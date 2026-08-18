"""Unit tests for the GitHub integration's shared core.

Covers the pieces every one of the 47 actions routes through:

- `handle_github_errors` -- the decorator that gates on token presence and
  converts exceptions into failed ActionResults
- `GitHubAPI.get_headers` -- Bearer auth and the pinned API version
- `GitHubAPI.paginated_fetch` -- GitHub's page-walking loop
- `GitHubConnectedAccountHandler` -- account info shown in the UI after connect

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from github.github import (  # noqa: E402
    GitHubAPI,
    GitHubConnectedAccountHandler,
    handle_github_errors,
    github as github_integration,
)

pytestmark = pytest.mark.unit

TOKEN = "gho_testtoken1234567890"  # nosec B105
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_USER = {
    "id": 1234567,
    "login": "octocat",
    "name": "Mona Lisa Octocat",
    "email": "octocat@github.com",
    "avatar_url": "https://avatars.githubusercontent.com/u/1234567",
    "company": "GitHub",
}


@pytest.fixture
def gh_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": TOKEN}}
    return ctx


# ---- get_headers ----


class TestGetHeaders:
    def test_uses_bearer_authorization(self, gh_context):
        assert GitHubAPI.get_headers(gh_context)["Authorization"] == f"Bearer {TOKEN}"

    def test_pins_the_api_version(self, gh_context):
        """Pinning avoids silent behaviour changes as GitHub ships new versions."""
        assert GitHubAPI.get_headers(gh_context)["X-GitHub-Api-Version"] == "2022-11-28"

    def test_requests_the_v3_json_media_type(self, gh_context):
        assert GitHubAPI.get_headers(gh_context)["Accept"] == "application/vnd.github.v3+json"

    def test_sets_json_content_type(self, gh_context):
        assert GitHubAPI.get_headers(gh_context)["Content-Type"] == "application/json"

    def test_missing_token_yields_a_bare_bearer(self, gh_context):
        """get_headers itself does not validate -- that's the decorator's job --
        so it produces `Bearer ` with an empty token."""
        gh_context.auth = {}

        assert GitHubAPI.get_headers(gh_context)["Authorization"] == "Bearer "

    def test_returns_a_fresh_dict(self, gh_context):
        headers = GitHubAPI.get_headers(gh_context)
        headers["X-Injected"] = "1"

        assert "X-Injected" not in GitHubAPI.get_headers(gh_context)

    def test_base_url_is_the_public_api(self):
        assert GitHubAPI.BASE_URL == "https://api.github.com"


# ---- handle_github_errors ----


class _Dummy:
    """Minimal action-like object for exercising the decorator."""

    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.calls = 0

    @handle_github_errors("dummy_action")
    async def execute(self, inputs, context):
        self.calls += 1
        return await self._behaviour(inputs, context)


class TestHandleGithubErrors:
    @pytest.mark.asyncio
    async def test_passes_through_on_success(self, gh_context):
        from autohive_integrations_sdk import ActionResult

        async def ok(_inputs, _context):
            return ActionResult(data={"result": True, "value": 42}, cost_usd=0.0)

        dummy = _Dummy(ok)

        result = await dummy.execute({}, gh_context)

        assert result.data["result"] is True
        assert result.data["value"] == 42

    @pytest.mark.asyncio
    async def test_missing_token_short_circuits_before_the_body(self, gh_context):
        """The wrapped function must not run at all without a token -- otherwise
        every action would issue an unauthenticated request."""
        gh_context.auth = {"credentials": {}}

        async def should_not_run(_inputs, _context):
            raise AssertionError("body ran without a token")

        dummy = _Dummy(should_not_run)

        result = await dummy.execute({}, gh_context)

        assert result.data["result"] is False
        assert "No access token found" in result.data["error"]
        assert dummy.calls == 0

    @pytest.mark.asyncio
    async def test_missing_credentials_key_also_short_circuits(self, gh_context):
        gh_context.auth = {}

        async def should_not_run(_inputs, _context):
            raise AssertionError("body ran without a token")

        result = await _Dummy(should_not_run).execute({}, gh_context)

        assert result.data["result"] is False

    @pytest.mark.asyncio
    async def test_empty_token_string_short_circuits(self, gh_context):
        gh_context.auth = {"credentials": {"access_token": ""}}

        async def should_not_run(_inputs, _context):
            raise AssertionError("body ran without a token")

        result = await _Dummy(should_not_run).execute({}, gh_context)

        assert result.data["result"] is False

    @pytest.mark.asyncio
    async def test_error_message_tells_the_user_to_reconnect(self, gh_context):
        """The message is user-facing, so it names the remedy rather than the
        internal cause."""
        gh_context.auth = {"credentials": {}}

        async def noop(_inputs, _context):
            return None

        result = await _Dummy(noop).execute({}, gh_context)

        assert "reconnect your GitHub account" in result.data["error"]

    @pytest.mark.asyncio
    async def test_exceptions_become_failed_action_results(self, gh_context):
        async def boom(_inputs, _context):
            raise Exception("HTTP 404: Not Found")

        result = await _Dummy(boom).execute({}, gh_context)

        assert result.data["result"] is False
        assert result.data["error"] == "HTTP 404: Not Found"

    @pytest.mark.asyncio
    async def test_key_errors_are_also_captured(self, gh_context):
        """A missing required input raises KeyError, which is caught the same way
        rather than propagating to the platform."""

        async def missing_input(inputs, _context):
            return inputs["required_but_absent"]

        result = await _Dummy(missing_input).execute({}, gh_context)

        assert result.data["result"] is False

    @pytest.mark.asyncio
    async def test_no_action_raises_out_of_the_decorator(self, gh_context):
        """Every failure path returns an ActionResult, so the platform never sees
        a raised exception from a decorated action."""

        async def boom(_inputs, _context):
            raise RuntimeError("unexpected")

        result = await _Dummy(boom).execute({}, gh_context)

        assert result.data["result"] is False
        assert result.cost_usd == 0.0

    def test_functools_wraps_preserves_the_name(self):
        """`@wraps` keeps the wrapped name so tracebacks stay readable."""
        assert _Dummy.execute.__name__ == "execute"


# ---- paginated_fetch ----


class TestPaginatedFetch:
    @pytest.mark.asyncio
    async def test_single_short_page_stops_immediately(self, gh_context):
        """Fewer items than per_page means this is the last page."""
        gh_context.fetch.return_value = [{"id": 1}, {"id": 2}]

        items = await GitHubAPI.paginated_fetch(gh_context, "https://api.github.com/x")

        assert items == [{"id": 1}, {"id": 2}]
        assert gh_context.fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_defaults_to_max_page_size(self, gh_context):
        """per_page=100 is GitHub's maximum, minimising round trips."""
        gh_context.fetch.return_value = []

        await GitHubAPI.paginated_fetch(gh_context, "https://api.github.com/x")

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["per_page"] == 100
        assert params["page"] == 1

    @pytest.mark.asyncio
    async def test_caller_params_are_preserved(self, gh_context):
        gh_context.fetch.return_value = []

        await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", params={"state": "open"}
        )

        assert gh_context.fetch.call_args.kwargs["params"]["state"] == "open"

    @pytest.mark.asyncio
    async def test_caller_can_override_per_page(self, gh_context):
        """setdefault means an explicit per_page wins."""
        gh_context.fetch.return_value = []

        await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", params={"per_page": 5}
        )

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 5

    @pytest.mark.asyncio
    async def test_walks_pages_until_a_short_one(self, gh_context):
        full = [{"id": i} for i in range(3)]
        gh_context.fetch.side_effect = [full, full, [{"id": 99}]]

        items = await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", params={"per_page": 3}
        )

        assert len(items) == 7
        assert gh_context.fetch.await_count == 3

    @pytest.mark.asyncio
    async def test_page_number_increments(self, gh_context):
        full = [{"id": i} for i in range(2)]
        gh_context.fetch.side_effect = [full, full, []]

        await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", params={"per_page": 2}
        )

        assert gh_context.fetch.await_count == 3

    @pytest.mark.asyncio
    async def test_empty_first_page_returns_empty(self, gh_context):
        gh_context.fetch.return_value = []

        assert await GitHubAPI.paginated_fetch(gh_context, "https://api.github.com/x") == []

    @pytest.mark.asyncio
    async def test_data_key_unwraps_a_dict_envelope(self, gh_context):
        """Some GitHub list endpoints wrap results, e.g. {"workflows": [...]}."""
        gh_context.fetch.return_value = {"workflows": [{"id": 1}], "total_count": 1}

        items = await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", data_key="workflows"
        )

        assert items == [{"id": 1}]

    @pytest.mark.asyncio
    async def test_missing_data_key_stops_the_loop(self, gh_context):
        gh_context.fetch.return_value = {"total_count": 0}

        items = await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", data_key="workflows"
        )

        assert items == []
        assert gh_context.fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_bare_dict_response_is_wrapped_in_a_list(self, gh_context):
        """Without a data_key a single object becomes a one-item list, so callers
        always get a list back."""
        gh_context.fetch.return_value = {"id": 1, "name": "single"}

        items = await GitHubAPI.paginated_fetch(gh_context, "https://api.github.com/x")

        assert items == [{"id": 1, "name": "single"}]

    @pytest.mark.asyncio
    async def test_none_response_terminates(self, gh_context):
        gh_context.fetch.return_value = None

        assert await GitHubAPI.paginated_fetch(gh_context, "https://api.github.com/x") == []

    @pytest.mark.asyncio
    async def test_sends_auth_headers(self, gh_context):
        gh_context.fetch.return_value = []

        await GitHubAPI.paginated_fetch(gh_context, "https://api.github.com/x")

        assert gh_context.fetch.call_args.kwargs["headers"]["Authorization"] == f"Bearer {TOKEN}"

    @pytest.mark.asyncio
    async def test_headers_are_built_once_for_the_whole_walk(self, gh_context):
        """Headers are computed before the loop, so a long walk doesn't rebuild
        them per page."""
        full = [{"id": i} for i in range(2)]
        gh_context.fetch.side_effect = [full, []]

        await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", params={"per_page": 2}
        )

        first, second = gh_context.fetch.await_args_list
        assert first.kwargs["headers"] == second.kwargs["headers"]

    @pytest.mark.asyncio
    async def test_exactly_per_page_items_triggers_another_request(self, gh_context):
        """A full page is indistinguishable from more data, so the loop must ask
        again -- the boundary case that decides whether pagination terminates
        early and silently truncates results."""
        gh_context.fetch.side_effect = [[{"id": 1}, {"id": 2}], []]

        items = await GitHubAPI.paginated_fetch(
            gh_context, "https://api.github.com/x", params={"per_page": 2}
        )

        assert len(items) == 2
        assert gh_context.fetch.await_count == 2

    @pytest.mark.asyncio
    async def test_mutates_the_caller_supplied_params_dict(self, gh_context):
        """The helper writes per_page/page and increments page in place, so a
        params dict reused across calls carries state over. No current caller
        does this, but it isn't defensive."""
        params = {"state": "open"}
        gh_context.fetch.return_value = []

        await GitHubAPI.paginated_fetch(gh_context, "https://api.github.com/x", params=params)

        assert params["per_page"] == 100
        assert params["page"] == 1


# ---- Connected account handler ----


class TestConnectedAccountHandler:
    @pytest.mark.asyncio
    async def test_maps_user_fields(self, gh_context):
        gh_context.fetch.return_value = SAMPLE_USER

        info = await GitHubConnectedAccountHandler().get_account_info(gh_context)

        assert info.username == "octocat"
        assert info.email == "octocat@github.com"
        assert info.avatar_url.endswith("1234567")
        assert info.organization == "GitHub"

    @pytest.mark.asyncio
    async def test_splits_name_into_first_and_last(self, gh_context):
        gh_context.fetch.return_value = SAMPLE_USER

        info = await GitHubConnectedAccountHandler().get_account_info(gh_context)

        assert info.first_name == "Mona"
        assert info.last_name == "Lisa Octocat"

    @pytest.mark.asyncio
    async def test_single_word_name_leaves_last_name_none(self, gh_context):
        """maxsplit=1 means a one-word name yields no surname rather than
        raising an IndexError."""
        gh_context.fetch.return_value = {**SAMPLE_USER, "name": "Octocat"}

        info = await GitHubConnectedAccountHandler().get_account_info(gh_context)

        assert info.first_name == "Octocat"
        assert info.last_name is None

    @pytest.mark.parametrize("name", ["", None])
    @pytest.mark.asyncio
    async def test_absent_name_yields_no_name_parts(self, gh_context, name):
        """GitHub profiles often have no display name at all."""
        gh_context.fetch.return_value = {**SAMPLE_USER, "name": name}

        info = await GitHubConnectedAccountHandler().get_account_info(gh_context)

        assert info.first_name is None
        assert info.last_name is None

    @pytest.mark.asyncio
    async def test_user_id_is_stringified(self, gh_context):
        """GitHub returns a numeric id; the platform expects a string."""
        gh_context.fetch.return_value = SAMPLE_USER

        info = await GitHubConnectedAccountHandler().get_account_info(gh_context)

        assert info.user_id == "1234567"

    @pytest.mark.asyncio
    async def test_missing_id_yields_none_not_the_string_none(self, gh_context):
        """A naive str() would produce the literal "None" here."""
        gh_context.fetch.return_value = {"login": "octocat"}

        info = await GitHubConnectedAccountHandler().get_account_info(gh_context)

        assert info.user_id is None

    @pytest.mark.asyncio
    async def test_private_email_yields_none(self, gh_context):
        """GitHub omits email when the user keeps it private."""
        gh_context.fetch.return_value = {**SAMPLE_USER, "email": None}

        info = await GitHubConnectedAccountHandler().get_account_info(gh_context)

        assert info.email is None


# ---- Config ----


class TestGithubConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(github_integration._action_handlers.keys())

        assert defined == registered

    def test_uses_platform_oauth(self, config):
        assert config["auth"]["type"] == "platform"

    def test_every_action_declares_an_output_schema(self, config):
        missing = [
            name for name, spec in config["actions"].items() if "output_schema" not in spec
        ]

        assert missing == []

    def test_every_action_has_a_display_name_and_description(self, config):
        incomplete = [
            name
            for name, spec in config["actions"].items()
            if not spec.get("display_name") or not spec.get("description")
        ]

        assert incomplete == []
