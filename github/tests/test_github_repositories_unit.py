"""Unit tests for the GitHub repository actions.

Covers `create_repository`, `get_repository`, `list_repositories`,
`update_repository`, `delete_repository`, `list_user_repositories`, and
`list_organization_repositories`.

The routing logic is the interesting part: three of these actions choose between
`/orgs/{org}/repos`, `/users/{username}/repos`, and `/user/repos` depending on
which inputs are present, and picking the wrong one either creates a repo in the
wrong place or lists someone else's.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from github.github import (  # noqa: E402
    CreateRepository,
    DeleteRepository,
    GetRepository,
    ListOrganizationRepositories,
    ListRepositories,
    ListUserRepositories,
    UpdateRepository,
)

pytestmark = pytest.mark.unit

TOKEN = "gho_testtoken1234567890"  # nosec B105
BASE = "https://api.github.com"
OWNER = "octocat"
REPO = "hello-world"
ORG = "acme-corp"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

FULL_REPO = {
    "id": 1296269,
    "name": REPO,
    "full_name": f"{OWNER}/{REPO}",
    "description": "My first repository",
    "private": False,
    "fork": False,
    "default_branch": "main",
    "created_at": "2026-01-26T19:01:12Z",
    "updated_at": "2026-01-26T19:14:43Z",
    "pushed_at": "2026-01-26T19:06:43Z",
    "clone_url": f"https://github.com/{OWNER}/{REPO}.git",
    "ssh_url": f"git@github.com:{OWNER}/{REPO}.git",
    "html_url": f"https://github.com/{OWNER}/{REPO}",
    "language": "Python",
    "visibility": "public",
    "forks_count": 9,
    "stargazers_count": 80,
    "watchers_count": 80,
    "open_issues_count": 0,
    "has_issues": True,
    "has_wiki": True,
}


@pytest.fixture
def gh_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": TOKEN}}
    return ctx


# ---- create_repository ----


class TestCreateRepository:
    @pytest.mark.asyncio
    async def test_creates_repository(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        result = await CreateRepository().execute({"name": REPO}, gh_context)

        assert result.data["id"] == 1296269
        assert result.data["full_name"] == f"{OWNER}/{REPO}"

    @pytest.mark.asyncio
    async def test_personal_repo_targets_user_repos(self, gh_context):
        """Without `org` the repo is created under the authenticated user."""
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute({"name": REPO}, gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/user/repos"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_org_repo_targets_org_repos(self, gh_context):
        """Passing `org` changes the owning account -- creating a repo in the
        wrong org is visible to the whole organisation."""
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute({"name": REPO, "org": ORG}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/orgs/{ORG}/repos"

    @pytest.mark.asyncio
    async def test_defaults_to_public(self, gh_context):
        """`private` defaults to False, so an unqualified create is world-visible.
        Asserted explicitly because the consequence of a wrong default here is
        publishing code."""
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute({"name": REPO}, gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["private"] is False

    @pytest.mark.asyncio
    async def test_private_true_is_forwarded(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute({"name": REPO, "private": True}, gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["private"] is True

    @pytest.mark.asyncio
    async def test_feature_flags_default_to_enabled(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute({"name": REPO}, gh_context)

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["has_issues"] is True
        assert body["has_projects"] is True
        assert body["has_wiki"] is True

    @pytest.mark.asyncio
    async def test_feature_flags_can_be_disabled(self, gh_context):
        """These are always present in the body, so False is sent rather than
        omitted."""
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute(
            {"name": REPO, "has_issues": False, "has_projects": False, "has_wiki": False},
            gh_context,
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["has_issues"] is False
        assert body["has_projects"] is False
        assert body["has_wiki"] is False

    @pytest.mark.asyncio
    async def test_auto_init_defaults_to_false(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute({"name": REPO}, gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["auto_init"] is False

    @pytest.mark.asyncio
    async def test_optional_templates_are_omitted_when_absent(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute({"name": REPO}, gh_context)

        body = gh_context.fetch.call_args.kwargs["json"]
        for key in ("description", "homepage", "gitignore_template", "license_template"):
            assert key not in body

    @pytest.mark.asyncio
    async def test_optional_fields_forwarded(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await CreateRepository().execute(
            {
                "name": REPO,
                "description": "A repo",
                "homepage": "https://example.com",
                "gitignore_template": "Python",
                "license_template": "mit",
                "auto_init": True,
            },
            gh_context,
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["description"] == "A repo"
        assert body["homepage"] == "https://example.com"
        assert body["gitignore_template"] == "Python"
        assert body["license_template"] == "mit"
        assert body["auto_init"] is True

    @pytest.mark.asyncio
    async def test_response_exposes_both_clone_urls(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        result = await CreateRepository().execute({"name": REPO}, gh_context)

        assert result.data["clone_url"].endswith(".git")
        assert result.data["ssh_url"].startswith("git@github.com:")

    @pytest.mark.asyncio
    async def test_missing_name_is_captured_by_the_decorator(self, gh_context):
        result = await CreateRepository().execute({}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_token_is_captured(self, gh_context):
        gh_context.auth = {"credentials": {}}

        result = await CreateRepository().execute({"name": REPO}, gh_context)

        assert result.data["result"] is False
        assert "No access token found" in result.data["error"]
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_api_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 422: name already exists")

        result = await CreateRepository().execute({"name": REPO}, gh_context)

        assert result.data["result"] is False
        assert "already exists" in result.data["error"]


# ---- get_repository ----


class TestGetRepository:
    @pytest.mark.asyncio
    async def test_returns_repository_details(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        result = await GetRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["full_name"] == f"{OWNER}/{REPO}"
        assert result.data["stargazers_count"] == 80

    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await GetRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/repos/{OWNER}/{REPO}"

    @pytest.mark.asyncio
    async def test_html_url_is_exposed_as_url(self, gh_context):
        """The response renames html_url to url."""
        gh_context.fetch.return_value = FULL_REPO

        result = await GetRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["url"] == FULL_REPO["html_url"]
        assert "html_url" not in result.data

    @pytest.mark.asyncio
    async def test_nullable_fields_use_get(self, gh_context):
        """description and language can legitimately be null on a fresh repo, so
        they are read defensively while required fields are not."""
        payload = {**FULL_REPO}
        del payload["description"]
        del payload["language"]
        gh_context.fetch.return_value = payload

        result = await GetRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["description"] is None
        assert result.data["language"] is None

    @pytest.mark.asyncio
    async def test_missing_required_response_field_is_captured(self, gh_context):
        """Required fields are read with [] so a truncated payload surfaces as a
        captured KeyError rather than a partial result."""
        gh_context.fetch.return_value = {"name": REPO}

        result = await GetRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False

    @pytest.mark.parametrize("missing", ["owner", "repo"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO}
        del inputs[missing]

        result = await GetRepository().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_not_found_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Not Found")

        result = await GetRepository().execute({"owner": OWNER, "repo": "nope"}, gh_context)

        assert result.data["result"] is False


# ---- list_repositories ----


class TestListRepositories:
    @pytest.mark.asyncio
    async def test_returns_a_bare_list(self, gh_context):
        """Unlike most actions this returns a list as `data`, not a dict."""
        gh_context.fetch.return_value = [FULL_REPO]

        result = await ListRepositories().execute({}, gh_context)

        assert isinstance(result.data, list)
        assert result.data[0]["name"] == REPO

    @pytest.mark.asyncio
    async def test_no_scope_lists_the_authenticated_user(self, gh_context):
        gh_context.fetch.return_value = []

        await ListRepositories().execute({}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/user/repos"

    @pytest.mark.asyncio
    async def test_username_lists_that_user(self, gh_context):
        gh_context.fetch.return_value = []

        await ListRepositories().execute({"username": "torvalds"}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/users/torvalds/repos"

    @pytest.mark.asyncio
    async def test_org_takes_precedence_over_username(self, gh_context):
        """The branch order is org, then username, then authenticated user."""
        gh_context.fetch.return_value = []

        await ListRepositories().execute({"org": ORG, "username": "torvalds"}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/orgs/{ORG}/repos"

    @pytest.mark.asyncio
    async def test_default_sort_and_direction(self, gh_context):
        gh_context.fetch.return_value = []

        await ListRepositories().execute({}, gh_context)

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["type"] == "all"
        assert params["sort"] == "updated"
        assert params["direction"] == "desc"

    @pytest.mark.asyncio
    async def test_explicit_sort_options_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListRepositories().execute(
            {"type": "private", "sort": "created", "direction": "asc"}, gh_context
        )

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["type"] == "private"
        assert params["sort"] == "created"
        assert params["direction"] == "asc"

    @pytest.mark.asyncio
    async def test_pagination_is_applied(self, gh_context):
        """Listing goes through paginated_fetch, so per_page is set."""
        gh_context.fetch.return_value = []

        await ListRepositories().execute({}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 100

    @pytest.mark.asyncio
    async def test_empty_list_yields_empty_data(self, gh_context):
        gh_context.fetch.return_value = []

        result = await ListRepositories().execute({}, gh_context)

        assert result.data == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 403: rate limit exceeded")

        result = await ListRepositories().execute({}, gh_context)

        assert result.data["result"] is False


# ---- list_user_repositories / list_organization_repositories ----


class TestListUserRepositories:
    @pytest.mark.asyncio
    async def test_named_user_targets_users_path(self, gh_context):
        gh_context.fetch.return_value = [FULL_REPO]

        await ListUserRepositories().execute({"username": "torvalds"}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/users/torvalds/repos"

    @pytest.mark.asyncio
    async def test_omitted_username_falls_back_to_authenticated_user(self, gh_context):
        """`/user/repos` also returns private repos the token can see, whereas
        `/users/{name}/repos` only ever returns public ones."""
        gh_context.fetch.return_value = []

        await ListUserRepositories().execute({}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/user/repos"

    @pytest.mark.asyncio
    async def test_returns_a_list(self, gh_context):
        gh_context.fetch.return_value = [FULL_REPO]

        result = await ListUserRepositories().execute({}, gh_context)

        assert isinstance(result.data, list)

    @pytest.mark.asyncio
    async def test_default_params(self, gh_context):
        gh_context.fetch.return_value = []

        await ListUserRepositories().execute({}, gh_context)

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["type"] == "all"
        assert params["sort"] == "updated"


class TestListOrganizationRepositories:
    @pytest.mark.asyncio
    async def test_targets_the_org_path(self, gh_context):
        gh_context.fetch.return_value = [FULL_REPO]

        await ListOrganizationRepositories().execute({"org": ORG}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/orgs/{ORG}/repos"

    @pytest.mark.asyncio
    async def test_org_is_required(self, gh_context):
        """Unlike list_user_repositories there is no fallback -- an org must be
        named."""
        result = await ListOrganizationRepositories().execute({}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_sort_options_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListOrganizationRepositories().execute(
            {"org": ORG, "type": "sources", "direction": "asc"}, gh_context
        )

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["type"] == "sources"
        assert params["direction"] == "asc"

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: org not found")

        result = await ListOrganizationRepositories().execute({"org": "nope"}, gh_context)

        assert result.data["result"] is False


# ---- update_repository ----


class TestUpdateRepository:
    @pytest.mark.asyncio
    async def test_request_uses_patch(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await UpdateRepository().execute(
            {"owner": OWNER, "repo": REPO, "description": "Updated"}, gh_context
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}"
        assert call.kwargs["method"] == "PATCH"

    @pytest.mark.asyncio
    async def test_none_values_are_stripped(self, gh_context):
        """The action always passes all five keys, so the API helper filters the
        Nones -- otherwise a partial update would blank the other fields."""
        gh_context.fetch.return_value = FULL_REPO

        await UpdateRepository().execute(
            {"owner": OWNER, "repo": REPO, "description": "Updated"}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {"description": "Updated"}

    @pytest.mark.asyncio
    async def test_private_false_survives_the_none_filter(self, gh_context):
        """The filter is `is not None`, so making a private repo public works --
        a truthiness filter would silently drop this."""
        gh_context.fetch.return_value = FULL_REPO

        await UpdateRepository().execute(
            {"owner": OWNER, "repo": REPO, "private": False}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {"private": False}

    @pytest.mark.asyncio
    async def test_feature_flags_false_survive(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await UpdateRepository().execute(
            {"owner": OWNER, "repo": REPO, "has_issues": False, "has_wiki": False}, gh_context
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["has_issues"] is False
        assert body["has_wiki"] is False

    @pytest.mark.asyncio
    async def test_empty_description_is_sent(self, gh_context):
        """An empty string is not None, so a description can be cleared."""
        gh_context.fetch.return_value = FULL_REPO

        await UpdateRepository().execute(
            {"owner": OWNER, "repo": REPO, "description": ""}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {"description": ""}

    @pytest.mark.asyncio
    async def test_no_updatable_fields_sends_empty_body(self, gh_context):
        gh_context.fetch.return_value = FULL_REPO

        await UpdateRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_has_projects_is_not_updatable(self, gh_context):
        """create_repository accepts has_projects but update omits it."""
        gh_context.fetch.return_value = FULL_REPO

        await UpdateRepository().execute(
            {"owner": OWNER, "repo": REPO, "has_projects": False}, gh_context
        )

        assert "has_projects" not in gh_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 403: admin rights required")

        result = await UpdateRepository().execute(
            {"owner": OWNER, "repo": REPO, "private": True}, gh_context
        )

        assert result.data["result"] is False


# ---- delete_repository ----


class TestDeleteRepository:
    @pytest.mark.asyncio
    async def test_request_uses_delete(self, gh_context):
        gh_context.fetch.return_value = None

        await DeleteRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_reports_the_deleted_slug(self, gh_context):
        """The response echoes owner/repo so the caller can confirm which
        repository was removed."""
        gh_context.fetch.return_value = None

        result = await DeleteRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["deleted"] is True
        assert result.data["repository"] == f"{OWNER}/{REPO}"

    @pytest.mark.asyncio
    async def test_sends_no_body(self, gh_context):
        gh_context.fetch.return_value = None

        await DeleteRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert "json" not in gh_context.fetch.call_args.kwargs

    @pytest.mark.parametrize("missing", ["owner", "repo"])
    @pytest.mark.asyncio
    async def test_required_inputs_prevent_the_request(self, gh_context, missing):
        """A missing owner or repo must not produce a request against a malformed
        path -- this is a destructive action."""
        inputs = {"owner": OWNER, "repo": REPO}
        del inputs[missing]

        result = await DeleteRepository().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_permission_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 403: Must have admin rights")

        result = await DeleteRepository().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False
        assert "admin rights" in result.data["error"]


# ---- Config ----


class TestGithubRepositoryConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action", ["get_repository", "update_repository", "delete_repository"]
    )
    def test_repo_scoped_actions_require_owner_and_repo(self, config, action):
        required = config["actions"][action]["input_schema"]["required"]
        assert "owner" in required
        assert "repo" in required

    def test_create_repository_requires_only_name(self, config):
        assert config["actions"]["create_repository"]["input_schema"]["required"] == ["name"]

    def test_list_organization_repositories_requires_org(self, config):
        required = config["actions"]["list_organization_repositories"]["input_schema"]["required"]
        assert "org" in required

    @pytest.mark.parametrize("action", ["list_repositories", "list_user_repositories"])
    def test_flexible_list_actions_require_nothing(self, config, action):
        assert not config["actions"][action]["input_schema"].get("required")

    def test_create_repository_private_defaults_to_false_in_schema(self, config):
        """Schema default and handler default must agree, or the UI will show one
        thing and the code do another."""
        props = config["actions"]["create_repository"]["input_schema"]["properties"]
        assert props["private"].get("default") is False
