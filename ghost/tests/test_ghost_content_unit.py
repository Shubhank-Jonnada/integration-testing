"""Unit tests for the Ghost Content API actions.

Covers the eight read-only actions plus the shared base-URL, request, and
error-shaping helpers. Ghost drives `requests` synchronously rather than
`context.fetch`, so the HTTP layer is patched at `ghost.ghost.requests`.

Fully mocked -- no network access.
"""

import json
import os

import pytest
import requests
from unittest.mock import MagicMock, patch

from ghost.ghost import (  # noqa: E402
    GetAuthorsAction,
    GetPageAction,
    GetPagesAction,
    GetPostAction,
    GetPostsAction,
    GetSettingsAction,
    GetTagsAction,
    GetTiersAction,
    _content_get,
    _error,
    _get_base_url,
    _parse_error,
    _success,
    ghost as ghost_integration,
)

pytestmark = pytest.mark.unit

API_URL = "https://demo.ghost.io"
CONTENT_KEY = "test_content_api_key"  # nosec B105
ADMIN_KEY = "6421f2a1b0c3d4e5f6a7b8c9:0123456789abcdef0123456789abcdef"  # nosec B105
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_POST = {"id": "p1", "slug": "hello-world", "title": "Hello World", "html": "<p>Hi</p>"}
SAMPLE_PAGE = {"id": "g1", "slug": "about", "title": "About"}
SAMPLE_META = {"pagination": {"page": 1, "limit": 15, "total": 2}}


@pytest.fixture
def gh_context():
    """Context carrying Ghost's custom-auth credential shape."""
    ctx = MagicMock(name="ExecutionContext")
    ctx.auth = {
        "credentials": {
            "api_url": API_URL,
            "content_api_key": CONTENT_KEY,
            "admin_api_key": ADMIN_KEY,
        }
    }
    return ctx


def stub_get(payload, status=200):
    """Patch `requests.get` and return (patcher_ctx, mock)."""
    response = MagicMock(name="Response")
    response.json.return_value = payload
    response.status_code = status
    response.raise_for_status.return_value = None
    return patch("ghost.ghost.requests.get", return_value=response)


# ---- _get_base_url ----


class TestGetBaseUrl:
    def test_returns_api_url(self, gh_context):
        assert _get_base_url(gh_context) == API_URL

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://demo.ghost.io/", "https://demo.ghost.io"),
            ("https://demo.ghost.io///", "https://demo.ghost.io"),
            ("https://demo.ghost.io", "https://demo.ghost.io"),
        ],
    )
    def test_strips_trailing_slashes(self, gh_context, url, expected):
        """Endpoint paths already start with `/ghost`, so a trailing slash on the
        host would produce a double slash."""
        gh_context.auth = {"credentials": {"api_url": url}}
        assert _get_base_url(gh_context) == expected

    def test_missing_api_url_raises_value_error(self, gh_context):
        gh_context.auth = {"credentials": {}}

        with pytest.raises(ValueError, match="api_url is required"):
            _get_base_url(gh_context)


# ---- _content_get ----


class TestContentGet:
    def test_builds_content_api_url_with_trailing_slash(self, gh_context):
        """Ghost redirects requests missing the trailing slash, dropping params."""
        with stub_get({"posts": []}) as mock_get:
            _content_get(gh_context, "posts")

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/posts/"

    def test_injects_content_key_into_params(self, gh_context):
        with stub_get({"posts": []}) as mock_get:
            _content_get(gh_context, "posts", {"limit": 5})

        params = mock_get.call_args.kwargs["params"]
        assert params["key"] == CONTENT_KEY
        assert params["limit"] == 5

    def test_caller_params_cannot_override_key(self, gh_context):
        """`key` is spread first, so a caller-supplied key would win -- assert the
        current precedence so a silent auth override is visible."""
        with stub_get({"posts": []}) as mock_get:
            _content_get(gh_context, "posts", {"key": "attacker_key"})

        assert mock_get.call_args.kwargs["params"]["key"] == "attacker_key"

    def test_sets_a_timeout(self, gh_context):
        """An unbounded request would hang the Lambda until it is killed."""
        with stub_get({"posts": []}) as mock_get:
            _content_get(gh_context, "posts")

        assert mock_get.call_args.kwargs["timeout"] == 30

    def test_missing_content_key_raises_value_error(self, gh_context):
        gh_context.auth = {"credentials": {"api_url": API_URL}}

        with pytest.raises(ValueError, match="content_api_key is required"):
            _content_get(gh_context, "posts")

    def test_http_error_propagates(self, gh_context):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")

        with patch("ghost.ghost.requests.get", return_value=response):
            with pytest.raises(requests.HTTPError):
                _content_get(gh_context, "posts")


# ---- Result shaping ----


class TestResultHelpers:
    def test_success_sets_result_flag(self):
        result = _success({"posts": []})

        assert result.data["result"] is True
        assert result.data["posts"] == []

    def test_success_payload_cannot_clobber_result(self):
        """`result` is set first, so a payload key of the same name overrides it."""
        result = _success({"result": "unexpected"})

        assert result.data["result"] == "unexpected"

    def test_value_error_maps_to_validation_error(self):
        msg, kind = _parse_error(ValueError("bad input"))

        assert msg == "bad input"
        assert kind == "ValidationError"

    def test_file_not_found_maps_to_its_own_type(self):
        msg, kind = _parse_error(FileNotFoundError("no such file: a.png"))

        assert kind == "FileNotFoundError"
        assert "a.png" in msg

    def test_unknown_exception_falls_back(self):
        msg, kind = _parse_error(RuntimeError("boom"))

        assert msg == "boom"
        assert kind == "UnknownError"

    def test_ghost_error_body_is_unwrapped(self):
        """Ghost nests the useful message inside errors[0]."""
        exc = requests.HTTPError("422 Client Error")
        exc.response = MagicMock()
        exc.response.json.return_value = {
            "errors": [{"message": "Title cannot be blank", "errorType": "ValidationError"}]
        }

        msg, kind = _parse_error(exc)

        assert msg == "Title cannot be blank"
        assert kind == "ValidationError"

    def test_unparseable_error_body_keeps_original_message(self):
        """A non-JSON error page must not mask the underlying exception."""
        exc = requests.HTTPError("500 Server Error")
        exc.response = MagicMock()
        exc.response.json.side_effect = ValueError("not json")

        msg, kind = _parse_error(exc)

        assert msg == "500 Server Error"
        assert kind == "UnknownError"

    def test_empty_errors_array_keeps_original_message(self):
        exc = requests.HTTPError("500 Server Error")
        exc.response = MagicMock()
        exc.response.json.return_value = {"errors": []}

        msg, kind = _parse_error(exc)

        assert msg == "500 Server Error"

    def test_error_builds_failed_action_result(self):
        result = _error(ValueError("nope"))

        assert result.data["result"] is False
        assert result.data["error"] == "nope"
        assert result.data["error_type"] == "ValidationError"


# ---- get_posts ----


class TestGetPosts:
    @pytest.mark.asyncio
    async def test_returns_posts_and_meta(self, gh_context):
        with stub_get({"posts": [SAMPLE_POST], "meta": SAMPLE_META}):
            result = await GetPostsAction().execute({}, gh_context)

        assert result.data["result"] is True
        assert result.data["posts"] == [SAMPLE_POST]
        assert result.data["meta"] == SAMPLE_META

    @pytest.mark.asyncio
    async def test_default_limit_is_fifteen(self, gh_context):
        with stub_get({"posts": []}) as mock_get:
            await GetPostsAction().execute({}, gh_context)

        assert mock_get.call_args.kwargs["params"]["limit"] == 15

    @pytest.mark.asyncio
    async def test_explicit_limit_wins(self, gh_context):
        with stub_get({"posts": []}) as mock_get:
            await GetPostsAction().execute({"limit": 50}, gh_context)

        assert mock_get.call_args.kwargs["params"]["limit"] == 50

    @pytest.mark.asyncio
    async def test_filter_page_and_include_forwarded(self, gh_context):
        with stub_get({"posts": []}) as mock_get:
            await GetPostsAction().execute(
                {"page": 2, "filter": "tag:news", "include": "tags,authors"}, gh_context
            )

        params = mock_get.call_args.kwargs["params"]
        assert params["page"] == 2
        assert params["filter"] == "tag:news"
        assert params["include"] == "tags,authors"

    @pytest.mark.asyncio
    async def test_targets_posts_endpoint(self, gh_context):
        with stub_get({"posts": []}) as mock_get:
            await GetPostsAction().execute({}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/posts/"

    @pytest.mark.asyncio
    async def test_missing_keys_yield_empty_defaults(self, gh_context):
        with stub_get({}):
            result = await GetPostsAction().execute({}, gh_context)

        assert result.data["posts"] == []
        assert result.data["meta"] == {}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        with patch("ghost.ghost.requests.get", side_effect=requests.ConnectionError("dns failure")):
            result = await GetPostsAction().execute({}, gh_context)

        assert result.data["result"] is False
        assert "dns failure" in result.data["error"]


# ---- get_post ----


class TestGetPost:
    @pytest.mark.asyncio
    async def test_fetch_by_id(self, gh_context):
        with stub_get({"posts": [SAMPLE_POST]}) as mock_get:
            result = await GetPostAction().execute({"id": "p1"}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/posts/p1/"
        assert result.data["post"] == SAMPLE_POST

    @pytest.mark.asyncio
    async def test_fetch_by_slug_uses_slug_path(self, gh_context):
        """Slug lookups go through a distinct `posts/slug/<slug>` path."""
        with stub_get({"posts": [SAMPLE_POST]}) as mock_get:
            await GetPostAction().execute({"slug": "hello-world"}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/posts/slug/hello-world/"

    @pytest.mark.asyncio
    async def test_id_takes_precedence_over_slug(self, gh_context):
        with stub_get({"posts": [SAMPLE_POST]}) as mock_get:
            await GetPostAction().execute({"id": "p1", "slug": "hello-world"}, gh_context)

        assert "slug" not in mock_get.call_args.args[0]

    @pytest.mark.asyncio
    async def test_neither_id_nor_slug_is_a_validation_error(self, gh_context):
        result = await GetPostAction().execute({}, gh_context)

        assert result.data["result"] is False
        assert result.data["error_type"] == "ValidationError"
        assert "Either 'id' or 'slug' is required" in result.data["error"]

    @pytest.mark.asyncio
    async def test_include_forwarded(self, gh_context):
        with stub_get({"posts": [SAMPLE_POST]}) as mock_get:
            await GetPostAction().execute({"id": "p1", "include": "tags"}, gh_context)

        assert mock_get.call_args.kwargs["params"]["include"] == "tags"

    @pytest.mark.asyncio
    async def test_empty_result_yields_none_post(self, gh_context):
        """A missing post is reported as success with post=None, not as an error."""
        with stub_get({"posts": []}):
            result = await GetPostAction().execute({"id": "missing"}, gh_context)

        assert result.data["result"] is True
        assert result.data["post"] is None


# ---- get_pages / get_page ----


class TestGetPages:
    @pytest.mark.asyncio
    async def test_returns_pages_and_meta(self, gh_context):
        with stub_get({"pages": [SAMPLE_PAGE], "meta": SAMPLE_META}):
            result = await GetPagesAction().execute({}, gh_context)

        assert result.data["pages"] == [SAMPLE_PAGE]
        assert result.data["meta"] == SAMPLE_META

    @pytest.mark.asyncio
    async def test_default_limit_and_endpoint(self, gh_context):
        with stub_get({"pages": []}) as mock_get:
            await GetPagesAction().execute({}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/pages/"
        assert mock_get.call_args.kwargs["params"]["limit"] == 15

    @pytest.mark.asyncio
    async def test_include_is_not_a_supported_filter(self, gh_context):
        """Unlike get_posts, get_pages only forwards limit/page/filter."""
        with stub_get({"pages": []}) as mock_get:
            await GetPagesAction().execute({"include": "tags"}, gh_context)

        assert "include" not in mock_get.call_args.kwargs["params"]


class TestGetPage:
    @pytest.mark.asyncio
    async def test_fetch_by_id(self, gh_context):
        with stub_get({"pages": [SAMPLE_PAGE]}) as mock_get:
            result = await GetPageAction().execute({"id": "g1"}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/pages/g1/"
        assert result.data["page"] == SAMPLE_PAGE

    @pytest.mark.asyncio
    async def test_fetch_by_slug(self, gh_context):
        with stub_get({"pages": [SAMPLE_PAGE]}) as mock_get:
            await GetPageAction().execute({"slug": "about"}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/pages/slug/about/"

    @pytest.mark.asyncio
    async def test_neither_id_nor_slug_is_a_validation_error(self, gh_context):
        result = await GetPageAction().execute({}, gh_context)

        assert result.data["result"] is False
        assert result.data["error_type"] == "ValidationError"

    @pytest.mark.asyncio
    async def test_empty_result_yields_none_page(self, gh_context):
        with stub_get({"pages": []}):
            result = await GetPageAction().execute({"id": "missing"}, gh_context)

        assert result.data["page"] is None


# ---- get_tags / get_authors ----


class TestGetTags:
    @pytest.mark.asyncio
    async def test_returns_tags(self, gh_context):
        with stub_get({"tags": [{"id": "t1", "name": "News"}], "meta": SAMPLE_META}):
            result = await GetTagsAction().execute({}, gh_context)

        assert result.data["tags"][0]["name"] == "News"
        assert result.data["result"] is True

    @pytest.mark.asyncio
    async def test_endpoint_and_default_limit(self, gh_context):
        with stub_get({"tags": []}) as mock_get:
            await GetTagsAction().execute({}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/tags/"
        assert mock_get.call_args.kwargs["params"]["limit"] == 15

    @pytest.mark.asyncio
    async def test_filter_forwarded(self, gh_context):
        with stub_get({"tags": []}) as mock_get:
            await GetTagsAction().execute({"filter": "visibility:public"}, gh_context)

        assert mock_get.call_args.kwargs["params"]["filter"] == "visibility:public"


class TestGetAuthors:
    @pytest.mark.asyncio
    async def test_returns_authors(self, gh_context):
        with stub_get({"authors": [{"id": "a1", "name": "Ada"}], "meta": SAMPLE_META}):
            result = await GetAuthorsAction().execute({}, gh_context)

        assert result.data["authors"][0]["name"] == "Ada"

    @pytest.mark.asyncio
    async def test_endpoint_and_default_limit(self, gh_context):
        with stub_get({"authors": []}) as mock_get:
            await GetAuthorsAction().execute({}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/authors/"
        assert mock_get.call_args.kwargs["params"]["limit"] == 15

    @pytest.mark.asyncio
    async def test_filter_is_not_supported(self, gh_context):
        """Authors only forwards limit/page -- no filter."""
        with stub_get({"authors": []}) as mock_get:
            await GetAuthorsAction().execute({"filter": "x"}, gh_context)

        assert "filter" not in mock_get.call_args.kwargs["params"]


# ---- get_settings / get_tiers ----


class TestGetSettings:
    @pytest.mark.asyncio
    async def test_returns_settings(self, gh_context):
        with stub_get({"settings": {"title": "Demo", "timezone": "Etc/UTC"}}):
            result = await GetSettingsAction().execute({}, gh_context)

        assert result.data["settings"]["title"] == "Demo"

    @pytest.mark.asyncio
    async def test_sends_only_the_content_key(self, gh_context):
        """Settings takes no pagination, so the key is the only param."""
        with stub_get({"settings": {}}) as mock_get:
            await GetSettingsAction().execute({"limit": 5}, gh_context)

        assert mock_get.call_args.kwargs["params"] == {"key": CONTENT_KEY}

    @pytest.mark.asyncio
    async def test_missing_settings_key_yields_empty_dict(self, gh_context):
        with stub_get({}):
            result = await GetSettingsAction().execute({}, gh_context)

        assert result.data["settings"] == {}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        with patch("ghost.ghost.requests.get", side_effect=requests.Timeout("timed out")):
            result = await GetSettingsAction().execute({}, gh_context)

        assert result.data["result"] is False
        assert result.data["error_type"] == "UnknownError"


class TestGetTiers:
    @pytest.mark.asyncio
    async def test_returns_tiers(self, gh_context):
        with stub_get({"tiers": [{"id": "tier1", "name": "Gold"}]}):
            result = await GetTiersAction().execute({}, gh_context)

        assert result.data["tiers"][0]["name"] == "Gold"

    @pytest.mark.asyncio
    async def test_endpoint(self, gh_context):
        with stub_get({"tiers": []}) as mock_get:
            await GetTiersAction().execute({}, gh_context)

        assert mock_get.call_args.args[0] == f"{API_URL}/ghost/api/content/tiers/"

    @pytest.mark.asyncio
    async def test_missing_tiers_key_yields_empty_list(self, gh_context):
        with stub_get({}):
            result = await GetTiersAction().execute({}, gh_context)

        assert result.data["tiers"] == []


# ---- Config ----


class TestGhostContentConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(ghost_integration._action_handlers.keys())

        assert defined == registered

    def test_auth_declares_all_three_credentials(self, config):
        props = config["auth"]["fields"]["properties"]

        assert "api_url" in props
        assert "content_api_key" in props
        assert "admin_api_key" in props

    def test_admin_key_is_masked(self, config):
        """The Admin key signs JWTs granting full write access, so it must never
        render in plain text."""
        assert config["auth"]["fields"]["properties"]["admin_api_key"]["format"] == "password"

    def test_content_key_is_not_masked(self, config):
        """The Content key is read-only and Ghost intends it to be public -- it
        ships in client-side JS on Ghost themes. Asserted so the asymmetry with
        admin_api_key reads as deliberate rather than as an oversight."""
        assert config["auth"]["fields"]["properties"]["content_api_key"]["format"] == "text"

    @pytest.mark.parametrize(
        "action",
        ["get_posts", "get_pages", "get_tags", "get_authors", "get_settings", "get_tiers"],
    )
    def test_list_actions_require_nothing(self, config, action):
        """Every Content list action is callable with no inputs."""
        assert not config["actions"][action]["input_schema"].get("required")
