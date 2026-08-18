"""Unit tests for the Ghost Admin API actions.

Covers the seven write actions plus the JWT signing and request helpers. The
Admin API authenticates with a short-lived HS256 token derived from a
`id:hex_secret` key pair, so `_make_admin_jwt` is decoded and inspected directly
rather than treated as opaque.

Fully mocked -- no network access, no files written outside tmp_path.
"""

import json
import os

import jwt
import pytest
import requests
from unittest.mock import MagicMock, patch

from ghost.ghost import (  # noqa: E402
    CreateMemberAction,
    CreatePageAction,
    CreatePostAction,
    SendNewsletterAction,
    UpdateMemberAction,
    UpdatePostAction,
    UploadImageAction,
    _admin_request,
    _make_admin_jwt,
)

pytestmark = pytest.mark.unit

API_URL = "https://demo.ghost.io"
KEY_ID = "6421f2a1b0c3d4e5f6a7b8c9"
KEY_SECRET = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
ADMIN_KEY = f"{KEY_ID}:{KEY_SECRET}"  # nosec B105
CONTENT_KEY = "test_content_api_key"  # nosec B105
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

UPDATED_AT = "2026-01-15T10:00:00.000Z"
SAMPLE_POST = {"id": "p1", "title": "Hello", "status": "draft", "updated_at": UPDATED_AT}
SAMPLE_MEMBER = {"id": "m1", "email": "ada@example.com", "name": "Ada"}


@pytest.fixture
def gh_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.auth = {
        "credentials": {
            "api_url": API_URL,
            "content_api_key": CONTENT_KEY,
            "admin_api_key": ADMIN_KEY,
        }
    }
    return ctx


def stub_request(payload, content=b"{}"):
    """Patch `requests.request` and expose the call for assertions."""
    response = MagicMock(name="Response")
    response.json.return_value = payload
    response.content = content
    response.raise_for_status.return_value = None
    return patch("ghost.ghost.requests.request", return_value=response)


# ---- _make_admin_jwt ----


class TestMakeAdminJwt:
    def test_token_is_signed_with_the_hex_decoded_secret(self, gh_context):
        """Ghost secrets are hex-encoded; signing with the raw string would
        produce a token Ghost rejects."""
        token = _make_admin_jwt(gh_context)

        decoded = jwt.decode(
            token, bytes.fromhex(KEY_SECRET), algorithms=["HS256"], audience="/admin/"
        )
        assert decoded["aud"] == "/admin/"

    def test_key_id_is_in_the_kid_header(self, gh_context):
        """Ghost looks up the signing secret by `kid`, so it must be in the
        header, not the payload."""
        token = _make_admin_jwt(gh_context)

        assert jwt.get_unverified_header(token)["kid"] == KEY_ID

    def test_algorithm_is_hs256(self, gh_context):
        assert jwt.get_unverified_header(_make_admin_jwt(gh_context))["alg"] == "HS256"

    def test_token_expires_five_minutes_out(self, gh_context):
        """Ghost rejects tokens with a lifetime over 5 minutes."""
        token = _make_admin_jwt(gh_context)

        decoded = jwt.decode(
            token, bytes.fromhex(KEY_SECRET), algorithms=["HS256"], audience="/admin/"
        )
        assert decoded["exp"] - decoded["iat"] == 300

    def test_missing_admin_key_raises_value_error(self, gh_context):
        gh_context.auth = {"credentials": {"api_url": API_URL}}

        with pytest.raises(ValueError, match="admin_api_key is required"):
            _make_admin_jwt(gh_context)

    def test_key_without_colon_raises_value_error(self, gh_context):
        gh_context.auth = {"credentials": {"api_url": API_URL, "admin_api_key": "no-colon-here"}}

        with pytest.raises(ValueError, match="must be in format id:secret"):
            _make_admin_jwt(gh_context)

    def test_only_the_first_colon_delimits(self, gh_context):
        """`split(":", 1)` means the id is everything before the FIRST colon and
        the secret keeps the rest. A key with a stray extra colon therefore
        yields a non-hex secret and fails at decode -- not a silently truncated
        id, which would be far harder to diagnose."""
        gh_context.auth = {
            "credentials": {"api_url": API_URL, "admin_api_key": f"{KEY_ID}:extra:{KEY_SECRET}"}
        }

        with pytest.raises(ValueError):
            _make_admin_jwt(gh_context)

    def test_non_hex_secret_raises(self, gh_context):
        gh_context.auth = {"credentials": {"api_url": API_URL, "admin_api_key": f"{KEY_ID}:zzzz"}}

        with pytest.raises(ValueError):
            _make_admin_jwt(gh_context)


# ---- _admin_request ----


class TestAdminRequest:
    def test_builds_admin_url_with_trailing_slash(self, gh_context):
        with stub_request({"posts": []}) as mock_req:
            _admin_request(gh_context, "GET", "posts")

        assert mock_req.call_args.args[1] == f"{API_URL}/ghost/api/admin/posts/"

    def test_method_is_forwarded(self, gh_context):
        with stub_request({}) as mock_req:
            _admin_request(gh_context, "PUT", "posts/p1")

        assert mock_req.call_args.args[0] == "PUT"

    def test_uses_ghost_authorization_scheme(self, gh_context):
        """Admin auth is `Ghost <jwt>`, not `Bearer <jwt>`."""
        with stub_request({}) as mock_req:
            _admin_request(gh_context, "GET", "posts")

        auth = mock_req.call_args.kwargs["headers"]["Authorization"]
        assert auth.startswith("Ghost ")

    def test_json_branch_sets_content_type(self, gh_context):
        with stub_request({}) as mock_req:
            _admin_request(gh_context, "POST", "posts", json={"posts": []})

        assert mock_req.call_args.kwargs["headers"]["Content-Type"] == "application/json"

    def test_multipart_branch_omits_content_type(self, gh_context):
        """requests must set the multipart boundary itself, so a manual
        Content-Type would corrupt the upload."""
        with stub_request({}) as mock_req:
            _admin_request(gh_context, "POST", "images/upload", files={"file": ("a.png", b"x", "image/png")})

        headers = mock_req.call_args.kwargs["headers"]
        assert "Content-Type" not in headers
        assert "files" in mock_req.call_args.kwargs

    def test_sets_a_timeout(self, gh_context):
        with stub_request({}) as mock_req:
            _admin_request(gh_context, "GET", "posts")

        assert mock_req.call_args.kwargs["timeout"] == 30

    def test_empty_response_body_yields_empty_dict(self, gh_context):
        """A 204 has no body, and calling .json() on it would raise."""
        with stub_request({}, content=b""):
            assert _admin_request(gh_context, "DELETE", "posts/p1") == {}

    def test_http_error_propagates(self, gh_context):
        response = MagicMock()
        response.raise_for_status.side_effect = requests.HTTPError("422")

        with patch("ghost.ghost.requests.request", return_value=response):
            with pytest.raises(requests.HTTPError):
                _admin_request(gh_context, "POST", "posts")


# ---- create_post ----


class TestCreatePost:
    @pytest.mark.asyncio
    async def test_creates_post(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}):
            result = await CreatePostAction().execute({"title": "Hello"}, gh_context)

        assert result.data["result"] is True
        assert result.data["post"]["id"] == "p1"

    @pytest.mark.asyncio
    async def test_body_is_wrapped_in_posts_array(self, gh_context):
        """The Admin API takes a single-element `posts` array, not a bare object."""
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await CreatePostAction().execute({"title": "Hello"}, gh_context)

        body = mock_req.call_args.kwargs["json"]
        assert list(body) == ["posts"]
        assert body["posts"][0]["title"] == "Hello"

    @pytest.mark.asyncio
    async def test_status_defaults_to_draft(self, gh_context):
        """Defaulting to draft prevents an accidental publish."""
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await CreatePostAction().execute({"title": "Hello"}, gh_context)

        assert mock_req.call_args.kwargs["json"]["posts"][0]["status"] == "draft"

    @pytest.mark.asyncio
    async def test_explicit_status_wins(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await CreatePostAction().execute({"title": "Hello", "status": "published"}, gh_context)

        assert mock_req.call_args.kwargs["json"]["posts"][0]["status"] == "published"

    @pytest.mark.asyncio
    async def test_html_sets_source_param(self, gh_context):
        """Without `?source=html` Ghost ignores the html field entirely."""
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await CreatePostAction().execute({"title": "Hello", "html": "<p>Hi</p>"}, gh_context)

        assert mock_req.call_args.kwargs["params"] == {"source": "html"}

    @pytest.mark.asyncio
    async def test_no_html_sends_no_source_param(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await CreatePostAction().execute({"title": "Hello", "lexical": "{}"}, gh_context)

        assert mock_req.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_optional_fields_forwarded(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await CreatePostAction().execute(
                {
                    "title": "Hello",
                    "tags": ["news"],
                    "authors": ["ada"],
                    "feature_image": "https://x/i.png",
                    "excerpt": "summary",
                },
                gh_context,
            )

        post = mock_req.call_args.kwargs["json"]["posts"][0]
        assert post["tags"] == ["news"]
        assert post["authors"] == ["ada"]
        assert post["feature_image"] == "https://x/i.png"
        assert post["excerpt"] == "summary"

    @pytest.mark.asyncio
    async def test_empty_string_fields_are_kept(self, gh_context):
        """Fields are gated on `is not None`, so an empty excerpt is sent -- which
        is how a caller clears one."""
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await CreatePostAction().execute({"title": "Hello", "excerpt": ""}, gh_context)

        assert mock_req.call_args.kwargs["json"]["posts"][0]["excerpt"] == ""

    @pytest.mark.asyncio
    async def test_missing_title_is_captured(self, gh_context):
        result = await CreatePostAction().execute({}, gh_context)

        assert result.data["result"] is False

    @pytest.mark.asyncio
    async def test_ghost_validation_error_is_unwrapped(self, gh_context):
        exc = requests.HTTPError("422")
        exc.response = MagicMock()
        exc.response.json.return_value = {
            "errors": [{"message": "Title cannot be blank", "errorType": "ValidationError"}]
        }

        with patch("ghost.ghost.requests.request", side_effect=exc):
            result = await CreatePostAction().execute({"title": ""}, gh_context)

        assert result.data["error"] == "Title cannot be blank"
        assert result.data["error_type"] == "ValidationError"


# ---- update_post ----


class TestUpdatePost:
    @pytest.mark.asyncio
    async def test_updates_post(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}):
            result = await UpdatePostAction().execute(
                {"id": "p1", "updated_at": UPDATED_AT, "title": "New"}, gh_context
            )

        assert result.data["result"] is True

    @pytest.mark.asyncio
    async def test_uses_put_to_post_id(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await UpdatePostAction().execute({"id": "p1", "updated_at": UPDATED_AT}, gh_context)

        assert mock_req.call_args.args[0] == "PUT"
        assert mock_req.call_args.args[1] == f"{API_URL}/ghost/api/admin/posts/p1/"

    @pytest.mark.asyncio
    async def test_updated_at_is_always_sent(self, gh_context):
        """Ghost uses updated_at for collision detection and rejects updates
        without it."""
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await UpdatePostAction().execute({"id": "p1", "updated_at": UPDATED_AT}, gh_context)

        assert mock_req.call_args.kwargs["json"]["posts"][0]["updated_at"] == UPDATED_AT

    @pytest.mark.asyncio
    async def test_status_is_not_defaulted_on_update(self, gh_context):
        """Unlike create, update must not inject a status -- doing so would
        silently unpublish a live post."""
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await UpdatePostAction().execute({"id": "p1", "updated_at": UPDATED_AT}, gh_context)

        assert "status" not in mock_req.call_args.kwargs["json"]["posts"][0]

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await UpdatePostAction().execute(
                {"id": "p1", "updated_at": UPDATED_AT, "title": "New"}, gh_context
            )

        assert set(mock_req.call_args.kwargs["json"]["posts"][0]) == {"updated_at", "title"}

    @pytest.mark.asyncio
    async def test_html_sets_source_param(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}) as mock_req:
            await UpdatePostAction().execute(
                {"id": "p1", "updated_at": UPDATED_AT, "html": "<p>x</p>"}, gh_context
            )

        assert mock_req.call_args.kwargs["params"] == {"source": "html"}

    @pytest.mark.asyncio
    async def test_missing_updated_at_is_captured(self, gh_context):
        result = await UpdatePostAction().execute({"id": "p1"}, gh_context)

        assert result.data["result"] is False


# ---- create_page ----


class TestCreatePage:
    @pytest.mark.asyncio
    async def test_creates_page(self, gh_context):
        with stub_request({"pages": [{"id": "g1", "title": "About"}]}):
            result = await CreatePageAction().execute({"title": "About"}, gh_context)

        assert result.data["page"]["id"] == "g1"

    @pytest.mark.asyncio
    async def test_body_wrapped_in_pages_array(self, gh_context):
        with stub_request({"pages": []}) as mock_req:
            await CreatePageAction().execute({"title": "About"}, gh_context)

        assert mock_req.call_args.args[1] == f"{API_URL}/ghost/api/admin/pages/"
        assert list(mock_req.call_args.kwargs["json"]) == ["pages"]

    @pytest.mark.asyncio
    async def test_status_defaults_to_draft(self, gh_context):
        with stub_request({"pages": []}) as mock_req:
            await CreatePageAction().execute({"title": "About"}, gh_context)

        assert mock_req.call_args.kwargs["json"]["pages"][0]["status"] == "draft"

    @pytest.mark.asyncio
    async def test_supports_fewer_fields_than_create_post(self, gh_context):
        """Pages only forward html/lexical/status -- tags and excerpt are dropped."""
        with stub_request({"pages": []}) as mock_req:
            await CreatePageAction().execute(
                {"title": "About", "tags": ["x"], "excerpt": "y"}, gh_context
            )

        page = mock_req.call_args.kwargs["json"]["pages"][0]
        assert "tags" not in page
        assert "excerpt" not in page

    @pytest.mark.asyncio
    async def test_empty_result_yields_none(self, gh_context):
        with stub_request({"pages": []}):
            result = await CreatePageAction().execute({"title": "About"}, gh_context)

        assert result.data["page"] is None


# ---- upload_image ----


class TestUploadImage:
    @pytest.fixture
    def png(self, tmp_path):
        path = tmp_path / "logo.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n")
        return path

    @pytest.mark.asyncio
    async def test_uploads_image(self, gh_context, png):
        with stub_request({"images": [{"url": f"{API_URL}/content/images/logo.png"}]}):
            result = await UploadImageAction().execute({"file_path": str(png)}, gh_context)

        assert result.data["result"] is True
        assert result.data["image"]["url"].endswith("logo.png")

    @pytest.mark.asyncio
    async def test_targets_upload_endpoint(self, gh_context, png):
        with stub_request({"images": []}) as mock_req:
            await UploadImageAction().execute({"file_path": str(png)}, gh_context)

        assert mock_req.call_args.args[1] == f"{API_URL}/ghost/api/admin/images/upload/"
        assert mock_req.call_args.args[0] == "POST"

    @pytest.mark.asyncio
    async def test_mime_type_guessed_from_extension(self, gh_context, png):
        with stub_request({"images": []}) as mock_req:
            await UploadImageAction().execute({"file_path": str(png)}, gh_context)

        assert mock_req.call_args.kwargs["files"]["file"][2] == "image/png"

    @pytest.mark.asyncio
    async def test_unknown_extension_falls_back_to_octet_stream(self, gh_context, tmp_path):
        path = tmp_path / "blob.weirdext"
        path.write_bytes(b"data")

        with stub_request({"images": []}) as mock_req:
            await UploadImageAction().execute({"file_path": str(path)}, gh_context)

        assert mock_req.call_args.kwargs["files"]["file"][2] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_filename_is_basename_only(self, gh_context, png):
        """The full path must not leak into the multipart filename."""
        with stub_request({"images": []}) as mock_req:
            await UploadImageAction().execute({"file_path": str(png)}, gh_context)

        assert mock_req.call_args.kwargs["files"]["file"][0] == "logo.png"

    @pytest.mark.asyncio
    async def test_purpose_defaults_to_image(self, gh_context, png):
        with stub_request({"images": []}) as mock_req:
            await UploadImageAction().execute({"file_path": str(png)}, gh_context)

        assert mock_req.call_args.kwargs["files"]["purpose"] == (None, "image")

    @pytest.mark.asyncio
    async def test_explicit_purpose_forwarded(self, gh_context, png):
        with stub_request({"images": []}) as mock_req:
            await UploadImageAction().execute(
                {"file_path": str(png), "purpose": "profile_image"}, gh_context
            )

        assert mock_req.call_args.kwargs["files"]["purpose"] == (None, "profile_image")

    @pytest.mark.asyncio
    async def test_missing_file_is_reported_as_file_not_found(self, gh_context, tmp_path):
        result = await UploadImageAction().execute(
            {"file_path": str(tmp_path / "nope.png")}, gh_context
        )

        assert result.data["result"] is False
        assert result.data["error_type"] == "FileNotFoundError"


# ---- create_member / update_member ----


class TestCreateMember:
    @pytest.mark.asyncio
    async def test_creates_member(self, gh_context):
        with stub_request({"members": [SAMPLE_MEMBER]}):
            result = await CreateMemberAction().execute({"email": "ada@example.com"}, gh_context)

        assert result.data["member"]["id"] == "m1"

    @pytest.mark.asyncio
    async def test_body_wrapped_in_members_array(self, gh_context):
        with stub_request({"members": []}) as mock_req:
            await CreateMemberAction().execute({"email": "ada@example.com"}, gh_context)

        assert mock_req.call_args.args[1] == f"{API_URL}/ghost/api/admin/members/"
        assert mock_req.call_args.kwargs["json"]["members"][0]["email"] == "ada@example.com"

    @pytest.mark.asyncio
    async def test_optional_fields_forwarded(self, gh_context):
        with stub_request({"members": []}) as mock_req:
            await CreateMemberAction().execute(
                {
                    "email": "ada@example.com",
                    "name": "Ada",
                    "labels": ["vip"],
                    "newsletters": [{"id": "n1"}],
                    "note": "referral",
                },
                gh_context,
            )

        member = mock_req.call_args.kwargs["json"]["members"][0]
        assert member["name"] == "Ada"
        assert member["labels"] == ["vip"]
        assert member["newsletters"] == [{"id": "n1"}]
        assert member["note"] == "referral"

    @pytest.mark.asyncio
    async def test_minimal_body_is_email_only(self, gh_context):
        with stub_request({"members": []}) as mock_req:
            await CreateMemberAction().execute({"email": "ada@example.com"}, gh_context)

        assert set(mock_req.call_args.kwargs["json"]["members"][0]) == {"email"}

    @pytest.mark.asyncio
    async def test_missing_email_is_captured(self, gh_context):
        result = await CreateMemberAction().execute({}, gh_context)

        assert result.data["result"] is False


class TestUpdateMember:
    @pytest.mark.asyncio
    async def test_uses_put_to_member_id(self, gh_context):
        with stub_request({"members": [SAMPLE_MEMBER]}) as mock_req:
            await UpdateMemberAction().execute({"id": "m1", "name": "Ada L"}, gh_context)

        assert mock_req.call_args.args[0] == "PUT"
        assert mock_req.call_args.args[1] == f"{API_URL}/ghost/api/admin/members/m1/"

    @pytest.mark.asyncio
    async def test_email_is_updatable(self, gh_context):
        """Unlike create, email is optional here -- it's one of the mutable fields."""
        with stub_request({"members": []}) as mock_req:
            await UpdateMemberAction().execute({"id": "m1", "email": "new@example.com"}, gh_context)

        assert mock_req.call_args.kwargs["json"]["members"][0]["email"] == "new@example.com"

    @pytest.mark.asyncio
    async def test_no_fields_sends_empty_member_object(self, gh_context):
        """An id-only update sends `{"members": [{}]}` -- a no-op request rather
        than a validation error."""
        with stub_request({"members": []}) as mock_req:
            await UpdateMemberAction().execute({"id": "m1"}, gh_context)

        assert mock_req.call_args.kwargs["json"] == {"members": [{}]}

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, gh_context):
        result = await UpdateMemberAction().execute({"name": "Ada"}, gh_context)

        assert result.data["result"] is False


# ---- send_newsletter ----


class TestSendNewsletter:
    @pytest.mark.asyncio
    async def test_sends_newsletter(self, gh_context):
        with stub_request({"posts": [SAMPLE_POST]}):
            result = await SendNewsletterAction().execute(
                {"post_id": "p1", "updated_at": UPDATED_AT, "newsletter_slug": "weekly"},
                gh_context,
            )

        assert result.data["result"] is True

    @pytest.mark.asyncio
    async def test_newsletter_slug_goes_in_query_params(self, gh_context):
        """The slug selects the recipient list and is a query param, not a body
        field."""
        with stub_request({"posts": []}) as mock_req:
            await SendNewsletterAction().execute(
                {"post_id": "p1", "updated_at": UPDATED_AT, "newsletter_slug": "weekly"},
                gh_context,
            )

        assert mock_req.call_args.kwargs["params"] == {"newsletter": "weekly"}

    @pytest.mark.asyncio
    async def test_forces_published_status(self, gh_context):
        """Email delivery is triggered by the draft to published transition, so
        the handler always sends status=published."""
        with stub_request({"posts": []}) as mock_req:
            await SendNewsletterAction().execute(
                {"post_id": "p1", "updated_at": UPDATED_AT, "newsletter_slug": "weekly"},
                gh_context,
            )

        assert mock_req.call_args.kwargs["json"]["posts"][0]["status"] == "published"

    @pytest.mark.asyncio
    async def test_uses_put_to_post_id(self, gh_context):
        with stub_request({"posts": []}) as mock_req:
            await SendNewsletterAction().execute(
                {"post_id": "p1", "updated_at": UPDATED_AT, "newsletter_slug": "weekly"},
                gh_context,
            )

        assert mock_req.call_args.args[0] == "PUT"
        assert mock_req.call_args.args[1] == f"{API_URL}/ghost/api/admin/posts/p1/"

    @pytest.mark.asyncio
    async def test_body_carries_only_status_and_updated_at(self, gh_context):
        with stub_request({"posts": []}) as mock_req:
            await SendNewsletterAction().execute(
                {"post_id": "p1", "updated_at": UPDATED_AT, "newsletter_slug": "weekly"},
                gh_context,
            )

        assert set(mock_req.call_args.kwargs["json"]["posts"][0]) == {"status", "updated_at"}

    @pytest.mark.parametrize("missing", ["post_id", "updated_at", "newsletter_slug"])
    @pytest.mark.asyncio
    async def test_all_three_inputs_are_required(self, gh_context, missing):
        inputs = {"post_id": "p1", "updated_at": UPDATED_AT, "newsletter_slug": "weekly"}
        del inputs[missing]

        result = await SendNewsletterAction().execute(inputs, gh_context)

        assert result.data["result"] is False


# ---- Config ----


class TestGhostAdminConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize("action", ["create_post", "create_page"])
    def test_create_actions_require_title(self, config, action):
        assert "title" in config["actions"][action]["input_schema"]["required"]

    def test_update_post_requires_id_and_updated_at(self, config):
        required = config["actions"]["update_post"]["input_schema"]["required"]
        assert "id" in required
        assert "updated_at" in required

    def test_create_member_requires_email(self, config):
        assert "email" in config["actions"]["create_member"]["input_schema"]["required"]

    def test_send_newsletter_requires_all_three_inputs(self, config):
        required = config["actions"]["send_newsletter"]["input_schema"]["required"]
        assert sorted(required) == ["newsletter_slug", "post_id", "updated_at"]

    def test_upload_image_requires_file_path(self, config):
        assert "file_path" in config["actions"]["upload_image"]["input_schema"]["required"]
