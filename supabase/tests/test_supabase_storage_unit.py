"""Unit tests for the Supabase Storage actions.

Covers bucket CRUD (`list_buckets`, `get_bucket`, `create_bucket`,
`delete_bucket`), object operations (`list_files`, `delete_files`), and the
purely local `get_public_url`.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from supabase.supabase import (  # noqa: E402
    CreateBucketAction,
    DeleteBucketAction,
    DeleteFilesAction,
    GetBucketAction,
    GetPublicUrlAction,
    ListBucketsAction,
    ListFilesAction,
)

pytestmark = pytest.mark.unit

SERVICE_KEY = "test_service_role_secret"  # nosec B105
HOST = "https://abcdefgh.supabase.co"
BUCKET = "avatars"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_BUCKET = {"id": BUCKET, "name": BUCKET, "public": True, "file_size_limit": None}
SAMPLE_FILES = [
    {"name": "a.png", "id": "f1", "metadata": {"size": 1024}},
    {"name": "b.png", "id": "f2", "metadata": {"size": 2048}},
]


@pytest.fixture
def sb_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"host": HOST, "service_role_secret": SERVICE_KEY}}
    return ctx


# ---- list_buckets ----


class TestListBuckets:
    @pytest.mark.asyncio
    async def test_returns_buckets(self, sb_context):
        sb_context.fetch.return_value = [SAMPLE_BUCKET]

        result = await ListBucketsAction().execute({}, sb_context)

        assert result.data["result"] is True
        assert result.data["buckets"] == [SAMPLE_BUCKET]

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sb_context):
        sb_context.fetch.return_value = []

        await ListBucketsAction().execute({}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/storage/v1/bucket"
        assert sb_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_sends_service_role_key(self, sb_context):
        sb_context.fetch.return_value = []

        await ListBucketsAction().execute({}, sb_context)

        headers = sb_context.fetch.call_args.kwargs["headers"]
        assert headers["apikey"] == SERVICE_KEY
        assert headers["Authorization"] == f"Bearer {SERVICE_KEY}"

    @pytest.mark.asyncio
    async def test_non_list_response_yields_empty(self, sb_context):
        sb_context.fetch.return_value = {"error": "unauthorized"}

        result = await ListBucketsAction().execute({}, sb_context)

        assert result.data["buckets"] == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListBucketsAction().execute({}, sb_context)

        assert result.data["result"] is False
        assert result.data["buckets"] == []


# ---- get_bucket ----


class TestGetBucket:
    @pytest.mark.asyncio
    async def test_returns_bucket(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_BUCKET

        result = await GetBucketAction().execute({"bucket_id": BUCKET}, sb_context)

        assert result.data["result"] is True
        assert result.data["bucket"]["name"] == BUCKET

    @pytest.mark.asyncio
    async def test_request_url_includes_bucket_id(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_BUCKET

        await GetBucketAction().execute({"bucket_id": BUCKET}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/storage/v1/bucket/{BUCKET}"
        assert sb_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_bucket_id_is_captured(self, sb_context):
        result = await GetBucketAction().execute({}, sb_context)

        assert result.data["result"] is False
        assert result.data["bucket"] == {}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("Bucket not found")

        result = await GetBucketAction().execute({"bucket_id": "nope"}, sb_context)

        assert result.data["result"] is False
        assert "Bucket not found" in result.data["error"]


# ---- create_bucket ----


class TestCreateBucket:
    @pytest.mark.asyncio
    async def test_creates_bucket(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_BUCKET

        result = await CreateBucketAction().execute({"name": BUCKET}, sb_context)

        assert result.data["result"] is True
        assert result.data["bucket"]["id"] == BUCKET

    @pytest.mark.asyncio
    async def test_name_is_used_as_both_id_and_name(self, sb_context):
        """Supabase requires an explicit id; the handler mirrors the name into it."""
        sb_context.fetch.return_value = {}

        await CreateBucketAction().execute({"name": BUCKET}, sb_context)

        body = sb_context.fetch.call_args.kwargs["json"]
        assert body["id"] == BUCKET
        assert body["name"] == BUCKET

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sb_context):
        sb_context.fetch.return_value = {}

        await CreateBucketAction().execute({"name": BUCKET}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/storage/v1/bucket"
        assert sb_context.fetch.call_args.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_minimal_body_omits_optionals(self, sb_context):
        sb_context.fetch.return_value = {}

        await CreateBucketAction().execute({"name": BUCKET}, sb_context)

        body = sb_context.fetch.call_args.kwargs["json"]
        assert set(body) == {"id", "name"}

    @pytest.mark.asyncio
    async def test_public_false_is_preserved(self, sb_context):
        """`public` is gated on `is not None`, so an explicit False survives --
        unlike the other optional fields, which are truthiness-gated."""
        sb_context.fetch.return_value = {}

        await CreateBucketAction().execute({"name": BUCKET, "public": False}, sb_context)

        assert sb_context.fetch.call_args.kwargs["json"]["public"] is False

    @pytest.mark.asyncio
    async def test_public_true_is_preserved(self, sb_context):
        sb_context.fetch.return_value = {}

        await CreateBucketAction().execute({"name": BUCKET, "public": True}, sb_context)

        assert sb_context.fetch.call_args.kwargs["json"]["public"] is True

    @pytest.mark.asyncio
    async def test_size_limit_and_mime_types_forwarded(self, sb_context):
        sb_context.fetch.return_value = {}

        await CreateBucketAction().execute(
            {"name": BUCKET, "file_size_limit": 5242880, "allowed_mime_types": ["image/png"]},
            sb_context,
        )

        body = sb_context.fetch.call_args.kwargs["json"]
        assert body["file_size_limit"] == 5242880
        assert body["allowed_mime_types"] == ["image/png"]

    @pytest.mark.asyncio
    async def test_zero_size_limit_is_dropped(self, sb_context):
        """file_size_limit is truthiness-gated, so 0 is silently discarded."""
        sb_context.fetch.return_value = {}

        await CreateBucketAction().execute({"name": BUCKET, "file_size_limit": 0}, sb_context)

        assert "file_size_limit" not in sb_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("Bucket already exists")

        result = await CreateBucketAction().execute({"name": BUCKET}, sb_context)

        assert result.data["result"] is False
        assert result.data["bucket"] == {}


# ---- delete_bucket ----


class TestDeleteBucket:
    @pytest.mark.asyncio
    async def test_reports_deleted(self, sb_context):
        sb_context.fetch.return_value = {}

        result = await DeleteBucketAction().execute({"bucket_id": BUCKET}, sb_context)

        assert result.data["result"] is True
        assert result.data["deleted"] is True

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sb_context):
        sb_context.fetch.return_value = {}

        await DeleteBucketAction().execute({"bucket_id": BUCKET}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/storage/v1/bucket/{BUCKET}"
        assert sb_context.fetch.call_args.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_content_type_removed_for_bodyless_delete(self, sb_context):
        """Sending Content-Type: application/json with no body makes some
        gateways expect one."""
        sb_context.fetch.return_value = {}

        await DeleteBucketAction().execute({"bucket_id": BUCKET}, sb_context)

        headers = sb_context.fetch.call_args.kwargs["headers"]
        assert "Content-Type" not in headers
        assert headers["apikey"] == SERVICE_KEY

    @pytest.mark.asyncio
    async def test_sends_no_body(self, sb_context):
        sb_context.fetch.return_value = {}

        await DeleteBucketAction().execute({"bucket_id": BUCKET}, sb_context)

        assert "json" not in sb_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_error_reports_not_deleted(self, sb_context):
        sb_context.fetch.side_effect = Exception("Bucket not empty")

        result = await DeleteBucketAction().execute({"bucket_id": BUCKET}, sb_context)

        assert result.data["result"] is False
        assert result.data["deleted"] is False


# ---- list_files ----


class TestListFiles:
    @pytest.mark.asyncio
    async def test_returns_files(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_FILES

        result = await ListFilesAction().execute({"bucket_id": BUCKET}, sb_context)

        assert result.data["result"] is True
        assert len(result.data["files"]) == 2

    @pytest.mark.asyncio
    async def test_request_uses_post_to_list_endpoint(self, sb_context):
        """Listing objects is a POST with a JSON body, not a GET."""
        sb_context.fetch.return_value = []

        await ListFilesAction().execute({"bucket_id": BUCKET}, sb_context)

        call = sb_context.fetch.call_args
        assert call.args[0] == f"{HOST}/storage/v1/object/list/{BUCKET}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_default_body(self, sb_context):
        sb_context.fetch.return_value = []

        await ListFilesAction().execute({"bucket_id": BUCKET}, sb_context)

        assert sb_context.fetch.call_args.kwargs["json"] == {"prefix": "", "limit": 100, "offset": 0}

    @pytest.mark.asyncio
    async def test_path_maps_to_prefix(self, sb_context):
        """The input is named `path` but the API field is `prefix`."""
        sb_context.fetch.return_value = []

        await ListFilesAction().execute({"bucket_id": BUCKET, "path": "uploads/2026"}, sb_context)

        assert sb_context.fetch.call_args.kwargs["json"]["prefix"] == "uploads/2026"

    @pytest.mark.asyncio
    async def test_pagination_forwarded(self, sb_context):
        sb_context.fetch.return_value = []

        await ListFilesAction().execute({"bucket_id": BUCKET, "limit": 10, "offset": 30}, sb_context)

        body = sb_context.fetch.call_args.kwargs["json"]
        assert body["limit"] == 10
        assert body["offset"] == 30

    @pytest.mark.asyncio
    async def test_search_only_sent_when_present(self, sb_context):
        sb_context.fetch.return_value = []

        await ListFilesAction().execute({"bucket_id": BUCKET}, sb_context)
        assert "search" not in sb_context.fetch.call_args.kwargs["json"]

        await ListFilesAction().execute({"bucket_id": BUCKET, "search": "logo"}, sb_context)
        assert sb_context.fetch.call_args.kwargs["json"]["search"] == "logo"

    @pytest.mark.asyncio
    async def test_non_list_response_yields_empty(self, sb_context):
        sb_context.fetch.return_value = {"error": "not found"}

        result = await ListFilesAction().execute({"bucket_id": BUCKET}, sb_context)

        assert result.data["files"] == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("HTTP 400")

        result = await ListFilesAction().execute({"bucket_id": BUCKET}, sb_context)

        assert result.data["result"] is False
        assert result.data["files"] == []


# ---- delete_files ----


class TestDeleteFiles:
    @pytest.mark.asyncio
    async def test_returns_deleted_objects(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_FILES

        result = await DeleteFilesAction().execute(
            {"bucket_id": BUCKET, "paths": ["a.png", "b.png"]}, sb_context
        )

        assert result.data["result"] is True
        assert len(result.data["deleted"]) == 2

    @pytest.mark.asyncio
    async def test_request_url_method_and_body(self, sb_context):
        """Paths are sent as `prefixes`, not `paths`."""
        sb_context.fetch.return_value = []

        await DeleteFilesAction().execute({"bucket_id": BUCKET, "paths": ["a.png"]}, sb_context)

        call = sb_context.fetch.call_args
        assert call.args[0] == f"{HOST}/storage/v1/object/{BUCKET}"
        assert call.kwargs["method"] == "DELETE"
        assert call.kwargs["json"] == {"prefixes": ["a.png"]}

    @pytest.mark.asyncio
    async def test_api_level_error_object_is_surfaced(self, sb_context):
        """Storage returns 200 with an error body in some failure modes, so a
        dict carrying `error` is treated as a failure rather than as data."""
        sb_context.fetch.return_value = {"error": "InvalidRequest", "message": "Object not found"}

        result = await DeleteFilesAction().execute(
            {"bucket_id": BUCKET, "paths": ["missing.png"]}, sb_context
        )

        assert result.data["result"] is False
        assert result.data["error"] == "Object not found"
        assert result.data["deleted"] == []

    @pytest.mark.asyncio
    async def test_error_object_without_message_falls_back_to_error(self, sb_context):
        sb_context.fetch.return_value = {"error": "InvalidRequest"}

        result = await DeleteFilesAction().execute(
            {"bucket_id": BUCKET, "paths": ["x.png"]}, sb_context
        )

        assert result.data["error"] == "InvalidRequest"

    @pytest.mark.asyncio
    async def test_dict_without_error_key_yields_empty_deleted(self, sb_context):
        sb_context.fetch.return_value = {"ok": True}

        result = await DeleteFilesAction().execute(
            {"bucket_id": BUCKET, "paths": ["x.png"]}, sb_context
        )

        assert result.data["result"] is True
        assert result.data["deleted"] == []

    @pytest.mark.asyncio
    async def test_empty_paths_list(self, sb_context):
        sb_context.fetch.return_value = []

        result = await DeleteFilesAction().execute({"bucket_id": BUCKET, "paths": []}, sb_context)

        assert sb_context.fetch.call_args.kwargs["json"] == {"prefixes": []}
        assert result.data["result"] is True

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("HTTP 403")

        result = await DeleteFilesAction().execute(
            {"bucket_id": BUCKET, "paths": ["a.png"]}, sb_context
        )

        assert result.data["result"] is False
        assert result.data["deleted"] == []


# ---- get_public_url ----


class TestGetPublicUrl:
    @pytest.mark.asyncio
    async def test_builds_public_url(self, sb_context):
        result = await GetPublicUrlAction().execute(
            {"bucket_id": BUCKET, "path": "profile/a.png"}, sb_context
        )

        assert result.data["result"] is True
        assert result.data["public_url"] == f"{HOST}/storage/v1/object/public/{BUCKET}/profile/a.png"

    @pytest.mark.asyncio
    async def test_makes_no_network_call(self, sb_context):
        """The URL is derived locally -- no request should be issued."""
        await GetPublicUrlAction().execute({"bucket_id": BUCKET, "path": "a.png"}, sb_context)

        sb_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_host_trailing_slash_does_not_double_up(self, sb_context):
        sb_context.auth = {"credentials": {"host": f"{HOST}/", "service_role_secret": SERVICE_KEY}}

        result = await GetPublicUrlAction().execute({"bucket_id": BUCKET, "path": "a.png"}, sb_context)

        assert "//storage" not in result.data["public_url"]

    @pytest.mark.asyncio
    async def test_missing_path_is_captured(self, sb_context):
        result = await GetPublicUrlAction().execute({"bucket_id": BUCKET}, sb_context)

        assert result.data["result"] is False
        assert result.data["public_url"] == ""

    @pytest.mark.asyncio
    async def test_nested_path_preserved_verbatim(self, sb_context):
        """Path segments are not URL-encoded, so a path with spaces is emitted
        as-is -- documented rather than assumed safe."""
        result = await GetPublicUrlAction().execute(
            {"bucket_id": BUCKET, "path": "my folder/a b.png"}, sb_context
        )

        assert result.data["public_url"].endswith("/my folder/a b.png")


# ---- Config ----


class TestSupabaseStorageConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        ["get_bucket", "delete_bucket", "list_files", "delete_files", "get_public_url"],
    )
    def test_bucket_scoped_actions_require_bucket_id(self, config, action):
        assert "bucket_id" in config["actions"][action]["input_schema"]["required"]

    def test_create_bucket_requires_name(self, config):
        assert "name" in config["actions"]["create_bucket"]["input_schema"]["required"]

    def test_delete_files_requires_paths(self, config):
        assert "paths" in config["actions"]["delete_files"]["input_schema"]["required"]

    def test_get_public_url_requires_path(self, config):
        assert "path" in config["actions"]["get_public_url"]["input_schema"]["required"]
