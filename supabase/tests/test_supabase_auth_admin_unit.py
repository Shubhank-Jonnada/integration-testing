"""Unit tests for the Supabase Auth Admin (GoTrue) actions.

Covers `list_users`, `get_user`, and `delete_user`. These hit the
`/auth/v1/admin` surface, which requires the service-role secret and returns a
paginated envelope rather than a bare array.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from supabase.supabase import (  # noqa: E402
    DeleteUserAction,
    GetUserAction,
    ListUsersAction,
)

pytestmark = pytest.mark.unit

SERVICE_KEY = "test_service_role_secret"  # nosec B105
HOST = "https://abcdefgh.supabase.co"
USER_ID = "3f8c1a2e-9b4d-4c7f-8e15-2a6b0d9f3c11"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_USERS = [
    {"id": USER_ID, "email": "ada@example.com", "created_at": "2026-01-02T00:00:00Z"},
    {"id": "b1c2d3e4-0000-0000-0000-000000000000", "email": "grace@example.com"},
]


@pytest.fixture
def sb_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"host": HOST, "service_role_secret": SERVICE_KEY}}
    return ctx


# ---- list_users ----


class TestListUsers:
    @pytest.mark.asyncio
    async def test_returns_users_and_total(self, sb_context):
        sb_context.fetch.return_value = {"users": SAMPLE_USERS, "total": 2}

        result = await ListUsersAction().execute({}, sb_context)

        assert result.data["result"] is True
        assert result.data["users"] == SAMPLE_USERS
        assert result.data["total"] == 2

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sb_context):
        sb_context.fetch.return_value = {"users": []}

        await ListUsersAction().execute({}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/auth/v1/admin/users"
        assert sb_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_requires_service_role_authorization(self, sb_context):
        """Admin endpoints reject the anon key -- the Bearer token must be the
        service-role secret."""
        sb_context.fetch.return_value = {"users": []}

        await ListUsersAction().execute({}, sb_context)

        headers = sb_context.fetch.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {SERVICE_KEY}"
        assert headers["apikey"] == SERVICE_KEY

    @pytest.mark.asyncio
    async def test_pagination_forwarded(self, sb_context):
        sb_context.fetch.return_value = {"users": []}

        await ListUsersAction().execute({"page": 2, "per_page": 50}, sb_context)

        assert sb_context.fetch.call_args.kwargs["params"] == {"page": 2, "per_page": 50}

    @pytest.mark.asyncio
    async def test_no_pagination_sends_none_params(self, sb_context):
        sb_context.fetch.return_value = {"users": []}

        await ListUsersAction().execute({}, sb_context)

        assert sb_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_page_zero_is_dropped(self, sb_context):
        """Pagination is truthiness-gated, so page=0 is discarded rather than
        sent. GoTrue pages are 1-indexed, so this is harmless -- but asserted so
        the gating is deliberate."""
        sb_context.fetch.return_value = {"users": []}

        await ListUsersAction().execute({"page": 0}, sb_context)

        assert sb_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_total_defaults_to_user_count(self, sb_context):
        """GoTrue omits `total` on some versions; fall back to what was returned."""
        sb_context.fetch.return_value = {"users": SAMPLE_USERS}

        result = await ListUsersAction().execute({}, sb_context)

        assert result.data["total"] == 2

    @pytest.mark.asyncio
    async def test_non_dict_response_yields_empty(self, sb_context):
        """A bare array would mean the envelope shape changed -- coerce rather
        than crash."""
        sb_context.fetch.return_value = SAMPLE_USERS

        result = await ListUsersAction().execute({}, sb_context)

        assert result.data["users"] == []
        assert result.data["total"] == 0

    @pytest.mark.asyncio
    async def test_empty_envelope(self, sb_context):
        sb_context.fetch.return_value = {}

        result = await ListUsersAction().execute({}, sb_context)

        assert result.data["users"] == []
        assert result.data["total"] == 0
        assert result.data["result"] is True

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("HTTP 403: not_admin")

        result = await ListUsersAction().execute({}, sb_context)

        assert result.data["result"] is False
        assert result.data["users"] == []
        assert result.data["total"] == 0
        assert "not_admin" in result.data["error"]


# ---- get_user ----


class TestGetUser:
    @pytest.mark.asyncio
    async def test_returns_user(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_USERS[0]

        result = await GetUserAction().execute({"user_id": USER_ID}, sb_context)

        assert result.data["result"] is True
        assert result.data["user"]["email"] == "ada@example.com"

    @pytest.mark.asyncio
    async def test_request_url_includes_user_id(self, sb_context):
        sb_context.fetch.return_value = {}

        await GetUserAction().execute({"user_id": USER_ID}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/auth/v1/admin/users/{USER_ID}"
        assert sb_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_sends_no_params_or_body(self, sb_context):
        sb_context.fetch.return_value = {}

        await GetUserAction().execute({"user_id": USER_ID}, sb_context)

        kwargs = sb_context.fetch.call_args.kwargs
        assert "params" not in kwargs
        assert "json" not in kwargs

    @pytest.mark.asyncio
    async def test_response_passed_through_unwrapped(self, sb_context):
        """GoTrue returns the user object directly, not wrapped in a `user` key."""
        sb_context.fetch.return_value = {"id": USER_ID, "app_metadata": {"provider": "email"}}

        result = await GetUserAction().execute({"user_id": USER_ID}, sb_context)

        assert result.data["user"]["app_metadata"]["provider"] == "email"

    @pytest.mark.asyncio
    async def test_missing_user_id_is_captured(self, sb_context):
        result = await GetUserAction().execute({}, sb_context)

        assert result.data["result"] is False
        assert result.data["user"] == {}
        sb_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("HTTP 404: User not found")

        result = await GetUserAction().execute({"user_id": "nope"}, sb_context)

        assert result.data["result"] is False
        assert result.data["user"] == {}
        assert "User not found" in result.data["error"]


# ---- delete_user ----


class TestDeleteUser:
    @pytest.mark.asyncio
    async def test_reports_deleted(self, sb_context):
        sb_context.fetch.return_value = {}

        result = await DeleteUserAction().execute({"user_id": USER_ID}, sb_context)

        assert result.data["result"] is True
        assert result.data["deleted"] is True

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sb_context):
        sb_context.fetch.return_value = {}

        await DeleteUserAction().execute({"user_id": USER_ID}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/auth/v1/admin/users/{USER_ID}"
        assert sb_context.fetch.call_args.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_content_type_removed_for_bodyless_delete(self, sb_context):
        sb_context.fetch.return_value = {}

        await DeleteUserAction().execute({"user_id": USER_ID}, sb_context)

        headers = sb_context.fetch.call_args.kwargs["headers"]
        assert "Content-Type" not in headers
        assert headers["Authorization"] == f"Bearer {SERVICE_KEY}"

    @pytest.mark.asyncio
    async def test_sends_no_body(self, sb_context):
        sb_context.fetch.return_value = {}

        await DeleteUserAction().execute({"user_id": USER_ID}, sb_context)

        assert "json" not in sb_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_response_body_is_ignored(self, sb_context):
        """Success is inferred from the absence of an exception, so whatever the
        API returns doesn't change the outcome."""
        sb_context.fetch.return_value = {"unexpected": "payload"}

        result = await DeleteUserAction().execute({"user_id": USER_ID}, sb_context)

        assert result.data["deleted"] is True

    @pytest.mark.asyncio
    async def test_missing_user_id_is_captured(self, sb_context):
        result = await DeleteUserAction().execute({}, sb_context)

        assert result.data["result"] is False
        assert result.data["deleted"] is False
        sb_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_reports_not_deleted(self, sb_context):
        sb_context.fetch.side_effect = Exception("HTTP 403: not_admin")

        result = await DeleteUserAction().execute({"user_id": USER_ID}, sb_context)

        assert result.data["result"] is False
        assert result.data["deleted"] is False
        assert "not_admin" in result.data["error"]


# ---- Config ----


class TestSupabaseAuthAdminConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize("action", ["get_user", "delete_user"])
    def test_user_scoped_actions_require_user_id(self, config, action):
        assert "user_id" in config["actions"][action]["input_schema"]["required"]

    def test_list_users_requires_nothing(self, config):
        """Listing is unfiltered by default, so no required inputs."""
        schema = config["actions"]["list_users"]["input_schema"]
        assert not schema.get("required")

    def test_list_users_exposes_pagination(self, config):
        props = config["actions"]["list_users"]["input_schema"]["properties"]
        assert "page" in props
        assert "per_page" in props

    def test_service_role_secret_is_a_password_field(self, config):
        """It grants admin access, so it must never render in plain text."""
        field = config["auth"]["fields"]["properties"]["service_role_secret"]
        assert field["format"] == "password"
