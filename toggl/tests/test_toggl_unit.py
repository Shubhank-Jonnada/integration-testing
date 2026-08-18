"""Unit tests for the Toggl Track integration.

Fully mocked -- no network access. Covers the Basic-auth header helper, the
`create_time_entry` request contract, payload construction, and config.json.
"""

import base64
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from toggl.toggl import (  # noqa: E402
    CreateTimeEntry,
    _basic_auth_header_for_api_token,
    toggl as toggl_integration,
)

pytestmark = pytest.mark.unit

API_TOKEN = "test_api_token"  # nosec B105
WORKSPACE_ID = 12345
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_TIME_ENTRY = {
    "id": 987654321,
    "workspace_id": WORKSPACE_ID,
    "description": "Writing tests",
    "start": "2026-01-15T09:00:00Z",
    "stop": "2026-01-15T10:30:00Z",
    "duration": 5400,
    "billable": False,
}


@pytest.fixture
def toggl_context():
    """Context carrying Toggl's custom-auth credential shape."""
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch", return_value=SAMPLE_TIME_ENTRY)
    ctx.auth = {"credentials": {"api_token": API_TOKEN}}
    return ctx


def minimal_inputs(**overrides):
    inputs = {"workspace_id": WORKSPACE_ID, "start": "2026-01-15T09:00:00Z"}
    inputs.update(overrides)
    return inputs


# ---- Auth Header Helper ----


class TestBasicAuthHeader:
    def test_encodes_token_with_api_token_password(self):
        """Toggl expects Basic auth with the literal password 'api_token'."""
        headers = _basic_auth_header_for_api_token(API_TOKEN)

        encoded = headers["Authorization"].removeprefix("Basic ")
        assert base64.b64decode(encoded).decode() == f"{API_TOKEN}:api_token"

    def test_authorization_uses_basic_scheme(self):
        headers = _basic_auth_header_for_api_token(API_TOKEN)
        assert headers["Authorization"].startswith("Basic ")

    def test_sets_json_content_type(self):
        assert _basic_auth_header_for_api_token(API_TOKEN)["Content-Type"] == "application/json"

    @pytest.mark.parametrize(
        "token",
        ["abc123", "token-with-dashes", "TOKEN_WITH_UNDERSCORES", "0" * 32],
    )
    def test_round_trips_arbitrary_tokens(self, token):
        headers = _basic_auth_header_for_api_token(token)
        decoded = base64.b64decode(headers["Authorization"].removeprefix("Basic ")).decode()
        assert decoded.startswith(f"{token}:")


# ---- create_time_entry ----


class TestCreateTimeEntry:
    @pytest.mark.asyncio
    async def test_returns_api_response(self, toggl_context):
        result = await CreateTimeEntry().execute(minimal_inputs(), toggl_context)
        assert result == SAMPLE_TIME_ENTRY

    @pytest.mark.asyncio
    async def test_request_url_targets_workspace_time_entries(self, toggl_context):
        await CreateTimeEntry().execute(minimal_inputs(), toggl_context)

        url = toggl_context.fetch.call_args.args[0]
        assert url == f"https://api.track.toggl.com/api/v9/workspaces/{WORKSPACE_ID}/time_entries"

    @pytest.mark.asyncio
    async def test_request_uses_post(self, toggl_context):
        await CreateTimeEntry().execute(minimal_inputs(), toggl_context)
        assert toggl_context.fetch.call_args.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_request_sends_auth_header(self, toggl_context):
        await CreateTimeEntry().execute(minimal_inputs(), toggl_context)

        headers = toggl_context.fetch.call_args.kwargs["headers"]
        assert headers == _basic_auth_header_for_api_token(API_TOKEN)

    @pytest.mark.asyncio
    async def test_payload_sets_created_with(self, toggl_context):
        """Toggl rejects entry creation without a created_with attribution."""
        await CreateTimeEntry().execute(minimal_inputs(), toggl_context)

        body = toggl_context.fetch.call_args.kwargs["json"]
        assert body["created_with"] == "autohive-integrations"

    @pytest.mark.asyncio
    async def test_payload_echoes_workspace_id(self, toggl_context):
        await CreateTimeEntry().execute(minimal_inputs(), toggl_context)
        assert toggl_context.fetch.call_args.kwargs["json"]["workspace_id"] == WORKSPACE_ID

    @pytest.mark.asyncio
    async def test_omitted_optional_fields_are_stripped(self, toggl_context):
        """None values are dropped so Toggl doesn't reject the payload."""
        await CreateTimeEntry().execute(minimal_inputs(), toggl_context)

        body = toggl_context.fetch.call_args.kwargs["json"]
        for absent in ("description", "stop", "duration", "project_id", "task_id", "tags", "tag_ids", "user_id"):
            assert absent not in body

    @pytest.mark.asyncio
    async def test_no_none_values_survive_in_payload(self, toggl_context):
        await CreateTimeEntry().execute(minimal_inputs(description=None, project_id=None), toggl_context)

        body = toggl_context.fetch.call_args.kwargs["json"]
        assert all(v is not None for v in body.values())

    @pytest.mark.asyncio
    async def test_billable_defaults_to_false(self, toggl_context):
        """billable is a real False, not stripped as if it were absent."""
        await CreateTimeEntry().execute(minimal_inputs(), toggl_context)

        body = toggl_context.fetch.call_args.kwargs["json"]
        assert body["billable"] is False

    @pytest.mark.asyncio
    async def test_billable_true_passed_through(self, toggl_context):
        await CreateTimeEntry().execute(minimal_inputs(billable=True), toggl_context)
        assert toggl_context.fetch.call_args.kwargs["json"]["billable"] is True

    @pytest.mark.asyncio
    async def test_full_payload_passed_through(self, toggl_context):
        inputs = minimal_inputs(
            description="Writing tests",
            stop="2026-01-15T10:30:00Z",
            duration=5400,
            project_id=555,
            task_id=666,
            tags=["dev", "testing"],
            tag_ids=[1, 2],
            user_id=777,
        )

        await CreateTimeEntry().execute(inputs, toggl_context)

        body = toggl_context.fetch.call_args.kwargs["json"]
        assert body["description"] == "Writing tests"
        assert body["stop"] == "2026-01-15T10:30:00Z"
        assert body["duration"] == 5400
        assert body["project_id"] == 555
        assert body["task_id"] == 666
        assert body["tags"] == ["dev", "testing"]
        assert body["tag_ids"] == [1, 2]
        assert body["user_id"] == 777

    @pytest.mark.asyncio
    async def test_running_entry_uses_negative_duration(self, toggl_context):
        """A still-running entry is signalled by duration == -1, not by omission."""
        await CreateTimeEntry().execute(minimal_inputs(duration=-1), toggl_context)

        assert toggl_context.fetch.call_args.kwargs["json"]["duration"] == -1

    @pytest.mark.asyncio
    async def test_missing_api_token_raises(self, toggl_context):
        toggl_context.auth = {"credentials": {}}

        with pytest.raises(Exception, match="Toggl API token is required"):
            await CreateTimeEntry().execute(minimal_inputs(), toggl_context)

        toggl_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_api_token_raises(self, toggl_context):
        toggl_context.auth = {"credentials": {"api_token": ""}}

        with pytest.raises(Exception, match="Toggl API token is required"):
            await CreateTimeEntry().execute(minimal_inputs(), toggl_context)

    @pytest.mark.asyncio
    async def test_fetch_error_propagates(self, toggl_context):
        toggl_context.fetch.side_effect = Exception("HTTP 403: Forbidden")

        with pytest.raises(Exception, match="HTTP 403"):
            await CreateTimeEntry().execute(minimal_inputs(), toggl_context)


# ---- Config ----


class TestTogglConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(toggl_integration._action_handlers.keys())

        assert defined == registered

    def test_create_time_entry_requires_workspace_and_start(self, config):
        schema = config["actions"]["create_time_entry"]["input_schema"]
        assert sorted(schema["required"]) == ["start", "workspace_id"]

    def test_billable_default_matches_handler(self, config):
        """config.json default and the handler's fallback must not drift apart."""
        props = config["actions"]["create_time_entry"]["input_schema"]["properties"]
        assert props["billable"]["default"] is False

    def test_custom_auth_declares_api_token(self, config):
        auth = config["auth"]
        assert auth["type"] == "custom"
        assert "api_token" in auth["fields"]["properties"]
        assert auth["fields"]["properties"]["api_token"]["format"] == "password"
