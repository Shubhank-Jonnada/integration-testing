"""Unit tests for the ElevenLabs voice, history, and subscription actions.

Covers the five actions that go through `context.fetch`. The two binary-audio
actions (`text_to_speech`, `download_history_audio`) bypass the SDK and use
aiohttp directly, so they're tested separately.

Fully mocked -- no network access, no credits consumed.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from elevenlabs.elevenlabs import (  # noqa: E402
    ELEVENLABS_API_BASE_URL,
    GetUserSubscriptionAction,
    GetVoiceAction,
    GetVoiceSettingsAction,
    ListHistoryAction,
    ListVoicesAction,
    get_auth_headers,
    elevenlabs as elevenlabs_integration,
)

pytestmark = pytest.mark.unit

API_KEY = "test_xi_api_key"  # nosec B105
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_VOICE = {
    "voice_id": VOICE_ID,
    "name": "Rachel",
    "category": "premade",
    "labels": {"accent": "american", "gender": "female"},
}

SAMPLE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0}

SAMPLE_HISTORY_ITEM = {
    "history_item_id": "hist_abc123",
    "voice_id": VOICE_ID,
    "text": "Hello world",
    "character_count_change_to": 11,
}

SAMPLE_SUBSCRIPTION = {
    "tier": "starter",
    "character_count": 1500,
    "character_limit": 30000,
    "can_extend_character_limit": False,
}


@pytest.fixture
def el_context():
    """Context carrying the ElevenLabs credential shape."""
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"api_key": API_KEY}}
    return ctx


# ---- Auth Headers ----


class TestGetAuthHeaders:
    def test_uses_xi_api_key_header(self, el_context):
        """ElevenLabs authenticates with a custom header, not a Bearer token."""
        headers = get_auth_headers(el_context)

        assert headers["xi-api-key"] == API_KEY
        assert "Authorization" not in headers

    def test_sets_json_content_type(self, el_context):
        assert get_auth_headers(el_context)["Content-Type"] == "application/json"

    def test_missing_credentials_yields_empty_key(self, el_context):
        """Absent credentials degrade to an empty key rather than raising."""
        el_context.auth = {}
        assert get_auth_headers(el_context)["xi-api-key"] == ""


# ---- list_voices ----


class TestListVoices:
    @pytest.mark.asyncio
    async def test_returns_voices(self, el_context):
        el_context.fetch.return_value = {"voices": [SAMPLE_VOICE]}

        result = await ListVoicesAction().execute({}, el_context)

        assert result["result"] is True
        assert result["voices"] == [SAMPLE_VOICE]

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, el_context):
        el_context.fetch.return_value = {"voices": []}

        await ListVoicesAction().execute({}, el_context)

        assert el_context.fetch.call_args.args[0] == f"{ELEVENLABS_API_BASE_URL}/voices"
        assert el_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_no_filters_sends_none_params(self, el_context):
        """An empty filter set is sent as None, not as an empty dict."""
        el_context.fetch.return_value = {"voices": []}

        await ListVoicesAction().execute({}, el_context)

        assert el_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_all_filters_forwarded(self, el_context):
        el_context.fetch.return_value = {"voices": []}

        await ListVoicesAction().execute(
            {"page_size": 20, "category": "premade", "use_cases": "narration", "search": "Rachel"},
            el_context,
        )

        params = el_context.fetch.call_args.kwargs["params"]
        assert params == {
            "page_size": 20,
            "category": "premade",
            "use_cases": "narration",
            "search": "Rachel",
        }

    @pytest.mark.asyncio
    async def test_falsy_filters_are_dropped(self, el_context):
        """The handler filters on truthiness, so 0 and "" are omitted."""
        el_context.fetch.return_value = {"voices": []}

        await ListVoicesAction().execute({"page_size": 0, "search": ""}, el_context)

        assert el_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_missing_voices_key_yields_empty_list(self, el_context):
        el_context.fetch.return_value = {}

        result = await ListVoicesAction().execute({}, el_context)

        assert result["voices"] == []
        assert result["result"] is True

    @pytest.mark.asyncio
    async def test_error_is_captured_not_raised(self, el_context):
        el_context.fetch.side_effect = Exception("HTTP 401: Unauthorized")

        result = await ListVoicesAction().execute({}, el_context)

        assert result["result"] is False
        assert "HTTP 401" in result["error"]
        assert result["voices"] == []


# ---- get_voice ----


class TestGetVoice:
    @pytest.mark.asyncio
    async def test_returns_voice(self, el_context):
        el_context.fetch.return_value = SAMPLE_VOICE

        result = await GetVoiceAction().execute({"voice_id": VOICE_ID}, el_context)

        assert result["result"] is True
        assert result["voice"]["name"] == "Rachel"

    @pytest.mark.asyncio
    async def test_request_url_includes_voice_id(self, el_context):
        el_context.fetch.return_value = SAMPLE_VOICE

        await GetVoiceAction().execute({"voice_id": VOICE_ID}, el_context)

        assert el_context.fetch.call_args.args[0] == f"{ELEVENLABS_API_BASE_URL}/voices/{VOICE_ID}"

    @pytest.mark.asyncio
    async def test_with_settings_lowercased(self, el_context):
        """The API expects the literal string 'true', not Python's 'True'."""
        el_context.fetch.return_value = SAMPLE_VOICE

        await GetVoiceAction().execute({"voice_id": VOICE_ID, "with_settings": True}, el_context)

        assert el_context.fetch.call_args.kwargs["params"] == {"with_settings": "true"}

    @pytest.mark.asyncio
    async def test_with_settings_false_is_omitted(self, el_context):
        el_context.fetch.return_value = SAMPLE_VOICE

        await GetVoiceAction().execute({"voice_id": VOICE_ID, "with_settings": False}, el_context)

        assert el_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_missing_voice_id_is_captured(self, el_context):
        """A KeyError from the missing input is caught by the handler."""
        result = await GetVoiceAction().execute({}, el_context)

        assert result["result"] is False
        assert result["voice"] == {}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, el_context):
        el_context.fetch.side_effect = Exception("HTTP 404: voice not found")

        result = await GetVoiceAction().execute({"voice_id": "nope"}, el_context)

        assert result["result"] is False
        assert "404" in result["error"]


# ---- get_voice_settings ----


class TestGetVoiceSettings:
    @pytest.mark.asyncio
    async def test_returns_settings(self, el_context):
        el_context.fetch.return_value = SAMPLE_SETTINGS

        result = await GetVoiceSettingsAction().execute({"voice_id": VOICE_ID}, el_context)

        assert result["result"] is True
        assert result["settings"]["stability"] == 0.5

    @pytest.mark.asyncio
    async def test_request_targets_settings_subresource(self, el_context):
        el_context.fetch.return_value = SAMPLE_SETTINGS

        await GetVoiceSettingsAction().execute({"voice_id": VOICE_ID}, el_context)

        assert el_context.fetch.call_args.args[0] == f"{ELEVENLABS_API_BASE_URL}/voices/{VOICE_ID}/settings"
        assert el_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_sends_auth_header(self, el_context):
        el_context.fetch.return_value = SAMPLE_SETTINGS

        await GetVoiceSettingsAction().execute({"voice_id": VOICE_ID}, el_context)

        assert el_context.fetch.call_args.kwargs["headers"]["xi-api-key"] == API_KEY

    @pytest.mark.asyncio
    async def test_error_is_captured(self, el_context):
        el_context.fetch.side_effect = Exception("boom")

        result = await GetVoiceSettingsAction().execute({"voice_id": VOICE_ID}, el_context)

        assert result["result"] is False
        assert result["settings"] == {}


# ---- list_history ----


class TestListHistory:
    @pytest.mark.asyncio
    async def test_returns_history(self, el_context):
        el_context.fetch.return_value = {"history": [SAMPLE_HISTORY_ITEM]}

        result = await ListHistoryAction().execute({}, el_context)

        assert result["result"] is True
        assert result["history"][0]["history_item_id"] == "hist_abc123"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, el_context):
        el_context.fetch.return_value = {"history": []}

        await ListHistoryAction().execute({}, el_context)

        assert el_context.fetch.call_args.args[0] == f"{ELEVENLABS_API_BASE_URL}/history"
        assert el_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_filters_forwarded(self, el_context):
        el_context.fetch.return_value = {"history": []}

        await ListHistoryAction().execute({"page_size": 50, "voice_id": VOICE_ID}, el_context)

        assert el_context.fetch.call_args.kwargs["params"] == {"page_size": 50, "voice_id": VOICE_ID}

    @pytest.mark.asyncio
    async def test_no_filters_sends_none_params(self, el_context):
        el_context.fetch.return_value = {"history": []}

        await ListHistoryAction().execute({}, el_context)

        assert el_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_missing_history_key_yields_empty_list(self, el_context):
        el_context.fetch.return_value = {}

        result = await ListHistoryAction().execute({}, el_context)

        assert result["history"] == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, el_context):
        el_context.fetch.side_effect = Exception("HTTP 500")

        result = await ListHistoryAction().execute({}, el_context)

        assert result["result"] is False
        assert result["history"] == []


# ---- get_user_subscription ----


class TestGetUserSubscription:
    @pytest.mark.asyncio
    async def test_returns_subscription(self, el_context):
        el_context.fetch.return_value = SAMPLE_SUBSCRIPTION

        result = await GetUserSubscriptionAction().execute({}, el_context)

        assert result["result"] is True
        assert result["subscription"]["character_limit"] == 30000

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, el_context):
        el_context.fetch.return_value = SAMPLE_SUBSCRIPTION

        await GetUserSubscriptionAction().execute({}, el_context)

        assert el_context.fetch.call_args.args[0] == f"{ELEVENLABS_API_BASE_URL}/user/subscription"
        assert el_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_takes_no_inputs(self, el_context):
        """Extra inputs are ignored rather than forwarded as params."""
        el_context.fetch.return_value = SAMPLE_SUBSCRIPTION

        await GetUserSubscriptionAction().execute({"ignored": "value"}, el_context)

        assert "params" not in el_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_error_is_captured(self, el_context):
        el_context.fetch.side_effect = Exception("HTTP 401")

        result = await GetUserSubscriptionAction().execute({}, el_context)

        assert result["result"] is False
        assert result["subscription"] == {}


# ---- Config ----


class TestElevenLabsConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(elevenlabs_integration._action_handlers.keys())

        assert defined == registered

    def test_base_url_is_v1(self):
        assert ELEVENLABS_API_BASE_URL == "https://api.elevenlabs.io/v1"

    @pytest.mark.parametrize(
        "action",
        ["get_voice", "get_voice_settings"],
    )
    def test_voice_scoped_actions_require_voice_id(self, config, action):
        assert "voice_id" in config["actions"][action]["input_schema"]["required"]
