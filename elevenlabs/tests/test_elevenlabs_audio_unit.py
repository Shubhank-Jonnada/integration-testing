"""Unit tests for the ElevenLabs binary-audio actions.

`text_to_speech` and `download_history_audio` bypass `context.fetch` and drive
aiohttp directly so they can read raw bytes. That means the request contract has
to be asserted against a stubbed `aiohttp.ClientSession` rather than the SDK's
fetch mock.

Fully mocked -- no network access, no credits consumed.
"""

import base64
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from elevenlabs.elevenlabs import (  # noqa: E402
    ELEVENLABS_API_BASE_URL,
    DownloadHistoryAudioAction,
    TextToSpeechAction,
)

pytestmark = pytest.mark.unit

API_KEY = "test_xi_api_key"  # nosec B105
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"
HISTORY_ITEM_ID = "hist_abc123"
AUDIO_BYTES = b"ID3\x04\x00\x00\x00fake-mp3-payload"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")


@pytest.fixture
def el_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"api_key": API_KEY}}
    return ctx


def stub_aiohttp(status=200, body=AUDIO_BYTES, text="", verb="post"):
    """Build an aiohttp.ClientSession stub and the response recorder.

    Returns (session_factory, calls) where `calls` captures the kwargs the
    handler passed to session.post / session.get.
    """
    calls = {}

    response = MagicMock(name="ClientResponse")
    response.status = status
    response.read = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value=text)

    response_ctx = MagicMock(name="response_ctx")
    response_ctx.__aenter__ = AsyncMock(return_value=response)
    response_ctx.__aexit__ = AsyncMock(return_value=False)

    def record(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return response_ctx

    session = MagicMock(name="ClientSession")
    setattr(session, verb, record)

    session_ctx = MagicMock(name="session_ctx")
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=session_ctx), calls


# ---- text_to_speech ----


class TestTextToSpeech:
    @pytest.mark.asyncio
    async def test_returns_base64_encoded_file(self, el_context):
        """Audio is returned base64-encoded in the Autohive file envelope."""
        factory, _ = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Hello"}, el_context)

        assert result["result"] is True
        assert base64.b64decode(result["file"]["content"]) == AUDIO_BYTES

    @pytest.mark.asyncio
    async def test_file_envelope_metadata(self, el_context):
        factory, _ = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Hello"}, el_context)

        assert result["file"]["name"] == "generated_audio.mp3"
        assert result["file"]["contentType"] == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_request_url_includes_voice_id(self, el_context):
        factory, calls = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Hello"}, el_context)

        assert calls["url"] == f"{ELEVENLABS_API_BASE_URL}/text-to-speech/{VOICE_ID}"

    @pytest.mark.asyncio
    async def test_output_format_appended_as_query_string(self, el_context):
        """output_format goes on the URL, not in the JSON body."""
        factory, calls = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await TextToSpeechAction().execute(
                {"voice_id": VOICE_ID, "text": "Hello", "output_format": "mp3_44100_128"},
                el_context,
            )

        assert calls["url"].endswith("?output_format=mp3_44100_128")
        assert "output_format" not in calls["json"]

    @pytest.mark.asyncio
    async def test_no_output_format_leaves_url_clean(self, el_context):
        factory, calls = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Hello"}, el_context)

        assert "?" not in calls["url"]

    @pytest.mark.asyncio
    async def test_body_carries_text(self, el_context):
        factory, calls = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Read this"}, el_context)

        assert calls["json"] == {"text": "Read this"}

    @pytest.mark.asyncio
    async def test_optional_body_fields_forwarded(self, el_context):
        factory, calls = stub_aiohttp()
        settings = {"stability": 0.4, "similarity_boost": 0.8}

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await TextToSpeechAction().execute(
                {
                    "voice_id": VOICE_ID,
                    "text": "Hello",
                    "model_id": "eleven_turbo_v2",
                    "voice_settings": settings,
                },
                el_context,
            )

        assert calls["json"]["model_id"] == "eleven_turbo_v2"
        assert calls["json"]["voice_settings"] == settings

    @pytest.mark.asyncio
    async def test_empty_optional_fields_omitted(self, el_context):
        factory, calls = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await TextToSpeechAction().execute(
                {"voice_id": VOICE_ID, "text": "Hello", "model_id": "", "voice_settings": {}},
                el_context,
            )

        assert calls["json"] == {"text": "Hello"}

    @pytest.mark.asyncio
    async def test_sends_api_key_and_json_content_type(self, el_context):
        factory, calls = stub_aiohttp()

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Hello"}, el_context)

        assert calls["headers"] == {"xi-api-key": API_KEY, "Content-Type": "application/json"}

    @pytest.mark.asyncio
    async def test_non_200_returns_status_and_body(self, el_context):
        factory, _ = stub_aiohttp(status=422, text="voice_not_found")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await TextToSpeechAction().execute({"voice_id": "bad", "text": "Hello"}, el_context)

        assert result["result"] is False
        assert result["error"] == "HTTP 422: voice_not_found"
        assert result["audio"] == {}

    @pytest.mark.asyncio
    async def test_error_key_is_audio_not_file_on_failure(self, el_context):
        """Failures return an `audio` key while successes return `file` -- assert
        the asymmetry so downstream consumers aren't surprised by it."""
        factory, _ = stub_aiohttp(status=500, text="server error")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Hi"}, el_context)

        assert "file" not in result
        assert "audio" in result

    @pytest.mark.asyncio
    async def test_missing_text_is_captured(self, el_context):
        result = await TextToSpeechAction().execute({"voice_id": VOICE_ID}, el_context)

        assert result["result"] is False
        assert result["audio"] == {}

    @pytest.mark.asyncio
    async def test_session_exception_is_captured(self, el_context):
        factory = MagicMock(side_effect=Exception("connection reset"))

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await TextToSpeechAction().execute({"voice_id": VOICE_ID, "text": "Hi"}, el_context)

        assert result["result"] is False
        assert "connection reset" in result["error"]


# ---- download_history_audio ----


class TestDownloadHistoryAudio:
    @pytest.mark.asyncio
    async def test_returns_base64_encoded_file(self, el_context):
        factory, _ = stub_aiohttp(verb="get")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await DownloadHistoryAudioAction().execute(
                {"history_item_id": HISTORY_ITEM_ID}, el_context
            )

        assert result["result"] is True
        assert base64.b64decode(result["file"]["content"]) == AUDIO_BYTES

    @pytest.mark.asyncio
    async def test_file_envelope_metadata(self, el_context):
        """Downloads are named differently from generated audio."""
        factory, _ = stub_aiohttp(verb="get")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await DownloadHistoryAudioAction().execute(
                {"history_item_id": HISTORY_ITEM_ID}, el_context
            )

        assert result["file"]["name"] == "downloaded_audio.mp3"
        assert result["file"]["contentType"] == "audio/mpeg"

    @pytest.mark.asyncio
    async def test_request_url_targets_history_audio(self, el_context):
        factory, calls = stub_aiohttp(verb="get")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await DownloadHistoryAudioAction().execute({"history_item_id": HISTORY_ITEM_ID}, el_context)

        assert calls["url"] == f"{ELEVENLABS_API_BASE_URL}/history/{HISTORY_ITEM_ID}/audio"

    @pytest.mark.asyncio
    async def test_sends_api_key_without_content_type(self, el_context):
        """A GET for binary content sends no JSON content type."""
        factory, calls = stub_aiohttp(verb="get")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            await DownloadHistoryAudioAction().execute({"history_item_id": HISTORY_ITEM_ID}, el_context)

        assert calls["headers"] == {"xi-api-key": API_KEY}

    @pytest.mark.asyncio
    async def test_non_200_returns_status_and_body(self, el_context):
        factory, _ = stub_aiohttp(status=404, text="history item not found", verb="get")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await DownloadHistoryAudioAction().execute({"history_item_id": "nope"}, el_context)

        assert result["result"] is False
        assert result["error"] == "HTTP 404: history item not found"

    @pytest.mark.asyncio
    async def test_empty_payload_still_succeeds(self, el_context):
        """A zero-byte 200 is reported as success with empty content."""
        factory, _ = stub_aiohttp(body=b"", verb="get")

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await DownloadHistoryAudioAction().execute(
                {"history_item_id": HISTORY_ITEM_ID}, el_context
            )

        assert result["result"] is True
        assert result["file"]["content"] == ""

    @pytest.mark.asyncio
    async def test_missing_history_item_id_is_captured(self, el_context):
        result = await DownloadHistoryAudioAction().execute({}, el_context)

        assert result["result"] is False
        assert result["audio"] == {}

    @pytest.mark.asyncio
    async def test_session_exception_is_captured(self, el_context):
        factory = MagicMock(side_effect=Exception("dns failure"))

        with patch("elevenlabs.elevenlabs.aiohttp.ClientSession", factory):
            result = await DownloadHistoryAudioAction().execute(
                {"history_item_id": HISTORY_ITEM_ID}, el_context
            )

        assert result["result"] is False
        assert "dns failure" in result["error"]


# ---- Config ----


class TestAudioActionConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_text_to_speech_requires_voice_and_text(self, config):
        required = config["actions"]["text_to_speech"]["input_schema"]["required"]
        assert sorted(required) == ["text", "voice_id"]

    def test_download_history_audio_requires_item_id(self, config):
        required = config["actions"]["download_history_audio"]["input_schema"]["required"]
        assert required == ["history_item_id"]

    @pytest.mark.parametrize("action", ["text_to_speech", "download_history_audio"])
    def test_audio_actions_declare_file_output(self, config, action):
        """Both return the Autohive file envelope, so the schema must expose it."""
        props = config["actions"][action]["output_schema"]["properties"]
        assert "file" in props
