import pytest
from unittest.mock import AsyncMock, MagicMock
from autohive_integrations_sdk import FetchResponse, ResultType

from test2.test2 import test2

pytestmark = pytest.mark.unit

API_BASE = "https://api.openweathermap.org/data/2.5"


@pytest.fixture
def mock_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "Custom", "credentials": {"api_key": "test_key"}}  # nosec B105
    return ctx


SAMPLE_RESPONSE = {
    "name": "Wellington",
    "sys": {"country": "NZ"},
    "weather": [{"description": "light rain"}],
    "main": {"temp": 13.4, "feels_like": 12.1, "humidity": 82},
    "wind": {"speed": 7.2},
}


# ---- get_current_weather ----


class TestGetCurrentWeather:
    async def test_returns_current_conditions(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        result = await test2.execute_action(
            "get_current_weather", {"city": "Wellington,NZ"}, mock_context
        )

        assert result.type != ResultType.ACTION_ERROR
        data = result.result.data
        assert data["result"] is True
        assert data["city"] == "Wellington"
        assert data["country"] == "NZ"
        assert data["conditions"] == "light rain"
        assert data["temperature"] == 13.4
        assert data["feels_like"] == 12.1
        assert data["humidity"] == 82
        assert data["wind_speed"] == 7.2

    async def test_calls_weather_endpoint(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test2.execute_action(
            "get_current_weather", {"city": "Wellington,NZ"}, mock_context
        )

        call = mock_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/weather"
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["params"]["q"] == "Wellington,NZ"

    async def test_api_key_passed_as_param(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test2.execute_action(
            "get_current_weather", {"city": "Wellington"}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["params"]["appid"] == "test_key"  # nosec B105

    async def test_defaults_to_metric_units(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test2.execute_action(
            "get_current_weather", {"city": "Wellington"}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["params"]["units"] == "metric"

    async def test_units_passed_through(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test2.execute_action(
            "get_current_weather",
            {"city": "Austin,TX,US", "units": "imperial"},
            mock_context,
        )

        assert mock_context.fetch.call_args.kwargs["params"]["units"] == "imperial"

    async def test_handles_sparse_response(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data={"name": "Nowhere"}
        )

        result = await test2.execute_action(
            "get_current_weather", {"city": "Nowhere"}, mock_context
        )

        assert result.type != ResultType.ACTION_ERROR
        data = result.result.data
        assert data == {"city": "Nowhere", "result": True}

    async def test_fetch_failure_returns_error(self, mock_context):
        mock_context.fetch.side_effect = Exception("401 Unauthorized")

        result = await test2.execute_action(
            "get_current_weather", {"city": "Wellington"}, mock_context
        )

        assert result.type == ResultType.ACTION_ERROR
