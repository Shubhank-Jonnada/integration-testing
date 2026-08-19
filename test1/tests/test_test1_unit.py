import pytest
from unittest.mock import AsyncMock, MagicMock
from autohive_integrations_sdk import FetchResponse, ResultType

from test1.test1 import test1

pytestmark = pytest.mark.unit

API_BASE = "https://ipinfo.io"


@pytest.fixture
def mock_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "Custom", "credentials": {"api_token": "test_token"}}  # nosec B105
    return ctx


SAMPLE_RESPONSE = {
    "ip": "8.8.8.8",
    "hostname": "dns.google",
    "city": "Mountain View",
    "region": "California",
    "country": "US",
    "loc": "37.4056,-122.0775",
    "org": "AS15169 Google LLC",
    "postal": "94043",
    "timezone": "America/Los_Angeles",
}


# ---- lookup_ip ----


class TestLookupIp:
    async def test_returns_ip_details(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        result = await test1.execute_action(
            "lookup_ip", {"ip_address": "8.8.8.8"}, mock_context
        )

        assert result.type != ResultType.ACTION_ERROR
        data = result.result.data
        assert data["result"] is True
        assert data["ip"] == "8.8.8.8"
        assert data["city"] == "Mountain View"
        assert data["org"] == "AS15169 Google LLC"

    async def test_requests_the_given_address(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test1.execute_action("lookup_ip", {"ip_address": "8.8.8.8"}, mock_context)

        call = mock_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/8.8.8.8/json"
        assert call.kwargs["method"] == "GET"

    async def test_omitting_address_looks_up_own_ip(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test1.execute_action("lookup_ip", {}, mock_context)

        assert mock_context.fetch.call_args.args[0] == f"{API_BASE}/json"

    async def test_blank_address_looks_up_own_ip(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test1.execute_action("lookup_ip", {"ip_address": "   "}, mock_context)

        assert mock_context.fetch.call_args.args[0] == f"{API_BASE}/json"

    async def test_auth_header_propagated(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200, headers={}, data=SAMPLE_RESPONSE
        )

        await test1.execute_action("lookup_ip", {"ip_address": "1.1.1.1"}, mock_context)

        headers = mock_context.fetch.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer test_token"  # nosec B105

    async def test_omits_empty_fields(self, mock_context):
        mock_context.fetch.return_value = FetchResponse(
            status=200,
            headers={},
            data={"ip": "203.0.113.7", "city": "", "hostname": None},
        )

        result = await test1.execute_action(
            "lookup_ip", {"ip_address": "203.0.113.7"}, mock_context
        )

        data = result.result.data
        assert data == {"ip": "203.0.113.7", "result": True}

    async def test_fetch_failure_returns_error(self, mock_context):
        mock_context.fetch.side_effect = Exception("429 Too Many Requests")

        result = await test1.execute_action(
            "lookup_ip", {"ip_address": "8.8.8.8"}, mock_context
        )

        assert result.type == ResultType.ACTION_ERROR
