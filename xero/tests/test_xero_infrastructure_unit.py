"""Unit tests for the Xero rate limiter, file resolution, and connection layer.

Covers the shared infrastructure every other Xero action depends on:

- `XeroRateLimiter` -- 429 retry/backoff and the bail-out-early path
- `_resolve_file_bytes` -- base64 vs pre-signed-URL file injection
- `get_all_connections` / `get_available_connections` -- tenant discovery
- `XeroConnectedAccountHandler` -- account info shown in the UI
- `find_contact_by_name` -- the one action that surfaces rate-limit state

`asyncio.sleep` is patched throughout so backoff paths run instantly.

Fully mocked -- no network access.
"""

import base64
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from xero.xero import (  # noqa: E402
    FindContactByNameAction,
    GetAvailableConnectionsAction,
    XeroConnectedAccountHandler,
    XeroRateLimitExceededException,
    XeroRateLimiter,
    _resolve_file_bytes,
    get_all_connections,
    xero as xero_integration,
)

pytestmark = pytest.mark.unit

TENANT_ID = "b2c3d4e5-f6a7-8901-bcde-f23456789012"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

CONNECTION = {"tenantId": TENANT_ID, "tenantName": "Demo Company (NZ)", "tenantType": "ORGANISATION"}

SAMPLE_CONTACT = {
    "ContactID": "c1",
    "Name": "Acme Ltd",
    "ContactStatus": "ACTIVE",
    "EmailAddress": "ap@acme.com",
    "HasAttachments": False,
}


@pytest.fixture
def xr_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": "test_token"}}  # nosec B105
    return ctx


def rate_limit_error(retry_after=None):
    """Build an exception that looks like a Xero 429 to the limiter."""
    exc = Exception("HTTP 429: Too Many Requests")
    if retry_after is not None:
        exc.headers = {"Retry-After": str(retry_after)}
    return exc


# ---- XeroRateLimiter: delay extraction ----


class TestExtractRetryDelay:
    def test_reads_retry_after_header(self):
        limiter = XeroRateLimiter()

        assert limiter._extract_retry_delay(rate_limit_error(retry_after=17)) == 17

    def test_falls_back_to_default_when_header_absent(self):
        limiter = XeroRateLimiter(default_retry_delay=45)

        assert limiter._extract_retry_delay(Exception("429")) == 45

    def test_non_numeric_header_falls_back_to_default(self):
        """Xero occasionally sends a date instead of seconds; that must not crash."""
        limiter = XeroRateLimiter(default_retry_delay=30)
        exc = Exception("429")
        exc.headers = {"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}

        assert limiter._extract_retry_delay(exc) == 30

    def test_empty_header_falls_back_to_default(self):
        limiter = XeroRateLimiter(default_retry_delay=60)
        exc = Exception("429")
        exc.headers = {"Retry-After": ""}

        assert limiter._extract_retry_delay(exc) == 60


# ---- XeroRateLimiter: request behaviour ----


class TestRateLimiterMakeRequest:
    @pytest.mark.asyncio
    async def test_injects_tenant_header(self, xr_context):
        """Every Xero API call must carry xero-tenant-id or it 401s."""
        xr_context.fetch.return_value = {"Contacts": []}

        await XeroRateLimiter().make_request(xr_context, "https://api.xero.com/x", TENANT_ID)

        assert xr_context.fetch.call_args.kwargs["headers"]["xero-tenant-id"] == TENANT_ID

    @pytest.mark.asyncio
    async def test_preserves_caller_supplied_headers(self, xr_context):
        xr_context.fetch.return_value = {}

        await XeroRateLimiter().make_request(
            xr_context, "https://api.xero.com/x", TENANT_ID, headers={"Accept": "application/json"}
        )

        headers = xr_context.fetch.call_args.kwargs["headers"]
        assert headers["Accept"] == "application/json"
        assert headers["xero-tenant-id"] == TENANT_ID

    @pytest.mark.asyncio
    async def test_forwards_method_and_params(self, xr_context):
        xr_context.fetch.return_value = {}

        await XeroRateLimiter().make_request(
            xr_context, "https://api.xero.com/x", TENANT_ID, method="GET", params={"where": "x"}
        )

        assert xr_context.fetch.call_args.kwargs["method"] == "GET"
        assert xr_context.fetch.call_args.kwargs["params"] == {"where": "x"}

    @pytest.mark.asyncio
    async def test_returns_response_without_retrying_on_success(self, xr_context):
        xr_context.fetch.return_value = {"ok": True}

        result = await XeroRateLimiter().make_request(xr_context, "https://x", TENANT_ID)

        assert result == {"ok": True}
        assert xr_context.fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self, xr_context):
        xr_context.fetch.side_effect = [rate_limit_error(retry_after=5), {"ok": True}]

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock) as sleep:
            result = await XeroRateLimiter().make_request(xr_context, "https://x", TENANT_ID)

        assert result == {"ok": True}
        assert xr_context.fetch.await_count == 2
        sleep.assert_awaited_once_with(5)

    @pytest.mark.asyncio
    async def test_waits_the_header_delay_between_attempts(self, xr_context):
        xr_context.fetch.side_effect = [
            rate_limit_error(retry_after=3),
            rate_limit_error(retry_after=7),
            {"ok": True},
        ]

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await XeroRateLimiter().make_request(xr_context, "https://x", TENANT_ID)

        assert [c.args[0] for c in sleep.await_args_list] == [3, 7]

    @pytest.mark.asyncio
    async def test_exhausting_retries_raises_the_last_error(self, xr_context):
        """max_retries=2 means 3 total attempts, then the final 429 propagates."""
        xr_context.fetch.side_effect = rate_limit_error(retry_after=1)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(Exception, match="429"):
                await XeroRateLimiter(max_retries=2).make_request(xr_context, "https://x", TENANT_ID)

        assert xr_context.fetch.await_count == 3

    @pytest.mark.asyncio
    async def test_no_sleep_on_the_final_attempt(self, xr_context):
        """The loop breaks before sleeping once retries are spent, so the lambda
        doesn't pay a pointless delay before failing."""
        xr_context.fetch.side_effect = rate_limit_error(retry_after=1)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with pytest.raises(Exception):
                await XeroRateLimiter(max_retries=2).make_request(xr_context, "https://x", TENANT_ID)

        assert sleep.await_count == 2

    @pytest.mark.asyncio
    async def test_delay_over_max_wait_bails_out_immediately(self, xr_context):
        """Rather than blocking the lambda for minutes, the limiter raises a
        structured exception so the caller can report back and retry later."""
        xr_context.fetch.side_effect = rate_limit_error(retry_after=300)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with pytest.raises(XeroRateLimitExceededException) as exc:
                await XeroRateLimiter(max_wait_time=60).make_request(xr_context, "https://x", TENANT_ID)

        sleep.assert_not_awaited()
        assert exc.value.requested_delay == 300
        assert exc.value.max_wait_time == 60
        assert exc.value.tenant_id == TENANT_ID

    @pytest.mark.asyncio
    async def test_delay_exactly_at_max_wait_is_accepted(self, xr_context):
        """The comparison is `>`, so a delay equal to the cap still waits."""
        xr_context.fetch.side_effect = [rate_limit_error(retry_after=60), {"ok": True}]

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock) as sleep:
            await XeroRateLimiter(max_wait_time=60).make_request(xr_context, "https://x", TENANT_ID)

        sleep.assert_awaited_once_with(60)

    @pytest.mark.asyncio
    async def test_non_rate_limit_errors_fail_immediately(self, xr_context):
        """A 400 must not be retried -- it will never succeed."""
        xr_context.fetch.side_effect = Exception("HTTP 400: Bad Request")

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock) as sleep:
            with pytest.raises(Exception, match="400"):
                await XeroRateLimiter().make_request(xr_context, "https://x", TENANT_ID)

        assert xr_context.fetch.await_count == 1
        sleep.assert_not_awaited()

    @pytest.mark.parametrize(
        "message",
        ["HTTP 429", "Rate limit exceeded", "TOO MANY REQUESTS", "too many requests"],
    )
    @pytest.mark.asyncio
    async def test_rate_limit_detection_is_case_insensitive(self, xr_context, message):
        """Detection is a substring match on the lowercased message."""
        xr_context.fetch.side_effect = [Exception(message), {"ok": True}]

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            result = await XeroRateLimiter().make_request(xr_context, "https://x", TENANT_ID)

        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_unrelated_error_containing_429_is_treated_as_rate_limit(self, xr_context):
        """Substring matching means an invoice numbered 429 in an error message
        triggers a retry. Documented rather than assumed correct."""
        xr_context.fetch.side_effect = [Exception("Invoice INV-429 not found"), {"ok": True}]

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            result = await XeroRateLimiter().make_request(xr_context, "https://x", TENANT_ID)

        assert result == {"ok": True}
        assert xr_context.fetch.await_count == 2

    def test_default_configuration(self):
        limiter = XeroRateLimiter()

        assert limiter.default_retry_delay == 60
        assert limiter.max_retries == 3
        assert limiter.max_wait_time == 60


class TestRateLimitExceededException:
    def test_message_reports_all_three_values(self):
        exc = XeroRateLimitExceededException(120, 60, TENANT_ID)

        assert "120s" in str(exc)
        assert "60s" in str(exc)
        assert TENANT_ID in str(exc)

    def test_attributes_are_exposed_for_structured_handling(self):
        exc = XeroRateLimitExceededException(120, 60, TENANT_ID)

        assert exc.requested_delay == 120
        assert exc.max_wait_time == 60
        assert exc.tenant_id == TENANT_ID


# ---- _resolve_file_bytes ----


class TestResolveFileBytes:
    @pytest.mark.asyncio
    async def test_decodes_base64_content(self):
        payload = b"\x89PNG\r\n\x1a\n binary"

        result = await _resolve_file_bytes({"content": base64.b64encode(payload).decode()})

        assert result == payload

    @pytest.mark.asyncio
    async def test_content_takes_precedence_over_url(self):
        """When both are present the inline bytes win -- no network call."""
        payload = b"inline"

        with patch("xero.xero.aiohttp.ClientSession") as session:
            result = await _resolve_file_bytes(
                {"content": base64.b64encode(payload).decode(), "url": "https://s3/x"}
            )

        assert result == payload
        session.assert_not_called()

    @pytest.mark.asyncio
    async def test_invalid_base64_raises_a_clear_error(self):
        with pytest.raises(ValueError, match="not valid base64"):
            await _resolve_file_bytes({"content": "!!!not base64!!!"})

    @pytest.mark.asyncio
    async def test_downloads_from_presigned_url(self):
        response = MagicMock()
        response.status = 200
        response.read = AsyncMock(return_value=b"downloaded")

        resp_ctx = MagicMock()
        resp_ctx.__aenter__ = AsyncMock(return_value=response)
        resp_ctx.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.get = MagicMock(return_value=resp_ctx)
        sess_ctx = MagicMock()
        sess_ctx.__aenter__ = AsyncMock(return_value=session)
        sess_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("xero.xero.aiohttp.ClientSession", return_value=sess_ctx):
            result = await _resolve_file_bytes({"url": "https://s3/presigned"})

        assert result == b"downloaded"

    @pytest.mark.asyncio
    async def test_failed_download_raises_with_status(self):
        response = MagicMock()
        response.status = 403
        response.read = AsyncMock(return_value=b"")

        resp_ctx = MagicMock()
        resp_ctx.__aenter__ = AsyncMock(return_value=response)
        resp_ctx.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.get = MagicMock(return_value=resp_ctx)
        sess_ctx = MagicMock()
        sess_ctx.__aenter__ = AsyncMock(return_value=session)
        sess_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("xero.xero.aiohttp.ClientSession", return_value=sess_ctx):
            with pytest.raises(ValueError, match="HTTP 403"):
                await _resolve_file_bytes({"url": "https://s3/presigned"})

    @pytest.mark.asyncio
    async def test_file_id_without_content_names_the_platform_failure(self):
        """A fileId with no injected content means the platform didn't hydrate
        the file -- the message says so rather than blaming the input."""
        with pytest.raises(ValueError, match="platform failed to inject file content"):
            await _resolve_file_bytes({"fileId": "f-123", "ownerType": "message"})

    @pytest.mark.asyncio
    async def test_empty_file_object_raises(self):
        with pytest.raises(ValueError, match="missing 'content'"):
            await _resolve_file_bytes({})

    @pytest.mark.asyncio
    async def test_empty_content_string_falls_through_to_url(self):
        """`content: ""` is falsy, so resolution continues to the url branch."""
        with pytest.raises(ValueError, match="missing 'content'"):
            await _resolve_file_bytes({"content": ""})


# ---- get_all_connections ----


class TestGetAllConnections:
    @pytest.mark.asyncio
    async def test_returns_connections(self, xr_context):
        xr_context.fetch.return_value = [CONNECTION]

        assert await get_all_connections(xr_context) == [CONNECTION]

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, xr_context):
        xr_context.fetch.return_value = [CONNECTION]

        await get_all_connections(xr_context)

        call = xr_context.fetch.call_args
        assert call.args[0] == "https://api.xero.com/connections"
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["headers"] == {"Accept": "application/json"}

    @pytest.mark.asyncio
    async def test_no_tenant_header_is_sent(self, xr_context):
        """The connections endpoint is tenant-agnostic -- it's how you discover
        tenants in the first place."""
        xr_context.fetch.return_value = [CONNECTION]

        await get_all_connections(xr_context)

        assert "xero-tenant-id" not in xr_context.fetch.call_args.kwargs["headers"]

    @pytest.mark.parametrize("empty", [[], None, {}])
    @pytest.mark.asyncio
    async def test_empty_response_raises(self, xr_context, empty):
        xr_context.fetch.return_value = empty

        with pytest.raises(Exception, match="No Xero connections found"):
            await get_all_connections(xr_context)

    @pytest.mark.asyncio
    async def test_errors_are_wrapped_with_context(self, xr_context):
        xr_context.fetch.side_effect = Exception("HTTP 401: token expired")

        with pytest.raises(Exception, match="Failed to get connections"):
            await get_all_connections(xr_context)


# ---- get_available_connections ----


class TestGetAvailableConnections:
    @pytest.mark.asyncio
    async def test_maps_tenants_to_companies(self, xr_context):
        xr_context.fetch.return_value = [CONNECTION]

        result = await GetAvailableConnectionsAction().execute({}, xr_context)

        assert result.data["success"] is True
        assert result.data["companies"] == [
            {"tenant_id": TENANT_ID, "company_name": "Demo Company (NZ)"}
        ]

    @pytest.mark.asyncio
    async def test_multiple_tenants_all_returned(self, xr_context):
        second = {"tenantId": "t2", "tenantName": "Second Co"}
        xr_context.fetch.return_value = [CONNECTION, second]

        result = await GetAvailableConnectionsAction().execute({}, xr_context)

        assert len(result.data["companies"]) == 2

    @pytest.mark.asyncio
    async def test_connections_missing_id_or_name_are_skipped(self, xr_context):
        """A partial connection would produce an unusable entry, so it's dropped
        rather than returned with a None field."""
        xr_context.fetch.return_value = [
            CONNECTION,
            {"tenantId": "t2"},
            {"tenantName": "No ID Co"},
        ]

        result = await GetAvailableConnectionsAction().execute({}, xr_context)

        assert len(result.data["companies"]) == 1

    @pytest.mark.asyncio
    async def test_failure_is_reported_not_raised(self, xr_context):
        xr_context.fetch.side_effect = Exception("HTTP 401")

        result = await GetAvailableConnectionsAction().execute({}, xr_context)

        assert result.data["success"] is False
        assert result.data["companies"] == []
        assert "Failed to get connections" in result.data["message"]


# ---- Connected account handler ----


class TestConnectedAccountHandler:
    @pytest.mark.asyncio
    async def test_uses_first_tenant_name_as_username(self, xr_context):
        xr_context.fetch.return_value = [CONNECTION, {"tenantId": "t2", "tenantName": "Second"}]

        info = await XeroConnectedAccountHandler().get_account_info(xr_context)

        assert info.username == "Demo Company (NZ)"
        assert info.user_id == TENANT_ID

    @pytest.mark.parametrize("empty", [[], None, {}])
    @pytest.mark.asyncio
    async def test_empty_response_degrades_gracefully(self, xr_context, empty):
        """This runs during authorization, so it must never raise -- a failure
        here would break the connect flow rather than one action."""
        xr_context.fetch.return_value = empty

        info = await XeroConnectedAccountHandler().get_account_info(xr_context)

        assert info.username == "Unknown Organization"

    @pytest.mark.asyncio
    async def test_missing_tenant_name_falls_back(self, xr_context):
        xr_context.fetch.return_value = [{"tenantId": TENANT_ID}]

        info = await XeroConnectedAccountHandler().get_account_info(xr_context)

        assert info.username == "Unknown Organization"
        assert info.user_id == TENANT_ID


# ---- find_contact_by_name ----


class TestFindContactByName:
    @pytest.mark.asyncio
    async def test_returns_mapped_contacts(self, xr_context):
        xr_context.fetch.return_value = {"Contacts": [SAMPLE_CONTACT]}

        result = await FindContactByNameAction().execute(
            {"tenant_id": TENANT_ID, "contact_name": "Acme"}, xr_context
        )

        contact = result.data["contacts"][0]
        assert contact["contact_id"] == "c1"
        assert contact["name"] == "Acme Ltd"
        assert contact["email_address"] == "ap@acme.com"

    @pytest.mark.asyncio
    async def test_builds_contains_where_filter(self, xr_context):
        """Xero filtering uses a `where` expression, not a search param."""
        xr_context.fetch.return_value = {"Contacts": []}

        await FindContactByNameAction().execute(
            {"tenant_id": TENANT_ID, "contact_name": "Acme"}, xr_context
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params == {"where": 'Name.Contains("Acme")'}

    @pytest.mark.asyncio
    async def test_request_url_and_tenant_header(self, xr_context):
        xr_context.fetch.return_value = {"Contacts": []}

        await FindContactByNameAction().execute(
            {"tenant_id": TENANT_ID, "contact_name": "Acme"}, xr_context
        )

        call = xr_context.fetch.call_args
        assert call.args[0] == "https://api.xero.com/api.xro/2.0/Contacts"
        assert call.kwargs["headers"]["xero-tenant-id"] == TENANT_ID

    @pytest.mark.asyncio
    async def test_quote_in_name_is_not_escaped(self, xr_context):
        """The filter is built by string interpolation, so a double quote in the
        contact name produces a malformed where expression. Documented as a real
        injection-shaped gap rather than assumed safe."""
        xr_context.fetch.return_value = {"Contacts": []}

        await FindContactByNameAction().execute(
            {"tenant_id": TENANT_ID, "contact_name": 'Ac"me'}, xr_context
        )

        assert xr_context.fetch.call_args.kwargs["params"]["where"] == 'Name.Contains("Ac"me")'

    @pytest.mark.asyncio
    async def test_all_contact_fields_are_mapped(self, xr_context):
        """The handler flattens 25 Xero fields to snake_case; a missing one
        becomes None rather than being omitted."""
        xr_context.fetch.return_value = {"Contacts": [{"ContactID": "c1"}]}

        result = await FindContactByNameAction().execute(
            {"tenant_id": TENANT_ID, "contact_name": "x"}, xr_context
        )

        contact = result.data["contacts"][0]
        assert len(contact) == 25
        assert contact["name"] is None
        assert contact["contact_persons"] is None

    @pytest.mark.parametrize("response", [{}, {"Contacts": []}, None])
    @pytest.mark.asyncio
    async def test_no_matches_yields_empty_list(self, xr_context, response):
        xr_context.fetch.return_value = response

        result = await FindContactByNameAction().execute(
            {"tenant_id": TENANT_ID, "contact_name": "nobody"}, xr_context
        )

        assert result.data["contacts"] == []

    @pytest.mark.parametrize("missing", ["tenant_id", "contact_name"])
    @pytest.mark.asyncio
    async def test_missing_required_inputs_raise(self, xr_context, missing):
        """These raise rather than returning a failed ActionResult -- unlike most
        other handlers in this integration."""
        inputs = {"tenant_id": TENANT_ID, "contact_name": "Acme"}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await FindContactByNameAction().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_structured_payload(self, xr_context):
        """Rather than raising, this action reports the retry delay so the caller
        can decide when to try again."""
        xr_context.fetch.side_effect = rate_limit_error(retry_after=600)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            result = await FindContactByNameAction().execute(
                {"tenant_id": TENANT_ID, "contact_name": "Acme"}, xr_context
            )

        assert result.data["success"] is False
        assert result.data["error_type"] == "rate_limit_exceeded"
        assert result.data["retry_delay_seconds"] == 600
        assert result.data["tenant_id"] == TENANT_ID
        assert result.data["contacts"] == []

    @pytest.mark.asyncio
    async def test_other_errors_are_wrapped_and_raised(self, xr_context):
        xr_context.fetch.side_effect = Exception("HTTP 500")

        with pytest.raises(Exception, match="Failed to find contact by name"):
            await FindContactByNameAction().execute(
                {"tenant_id": TENANT_ID, "contact_name": "Acme"}, xr_context
            )


# ---- Config ----


class TestXeroConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(xero_integration._action_handlers.keys())

        assert defined == registered

    def test_uses_platform_oauth(self, config):
        assert config["auth"]["type"] == "platform"

    def test_find_contact_requires_tenant_and_name(self, config):
        required = config["actions"]["find_contact_by_name"]["input_schema"]["required"]
        assert sorted(required) == ["contact_name", "tenant_id"]

    def test_get_available_connections_requires_nothing(self, config):
        """Tenant discovery must be callable before any tenant is known."""
        assert not config["actions"]["get_available_connections"]["input_schema"].get("required")
