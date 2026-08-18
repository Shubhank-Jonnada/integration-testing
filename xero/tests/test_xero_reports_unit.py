"""Unit tests for the Xero report actions.

Covers `get_aged_payables`, `get_aged_receivables`, `get_balance_sheet`,
`get_profit_and_loss`, `get_trial_balance`, and `get_accounts`.

These six share a structure: validate inputs, build a `params` dict of optional
Xero query arguments, call through the rate limiter, and return the raw report
payload. The tests focus on the per-action parameter mapping (snake_case inputs
to Xero's camelCase query args) and the shared rate-limit fallback.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from xero.xero import (  # noqa: E402
    GetAccountsAction,
    GetAgedPayablesAction,
    GetAgedReceivablesAction,
    GetBalanceSheetAction,
    GetProfitAndLossAction,
    GetTrialBalanceAction,
)

pytestmark = pytest.mark.unit

TENANT_ID = "b2c3d4e5-f6a7-8901-bcde-f23456789012"
CONTACT_ID = "c1d2e3f4-a5b6-7890-cdef-123456789abc"
API_BASE = "https://api.xero.com/api.xro/2.0"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

REPORT_PAYLOAD = {
    "Reports": [
        {
            "ReportID": "AgedPayablesByContact",
            "ReportName": "Aged Payables By Contact",
            "ReportDate": "15 January 2026",
            "Rows": [],
        }
    ]
}

ACCOUNTS_PAYLOAD = {
    "Accounts": [
        {"AccountID": "a1", "Code": "200", "Name": "Sales", "Type": "REVENUE", "Status": "ACTIVE"}
    ]
}


@pytest.fixture
def xr_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": "test_token"}}  # nosec B105
    return ctx


def rate_limit_error(retry_after):
    exc = Exception("HTTP 429: Too Many Requests")
    exc.headers = {"Retry-After": str(retry_after)}
    return exc


# Each entry: (action class, extra required inputs, expected URL suffix, error fragment)
CONTACT_REPORTS = [
    (GetAgedPayablesAction, "Reports/AgedPayablesByContact", "aged payables"),
    (GetAgedReceivablesAction, "Reports/AgedReceivablesByContact", "aged receivables"),
]

TENANT_ONLY_REPORTS = [
    (GetBalanceSheetAction, "Reports/BalanceSheet", "balance sheet"),
    (GetProfitAndLossAction, "Reports/ProfitAndLoss", "profit and loss"),
    (GetTrialBalanceAction, "Reports/TrialBalance", "trial balance"),
    (GetAccountsAction, "Accounts", "accounts"),
]

ALL_REPORTS = [(a, u, e) for a, u, e in CONTACT_REPORTS] + TENANT_ONLY_REPORTS


def inputs_for(action_cls):
    """Minimum valid inputs for a given report action."""
    base = {"tenant_id": TENANT_ID}
    if action_cls in (GetAgedPayablesAction, GetAgedReceivablesAction):
        base["contact_id"] = CONTACT_ID
    return base


# ---- Shared contract across all six reports ----


class TestSharedReportContract:
    @pytest.mark.parametrize("action_cls, url_suffix, _err", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_request_url_and_method(self, xr_context, action_cls, url_suffix, _err):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await action_cls().execute(inputs_for(action_cls), xr_context)

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/{url_suffix}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.parametrize("action_cls, _url, _err", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_tenant_header_is_injected(self, xr_context, action_cls, _url, _err):
        """Reports go through the rate limiter, which adds xero-tenant-id."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await action_cls().execute(inputs_for(action_cls), xr_context)

        assert xr_context.fetch.call_args.kwargs["headers"]["xero-tenant-id"] == TENANT_ID

    @pytest.mark.parametrize("action_cls, _url, _err", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_accept_json_header(self, xr_context, action_cls, _url, _err):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await action_cls().execute(inputs_for(action_cls), xr_context)

        assert xr_context.fetch.call_args.kwargs["headers"]["Accept"] == "application/json"

    @pytest.mark.parametrize("action_cls, _url, _err", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_raw_payload_returned_unwrapped(self, xr_context, action_cls, _url, _err):
        """Reports are passed through verbatim -- no reshaping, so the caller sees
        Xero's own Reports/Rows structure."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        result = await action_cls().execute(inputs_for(action_cls), xr_context)

        assert result.data == REPORT_PAYLOAD

    @pytest.mark.parametrize("action_cls, _url, _err", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, xr_context, action_cls, _url, _err):
        inputs = inputs_for(action_cls)
        del inputs["tenant_id"]

        with pytest.raises(ValueError, match="tenant_id is required"):
            await action_cls().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.parametrize("action_cls, _url, err_fragment", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_empty_response_raises_with_action_context(
        self, xr_context, action_cls, _url, err_fragment
    ):
        """An empty body is treated as a failure, and the wrapped message names
        which report failed."""
        xr_context.fetch.return_value = None

        with pytest.raises(Exception, match=err_fragment):
            await action_cls().execute(inputs_for(action_cls), xr_context)

    @pytest.mark.parametrize("action_cls, _url, err_fragment", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_errors_are_wrapped_with_action_context(
        self, xr_context, action_cls, _url, err_fragment
    ):
        xr_context.fetch.side_effect = Exception("HTTP 500")

        with pytest.raises(Exception, match=err_fragment):
            await action_cls().execute(inputs_for(action_cls), xr_context)

    @pytest.mark.parametrize("action_cls, _url, _err", ALL_REPORTS)
    @pytest.mark.asyncio
    async def test_rate_limit_returns_structured_payload(self, xr_context, action_cls, _url, _err):
        """Every report degrades to a structured rate-limit report rather than
        raising, so the caller can surface a retry delay."""
        xr_context.fetch.side_effect = rate_limit_error(600)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            result = await action_cls().execute(inputs_for(action_cls), xr_context)

        assert result.data["success"] is False
        assert result.data["error_type"] == "rate_limit_exceeded"
        assert result.data["retry_delay_seconds"] == 600
        assert result.data["tenant_id"] == TENANT_ID


# ---- Contact-scoped reports ----


class TestContactScopedReports:
    @pytest.mark.parametrize("action_cls, _url, _err", CONTACT_REPORTS)
    @pytest.mark.asyncio
    async def test_contact_id_sent_as_camel_case_param(self, xr_context, action_cls, _url, _err):
        """The input is `contact_id` but Xero expects `contactId`."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await action_cls().execute(inputs_for(action_cls), xr_context)

        assert xr_context.fetch.call_args.kwargs["params"]["contactId"] == CONTACT_ID

    @pytest.mark.parametrize("action_cls, _url, _err", CONTACT_REPORTS)
    @pytest.mark.asyncio
    async def test_missing_contact_id_raises(self, xr_context, action_cls, _url, _err):
        with pytest.raises(ValueError, match="contact_id is required"):
            await action_cls().execute({"tenant_id": TENANT_ID}, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.parametrize("action_cls, _url, _err", CONTACT_REPORTS)
    @pytest.mark.asyncio
    async def test_date_is_optional(self, xr_context, action_cls, _url, _err):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await action_cls().execute(inputs_for(action_cls), xr_context)
        assert "date" not in xr_context.fetch.call_args.kwargs["params"]

        await action_cls().execute({**inputs_for(action_cls), "date": "2026-01-31"}, xr_context)
        assert xr_context.fetch.call_args.kwargs["params"]["date"] == "2026-01-31"

    def test_payables_and_receivables_use_distinct_endpoints(self):
        """The two reports are otherwise identical, so a copy-paste slip between
        them would silently return the wrong side of the ledger."""
        assert CONTACT_REPORTS[0][1] != CONTACT_REPORTS[1][1]


# ---- get_balance_sheet ----


class TestGetBalanceSheet:
    @pytest.mark.asyncio
    async def test_no_optional_params_sends_empty_dict(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetBalanceSheetAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert xr_context.fetch.call_args.kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_date_forwarded(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetBalanceSheetAction().execute(
            {"tenant_id": TENANT_ID, "date": "2026-01-31"}, xr_context
        )

        assert xr_context.fetch.call_args.kwargs["params"]["date"] == "2026-01-31"

    @pytest.mark.asyncio
    async def test_periods_is_stringified(self, xr_context):
        """Xero query args must be strings; an int would be rejected."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetBalanceSheetAction().execute({"tenant_id": TENANT_ID, "periods": 3}, xr_context)

        assert xr_context.fetch.call_args.kwargs["params"]["periods"] == "3"

    @pytest.mark.asyncio
    async def test_periods_zero_is_dropped(self, xr_context):
        """periods is truthiness-gated, so 0 never reaches the API."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetBalanceSheetAction().execute({"tenant_id": TENANT_ID, "periods": 0}, xr_context)

        assert "periods" not in xr_context.fetch.call_args.kwargs["params"]


# ---- get_profit_and_loss ----


class TestGetProfitAndLoss:
    @pytest.mark.asyncio
    async def test_date_range_uses_camel_case_params(self, xr_context):
        """`from_date`/`to_date` inputs map to Xero's `fromDate`/`toDate`."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetProfitAndLossAction().execute(
            {"tenant_id": TENANT_ID, "from_date": "2026-01-01", "to_date": "2026-01-31"},
            xr_context,
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params["fromDate"] == "2026-01-01"
        assert params["toDate"] == "2026-01-31"
        assert "from_date" not in params
        assert "to_date" not in params

    @pytest.mark.asyncio
    async def test_timeframe_forwarded(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetProfitAndLossAction().execute(
            {"tenant_id": TENANT_ID, "timeframe": "MONTH"}, xr_context
        )

        assert xr_context.fetch.call_args.kwargs["params"]["timeframe"] == "MONTH"

    @pytest.mark.asyncio
    async def test_periods_is_stringified(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetProfitAndLossAction().execute({"tenant_id": TENANT_ID, "periods": 12}, xr_context)

        assert xr_context.fetch.call_args.kwargs["params"]["periods"] == "12"

    @pytest.mark.asyncio
    async def test_all_five_optional_params_together(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetProfitAndLossAction().execute(
            {
                "tenant_id": TENANT_ID,
                "date": "2026-01-31",
                "from_date": "2026-01-01",
                "to_date": "2026-01-31",
                "timeframe": "QUARTER",
                "periods": 4,
            },
            xr_context,
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert set(params) == {"date", "fromDate", "toDate", "timeframe", "periods"}

    @pytest.mark.asyncio
    async def test_no_optional_params_sends_empty_dict(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetProfitAndLossAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert xr_context.fetch.call_args.kwargs["params"] == {}


# ---- get_trial_balance ----


class TestGetTrialBalance:
    @pytest.mark.asyncio
    async def test_date_forwarded(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetTrialBalanceAction().execute(
            {"tenant_id": TENANT_ID, "date": "2026-01-31"}, xr_context
        )

        assert xr_context.fetch.call_args.kwargs["params"]["date"] == "2026-01-31"

    @pytest.mark.parametrize("value, expected", [(True, "true"), (False, "false")])
    @pytest.mark.asyncio
    async def test_payments_only_is_lowercased_string(self, xr_context, value, expected):
        """Xero expects the literal 'true'/'false', not Python's 'True'/'False'."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetTrialBalanceAction().execute(
            {"tenant_id": TENANT_ID, "payments_only": value}, xr_context
        )

        assert xr_context.fetch.call_args.kwargs["params"]["paymentsOnly"] == expected

    @pytest.mark.asyncio
    async def test_payments_only_false_is_still_sent(self, xr_context):
        """Unlike the other optionals, this one is gated on `is not None`, so an
        explicit False reaches the API rather than being dropped."""
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetTrialBalanceAction().execute(
            {"tenant_id": TENANT_ID, "payments_only": False}, xr_context
        )

        assert "paymentsOnly" in xr_context.fetch.call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_payments_only_omitted_when_absent(self, xr_context):
        xr_context.fetch.return_value = REPORT_PAYLOAD

        await GetTrialBalanceAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert "paymentsOnly" not in xr_context.fetch.call_args.kwargs["params"]


# ---- get_accounts ----


class TestGetAccounts:
    @pytest.mark.asyncio
    async def test_returns_accounts_payload(self, xr_context):
        xr_context.fetch.return_value = ACCOUNTS_PAYLOAD

        result = await GetAccountsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert result.data["Accounts"][0]["Code"] == "200"

    @pytest.mark.asyncio
    async def test_targets_accounts_not_reports(self, xr_context):
        """get_accounts hits the Accounts resource, not a Reports endpoint,
        despite being grouped with the reporting actions."""
        xr_context.fetch.return_value = ACCOUNTS_PAYLOAD

        await GetAccountsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        url = xr_context.fetch.call_args.args[0]
        assert url == f"{API_BASE}/Accounts"
        assert "Reports" not in url

    @pytest.mark.asyncio
    async def test_where_and_order_forwarded(self, xr_context):
        xr_context.fetch.return_value = ACCOUNTS_PAYLOAD

        await GetAccountsAction().execute(
            {"tenant_id": TENANT_ID, "where": 'Type=="REVENUE"', "order": "Code ASC"}, xr_context
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params["where"] == 'Type=="REVENUE"'
        assert params["order"] == "Code ASC"

    @pytest.mark.asyncio
    async def test_where_is_passed_through_verbatim(self, xr_context):
        """The caller supplies a raw Xero filter expression; the handler does no
        validation or escaping of it."""
        xr_context.fetch.return_value = ACCOUNTS_PAYLOAD

        await GetAccountsAction().execute(
            {"tenant_id": TENANT_ID, "where": 'Name.StartsWith("Bank")'}, xr_context
        )

        assert xr_context.fetch.call_args.kwargs["params"]["where"] == 'Name.StartsWith("Bank")'

    @pytest.mark.asyncio
    async def test_no_filters_sends_empty_params(self, xr_context):
        xr_context.fetch.return_value = ACCOUNTS_PAYLOAD

        await GetAccountsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert xr_context.fetch.call_args.kwargs["params"] == {}


# ---- Config ----


class TestXeroReportConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        [
            "get_aged_payables",
            "get_aged_receivables",
            "get_balance_sheet",
            "get_profit_and_loss",
            "get_trial_balance",
            "get_accounts",
        ],
    )
    def test_all_reports_require_tenant_id(self, config, action):
        assert "tenant_id" in config["actions"][action]["input_schema"]["required"]

    @pytest.mark.parametrize("action", ["get_aged_payables", "get_aged_receivables"])
    def test_aged_reports_require_contact_id(self, config, action):
        assert "contact_id" in config["actions"][action]["input_schema"]["required"]

    def test_profit_and_loss_exposes_the_full_date_range(self, config):
        props = config["actions"]["get_profit_and_loss"]["input_schema"]["properties"]

        assert "from_date" in props
        assert "to_date" in props
        assert "timeframe" in props
        assert "periods" in props

    def test_trial_balance_exposes_payments_only(self, config):
        props = config["actions"]["get_trial_balance"]["input_schema"]["properties"]
        assert props["payments_only"]["type"] == "boolean"
