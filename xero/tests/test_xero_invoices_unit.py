"""Unit tests for the Xero invoice, bill, payment, and bank transaction actions.

Covers the four read actions (`get_payments`, `get_invoices`, `get_invoice_pdf`,
`get_bank_transactions`) and the four write actions (`create_sales_invoice`,
`create_purchase_bill`, `update_sales_invoice`, `update_purchase_bill`).

The ACCREC/ACCPAY distinction is the load-bearing detail here: sales invoices and
purchase bills post to the same endpoint and differ only by a `Type` field, so
getting it wrong books money on the wrong side of the ledger.

Fully mocked -- no network access.
"""

import base64
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from xero.xero import (  # noqa: E402
    CreatePurchaseBillAction,
    CreateSalesInvoiceAction,
    GetBankTransactionsAction,
    GetInvoicePdfAction,
    GetInvoicesAction,
    GetPaymentsAction,
    UpdatePurchaseBillAction,
    UpdateSalesInvoiceAction,
)

pytestmark = pytest.mark.unit

TENANT_ID = "b2c3d4e5-f6a7-8901-bcde-f23456789012"
INVOICE_ID = "d4e5f6a7-b8c9-0123-def0-3456789abcde"
API_BASE = "https://api.xero.com/api.xro/2.0"
ACCESS_TOKEN = "test_access_token"  # nosec B105
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

CONTACT = {"ContactID": "c1"}
LINE_ITEMS = [{"Description": "Consulting", "Quantity": 2, "UnitAmount": 250.0, "AccountCode": "200"}]

INVOICE_PAYLOAD = {"Invoices": [{"InvoiceID": INVOICE_ID, "Type": "ACCREC", "Status": "DRAFT"}]}
PAYMENTS_PAYLOAD = {"Payments": [{"PaymentID": "p1", "Amount": 500.0}]}
PDF_BYTES = b"%PDF-1.4 fake pdf payload"


@pytest.fixture
def xr_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": ACCESS_TOKEN}}
    return ctx


def rate_limit_error(retry_after):
    exc = Exception("HTTP 429: Too Many Requests")
    exc.headers = {"Retry-After": str(retry_after)}
    return exc


def stub_pdf_session(status=200, body=PDF_BYTES, content_type="application/pdf", text=""):
    """Stub aiohttp for the PDF download path; returns (factory, calls)."""
    calls = {}

    response = MagicMock(name="ClientResponse")
    response.status = status
    response.read = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value=text)
    response.headers = {"content-type": content_type} if content_type else {}

    resp_ctx = MagicMock()
    resp_ctx.__aenter__ = AsyncMock(return_value=response)
    resp_ctx.__aexit__ = AsyncMock(return_value=False)

    def record(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return resp_ctx

    session = MagicMock()
    session.get = record
    sess_ctx = MagicMock()
    sess_ctx.__aenter__ = AsyncMock(return_value=session)
    sess_ctx.__aexit__ = AsyncMock(return_value=False)

    return MagicMock(return_value=sess_ctx), calls


# ---- get_payments ----


class TestGetPayments:
    @pytest.mark.asyncio
    async def test_returns_payments(self, xr_context):
        xr_context.fetch.return_value = PAYMENTS_PAYLOAD

        result = await GetPaymentsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert result.data["Payments"][0]["PaymentID"] == "p1"

    @pytest.mark.asyncio
    async def test_request_url_method_and_tenant(self, xr_context):
        xr_context.fetch.return_value = PAYMENTS_PAYLOAD

        await GetPaymentsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/Payments"
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["headers"]["xero-tenant-id"] == TENANT_ID

    @pytest.mark.asyncio
    async def test_where_and_order_forwarded(self, xr_context):
        xr_context.fetch.return_value = PAYMENTS_PAYLOAD

        await GetPaymentsAction().execute(
            {"tenant_id": TENANT_ID, "where": "Date>=DateTime(2026,01,01)", "order": "Date DESC"},
            xr_context,
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params["where"] == "Date>=DateTime(2026,01,01)"
        assert params["order"] == "Date DESC"

    @pytest.mark.asyncio
    async def test_pagination_is_stringified(self, xr_context):
        """Xero query args must be strings; ints are rejected."""
        xr_context.fetch.return_value = PAYMENTS_PAYLOAD

        await GetPaymentsAction().execute(
            {"tenant_id": TENANT_ID, "page": 2, "pageSize": 50}, xr_context
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params["page"] == "2"
        assert params["pageSize"] == "50"

    @pytest.mark.asyncio
    async def test_page_size_uses_camel_case_input_name(self, xr_context):
        """Unlike most inputs here, `pageSize` is already camelCase on the input
        side -- an inconsistency with `from_date`/`to_date` elsewhere."""
        xr_context.fetch.return_value = PAYMENTS_PAYLOAD

        await GetPaymentsAction().execute(
            {"tenant_id": TENANT_ID, "page_size": 50}, xr_context
        )

        assert "pageSize" not in xr_context.fetch.call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, xr_context):
        with pytest.raises(ValueError, match="tenant_id is required"):
            await GetPaymentsAction().execute({}, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_wrapped(self, xr_context):
        xr_context.fetch.side_effect = Exception("HTTP 500")

        with pytest.raises(Exception, match="Failed to fetch payments"):
            await GetPaymentsAction().execute({"tenant_id": TENANT_ID}, xr_context)

    @pytest.mark.asyncio
    async def test_rate_limit_returns_structured_payload(self, xr_context):
        xr_context.fetch.side_effect = rate_limit_error(600)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            result = await GetPaymentsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert result.data["error_type"] == "rate_limit_exceeded"


# ---- get_invoices ----


class TestGetInvoices:
    @pytest.mark.asyncio
    async def test_list_mode_targets_collection(self, xr_context):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await GetInvoicesAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert xr_context.fetch.call_args.args[0] == f"{API_BASE}/Invoices"

    @pytest.mark.asyncio
    async def test_single_mode_targets_invoice_id(self, xr_context):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await GetInvoicesAction().execute(
            {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
        )

        assert xr_context.fetch.call_args.args[0] == f"{API_BASE}/Invoices/{INVOICE_ID}"

    @pytest.mark.asyncio
    async def test_filters_are_ignored_in_single_mode(self, xr_context):
        """When invoice_id is supplied the filter block is skipped entirely, so a
        `where` passed alongside it is silently dropped."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await GetInvoicesAction().execute(
            {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID, "where": 'Status=="DRAFT"'},
            xr_context,
        )

        assert xr_context.fetch.call_args.kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_all_list_filters_forwarded(self, xr_context):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await GetInvoicesAction().execute(
            {
                "tenant_id": TENANT_ID,
                "where": 'Status=="AUTHORISED"',
                "order": "Date DESC",
                "page": 2,
                "pageSize": 100,
                "statuses": "DRAFT,AUTHORISED",
                "invoice_numbers": "INV-001,INV-002",
                "contact_ids": "c1,c2",
            },
            xr_context,
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params["where"] == 'Status=="AUTHORISED"'
        assert params["order"] == "Date DESC"
        assert params["page"] == "2"
        assert params["pageSize"] == "100"
        assert params["statuses"] == "DRAFT,AUTHORISED"
        assert params["InvoiceNumbers"] == "INV-001,INV-002"
        assert params["ContactIDs"] == "c1,c2"

    @pytest.mark.asyncio
    async def test_id_list_params_use_xero_pascal_case(self, xr_context):
        """`invoice_numbers`→`InvoiceNumbers` and `contact_ids`→`ContactIDs`, but
        `statuses` stays lowercase -- three different casing conventions in one
        params dict."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await GetInvoicesAction().execute(
            {"tenant_id": TENANT_ID, "invoice_numbers": "INV-1", "contact_ids": "c1", "statuses": "DRAFT"},
            xr_context,
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert set(params) == {"InvoiceNumbers", "ContactIDs", "statuses"}

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, xr_context):
        with pytest.raises(ValueError, match="tenant_id is required"):
            await GetInvoicesAction().execute({}, xr_context)

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, xr_context):
        xr_context.fetch.return_value = None

        with pytest.raises(Exception, match="Failed to fetch invoices"):
            await GetInvoicesAction().execute({"tenant_id": TENANT_ID}, xr_context)


# ---- get_bank_transactions ----


class TestGetBankTransactions:
    @pytest.mark.asyncio
    async def test_request_url_and_method(self, xr_context):
        xr_context.fetch.return_value = {"BankTransactions": []}

        await GetBankTransactionsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/BankTransactions"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_filters_forwarded(self, xr_context):
        xr_context.fetch.return_value = {"BankTransactions": []}

        await GetBankTransactionsAction().execute(
            {"tenant_id": TENANT_ID, "where": "Type==\"SPEND\"", "order": "Date DESC", "page": 3},
            xr_context,
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params["where"] == 'Type=="SPEND"'
        assert params["order"] == "Date DESC"
        assert params["page"] == "3"

    @pytest.mark.asyncio
    async def test_returns_raw_payload(self, xr_context):
        payload = {"BankTransactions": [{"BankTransactionID": "bt1", "Total": 99.5}]}
        xr_context.fetch.return_value = payload

        result = await GetBankTransactionsAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert result.data == payload

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, xr_context):
        with pytest.raises(ValueError, match="tenant_id is required"):
            await GetBankTransactionsAction().execute({}, xr_context)


# ---- get_invoice_pdf ----


class TestGetInvoicePdf:
    @pytest.mark.asyncio
    async def test_returns_base64_pdf(self, xr_context):
        factory, _ = stub_pdf_session()

        with patch("xero.xero.aiohttp.ClientSession", factory):
            result = await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert result.data["success"] is True
        assert base64.b64decode(result.data["file"]["content"]) == PDF_BYTES

    @pytest.mark.asyncio
    async def test_filename_includes_invoice_id(self, xr_context):
        factory, _ = stub_pdf_session()

        with patch("xero.xero.aiohttp.ClientSession", factory):
            result = await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert result.data["file"]["name"] == f"invoice_{INVOICE_ID}.pdf"

    @pytest.mark.asyncio
    async def test_requests_pdf_accept_header(self, xr_context):
        """The same URL returns JSON or PDF depending on Accept, so this header is
        what makes it a PDF download at all."""
        factory, calls = stub_pdf_session()

        with patch("xero.xero.aiohttp.ClientSession", factory):
            await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert calls["headers"]["Accept"] == "application/pdf"
        assert calls["url"] == f"{API_BASE}/Invoices/{INVOICE_ID}"

    @pytest.mark.asyncio
    async def test_sends_bearer_token_and_tenant_header(self, xr_context):
        """This action bypasses the SDK fetch, so it has to attach the OAuth token
        itself rather than relying on the platform to inject it."""
        factory, calls = stub_pdf_session()

        with patch("xero.xero.aiohttp.ClientSession", factory):
            await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert calls["headers"]["Authorization"] == f"Bearer {ACCESS_TOKEN}"
        assert calls["headers"]["xero-tenant-id"] == TENANT_ID

    @pytest.mark.asyncio
    async def test_content_type_taken_from_response(self, xr_context):
        factory, _ = stub_pdf_session(content_type="application/pdf; charset=utf-8")

        with patch("xero.xero.aiohttp.ClientSession", factory):
            result = await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert result.data["file"]["contentType"] == "application/pdf; charset=utf-8"

    @pytest.mark.asyncio
    async def test_content_type_defaults_when_header_absent(self, xr_context):
        factory, _ = stub_pdf_session(content_type=None)

        with patch("xero.xero.aiohttp.ClientSession", factory):
            result = await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert result.data["file"]["contentType"] == "application/pdf"

    @pytest.mark.asyncio
    async def test_non_200_returns_failure_with_empty_file(self, xr_context):
        factory, _ = stub_pdf_session(status=404, text="Invoice not found")

        with patch("xero.xero.aiohttp.ClientSession", factory):
            result = await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert result.data["success"] is False
        assert result.data["file"]["content"] == ""
        assert "404" in result.data["error"]

    @pytest.mark.asyncio
    async def test_missing_access_token_is_reported_not_raised(self, xr_context):
        xr_context.auth = {"credentials": {}}

        result = await GetInvoicePdfAction().execute(
            {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
        )

        assert result.data["success"] is False
        assert "No authentication token available" in result.data["error"]

    @pytest.mark.parametrize("missing", ["tenant_id", "invoice_id"])
    @pytest.mark.asyncio
    async def test_required_inputs_raise(self, xr_context, missing):
        inputs = {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await GetInvoicePdfAction().execute(inputs, xr_context)

    @pytest.mark.asyncio
    async def test_failure_envelope_always_carries_a_file_object(self, xr_context):
        """Success and failure both return a `file` key, so callers don't have to
        branch on shape -- unlike the ElevenLabs audio actions."""
        factory, _ = stub_pdf_session(status=500, text="boom")

        with patch("xero.xero.aiohttp.ClientSession", factory):
            result = await GetInvoicePdfAction().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )

        assert set(result.data["file"]) == {"name", "content", "contentType"}


# ---- create_sales_invoice / create_purchase_bill ----

CREATE_ACTIONS = [
    (CreateSalesInvoiceAction, "ACCREC", "create sales invoice"),
    (CreatePurchaseBillAction, "ACCPAY", "create purchase bill"),
]


class TestCreateInvoiceAndBill:
    @pytest.mark.parametrize("action_cls, expected_type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_type_is_hardcoded_per_action(self, xr_context, action_cls, expected_type, _err):
        """Sales invoices and purchase bills POST to the same endpoint and differ
        only by Type. ACCREC is money owed TO you, ACCPAY is money you owe -- a
        swap here books revenue as an expense."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
        )

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert invoice["Type"] == expected_type

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_posts_to_invoices_collection(self, xr_context, action_cls, _type, _err):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
        )

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/Invoices"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_body_wrapped_in_invoices_array(self, xr_context, action_cls, _type, _err):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
        )

        body = xr_context.fetch.call_args.kwargs["json"]
        assert list(body) == ["Invoices"]
        assert len(body["Invoices"]) == 1

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_contact_and_line_items_passed_through(self, xr_context, action_cls, _type, _err):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
        )

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert invoice["Contact"] == CONTACT
        assert invoice["LineItems"] == LINE_ITEMS

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_minimal_body_has_no_status(self, xr_context, action_cls, _type, _err):
        """Status is not defaulted, so Xero applies its own default (DRAFT)
        rather than the integration forcing one."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
        )

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert set(invoice) == {"Type", "Contact", "LineItems"}

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_all_optional_fields_mapped_to_pascal_case(
        self, xr_context, action_cls, _type, _err
    ):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {
                "tenant_id": TENANT_ID,
                "contact": CONTACT,
                "line_items": LINE_ITEMS,
                "date": "2026-01-15",
                "due_date": "2026-02-15",
                "invoice_number": "INV-001",
                "reference": "PO-99",
                "branding_theme_id": "bt1",
                "currency_code": "NZD",
                "status": "AUTHORISED",
                "line_amount_types": "Exclusive",
            },
            xr_context,
        )

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert invoice["Date"] == "2026-01-15"
        assert invoice["DueDate"] == "2026-02-15"
        assert invoice["InvoiceNumber"] == "INV-001"
        assert invoice["Reference"] == "PO-99"
        assert invoice["CurrencyCode"] == "NZD"
        assert invoice["Status"] == "AUTHORISED"
        assert invoice["LineAmountTypes"] == "Exclusive"

    @pytest.mark.asyncio
    async def test_branding_theme_is_sales_invoice_only(self, xr_context):
        """A branding theme controls the look of an outbound document, so it
        applies to sales invoices you send but not to bills you receive.
        create_purchase_bill deliberately omits it -- asserted both ways so the
        asymmetry reads as intentional."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD
        inputs = {
            "tenant_id": TENANT_ID,
            "contact": CONTACT,
            "line_items": LINE_ITEMS,
            "branding_theme_id": "bt1",
        }

        await CreateSalesInvoiceAction().execute(inputs, xr_context)
        assert xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]["BrandingThemeID"] == "bt1"

        await CreatePurchaseBillAction().execute(inputs, xr_context)
        assert "BrandingThemeID" not in xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_json_content_type_header(self, xr_context, action_cls, _type, _err):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
        )

        headers = xr_context.fetch.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.parametrize("missing", ["tenant_id", "contact", "line_items"])
    @pytest.mark.asyncio
    async def test_required_inputs_raise(self, xr_context, action_cls, _type, _err, missing):
        inputs = {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await action_cls().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_empty_line_items_rejected(self, xr_context, action_cls, _type, _err):
        """An invoice with no lines would post a zero-total document."""
        with pytest.raises(ValueError, match="line_items is required"):
            await action_cls().execute(
                {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": []}, xr_context
            )

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_non_list_line_items_rejected(self, xr_context, action_cls, _type, _err):
        """A single dict is a plausible mistake and is caught explicitly."""
        with pytest.raises(ValueError, match="must be a list"):
            await action_cls().execute(
                {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS[0]},
                xr_context,
            )

    @pytest.mark.parametrize("action_cls, _type, err_fragment", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_errors_are_wrapped_with_action_context(
        self, xr_context, action_cls, _type, err_fragment
    ):
        xr_context.fetch.side_effect = Exception("HTTP 400: validation failed")

        with pytest.raises(Exception, match=err_fragment):
            await action_cls().execute(
                {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
            )

    @pytest.mark.parametrize("action_cls, _type, _err", CREATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_rate_limit_returns_structured_payload(self, xr_context, action_cls, _type, _err):
        xr_context.fetch.side_effect = rate_limit_error(600)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            result = await action_cls().execute(
                {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}, xr_context
            )

        assert result.data["error_type"] == "rate_limit_exceeded"


# ---- update_sales_invoice / update_purchase_bill ----

UPDATE_ACTIONS = [
    (UpdateSalesInvoiceAction, "ACCREC", "update sales invoice"),
    (UpdatePurchaseBillAction, "ACCPAY", "update purchase bill"),
]


class TestUpdateInvoiceAndBill:
    @pytest.mark.parametrize("action_cls, expected_type, _err", UPDATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_type_is_hardcoded_per_action(self, xr_context, action_cls, expected_type, _err):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
        )

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert invoice["Type"] == expected_type

    @pytest.mark.parametrize("action_cls, _type, _err", UPDATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_posts_to_specific_invoice_url(self, xr_context, action_cls, _type, _err):
        """Xero updates via POST to the item URL, not PUT."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute({"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context)

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/Invoices/{INVOICE_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.parametrize("action_cls, _type, _err", UPDATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_invoice_id_is_echoed_in_the_body(self, xr_context, action_cls, _type, _err):
        """Xero requires InvoiceID in the payload as well as the path."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute({"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context)

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert invoice["InvoiceID"] == INVOICE_ID

    @pytest.mark.parametrize("action_cls, _type, _err", UPDATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_minimal_update_sends_only_id_and_type(self, xr_context, action_cls, _type, _err):
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute({"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context)

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert set(invoice) == {"InvoiceID", "Type"}

    @pytest.mark.parametrize("action_cls, _type, _err", UPDATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_contact_and_line_items_are_optional_on_update(
        self, xr_context, action_cls, _type, _err
    ):
        """Unlike create, these are optional -- so a status-only update won't wipe
        the line items."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID, "status": "AUTHORISED"}, xr_context
        )

        invoice = xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]
        assert "LineItems" not in invoice
        assert invoice["Status"] == "AUTHORISED"

    @pytest.mark.parametrize("action_cls, _type, _err", UPDATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_supplied_line_items_replace_wholesale(self, xr_context, action_cls, _type, _err):
        """Xero replaces the whole LineItems array, so a partial list drops lines."""
        xr_context.fetch.return_value = INVOICE_PAYLOAD

        await action_cls().execute(
            {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID, "line_items": LINE_ITEMS}, xr_context
        )

        assert xr_context.fetch.call_args.kwargs["json"]["Invoices"][0]["LineItems"] == LINE_ITEMS

    @pytest.mark.parametrize("action_cls, _type, _err", UPDATE_ACTIONS)
    @pytest.mark.parametrize("missing", ["tenant_id", "invoice_id"])
    @pytest.mark.asyncio
    async def test_required_inputs_raise(self, xr_context, action_cls, _type, _err, missing):
        inputs = {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await action_cls().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.parametrize("action_cls, _type, err_fragment", UPDATE_ACTIONS)
    @pytest.mark.asyncio
    async def test_errors_are_wrapped_with_action_context(
        self, xr_context, action_cls, _type, err_fragment
    ):
        xr_context.fetch.side_effect = Exception("HTTP 400")

        with pytest.raises(Exception, match=err_fragment):
            await action_cls().execute(
                {"tenant_id": TENANT_ID, "invoice_id": INVOICE_ID}, xr_context
            )


# ---- Config ----


class TestXeroInvoiceConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        [
            "get_payments",
            "get_invoices",
            "get_invoice_pdf",
            "get_bank_transactions",
            "create_sales_invoice",
            "create_purchase_bill",
            "update_sales_invoice",
            "update_purchase_bill",
        ],
    )
    def test_all_require_tenant_id(self, config, action):
        assert "tenant_id" in config["actions"][action]["input_schema"]["required"]

    @pytest.mark.parametrize("action", ["create_sales_invoice", "create_purchase_bill"])
    def test_create_actions_require_contact_and_line_items(self, config, action):
        required = config["actions"][action]["input_schema"]["required"]
        assert "contact" in required
        assert "line_items" in required

    @pytest.mark.parametrize(
        "action", ["update_sales_invoice", "update_purchase_bill", "get_invoice_pdf"]
    )
    def test_invoice_scoped_actions_require_invoice_id(self, config, action):
        assert "invoice_id" in config["actions"][action]["input_schema"]["required"]

    def test_get_invoices_invoice_id_is_optional(self, config):
        """get_invoices serves both list and single-item modes."""
        required = config["actions"]["get_invoices"]["input_schema"]["required"]
        assert "invoice_id" not in required
