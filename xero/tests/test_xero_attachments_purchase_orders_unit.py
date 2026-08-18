"""Unit tests for the Xero attachment and purchase order actions.

Covers file attachment (`attach_file_to_invoice`, `attach_file_to_bill`),
attachment retrieval (`get_attachments`, `get_attachment_content`), and the six
purchase order actions.

Fully mocked -- no network access.
"""

import base64
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from xero.xero import (  # noqa: E402
    AddNoteToPurchaseOrderAction,
    AttachFileToBillAction,
    AttachFileToInvoiceAction,
    CreatePurchaseOrderAction,
    DeletePurchaseOrderAction,
    GetAttachmentContentAction,
    GetAttachmentsAction,
    GetPurchaseOrderHistoryAction,
    GetPurchaseOrdersAction,
    UpdatePurchaseOrderAction,
)

pytestmark = pytest.mark.unit

TENANT_ID = "b2c3d4e5-f6a7-8901-bcde-f23456789012"
INVOICE_ID = "d4e5f6a7-b8c9-0123-def0-3456789abcde"
PO_ID = "e5f6a7b8-c9d0-1234-ef01-456789abcdef"
API_BASE = "https://api.xero.com/api.xro/2.0"
ACCESS_TOKEN = "test_access_token"  # nosec B105
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

FILE_BYTES = b"%PDF-1.4 attachment payload"
FILE_OBJ = {
    "name": "receipt.pdf",
    "contentType": "application/pdf",
    "content": base64.b64encode(FILE_BYTES).decode(),
}

CONTACT = {"ContactID": "c1"}
LINE_ITEMS = [{"Description": "Widgets", "Quantity": 10, "UnitAmount": 9.99}]
PO_PAYLOAD = {"PurchaseOrders": [{"PurchaseOrderID": PO_ID, "Status": "DRAFT"}]}
ATTACHMENT_PAYLOAD = {"Attachments": [{"AttachmentID": "at1", "FileName": "receipt.pdf"}]}


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


def wire_context_session(ctx, status=200, body=FILE_BYTES, text=""):
    """Wire the context's own aiohttp session for the attachment-content path.

    `get_attachment_content` does not open its own ClientSession -- it enters the
    ExecutionContext as an async context manager and reuses `context._session`,
    so the SDK's connection pool and auth plumbing are shared. The double has to
    mirror that shape rather than patching `aiohttp.ClientSession`.
    """
    calls = {}

    response = MagicMock(name="ClientResponse")
    response.status = status
    response.read = AsyncMock(return_value=body)
    response.text = AsyncMock(return_value=text)
    response.headers = {"content-type": "application/octet-stream"}

    resp_ctx = MagicMock()
    resp_ctx.__aenter__ = AsyncMock(return_value=response)
    resp_ctx.__aexit__ = AsyncMock(return_value=False)

    def record(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return resp_ctx

    session = MagicMock(name="session")
    session.get = record

    ctx._session = session
    ctx.__aenter__ = AsyncMock(return_value=ctx)
    ctx.__aexit__ = AsyncMock(return_value=False)

    return calls


# ---- attach_file_to_invoice / attach_file_to_bill ----

ATTACH_ACTIONS = [
    (AttachFileToInvoiceAction, "invoice_id", "attach file to invoice"),
    (AttachFileToBillAction, "bill_id", "attach file to bill"),
]


def attach_inputs(id_key, **overrides):
    inputs = {"tenant_id": TENANT_ID, id_key: INVOICE_ID, "file": dict(FILE_OBJ)}
    inputs.update(overrides)
    return inputs


class TestAttachFile:
    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_url_embeds_the_filename(self, xr_context, action_cls, id_key, _err):
        """Xero derives the attachment name from the URL path, not the body."""
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await action_cls().execute(attach_inputs(id_key), xr_context)

        url = xr_context.fetch.call_args.args[0]
        assert url == f"{API_BASE}/Invoices/{INVOICE_ID}/Attachments/receipt.pdf"

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_both_actions_target_the_invoices_resource(
        self, xr_context, action_cls, id_key, _err
    ):
        """Bills are invoices in Xero's data model, so attach_file_to_bill also
        posts under /Invoices -- there is no /Bills resource."""
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await action_cls().execute(attach_inputs(id_key), xr_context)

        assert "/Invoices/" in xr_context.fetch.call_args.args[0]
        assert "/Bills/" not in xr_context.fetch.call_args.args[0]

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_sends_raw_bytes_not_json(self, xr_context, action_cls, id_key, _err):
        """The attachment body is the decoded file, sent as `data`."""
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await action_cls().execute(attach_inputs(id_key), xr_context)

        call = xr_context.fetch.call_args
        assert call.kwargs["data"] == FILE_BYTES
        assert "json" not in call.kwargs
        assert call.kwargs["method"] == "POST"

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_content_type_header_comes_from_the_file(
        self, xr_context, action_cls, id_key, _err
    ):
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await action_cls().execute(
            attach_inputs(id_key, file={**FILE_OBJ, "contentType": "image/png"}), xr_context
        )

        assert xr_context.fetch.call_args.kwargs["headers"]["Content-Type"] == "image/png"

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_files_array_falls_back_to_first_entry(
        self, xr_context, action_cls, id_key, _err
    ):
        """The platform may deliver `files` (plural); only the first is attached."""
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD
        inputs = attach_inputs(id_key)
        del inputs["file"]
        inputs["files"] = [FILE_OBJ, {**FILE_OBJ, "name": "second.pdf"}]

        await action_cls().execute(inputs, xr_context)

        assert xr_context.fetch.call_args.args[0].endswith("receipt.pdf")

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_include_online_is_lowercased(self, xr_context, action_cls, id_key, _err):
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await action_cls().execute(attach_inputs(id_key, include_online=True), xr_context)

        assert xr_context.fetch.call_args.kwargs["params"]["IncludeOnline"] == "true"

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_include_online_false_is_still_sent(self, xr_context, action_cls, id_key, _err):
        """Gated on `is not None`, so an explicit False reaches Xero -- which
        matters, since it controls whether a customer can see the attachment."""
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await action_cls().execute(attach_inputs(id_key, include_online=False), xr_context)

        assert xr_context.fetch.call_args.kwargs["params"]["IncludeOnline"] == "false"

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_include_online_omitted_when_absent(self, xr_context, action_cls, id_key, _err):
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await action_cls().execute(attach_inputs(id_key), xr_context)

        assert xr_context.fetch.call_args.kwargs["params"] == {}

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_missing_file_object_is_rejected(self, xr_context, action_cls, id_key, _err):
        inputs = attach_inputs(id_key)
        del inputs["file"]

        with pytest.raises(Exception, match="file object is required"):
            await action_cls().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_invoice_variant_enumerates_file_keys_but_bill_does_not(self, xr_context):
        """attach_file_to_invoice appends the available file-related input keys to
        its error, which is what makes a platform file-injection failure
        diagnosable. attach_file_to_bill raises the bare message. Asserted both
        ways so the gap is visible rather than assumed absent."""
        inputs = attach_inputs("invoice_id")
        del inputs["file"]
        inputs["file_path"] = "/tmp/x.pdf"  # nosec B108

        with pytest.raises(Exception, match="file_path"):
            await AttachFileToInvoiceAction().execute(inputs, xr_context)

        bill_inputs = attach_inputs("bill_id")
        del bill_inputs["file"]
        bill_inputs["file_path"] = "/tmp/x.pdf"  # nosec B108

        with pytest.raises(Exception) as exc:
            await AttachFileToBillAction().execute(bill_inputs, xr_context)

        assert "file_path" not in str(exc.value)

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_missing_name_is_rejected(self, xr_context, action_cls, id_key, _err):
        with pytest.raises(Exception, match="missing 'name'"):
            await action_cls().execute(
                attach_inputs(id_key, file={**FILE_OBJ, "name": ""}), xr_context
            )

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_whitespace_only_name_is_rejected(self, xr_context, action_cls, id_key, _err):
        """The name is stripped before the check, so "   " counts as missing."""
        with pytest.raises(Exception, match="missing 'name'"):
            await action_cls().execute(
                attach_inputs(id_key, file={**FILE_OBJ, "name": "   "}), xr_context
            )

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_missing_content_type_is_rejected(self, xr_context, action_cls, id_key, _err):
        """Xero stores whatever Content-Type it is given, so a blank one would
        make the attachment undownloadable."""
        with pytest.raises(Exception, match="missing 'contentType'"):
            await action_cls().execute(
                attach_inputs(id_key, file={**FILE_OBJ, "contentType": ""}), xr_context
            )

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_invalid_base64_content_is_rejected(self, xr_context, action_cls, id_key, _err):
        with pytest.raises(Exception, match="not valid base64"):
            await action_cls().execute(
                attach_inputs(id_key, file={**FILE_OBJ, "content": "!!!bad!!!"}), xr_context
            )

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, xr_context, action_cls, id_key, _err):
        inputs = attach_inputs(id_key)
        del inputs["tenant_id"]

        with pytest.raises(ValueError, match="tenant_id is required"):
            await action_cls().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_missing_id_raises(self, xr_context, action_cls, id_key, _err):
        inputs = attach_inputs(id_key)
        del inputs[id_key]

        with pytest.raises(ValueError, match=f"{id_key} is required"):
            await action_cls().execute(inputs, xr_context)

    @pytest.mark.parametrize("action_cls, id_key, _err", ATTACH_ACTIONS)
    @pytest.mark.asyncio
    async def test_rate_limit_returns_structured_payload(
        self, xr_context, action_cls, id_key, _err
    ):
        xr_context.fetch.side_effect = rate_limit_error(600)

        with patch("xero.xero.asyncio.sleep", new_callable=AsyncMock):
            result = await action_cls().execute(attach_inputs(id_key), xr_context)

        assert result.data["error_type"] == "rate_limit_exceeded"


# ---- get_attachments ----


class TestGetAttachments:
    @pytest.mark.asyncio
    async def test_returns_attachments(self, xr_context):
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        result = await GetAttachmentsAction().execute(
            {"tenant_id": TENANT_ID, "endpoint": "Invoices", "guid": INVOICE_ID}, xr_context
        )

        assert result.data["Attachments"][0]["FileName"] == "receipt.pdf"

    @pytest.mark.parametrize("endpoint", ["Invoices", "PurchaseOrders", "BankTransactions", "Contacts"])
    @pytest.mark.asyncio
    async def test_endpoint_is_interpolated_verbatim(self, xr_context, endpoint):
        """The caller names the parent resource, so one action serves every
        attachable type -- but the value is unvalidated and goes straight into
        the URL path."""
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await GetAttachmentsAction().execute(
            {"tenant_id": TENANT_ID, "endpoint": endpoint, "guid": INVOICE_ID}, xr_context
        )

        assert xr_context.fetch.call_args.args[0] == f"{API_BASE}/{endpoint}/{INVOICE_ID}/Attachments"

    @pytest.mark.asyncio
    async def test_request_method_and_headers(self, xr_context):
        xr_context.fetch.return_value = ATTACHMENT_PAYLOAD

        await GetAttachmentsAction().execute(
            {"tenant_id": TENANT_ID, "endpoint": "Invoices", "guid": INVOICE_ID}, xr_context
        )

        call = xr_context.fetch.call_args
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["headers"]["xero-tenant-id"] == TENANT_ID

    @pytest.mark.parametrize("missing", ["tenant_id", "endpoint", "guid"])
    @pytest.mark.asyncio
    async def test_required_inputs_raise(self, xr_context, missing):
        inputs = {"tenant_id": TENANT_ID, "endpoint": "Invoices", "guid": INVOICE_ID}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await GetAttachmentsAction().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_response_raises(self, xr_context):
        xr_context.fetch.return_value = None

        with pytest.raises(Exception, match="attachments"):
            await GetAttachmentsAction().execute(
                {"tenant_id": TENANT_ID, "endpoint": "Invoices", "guid": INVOICE_ID}, xr_context
            )


# ---- get_attachment_content ----


class TestGetAttachmentContent:
    def base_inputs(self):
        return {
            "tenant_id": TENANT_ID,
            "endpoint": "Invoices",
            "guid": INVOICE_ID,
            "file_name": "receipt.pdf",
        }

    @pytest.mark.asyncio
    async def test_returns_base64_content(self, xr_context):
        wire_context_session(xr_context)

        result = await GetAttachmentContentAction().execute(self.base_inputs(), xr_context)

        assert base64.b64decode(result.data["file"]["content"]) == FILE_BYTES

    @pytest.mark.asyncio
    async def test_reuses_the_context_session(self, xr_context):
        """Unlike get_invoice_pdf, this action reuses the SDK's session via
        `context._session` instead of opening its own ClientSession -- so it
        inherits the connection pool. Asserted so the two binary-download paths
        aren't assumed to work the same way."""
        wire_context_session(xr_context)

        await GetAttachmentContentAction().execute(self.base_inputs(), xr_context)

        xr_context.__aenter__.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_url_includes_endpoint_guid_and_filename(self, xr_context):
        calls = wire_context_session(xr_context)

        await GetAttachmentContentAction().execute(self.base_inputs(), xr_context)

        assert calls["url"] == f"{API_BASE}/Invoices/{INVOICE_ID}/Attachments/receipt.pdf"

    @pytest.mark.asyncio
    async def test_requests_octet_stream(self, xr_context):
        """Without this Accept header Xero returns attachment metadata as JSON
        rather than the file itself."""
        calls = wire_context_session(xr_context)

        await GetAttachmentContentAction().execute(self.base_inputs(), xr_context)

        assert calls["headers"]["Accept"] == "application/octet-stream"
        assert calls["headers"]["xero-tenant-id"] == TENANT_ID

    @pytest.mark.asyncio
    async def test_sends_bearer_token(self, xr_context):
        calls = wire_context_session(xr_context)

        await GetAttachmentContentAction().execute(self.base_inputs(), xr_context)

        assert calls["headers"]["Authorization"] == f"Bearer {ACCESS_TOKEN}"

    @pytest.mark.asyncio
    async def test_non_200_is_reported_not_raised(self, xr_context):
        wire_context_session(xr_context, status=404, text="Attachment not found")

        result = await GetAttachmentContentAction().execute(self.base_inputs(), xr_context)

        assert result.data["success"] is False
        assert result.data["file"]["content"] == ""

    @pytest.mark.parametrize("missing", ["tenant_id", "endpoint", "guid", "file_name"])
    @pytest.mark.asyncio
    async def test_all_four_inputs_are_required(self, xr_context, missing):
        inputs = self.base_inputs()
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await GetAttachmentContentAction().execute(inputs, xr_context)


# ---- get_purchase_orders ----


class TestGetPurchaseOrders:
    @pytest.mark.asyncio
    async def test_list_mode_targets_collection(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await GetPurchaseOrdersAction().execute({"tenant_id": TENANT_ID}, xr_context)

        assert xr_context.fetch.call_args.args[0] == f"{API_BASE}/PurchaseOrders"

    @pytest.mark.asyncio
    async def test_single_mode_targets_purchase_order_id(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await GetPurchaseOrdersAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        assert xr_context.fetch.call_args.args[0] == f"{API_BASE}/PurchaseOrders/{PO_ID}"

    @pytest.mark.asyncio
    async def test_filters_ignored_in_single_mode(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await GetPurchaseOrdersAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID, "where": 'Status=="DRAFT"'},
            xr_context,
        )

        assert xr_context.fetch.call_args.kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_list_filters_forwarded(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await GetPurchaseOrdersAction().execute(
            {
                "tenant_id": TENANT_ID,
                "where": 'Status=="AUTHORISED"',
                "order": "Date DESC",
                "page": 2,
                "statuses": "DRAFT,AUTHORISED",
            },
            xr_context,
        )

        params = xr_context.fetch.call_args.kwargs["params"]
        assert params["where"] == 'Status=="AUTHORISED"'
        assert params["order"] == "Date DESC"
        assert params["page"] == "2"
        assert params["statuses"] == "DRAFT,AUTHORISED"

    @pytest.mark.asyncio
    async def test_no_page_size_filter_unlike_invoices(self, xr_context):
        """get_invoices supports pageSize; purchase orders do not."""
        xr_context.fetch.return_value = PO_PAYLOAD

        await GetPurchaseOrdersAction().execute(
            {"tenant_id": TENANT_ID, "pageSize": 100}, xr_context
        )

        assert "pageSize" not in xr_context.fetch.call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_missing_tenant_id_raises(self, xr_context):
        with pytest.raises(ValueError, match="tenant_id is required"):
            await GetPurchaseOrdersAction().execute({}, xr_context)


# ---- create_purchase_order ----


class TestCreatePurchaseOrder:
    def base_inputs(self, **overrides):
        inputs = {"tenant_id": TENANT_ID, "contact": CONTACT, "line_items": LINE_ITEMS}
        inputs.update(overrides)
        return inputs

    @pytest.mark.asyncio
    async def test_posts_to_purchase_orders_collection(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await CreatePurchaseOrderAction().execute(self.base_inputs(), xr_context)

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/PurchaseOrders"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_body_wrapped_in_purchase_orders_array(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await CreatePurchaseOrderAction().execute(self.base_inputs(), xr_context)

        body = xr_context.fetch.call_args.kwargs["json"]
        assert list(body) == ["PurchaseOrders"]
        assert body["PurchaseOrders"][0]["Contact"] == CONTACT

    @pytest.mark.asyncio
    async def test_no_type_field_is_sent(self, xr_context):
        """Purchase orders are their own resource, so unlike invoices and bills
        there is no ACCREC/ACCPAY discriminator."""
        xr_context.fetch.return_value = PO_PAYLOAD

        await CreatePurchaseOrderAction().execute(self.base_inputs(), xr_context)

        assert "Type" not in xr_context.fetch.call_args.kwargs["json"]["PurchaseOrders"][0]

    @pytest.mark.asyncio
    async def test_all_optional_fields_mapped(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await CreatePurchaseOrderAction().execute(
            self.base_inputs(
                date="2026-01-15",
                delivery_date="2026-02-01",
                purchase_order_number="PO-001",
                reference="REQ-9",
                currency_code="NZD",
                status="AUTHORISED",
                line_amount_types="Exclusive",
                delivery_address="1 Queen St",
                attention_to="Ada",
                telephone="+64123",
                delivery_instructions="Leave at reception",
            ),
            xr_context,
        )

        po = xr_context.fetch.call_args.kwargs["json"]["PurchaseOrders"][0]
        assert po["Date"] == "2026-01-15"
        assert po["DeliveryDate"] == "2026-02-01"
        assert po["PurchaseOrderNumber"] == "PO-001"
        assert po["Reference"] == "REQ-9"
        assert po["CurrencyCode"] == "NZD"
        assert po["Status"] == "AUTHORISED"
        assert po["LineAmountTypes"] == "Exclusive"
        assert po["DeliveryAddress"] == "1 Queen St"
        assert po["AttentionTo"] == "Ada"
        assert po["Telephone"] == "+64123"
        assert po["DeliveryInstructions"] == "Leave at reception"

    @pytest.mark.asyncio
    async def test_minimal_body_omits_optionals(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await CreatePurchaseOrderAction().execute(self.base_inputs(), xr_context)

        po = xr_context.fetch.call_args.kwargs["json"]["PurchaseOrders"][0]
        assert set(po) == {"Contact", "LineItems"}

    @pytest.mark.parametrize("missing", ["tenant_id", "contact", "line_items"])
    @pytest.mark.asyncio
    async def test_required_inputs_raise(self, xr_context, missing):
        inputs = self.base_inputs()
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await CreatePurchaseOrderAction().execute(inputs, xr_context)

    @pytest.mark.asyncio
    async def test_non_list_line_items_rejected(self, xr_context):
        with pytest.raises(ValueError, match="must be a list"):
            await CreatePurchaseOrderAction().execute(
                self.base_inputs(line_items=LINE_ITEMS[0]), xr_context
            )


# ---- update_purchase_order ----


class TestUpdatePurchaseOrder:
    @pytest.mark.asyncio
    async def test_posts_to_specific_purchase_order(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await UpdatePurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/PurchaseOrders/{PO_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_purchase_order_id_echoed_in_body(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await UpdatePurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        po = xr_context.fetch.call_args.kwargs["json"]["PurchaseOrders"][0]
        assert po["PurchaseOrderID"] == PO_ID

    @pytest.mark.asyncio
    async def test_line_items_optional_on_update(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await UpdatePurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID, "status": "AUTHORISED"}, xr_context
        )

        po = xr_context.fetch.call_args.kwargs["json"]["PurchaseOrders"][0]
        assert "LineItems" not in po
        assert po["Status"] == "AUTHORISED"

    @pytest.mark.parametrize("missing", ["tenant_id", "purchase_order_id"])
    @pytest.mark.asyncio
    async def test_required_inputs_raise(self, xr_context, missing):
        inputs = {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await UpdatePurchaseOrderAction().execute(inputs, xr_context)


# ---- delete_purchase_order ----


class TestDeletePurchaseOrder:
    @pytest.mark.asyncio
    async def test_deletion_is_a_status_change_not_a_delete_verb(self, xr_context):
        """Xero has no DELETE for purchase orders -- deletion is a POST setting
        Status to DELETED. A caller expecting a hard delete gets a soft one."""
        xr_context.fetch.return_value = PO_PAYLOAD

        await DeletePurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        call = xr_context.fetch.call_args
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["json"]["PurchaseOrders"][0]["Status"] == "DELETED"

    @pytest.mark.asyncio
    async def test_targets_the_specific_purchase_order(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await DeletePurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        assert xr_context.fetch.call_args.args[0] == f"{API_BASE}/PurchaseOrders/{PO_ID}"

    @pytest.mark.asyncio
    async def test_body_carries_only_id_and_status(self, xr_context):
        xr_context.fetch.return_value = PO_PAYLOAD

        await DeletePurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        po = xr_context.fetch.call_args.kwargs["json"]["PurchaseOrders"][0]
        assert set(po) == {"PurchaseOrderID", "Status"}

    @pytest.mark.parametrize("missing", ["tenant_id", "purchase_order_id"])
    @pytest.mark.asyncio
    async def test_required_inputs_raise(self, xr_context, missing):
        inputs = {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await DeletePurchaseOrderAction().execute(inputs, xr_context)


# ---- get_purchase_order_history / add_note_to_purchase_order ----


class TestPurchaseOrderHistory:
    @pytest.mark.asyncio
    async def test_history_uses_get_on_history_subresource(self, xr_context):
        xr_context.fetch.return_value = {"HistoryRecords": []}

        await GetPurchaseOrderHistoryAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/PurchaseOrders/{PO_ID}/History"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_history_returns_raw_payload(self, xr_context):
        payload = {"HistoryRecords": [{"Details": "Note added", "DateUTC": "2026-01-15"}]}
        xr_context.fetch.return_value = payload

        result = await GetPurchaseOrderHistoryAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}, xr_context
        )

        assert result.data == payload

    @pytest.mark.asyncio
    async def test_add_note_uses_put_on_the_same_url(self, xr_context):
        """Reading history is a GET and writing a note is a PUT -- same URL, so
        the verb is the only thing distinguishing them."""
        xr_context.fetch.return_value = {"HistoryRecords": []}

        await AddNoteToPurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID, "note": "Approved"}, xr_context
        )

        call = xr_context.fetch.call_args
        assert call.args[0] == f"{API_BASE}/PurchaseOrders/{PO_ID}/History"
        assert call.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_note_is_wrapped_as_a_history_record(self, xr_context):
        xr_context.fetch.return_value = {"HistoryRecords": []}

        await AddNoteToPurchaseOrderAction().execute(
            {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID, "note": "Approved by Ada"},
            xr_context,
        )

        assert xr_context.fetch.call_args.kwargs["json"] == {
            "HistoryRecords": [{"Details": "Approved by Ada"}]
        }

    @pytest.mark.parametrize("missing", ["tenant_id", "purchase_order_id", "note"])
    @pytest.mark.asyncio
    async def test_add_note_required_inputs_raise(self, xr_context, missing):
        inputs = {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID, "note": "n"}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await AddNoteToPurchaseOrderAction().execute(inputs, xr_context)

        xr_context.fetch.assert_not_called()

    @pytest.mark.parametrize("missing", ["tenant_id", "purchase_order_id"])
    @pytest.mark.asyncio
    async def test_history_required_inputs_raise(self, xr_context, missing):
        inputs = {"tenant_id": TENANT_ID, "purchase_order_id": PO_ID}
        del inputs[missing]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            await GetPurchaseOrderHistoryAction().execute(inputs, xr_context)


# ---- Config ----


class TestXeroAttachmentAndPoConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        [
            "attach_file_to_invoice",
            "attach_file_to_bill",
            "get_attachments",
            "get_attachment_content",
            "get_purchase_orders",
            "create_purchase_order",
            "update_purchase_order",
            "delete_purchase_order",
            "get_purchase_order_history",
            "add_note_to_purchase_order",
        ],
    )
    def test_all_require_tenant_id(self, config, action):
        assert "tenant_id" in config["actions"][action]["input_schema"]["required"]

    @pytest.mark.parametrize("action", ["get_attachments", "get_attachment_content"])
    def test_attachment_reads_require_endpoint_and_guid(self, config, action):
        required = config["actions"][action]["input_schema"]["required"]
        assert "endpoint" in required
        assert "guid" in required

    def test_get_attachment_content_requires_file_name(self, config):
        assert "file_name" in config["actions"]["get_attachment_content"]["input_schema"]["required"]

    def test_create_purchase_order_requires_contact_and_line_items(self, config):
        required = config["actions"]["create_purchase_order"]["input_schema"]["required"]
        assert "contact" in required
        assert "line_items" in required

    def test_add_note_requires_note(self, config):
        assert "note" in config["actions"]["add_note_to_purchase_order"]["input_schema"]["required"]

    @pytest.mark.parametrize(
        "action",
        ["update_purchase_order", "delete_purchase_order", "get_purchase_order_history", "add_note_to_purchase_order"],
    )
    def test_po_scoped_actions_require_purchase_order_id(self, config, action):
        assert "purchase_order_id" in config["actions"][action]["input_schema"]["required"]
