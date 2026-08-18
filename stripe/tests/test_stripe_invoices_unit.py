"""Unit tests for the Stripe invoice actions.

Covers the nine invoice actions: `list_invoices`, `get_invoice`,
`create_invoice`, `update_invoice`, `delete_invoice`, and the four lifecycle
transitions (`finalize_invoice`, `send_invoice`, `pay_invoice`,
`void_invoice`).

The lifecycle transitions are the interesting part -- each is a POST to a
distinct sub-path, and mixing them up moves an invoice to the wrong terminal
state.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from stripe.stripe import (  # noqa: E402
    API_VERSION,
    STRIPE_API_BASE_URL,
    CreateInvoiceAction,
    DeleteInvoiceAction,
    FinalizeInvoiceAction,
    GetInvoiceAction,
    ListInvoicesAction,
    PayInvoiceAction,
    SendInvoiceAction,
    UpdateInvoiceAction,
    VoidInvoiceAction,
)

pytestmark = pytest.mark.unit

CUSTOMER_ID = "cus_NffrFeUfNV2Hib"
INVOICE_ID = "in_1MtHbELkdIwHu7ixl4OzzPMv"
API_ROOT = f"{STRIPE_API_BASE_URL}/{API_VERSION}"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_INVOICE = {
    "id": INVOICE_ID,
    "object": "invoice",
    "customer": CUSTOMER_ID,
    "status": "draft",
    "total": 2500,
    "currency": "nzd",
}


@pytest.fixture
def sp_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": "test_token"}}  # nosec B105
    return ctx


# ---- list_invoices ----


class TestListInvoices:
    @pytest.mark.asyncio
    async def test_returns_invoices_and_has_more(self, sp_context):
        sp_context.fetch.return_value = {"data": [SAMPLE_INVOICE], "has_more": True}

        result = await ListInvoicesAction().execute({}, sp_context)

        assert result.data["result"] is True
        assert result.data["invoices"] == [SAMPLE_INVOICE]
        assert result.data["has_more"] is True

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListInvoicesAction().execute({}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoices"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_customer_and_status_filters_forwarded(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListInvoicesAction().execute(
            {"customer": CUSTOMER_ID, "status": "open"}, sp_context
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["customer"] == CUSTOMER_ID
        assert params["status"] == "open"

    @pytest.mark.asyncio
    async def test_created_filters_use_bracket_syntax(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListInvoicesAction().execute(
            {"created_gte": 1704067200, "created_lte": 1735689600}, sp_context
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["created[gte]"] == 1704067200
        assert params["created[lte]"] == 1735689600

    @pytest.mark.asyncio
    async def test_pagination_is_inherited_from_the_shared_helper(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListInvoicesAction().execute({"limit": 500, "starting_after": "in_a"}, sp_context)

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["limit"] == 100
        assert params["starting_after"] == "in_a"

    @pytest.mark.asyncio
    async def test_missing_data_key_yields_empty_list(self, sp_context):
        sp_context.fetch.return_value = {}

        result = await ListInvoicesAction().execute({}, sp_context)

        assert result.data["invoices"] == []
        assert result.data["has_more"] is False

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListInvoicesAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["invoices"] == []


# ---- get_invoice ----


class TestGetInvoice:
    @pytest.mark.asyncio
    async def test_returns_invoice(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        result = await GetInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["invoice"]["status"] == "draft"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await GetInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoices/{INVOICE_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_invoice_id_is_captured(self, sp_context):
        result = await GetInvoiceAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["invoice"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 404: No such invoice")

        result = await GetInvoiceAction().execute({"invoice_id": "in_missing"}, sp_context)

        assert result.data["result"] is False
        assert "No such invoice" in result.data["error"]


# ---- create_invoice ----


class TestCreateInvoice:
    @pytest.mark.asyncio
    async def test_creates_invoice(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        result = await CreateInvoiceAction().execute({"customer": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["invoice"]["id"] == INVOICE_ID

    @pytest.mark.asyncio
    async def test_request_url_method_and_form_body(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await CreateInvoiceAction().execute({"customer": CUSTOMER_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoices"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["data"] == {"customer": CUSTOMER_ID}
        assert "json" not in call.kwargs

    @pytest.mark.asyncio
    async def test_all_optional_fields_forwarded(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await CreateInvoiceAction().execute(
            {
                "customer": CUSTOMER_ID,
                "currency": "nzd",
                "description": "January services",
                "collection_method": "send_invoice",
                "days_until_due": 14,
                "metadata": {"po": "PO-9"},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["currency"] == "nzd"
        assert data["description"] == "January services"
        assert data["collection_method"] == "send_invoice"
        assert data["days_until_due"] == "14"
        assert data["metadata[po]"] == "PO-9"

    @pytest.mark.asyncio
    async def test_auto_advance_false_is_sent(self, sp_context):
        """auto_advance is presence-gated, not truthiness-gated -- so an explicit
        False reaches Stripe. That matters: auto_advance controls whether Stripe
        automatically finalizes and charges the invoice, so silently dropping
        False would let an invoice auto-collect against the caller's intent."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await CreateInvoiceAction().execute(
            {"customer": CUSTOMER_ID, "auto_advance": False}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["auto_advance"] == "false"

    @pytest.mark.asyncio
    async def test_auto_advance_true_is_sent_as_lowercase(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await CreateInvoiceAction().execute(
            {"customer": CUSTOMER_ID, "auto_advance": True}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["auto_advance"] == "true"

    @pytest.mark.asyncio
    async def test_auto_advance_omitted_when_absent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await CreateInvoiceAction().execute({"customer": CUSTOMER_ID}, sp_context)

        assert "auto_advance" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_days_until_due_zero_is_dropped(self, sp_context):
        """Truthiness-gated, so 0 (due immediately) can't be expressed."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await CreateInvoiceAction().execute(
            {"customer": CUSTOMER_ID, "days_until_due": 0}, sp_context
        )

        assert "days_until_due" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_missing_customer_is_captured(self, sp_context):
        result = await CreateInvoiceAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["invoice"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 400: No such customer")

        result = await CreateInvoiceAction().execute({"customer": "cus_bad"}, sp_context)

        assert result.data["result"] is False


# ---- update_invoice ----


class TestUpdateInvoice:
    @pytest.mark.asyncio
    async def test_request_uses_post_to_invoice_id(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await UpdateInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "description": "Updated"}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoices/{INVOICE_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await UpdateInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "description": "Updated"}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"description": "Updated"}

    @pytest.mark.asyncio
    async def test_customer_is_not_updatable(self, sp_context):
        """create_invoice takes `customer` but update omits it from its field
        list -- an invoice cannot be reassigned to another customer here."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await UpdateInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "customer": "cus_other"}, sp_context
        )

        assert "customer" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_currency_is_not_updatable(self, sp_context):
        """Stripe forbids changing an invoice's currency, so its absence here is
        correct rather than an oversight."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await UpdateInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "currency": "usd"}, sp_context
        )

        assert "currency" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_auto_advance_false_survives_on_update(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await UpdateInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "auto_advance": False}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["auto_advance"] == "false"

    @pytest.mark.asyncio
    async def test_id_only_update_sends_empty_form(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await UpdateInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_missing_invoice_id_is_captured(self, sp_context):
        result = await UpdateInvoiceAction().execute({"description": "x"}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- delete_invoice ----


class TestDeleteInvoice:
    @pytest.mark.asyncio
    async def test_reports_deletion(self, sp_context):
        sp_context.fetch.return_value = {"id": INVOICE_ID, "deleted": True}

        result = await DeleteInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["deleted"] is True
        assert result.data["id"] == INVOICE_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"deleted": True}

        await DeleteInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoices/{INVOICE_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_delete_only_works_on_drafts_at_the_api_level(self, sp_context):
        """Stripe rejects deleting a finalized invoice; the handler surfaces that
        as a captured error rather than translating it."""
        sp_context.fetch.side_effect = Exception(
            "HTTP 400: You can only delete draft invoices"
        )

        result = await DeleteInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["result"] is False
        assert result.data["deleted"] is False
        assert "draft invoices" in result.data["error"]

    @pytest.mark.asyncio
    async def test_deleted_defaults_to_true(self, sp_context):
        sp_context.fetch.return_value = {"id": INVOICE_ID}

        result = await DeleteInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["deleted"] is True

    @pytest.mark.asyncio
    async def test_error_echoes_the_requested_id(self, sp_context):
        sp_context.fetch.side_effect = Exception("boom")

        result = await DeleteInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["id"] == INVOICE_ID

    @pytest.mark.asyncio
    async def test_missing_invoice_id_yields_empty_id(self, sp_context):
        result = await DeleteInvoiceAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["id"] == ""


# ---- Lifecycle transitions ----

# Each transition is a POST to a distinct sub-path off the invoice.
TRANSITIONS = [
    (FinalizeInvoiceAction, "finalize"),
    (SendInvoiceAction, "send"),
    (PayInvoiceAction, "pay"),
    (VoidInvoiceAction, "void"),
]


class TestInvoiceLifecycleTransitions:
    @pytest.mark.parametrize("action_cls, sub_path", TRANSITIONS)
    @pytest.mark.asyncio
    async def test_each_transition_has_its_own_sub_path(self, sp_context, action_cls, sub_path):
        """finalize, send, pay, and void are four different terminal outcomes --
        voiding an invoice you meant to pay is not recoverable, so the paths must
        not drift."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await action_cls().execute({"invoice_id": INVOICE_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoices/{INVOICE_ID}/{sub_path}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.parametrize("action_cls, _sub_path", TRANSITIONS)
    @pytest.mark.asyncio
    async def test_returns_the_updated_invoice(self, sp_context, action_cls, _sub_path):
        sp_context.fetch.return_value = {**SAMPLE_INVOICE, "status": "paid"}

        result = await action_cls().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["invoice"]["status"] == "paid"

    @pytest.mark.parametrize("action_cls, _sub_path", TRANSITIONS)
    @pytest.mark.asyncio
    async def test_missing_invoice_id_is_captured(self, sp_context, action_cls, _sub_path):
        result = await action_cls().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["invoice"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.parametrize("action_cls, _sub_path", TRANSITIONS)
    @pytest.mark.asyncio
    async def test_errors_are_captured(self, sp_context, action_cls, _sub_path):
        sp_context.fetch.side_effect = Exception("HTTP 400: Invalid invoice status")

        result = await action_cls().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["result"] is False
        assert "Invalid invoice status" in result.data["error"]

    @pytest.mark.parametrize("action_cls, _sub_path", [(SendInvoiceAction, "send"), (VoidInvoiceAction, "void")])
    @pytest.mark.asyncio
    async def test_send_and_void_take_no_body(self, sp_context, action_cls, _sub_path):
        """These two accept no parameters at all, so no form body is built."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await action_cls().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert "data" not in sp_context.fetch.call_args.kwargs


class TestFinalizeInvoice:
    @pytest.mark.asyncio
    async def test_auto_advance_forwarded_when_supplied(self, sp_context):
        """Finalizing with auto_advance controls whether Stripe then attempts
        collection automatically."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await FinalizeInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "auto_advance": True}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["auto_advance"] == "true"

    @pytest.mark.asyncio
    async def test_auto_advance_false_is_forwarded(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await FinalizeInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "auto_advance": False}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["auto_advance"] == "false"

    @pytest.mark.asyncio
    async def test_empty_body_when_auto_advance_absent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await FinalizeInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}


class TestPayInvoice:
    @pytest.mark.asyncio
    async def test_payment_method_forwarded(self, sp_context):
        sp_context.fetch.return_value = {**SAMPLE_INVOICE, "status": "paid"}

        await PayInvoiceAction().execute(
            {"invoice_id": INVOICE_ID, "payment_method": "pm_card_visa"}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["payment_method"] == "pm_card_visa"

    @pytest.mark.asyncio
    async def test_omitted_payment_method_uses_the_default(self, sp_context):
        """With no payment_method Stripe charges the customer's default source."""
        sp_context.fetch.return_value = SAMPLE_INVOICE

        await PayInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_card_decline_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 402: Your card was declined")

        result = await PayInvoiceAction().execute({"invoice_id": INVOICE_ID}, sp_context)

        assert result.data["result"] is False
        assert "declined" in result.data["error"]


# ---- Config ----


class TestStripeInvoiceConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        [
            "get_invoice",
            "update_invoice",
            "delete_invoice",
            "finalize_invoice",
            "send_invoice",
            "pay_invoice",
            "void_invoice",
        ],
    )
    def test_invoice_scoped_actions_require_invoice_id(self, config, action):
        assert "invoice_id" in config["actions"][action]["input_schema"]["required"]

    def test_create_invoice_requires_customer(self, config):
        assert "customer" in config["actions"]["create_invoice"]["input_schema"]["required"]

    def test_list_invoices_requires_nothing(self, config):
        assert not config["actions"]["list_invoices"]["input_schema"].get("required")

    def test_auto_advance_is_declared_boolean(self, config):
        props = config["actions"]["create_invoice"]["input_schema"]["properties"]
        assert props["auto_advance"]["type"] == "boolean"

    def test_all_four_transitions_are_registered(self, config):
        for action in ("finalize_invoice", "send_invoice", "pay_invoice", "void_invoice"):
            assert action in config["actions"]
