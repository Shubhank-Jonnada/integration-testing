"""Unit tests for the Stripe invoice item and payment method actions.

Covers the five invoice item actions and the four payment method actions.

Invoice items are where money amounts are set, so the `is not None` guards on
`amount`, `quantity`, and `unit_amount` get particular attention -- a zero
amount is a legitimate line item and must not be silently dropped.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from stripe.stripe import (  # noqa: E402
    API_VERSION,
    STRIPE_API_BASE_URL,
    AttachPaymentMethodAction,
    CreateInvoiceItemAction,
    DeleteInvoiceItemAction,
    DetachPaymentMethodAction,
    GetInvoiceItemAction,
    GetPaymentMethodAction,
    ListInvoiceItemsAction,
    ListPaymentMethodsAction,
    UpdateInvoiceItemAction,
)

pytestmark = pytest.mark.unit

CUSTOMER_ID = "cus_NffrFeUfNV2Hib"
INVOICE_ID = "in_1MtHbELkdIwHu7ixl4OzzPMv"
ITEM_ID = "ii_1MtHbELkdIwHu7ix4Xn2ZQZk"
PM_ID = "pm_1MtHbELkdIwHu7ix0Snn9K8W"
API_ROOT = f"{STRIPE_API_BASE_URL}/{API_VERSION}"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_ITEM = {
    "id": ITEM_ID,
    "object": "invoiceitem",
    "customer": CUSTOMER_ID,
    "amount": 2500,
    "currency": "nzd",
    "description": "Consulting",
}

SAMPLE_PM = {
    "id": PM_ID,
    "object": "payment_method",
    "type": "card",
    "card": {"brand": "visa", "last4": "4242"},
}


@pytest.fixture
def sp_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": "test_token"}}  # nosec B105
    return ctx


# ---- list_invoice_items ----


class TestListInvoiceItems:
    @pytest.mark.asyncio
    async def test_returns_items_and_has_more(self, sp_context):
        sp_context.fetch.return_value = {"data": [SAMPLE_ITEM], "has_more": False}

        result = await ListInvoiceItemsAction().execute({}, sp_context)

        assert result.data["result"] is True
        assert result.data["invoice_items"] == [SAMPLE_ITEM]

    @pytest.mark.asyncio
    async def test_request_url_uses_invoiceitems_without_underscore(self, sp_context):
        """Stripe's path is `/invoiceitems`, not `/invoice_items` -- the action
        name and the endpoint disagree, which is easy to get wrong."""
        sp_context.fetch.return_value = {"data": []}

        await ListInvoiceItemsAction().execute({}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoiceitems"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_customer_and_invoice_filters_forwarded(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListInvoiceItemsAction().execute(
            {"customer": CUSTOMER_ID, "invoice": INVOICE_ID}, sp_context
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["customer"] == CUSTOMER_ID
        assert params["invoice"] == INVOICE_ID

    @pytest.mark.parametrize("value, expected", [(True, "true"), (False, "false")])
    @pytest.mark.asyncio
    async def test_pending_is_sent_as_a_lowercase_string(self, sp_context, value, expected):
        """`pending` is presence-gated, so False is sent -- which is how you ask
        for items already attached to an invoice rather than unbilled ones."""
        sp_context.fetch.return_value = {"data": []}

        await ListInvoiceItemsAction().execute({"pending": value}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"]["pending"] == expected

    @pytest.mark.asyncio
    async def test_pending_omitted_when_absent(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListInvoiceItemsAction().execute({}, sp_context)

        assert "pending" not in sp_context.fetch.call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_pagination_inherited_and_clamped(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListInvoiceItemsAction().execute({"limit": 250}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"]["limit"] == 100

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListInvoiceItemsAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["invoice_items"] == []


# ---- get_invoice_item ----


class TestGetInvoiceItem:
    @pytest.mark.asyncio
    async def test_returns_item(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        result = await GetInvoiceItemAction().execute({"invoice_item_id": ITEM_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["invoice_item"]["amount"] == 2500

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await GetInvoiceItemAction().execute({"invoice_item_id": ITEM_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoiceitems/{ITEM_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await GetInvoiceItemAction().execute({}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 404")

        result = await GetInvoiceItemAction().execute({"invoice_item_id": "ii_x"}, sp_context)

        assert result.data["result"] is False
        assert result.data["invoice_item"] == {}


# ---- create_invoice_item ----


class TestCreateInvoiceItem:
    @pytest.mark.asyncio
    async def test_creates_item(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        result = await CreateInvoiceItemAction().execute({"customer": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["invoice_item"]["id"] == ITEM_ID

    @pytest.mark.asyncio
    async def test_request_url_method_and_form_body(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute({"customer": CUSTOMER_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoiceitems"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["data"] == {"customer": CUSTOMER_ID}

    @pytest.mark.asyncio
    async def test_amount_zero_is_preserved(self, sp_context):
        """`amount` is guarded on `is not None`, so a zero-amount line item -- a
        real thing, e.g. a comped item on an invoice -- reaches Stripe."""
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute(
            {"customer": CUSTOMER_ID, "amount": 0}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["amount"] == "0"

    @pytest.mark.asyncio
    async def test_quantity_zero_is_preserved(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute(
            {"customer": CUSTOMER_ID, "quantity": 0}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["quantity"] == "0"

    @pytest.mark.asyncio
    async def test_unit_amount_maps_to_unit_amount_decimal(self, sp_context):
        """The input is `unit_amount` but Stripe is sent `unit_amount_decimal`,
        which accepts fractional cents."""
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute(
            {"customer": CUSTOMER_ID, "unit_amount": 12.5}, sp_context
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["unit_amount_decimal"] == "12.5"
        assert "unit_amount" not in data

    @pytest.mark.asyncio
    async def test_unit_amount_zero_is_preserved(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute(
            {"customer": CUSTOMER_ID, "unit_amount": 0}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["unit_amount_decimal"] == "0"

    @pytest.mark.asyncio
    async def test_all_fields_forwarded(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute(
            {
                "customer": CUSTOMER_ID,
                "invoice": INVOICE_ID,
                "amount": 2500,
                "currency": "nzd",
                "description": "Consulting",
                "quantity": 2,
                "metadata": {"project": "apollo"},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["invoice"] == INVOICE_ID
        assert data["amount"] == "2500"
        assert data["currency"] == "nzd"
        assert data["description"] == "Consulting"
        assert data["quantity"] == "2"
        assert data["metadata[project]"] == "apollo"

    @pytest.mark.asyncio
    async def test_omitting_invoice_creates_a_pending_item(self, sp_context):
        """With no `invoice` the item sits unbilled until the next invoice is
        created for that customer."""
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute(
            {"customer": CUSTOMER_ID, "amount": 100}, sp_context
        )

        assert "invoice" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_empty_description_is_dropped(self, sp_context):
        """description is truthiness-gated, unlike the numeric fields."""
        sp_context.fetch.return_value = SAMPLE_ITEM

        await CreateInvoiceItemAction().execute(
            {"customer": CUSTOMER_ID, "description": ""}, sp_context
        )

        assert "description" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_missing_customer_is_captured(self, sp_context):
        result = await CreateInvoiceItemAction().execute({}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 400: currency required")

        result = await CreateInvoiceItemAction().execute({"customer": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is False


# ---- update_invoice_item ----


class TestUpdateInvoiceItem:
    @pytest.mark.asyncio
    async def test_request_uses_post_to_item_id(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await UpdateInvoiceItemAction().execute(
            {"invoice_item_id": ITEM_ID, "amount": 3000}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoiceitems/{ITEM_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await UpdateInvoiceItemAction().execute(
            {"invoice_item_id": ITEM_ID, "amount": 3000}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"amount": "3000"}

    @pytest.mark.asyncio
    async def test_amount_zero_survives_on_update(self, sp_context):
        """Zeroing out a line item is a legitimate correction."""
        sp_context.fetch.return_value = SAMPLE_ITEM

        await UpdateInvoiceItemAction().execute(
            {"invoice_item_id": ITEM_ID, "amount": 0}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["amount"] == "0"

    @pytest.mark.asyncio
    async def test_customer_and_currency_are_not_updatable(self, sp_context):
        """Both are accepted on create but omitted from update's field list."""
        sp_context.fetch.return_value = SAMPLE_ITEM

        await UpdateInvoiceItemAction().execute(
            {"invoice_item_id": ITEM_ID, "customer": "cus_other", "currency": "usd"}, sp_context
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert "customer" not in data
        assert "currency" not in data

    @pytest.mark.asyncio
    async def test_invoice_reassignment_is_not_supported(self, sp_context):
        """`invoice` is create-only, so an item cannot be moved to another
        invoice through this action."""
        sp_context.fetch.return_value = SAMPLE_ITEM

        await UpdateInvoiceItemAction().execute(
            {"invoice_item_id": ITEM_ID, "invoice": "in_other"}, sp_context
        )

        assert "invoice" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_id_only_update_sends_empty_form(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_ITEM

        await UpdateInvoiceItemAction().execute({"invoice_item_id": ITEM_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await UpdateInvoiceItemAction().execute({"amount": 100}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- delete_invoice_item ----


class TestDeleteInvoiceItem:
    @pytest.mark.asyncio
    async def test_reports_deletion(self, sp_context):
        sp_context.fetch.return_value = {"id": ITEM_ID, "deleted": True}

        result = await DeleteInvoiceItemAction().execute({"invoice_item_id": ITEM_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["deleted"] is True
        assert result.data["id"] == ITEM_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"deleted": True}

        await DeleteInvoiceItemAction().execute({"invoice_item_id": ITEM_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/invoiceitems/{ITEM_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_deleted_defaults_to_true(self, sp_context):
        sp_context.fetch.return_value = {"id": ITEM_ID}

        result = await DeleteInvoiceItemAction().execute({"invoice_item_id": ITEM_ID}, sp_context)

        assert result.data["deleted"] is True

    @pytest.mark.asyncio
    async def test_error_echoes_requested_id_and_reports_not_deleted(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 400: already attached to a finalized invoice")

        result = await DeleteInvoiceItemAction().execute({"invoice_item_id": ITEM_ID}, sp_context)

        assert result.data["result"] is False
        assert result.data["deleted"] is False
        assert result.data["id"] == ITEM_ID

    @pytest.mark.asyncio
    async def test_missing_id_yields_empty_id(self, sp_context):
        result = await DeleteInvoiceItemAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["id"] == ""


# ---- list_payment_methods ----


class TestListPaymentMethods:
    @pytest.mark.asyncio
    async def test_returns_payment_methods(self, sp_context):
        sp_context.fetch.return_value = {"data": [SAMPLE_PM], "has_more": False}

        result = await ListPaymentMethodsAction().execute({"customer": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["payment_methods"][0]["card"]["last4"] == "4242"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListPaymentMethodsAction().execute({"customer": CUSTOMER_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/payment_methods"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_type_defaults_to_card(self, sp_context):
        """Stripe requires a `type` on this endpoint, so the handler always sends
        one -- defaulting to card rather than erroring."""
        sp_context.fetch.return_value = {"data": []}

        await ListPaymentMethodsAction().execute({"customer": CUSTOMER_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"]["type"] == "card"

    @pytest.mark.asyncio
    async def test_explicit_type_wins(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListPaymentMethodsAction().execute(
            {"customer": CUSTOMER_ID, "type": "sepa_debit"}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["params"]["type"] == "sepa_debit"

    @pytest.mark.asyncio
    async def test_type_is_always_present_even_without_a_customer(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListPaymentMethodsAction().execute({}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"] == {"type": "card"}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListPaymentMethodsAction().execute({"customer": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is False
        assert result.data["payment_methods"] == []


# ---- get_payment_method ----


class TestGetPaymentMethod:
    @pytest.mark.asyncio
    async def test_returns_payment_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PM

        result = await GetPaymentMethodAction().execute({"payment_method_id": PM_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["payment_method"]["type"] == "card"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PM

        await GetPaymentMethodAction().execute({"payment_method_id": PM_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/payment_methods/{PM_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await GetPaymentMethodAction().execute({}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- attach / detach ----


class TestAttachPaymentMethod:
    @pytest.mark.asyncio
    async def test_attaches_to_customer(self, sp_context):
        sp_context.fetch.return_value = {**SAMPLE_PM, "customer": CUSTOMER_ID}

        result = await AttachPaymentMethodAction().execute(
            {"payment_method_id": PM_ID, "customer": CUSTOMER_ID}, sp_context
        )

        assert result.data["result"] is True
        assert result.data["payment_method"]["customer"] == CUSTOMER_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PM

        await AttachPaymentMethodAction().execute(
            {"payment_method_id": PM_ID, "customer": CUSTOMER_ID}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/payment_methods/{PM_ID}/attach"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_customer_is_sent_in_the_form_body(self, sp_context):
        """Attach takes the customer in the body, not the path."""
        sp_context.fetch.return_value = SAMPLE_PM

        await AttachPaymentMethodAction().execute(
            {"payment_method_id": PM_ID, "customer": CUSTOMER_ID}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"customer": CUSTOMER_ID}

    @pytest.mark.parametrize("missing", ["payment_method_id", "customer"])
    @pytest.mark.asyncio
    async def test_both_inputs_are_required(self, sp_context, missing):
        inputs = {"payment_method_id": PM_ID, "customer": CUSTOMER_ID}
        del inputs[missing]

        result = await AttachPaymentMethodAction().execute(inputs, sp_context)

        assert result.data["result"] is False
        assert result.data["payment_method"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception(
            "HTTP 400: previously used payment method cannot be attached"
        )

        result = await AttachPaymentMethodAction().execute(
            {"payment_method_id": PM_ID, "customer": CUSTOMER_ID}, sp_context
        )

        assert result.data["result"] is False
        assert "cannot be attached" in result.data["error"]


class TestDetachPaymentMethod:
    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {**SAMPLE_PM, "customer": None}

        await DetachPaymentMethodAction().execute({"payment_method_id": PM_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/payment_methods/{PM_ID}/detach"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_detach_takes_no_customer(self, sp_context):
        """Detach needs no customer -- the payment method already knows which one
        it is attached to, so no body is sent."""
        sp_context.fetch.return_value = SAMPLE_PM

        await DetachPaymentMethodAction().execute(
            {"payment_method_id": PM_ID, "customer": CUSTOMER_ID}, sp_context
        )

        assert "data" not in sp_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_returns_the_detached_method(self, sp_context):
        sp_context.fetch.return_value = {**SAMPLE_PM, "customer": None}

        result = await DetachPaymentMethodAction().execute({"payment_method_id": PM_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["payment_method"]["customer"] is None

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await DetachPaymentMethodAction().execute({}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 404: No such payment method")

        result = await DetachPaymentMethodAction().execute({"payment_method_id": PM_ID}, sp_context)

        assert result.data["result"] is False


# ---- Config ----


class TestStripeItemAndPmConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action", ["get_invoice_item", "update_invoice_item", "delete_invoice_item"]
    )
    def test_item_actions_require_invoice_item_id(self, config, action):
        assert "invoice_item_id" in config["actions"][action]["input_schema"]["required"]

    def test_create_invoice_item_requires_customer(self, config):
        assert "customer" in config["actions"]["create_invoice_item"]["input_schema"]["required"]

    @pytest.mark.parametrize(
        "action", ["get_payment_method", "attach_payment_method", "detach_payment_method"]
    )
    def test_pm_actions_require_payment_method_id(self, config, action):
        assert "payment_method_id" in config["actions"][action]["input_schema"]["required"]

    def test_attach_also_requires_customer(self, config):
        assert "customer" in config["actions"]["attach_payment_method"]["input_schema"]["required"]

    def test_detach_does_not_require_customer(self, config):
        required = config["actions"]["detach_payment_method"]["input_schema"]["required"]
        assert "customer" not in required
