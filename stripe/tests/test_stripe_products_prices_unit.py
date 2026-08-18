"""Unit tests for the Stripe product and price actions.

Covers the four product actions and the four price actions.

Prices are immutable in Stripe once created -- only a handful of presentational
fields can be changed afterwards -- so `update_price`'s deliberately narrow field
list is a correctness property rather than an omission.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from stripe.stripe import (  # noqa: E402
    API_VERSION,
    STRIPE_API_BASE_URL,
    CreatePriceAction,
    CreateProductAction,
    GetPriceAction,
    GetProductAction,
    ListPricesAction,
    ListProductsAction,
    UpdatePriceAction,
    UpdateProductAction,
)

pytestmark = pytest.mark.unit

PRODUCT_ID = "prod_NWjs8kKbJWmuuc"
PRICE_ID = "price_1MoBy5LkdIwHu7ixZhnattbh"
API_ROOT = f"{STRIPE_API_BASE_URL}/{API_VERSION}"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_PRODUCT = {
    "id": PRODUCT_ID,
    "object": "product",
    "name": "Gold Plan",
    "active": True,
}

SAMPLE_PRICE = {
    "id": PRICE_ID,
    "object": "price",
    "product": PRODUCT_ID,
    "currency": "usd",
    "unit_amount": 2000,
    "recurring": {"interval": "month"},
}


@pytest.fixture
def sp_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": "test_token"}}  # nosec B105
    return ctx


# ---- list_products ----


class TestListProducts:
    @pytest.mark.asyncio
    async def test_returns_products_and_has_more(self, sp_context):
        sp_context.fetch.return_value = {"data": [SAMPLE_PRODUCT], "has_more": False}

        result = await ListProductsAction().execute({}, sp_context)

        assert result.data["result"] is True
        assert result.data["products"] == [SAMPLE_PRODUCT]

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListProductsAction().execute({}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/products"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.parametrize("value, expected", [(True, "true"), (False, "false")])
    @pytest.mark.asyncio
    async def test_active_filter_is_a_lowercase_string(self, sp_context, value, expected):
        """`active` is presence-gated, so `active=False` is a usable filter for
        listing archived products rather than being dropped."""
        sp_context.fetch.return_value = {"data": []}

        await ListProductsAction().execute({"active": value}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"]["active"] == expected

    @pytest.mark.asyncio
    async def test_active_omitted_when_absent(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListProductsAction().execute({}, sp_context)

        assert "active" not in sp_context.fetch.call_args.kwargs["params"]

    @pytest.mark.asyncio
    async def test_type_and_created_filters_forwarded(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListProductsAction().execute(
            {"type": "service", "created_gte": 1704067200, "created_lte": 1735689600}, sp_context
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["type"] == "service"
        assert params["created[gte]"] == 1704067200
        assert params["created[lte]"] == 1735689600

    @pytest.mark.asyncio
    async def test_pagination_inherited_and_clamped(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListProductsAction().execute({"limit": 300}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"]["limit"] == 100

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListProductsAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["products"] == []


# ---- get_product ----


class TestGetProduct:
    @pytest.mark.asyncio
    async def test_returns_product(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        result = await GetProductAction().execute({"product_id": PRODUCT_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["product"]["name"] == "Gold Plan"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await GetProductAction().execute({"product_id": PRODUCT_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/products/{PRODUCT_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await GetProductAction().execute({}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- create_product ----


class TestCreateProduct:
    @pytest.mark.asyncio
    async def test_creates_product(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        result = await CreateProductAction().execute({"name": "Gold Plan"}, sp_context)

        assert result.data["result"] is True
        assert result.data["product"]["id"] == PRODUCT_ID

    @pytest.mark.asyncio
    async def test_request_url_method_and_minimal_body(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await CreateProductAction().execute({"name": "Gold Plan"}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/products"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["data"] == {"name": "Gold Plan"}

    @pytest.mark.asyncio
    async def test_active_false_is_preserved(self, sp_context):
        """Creating an already-archived product is legitimate, and `active` is
        presence-gated so False survives."""
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await CreateProductAction().execute({"name": "Legacy", "active": False}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"]["active"] == "false"

    @pytest.mark.asyncio
    async def test_images_list_is_indexed(self, sp_context):
        """Stripe expects images as `images[0]`, `images[1]`."""
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await CreateProductAction().execute(
            {"name": "Gold", "images": ["https://x/a.png", "https://x/b.png"]}, sp_context
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["images[0]"] == "https://x/a.png"
        assert data["images[1]"] == "https://x/b.png"

    @pytest.mark.asyncio
    async def test_default_price_data_is_flattened(self, sp_context):
        """A nested price can be created inline with the product."""
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await CreateProductAction().execute(
            {
                "name": "Gold",
                "default_price_data": {
                    "currency": "usd",
                    "unit_amount": 2000,
                    "recurring": {"interval": "month"},
                },
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["default_price_data[currency]"] == "usd"
        assert data["default_price_data[unit_amount]"] == "2000"
        assert data["default_price_data[recurring][interval]"] == "month"

    @pytest.mark.asyncio
    async def test_all_scalar_fields_forwarded(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await CreateProductAction().execute(
            {
                "name": "Gold",
                "description": "Best plan",
                "tax_code": "txcd_10000000",
                "unit_label": "seat",
                "url": "https://example.com/gold",
                "metadata": {"tier": "gold"},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["description"] == "Best plan"
        assert data["tax_code"] == "txcd_10000000"
        assert data["unit_label"] == "seat"
        assert data["url"] == "https://example.com/gold"
        assert data["metadata[tier]"] == "gold"

    @pytest.mark.asyncio
    async def test_missing_name_is_captured(self, sp_context):
        result = await CreateProductAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["product"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 400")

        result = await CreateProductAction().execute({"name": "Gold"}, sp_context)

        assert result.data["result"] is False


# ---- update_product ----


class TestUpdateProduct:
    @pytest.mark.asyncio
    async def test_request_uses_post_to_product_id(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await UpdateProductAction().execute(
            {"product_id": PRODUCT_ID, "name": "Gold Plus"}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/products/{PRODUCT_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_archiving_sends_active_false(self, sp_context):
        """Setting active=False is how a product is archived, so this must not be
        dropped by a truthiness gate."""
        sp_context.fetch.return_value = {**SAMPLE_PRODUCT, "active": False}

        await UpdateProductAction().execute({"product_id": PRODUCT_ID, "active": False}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"]["active"] == "false"

    @pytest.mark.asyncio
    async def test_default_price_replaces_default_price_data(self, sp_context):
        """Create takes `default_price_data` (inline creation); update takes
        `default_price` (a reference to an existing price)."""
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await UpdateProductAction().execute(
            {"product_id": PRODUCT_ID, "default_price": PRICE_ID, "default_price_data": {"x": 1}},
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["default_price"] == PRICE_ID
        assert not any(k.startswith("default_price_data") for k in data)

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await UpdateProductAction().execute(
            {"product_id": PRODUCT_ID, "unit_label": "user"}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"unit_label": "user"}

    @pytest.mark.asyncio
    async def test_id_only_update_sends_empty_form(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRODUCT

        await UpdateProductAction().execute({"product_id": PRODUCT_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await UpdateProductAction().execute({"name": "X"}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- list_prices ----


class TestListPrices:
    @pytest.mark.asyncio
    async def test_returns_prices(self, sp_context):
        sp_context.fetch.return_value = {"data": [SAMPLE_PRICE], "has_more": False}

        result = await ListPricesAction().execute({}, sp_context)

        assert result.data["result"] is True
        assert result.data["prices"][0]["unit_amount"] == 2000

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListPricesAction().execute({}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/prices"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_all_filters_forwarded(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListPricesAction().execute(
            {
                "active": True,
                "product": PRODUCT_ID,
                "type": "recurring",
                "currency": "usd",
                "created_gte": 1704067200,
            },
            sp_context,
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["active"] == "true"
        assert params["product"] == PRODUCT_ID
        assert params["type"] == "recurring"
        assert params["currency"] == "usd"
        assert params["created[gte]"] == 1704067200

    @pytest.mark.asyncio
    async def test_active_false_filters_archived_prices(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListPricesAction().execute({"active": False}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"]["active"] == "false"

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListPricesAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["prices"] == []


# ---- get_price ----


class TestGetPrice:
    @pytest.mark.asyncio
    async def test_returns_price(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        result = await GetPriceAction().execute({"price_id": PRICE_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["price"]["recurring"]["interval"] == "month"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        await GetPriceAction().execute({"price_id": PRICE_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/prices/{PRICE_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await GetPriceAction().execute({}, sp_context)

        assert result.data["result"] is False


# ---- create_price ----


class TestCreatePrice:
    @pytest.mark.asyncio
    async def test_creates_price(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        result = await CreatePriceAction().execute(
            {"product": PRODUCT_ID, "unit_amount": 2000}, sp_context
        )

        assert result.data["result"] is True
        assert result.data["price"]["id"] == PRICE_ID

    @pytest.mark.asyncio
    async def test_currency_defaults_to_usd(self, sp_context):
        """Stripe requires a currency, so the handler supplies one rather than
        erroring -- worth knowing, since an unspecified currency silently becomes
        USD rather than the account default."""
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute({"product": PRODUCT_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"]["currency"] == "usd"

    @pytest.mark.asyncio
    async def test_explicit_currency_wins(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {"product": PRODUCT_ID, "currency": "nzd"}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["currency"] == "nzd"

    @pytest.mark.asyncio
    async def test_unit_amount_zero_is_preserved(self, sp_context):
        """A free price is legitimate, and `unit_amount` is presence-gated."""
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {"product": PRODUCT_ID, "unit_amount": 0}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["unit_amount"] == "0"

    @pytest.mark.asyncio
    async def test_unit_amount_takes_precedence_over_decimal(self, sp_context):
        """The two are mutually exclusive in Stripe, so the handler picks one via
        if/elif -- unit_amount wins when both are supplied."""
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {"product": PRODUCT_ID, "unit_amount": 2000, "unit_amount_decimal": "19.99"},
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["unit_amount"] == "2000"
        assert "unit_amount_decimal" not in data

    @pytest.mark.asyncio
    async def test_decimal_used_when_unit_amount_absent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {"product": PRODUCT_ID, "unit_amount_decimal": "19.99"}, sp_context
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["unit_amount_decimal"] == "19.99"
        assert "unit_amount" not in data

    @pytest.mark.asyncio
    async def test_zero_unit_amount_still_blocks_the_decimal_branch(self, sp_context):
        """Because the guard is `is not None`, unit_amount=0 takes the first
        branch -- so a caller passing both gets a free price, not the decimal."""
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {"product": PRODUCT_ID, "unit_amount": 0, "unit_amount_decimal": "19.99"},
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["unit_amount"] == "0"
        assert "unit_amount_decimal" not in data

    @pytest.mark.asyncio
    async def test_recurring_is_flattened(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {
                "product": PRODUCT_ID,
                "unit_amount": 2000,
                "recurring": {"interval": "month", "interval_count": 3},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["recurring[interval]"] == "month"
        assert data["recurring[interval_count]"] == "3"

    @pytest.mark.asyncio
    async def test_tiers_list_of_dicts_is_indexed(self, sp_context):
        """Graduated pricing sends tiers as `tiers[0][up_to]` etc."""
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {
                "product": PRODUCT_ID,
                "billing_scheme": "tiered",
                "tiers_mode": "graduated",
                "tiers": [{"up_to": 10, "unit_amount": 1000}, {"up_to": "inf", "unit_amount": 500}],
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["tiers[0][up_to]"] == "10"
        assert data["tiers[0][unit_amount]"] == "1000"
        assert data["tiers[1][up_to]"] == "inf"
        assert data["billing_scheme"] == "tiered"
        assert data["tiers_mode"] == "graduated"

    @pytest.mark.asyncio
    async def test_nickname_and_tax_behavior_forwarded(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        await CreatePriceAction().execute(
            {
                "product": PRODUCT_ID,
                "unit_amount": 2000,
                "nickname": "Monthly gold",
                "tax_behavior": "exclusive",
                "metadata": {"tier": "gold"},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["nickname"] == "Monthly gold"
        assert data["tax_behavior"] == "exclusive"
        assert data["metadata[tier]"] == "gold"

    @pytest.mark.asyncio
    async def test_missing_product_is_captured(self, sp_context):
        result = await CreatePriceAction().execute({"unit_amount": 2000}, sp_context)

        assert result.data["result"] is False
        assert result.data["price"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 400: No such product")

        result = await CreatePriceAction().execute({"product": "prod_bad"}, sp_context)

        assert result.data["result"] is False


# ---- update_price ----


class TestUpdatePrice:
    @pytest.mark.asyncio
    async def test_request_uses_post_to_price_id(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        await UpdatePriceAction().execute({"price_id": PRICE_ID, "nickname": "New"}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/prices/{PRICE_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_only_four_fields_are_updatable(self, sp_context):
        """Stripe prices are immutable apart from presentational fields, so the
        narrow field list is correct. Amount, currency, product, and recurring
        all require creating a new price instead."""
        sp_context.fetch.return_value = SAMPLE_PRICE

        await UpdatePriceAction().execute(
            {
                "price_id": PRICE_ID,
                "active": True,
                "nickname": "New",
                "tax_behavior": "inclusive",
                "metadata": {"a": "b"},
                "unit_amount": 9999,
                "currency": "nzd",
                "product": "prod_other",
                "recurring": {"interval": "year"},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["active"] == "true"
        assert data["nickname"] == "New"
        assert data["tax_behavior"] == "inclusive"
        assert data["metadata[a]"] == "b"
        assert "unit_amount" not in data
        assert "currency" not in data
        assert "product" not in data
        assert not any(k.startswith("recurring") for k in data)

    @pytest.mark.asyncio
    async def test_archiving_sends_active_false(self, sp_context):
        sp_context.fetch.return_value = {**SAMPLE_PRICE, "active": False}

        await UpdatePriceAction().execute({"price_id": PRICE_ID, "active": False}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"]["active"] == "false"

    @pytest.mark.asyncio
    async def test_id_only_update_sends_empty_form(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_PRICE

        await UpdatePriceAction().execute({"price_id": PRICE_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await UpdatePriceAction().execute({"nickname": "X"}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- Config ----


class TestStripeProductPriceConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize("action", ["get_product", "update_product"])
    def test_product_actions_require_product_id(self, config, action):
        assert "product_id" in config["actions"][action]["input_schema"]["required"]

    def test_create_product_requires_name(self, config):
        assert "name" in config["actions"]["create_product"]["input_schema"]["required"]

    @pytest.mark.parametrize("action", ["get_price", "update_price"])
    def test_price_actions_require_price_id(self, config, action):
        assert "price_id" in config["actions"][action]["input_schema"]["required"]

    def test_create_price_requires_product(self, config):
        assert "product" in config["actions"]["create_price"]["input_schema"]["required"]

    def test_create_price_schema_is_stricter_than_the_handler(self, config):
        """The schema requires `currency` and `unit_amount`, but the handler
        defaults currency to "usd" and treats unit_amount as optional. Since the
        SDK validates inputs against the schema before the handler runs, the
        handler's defaults are unreachable in production -- a request omitting
        either field fails validation rather than picking up the default.

        Asserted as-is to document the mismatch. Either the schema should relax
        or the handler's fallbacks should go; the tests above exercise the
        handler's own logic directly, which is why they still pass."""
        required = config["actions"]["create_price"]["input_schema"]["required"]

        assert "currency" in required
        assert "unit_amount" in required

    def test_no_delete_actions_for_products_or_prices(self, config):
        """Stripe has no delete for either -- archiving via active=False is the
        supported path, so these actions should not exist."""
        assert "delete_product" not in config["actions"]
        assert "delete_price" not in config["actions"]
