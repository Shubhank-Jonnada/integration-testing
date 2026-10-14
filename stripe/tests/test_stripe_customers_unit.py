"""Unit tests for the Stripe form-encoding helpers and customer actions.

Stripe's API is form-encoded rather than JSON, so `build_form_data` has to
flatten arbitrarily nested dicts and lists into Stripe's `key[sub][0]` bracket
notation. That recursion is the highest-risk code in the integration -- every
write action passes its payload through it -- so it gets tested directly and
thoroughly.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from stripe.stripe import (  # noqa: E402
    API_VERSION,
    STRIPE_API_BASE_URL,
    CreateCustomerAction,
    DeleteCustomerAction,
    GetCustomerAction,
    ListCustomersAction,
    UpdateCustomerAction,
    build_form_data,
    build_list_params,
    get_common_headers,
    stripe as stripe_integration,
)

pytestmark = pytest.mark.unit

CUSTOMER_ID = "cus_NffrFeUfNV2Hib"
API_ROOT = f"{STRIPE_API_BASE_URL}/{API_VERSION}"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_CUSTOMER = {
    "id": CUSTOMER_ID,
    "object": "customer",
    "email": "ada@example.com",
    "name": "Ada Lovelace",
}


@pytest.fixture
def sp_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": "test_token"}}  # nosec B105
    return ctx


# ---- get_common_headers ----


class TestGetCommonHeaders:
    def test_content_type_is_form_encoded(self):
        """Stripe rejects JSON bodies on write endpoints."""
        assert get_common_headers()["Content-Type"] == "application/x-www-form-urlencoded"

    def test_pins_an_api_version(self):
        """Pinning avoids silent behaviour changes when Stripe ships a new
        version."""
        assert get_common_headers()["Stripe-Version"] == "2025-12-15.preview"

    def test_sets_no_authorization_header(self):
        """Platform OAuth means the SDK injects auth; a manual header here would
        override the platform token."""
        assert "Authorization" not in get_common_headers()

    def test_returns_a_fresh_dict(self):
        headers = get_common_headers()
        headers["X-Injected"] = "1"

        assert "X-Injected" not in get_common_headers()


# ---- build_form_data ----


class TestBuildFormData:
    def test_flat_values_are_stringified(self):
        assert build_form_data({"name": "Ada", "count": 3}) == {"name": "Ada", "count": "3"}

    def test_nested_dict_uses_bracket_notation(self):
        result = build_form_data({"address": {"city": "Wellington", "country": "NZ"}})

        assert result == {"address[city]": "Wellington", "address[country]": "NZ"}

    def test_deeply_nested_dicts_recurse(self):
        result = build_form_data({"a": {"b": {"c": "d"}}})

        assert result == {"a[b][c]": "d"}

    def test_list_of_scalars_is_indexed(self):
        result = build_form_data({"tags": ["a", "b"]})

        assert result == {"tags[0]": "a", "tags[1]": "b"}

    def test_list_of_dicts_is_indexed_and_recursed(self):
        """This is the shape subscription items and invoice line items use."""
        result = build_form_data({"items": [{"price": "p1", "quantity": 2}, {"price": "p2"}]})

        assert result == {
            "items[0][price]": "p1",
            "items[0][quantity]": "2",
            "items[1][price]": "p2",
        }

    def test_booleans_become_lowercase_strings(self):
        """Python's str(True) is "True", which Stripe rejects."""
        result = build_form_data({"active": True, "livemode": False})

        assert result == {"active": "true", "livemode": "false"}

    def test_nested_booleans_are_also_converted(self):
        result = build_form_data({"settings": {"enabled": True}})

        assert result == {"settings[enabled]": "true"}

    def test_none_values_are_dropped(self):
        result = build_form_data({"name": "Ada", "phone": None})

        assert result == {"name": "Ada"}

    def test_zero_and_empty_string_are_preserved(self):
        """The guard is `is not None`, so falsy-but-meaningful values survive --
        important because amount=0 and description="" are both valid to Stripe."""
        result = build_form_data({"amount": 0, "description": ""})

        assert result == {"amount": "0", "description": ""}

    def test_empty_dict_yields_empty_result(self):
        assert build_form_data({}) == {}

    def test_empty_nested_dict_contributes_nothing(self):
        assert build_form_data({"metadata": {}}) == {}

    def test_empty_list_contributes_nothing(self):
        assert build_form_data({"tags": []}) == {}

    def test_none_inside_a_list_is_stringified_not_dropped(self):
        """The None guard only applies to dict values, so a None inside a list
        becomes the literal string "None". Documented as a real edge-case gap."""
        result = build_form_data({"tags": ["a", None]})

        assert result == {"tags[0]": "a", "tags[1]": "None"}

    def test_prefix_argument_wraps_top_level_keys(self):
        result = build_form_data({"city": "Wellington"}, prefix="address")

        assert result == {"address[city]": "Wellington"}

    def test_float_amounts_are_stringified(self):
        assert build_form_data({"unit_amount_decimal": 12.5}) == {"unit_amount_decimal": "12.5"}

    def test_metadata_keys_are_not_escaped(self):
        """A metadata key containing a bracket is emitted verbatim, producing an
        ambiguous form key. No current caller does this, but it isn't guarded."""
        result = build_form_data({"metadata": {"a[b]": "c"}})

        assert result == {"metadata[a[b]]": "c"}


# ---- build_list_params ----


class TestBuildListParams:
    def test_empty_inputs_yield_no_params(self):
        assert build_list_params({}) == {}

    def test_limit_forwarded(self):
        assert build_list_params({"limit": 25}) == {"limit": 25}

    def test_limit_is_clamped_to_stripe_maximum(self):
        """Stripe rejects limit > 100, so the helper clamps rather than erroring."""
        assert build_list_params({"limit": 500}) == {"limit": 100}

    def test_limit_at_the_boundary_is_unchanged(self):
        assert build_list_params({"limit": 100}) == {"limit": 100}

    def test_limit_zero_is_dropped(self):
        """Truthiness-gated, so limit=0 is omitted rather than sent."""
        assert build_list_params({"limit": 0}) == {}

    def test_cursors_forwarded(self):
        params = build_list_params({"starting_after": "cus_a", "ending_before": "cus_b"})

        assert params == {"starting_after": "cus_a", "ending_before": "cus_b"}

    def test_unknown_inputs_are_ignored(self):
        """Only the three pagination keys are read; everything else is the
        caller's job to add."""
        assert build_list_params({"email": "a@b.com", "limit": 5}) == {"limit": 5}


# ---- list_customers ----


class TestListCustomers:
    @pytest.mark.asyncio
    async def test_returns_customers_and_has_more(self, sp_context):
        sp_context.fetch.return_value = {"data": [SAMPLE_CUSTOMER], "has_more": True}

        result = await ListCustomersAction().execute({}, sp_context)

        assert result.data["result"] is True
        assert result.data["customers"] == [SAMPLE_CUSTOMER]
        assert result.data["has_more"] is True

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListCustomersAction().execute({}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/customers"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_pagination_and_email_filter_forwarded(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListCustomersAction().execute(
            {"limit": 10, "starting_after": "cus_a", "email": "ada@example.com"}, sp_context
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["limit"] == 10
        assert params["starting_after"] == "cus_a"
        assert params["email"] == "ada@example.com"

    @pytest.mark.asyncio
    async def test_created_filters_use_stripe_bracket_syntax(self, sp_context):
        """`created_gte` maps to Stripe's `created[gte]` range operator."""
        sp_context.fetch.return_value = {"data": []}

        await ListCustomersAction().execute(
            {"created_gte": 1704067200, "created_lte": 1735689600}, sp_context
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["created[gte]"] == 1704067200
        assert params["created[lte]"] == 1735689600
        assert "created_gte" not in params

    @pytest.mark.asyncio
    async def test_has_more_defaults_to_false(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        result = await ListCustomersAction().execute({}, sp_context)

        assert result.data["has_more"] is False

    @pytest.mark.asyncio
    async def test_missing_data_key_yields_empty_list(self, sp_context):
        sp_context.fetch.return_value = {}

        result = await ListCustomersAction().execute({}, sp_context)

        assert result.data["customers"] == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 401: Invalid API Key")

        result = await ListCustomersAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["customers"] == []
        assert result.data["has_more"] is False
        assert "Invalid API Key" in result.data["error"]


# ---- get_customer ----


class TestGetCustomer:
    @pytest.mark.asyncio
    async def test_returns_customer(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        result = await GetCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["customer"]["email"] == "ada@example.com"

    @pytest.mark.asyncio
    async def test_request_url_includes_customer_id(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await GetCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/customers/{CUSTOMER_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_sends_no_params_or_body(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await GetCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        kwargs = sp_context.fetch.call_args.kwargs
        assert "params" not in kwargs
        assert "data" not in kwargs

    @pytest.mark.asyncio
    async def test_missing_customer_id_is_captured(self, sp_context):
        result = await GetCustomerAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["customer"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 404: No such customer")

        result = await GetCustomerAction().execute({"customer_id": "cus_missing"}, sp_context)

        assert result.data["result"] is False
        assert "No such customer" in result.data["error"]


# ---- create_customer ----


class TestCreateCustomer:
    @pytest.mark.asyncio
    async def test_creates_customer(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        result = await CreateCustomerAction().execute({"email": "ada@example.com"}, sp_context)

        assert result.data["result"] is True
        assert result.data["customer"]["id"] == CUSTOMER_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await CreateCustomerAction().execute({"email": "ada@example.com"}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/customers"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_body_is_sent_as_form_data_not_json(self, sp_context):
        """Stripe requires form encoding; a `json` kwarg would be rejected."""
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await CreateCustomerAction().execute({"email": "ada@example.com"}, sp_context)

        kwargs = sp_context.fetch.call_args.kwargs
        assert kwargs["data"] == {"email": "ada@example.com"}
        assert "json" not in kwargs

    @pytest.mark.asyncio
    async def test_all_optional_fields_forwarded(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await CreateCustomerAction().execute(
            {
                "email": "ada@example.com",
                "name": "Ada",
                "description": "VIP",
                "phone": "+64211234567",
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["email"] == "ada@example.com"
        assert data["name"] == "Ada"
        assert data["description"] == "VIP"
        assert data["phone"] == "+64211234567"

    @pytest.mark.asyncio
    async def test_nested_address_is_flattened(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await CreateCustomerAction().execute(
            {"email": "a@b.com", "address": {"city": "Wellington", "country": "NZ"}}, sp_context
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["address[city]"] == "Wellington"
        assert data["address[country]"] == "NZ"
        assert "address" not in data

    @pytest.mark.asyncio
    async def test_metadata_is_flattened(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await CreateCustomerAction().execute(
            {"email": "a@b.com", "metadata": {"plan": "gold", "seats": 5}}, sp_context
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["metadata[plan]"] == "gold"
        assert data["metadata[seats]"] == "5"

    @pytest.mark.asyncio
    async def test_no_inputs_sends_empty_form(self, sp_context):
        """Stripe allows creating a bare customer with no attributes."""
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await CreateCustomerAction().execute({}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_empty_values_are_dropped(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await CreateCustomerAction().execute(
            {"email": "a@b.com", "name": "", "metadata": {}}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"email": "a@b.com"}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 400: Invalid email address")

        result = await CreateCustomerAction().execute({"email": "bad"}, sp_context)

        assert result.data["result"] is False
        assert result.data["customer"] == {}


# ---- update_customer ----


class TestUpdateCustomer:
    @pytest.mark.asyncio
    async def test_request_uses_post_to_customer_id(self, sp_context):
        """Stripe updates via POST, not PUT or PATCH."""
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await UpdateCustomerAction().execute(
            {"customer_id": CUSTOMER_ID, "name": "Ada L"}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/customers/{CUSTOMER_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await UpdateCustomerAction().execute(
            {"customer_id": CUSTOMER_ID, "phone": "+64211234567"}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"phone": "+64211234567"}

    @pytest.mark.asyncio
    async def test_customer_id_not_duplicated_into_body(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await UpdateCustomerAction().execute({"customer_id": CUSTOMER_ID, "name": "X"}, sp_context)

        assert "customer_id" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_id_only_update_sends_empty_form(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await UpdateCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_empty_values_cannot_clear_a_field(self, sp_context):
        """Truthiness gating means "" can't blank a description -- even though
        Stripe accepts an empty string for exactly that purpose."""
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await UpdateCustomerAction().execute(
            {"customer_id": CUSTOMER_ID, "description": ""}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_metadata_replacement_is_flattened(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_CUSTOMER

        await UpdateCustomerAction().execute(
            {"customer_id": CUSTOMER_ID, "metadata": {"tier": "platinum"}}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["metadata[tier]"] == "platinum"

    @pytest.mark.asyncio
    async def test_missing_customer_id_is_captured(self, sp_context):
        result = await UpdateCustomerAction().execute({"name": "X"}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 404")

        result = await UpdateCustomerAction().execute(
            {"customer_id": CUSTOMER_ID, "name": "X"}, sp_context
        )

        assert result.data["result"] is False


# ---- delete_customer ----


class TestDeleteCustomer:
    @pytest.mark.asyncio
    async def test_reports_deletion(self, sp_context):
        sp_context.fetch.return_value = {"id": CUSTOMER_ID, "object": "customer", "deleted": True}

        result = await DeleteCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["deleted"] is True
        assert result.data["id"] == CUSTOMER_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"deleted": True}

        await DeleteCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/customers/{CUSTOMER_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_id_falls_back_to_the_input(self, sp_context):
        """If Stripe omits the id, echo back what was requested so the caller can
        still correlate the response."""
        sp_context.fetch.return_value = {"deleted": True}

        result = await DeleteCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        assert result.data["id"] == CUSTOMER_ID

    @pytest.mark.asyncio
    async def test_deleted_defaults_to_true_on_success(self, sp_context):
        """A 2xx with no `deleted` flag is treated as a successful deletion."""
        sp_context.fetch.return_value = {"id": CUSTOMER_ID}

        result = await DeleteCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        assert result.data["deleted"] is True

    @pytest.mark.asyncio
    async def test_stripe_reporting_not_deleted_is_respected(self, sp_context):
        sp_context.fetch.return_value = {"id": CUSTOMER_ID, "deleted": False}

        result = await DeleteCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        assert result.data["deleted"] is False
        assert result.data["result"] is True

    @pytest.mark.asyncio
    async def test_error_reports_not_deleted_and_echoes_id(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 404: No such customer")

        result = await DeleteCustomerAction().execute({"customer_id": CUSTOMER_ID}, sp_context)

        assert result.data["result"] is False
        assert result.data["deleted"] is False
        assert result.data["id"] == CUSTOMER_ID

    @pytest.mark.asyncio
    async def test_missing_customer_id_yields_empty_id(self, sp_context):
        result = await DeleteCustomerAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["id"] == ""


# ---- Config ----


class TestStripeCustomerConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(stripe_integration._action_handlers.keys())

        assert defined == registered

    @pytest.mark.parametrize("action", ["get_customer", "update_customer", "delete_customer"])
    def test_id_scoped_actions_require_customer_id(self, config, action):
        assert "customer_id" in config["actions"][action]["input_schema"]["required"]

    @pytest.mark.parametrize("action", ["list_customers", "create_customer"])
    def test_collection_actions_require_nothing(self, config, action):
        """Stripe allows both an unfiltered list and a bare customer create."""
        assert not config["actions"][action]["input_schema"].get("required")

    def test_list_customers_exposes_cursor_pagination(self, config):
        props = config["actions"]["list_customers"]["input_schema"]["properties"]

        assert "starting_after" in props
        assert "ending_before" in props

    def test_list_limit_declares_stripe_maximum(self, config):
        """The schema cap should match the clamp in build_list_params."""
        limit = config["actions"]["list_customers"]["input_schema"]["properties"]["limit"]
        assert limit["maximum"] == 100
