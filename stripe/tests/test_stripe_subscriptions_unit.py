"""Unit tests for the Stripe subscription actions.

Covers `list_subscriptions`, `get_subscription`, `create_subscription`,
`update_subscription`, and `cancel_subscription`.

`cancel_subscription` is the most consequential handler in the integration: it
branches between "cancel at period end" (a POST that keeps the subscription
alive until the term expires) and "cancel immediately" (a DELETE that stops
billing now). Choosing the wrong branch either keeps charging a customer who
asked to leave, or cuts off service that was paid for.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from stripe.stripe import (  # noqa: E402
    API_VERSION,
    STRIPE_API_BASE_URL,
    CancelSubscriptionAction,
    CreateSubscriptionAction,
    GetSubscriptionAction,
    ListSubscriptionsAction,
    UpdateSubscriptionAction,
)

pytestmark = pytest.mark.unit

CUSTOMER_ID = "cus_NffrFeUfNV2Hib"
SUB_ID = "sub_1MowQVLkdIwHu7ixeRlqHVzs"
PRICE_ID = "price_1MoBy5LkdIwHu7ixZhnattbh"
API_ROOT = f"{STRIPE_API_BASE_URL}/{API_VERSION}"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

ITEMS = [{"price": PRICE_ID, "quantity": 2}]

SAMPLE_SUB = {
    "id": SUB_ID,
    "object": "subscription",
    "customer": CUSTOMER_ID,
    "status": "active",
    "cancel_at_period_end": False,
}


@pytest.fixture
def sp_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": "test_token"}}  # nosec B105
    return ctx


# ---- list_subscriptions ----


class TestListSubscriptions:
    @pytest.mark.asyncio
    async def test_returns_subscriptions_and_has_more(self, sp_context):
        sp_context.fetch.return_value = {"data": [SAMPLE_SUB], "has_more": False}

        result = await ListSubscriptionsAction().execute({}, sp_context)

        assert result.data["result"] is True
        assert result.data["subscriptions"] == [SAMPLE_SUB]

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListSubscriptionsAction().execute({}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/subscriptions"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_customer_price_and_status_filters_forwarded(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListSubscriptionsAction().execute(
            {"customer": CUSTOMER_ID, "price": PRICE_ID, "status": "active"}, sp_context
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["customer"] == CUSTOMER_ID
        assert params["price"] == PRICE_ID
        assert params["status"] == "active"

    @pytest.mark.asyncio
    async def test_created_and_period_filters_use_bracket_syntax(self, sp_context):
        """Subscriptions support a second range filter on
        `current_period_start` in addition to `created`."""
        sp_context.fetch.return_value = {"data": []}

        await ListSubscriptionsAction().execute(
            {
                "created_gte": 1704067200,
                "created_lte": 1735689600,
                "current_period_start_gte": 1704067200,
                "current_period_start_lte": 1735689600,
            },
            sp_context,
        )

        params = sp_context.fetch.call_args.kwargs["params"]
        assert params["created[gte]"] == 1704067200
        assert params["created[lte]"] == 1735689600
        assert params["current_period_start[gte]"] == 1704067200
        assert params["current_period_start[lte]"] == 1735689600
        assert "current_period_start_gte" not in params

    @pytest.mark.asyncio
    async def test_pagination_inherited_and_clamped(self, sp_context):
        sp_context.fetch.return_value = {"data": []}

        await ListSubscriptionsAction().execute({"limit": 400}, sp_context)

        assert sp_context.fetch.call_args.kwargs["params"]["limit"] == 100

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListSubscriptionsAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["subscriptions"] == []


# ---- get_subscription ----


class TestGetSubscription:
    @pytest.mark.asyncio
    async def test_returns_subscription(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        result = await GetSubscriptionAction().execute({"subscription_id": SUB_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["subscription"]["status"] == "active"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        await GetSubscriptionAction().execute({"subscription_id": SUB_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/subscriptions/{SUB_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await GetSubscriptionAction().execute({}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- create_subscription ----


class TestCreateSubscription:
    @pytest.mark.asyncio
    async def test_creates_subscription(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        result = await CreateSubscriptionAction().execute(
            {"customer": CUSTOMER_ID, "items": ITEMS}, sp_context
        )

        assert result.data["result"] is True
        assert result.data["subscription"]["id"] == SUB_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        await CreateSubscriptionAction().execute(
            {"customer": CUSTOMER_ID, "items": ITEMS}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/subscriptions"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_items_are_flattened_to_indexed_form_keys(self, sp_context):
        """This is the shape build_form_data's list-of-dicts branch exists for."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CreateSubscriptionAction().execute(
            {"customer": CUSTOMER_ID, "items": [{"price": "p1", "quantity": 2}, {"price": "p2"}]},
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["items[0][price]"] == "p1"
        assert data["items[0][quantity]"] == "2"
        assert data["items[1][price]"] == "p2"

    @pytest.mark.asyncio
    async def test_cancel_at_period_end_false_is_sent(self, sp_context):
        """Presence-gated, so an explicit False reaches Stripe."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CreateSubscriptionAction().execute(
            {"customer": CUSTOMER_ID, "items": ITEMS, "cancel_at_period_end": False}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["cancel_at_period_end"] == "false"

    @pytest.mark.asyncio
    async def test_trial_and_billing_fields_forwarded(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        await CreateSubscriptionAction().execute(
            {
                "customer": CUSTOMER_ID,
                "items": ITEMS,
                "trial_period_days": 14,
                "trial_end": 1735689600,
                "billing_cycle_anchor": 1735689600,
                "payment_behavior": "default_incomplete",
                "proration_behavior": "create_prorations",
                "collection_method": "charge_automatically",
                "default_payment_method": "pm_card_visa",
                "metadata": {"plan": "gold"},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["trial_period_days"] == "14"
        assert data["trial_end"] == "1735689600"
        assert data["billing_cycle_anchor"] == "1735689600"
        assert data["payment_behavior"] == "default_incomplete"
        assert data["proration_behavior"] == "create_prorations"
        assert data["collection_method"] == "charge_automatically"
        assert data["default_payment_method"] == "pm_card_visa"
        assert data["metadata[plan]"] == "gold"

    @pytest.mark.asyncio
    async def test_trial_period_days_zero_is_dropped(self, sp_context):
        """Truthiness-gated, so "no trial" can't be expressed as 0 -- omitting the
        field entirely has the same effect, so this is harmless but inconsistent
        with cancel_at_period_end above."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CreateSubscriptionAction().execute(
            {"customer": CUSTOMER_ID, "items": ITEMS, "trial_period_days": 0}, sp_context
        )

        assert "trial_period_days" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_empty_items_list_is_dropped(self, sp_context):
        """items is truthiness-gated, so an empty list produces a body with only
        the customer -- which Stripe rejects, since a subscription needs items."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CreateSubscriptionAction().execute(
            {"customer": CUSTOMER_ID, "items": []}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"customer": CUSTOMER_ID}

    @pytest.mark.asyncio
    async def test_missing_customer_is_captured(self, sp_context):
        result = await CreateSubscriptionAction().execute({"items": ITEMS}, sp_context)

        assert result.data["result"] is False
        assert result.data["subscription"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 400: no payment method")

        result = await CreateSubscriptionAction().execute(
            {"customer": CUSTOMER_ID, "items": ITEMS}, sp_context
        )

        assert result.data["result"] is False
        assert "no payment method" in result.data["error"]


# ---- update_subscription ----


class TestUpdateSubscription:
    @pytest.mark.asyncio
    async def test_request_uses_post_to_subscription_id(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "items": ITEMS}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/subscriptions/{SUB_ID}"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "proration_behavior": "none"}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"] == {"proration_behavior": "none"}

    @pytest.mark.asyncio
    async def test_customer_is_not_updatable(self, sp_context):
        """A subscription cannot be moved between customers."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "customer": "cus_other"}, sp_context
        )

        assert "customer" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_billing_cycle_anchor_is_create_only(self, sp_context):
        """Accepted on create but omitted from update's field list."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "billing_cycle_anchor": 1735689600}, sp_context
        )

        assert "billing_cycle_anchor" not in sp_context.fetch.call_args.kwargs["data"]

    @pytest.mark.asyncio
    async def test_trial_period_days_is_create_only(self, sp_context):
        """Only `trial_end` can adjust a trial after the fact."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "trial_period_days": 7, "trial_end": 1735689600},
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert "trial_period_days" not in data
        assert data["trial_end"] == "1735689600"

    @pytest.mark.parametrize("value, expected", [(True, "true"), (False, "false")])
    @pytest.mark.asyncio
    async def test_cancel_at_period_end_both_ways(self, sp_context, value, expected):
        """False is how a scheduled cancellation is reversed, so it must survive."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "cancel_at_period_end": value}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["cancel_at_period_end"] == expected

    @pytest.mark.asyncio
    async def test_items_replace_wholesale(self, sp_context):
        """Stripe replaces the item set, so a partial list removes the rest."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "items": [{"price": "p_new"}]}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["items[0][price]"] == "p_new"

    @pytest.mark.asyncio
    async def test_id_only_update_sends_empty_form(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        await UpdateSubscriptionAction().execute({"subscription_id": SUB_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] == {}

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await UpdateSubscriptionAction().execute({"items": ITEMS}, sp_context)

        assert result.data["result"] is False
        sp_context.fetch.assert_not_called()


# ---- cancel_subscription ----


class TestCancelSubscription:
    @pytest.mark.asyncio
    async def test_default_is_immediate_cancellation_via_delete(self, sp_context):
        """`cancel_at_period_end` defaults to False, so an unqualified cancel
        stops billing immediately rather than at the end of the term."""
        sp_context.fetch.return_value = {**SAMPLE_SUB, "status": "canceled"}

        await CancelSubscriptionAction().execute({"subscription_id": SUB_ID}, sp_context)

        call = sp_context.fetch.call_args
        assert call.args[0] == f"{API_ROOT}/subscriptions/{SUB_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_period_end_cancellation_uses_post_not_delete(self, sp_context):
        """Scheduling a cancellation is an update, not a deletion -- the
        subscription stays active until the term expires."""
        sp_context.fetch.return_value = {**SAMPLE_SUB, "cancel_at_period_end": True}

        await CancelSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "cancel_at_period_end": True}, sp_context
        )

        call = sp_context.fetch.call_args
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["data"] == {"cancel_at_period_end": "true"}

    @pytest.mark.asyncio
    async def test_period_end_branch_ignores_the_immediate_options(self, sp_context):
        """invoice_now and prorate only apply to immediate cancellation, so they
        are silently dropped on the scheduled path."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CancelSubscriptionAction().execute(
            {
                "subscription_id": SUB_ID,
                "cancel_at_period_end": True,
                "invoice_now": True,
                "prorate": True,
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data == {"cancel_at_period_end": "true"}

    @pytest.mark.asyncio
    async def test_immediate_cancellation_sends_no_body_by_default(self, sp_context):
        """With no options the DELETE carries `data=None` rather than an empty
        dict."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CancelSubscriptionAction().execute({"subscription_id": SUB_ID}, sp_context)

        assert sp_context.fetch.call_args.kwargs["data"] is None

    @pytest.mark.parametrize("value, expected", [(True, "true"), (False, "false")])
    @pytest.mark.asyncio
    async def test_invoice_now_both_ways(self, sp_context, value, expected):
        """invoice_now controls whether Stripe bills for unbilled usage on the
        way out, so False must reach the API rather than being dropped."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CancelSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "invoice_now": value}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["invoice_now"] == expected

    @pytest.mark.parametrize("value, expected", [(True, "true"), (False, "false")])
    @pytest.mark.asyncio
    async def test_prorate_both_ways(self, sp_context, value, expected):
        """prorate decides whether the customer is credited for the unused part
        of the period -- a dropped False would refund money unintentionally."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CancelSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "prorate": value}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["data"]["prorate"] == expected

    @pytest.mark.asyncio
    async def test_cancellation_details_are_flattened(self, sp_context):
        sp_context.fetch.return_value = SAMPLE_SUB

        await CancelSubscriptionAction().execute(
            {
                "subscription_id": SUB_ID,
                "cancellation_details": {"comment": "Too expensive", "feedback": "too_expensive"},
            },
            sp_context,
        )

        data = sp_context.fetch.call_args.kwargs["data"]
        assert data["cancellation_details[comment]"] == "Too expensive"
        assert data["cancellation_details[feedback]"] == "too_expensive"

    @pytest.mark.asyncio
    async def test_falsy_cancel_at_period_end_takes_the_immediate_branch(self, sp_context):
        """The branch guard is plain truthiness on `inputs.get(..., False)`, so
        an explicit False behaves the same as omission."""
        sp_context.fetch.return_value = SAMPLE_SUB

        await CancelSubscriptionAction().execute(
            {"subscription_id": SUB_ID, "cancel_at_period_end": False}, sp_context
        )

        assert sp_context.fetch.call_args.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_returns_the_cancelled_subscription(self, sp_context):
        sp_context.fetch.return_value = {**SAMPLE_SUB, "status": "canceled"}

        result = await CancelSubscriptionAction().execute({"subscription_id": SUB_ID}, sp_context)

        assert result.data["result"] is True
        assert result.data["subscription"]["status"] == "canceled"

    @pytest.mark.asyncio
    async def test_missing_id_is_captured(self, sp_context):
        result = await CancelSubscriptionAction().execute({}, sp_context)

        assert result.data["result"] is False
        assert result.data["subscription"] == {}
        sp_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sp_context):
        sp_context.fetch.side_effect = Exception("HTTP 404: No such subscription")

        result = await CancelSubscriptionAction().execute({"subscription_id": SUB_ID}, sp_context)

        assert result.data["result"] is False
        assert "No such subscription" in result.data["error"]


# ---- Config ----


class TestStripeSubscriptionConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action", ["get_subscription", "update_subscription", "cancel_subscription"]
    )
    def test_id_scoped_actions_require_subscription_id(self, config, action):
        assert "subscription_id" in config["actions"][action]["input_schema"]["required"]

    def test_create_subscription_requires_customer(self, config):
        assert "customer" in config["actions"]["create_subscription"]["input_schema"]["required"]

    def test_cancel_at_period_end_is_declared_boolean(self, config):
        props = config["actions"]["cancel_subscription"]["input_schema"]["properties"]
        assert props["cancel_at_period_end"]["type"] == "boolean"

    def test_list_subscriptions_exposes_both_range_filters(self, config):
        props = config["actions"]["list_subscriptions"]["input_schema"]["properties"]

        assert "created_gte" in props
        assert "current_period_start_gte" in props
