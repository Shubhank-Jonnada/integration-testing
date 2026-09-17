"""Unit tests for the Freshdesk contact actions.

Covers `create_contact`, `list_contacts`, `get_contact`, `update_contact`,
`delete_contact`, and `search_contacts`.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from freshdesk.freshdesk import (  # noqa: E402
    CreateContactAction,
    DeleteContactAction,
    GetContactAction,
    ListContactsAction,
    SearchContactsAction,
    UpdateContactAction,
)

pytestmark = pytest.mark.unit

API_KEY = "test_freshdesk_api_key"  # nosec B105
DOMAIN = "testcompany"
BASE_URL = f"https://{DOMAIN}.freshdesk.com/api/v2"
CONTACT_ID = 501
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_CONTACT = {
    "id": CONTACT_ID,
    "name": "John Jonz",
    "email": "john@example.com",
    "phone": "+15551234",
    "company_id": 42,
}


@pytest.fixture
def fd_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"api_key": API_KEY, "domain": DOMAIN}}
    return ctx


def minimal_contact(**overrides):
    inputs = {"name": "John Jonz", "email": "john@example.com"}
    inputs.update(overrides)
    return inputs


# ---- create_contact ----


class TestCreateContact:
    @pytest.mark.asyncio
    async def test_creates_contact(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_CONTACT

        result = await CreateContactAction().execute(minimal_contact(), fd_context)

        assert result["result"] is True
        assert result["contact"]["id"] == CONTACT_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateContactAction().execute(minimal_contact(), fd_context)

        assert fd_context.fetch.call_args.args[0] == f"{BASE_URL}/contacts"
        assert fd_context.fetch.call_args.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_minimal_body_is_name_and_email(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateContactAction().execute(minimal_contact(), fd_context)

        assert fd_context.fetch.call_args.kwargs["json"] == {
            "name": "John Jonz",
            "email": "john@example.com",
        }

    @pytest.mark.asyncio
    async def test_all_optional_fields_forwarded(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateContactAction().execute(
            minimal_contact(
                phone="+15551234",
                mobile="+15559876",
                company_id=42,
                job_title="Engineer",
                description="A contact",
                tags=["vip"],
            ),
            fd_context,
        )

        body = fd_context.fetch.call_args.kwargs["json"]
        assert body["phone"] == "+15551234"
        assert body["mobile"] == "+15559876"
        assert body["company_id"] == 42
        assert body["job_title"] == "Engineer"
        assert body["description"] == "A contact"
        assert body["tags"] == ["vip"]

    @pytest.mark.asyncio
    async def test_company_id_zero_survives(self, fd_context):
        """company_id is presence-gated while every other optional here is
        truthiness-gated, so 0 reaches the API."""
        fd_context.fetch.return_value = {}

        await CreateContactAction().execute(minimal_contact(company_id=0), fd_context)

        assert fd_context.fetch.call_args.kwargs["json"]["company_id"] == 0

    @pytest.mark.asyncio
    async def test_empty_text_fields_are_dropped(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateContactAction().execute(
            minimal_contact(phone="", mobile="", job_title="", description="", tags=[]),
            fd_context,
        )

        assert fd_context.fetch.call_args.kwargs["json"] == {
            "name": "John Jonz",
            "email": "john@example.com",
        }

    @pytest.mark.parametrize("missing", ["name", "email"])
    @pytest.mark.asyncio
    async def test_both_required_inputs_enforced(self, fd_context, missing):
        inputs = minimal_contact()
        del inputs[missing]

        result = await CreateContactAction().execute(inputs, fd_context)

        assert result["result"] is False
        assert result["contact"] == {}
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 409: email already exists")

        result = await CreateContactAction().execute(minimal_contact(), fd_context)

        assert result["result"] is False
        assert "already exists" in result["error"]


# ---- list_contacts ----


class TestListContacts:
    @pytest.mark.asyncio
    async def test_returns_contacts_and_total(self, fd_context):
        fd_context.fetch.return_value = [SAMPLE_CONTACT]

        result = await ListContactsAction().execute({}, fd_context)

        assert result["result"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_request_url_and_default_pagination(self, fd_context):
        fd_context.fetch.return_value = []

        await ListContactsAction().execute({}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/contacts"
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["params"] == {"page": 1, "per_page": 30}

    @pytest.mark.asyncio
    async def test_explicit_pagination_forwarded(self, fd_context):
        fd_context.fetch.return_value = []

        await ListContactsAction().execute({"page": 4, "per_page": 100}, fd_context)

        assert fd_context.fetch.call_args.kwargs["params"] == {"page": 4, "per_page": 100}

    @pytest.mark.asyncio
    async def test_total_is_page_scoped(self, fd_context):
        """`total` is len() of the returned page, so it caps at per_page."""
        fd_context.fetch.return_value = [SAMPLE_CONTACT] * 30

        result = await ListContactsAction().execute({}, fd_context)

        assert result["total"] == 30

    @pytest.mark.asyncio
    async def test_non_list_response_yields_empty(self, fd_context):
        fd_context.fetch.return_value = {"message": "unexpected"}

        result = await ListContactsAction().execute({}, fd_context)

        assert result["contacts"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 401")

        result = await ListContactsAction().execute({}, fd_context)

        assert result["result"] is False
        assert result["contacts"] == []


# ---- get_contact ----


class TestGetContact:
    @pytest.mark.asyncio
    async def test_returns_contact(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_CONTACT

        result = await GetContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        assert result["result"] is True
        assert result["contact"]["name"] == "John Jonz"

    @pytest.mark.asyncio
    async def test_request_url_includes_contact_id(self, fd_context):
        fd_context.fetch.return_value = {}

        await GetContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        assert fd_context.fetch.call_args.args[0] == f"{BASE_URL}/contacts/{CONTACT_ID}"
        assert fd_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_sends_no_params_or_body(self, fd_context):
        fd_context.fetch.return_value = {}

        await GetContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        kwargs = fd_context.fetch.call_args.kwargs
        assert "params" not in kwargs
        assert "json" not in kwargs

    @pytest.mark.asyncio
    async def test_missing_contact_id_is_captured(self, fd_context):
        result = await GetContactAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 404: Contact not found")

        result = await GetContactAction().execute({"contact_id": 999}, fd_context)

        assert result["result"] is False
        assert result["contact"] == {}


# ---- update_contact ----


class TestUpdateContact:
    @pytest.mark.asyncio
    async def test_request_uses_put_to_contact_id(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_CONTACT

        await UpdateContactAction().execute(
            {"contact_id": CONTACT_ID, "name": "John J"}, fd_context
        )

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/contacts/{CONTACT_ID}"
        assert call.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, fd_context):
        fd_context.fetch.return_value = {}

        await UpdateContactAction().execute(
            {"contact_id": CONTACT_ID, "job_title": "Staff Engineer"}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"] == {"job_title": "Staff Engineer"}

    @pytest.mark.asyncio
    async def test_contact_id_not_duplicated_into_body(self, fd_context):
        fd_context.fetch.return_value = {}

        await UpdateContactAction().execute({"contact_id": CONTACT_ID, "name": "X"}, fd_context)

        assert "contact_id" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_company_id_is_not_updatable(self, fd_context):
        """update_contact omits company_id from its field list, so a contact
        cannot be reassigned to another company through this action -- even
        though create_contact accepts it."""
        fd_context.fetch.return_value = {}

        await UpdateContactAction().execute(
            {"contact_id": CONTACT_ID, "company_id": 99}, fd_context
        )

        assert "company_id" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_tags_are_not_updatable(self, fd_context):
        """tags is accepted on create but absent from update's field list."""
        fd_context.fetch.return_value = {}

        await UpdateContactAction().execute(
            {"contact_id": CONTACT_ID, "tags": ["vip"]}, fd_context
        )

        assert "tags" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_all_updatable_fields_forwarded(self, fd_context):
        fd_context.fetch.return_value = {}

        await UpdateContactAction().execute(
            {
                "contact_id": CONTACT_ID,
                "name": "John J",
                "email": "new@example.com",
                "phone": "+1555",
                "mobile": "+1666",
                "job_title": "Staff",
                "description": "updated",
            },
            fd_context,
        )

        body = fd_context.fetch.call_args.kwargs["json"]
        assert set(body) == {"name", "email", "phone", "mobile", "job_title", "description"}

    @pytest.mark.asyncio
    async def test_empty_values_cannot_clear_a_field(self, fd_context):
        """All fields are truthiness-gated, so "" cannot blank a phone number."""
        fd_context.fetch.return_value = {}

        await UpdateContactAction().execute(
            {"contact_id": CONTACT_ID, "phone": "", "description": ""}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_missing_contact_id_is_captured(self, fd_context):
        result = await UpdateContactAction().execute({"name": "X"}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 400: invalid email")

        result = await UpdateContactAction().execute(
            {"contact_id": CONTACT_ID, "email": "bad"}, fd_context
        )

        assert result["result"] is False


# ---- delete_contact ----


class TestDeleteContact:
    @pytest.mark.asyncio
    async def test_reports_success(self, fd_context):
        fd_context.fetch.return_value = None

        result = await DeleteContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        assert result == {"result": True}

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, fd_context):
        fd_context.fetch.return_value = None

        await DeleteContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/contacts/{CONTACT_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_deletion_is_soft_and_uses_the_same_path(self, fd_context):
        """Freshdesk soft-deletes contacts on DELETE /contacts/<id> -- there is no
        separate archive endpoint, so the path must not gain a suffix."""
        fd_context.fetch.return_value = None

        await DeleteContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        assert fd_context.fetch.call_args.args[0].endswith(f"/contacts/{CONTACT_ID}")

    @pytest.mark.asyncio
    async def test_sends_no_body(self, fd_context):
        fd_context.fetch.return_value = None

        await DeleteContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        assert "json" not in fd_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_missing_contact_id_is_captured(self, fd_context):
        result = await DeleteContactAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 403: forbidden")

        result = await DeleteContactAction().execute({"contact_id": CONTACT_ID}, fd_context)

        assert result["result"] is False
        assert "403" in result["error"]


# ---- search_contacts ----


class TestSearchContacts:
    @pytest.mark.asyncio
    async def test_returns_matches(self, fd_context):
        fd_context.fetch.return_value = [SAMPLE_CONTACT]

        result = await SearchContactsAction().execute({"term": "john"}, fd_context)

        assert result["result"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_uses_autocomplete_endpoint_with_term_param(self, fd_context):
        """Contacts search takes `term`, whereas company search takes `name`."""
        fd_context.fetch.return_value = []

        await SearchContactsAction().execute({"term": "john"}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/contacts/autocomplete"
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["params"] == {"term": "john"}

    @pytest.mark.asyncio
    async def test_response_is_a_bare_array(self, fd_context):
        """Contact autocomplete returns an array directly, unlike company
        autocomplete which wraps results in a `companies` key."""
        fd_context.fetch.return_value = [SAMPLE_CONTACT, SAMPLE_CONTACT]

        result = await SearchContactsAction().execute({"term": "john"}, fd_context)

        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_dict_response_yields_empty(self, fd_context):
        fd_context.fetch.return_value = {"contacts": [SAMPLE_CONTACT]}

        result = await SearchContactsAction().execute({"term": "john"}, fd_context)

        assert result["contacts"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_no_matches(self, fd_context):
        fd_context.fetch.return_value = []

        result = await SearchContactsAction().execute({"term": "nobody"}, fd_context)

        assert result["result"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_missing_term_is_captured(self, fd_context):
        result = await SearchContactsAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 429: rate limited")

        result = await SearchContactsAction().execute({"term": "john"}, fd_context)

        assert result["result"] is False
        assert result["total"] == 0


# ---- Config ----


class TestFreshdeskContactConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_create_contact_requires_name_and_email(self, config):
        required = config["actions"]["create_contact"]["input_schema"]["required"]
        assert sorted(required) == ["email", "name"]

    @pytest.mark.parametrize("action", ["get_contact", "update_contact", "delete_contact"])
    def test_id_scoped_actions_require_contact_id(self, config, action):
        assert "contact_id" in config["actions"][action]["input_schema"]["required"]

    def test_search_contacts_requires_term(self, config):
        """The input is `term` here and `name` for companies -- asserted so the
        two search actions don't get conflated."""
        assert "term" in config["actions"]["search_contacts"]["input_schema"]["required"]

    def test_list_contacts_exposes_pagination(self, config):
        props = config["actions"]["list_contacts"]["input_schema"]["properties"]
        assert "page" in props
        assert "per_page" in props
