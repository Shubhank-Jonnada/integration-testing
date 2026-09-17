"""Unit tests for the Freshdesk company actions.

Covers the six company actions plus the shared Basic-auth header and base-URL
helpers.

Fully mocked -- no network access.
"""

import base64
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from freshdesk.freshdesk import (  # noqa: E402
    FRESHDESK_API_VERSION,
    CreateCompanyAction,
    DeleteCompanyAction,
    GetCompanyAction,
    ListCompaniesAction,
    SearchCompaniesAction,
    UpdateCompanyAction,
    get_auth_headers,
    get_base_url,
    freshdesk as freshdesk_integration,
)

pytestmark = pytest.mark.unit

API_KEY = "test_freshdesk_api_key"  # nosec B105
DOMAIN = "testcompany"
BASE_URL = f"https://{DOMAIN}.freshdesk.com/api/v2"
COMPANY_ID = 42
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_COMPANY = {
    "id": COMPANY_ID,
    "name": "Acme Corporation",
    "domains": ["acme.com"],
    "description": "A test company",
}


@pytest.fixture
def fd_context():
    """Context carrying Freshdesk's custom-auth credential shape."""
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"api_key": API_KEY, "domain": DOMAIN}}
    return ctx


# ---- Helpers ----


class TestGetAuthHeaders:
    def test_basic_auth_uses_api_key_with_x_password(self, fd_context):
        """Freshdesk expects the API key as the username and a literal 'X' as
        the password."""
        headers = get_auth_headers(fd_context)

        encoded = headers["Authorization"].removeprefix("Basic ")
        assert base64.b64decode(encoded).decode() == f"{API_KEY}:X"

    def test_uses_basic_scheme(self, fd_context):
        assert get_auth_headers(fd_context)["Authorization"].startswith("Basic ")

    def test_sets_json_content_type(self, fd_context):
        assert get_auth_headers(fd_context)["Content-Type"] == "application/json"

    def test_missing_api_key_still_encodes(self, fd_context):
        """Absent credentials produce a well-formed but unauthorised header
        rather than raising."""
        fd_context.auth = {}
        headers = get_auth_headers(fd_context)

        encoded = headers["Authorization"].removeprefix("Basic ")
        assert base64.b64decode(encoded).decode() == ":X"

    def test_non_ascii_api_key_raises(self, fd_context):
        """The helper encodes as ASCII, so a non-ASCII key fails loudly."""
        fd_context.auth = {"credentials": {"api_key": "kéy"}}

        with pytest.raises(UnicodeEncodeError):
            get_auth_headers(fd_context)


class TestGetBaseUrl:
    def test_builds_domain_scoped_url(self, fd_context):
        assert get_base_url(fd_context) == BASE_URL

    def test_uses_api_v2(self, fd_context):
        assert FRESHDESK_API_VERSION == "v2"
        assert get_base_url(fd_context).endswith("/api/v2")

    def test_domain_is_a_subdomain_not_a_full_host(self, fd_context):
        """The credential is the subdomain only -- passing a full host would
        produce a malformed URL."""
        fd_context.auth = {"credentials": {"domain": "acme"}}

        assert get_base_url(fd_context) == "https://acme.freshdesk.com/api/v2"

    def test_missing_domain_yields_malformed_host(self, fd_context):
        """Documented rather than guarded: an empty domain produces a URL that
        resolves to the wrong host."""
        fd_context.auth = {}

        assert get_base_url(fd_context) == "https://.freshdesk.com/api/v2"


# ---- list_companies ----


class TestListCompanies:
    @pytest.mark.asyncio
    async def test_returns_companies_and_total(self, fd_context):
        fd_context.fetch.return_value = [SAMPLE_COMPANY]

        result = await ListCompaniesAction().execute({}, fd_context)

        assert result["result"] is True
        assert result["companies"] == [SAMPLE_COMPANY]
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, fd_context):
        fd_context.fetch.return_value = []

        await ListCompaniesAction().execute({}, fd_context)

        assert fd_context.fetch.call_args.args[0] == f"{BASE_URL}/companies"
        assert fd_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_default_pagination(self, fd_context):
        fd_context.fetch.return_value = []

        await ListCompaniesAction().execute({}, fd_context)

        assert fd_context.fetch.call_args.kwargs["params"] == {"page": 1, "per_page": 30}

    @pytest.mark.asyncio
    async def test_explicit_pagination_forwarded(self, fd_context):
        fd_context.fetch.return_value = []

        await ListCompaniesAction().execute({"page": 3, "per_page": 100}, fd_context)

        assert fd_context.fetch.call_args.kwargs["params"] == {"page": 3, "per_page": 100}

    @pytest.mark.asyncio
    async def test_total_counts_the_page_not_the_account(self, fd_context):
        """`total` is the length of the returned page, so it caps at per_page --
        it is not an account-wide count."""
        fd_context.fetch.return_value = [SAMPLE_COMPANY] * 30

        result = await ListCompaniesAction().execute({"per_page": 30}, fd_context)

        assert result["total"] == 30

    @pytest.mark.asyncio
    async def test_non_list_response_yields_empty(self, fd_context):
        fd_context.fetch.return_value = {"message": "unexpected"}

        result = await ListCompaniesAction().execute({}, fd_context)

        assert result["companies"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 401: invalid credentials")

        result = await ListCompaniesAction().execute({}, fd_context)

        assert result["result"] is False
        assert result["companies"] == []
        assert "401" in result["error"]


# ---- create_company ----


class TestCreateCompany:
    @pytest.mark.asyncio
    async def test_creates_company(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_COMPANY

        result = await CreateCompanyAction().execute({"name": "Acme Corporation"}, fd_context)

        assert result["result"] is True
        assert result["company"]["id"] == COMPANY_ID

    @pytest.mark.asyncio
    async def test_request_url_method_and_minimal_body(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateCompanyAction().execute({"name": "Acme"}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/companies"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["json"] == {"name": "Acme"}

    @pytest.mark.asyncio
    async def test_optional_fields_forwarded(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateCompanyAction().execute(
            {
                "name": "Acme",
                "description": "desc",
                "domains": ["acme.com"],
                "note": "a note",
                "custom_fields": {"tier": "gold"},
            },
            fd_context,
        )

        body = fd_context.fetch.call_args.kwargs["json"]
        assert body["description"] == "desc"
        assert body["domains"] == ["acme.com"]
        assert body["note"] == "a note"
        assert body["custom_fields"] == {"tier": "gold"}

    @pytest.mark.asyncio
    async def test_empty_optional_fields_are_dropped(self, fd_context):
        """Optionals are truthiness-gated, so empty strings and lists are
        omitted -- meaning they cannot be used to clear a field."""
        fd_context.fetch.return_value = {}

        await CreateCompanyAction().execute(
            {"name": "Acme", "description": "", "domains": [], "custom_fields": {}}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"] == {"name": "Acme"}

    @pytest.mark.asyncio
    async def test_missing_name_is_captured(self, fd_context):
        result = await CreateCompanyAction().execute({}, fd_context)

        assert result["result"] is False
        assert result["company"] == {}
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 409: name already exists")

        result = await CreateCompanyAction().execute({"name": "Acme"}, fd_context)

        assert result["result"] is False
        assert "409" in result["error"]


# ---- get_company ----


class TestGetCompany:
    @pytest.mark.asyncio
    async def test_returns_company(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_COMPANY

        result = await GetCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        assert result["result"] is True
        assert result["company"]["name"] == "Acme Corporation"

    @pytest.mark.asyncio
    async def test_request_url_includes_company_id(self, fd_context):
        fd_context.fetch.return_value = {}

        await GetCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        assert fd_context.fetch.call_args.args[0] == f"{BASE_URL}/companies/{COMPANY_ID}"
        assert fd_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_sends_no_params_or_body(self, fd_context):
        fd_context.fetch.return_value = {}

        await GetCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        kwargs = fd_context.fetch.call_args.kwargs
        assert "params" not in kwargs
        assert "json" not in kwargs

    @pytest.mark.asyncio
    async def test_missing_company_id_is_captured(self, fd_context):
        result = await GetCompanyAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 404: Company not found")

        result = await GetCompanyAction().execute({"company_id": 999}, fd_context)

        assert result["result"] is False
        assert result["company"] == {}


# ---- update_company ----


class TestUpdateCompany:
    @pytest.mark.asyncio
    async def test_updates_company(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_COMPANY

        result = await UpdateCompanyAction().execute(
            {"company_id": COMPANY_ID, "name": "Acme Inc"}, fd_context
        )

        assert result["result"] is True

    @pytest.mark.asyncio
    async def test_request_uses_put_to_company_id(self, fd_context):
        fd_context.fetch.return_value = {}

        await UpdateCompanyAction().execute({"company_id": COMPANY_ID, "name": "Acme Inc"}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/companies/{COMPANY_ID}"
        assert call.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, fd_context):
        fd_context.fetch.return_value = {}

        await UpdateCompanyAction().execute({"company_id": COMPANY_ID, "note": "n"}, fd_context)

        assert fd_context.fetch.call_args.kwargs["json"] == {"note": "n"}

    @pytest.mark.asyncio
    async def test_id_is_not_duplicated_into_the_body(self, fd_context):
        """company_id belongs in the path only."""
        fd_context.fetch.return_value = {}

        await UpdateCompanyAction().execute({"company_id": COMPANY_ID, "name": "X"}, fd_context)

        assert "company_id" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_no_updatable_fields_sends_empty_body(self, fd_context):
        """An id-only update issues a PUT with `{}` rather than short-circuiting."""
        fd_context.fetch.return_value = {}

        await UpdateCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        assert fd_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_empty_values_cannot_clear_a_field(self, fd_context):
        """Truthiness gating means passing "" to blank a description is a no-op."""
        fd_context.fetch.return_value = {}

        await UpdateCompanyAction().execute(
            {"company_id": COMPANY_ID, "description": ""}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 400: validation failed")

        result = await UpdateCompanyAction().execute(
            {"company_id": COMPANY_ID, "name": "X"}, fd_context
        )

        assert result["result"] is False


# ---- delete_company ----


class TestDeleteCompany:
    @pytest.mark.asyncio
    async def test_reports_success(self, fd_context):
        fd_context.fetch.return_value = None

        result = await DeleteCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        assert result["result"] is True

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, fd_context):
        fd_context.fetch.return_value = None

        await DeleteCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/companies/{COMPANY_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_response_carries_no_company_payload(self, fd_context):
        """Deletion returns only the result flag -- there is no deleted record to
        hand back."""
        fd_context.fetch.return_value = None

        result = await DeleteCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        assert result == {"result": True}

    @pytest.mark.asyncio
    async def test_missing_company_id_is_captured(self, fd_context):
        result = await DeleteCompanyAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 403: forbidden")

        result = await DeleteCompanyAction().execute({"company_id": COMPANY_ID}, fd_context)

        assert result["result"] is False
        assert "403" in result["error"]


# ---- search_companies ----


class TestSearchCompanies:
    @pytest.mark.asyncio
    async def test_returns_matches(self, fd_context):
        fd_context.fetch.return_value = {"companies": [SAMPLE_COMPANY]}

        result = await SearchCompaniesAction().execute({"name": "acme"}, fd_context)

        assert result["result"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_uses_autocomplete_endpoint(self, fd_context):
        """Company search goes through /autocomplete, not the generic /search."""
        fd_context.fetch.return_value = {"companies": []}

        await SearchCompaniesAction().execute({"name": "acme"}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/companies/autocomplete"
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["params"] == {"name": "acme"}

    @pytest.mark.asyncio
    async def test_results_unwrapped_from_companies_key(self, fd_context):
        """Autocomplete returns an object, unlike list_companies which returns a
        bare array."""
        fd_context.fetch.return_value = {"companies": [SAMPLE_COMPANY, SAMPLE_COMPANY]}

        result = await SearchCompaniesAction().execute({"name": "acme"}, fd_context)

        assert result["total"] == 2

    @pytest.mark.asyncio
    async def test_list_response_yields_empty(self, fd_context):
        """A bare array means the envelope changed -- coerced rather than crashing."""
        fd_context.fetch.return_value = [SAMPLE_COMPANY]

        result = await SearchCompaniesAction().execute({"name": "acme"}, fd_context)

        assert result["companies"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_no_matches(self, fd_context):
        fd_context.fetch.return_value = {"companies": []}

        result = await SearchCompaniesAction().execute({"name": "nonexistent"}, fd_context)

        assert result["result"] is True
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_missing_name_is_captured(self, fd_context):
        result = await SearchCompaniesAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 429: rate limited")

        result = await SearchCompaniesAction().execute({"name": "acme"}, fd_context)

        assert result["result"] is False
        assert "429" in result["error"]


# ---- Config ----


class TestFreshdeskCompanyConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(freshdesk_integration._action_handlers.keys())

        assert defined == registered

    @pytest.mark.parametrize("action", ["get_company", "update_company", "delete_company"])
    def test_id_scoped_actions_require_company_id(self, config, action):
        assert "company_id" in config["actions"][action]["input_schema"]["required"]

    def test_create_company_requires_name(self, config):
        assert "name" in config["actions"]["create_company"]["input_schema"]["required"]

    def test_search_companies_requires_name(self, config):
        assert "name" in config["actions"]["search_companies"]["input_schema"]["required"]

    def test_list_companies_requires_nothing(self, config):
        assert not config["actions"]["list_companies"]["input_schema"].get("required")

    def test_auth_declares_api_key_and_domain(self, config):
        props = config["auth"]["fields"]["properties"]

        assert "api_key" in props
        assert "domain" in props

    def test_api_key_is_masked(self, config):
        assert config["auth"]["fields"]["properties"]["api_key"]["format"] == "password"
