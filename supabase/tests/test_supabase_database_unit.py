"""Unit tests for the Supabase database (PostgREST) actions.

Covers `select_records`, `insert_records`, `update_records`, `delete_records`,
and `call_function`, plus the shared header and base-URL helpers.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from supabase.supabase import (  # noqa: E402
    CallFunctionAction,
    DeleteRecordsAction,
    InsertRecordsAction,
    SelectRecordsAction,
    UpdateRecordsAction,
    get_base_url,
    get_headers,
    supabase as supabase_integration,
)

pytestmark = pytest.mark.unit

SERVICE_KEY = "test_service_role_secret"  # nosec B105
HOST = "https://abcdefgh.supabase.co"
TABLE = "customers"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_ROWS = [
    {"id": 1, "email": "a@example.com", "name": "Ada"},
    {"id": 2, "email": "b@example.com", "name": "Grace"},
]


@pytest.fixture
def sb_context():
    """Context carrying Supabase's host + service-role credential shape."""
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"host": HOST, "service_role_secret": SERVICE_KEY}}
    return ctx


# ---- Helpers ----


class TestGetHeaders:
    def test_sends_key_in_both_apikey_and_bearer(self, sb_context):
        """PostgREST needs `apikey`; GoTrue needs `Authorization`. Both carry the
        same service-role secret."""
        headers = get_headers(sb_context)

        assert headers["apikey"] == SERVICE_KEY
        assert headers["Authorization"] == f"Bearer {SERVICE_KEY}"

    def test_defaults_to_return_representation(self, sb_context):
        """Without this Prefer header PostgREST returns no body on writes."""
        assert get_headers(sb_context)["Prefer"] == "return=representation"

    def test_sets_json_content_type(self, sb_context):
        assert get_headers(sb_context)["Content-Type"] == "application/json"

    def test_missing_credentials_yield_empty_key(self, sb_context):
        sb_context.auth = {}
        headers = get_headers(sb_context)

        assert headers["apikey"] == ""
        assert headers["Authorization"] == "Bearer "

    def test_returns_a_fresh_dict_each_call(self, sb_context):
        """Handlers mutate the returned headers, so sharing one dict would leak
        Prefer/Range values between actions."""
        first = get_headers(sb_context)
        first["Prefer"] = "mutated"

        assert get_headers(sb_context)["Prefer"] == "return=representation"


class TestGetBaseUrl:
    def test_returns_host(self, sb_context):
        assert get_base_url(sb_context) == HOST

    @pytest.mark.parametrize(
        "host, expected",
        [
            ("https://x.supabase.co/", "https://x.supabase.co"),
            ("https://x.supabase.co///", "https://x.supabase.co"),
            ("https://x.supabase.co", "https://x.supabase.co"),
        ],
    )
    def test_strips_trailing_slashes(self, sb_context, host, expected):
        """A trailing slash would produce a double slash in every request path."""
        sb_context.auth = {"credentials": {"host": host}}
        assert get_base_url(sb_context) == expected

    def test_missing_host_returns_empty_string(self, sb_context):
        sb_context.auth = {}
        assert get_base_url(sb_context) == ""


# ---- select_records ----


class TestSelectRecords:
    @pytest.mark.asyncio
    async def test_returns_rows_and_count(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_ROWS

        result = await SelectRecordsAction().execute({"table": TABLE}, sb_context)

        assert result.data["result"] is True
        assert result.data["records"] == SAMPLE_ROWS
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, sb_context):
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute({"table": TABLE}, sb_context)

        assert sb_context.fetch.call_args.args[0] == f"{HOST}/rest/v1/{TABLE}"
        assert sb_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_select_and_order_forwarded(self, sb_context):
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute(
            {"table": TABLE, "select": "id,email", "order": "id.desc"}, sb_context
        )

        params = sb_context.fetch.call_args.kwargs["params"]
        assert params["select"] == "id,email"
        assert params["order"] == "id.desc"

    @pytest.mark.asyncio
    async def test_filters_flattened_into_params(self, sb_context):
        """PostgREST filters are query params, e.g. ?id=eq.1"""
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute(
            {"table": TABLE, "filters": {"id": "eq.1", "name": "like.*Ada*"}}, sb_context
        )

        params = sb_context.fetch.call_args.kwargs["params"]
        assert params["id"] == "eq.1"
        assert params["name"] == "like.*Ada*"

    @pytest.mark.asyncio
    async def test_limit_sets_range_header(self, sb_context):
        """PostgREST paginates via a Range header, not limit/offset params."""
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute({"table": TABLE, "limit": 10}, sb_context)

        headers = sb_context.fetch.call_args.kwargs["headers"]
        assert headers["Range"] == "0-9"
        assert headers["Range-Unit"] == "items"

    @pytest.mark.asyncio
    async def test_limit_with_offset_shifts_range(self, sb_context):
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute({"table": TABLE, "limit": 25, "offset": 50}, sb_context)

        assert sb_context.fetch.call_args.kwargs["headers"]["Range"] == "50-74"

    @pytest.mark.asyncio
    async def test_limit_one_is_a_single_item_range(self, sb_context):
        """Boundary: the range end is inclusive, so limit=1 is `n-n`."""
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute({"table": TABLE, "limit": 1, "offset": 5}, sb_context)

        assert sb_context.fetch.call_args.kwargs["headers"]["Range"] == "5-5"

    @pytest.mark.asyncio
    async def test_no_limit_sends_no_range_header(self, sb_context):
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute({"table": TABLE}, sb_context)

        assert "Range" not in sb_context.fetch.call_args.kwargs["headers"]

    @pytest.mark.asyncio
    async def test_limit_zero_still_sets_range(self, sb_context):
        """The guard is `is not None`, so limit=0 is honoured -- and produces an
        inverted range, which PostgREST rejects."""
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute({"table": TABLE, "limit": 0}, sb_context)

        assert sb_context.fetch.call_args.kwargs["headers"]["Range"] == "0--1"

    @pytest.mark.asyncio
    async def test_no_params_sends_none(self, sb_context):
        sb_context.fetch.return_value = []

        await SelectRecordsAction().execute({"table": TABLE}, sb_context)

        assert sb_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_non_list_response_yields_empty_records(self, sb_context):
        """A PostgREST error object is a dict, not a list -- coerced to empty."""
        sb_context.fetch.return_value = {"message": "unexpected"}

        result = await SelectRecordsAction().execute({"table": TABLE}, sb_context)

        assert result.data["records"] == []
        assert result.data["count"] == 0

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("relation does not exist")

        result = await SelectRecordsAction().execute({"table": "nope"}, sb_context)

        assert result.data["result"] is False
        assert "relation does not exist" in result.data["error"]


# ---- insert_records ----


class TestInsertRecords:
    @pytest.mark.asyncio
    async def test_inserts_and_returns_rows(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_ROWS

        result = await InsertRecordsAction().execute(
            {"table": TABLE, "records": SAMPLE_ROWS}, sb_context
        )

        assert result.data["result"] is True
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_request_url_method_and_body(self, sb_context):
        sb_context.fetch.return_value = []
        rows = [{"email": "c@example.com"}]

        await InsertRecordsAction().execute({"table": TABLE, "records": rows}, sb_context)

        call = sb_context.fetch.call_args
        assert call.args[0] == f"{HOST}/rest/v1/{TABLE}"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["json"] == rows

    @pytest.mark.asyncio
    async def test_plain_insert_sends_no_params(self, sb_context):
        sb_context.fetch.return_value = []

        await InsertRecordsAction().execute({"table": TABLE, "records": []}, sb_context)

        assert sb_context.fetch.call_args.kwargs["params"] is None

    @pytest.mark.asyncio
    async def test_on_conflict_triggers_upsert(self, sb_context):
        """on_conflict switches PostgREST into merge-duplicates mode."""
        sb_context.fetch.return_value = []

        await InsertRecordsAction().execute(
            {"table": TABLE, "records": SAMPLE_ROWS, "on_conflict": "email"}, sb_context
        )

        call = sb_context.fetch.call_args
        assert call.kwargs["params"] == {"on_conflict": "email"}
        assert call.kwargs["headers"]["Prefer"] == "resolution=merge-duplicates,return=representation"

    @pytest.mark.asyncio
    async def test_return_records_false_requests_minimal(self, sb_context):
        sb_context.fetch.return_value = []

        await InsertRecordsAction().execute(
            {"table": TABLE, "records": SAMPLE_ROWS, "return_records": False}, sb_context
        )

        assert sb_context.fetch.call_args.kwargs["headers"]["Prefer"] == "return=minimal"

    @pytest.mark.asyncio
    async def test_upsert_with_minimal_return_combines_both(self, sb_context):
        """The resolution directive must survive when the caller also opts out of
        the response body."""
        sb_context.fetch.return_value = []

        await InsertRecordsAction().execute(
            {"table": TABLE, "records": SAMPLE_ROWS, "on_conflict": "email", "return_records": False},
            sb_context,
        )

        prefer = sb_context.fetch.call_args.kwargs["headers"]["Prefer"]
        assert prefer == "resolution=merge-duplicates,return=minimal"

    @pytest.mark.asyncio
    async def test_count_falls_back_to_input_length(self, sb_context):
        """With return=minimal the API sends no rows back, so the count comes
        from what was submitted."""
        sb_context.fetch.return_value = None

        result = await InsertRecordsAction().execute(
            {"table": TABLE, "records": SAMPLE_ROWS, "return_records": False}, sb_context
        )

        assert result.data["records"] == []
        assert result.data["count"] == 2

    @pytest.mark.asyncio
    async def test_return_records_true_keeps_representation(self, sb_context):
        sb_context.fetch.return_value = SAMPLE_ROWS

        await InsertRecordsAction().execute(
            {"table": TABLE, "records": SAMPLE_ROWS, "return_records": True}, sb_context
        )

        assert sb_context.fetch.call_args.kwargs["headers"]["Prefer"] == "return=representation"

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("duplicate key value")

        result = await InsertRecordsAction().execute(
            {"table": TABLE, "records": SAMPLE_ROWS}, sb_context
        )

        assert result.data["result"] is False
        assert "duplicate key" in result.data["error"]


# ---- update_records ----


class TestUpdateRecords:
    @pytest.mark.asyncio
    async def test_updates_and_returns_rows(self, sb_context):
        sb_context.fetch.return_value = [SAMPLE_ROWS[0]]

        result = await UpdateRecordsAction().execute(
            {"table": TABLE, "data": {"name": "Ada L"}, "filters": {"id": "eq.1"}}, sb_context
        )

        assert result.data["result"] is True
        assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_request_uses_patch_with_filters(self, sb_context):
        sb_context.fetch.return_value = []

        await UpdateRecordsAction().execute(
            {"table": TABLE, "data": {"name": "X"}, "filters": {"id": "eq.1"}}, sb_context
        )

        call = sb_context.fetch.call_args
        assert call.args[0] == f"{HOST}/rest/v1/{TABLE}"
        assert call.kwargs["method"] == "PATCH"
        assert call.kwargs["params"] == {"id": "eq.1"}
        assert call.kwargs["json"] == {"name": "X"}

    @pytest.mark.asyncio
    async def test_multiple_filters_forwarded(self, sb_context):
        sb_context.fetch.return_value = []

        await UpdateRecordsAction().execute(
            {"table": TABLE, "data": {"n": 1}, "filters": {"id": "gt.5", "active": "is.true"}},
            sb_context,
        )

        assert sb_context.fetch.call_args.kwargs["params"] == {"id": "gt.5", "active": "is.true"}

    @pytest.mark.asyncio
    async def test_return_records_false_yields_none_count(self, sb_context):
        """count is None rather than 0 -- 'unknown', not 'nothing matched'."""
        sb_context.fetch.return_value = None

        result = await UpdateRecordsAction().execute(
            {"table": TABLE, "data": {"n": 1}, "filters": {"id": "eq.1"}, "return_records": False},
            sb_context,
        )

        assert result.data["count"] is None
        assert sb_context.fetch.call_args.kwargs["headers"]["Prefer"] == "return=minimal"

    @pytest.mark.asyncio
    async def test_empty_filters_send_empty_params(self, sb_context):
        """An unfiltered PATCH updates every row -- asserted so the wide-open
        request shape is at least visible."""
        sb_context.fetch.return_value = []

        await UpdateRecordsAction().execute(
            {"table": TABLE, "data": {"n": 1}, "filters": {}}, sb_context
        )

        assert sb_context.fetch.call_args.kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_missing_filters_is_captured(self, sb_context):
        result = await UpdateRecordsAction().execute(
            {"table": TABLE, "data": {"n": 1}}, sb_context
        )

        assert result.data["result"] is False

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("permission denied")

        result = await UpdateRecordsAction().execute(
            {"table": TABLE, "data": {"n": 1}, "filters": {"id": "eq.1"}}, sb_context
        )

        assert result.data["result"] is False
        assert "permission denied" in result.data["error"]


# ---- delete_records ----


class TestDeleteRecords:
    @pytest.mark.asyncio
    async def test_request_uses_delete_with_filters(self, sb_context):
        sb_context.fetch.return_value = []

        await DeleteRecordsAction().execute({"table": TABLE, "filters": {"id": "eq.1"}}, sb_context)

        call = sb_context.fetch.call_args
        assert call.args[0] == f"{HOST}/rest/v1/{TABLE}"
        assert call.kwargs["method"] == "DELETE"
        assert call.kwargs["params"] == {"id": "eq.1"}

    @pytest.mark.asyncio
    async def test_delete_sends_no_body(self, sb_context):
        sb_context.fetch.return_value = []

        await DeleteRecordsAction().execute({"table": TABLE, "filters": {"id": "eq.1"}}, sb_context)

        assert "json" not in sb_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_defaults_to_minimal_return(self, sb_context):
        """Unlike insert/update, delete defaults to NOT returning rows -- the
        guard is `not inputs.get(...)`, so an omitted flag means minimal."""
        sb_context.fetch.return_value = None

        result = await DeleteRecordsAction().execute(
            {"table": TABLE, "filters": {"id": "eq.1"}}, sb_context
        )

        assert sb_context.fetch.call_args.kwargs["headers"]["Prefer"] == "return=minimal"
        assert result.data["count"] is None

    @pytest.mark.asyncio
    async def test_return_records_true_requests_representation(self, sb_context):
        sb_context.fetch.return_value = [SAMPLE_ROWS[0]]

        result = await DeleteRecordsAction().execute(
            {"table": TABLE, "filters": {"id": "eq.1"}, "return_records": True}, sb_context
        )

        assert sb_context.fetch.call_args.kwargs["headers"]["Prefer"] == "return=representation"
        assert result.data["count"] == 1

    @pytest.mark.asyncio
    async def test_error_is_captured(self, sb_context):
        sb_context.fetch.side_effect = Exception("foreign key violation")

        result = await DeleteRecordsAction().execute(
            {"table": TABLE, "filters": {"id": "eq.1"}}, sb_context
        )

        assert result.data["result"] is False
        assert "foreign key" in result.data["error"]


# ---- call_function ----


class TestCallFunction:
    @pytest.mark.asyncio
    async def test_returns_function_result(self, sb_context):
        sb_context.fetch.return_value = {"total": 42}

        result = await CallFunctionAction().execute({"function_name": "get_total"}, sb_context)

        assert result.data["result"] is True
        assert result.data["data"] == {"total": 42}

    @pytest.mark.asyncio
    async def test_request_targets_rpc_endpoint(self, sb_context):
        sb_context.fetch.return_value = {}

        await CallFunctionAction().execute({"function_name": "get_total"}, sb_context)

        call = sb_context.fetch.call_args
        assert call.args[0] == f"{HOST}/rest/v1/rpc/get_total"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_params_sent_as_body(self, sb_context):
        """RPC arguments go in the JSON body, not the query string."""
        sb_context.fetch.return_value = {}

        await CallFunctionAction().execute(
            {"function_name": "add", "params": {"a": 1, "b": 2}}, sb_context
        )

        assert sb_context.fetch.call_args.kwargs["json"] == {"a": 1, "b": 2}

    @pytest.mark.asyncio
    async def test_omitted_params_send_empty_object(self, sb_context):
        """PostgREST requires a JSON body on RPC POSTs, so `{}` not None."""
        sb_context.fetch.return_value = {}

        await CallFunctionAction().execute({"function_name": "noop"}, sb_context)

        assert sb_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_scalar_and_list_returns_pass_through(self, sb_context):
        """RPC can return a scalar or an array, not just an object."""
        sb_context.fetch.return_value = [1, 2, 3]

        result = await CallFunctionAction().execute({"function_name": "series"}, sb_context)

        assert result.data["data"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_error_yields_none_data(self, sb_context):
        sb_context.fetch.side_effect = Exception("function does not exist")

        result = await CallFunctionAction().execute({"function_name": "nope"}, sb_context)

        assert result.data["result"] is False
        assert result.data["data"] is None


# ---- Config ----


class TestSupabaseDatabaseConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_actions_match_registered_handlers(self, config):
        defined = set(config["actions"].keys())
        registered = set(supabase_integration._action_handlers.keys())

        assert defined == registered

    @pytest.mark.parametrize(
        "action",
        ["select_records", "insert_records", "update_records", "delete_records"],
    )
    def test_table_actions_require_table(self, config, action):
        assert "table" in config["actions"][action]["input_schema"]["required"]

    def test_mutating_actions_require_filters(self, config):
        """An update or delete without filters would hit every row."""
        for action in ("update_records", "delete_records"):
            assert "filters" in config["actions"][action]["input_schema"]["required"]

    def test_call_function_requires_function_name(self, config):
        assert "function_name" in config["actions"]["call_function"]["input_schema"]["required"]

    def test_auth_declares_host_and_service_role_secret(self, config):
        props = config["auth"]["fields"]["properties"]
        assert "host" in props
        assert "service_role_secret" in props
