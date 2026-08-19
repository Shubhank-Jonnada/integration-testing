"""Unit tests for the Notion data source, block-append, and page-property actions.

`test_notion_unit.py` covers nine of the fourteen actions. These are the five it
doesn't:

- `list_data_sources`
- `get_data_source`
- `query_notion_data_source`
- `append_notion_block_children`
- `get_notion_page_property`

Data sources are Notion's 2025-09-03 replacement for querying databases
directly, so these paths are newer and less exercised than the rest.

Fully mocked -- no network access.
"""

import json
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from autohive_integrations_sdk import ActionError, FetchResponse  # noqa: E402

from notion.notion import (  # noqa: E402
    NOTION_API_VERSION,
    NotionAppendBlockChildrenHandler,
    NotionGetDataSourceHandler,
    NotionGetPagePropertyHandler,
    NotionListDataSourcesHandler,
    NotionQueryDataSourceHandler,
)

pytestmark = pytest.mark.unit

DATABASE_ID = "668d797c-76fa-4934-9b05-ad288df2d136"
DATA_SOURCE_ID = "b9bb7d5a-7c1e-4d3f-8f2a-1c4e5b6a7d8e"
BLOCK_ID = "c02fc1d3-db8b-45c5-a222-27595b15aea7"
PAGE_ID = "59833787-2cf9-4fdf-8782-e53db20768a5"
PROPERTY_ID = "title"
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

API = "https://api.notion.com/v1"


@pytest.fixture
def mock_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {
        "auth_type": "PlatformOauth2",
        "credentials": {"access_token": "test_token"},  # nosec B105
    }
    return ctx


def response(data):
    return FetchResponse(status=200, headers={}, data=data)


# ---- list_data_sources ----


class TestListDataSources:
    @pytest.mark.asyncio
    async def test_returns_data_sources(self, mock_context):
        mock_context.fetch.return_value = response(
            {"results": [{"id": DATA_SOURCE_ID, "name": "Tasks"}], "has_more": False}
        )

        result = await NotionListDataSourcesHandler().execute(
            {"database_id": DATABASE_ID}, mock_context
        )

        assert result.data["data_sources"][0]["id"] == DATA_SOURCE_ID
        assert result.data["has_more"] is False

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, mock_context):
        mock_context.fetch.return_value = response({"results": []})

        await NotionListDataSourcesHandler().execute({"database_id": DATABASE_ID}, mock_context)

        call = mock_context.fetch.call_args
        assert call.kwargs["url"] == f"{API}/databases/{DATABASE_ID}/data_sources"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_sends_the_api_version_header(self, mock_context):
        """Notion requires a version header on every request and rejects calls
        without it."""
        mock_context.fetch.return_value = response({"results": []})

        await NotionListDataSourcesHandler().execute({"database_id": DATABASE_ID}, mock_context)

        assert mock_context.fetch.call_args.kwargs["headers"]["Notion-Version"] == NOTION_API_VERSION

    @pytest.mark.asyncio
    async def test_pagination_forwarded(self, mock_context):
        mock_context.fetch.return_value = response({"results": []})

        await NotionListDataSourcesHandler().execute(
            {"database_id": DATABASE_ID, "page_size": 50, "start_cursor": "cur_1"}, mock_context
        )

        params = mock_context.fetch.call_args.kwargs["params"]
        assert params == {"page_size": 50, "start_cursor": "cur_1"}

    @pytest.mark.asyncio
    async def test_no_pagination_sends_empty_params(self, mock_context):
        mock_context.fetch.return_value = response({"results": []})

        await NotionListDataSourcesHandler().execute({"database_id": DATABASE_ID}, mock_context)

        assert mock_context.fetch.call_args.kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_next_cursor_is_surfaced(self, mock_context):
        """Without the cursor a caller can't page through a large database."""
        mock_context.fetch.return_value = response(
            {"results": [], "has_more": True, "next_cursor": "cur_2"}
        )

        result = await NotionListDataSourcesHandler().execute(
            {"database_id": DATABASE_ID}, mock_context
        )

        assert result.data["has_more"] is True
        assert result.data["next_cursor"] == "cur_2"

    @pytest.mark.asyncio
    async def test_missing_keys_default_safely(self, mock_context):
        mock_context.fetch.return_value = response({})

        result = await NotionListDataSourcesHandler().execute(
            {"database_id": DATABASE_ID}, mock_context
        )

        assert result.data["data_sources"] == []
        assert result.data["has_more"] is False
        assert result.data["next_cursor"] is None

    @pytest.mark.asyncio
    async def test_error_returns_action_error(self, mock_context):
        mock_context.fetch.side_effect = Exception("HTTP 404: database not found")

        result = await NotionListDataSourcesHandler().execute(
            {"database_id": "missing"}, mock_context
        )

        assert isinstance(result, ActionError)
        assert "database not found" in result.message

    @pytest.mark.asyncio
    async def test_missing_database_id_raises(self, mock_context):
        """The id lookup happens before the try block, so a KeyError propagates
        rather than becoming an ActionError."""
        with pytest.raises(KeyError):
            await NotionListDataSourcesHandler().execute({}, mock_context)

        mock_context.fetch.assert_not_called()


# ---- get_data_source ----


class TestGetDataSource:
    @pytest.mark.asyncio
    async def test_returns_the_schema(self, mock_context):
        schema = {"id": DATA_SOURCE_ID, "properties": {"Name": {"type": "title"}}}
        mock_context.fetch.return_value = response(schema)

        result = await NotionGetDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        assert result.data["data_source"] == schema

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, mock_context):
        mock_context.fetch.return_value = response({})

        await NotionGetDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        call = mock_context.fetch.call_args
        assert call.kwargs["url"] == f"{API}/data_sources/{DATA_SOURCE_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_response_is_returned_whole(self, mock_context):
        """The schema is passed through unreshaped, so callers see Notion's own
        property definitions."""
        mock_context.fetch.return_value = response({"properties": {"Status": {"type": "select"}}})

        result = await NotionGetDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        assert result.data["data_source"]["properties"]["Status"]["type"] == "select"

    @pytest.mark.asyncio
    async def test_error_returns_action_error(self, mock_context):
        mock_context.fetch.side_effect = Exception("HTTP 404")

        result = await NotionGetDataSourceHandler().execute(
            {"data_source_id": "missing"}, mock_context
        )

        assert isinstance(result, ActionError)

    @pytest.mark.asyncio
    async def test_missing_id_raises(self, mock_context):
        with pytest.raises(KeyError):
            await NotionGetDataSourceHandler().execute({}, mock_context)


# ---- query_notion_data_source ----


class TestQueryDataSource:
    @pytest.mark.asyncio
    async def test_returns_results(self, mock_context):
        mock_context.fetch.return_value = response(
            {"results": [{"id": PAGE_ID}], "has_more": False, "type": "page_or_data_source"}
        )

        result = await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        assert result.data["results"][0]["id"] == PAGE_ID
        assert result.data["type"] == "page_or_data_source"

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, mock_context):
        mock_context.fetch.return_value = response({"results": []})

        await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        call = mock_context.fetch.call_args
        assert call.kwargs["url"] == f"{API}/data_sources/{DATA_SOURCE_ID}/query"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_query_sends_a_json_content_type(self, mock_context):
        """Unlike the GET actions, the query POST also sets Content-Type."""
        mock_context.fetch.return_value = response({"results": []})

        await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        headers = mock_context.fetch.call_args.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Notion-Version"] == NOTION_API_VERSION

    @pytest.mark.asyncio
    async def test_empty_query_sends_an_empty_body(self, mock_context):
        """An unfiltered query returns every row, so the empty-body case is the
        "fetch everything" path."""
        mock_context.fetch.return_value = response({"results": []})

        await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_filter_is_passed_through_verbatim(self, mock_context):
        """Notion filters are deeply nested and caller-authored, so the handler
        must not reshape them."""
        mock_context.fetch.return_value = response({"results": []})
        notion_filter = {
            "and": [
                {"property": "Status", "select": {"equals": "Done"}},
                {"property": "Due", "date": {"before": "2026-02-01"}},
            ]
        }

        await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID, "filter": notion_filter}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["json"]["filter"] == notion_filter

    @pytest.mark.asyncio
    async def test_sorts_are_passed_through(self, mock_context):
        mock_context.fetch.return_value = response({"results": []})
        sorts = [{"property": "Due", "direction": "ascending"}]

        await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID, "sorts": sorts}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["json"]["sorts"] == sorts

    @pytest.mark.asyncio
    async def test_pagination_goes_in_the_body_not_the_query_string(self, mock_context):
        """This is a POST, so page_size and start_cursor are body fields -- the
        opposite of list_data_sources."""
        mock_context.fetch.return_value = response({"results": []})

        await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID, "page_size": 25, "start_cursor": "cur_1"},
            mock_context,
        )

        body = mock_context.fetch.call_args.kwargs["json"]
        assert body["page_size"] == 25
        assert body["start_cursor"] == "cur_1"
        assert "params" not in mock_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_empty_filter_is_dropped(self, mock_context):
        """An empty dict is falsy, so it is omitted rather than sent as `{}` --
        which Notion would reject as a malformed filter."""
        mock_context.fetch.return_value = response({"results": []})

        await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID, "filter": {}, "sorts": []}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_missing_keys_default_safely(self, mock_context):
        mock_context.fetch.return_value = response({})

        result = await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID}, mock_context
        )

        assert result.data["results"] == []
        assert result.data["has_more"] is False
        assert result.data["next_cursor"] is None
        assert result.data["type"] is None

    @pytest.mark.asyncio
    async def test_error_returns_action_error(self, mock_context):
        mock_context.fetch.side_effect = Exception("HTTP 400: invalid filter")

        result = await NotionQueryDataSourceHandler().execute(
            {"data_source_id": DATA_SOURCE_ID, "filter": {"bad": True}}, mock_context
        )

        assert isinstance(result, ActionError)
        assert "invalid filter" in result.message


# ---- append_notion_block_children ----


class TestAppendBlockChildren:
    CHILDREN = [
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": "Hello"}}]},
        }
    ]

    @pytest.mark.asyncio
    async def test_appends_children(self, mock_context):
        mock_context.fetch.return_value = response(
            {"results": [{"id": "new-block", "type": "paragraph"}], "has_more": False}
        )

        result = await NotionAppendBlockChildrenHandler().execute(
            {"block_id": BLOCK_ID, "children": self.CHILDREN}, mock_context
        )

        assert result.data["blocks"][0]["id"] == "new-block"

    @pytest.mark.asyncio
    async def test_request_uses_patch(self, mock_context):
        """Appending is a PATCH on the children collection, not a POST."""
        mock_context.fetch.return_value = response({"results": []})

        await NotionAppendBlockChildrenHandler().execute(
            {"block_id": BLOCK_ID, "children": self.CHILDREN}, mock_context
        )

        call = mock_context.fetch.call_args
        assert call.kwargs["url"] == f"{API}/blocks/{BLOCK_ID}/children"
        assert call.kwargs["method"] == "PATCH"

    @pytest.mark.asyncio
    async def test_children_are_wrapped_in_a_children_key(self, mock_context):
        mock_context.fetch.return_value = response({"results": []})

        await NotionAppendBlockChildrenHandler().execute(
            {"block_id": BLOCK_ID, "children": self.CHILDREN}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["json"] == {"children": self.CHILDREN}

    @pytest.mark.asyncio
    async def test_after_positions_the_insertion(self, mock_context):
        """`after` inserts following a specific sibling instead of at the end."""
        mock_context.fetch.return_value = response({"results": []})

        await NotionAppendBlockChildrenHandler().execute(
            {"block_id": BLOCK_ID, "children": self.CHILDREN, "after": "sibling-block"},
            mock_context,
        )

        assert mock_context.fetch.call_args.kwargs["json"]["after"] == "sibling-block"

    @pytest.mark.asyncio
    async def test_after_omitted_appends_at_the_end(self, mock_context):
        mock_context.fetch.return_value = response({"results": []})

        await NotionAppendBlockChildrenHandler().execute(
            {"block_id": BLOCK_ID, "children": self.CHILDREN}, mock_context
        )

        assert "after" not in mock_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_results_are_exposed_as_blocks(self, mock_context):
        """The API returns `results`; the action renames it to `blocks`."""
        mock_context.fetch.return_value = response({"results": [{"id": "b1"}]})

        result = await NotionAppendBlockChildrenHandler().execute(
            {"block_id": BLOCK_ID, "children": self.CHILDREN}, mock_context
        )

        assert "blocks" in result.data
        assert "results" not in result.data

    @pytest.mark.asyncio
    async def test_missing_children_raises(self, mock_context):
        with pytest.raises(KeyError):
            await NotionAppendBlockChildrenHandler().execute({"block_id": BLOCK_ID}, mock_context)

        mock_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_returns_action_error(self, mock_context):
        mock_context.fetch.side_effect = Exception("HTTP 400: validation_error")

        result = await NotionAppendBlockChildrenHandler().execute(
            {"block_id": BLOCK_ID, "children": self.CHILDREN}, mock_context
        )

        assert isinstance(result, ActionError)


# ---- get_notion_page_property ----


class TestGetPageProperty:
    @pytest.mark.asyncio
    async def test_returns_the_property(self, mock_context):
        mock_context.fetch.return_value = response(
            {"object": "property_item", "type": "title", "title": {"plain_text": "My page"}}
        )

        result = await NotionGetPagePropertyHandler().execute(
            {"page_id": PAGE_ID, "property_id": PROPERTY_ID}, mock_context
        )

        assert result.data["property"]["type"] == "title"

    @pytest.mark.asyncio
    async def test_request_url_nests_page_and_property(self, mock_context):
        mock_context.fetch.return_value = response({})

        await NotionGetPagePropertyHandler().execute(
            {"page_id": PAGE_ID, "property_id": PROPERTY_ID}, mock_context
        )

        call = mock_context.fetch.call_args
        assert call.kwargs["url"] == f"{API}/pages/{PAGE_ID}/properties/{PROPERTY_ID}"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_pagination_forwarded_as_params(self, mock_context):
        """Rollup and relation properties paginate, so these matter here."""
        mock_context.fetch.return_value = response({})

        await NotionGetPagePropertyHandler().execute(
            {
                "page_id": PAGE_ID,
                "property_id": PROPERTY_ID,
                "page_size": 10,
                "start_cursor": "cur_1",
            },
            mock_context,
        )

        assert mock_context.fetch.call_args.kwargs["params"] == {
            "page_size": 10,
            "start_cursor": "cur_1",
        }

    @pytest.mark.asyncio
    async def test_no_pagination_sends_empty_params(self, mock_context):
        mock_context.fetch.return_value = response({})

        await NotionGetPagePropertyHandler().execute(
            {"page_id": PAGE_ID, "property_id": PROPERTY_ID}, mock_context
        )

        assert mock_context.fetch.call_args.kwargs["params"] == {}

    @pytest.mark.asyncio
    async def test_paginated_property_list_is_returned_whole(self, mock_context):
        """A relation property returns a property_item *list*, not a single value."""
        mock_context.fetch.return_value = response(
            {
                "object": "list",
                "results": [{"type": "relation", "relation": {"id": "r1"}}],
                "has_more": True,
                "next_cursor": "cur_2",
            }
        )

        result = await NotionGetPagePropertyHandler().execute(
            {"page_id": PAGE_ID, "property_id": "relation-prop"}, mock_context
        )

        assert result.data["property"]["has_more"] is True
        assert result.data["property"]["next_cursor"] == "cur_2"

    @pytest.mark.parametrize("missing", ["page_id", "property_id"])
    @pytest.mark.asyncio
    async def test_both_ids_are_required(self, mock_context, missing):
        inputs = {"page_id": PAGE_ID, "property_id": PROPERTY_ID}
        del inputs[missing]

        with pytest.raises(KeyError):
            await NotionGetPagePropertyHandler().execute(inputs, mock_context)

        mock_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_returns_action_error(self, mock_context):
        mock_context.fetch.side_effect = Exception("HTTP 404: property not found")

        result = await NotionGetPagePropertyHandler().execute(
            {"page_id": PAGE_ID, "property_id": "nope"}, mock_context
        )

        assert isinstance(result, ActionError)
        assert "property not found" in result.message


# ---- Config ----


class TestNotionDataSourceConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_list_data_sources_requires_database_id(self, config):
        required = config["actions"]["list_data_sources"]["input_schema"]["required"]
        assert required == ["database_id"]

    @pytest.mark.parametrize("action", ["get_data_source", "query_notion_data_source"])
    def test_data_source_actions_require_data_source_id(self, config, action):
        assert "data_source_id" in config["actions"][action]["input_schema"]["required"]

    def test_append_requires_block_id_and_children(self, config):
        required = config["actions"]["append_notion_block_children"]["input_schema"]["required"]
        assert "block_id" in required
        assert "children" in required

    def test_get_page_property_requires_both_ids(self, config):
        required = config["actions"]["get_notion_page_property"]["input_schema"]["required"]
        assert sorted(required) == ["page_id", "property_id"]

    def test_api_version_is_the_data_sources_release(self, config):
        """Data sources only exist from 2025-09-03 onward, so an older pinned
        version would 404 on every action in this file."""
        assert NOTION_API_VERSION >= "2025-09-03"

    def test_query_page_size_is_capped_at_notion_maximum(self, config):
        props = config["actions"]["query_notion_data_source"]["input_schema"]["properties"]
        assert props["page_size"]["maximum"] == 100
