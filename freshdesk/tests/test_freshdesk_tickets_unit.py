"""Unit tests for the Freshdesk ticket and conversation actions.

Covers the five ticket actions plus the three conversation actions
(`list_conversations`, `create_note`, `create_reply`).

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from freshdesk.freshdesk import (  # noqa: E402
    CreateNoteAction,
    CreateReplyAction,
    CreateTicketAction,
    DeleteTicketAction,
    GetTicketAction,
    ListConversationsAction,
    ListTicketsAction,
    UpdateTicketAction,
)

pytestmark = pytest.mark.unit

API_KEY = "test_freshdesk_api_key"  # nosec B105
DOMAIN = "testcompany"
BASE_URL = f"https://{DOMAIN}.freshdesk.com/api/v2"
TICKET_ID = 1001
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SAMPLE_TICKET = {
    "id": TICKET_ID,
    "subject": "Cannot log in",
    "status": 2,
    "priority": 1,
    "requester_id": 55,
}

SAMPLE_CONVERSATION = {"id": 9001, "body": "<p>Looking into it</p>", "private": True}


@pytest.fixture
def fd_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"credentials": {"api_key": API_KEY, "domain": DOMAIN}}
    return ctx


def minimal_ticket(**overrides):
    inputs = {"subject": "Cannot log in", "email": "user@example.com"}
    inputs.update(overrides)
    return inputs


# ---- create_ticket ----


class TestCreateTicket:
    @pytest.mark.asyncio
    async def test_creates_ticket(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_TICKET

        result = await CreateTicketAction().execute(minimal_ticket(), fd_context)

        assert result["result"] is True
        assert result["ticket"]["id"] == TICKET_ID

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateTicketAction().execute(minimal_ticket(), fd_context)

        assert fd_context.fetch.call_args.args[0] == f"{BASE_URL}/tickets"
        assert fd_context.fetch.call_args.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_minimal_body_is_subject_and_email(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateTicketAction().execute(minimal_ticket(), fd_context)

        assert fd_context.fetch.call_args.kwargs["json"] == {
            "subject": "Cannot log in",
            "email": "user@example.com",
        }

    @pytest.mark.asyncio
    async def test_numeric_enums_are_presence_gated_not_truthiness_gated(self, fd_context):
        """priority, status, source and company_id use `in inputs`, so a
        legitimate 0 survives. Freshdesk enums start at 1, but the distinction
        matters: truthiness gating here would silently drop valid values."""
        fd_context.fetch.return_value = {}

        await CreateTicketAction().execute(
            minimal_ticket(priority=0, status=0, source=0, company_id=0), fd_context
        )

        body = fd_context.fetch.call_args.kwargs["json"]
        assert body["priority"] == 0
        assert body["status"] == 0
        assert body["source"] == 0
        assert body["company_id"] == 0

    @pytest.mark.asyncio
    async def test_text_fields_are_truthiness_gated(self, fd_context):
        """description, name and tags are dropped when empty -- unlike the
        numeric fields above."""
        fd_context.fetch.return_value = {}

        await CreateTicketAction().execute(
            minimal_ticket(description="", name="", tags=[]), fd_context
        )

        body = fd_context.fetch.call_args.kwargs["json"]
        assert "description" not in body
        assert "name" not in body
        assert "tags" not in body

    @pytest.mark.asyncio
    async def test_all_optional_fields_forwarded(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateTicketAction().execute(
            minimal_ticket(
                description="Details here",
                priority=3,
                status=2,
                source=1,
                name="Ada",
                company_id=42,
                tags=["login", "urgent"],
            ),
            fd_context,
        )

        body = fd_context.fetch.call_args.kwargs["json"]
        assert body["description"] == "Details here"
        assert body["priority"] == 3
        assert body["status"] == 2
        assert body["source"] == 1
        assert body["name"] == "Ada"
        assert body["company_id"] == 42
        assert body["tags"] == ["login", "urgent"]

    @pytest.mark.parametrize("missing", ["subject", "email"])
    @pytest.mark.asyncio
    async def test_both_required_inputs_are_enforced(self, fd_context, missing):
        inputs = minimal_ticket()
        del inputs[missing]

        result = await CreateTicketAction().execute(inputs, fd_context)

        assert result["result"] is False
        assert result["ticket"] == {}
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 400: email is invalid")

        result = await CreateTicketAction().execute(minimal_ticket(), fd_context)

        assert result["result"] is False
        assert "email is invalid" in result["error"]


# ---- list_tickets ----


class TestListTickets:
    @pytest.mark.asyncio
    async def test_returns_tickets_and_total(self, fd_context):
        fd_context.fetch.return_value = [SAMPLE_TICKET]

        result = await ListTicketsAction().execute({}, fd_context)

        assert result["result"] is True
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_request_url_and_default_pagination(self, fd_context):
        fd_context.fetch.return_value = []

        await ListTicketsAction().execute({}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/tickets"
        assert call.kwargs["method"] == "GET"
        assert call.kwargs["params"] == {"page": 1, "per_page": 30}

    @pytest.mark.asyncio
    async def test_explicit_pagination_forwarded(self, fd_context):
        fd_context.fetch.return_value = []

        await ListTicketsAction().execute({"page": 2, "per_page": 50}, fd_context)

        assert fd_context.fetch.call_args.kwargs["params"] == {"page": 2, "per_page": 50}

    @pytest.mark.asyncio
    async def test_non_list_response_yields_empty(self, fd_context):
        fd_context.fetch.return_value = {"message": "unexpected"}

        result = await ListTicketsAction().execute({}, fd_context)

        assert result["tickets"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 429: rate limited")

        result = await ListTicketsAction().execute({}, fd_context)

        assert result["result"] is False
        assert result["total"] == 0


# ---- get_ticket ----


class TestGetTicket:
    @pytest.mark.asyncio
    async def test_returns_ticket(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_TICKET

        result = await GetTicketAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert result["result"] is True
        assert result["ticket"]["subject"] == "Cannot log in"

    @pytest.mark.asyncio
    async def test_request_url_includes_ticket_id(self, fd_context):
        fd_context.fetch.return_value = {}

        await GetTicketAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert fd_context.fetch.call_args.args[0] == f"{BASE_URL}/tickets/{TICKET_ID}"
        assert fd_context.fetch.call_args.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_missing_ticket_id_is_captured(self, fd_context):
        result = await GetTicketAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 404: Ticket not found")

        result = await GetTicketAction().execute({"ticket_id": 999}, fd_context)

        assert result["result"] is False
        assert result["ticket"] == {}


# ---- update_ticket ----


class TestUpdateTicket:
    @pytest.mark.asyncio
    async def test_request_uses_put_to_ticket_id(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_TICKET

        await UpdateTicketAction().execute({"ticket_id": TICKET_ID, "status": 4}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/tickets/{TICKET_ID}"
        assert call.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, fd_context):
        fd_context.fetch.return_value = {}

        await UpdateTicketAction().execute({"ticket_id": TICKET_ID, "status": 5}, fd_context)

        assert fd_context.fetch.call_args.kwargs["json"] == {"status": 5}

    @pytest.mark.asyncio
    async def test_status_zero_survives(self, fd_context):
        """status and priority are presence-gated, so 0 is forwarded."""
        fd_context.fetch.return_value = {}

        await UpdateTicketAction().execute(
            {"ticket_id": TICKET_ID, "status": 0, "priority": 0}, fd_context
        )

        body = fd_context.fetch.call_args.kwargs["json"]
        assert body["status"] == 0
        assert body["priority"] == 0

    @pytest.mark.asyncio
    async def test_empty_subject_cannot_be_set(self, fd_context):
        """subject and description are truthiness-gated, so they can't be cleared."""
        fd_context.fetch.return_value = {}

        await UpdateTicketAction().execute(
            {"ticket_id": TICKET_ID, "subject": "", "description": ""}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_ticket_id_not_duplicated_into_body(self, fd_context):
        fd_context.fetch.return_value = {}

        await UpdateTicketAction().execute({"ticket_id": TICKET_ID, "status": 3}, fd_context)

        assert "ticket_id" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_tags_replace_wholesale(self, fd_context):
        """Freshdesk replaces the tag list rather than appending, so sending a
        partial list silently drops the rest."""
        fd_context.fetch.return_value = {}

        await UpdateTicketAction().execute(
            {"ticket_id": TICKET_ID, "tags": ["billing"]}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"]["tags"] == ["billing"]

    @pytest.mark.asyncio
    async def test_missing_ticket_id_is_captured(self, fd_context):
        result = await UpdateTicketAction().execute({"status": 3}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 400: invalid status")

        result = await UpdateTicketAction().execute(
            {"ticket_id": TICKET_ID, "status": 99}, fd_context
        )

        assert result["result"] is False


# ---- delete_ticket ----


class TestDeleteTicket:
    @pytest.mark.asyncio
    async def test_reports_success(self, fd_context):
        fd_context.fetch.return_value = None

        result = await DeleteTicketAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert result == {"result": True}

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, fd_context):
        fd_context.fetch.return_value = None

        await DeleteTicketAction().execute({"ticket_id": TICKET_ID}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/tickets/{TICKET_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_sends_no_body(self, fd_context):
        fd_context.fetch.return_value = None

        await DeleteTicketAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert "json" not in fd_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_missing_ticket_id_is_captured(self, fd_context):
        result = await DeleteTicketAction().execute({}, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 403: forbidden")

        result = await DeleteTicketAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert result["result"] is False
        assert "403" in result["error"]


# ---- list_conversations ----


class TestListConversations:
    @pytest.mark.asyncio
    async def test_returns_conversations(self, fd_context):
        fd_context.fetch.return_value = [SAMPLE_CONVERSATION]

        result = await ListConversationsAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert result["result"] is True
        assert result["conversations"] == [SAMPLE_CONVERSATION]

    @pytest.mark.asyncio
    async def test_request_targets_ticket_conversations(self, fd_context):
        fd_context.fetch.return_value = []

        await ListConversationsAction().execute({"ticket_id": TICKET_ID}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/tickets/{TICKET_ID}/conversations"
        assert call.kwargs["method"] == "GET"

    @pytest.mark.asyncio
    async def test_no_total_is_reported(self, fd_context):
        """Unlike the list actions, conversations returns no `total` key."""
        fd_context.fetch.return_value = [SAMPLE_CONVERSATION]

        result = await ListConversationsAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert "total" not in result

    @pytest.mark.asyncio
    async def test_no_pagination_is_sent(self, fd_context):
        """The endpoint is unpaginated here, so a ticket with many replies
        returns whatever the API defaults to."""
        fd_context.fetch.return_value = []

        await ListConversationsAction().execute(
            {"ticket_id": TICKET_ID, "page": 2}, fd_context
        )

        assert "params" not in fd_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_non_list_response_yields_empty(self, fd_context):
        fd_context.fetch.return_value = {"message": "unexpected"}

        result = await ListConversationsAction().execute({"ticket_id": TICKET_ID}, fd_context)

        assert result["conversations"] == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 404")

        result = await ListConversationsAction().execute({"ticket_id": 999}, fd_context)

        assert result["result"] is False
        assert result["conversations"] == []


# ---- create_note ----


class TestCreateNote:
    @pytest.mark.asyncio
    async def test_creates_note(self, fd_context):
        fd_context.fetch.return_value = SAMPLE_CONVERSATION

        result = await CreateNoteAction().execute(
            {"ticket_id": TICKET_ID, "body": "Internal note"}, fd_context
        )

        assert result["result"] is True
        assert result["conversation"]["id"] == 9001

    @pytest.mark.asyncio
    async def test_request_targets_notes_endpoint(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateNoteAction().execute({"ticket_id": TICKET_ID, "body": "n"}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/tickets/{TICKET_ID}/notes"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_note_is_always_private(self, fd_context):
        """`private` is hardcoded True -- a note must never become customer
        visible. This is the highest-consequence assertion in this file."""
        fd_context.fetch.return_value = {}

        await CreateNoteAction().execute({"ticket_id": TICKET_ID, "body": "n"}, fd_context)

        assert fd_context.fetch.call_args.kwargs["json"]["private"] is True

    @pytest.mark.asyncio
    async def test_private_cannot_be_overridden_by_input(self, fd_context):
        """Even an explicit private=False input must not leak the note."""
        fd_context.fetch.return_value = {}

        await CreateNoteAction().execute(
            {"ticket_id": TICKET_ID, "body": "n", "private": False}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"]["private"] is True

    @pytest.mark.asyncio
    async def test_notify_emails_forwarded(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateNoteAction().execute(
            {"ticket_id": TICKET_ID, "body": "n", "notify_emails": ["a@x.com"]}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"]["notify_emails"] == ["a@x.com"]

    @pytest.mark.asyncio
    async def test_empty_notify_emails_dropped(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateNoteAction().execute(
            {"ticket_id": TICKET_ID, "body": "n", "notify_emails": []}, fd_context
        )

        assert "notify_emails" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.parametrize("missing", ["ticket_id", "body"])
    @pytest.mark.asyncio
    async def test_required_inputs_enforced(self, fd_context, missing):
        inputs = {"ticket_id": TICKET_ID, "body": "n"}
        del inputs[missing]

        result = await CreateNoteAction().execute(inputs, fd_context)

        assert result["result"] is False
        fd_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 400")

        result = await CreateNoteAction().execute(
            {"ticket_id": TICKET_ID, "body": "n"}, fd_context
        )

        assert result["result"] is False
        assert result["conversation"] == {}


# ---- create_reply ----


class TestCreateReply:
    @pytest.mark.asyncio
    async def test_creates_reply(self, fd_context):
        fd_context.fetch.return_value = {"id": 9002, "body": "<p>Fixed</p>"}

        result = await CreateReplyAction().execute(
            {"ticket_id": TICKET_ID, "body": "Fixed"}, fd_context
        )

        assert result["result"] is True
        assert result["conversation"]["id"] == 9002

    @pytest.mark.asyncio
    async def test_request_targets_reply_endpoint(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateReplyAction().execute({"ticket_id": TICKET_ID, "body": "r"}, fd_context)

        call = fd_context.fetch.call_args
        assert call.args[0] == f"{BASE_URL}/tickets/{TICKET_ID}/reply"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_reply_never_sets_private(self, fd_context):
        """A reply is public by definition -- the opposite of create_note. The
        `private` key must be absent entirely."""
        fd_context.fetch.return_value = {}

        await CreateReplyAction().execute({"ticket_id": TICKET_ID, "body": "r"}, fd_context)

        assert "private" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_minimal_body_is_body_only(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateReplyAction().execute({"ticket_id": TICKET_ID, "body": "r"}, fd_context)

        assert fd_context.fetch.call_args.kwargs["json"] == {"body": "r"}

    @pytest.mark.asyncio
    async def test_from_email_forwarded(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateReplyAction().execute(
            {"ticket_id": TICKET_ID, "body": "r", "from_email": "support@acme.com"}, fd_context
        )

        assert fd_context.fetch.call_args.kwargs["json"]["from_email"] == "support@acme.com"

    @pytest.mark.asyncio
    async def test_empty_from_email_dropped(self, fd_context):
        fd_context.fetch.return_value = {}

        await CreateReplyAction().execute(
            {"ticket_id": TICKET_ID, "body": "r", "from_email": ""}, fd_context
        )

        assert "from_email" not in fd_context.fetch.call_args.kwargs["json"]

    @pytest.mark.parametrize("missing", ["ticket_id", "body"])
    @pytest.mark.asyncio
    async def test_required_inputs_enforced(self, fd_context, missing):
        inputs = {"ticket_id": TICKET_ID, "body": "r"}
        del inputs[missing]

        result = await CreateReplyAction().execute(inputs, fd_context)

        assert result["result"] is False

    @pytest.mark.asyncio
    async def test_error_is_captured(self, fd_context):
        fd_context.fetch.side_effect = Exception("HTTP 400")

        result = await CreateReplyAction().execute(
            {"ticket_id": TICKET_ID, "body": "r"}, fd_context
        )

        assert result["result"] is False


# ---- Config ----


class TestFreshdeskTicketConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    def test_create_ticket_requires_subject_and_email(self, config):
        required = config["actions"]["create_ticket"]["input_schema"]["required"]
        assert sorted(required) == ["email", "subject"]

    @pytest.mark.parametrize(
        "action",
        ["get_ticket", "update_ticket", "delete_ticket", "list_conversations", "create_note", "create_reply"],
    )
    def test_ticket_scoped_actions_require_ticket_id(self, config, action):
        assert "ticket_id" in config["actions"][action]["input_schema"]["required"]

    @pytest.mark.parametrize("action", ["create_note", "create_reply"])
    def test_conversation_actions_require_body(self, config, action):
        assert "body" in config["actions"][action]["input_schema"]["required"]

    def test_list_tickets_exposes_pagination(self, config):
        props = config["actions"]["list_tickets"]["input_schema"]["properties"]
        assert "page" in props
        assert "per_page" in props
