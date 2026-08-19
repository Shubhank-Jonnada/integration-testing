"""Unit tests for the GitHub pull request actions.

Covers `list_pull_requests`, `get_pull_request`, `create_pull_request`,
`merge_pull_request`, the three reviewer actions, and
`create_pull_request_review`.

Two things get particular attention:

- `list_pull_requests` is the only list action built on the Search API, so it
  assembles a `q` query string rather than sending normal filter params.
- `merge_pull_request` is irreversible, and `merge_method` decides whether
  history is preserved, squashed, or rebased.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from github.github import (  # noqa: E402
    AddPullRequestReviewers,
    CreatePullRequest,
    CreatePullRequestReview,
    GetPullRequest,
    ListPullRequestReviewers,
    ListPullRequests,
    MergePullRequest,
    RemovePullRequestReviewers,
)

pytestmark = pytest.mark.unit

TOKEN = "gho_testtoken1234567890"  # nosec B105
BASE = "https://api.github.com"
OWNER = "octocat"
REPO = "hello-world"
PULL_NUMBER = 42
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

SEARCH_ITEM = {
    "number": PULL_NUMBER,
    "title": "Add feature",
    "state": "open",
    "html_url": f"https://github.com/{OWNER}/{REPO}/pull/{PULL_NUMBER}",
    "user": {"login": OWNER, "avatar_url": "https://avatars/1"},
    "created_at": "2026-01-26T19:01:12Z",
    "updated_at": "2026-01-26T19:01:12Z",
}

MERGE_RESULT = {
    "sha": "6dcb09b5b57875f334f61aebed695e2e4193db5e",
    "merged": True,
    "message": "Pull Request successfully merged",
}


@pytest.fixture
def gh_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": TOKEN}}
    return ctx


def search_page(items):
    return {"items": items, "total_count": len(items)}


# ---- list_pull_requests ----


class TestListPullRequests:
    @pytest.mark.asyncio
    async def test_uses_the_search_api_not_the_pulls_endpoint(self, gh_context):
        """This is the only list action built on /search/issues -- it trades exact
        pagination for cross-cutting query support, and is subject to the Search
        API's separate, much lower rate limit."""
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/search/issues"

    @pytest.mark.asyncio
    async def test_query_scopes_to_prs_in_the_repo(self, gh_context):
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute({"owner": OWNER, "repo": REPO}, gh_context)

        q = gh_context.fetch.call_args.kwargs["params"]["q"]
        assert f"is:pr repo:{OWNER}/{REPO}" in q

    @pytest.mark.asyncio
    async def test_state_all_adds_no_state_qualifier(self, gh_context):
        """`all` is the default and means "don't filter" -- neither is:open nor
        is:closed is appended."""
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute({"owner": OWNER, "repo": REPO}, gh_context)

        q = gh_context.fetch.call_args.kwargs["params"]["q"]
        assert "is:open" not in q
        assert "is:closed" not in q

    @pytest.mark.parametrize("state, qualifier", [("open", "is:open"), ("closed", "is:closed")])
    @pytest.mark.asyncio
    async def test_state_becomes_a_query_qualifier(self, gh_context, state, qualifier):
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute(
            {"owner": OWNER, "repo": REPO, "state": state}, gh_context
        )

        assert qualifier in gh_context.fetch.call_args.kwargs["params"]["q"]

    @pytest.mark.asyncio
    async def test_author_and_date_range_become_qualifiers(self, gh_context):
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "author": "hubot",
                "after": "2026-01-01",
                "before": "2026-02-01",
            },
            gh_context,
        )

        q = gh_context.fetch.call_args.kwargs["params"]["q"]
        assert "author:hubot" in q
        assert "created:>=2026-01-01" in q
        assert "created:<=2026-02-01" in q

    @pytest.mark.asyncio
    async def test_popularity_sort_maps_to_comments(self, gh_context):
        """The Search API has no `popularity` sort, so it is translated."""
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute(
            {"owner": OWNER, "repo": REPO, "sort": "popularity"}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"]["sort"] == "comments"

    @pytest.mark.asyncio
    async def test_long_running_sort_maps_to_created(self, gh_context):
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute(
            {"owner": OWNER, "repo": REPO, "sort": "long-running"}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"]["sort"] == "created"

    @pytest.mark.asyncio
    async def test_unknown_sort_falls_back_to_updated(self, gh_context):
        """An unmapped sort value must not be forwarded verbatim -- the Search API
        would reject it with a 422."""
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute(
            {"owner": OWNER, "repo": REPO, "sort": "nonsense"}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"]["sort"] == "updated"

    @pytest.mark.asyncio
    async def test_limit_caps_per_page_at_one_hundred(self, gh_context):
        gh_context.fetch.return_value = search_page([])

        await ListPullRequests().execute(
            {"owner": OWNER, "repo": REPO, "limit": 500}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 100

    @pytest.mark.asyncio
    async def test_small_limit_reduces_per_page(self, gh_context):
        gh_context.fetch.return_value = search_page([SEARCH_ITEM])

        await ListPullRequests().execute({"owner": OWNER, "repo": REPO, "limit": 5}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 5

    @pytest.mark.asyncio
    async def test_results_are_truncated_to_the_limit(self, gh_context):
        """The walk stops as soon as enough results accumulate, then slices."""
        gh_context.fetch.return_value = search_page([SEARCH_ITEM] * 5)

        result = await ListPullRequests().execute(
            {"owner": OWNER, "repo": REPO, "limit": 3}, gh_context
        )

        assert len(result.data) == 3

    @pytest.mark.asyncio
    async def test_empty_results(self, gh_context):
        gh_context.fetch.return_value = search_page([])

        result = await ListPullRequests().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 403: search rate limit exceeded")

        result = await ListPullRequests().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False
        assert "rate limit" in result.data["error"]


# ---- get_pull_request ----


class TestGetPullRequest:
    @pytest.mark.asyncio
    async def test_request_url_uses_pulls_not_issues(self, gh_context):
        """A PR is readable via /issues too, but only /pulls returns the diff
        metadata (mergeable, head, base)."""
        gh_context.fetch.return_value = {
            "number": PULL_NUMBER,
            "title": "Add feature",
            "state": "open",
            "html_url": SEARCH_ITEM["html_url"],
        }

        await GetPullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}"
        )

    @pytest.mark.parametrize("missing", ["owner", "repo", "pull_number"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}
        del inputs[missing]

        result = await GetPullRequest().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Not Found")

        result = await GetPullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": 9999}, gh_context
        )

        assert result.data["result"] is False


# ---- create_pull_request ----


class TestCreatePullRequest:
    def base_inputs(self, **overrides):
        inputs = {
            "owner": OWNER,
            "repo": REPO,
            "title": "Add feature",
            "head": "feature-branch",
            "base": "main",
        }
        inputs.update(overrides)
        return inputs

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, gh_context):
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(self.base_inputs(), gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/pulls"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_head_and_base_are_not_swapped(self, gh_context):
        """head is the source branch, base is the target. Swapping them opens a
        PR in the opposite direction -- which looks plausible and merges the
        wrong way."""
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(self.base_inputs(), gh_context)

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["head"] == "feature-branch"
        assert body["base"] == "main"

    @pytest.mark.asyncio
    async def test_draft_defaults_to_false(self, gh_context):
        """So an unqualified create opens a review-ready PR and notifies
        reviewers."""
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["draft"] is False

    @pytest.mark.asyncio
    async def test_draft_true_is_forwarded(self, gh_context):
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(self.base_inputs(draft=True), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["draft"] is True

    @pytest.mark.asyncio
    async def test_maintainer_can_modify_defaults_to_true(self, gh_context):
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["maintainer_can_modify"] is True

    @pytest.mark.asyncio
    async def test_maintainer_can_modify_false_is_sent(self, gh_context):
        """Always present in the body, so False is transmitted rather than
        omitted."""
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(
            self.base_inputs(maintainer_can_modify=False), gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"]["maintainer_can_modify"] is False

    @pytest.mark.asyncio
    async def test_body_is_optional(self, gh_context):
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(self.base_inputs(), gh_context)
        assert "body" not in gh_context.fetch.call_args.kwargs["json"]

        await CreatePullRequest().execute(self.base_inputs(body="Describes it"), gh_context)
        assert gh_context.fetch.call_args.kwargs["json"]["body"] == "Describes it"

    @pytest.mark.asyncio
    async def test_cross_fork_head_is_passed_through(self, gh_context):
        """A fork PR uses `owner:branch` as head; the handler must not mangle it."""
        gh_context.fetch.return_value = SEARCH_ITEM

        await CreatePullRequest().execute(
            self.base_inputs(head="contributor:feature"), gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"]["head"] == "contributor:feature"

    @pytest.mark.parametrize("missing", ["owner", "repo", "title", "head", "base"])
    @pytest.mark.asyncio
    async def test_all_five_inputs_are_required(self, gh_context, missing):
        inputs = self.base_inputs()
        del inputs[missing]

        result = await CreatePullRequest().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 422: No commits between branches")

        result = await CreatePullRequest().execute(self.base_inputs(), gh_context)

        assert result.data["result"] is False
        assert "No commits" in result.data["error"]


# ---- merge_pull_request ----


class TestMergePullRequest:
    @pytest.mark.asyncio
    async def test_request_uses_put_to_merge_subpath(self, gh_context):
        gh_context.fetch.return_value = MERGE_RESULT

        await MergePullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}/merge"
        assert call.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_merge_method_defaults_to_merge(self, gh_context):
        """The default preserves individual commits. `squash` and `rebase` rewrite
        history irreversibly, so the default matters."""
        gh_context.fetch.return_value = MERGE_RESULT

        await MergePullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"]["merge_method"] == "merge"

    @pytest.mark.parametrize("method", ["merge", "squash", "rebase"])
    @pytest.mark.asyncio
    async def test_all_merge_methods_are_forwarded(self, gh_context, method):
        gh_context.fetch.return_value = MERGE_RESULT

        await MergePullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER, "merge_method": method},
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"]["merge_method"] == method

    @pytest.mark.asyncio
    async def test_commit_title_and_message_forwarded(self, gh_context):
        gh_context.fetch.return_value = MERGE_RESULT

        await MergePullRequest().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "pull_number": PULL_NUMBER,
                "commit_title": "Merge feature",
                "commit_message": "Detail",
            },
            gh_context,
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["commit_title"] == "Merge feature"
        assert body["commit_message"] == "Detail"

    @pytest.mark.asyncio
    async def test_omitted_commit_fields_let_github_generate_them(self, gh_context):
        gh_context.fetch.return_value = MERGE_RESULT

        await MergePullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body == {"merge_method": "merge"}

    @pytest.mark.asyncio
    async def test_response_prefers_the_supplied_commit_title(self, gh_context):
        """The result echoes the caller's title when given, else GitHub's."""
        gh_context.fetch.return_value = {**MERGE_RESULT, "commit_title": "Generated"}

        result = await MergePullRequest().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "pull_number": PULL_NUMBER,
                "commit_title": "Mine",
            },
            gh_context,
        )

        assert result.data["commit_title"] == "Mine"

    @pytest.mark.asyncio
    async def test_response_falls_back_to_github_commit_title(self, gh_context):
        gh_context.fetch.return_value = {**MERGE_RESULT, "commit_title": "Generated"}

        result = await MergePullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        assert result.data["commit_title"] == "Generated"

    @pytest.mark.asyncio
    async def test_merge_conflict_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 405: Pull Request is not mergeable")

        result = await MergePullRequest().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        assert result.data["result"] is False
        assert "not mergeable" in result.data["error"]

    @pytest.mark.parametrize("missing", ["owner", "repo", "pull_number"])
    @pytest.mark.asyncio
    async def test_required_inputs_prevent_the_merge(self, gh_context, missing):
        """Merging is irreversible, so a missing input must not produce a request
        against a malformed path."""
        inputs = {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}
        del inputs[missing]

        result = await MergePullRequest().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- Reviewer actions ----


REVIEWER_ACTIONS = [
    (AddPullRequestReviewers, "POST"),
    (RemovePullRequestReviewers, "DELETE"),
]


class TestReviewerActions:
    @pytest.mark.parametrize("action_cls, method", REVIEWER_ACTIONS)
    @pytest.mark.asyncio
    async def test_same_url_different_verb(self, gh_context, action_cls, method):
        """Add and remove share the requested_reviewers URL and differ only by
        method, so the verb is the only thing preventing the opposite effect."""
        gh_context.fetch.return_value = {"requested_reviewers": []}

        await action_cls().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER, "reviewers": ["hubot"]},
            gh_context,
        )

        call = gh_context.fetch.call_args
        assert (
            call.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}/requested_reviewers"
        )
        assert call.kwargs["method"] == method

    @pytest.mark.parametrize("action_cls, _method", REVIEWER_ACTIONS)
    @pytest.mark.asyncio
    async def test_user_reviewers_forwarded(self, gh_context, action_cls, _method):
        gh_context.fetch.return_value = {"requested_reviewers": []}

        await action_cls().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "pull_number": PULL_NUMBER,
                "reviewers": ["hubot", "octocat"],
            },
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"]["reviewers"] == ["hubot", "octocat"]

    @pytest.mark.parametrize("action_cls, _method", REVIEWER_ACTIONS)
    @pytest.mark.asyncio
    async def test_team_reviewers_are_a_separate_field(self, gh_context, action_cls, _method):
        """Teams cannot be mixed into the `reviewers` list -- GitHub requires the
        distinct `team_reviewers` key."""
        gh_context.fetch.return_value = {"requested_reviewers": []}

        await action_cls().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "pull_number": PULL_NUMBER,
                "team_reviewers": ["platform-team"],
            },
            gh_context,
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["team_reviewers"] == ["platform-team"]
        assert "reviewers" not in body

    @pytest.mark.parametrize("action_cls, _method", REVIEWER_ACTIONS)
    @pytest.mark.asyncio
    async def test_empty_lists_produce_an_empty_body(self, gh_context, action_cls, _method):
        gh_context.fetch.return_value = {"requested_reviewers": []}

        await action_cls().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "pull_number": PULL_NUMBER,
                "reviewers": [],
                "team_reviewers": [],
            },
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.parametrize("action_cls, _method", REVIEWER_ACTIONS)
    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context, action_cls, _method):
        gh_context.fetch.side_effect = Exception("HTTP 422: Reviews may only be requested")

        result = await action_cls().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER, "reviewers": ["x"]},
            gh_context,
        )

        assert result.data["result"] is False


class TestListPullRequestReviewers:
    @pytest.mark.asyncio
    async def test_uses_get_on_the_same_url(self, gh_context):
        gh_context.fetch.return_value = {"users": [], "teams": []}

        await ListPullRequestReviewers().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        call = gh_context.fetch.call_args
        assert (
            call.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}/requested_reviewers"
        )
        assert "method" not in call.kwargs

    @pytest.mark.asyncio
    async def test_sends_no_body(self, gh_context):
        gh_context.fetch.return_value = {"users": [], "teams": []}

        await ListPullRequestReviewers().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        assert "json" not in gh_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_missing_pull_number_is_captured(self, gh_context):
        result = await ListPullRequestReviewers().execute(
            {"owner": OWNER, "repo": REPO}, gh_context
        )

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- create_pull_request_review ----


class TestCreatePullRequestReview:
    @pytest.mark.asyncio
    async def test_request_url_and_method(self, gh_context):
        gh_context.fetch.return_value = {"id": 1, "state": "APPROVED"}

        await CreatePullRequestReview().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER, "event": "APPROVE"},
            gh_context,
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/pulls/{PULL_NUMBER}/reviews"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.parametrize("event", ["APPROVE", "REQUEST_CHANGES", "COMMENT"])
    @pytest.mark.asyncio
    async def test_all_review_events_forwarded(self, gh_context, event):
        """APPROVE can satisfy a branch protection rule, so the event value is
        security-relevant."""
        gh_context.fetch.return_value = {"id": 1}

        await CreatePullRequestReview().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER, "event": event},
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"]["event"] == event

    @pytest.mark.asyncio
    async def test_omitted_event_creates_a_pending_review(self, gh_context):
        """With no event GitHub stores the review as PENDING rather than
        submitting it."""
        gh_context.fetch.return_value = {"id": 1, "state": "PENDING"}

        await CreatePullRequestReview().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER, "body": "Notes"},
            gh_context,
        )

        assert "event" not in gh_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_inline_comments_forwarded(self, gh_context):
        gh_context.fetch.return_value = {"id": 1}

        comments = [{"path": "a.py", "line": 10, "body": "Nit"}]

        await CreatePullRequestReview().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "pull_number": PULL_NUMBER,
                "event": "COMMENT",
                "comments": comments,
            },
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"]["comments"] == comments

    @pytest.mark.asyncio
    async def test_all_fields_optional_yields_empty_body(self, gh_context):
        gh_context.fetch.return_value = {"id": 1}

        await CreatePullRequestReview().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception(
            "HTTP 422: Can not approve your own pull request"
        )

        result = await CreatePullRequestReview().execute(
            {"owner": OWNER, "repo": REPO, "pull_number": PULL_NUMBER, "event": "APPROVE"},
            gh_context,
        )

        assert result.data["result"] is False
        assert "own pull request" in result.data["error"]


# ---- Config ----


class TestGithubPullRequestConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        [
            "list_pull_requests",
            "get_pull_request",
            "create_pull_request",
            "merge_pull_request",
            "add_pull_request_reviewers",
            "remove_pull_request_reviewers",
            "list_pull_request_reviewers",
            "create_pull_request_review",
        ],
    )
    def test_all_require_owner_and_repo(self, config, action):
        required = config["actions"][action]["input_schema"]["required"]
        assert "owner" in required
        assert "repo" in required

    def test_create_pull_request_requires_title_head_and_base(self, config):
        required = config["actions"]["create_pull_request"]["input_schema"]["required"]

        assert "title" in required
        assert "head" in required
        assert "base" in required

    @pytest.mark.parametrize(
        "action",
        [
            "get_pull_request",
            "merge_pull_request",
            "add_pull_request_reviewers",
            "remove_pull_request_reviewers",
            "list_pull_request_reviewers",
            "create_pull_request_review",
        ],
    )
    def test_pr_scoped_actions_require_pull_number(self, config, action):
        assert "pull_number" in config["actions"][action]["input_schema"]["required"]

    def test_merge_method_is_constrained_to_valid_values(self, config):
        """An arbitrary merge_method would be rejected by GitHub, so the schema
        should enumerate the three valid options."""
        props = config["actions"]["merge_pull_request"]["input_schema"]["properties"]
        assert set(props["merge_method"]["enum"]) == {"merge", "squash", "rebase"}
