"""Unit tests for the GitHub commit and issue actions.

Covers `list_commits`, `get_commit`, `list_issues`, `get_issue`, `create_issue`,
`update_issue`, `create_issue_comment`, and `get_issue_comments`.

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from github.github import (  # noqa: E402
    CreateIssue,
    CreateIssueComment,
    GetCommit,
    GetIssue,
    GetIssueComments,
    ListCommits,
    ListIssues,
    UpdateIssue,
)

pytestmark = pytest.mark.unit

TOKEN = "gho_testtoken1234567890"  # nosec B105
BASE = "https://api.github.com"
OWNER = "octocat"
REPO = "hello-world"
SHA = "6dcb09b5b57875f334f61aebed695e2e4193db5e"
ISSUE_NUMBER = 1347
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

COMMIT = {
    "sha": SHA,
    "html_url": f"https://github.com/{OWNER}/{REPO}/commit/{SHA}",
    "commit": {
        "author": {"name": "Monalisa Octocat", "email": "mona@github.com", "date": "2026-01-26T19:01:12Z"},
        "committer": {"name": "GitHub", "email": "noreply@github.com", "date": "2026-01-26T19:01:12Z"},
        "message": "Fix all the bugs",
    },
    "stats": {"total": 10, "additions": 7, "deletions": 3},
    "files": [{"filename": "a.py", "status": "modified"}],
}

ISSUE = {
    "number": ISSUE_NUMBER,
    "title": "Found a bug",
    "body": "It broke",
    "state": "open",
    "created_at": "2026-01-26T19:01:12Z",
    "updated_at": "2026-01-26T19:01:12Z",
    "closed_at": None,
    "html_url": f"https://github.com/{OWNER}/{REPO}/issues/{ISSUE_NUMBER}",
    "user": {"login": OWNER, "avatar_url": "https://avatars.githubusercontent.com/u/1"},
    "assignees": [{"login": "hubot"}],
    "labels": [{"name": "bug", "color": "d73a4a"}],
}


@pytest.fixture
def gh_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": TOKEN}}
    return ctx


# ---- list_commits ----


class TestListCommits:
    @pytest.mark.asyncio
    async def test_returns_flattened_commits(self, gh_context):
        """The nested commit.author/committer structure is flattened one level."""
        gh_context.fetch.return_value = [COMMIT]

        result = await ListCommits().execute({"owner": OWNER, "repo": REPO}, gh_context)

        entry = result.data[0]
        assert entry["sha"] == SHA
        assert entry["author"]["name"] == "Monalisa Octocat"
        assert entry["message"] == "Fix all the bugs"

    @pytest.mark.asyncio
    async def test_author_and_committer_are_distinct(self, gh_context):
        """They differ on any commit made through the web UI or a bot, so the two
        must not be collapsed."""
        gh_context.fetch.return_value = [COMMIT]

        result = await ListCommits().execute({"owner": OWNER, "repo": REPO}, gh_context)

        entry = result.data[0]
        assert entry["author"]["name"] == "Monalisa Octocat"
        assert entry["committer"]["name"] == "GitHub"

    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = []

        await ListCommits().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/commits"

    @pytest.mark.asyncio
    async def test_no_filters_sends_only_pagination(self, gh_context):
        gh_context.fetch.return_value = []

        await ListCommits().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"] == {"per_page": 100, "page": 1}

    @pytest.mark.asyncio
    async def test_all_filters_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListCommits().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "sha": "main",
                "path": "src/app.py",
                "since": "2026-01-01T00:00:00Z",
                "until": "2026-02-01T00:00:00Z",
            },
            gh_context,
        )

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["sha"] == "main"
        assert params["path"] == "src/app.py"
        assert params["since"] == "2026-01-01T00:00:00Z"
        assert params["until"] == "2026-02-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_sha_filter_accepts_a_branch_name(self, gh_context):
        """GitHub's `sha` param is really "sha or ref", so a branch name is
        valid -- worth documenting since the name suggests otherwise."""
        gh_context.fetch.return_value = []

        await ListCommits().execute({"owner": OWNER, "repo": REPO, "sha": "develop"}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["sha"] == "develop"

    @pytest.mark.asyncio
    async def test_returns_a_bare_list(self, gh_context):
        gh_context.fetch.return_value = []

        result = await ListCommits().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data == []

    @pytest.mark.asyncio
    async def test_malformed_commit_is_captured(self, gh_context):
        """The nested reads use [] throughout, so a commit missing its author
        surfaces as a captured error rather than a partial list."""
        gh_context.fetch.return_value = [{"sha": SHA, "commit": {}}]

        result = await ListCommits().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 409: Git Repository is empty")

        result = await ListCommits().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False
        assert "empty" in result.data["error"]


# ---- get_commit ----


class TestGetCommit:
    @pytest.mark.asyncio
    async def test_returns_commit_with_stats_and_files(self, gh_context):
        """Unlike list_commits, the single-commit response includes diff stats."""
        gh_context.fetch.return_value = COMMIT

        result = await GetCommit().execute({"owner": OWNER, "repo": REPO, "sha": SHA}, gh_context)

        assert result.data["stats"]["additions"] == 7
        assert result.data["files"][0]["filename"] == "a.py"

    @pytest.mark.asyncio
    async def test_request_url_includes_sha(self, gh_context):
        gh_context.fetch.return_value = COMMIT

        await GetCommit().execute({"owner": OWNER, "repo": REPO, "sha": SHA}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/commits/{SHA}"

    @pytest.mark.asyncio
    async def test_stats_and_files_default_to_empty(self, gh_context):
        """Both are read with .get(), so a merge commit without them still works."""
        payload = {k: v for k, v in COMMIT.items() if k not in ("stats", "files")}
        gh_context.fetch.return_value = payload

        result = await GetCommit().execute({"owner": OWNER, "repo": REPO, "sha": SHA}, gh_context)

        assert result.data["stats"] == {}
        assert result.data["files"] == []

    @pytest.mark.parametrize("missing", ["owner", "repo", "sha"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "sha": SHA}
        del inputs[missing]

        result = await GetCommit().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: No commit found")

        result = await GetCommit().execute(
            {"owner": OWNER, "repo": REPO, "sha": "deadbeef"}, gh_context
        )

        assert result.data["result"] is False


# ---- list_issues ----


class TestListIssues:
    @pytest.mark.asyncio
    async def test_returns_mapped_issues(self, gh_context):
        gh_context.fetch.return_value = [ISSUE]

        result = await ListIssues().execute({"owner": OWNER, "repo": REPO}, gh_context)

        entry = result.data[0]
        assert entry["number"] == ISSUE_NUMBER
        assert entry["title"] == "Found a bug"
        assert entry["author"]["login"] == OWNER

    @pytest.mark.asyncio
    async def test_body_is_renamed_to_description(self, gh_context):
        """GitHub calls it `body`; the action exposes `description`."""
        gh_context.fetch.return_value = [ISSUE]

        result = await ListIssues().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data[0]["description"] == "It broke"
        assert "body" not in result.data[0]

    @pytest.mark.asyncio
    async def test_defaults_to_all_states(self, gh_context):
        """GitHub's own default is `open`; this action deliberately widens it to
        `all`, so an unqualified list includes closed issues."""
        gh_context.fetch.return_value = []

        await ListIssues().execute({"owner": OWNER, "repo": REPO}, gh_context)

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["state"] == "all"
        assert params["sort"] == "created"
        assert params["direction"] == "desc"

    @pytest.mark.asyncio
    async def test_state_filter_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListIssues().execute(
            {"owner": OWNER, "repo": REPO, "state": "closed"}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"]["state"] == "closed"

    @pytest.mark.asyncio
    async def test_since_and_labels_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListIssues().execute(
            {"owner": OWNER, "repo": REPO, "since": "2026-01-01T00:00:00Z", "labels": "bug,urgent"},
            gh_context,
        )

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["since"] == "2026-01-01T00:00:00Z"
        assert params["labels"] == "bug,urgent"

    @pytest.mark.asyncio
    async def test_assignees_and_labels_are_reshaped(self, gh_context):
        """Only login is kept from assignees; name and color from labels."""
        gh_context.fetch.return_value = [ISSUE]

        result = await ListIssues().execute({"owner": OWNER, "repo": REPO}, gh_context)

        entry = result.data[0]
        assert entry["assignees"] == [{"login": "hubot"}]
        assert entry["labels"] == [{"name": "bug", "color": "d73a4a"}]

    @pytest.mark.asyncio
    async def test_missing_assignees_and_labels_default_to_empty(self, gh_context):
        payload = {k: v for k, v in ISSUE.items() if k not in ("assignees", "labels")}
        gh_context.fetch.return_value = [payload]

        result = await ListIssues().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data[0]["assignees"] == []
        assert result.data[0]["labels"] == []

    @pytest.mark.asyncio
    async def test_open_issue_has_null_closed_at(self, gh_context):
        gh_context.fetch.return_value = [ISSUE]

        result = await ListIssues().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data[0]["closed_at"] is None

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404")

        result = await ListIssues().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False


# ---- get_issue ----


class TestGetIssue:
    @pytest.mark.asyncio
    async def test_returns_issue(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        result = await GetIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}, gh_context
        )

        assert result.data["number"] == ISSUE_NUMBER
        assert result.data["state"] == "open"

    @pytest.mark.asyncio
    async def test_request_url_includes_issue_number(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        await GetIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}, gh_context
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/issues/{ISSUE_NUMBER}"
        )

    @pytest.mark.parametrize("missing", ["owner", "repo", "issue_number"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}
        del inputs[missing]

        result = await GetIssue().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_pull_requests_are_also_issues(self, gh_context):
        """GitHub returns PRs from the issues endpoint too -- the action doesn't
        filter them out, so a caller may receive PRs when listing issues."""
        gh_context.fetch.return_value = {**ISSUE, "pull_request": {"url": "..."}}

        result = await GetIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}, gh_context
        )

        assert result.data["number"] == ISSUE_NUMBER


# ---- create_issue ----


class TestCreateIssue:
    @pytest.mark.asyncio
    async def test_creates_issue(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        result = await CreateIssue().execute(
            {"owner": OWNER, "repo": REPO, "title": "Found a bug"}, gh_context
        )

        assert result.data["number"] == ISSUE_NUMBER

    @pytest.mark.asyncio
    async def test_request_url_method_and_minimal_body(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        await CreateIssue().execute(
            {"owner": OWNER, "repo": REPO, "title": "Found a bug"}, gh_context
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/issues"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["json"] == {"title": "Found a bug"}

    @pytest.mark.asyncio
    async def test_all_optional_fields_forwarded(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        await CreateIssue().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "title": "Found a bug",
                "body": "Details",
                "assignees": ["hubot"],
                "labels": ["bug"],
                "milestone": 3,
            },
            gh_context,
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["body"] == "Details"
        assert body["assignees"] == ["hubot"]
        assert body["labels"] == ["bug"]
        assert body["milestone"] == 3

    @pytest.mark.asyncio
    async def test_empty_collections_are_dropped(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        await CreateIssue().execute(
            {"owner": OWNER, "repo": REPO, "title": "T", "assignees": [], "labels": []},
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {"title": "T"}

    @pytest.mark.asyncio
    async def test_milestone_zero_is_dropped(self, gh_context):
        """create_issue gates milestone on truthiness, unlike update_issue which
        uses `is not None` -- so 0 behaves differently between the two."""
        gh_context.fetch.return_value = ISSUE

        await CreateIssue().execute(
            {"owner": OWNER, "repo": REPO, "title": "T", "milestone": 0}, gh_context
        )

        assert "milestone" not in gh_context.fetch.call_args.kwargs["json"]

    @pytest.mark.parametrize("missing", ["owner", "repo", "title"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "title": "T"}
        del inputs[missing]

        result = await CreateIssue().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 410: Issues are disabled")

        result = await CreateIssue().execute(
            {"owner": OWNER, "repo": REPO, "title": "T"}, gh_context
        )

        assert result.data["result"] is False
        assert "disabled" in result.data["error"]


# ---- update_issue ----


class TestUpdateIssue:
    @pytest.mark.asyncio
    async def test_request_uses_patch(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        await UpdateIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "title": "New"},
            gh_context,
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/issues/{ISSUE_NUMBER}"
        assert call.kwargs["method"] == "PATCH"

    @pytest.mark.asyncio
    async def test_only_supplied_fields_are_sent(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        await UpdateIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "state": "closed"},
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {"state": "closed"}

    @pytest.mark.asyncio
    async def test_closing_an_issue(self, gh_context):
        gh_context.fetch.return_value = {**ISSUE, "state": "closed"}

        result = await UpdateIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "state": "closed"},
            gh_context,
        )

        assert result.data["state"] == "closed"

    @pytest.mark.asyncio
    async def test_milestone_zero_is_sent_on_update(self, gh_context):
        """update_issue guards milestone on `is not None`, so 0 reaches the API --
        the opposite of create_issue. Both are asserted so the inconsistency is
        documented rather than assumed."""
        gh_context.fetch.return_value = ISSUE

        await UpdateIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "milestone": 0},
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"]["milestone"] == 0

    @pytest.mark.asyncio
    async def test_empty_body_cannot_clear_the_description(self, gh_context):
        """body is truthiness-gated, so "" is dropped rather than clearing it."""
        gh_context.fetch.return_value = ISSUE

        await UpdateIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "body": ""}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_assignees_and_labels_replace_wholesale(self, gh_context):
        """GitHub replaces both lists, so a partial update removes the rest."""
        gh_context.fetch.return_value = ISSUE

        await UpdateIssue().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "issue_number": ISSUE_NUMBER,
                "assignees": ["newuser"],
                "labels": ["wontfix"],
            },
            gh_context,
        )

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["assignees"] == ["newuser"]
        assert body["labels"] == ["wontfix"]

    @pytest.mark.asyncio
    async def test_id_only_update_sends_empty_body(self, gh_context):
        gh_context.fetch.return_value = ISSUE

        await UpdateIssue().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"] == {}

    @pytest.mark.asyncio
    async def test_missing_issue_number_is_captured(self, gh_context):
        result = await UpdateIssue().execute(
            {"owner": OWNER, "repo": REPO, "state": "closed"}, gh_context
        )

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- create_issue_comment ----


class TestCreateIssueComment:
    @pytest.mark.asyncio
    async def test_request_url_method_and_body(self, gh_context):
        gh_context.fetch.return_value = {
            "id": 1,
            "body": "Thanks!",
            "created_at": "2026-01-26T19:01:12Z",
            "updated_at": "2026-01-26T19:01:12Z",
            "html_url": "https://github.com/x/y/issues/1#issuecomment-1",
            "user": {"login": OWNER, "avatar_url": "https://avatars/1"},
        }

        await CreateIssueComment().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "body": "Thanks!"},
            gh_context,
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/issues/{ISSUE_NUMBER}/comments"
        assert call.kwargs["method"] == "POST"
        assert call.kwargs["json"] == {"body": "Thanks!"}

    @pytest.mark.parametrize("missing", ["owner", "repo", "issue_number", "body"])
    @pytest.mark.asyncio
    async def test_all_four_inputs_are_required(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "body": "Hi"}
        del inputs[missing]

        result = await CreateIssueComment().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 403: locked conversation")

        result = await CreateIssueComment().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER, "body": "Hi"},
            gh_context,
        )

        assert result.data["result"] is False
        assert "locked" in result.data["error"]


# ---- get_issue_comments ----


class TestGetIssueComments:
    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = []

        await GetIssueComments().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}, gh_context
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/issues/{ISSUE_NUMBER}/comments"
        )

    @pytest.mark.asyncio
    async def test_returns_a_list(self, gh_context):
        gh_context.fetch.return_value = []

        result = await GetIssueComments().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}, gh_context
        )

        assert result.data == []

    @pytest.mark.asyncio
    async def test_pagination_is_applied(self, gh_context):
        """Long comment threads need paging, so this goes through
        paginated_fetch."""
        gh_context.fetch.return_value = []

        await GetIssueComments().execute(
            {"owner": OWNER, "repo": REPO, "issue_number": ISSUE_NUMBER}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 100

    @pytest.mark.asyncio
    async def test_missing_issue_number_is_captured(self, gh_context):
        result = await GetIssueComments().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- Config ----


class TestGithubCommitIssueConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        [
            "list_commits",
            "get_commit",
            "list_issues",
            "get_issue",
            "create_issue",
            "update_issue",
            "create_issue_comment",
            "get_issue_comments",
        ],
    )
    def test_all_require_owner_and_repo(self, config, action):
        required = config["actions"][action]["input_schema"]["required"]
        assert "owner" in required
        assert "repo" in required

    def test_get_commit_requires_sha(self, config):
        assert "sha" in config["actions"]["get_commit"]["input_schema"]["required"]

    @pytest.mark.parametrize(
        "action", ["get_issue", "update_issue", "create_issue_comment", "get_issue_comments"]
    )
    def test_issue_scoped_actions_require_issue_number(self, config, action):
        assert "issue_number" in config["actions"][action]["input_schema"]["required"]

    def test_create_issue_requires_title(self, config):
        assert "title" in config["actions"]["create_issue"]["input_schema"]["required"]

    def test_create_issue_comment_requires_body(self, config):
        assert "body" in config["actions"]["create_issue_comment"]["input_schema"]["required"]
