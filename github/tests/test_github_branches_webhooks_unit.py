"""Unit tests for the GitHub branch and webhook actions.

Covers `list_branches`, `get_branch`, `create_branch`, `delete_branch`,
`get_branch_protection`, `diff_branch_to_branch`, `create_webhook`,
`list_webhooks`, and `delete_webhook`.

Branch creation and deletion go through the low-level git refs API rather than
the `/branches` collection, which is easy to get wrong. `get_branch_protection`
is the one action in the integration with its own inner try/except, treating any
failure as "protection not enabled".

Fully mocked -- no network access.
"""

import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from github.github import (  # noqa: E402
    CreateBranch,
    CreateWebhook,
    DeleteBranch,
    DeleteWebhook,
    DiffBranchToBranch,
    GetBranch,
    GetBranchProtection,
    ListBranches,
    ListWebhooks,
)

pytestmark = pytest.mark.unit

TOKEN = "gho_testtoken1234567890"  # nosec B105
BASE = "https://api.github.com"
OWNER = "octocat"
REPO = "hello-world"
BRANCH = "feature-branch"
SHA = "6dcb09b5b57875f334f61aebed695e2e4193db5e"
HOOK_ID = 12345678
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

BRANCH_PAYLOAD = {
    "name": "main",
    "protected": True,
    "commit": {"sha": SHA, "url": f"{BASE}/repos/{OWNER}/{REPO}/commits/{SHA}"},
}

PROTECTION = {
    "required_status_checks": {"strict": True, "contexts": ["ci/build", "ci/test"]},
    "enforce_admins": {"enabled": True},
    "required_pull_request_reviews": {
        "required_approving_review_count": 2,
        "dismiss_stale_reviews": True,
        "require_code_owner_reviews": True,
    },
    "restrictions": {"users": [{"login": "octocat"}], "teams": [{"slug": "platform"}]},
}


@pytest.fixture
def gh_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": TOKEN}}
    return ctx


# ---- list_branches ----


class TestListBranches:
    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = [BRANCH_PAYLOAD]

        await ListBranches().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/branches"

    @pytest.mark.asyncio
    async def test_pagination_is_applied(self, gh_context):
        """Repos can have hundreds of branches, so this pages."""
        gh_context.fetch.return_value = []

        await ListBranches().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 100

    @pytest.mark.asyncio
    async def test_returns_branch_entries(self, gh_context):
        gh_context.fetch.return_value = [BRANCH_PAYLOAD]

        result = await ListBranches().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data[0]["name"] == "main"

    @pytest.mark.asyncio
    async def test_empty_repo_yields_empty_list(self, gh_context):
        gh_context.fetch.return_value = []

        result = await ListBranches().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404")

        result = await ListBranches().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False


# ---- get_branch ----


class TestGetBranch:
    @pytest.mark.asyncio
    async def test_request_url_includes_branch(self, gh_context):
        gh_context.fetch.return_value = BRANCH_PAYLOAD

        await GetBranch().execute({"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context)

        assert (
            gh_context.fetch.call_args.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/branches/main"
        )

    @pytest.mark.asyncio
    async def test_returns_branch_details(self, gh_context):
        gh_context.fetch.return_value = BRANCH_PAYLOAD

        result = await GetBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        assert result.data["name"] == "main"

    @pytest.mark.parametrize("missing", ["owner", "repo", "branch"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "branch": "main"}
        del inputs[missing]

        result = await GetBranch().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Branch not found")

        result = await GetBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch": "nope"}, gh_context
        )

        assert result.data["result"] is False


# ---- create_branch ----


class TestCreateBranch:
    @pytest.mark.asyncio
    async def test_posts_to_the_git_refs_endpoint(self, gh_context):
        """Branch creation is a git ref operation, not a POST to /branches --
        there is no such endpoint."""
        gh_context.fetch.return_value = {"ref": f"refs/heads/{BRANCH}", "object": {"sha": SHA}}

        await CreateBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch_name": BRANCH, "sha": SHA}, gh_context
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/git/refs"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_ref_is_fully_qualified(self, gh_context):
        """GitHub requires the full `refs/heads/<name>` form; a bare branch name
        is rejected with a 422."""
        gh_context.fetch.return_value = {"ref": f"refs/heads/{BRANCH}"}

        await CreateBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch_name": BRANCH, "sha": SHA}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"]["ref"] == f"refs/heads/{BRANCH}"

    @pytest.mark.asyncio
    async def test_sha_is_the_starting_point(self, gh_context):
        gh_context.fetch.return_value = {"ref": f"refs/heads/{BRANCH}"}

        await CreateBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch_name": BRANCH, "sha": SHA}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"]["sha"] == SHA

    @pytest.mark.asyncio
    async def test_body_carries_only_ref_and_sha(self, gh_context):
        gh_context.fetch.return_value = {"ref": f"refs/heads/{BRANCH}"}

        await CreateBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch_name": BRANCH, "sha": SHA}, gh_context
        )

        assert set(gh_context.fetch.call_args.kwargs["json"]) == {"ref", "sha"}

    @pytest.mark.asyncio
    async def test_nested_branch_name_is_preserved(self, gh_context):
        """Slashes in branch names are common (feature/foo) and must survive."""
        gh_context.fetch.return_value = {"ref": "refs/heads/feature/nested"}

        await CreateBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch_name": "feature/nested", "sha": SHA},
            gh_context,
        )

        assert gh_context.fetch.call_args.kwargs["json"]["ref"] == "refs/heads/feature/nested"

    @pytest.mark.parametrize("missing", ["owner", "repo", "branch_name", "sha"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "branch_name": BRANCH, "sha": SHA}
        del inputs[missing]

        result = await CreateBranch().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_existing_branch_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 422: Reference already exists")

        result = await CreateBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch_name": BRANCH, "sha": SHA}, gh_context
        )

        assert result.data["result"] is False
        assert "already exists" in result.data["error"]


# ---- delete_branch ----


class TestDeleteBranch:
    @pytest.mark.asyncio
    async def test_deletes_via_the_heads_ref_path(self, gh_context):
        """Deletion targets `git/refs/heads/<branch>` -- note the asymmetry with
        creation, which posts to `git/refs` with the ref in the body."""
        gh_context.fetch.return_value = None

        await DeleteBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch": BRANCH}, gh_context
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_sends_no_body(self, gh_context):
        gh_context.fetch.return_value = None

        await DeleteBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch": BRANCH}, gh_context
        )

        assert "json" not in gh_context.fetch.call_args.kwargs

    @pytest.mark.parametrize("missing", ["owner", "repo", "branch"])
    @pytest.mark.asyncio
    async def test_required_inputs_prevent_the_request(self, gh_context, missing):
        """Deleting a branch discards unmerged commits, so a missing input must
        not reach a malformed path."""
        inputs = {"owner": OWNER, "repo": REPO, "branch": BRANCH}
        del inputs[missing]

        result = await DeleteBranch().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_protected_branch_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 422: branch is protected")

        result = await DeleteBranch().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        assert result.data["result"] is False
        assert "protected" in result.data["error"]


# ---- get_branch_protection ----


class TestGetBranchProtection:
    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = PROTECTION

        await GetBranchProtection().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/branches/main/protection"
        )

    @pytest.mark.asyncio
    async def test_flattens_the_protection_payload(self, gh_context):
        gh_context.fetch.return_value = PROTECTION

        result = await GetBranchProtection().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        assert result.data["enabled"] is True
        assert result.data["required_status_checks"] == ["ci/build", "ci/test"]
        assert result.data["enforce_admins"] is True

    @pytest.mark.asyncio
    async def test_review_requirements_are_extracted(self, gh_context):
        gh_context.fetch.return_value = PROTECTION

        result = await GetBranchProtection().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        reviews = result.data["required_pull_request_reviews"]
        assert reviews["required_approving_review_count"] == 2
        assert reviews["dismiss_stale_reviews"] is True
        assert reviews["require_code_owner_reviews"] is True

    @pytest.mark.asyncio
    async def test_restrictions_are_reduced_to_logins_and_slugs(self, gh_context):
        """Users are identified by login, teams by slug -- two different keys."""
        gh_context.fetch.return_value = PROTECTION

        result = await GetBranchProtection().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        assert result.data["restrictions"]["users"] == ["octocat"]
        assert result.data["restrictions"]["teams"] == ["platform"]

    @pytest.mark.asyncio
    async def test_unprotected_branch_reports_disabled_not_an_error(self, gh_context):
        """GitHub returns 404 when protection isn't configured. This action has
        its own inner try/except that translates that into enabled=False, which
        is the useful answer -- a caller asking "is this protected?" gets a
        definite no rather than a failure."""
        gh_context.fetch.side_effect = Exception("HTTP 404: Branch not protected")

        result = await GetBranchProtection().execute(
            {"owner": OWNER, "repo": REPO, "branch": BRANCH}, gh_context
        )

        assert result.data["enabled"] is False
        assert "not protected" in result.data["error"]

    @pytest.mark.asyncio
    async def test_a_permissions_failure_also_reports_disabled(self, gh_context):
        """The inner except is broad, so a 403 from insufficient scope is
        indistinguishable from "no protection configured". That's a real
        false-negative risk: an unprotected-looking branch may just be one the
        token cannot inspect."""
        gh_context.fetch.side_effect = Exception("HTTP 403: Resource not accessible")

        result = await GetBranchProtection().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        assert result.data["enabled"] is False

    @pytest.mark.asyncio
    async def test_partial_protection_defaults_missing_sections(self, gh_context):
        """Each section is guarded independently, so protection with only status
        checks configured still returns a complete shape."""
        gh_context.fetch.return_value = {"required_status_checks": {"contexts": ["ci"]}}

        result = await GetBranchProtection().execute(
            {"owner": OWNER, "repo": REPO, "branch": "main"}, gh_context
        )

        assert result.data["required_status_checks"] == ["ci"]
        assert result.data["enforce_admins"] is False
        assert result.data["restrictions"]["users"] == []


# ---- diff_branch_to_branch ----


class TestDiffBranchToBranch:
    @pytest.mark.asyncio
    async def test_compare_url_uses_the_triple_dot_form(self, gh_context):
        """GitHub's compare endpoint takes `base...head`."""
        gh_context.fetch.return_value = {
            "status": "ahead",
            "ahead_by": 3,
            "behind_by": 0,
            "total_commits": 3,
            "commits": [],
            "files": [],
        }

        await DiffBranchToBranch().execute(
            {"owner": OWNER, "repo": REPO, "base_branch": "main", "head_branch": BRANCH},
            gh_context,
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/compare/main...{BRANCH}"
        )

    @pytest.mark.asyncio
    async def test_base_and_head_order_is_preserved(self, gh_context):
        """Reversing them inverts ahead_by/behind_by, so the order is meaningful."""
        gh_context.fetch.return_value = {
            "status": "behind",
            "ahead_by": 0,
            "behind_by": 2,
            "total_commits": 2,
            "commits": [],
            "files": [],
        }

        await DiffBranchToBranch().execute(
            {"owner": OWNER, "repo": REPO, "base_branch": BRANCH, "head_branch": "main"},
            gh_context,
        )

        url = gh_context.fetch.call_args.args[0]
        assert url.endswith(f"compare/{BRANCH}...main")

    @pytest.mark.parametrize("missing", ["owner", "repo", "base_branch", "head_branch"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "base_branch": "main", "head_branch": BRANCH}
        del inputs[missing]

        result = await DiffBranchToBranch().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Not Found")

        result = await DiffBranchToBranch().execute(
            {"owner": OWNER, "repo": REPO, "base_branch": "main", "head_branch": "nope"},
            gh_context,
        )

        assert result.data["result"] is False


# ---- create_webhook ----


class TestCreateWebhook:
    def base_inputs(self, **overrides):
        inputs = {
            "owner": OWNER,
            "repo": REPO,
            "url": "https://example.com/hook",
            "events": ["push", "pull_request"],
        }
        inputs.update(overrides)
        return inputs

    @pytest.mark.asyncio
    async def test_request_url_and_method(self, gh_context):
        gh_context.fetch.return_value = {"id": HOOK_ID, "active": True}

        await CreateWebhook().execute(self.base_inputs(), gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/hooks"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_name_is_always_web(self, gh_context):
        """GitHub only supports the `web` hook name for repository webhooks."""
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["name"] == "web"

    @pytest.mark.asyncio
    async def test_url_goes_inside_config_not_the_top_level(self, gh_context):
        """The delivery URL is nested under `config`, which is easy to get wrong
        since the input is a flat `url`."""
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(), gh_context)

        body = gh_context.fetch.call_args.kwargs["json"]
        assert body["config"]["url"] == "https://example.com/hook"
        assert "url" not in body

    @pytest.mark.asyncio
    async def test_content_type_defaults_to_json(self, gh_context):
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["config"]["content_type"] == "json"

    @pytest.mark.asyncio
    async def test_content_type_can_be_form(self, gh_context):
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(content_type="form"), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["config"]["content_type"] == "form"

    @pytest.mark.asyncio
    async def test_secret_is_included_when_supplied(self, gh_context):
        """The secret is what lets the receiver verify the HMAC signature."""
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(
            self.base_inputs(secret="s3cr3t"), gh_context  # nosec B106
        )

        assert gh_context.fetch.call_args.kwargs["json"]["config"]["secret"] == "s3cr3t"

    @pytest.mark.asyncio
    async def test_secret_is_omitted_when_absent(self, gh_context):
        """No secret means unsigned deliveries -- the receiver cannot verify
        origin. Asserted so the unsigned default is explicit."""
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(), gh_context)

        assert "secret" not in gh_context.fetch.call_args.kwargs["json"]["config"]

    @pytest.mark.asyncio
    async def test_events_are_forwarded(self, gh_context):
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["events"] == ["push", "pull_request"]

    @pytest.mark.asyncio
    async def test_active_defaults_to_true(self, gh_context):
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["active"] is True

    @pytest.mark.asyncio
    async def test_active_false_is_sent(self, gh_context):
        """`active` is always present in the body, so False creates a paused
        hook rather than being dropped."""
        gh_context.fetch.return_value = {"id": HOOK_ID}

        await CreateWebhook().execute(self.base_inputs(active=False), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["active"] is False

    @pytest.mark.parametrize("missing", ["owner", "repo", "url", "events"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = self.base_inputs()
        del inputs[missing]

        result = await CreateWebhook().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 422: Hook already exists")

        result = await CreateWebhook().execute(self.base_inputs(), gh_context)

        assert result.data["result"] is False


# ---- list_webhooks / delete_webhook ----


class TestListWebhooks:
    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = []

        await ListWebhooks().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/hooks"

    @pytest.mark.asyncio
    async def test_pagination_is_applied(self, gh_context):
        gh_context.fetch.return_value = []

        await ListWebhooks().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 100

    @pytest.mark.asyncio
    async def test_returns_a_list(self, gh_context):
        gh_context.fetch.return_value = []

        result = await ListWebhooks().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data == []

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 403: admin rights required")

        result = await ListWebhooks().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False


class TestDeleteWebhook:
    @pytest.mark.asyncio
    async def test_request_url_and_method(self, gh_context):
        gh_context.fetch.return_value = None

        await DeleteWebhook().execute(
            {"owner": OWNER, "repo": REPO, "hook_id": HOOK_ID}, gh_context
        )

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/hooks/{HOOK_ID}"
        assert call.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_sends_no_body(self, gh_context):
        gh_context.fetch.return_value = None

        await DeleteWebhook().execute(
            {"owner": OWNER, "repo": REPO, "hook_id": HOOK_ID}, gh_context
        )

        assert "json" not in gh_context.fetch.call_args.kwargs

    @pytest.mark.parametrize("missing", ["owner", "repo", "hook_id"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "hook_id": HOOK_ID}
        del inputs[missing]

        result = await DeleteWebhook().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Not Found")

        result = await DeleteWebhook().execute(
            {"owner": OWNER, "repo": REPO, "hook_id": 999}, gh_context
        )

        assert result.data["result"] is False


# ---- Config ----


class TestGithubBranchWebhookConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action",
        [
            "list_branches",
            "get_branch",
            "create_branch",
            "delete_branch",
            "get_branch_protection",
            "diff_branch_to_branch",
            "create_webhook",
            "list_webhooks",
            "delete_webhook",
        ],
    )
    def test_all_require_owner_and_repo(self, config, action):
        required = config["actions"][action]["input_schema"]["required"]
        assert "owner" in required
        assert "repo" in required

    def test_create_branch_requires_name_and_sha(self, config):
        required = config["actions"]["create_branch"]["input_schema"]["required"]
        assert "branch_name" in required
        assert "sha" in required

    def test_diff_requires_both_branches(self, config):
        required = config["actions"]["diff_branch_to_branch"]["input_schema"]["required"]
        assert "base_branch" in required
        assert "head_branch" in required

    def test_create_webhook_requires_url_and_events(self, config):
        required = config["actions"]["create_webhook"]["input_schema"]["required"]
        assert "url" in required
        assert "events" in required

    def test_delete_webhook_requires_hook_id(self, config):
        assert "hook_id" in config["actions"]["delete_webhook"]["input_schema"]["required"]
