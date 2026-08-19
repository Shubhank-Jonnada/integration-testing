"""Unit tests for the GitHub file, gist, user, workflow, tag, and release actions.

Covers the remaining fifteen actions:

- Files: `get_file_content`, `create_file`, `update_file`, `delete_file`
- Gists: `create_gist`
- Users: `get_user`, `list_organization_members`
- Workflows: `list_workflows`, `get_workflow_runs`, `get_rate_limit`
- Tags and releases: `list_tags`, `list_releases`, `get_release`,
  `get_latest_release`, `get_release_by_tag`

File writes are the interesting part: content is base64-encoded on the way out
and decoded on the way back, and the `sha` requirement is what makes updates and
deletes safe against concurrent edits.

Fully mocked -- no network access.
"""

import base64
import json
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from github.github import (  # noqa: E402
    CreateFile,
    CreateGist,
    DeleteFile,
    GetFileContent,
    GetLatestRelease,
    GetRateLimit,
    GetRelease,
    GetReleaseByTag,
    GetUser,
    GetWorkflowRuns,
    ListOrganizationMembers,
    ListReleases,
    ListTags,
    ListWorkflows,
    UpdateFile,
)

pytestmark = pytest.mark.unit

TOKEN = "gho_testtoken1234567890"  # nosec B105
BASE = "https://api.github.com"
OWNER = "octocat"
REPO = "hello-world"
ORG = "acme-corp"
PATH = "src/app.py"
SHA = "6dcb09b5b57875f334f61aebed695e2e4193db5e"
RELEASE_ID = 987654
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")

FILE_TEXT = "print('hello')\n"
FILE_RESPONSE = {
    "name": "app.py",
    "path": PATH,
    "sha": SHA,
    "size": len(FILE_TEXT),
    "content": base64.b64encode(FILE_TEXT.encode()).decode(),
}

RELEASE = {
    "id": RELEASE_ID,
    "tag_name": "v1.0.0",
    "name": "Version 1.0.0",
    "body": "Release notes",
    "draft": False,
    "prerelease": False,
    "created_at": "2026-01-26T19:01:12Z",
    "published_at": "2026-01-26T19:05:00Z",
    "html_url": f"https://github.com/{OWNER}/{REPO}/releases/tag/v1.0.0",
    "tarball_url": f"{BASE}/repos/{OWNER}/{REPO}/tarball/v1.0.0",
    "zipball_url": f"{BASE}/repos/{OWNER}/{REPO}/zipball/v1.0.0",
    "author": {"login": OWNER, "id": 1, "avatar_url": "https://avatars/1"},
    "assets": [],
}

WORKFLOW = {
    "id": 161335,
    "name": "CI",
    "path": ".github/workflows/ci.yml",
    "state": "active",
    "created_at": "2026-01-26T19:01:12Z",
    "updated_at": "2026-01-26T19:01:12Z",
    "html_url": f"https://github.com/{OWNER}/{REPO}/actions/workflows/ci.yml",
}

WORKFLOW_RUN = {
    "id": 30433642,
    "name": "CI",
    "workflow_id": 161335,
    "head_branch": "main",
    "head_sha": SHA,
    "run_number": 42,
    "event": "push",
    "status": "completed",
    "conclusion": "success",
    "created_at": "2026-01-26T19:01:12Z",
    "updated_at": "2026-01-26T19:06:00Z",
    "actor": {"login": OWNER, "avatar_url": "https://avatars/1"},
    "html_url": f"https://github.com/{OWNER}/{REPO}/actions/runs/30433642",
}


def tag_payload(name):
    return {
        "name": name,
        "commit": {"sha": SHA, "url": f"{BASE}/repos/{OWNER}/{REPO}/commits/{SHA}"},
    }


@pytest.fixture
def gh_context():
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {"auth_type": "PlatformOauth2", "credentials": {"access_token": TOKEN}}
    return ctx


# ---- get_file_content ----


class TestGetFileContent:
    @pytest.mark.asyncio
    async def test_decodes_base64_content(self, gh_context):
        gh_context.fetch.return_value = FILE_RESPONSE

        result = await GetFileContent().execute(
            {"owner": OWNER, "repo": REPO, "path": PATH}, gh_context
        )

        assert result.data["content"] == FILE_TEXT

    @pytest.mark.asyncio
    async def test_strips_newlines_before_decoding(self, gh_context):
        """GitHub wraps base64 at 60 characters, and those newlines break a naive
        b64decode -- so they are stripped first."""
        wrapped = "\n".join(
            base64.b64encode(FILE_TEXT.encode()).decode()[i : i + 4]
            for i in range(0, len(base64.b64encode(FILE_TEXT.encode()).decode()), 4)
        )
        gh_context.fetch.return_value = {**FILE_RESPONSE, "content": wrapped}

        result = await GetFileContent().execute(
            {"owner": OWNER, "repo": REPO, "path": PATH}, gh_context
        )

        assert result.data["content"] == FILE_TEXT

    @pytest.mark.asyncio
    async def test_request_url_includes_the_path(self, gh_context):
        gh_context.fetch.return_value = FILE_RESPONSE

        await GetFileContent().execute({"owner": OWNER, "repo": REPO, "path": PATH}, gh_context)

        assert (
            gh_context.fetch.call_args.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/contents/{PATH}"
        )

    @pytest.mark.asyncio
    async def test_returns_the_sha_for_later_writes(self, gh_context):
        """The sha returned here is what update_file and delete_file require."""
        gh_context.fetch.return_value = FILE_RESPONSE

        result = await GetFileContent().execute(
            {"owner": OWNER, "repo": REPO, "path": PATH}, gh_context
        )

        assert result.data["sha"] == SHA

    @pytest.mark.asyncio
    async def test_ref_is_optional(self, gh_context):
        gh_context.fetch.return_value = FILE_RESPONSE

        await GetFileContent().execute({"owner": OWNER, "repo": REPO, "path": PATH}, gh_context)
        assert gh_context.fetch.call_args.kwargs["params"] is None

        await GetFileContent().execute(
            {"owner": OWNER, "repo": REPO, "path": PATH, "ref": "develop"}, gh_context
        )
        assert gh_context.fetch.call_args.kwargs["params"] == {"ref": "develop"}

    @pytest.mark.asyncio
    async def test_empty_file_decodes_to_empty_string(self, gh_context):
        gh_context.fetch.return_value = {**FILE_RESPONSE, "content": "", "size": 0}

        result = await GetFileContent().execute(
            {"owner": OWNER, "repo": REPO, "path": PATH}, gh_context
        )

        assert result.data["content"] == ""

    @pytest.mark.asyncio
    async def test_binary_file_is_captured_as_an_error(self, gh_context):
        """The decoder assumes UTF-8, so a binary file raises UnicodeDecodeError,
        which the decorator captures. Documented rather than silently mangled."""
        gh_context.fetch.return_value = {
            **FILE_RESPONSE,
            "content": base64.b64encode(b"\x89PNG\r\n\x1a\n\xff\xfe").decode(),
        }

        result = await GetFileContent().execute(
            {"owner": OWNER, "repo": REPO, "path": "logo.png"}, gh_context
        )

        assert result.data["result"] is False

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Not Found")

        result = await GetFileContent().execute(
            {"owner": OWNER, "repo": REPO, "path": "nope.py"}, gh_context
        )

        assert result.data["result"] is False


# ---- create_file / update_file ----


class TestCreateFile:
    def base_inputs(self, **overrides):
        inputs = {
            "owner": OWNER,
            "repo": REPO,
            "path": PATH,
            "message": "Add app.py",
            "content": FILE_TEXT,
        }
        inputs.update(overrides)
        return inputs

    @pytest.mark.asyncio
    async def test_uses_put_to_the_contents_path(self, gh_context):
        gh_context.fetch.return_value = {"content": FILE_RESPONSE, "commit": {"sha": SHA}}

        await CreateFile().execute(self.base_inputs(), gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/contents/{PATH}"
        assert call.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_content_is_base64_encoded(self, gh_context):
        """GitHub requires base64; sending raw text is rejected."""
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await CreateFile().execute(self.base_inputs(), gh_context)

        sent = gh_context.fetch.call_args.kwargs["json"]["content"]
        assert base64.b64decode(sent).decode() == FILE_TEXT

    @pytest.mark.asyncio
    async def test_no_sha_is_sent_on_create(self, gh_context):
        """A sha here would make GitHub treat it as an update of an existing file."""
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await CreateFile().execute(self.base_inputs(), gh_context)

        assert "sha" not in gh_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_branch_is_optional(self, gh_context):
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await CreateFile().execute(self.base_inputs(), gh_context)
        assert "branch" not in gh_context.fetch.call_args.kwargs["json"]

        await CreateFile().execute(self.base_inputs(branch="develop"), gh_context)
        assert gh_context.fetch.call_args.kwargs["json"]["branch"] == "develop"

    @pytest.mark.asyncio
    async def test_commit_message_is_required_in_the_body(self, gh_context):
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await CreateFile().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["message"] == "Add app.py"

    @pytest.mark.asyncio
    async def test_unicode_content_round_trips(self, gh_context):
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}
        text = "héllo wörld — ✓\n"

        await CreateFile().execute(self.base_inputs(content=text), gh_context)

        sent = gh_context.fetch.call_args.kwargs["json"]["content"]
        assert base64.b64decode(sent).decode("utf-8") == text

    @pytest.mark.parametrize("missing", ["owner", "repo", "path", "message", "content"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = self.base_inputs()
        del inputs[missing]

        result = await CreateFile().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_existing_file_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception('HTTP 422: "sha" wasn\'t supplied')

        result = await CreateFile().execute(self.base_inputs(), gh_context)

        assert result.data["result"] is False


class TestUpdateFile:
    def base_inputs(self, **overrides):
        inputs = {
            "owner": OWNER,
            "repo": REPO,
            "path": PATH,
            "message": "Update app.py",
            "content": FILE_TEXT,
            "sha": SHA,
        }
        inputs.update(overrides)
        return inputs

    @pytest.mark.asyncio
    async def test_sends_the_sha(self, gh_context):
        """The sha is GitHub's optimistic-concurrency check -- without it the
        update is rejected, and with a stale one it fails rather than clobbering
        a concurrent change."""
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await UpdateFile().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["sha"] == SHA

    @pytest.mark.asyncio
    async def test_uses_put_like_create(self, gh_context):
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await UpdateFile().execute(self.base_inputs(), gh_context)

        assert gh_context.fetch.call_args.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_content_is_base64_encoded(self, gh_context):
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await UpdateFile().execute(self.base_inputs(content="new body\n"), gh_context)

        sent = gh_context.fetch.call_args.kwargs["json"]["content"]
        assert base64.b64decode(sent).decode() == "new body\n"

    @pytest.mark.asyncio
    async def test_body_carries_message_content_and_sha(self, gh_context):
        gh_context.fetch.return_value = {"content": FILE_RESPONSE}

        await UpdateFile().execute(self.base_inputs(), gh_context)

        assert set(gh_context.fetch.call_args.kwargs["json"]) == {"message", "content", "sha"}

    @pytest.mark.parametrize("missing", ["owner", "repo", "path", "message", "content", "sha"])
    @pytest.mark.asyncio
    async def test_all_six_inputs_are_required(self, gh_context, missing):
        inputs = self.base_inputs()
        del inputs[missing]

        result = await UpdateFile().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_stale_sha_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 409: is at another sha")

        result = await UpdateFile().execute(self.base_inputs(), gh_context)

        assert result.data["result"] is False
        assert "another sha" in result.data["error"]


class TestDeleteFile:
    def base_inputs(self, **overrides):
        inputs = {
            "owner": OWNER,
            "repo": REPO,
            "path": PATH,
            "message": "Remove app.py",
            "sha": SHA,
        }
        inputs.update(overrides)
        return inputs

    @pytest.mark.asyncio
    async def test_uses_delete_with_a_body(self, gh_context):
        """Unusually for a DELETE, GitHub requires a JSON body here -- the commit
        message and sha."""
        gh_context.fetch.return_value = {"commit": {"sha": SHA}}

        await DeleteFile().execute(self.base_inputs(), gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/contents/{PATH}"
        assert call.kwargs["method"] == "DELETE"
        assert call.kwargs["json"] == {"message": "Remove app.py", "sha": SHA}

    @pytest.mark.asyncio
    async def test_no_content_field_is_sent(self, gh_context):
        gh_context.fetch.return_value = {"commit": {"sha": SHA}}

        await DeleteFile().execute(self.base_inputs(), gh_context)

        assert "content" not in gh_context.fetch.call_args.kwargs["json"]

    @pytest.mark.asyncio
    async def test_branch_is_optional(self, gh_context):
        gh_context.fetch.return_value = {"commit": {"sha": SHA}}

        await DeleteFile().execute(self.base_inputs(branch="develop"), gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["branch"] == "develop"

    @pytest.mark.parametrize("missing", ["owner", "repo", "path", "message", "sha"])
    @pytest.mark.asyncio
    async def test_required_inputs_prevent_deletion(self, gh_context, missing):
        inputs = self.base_inputs()
        del inputs[missing]

        result = await DeleteFile().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- create_gist ----


class TestCreateGist:
    @pytest.mark.asyncio
    async def test_posts_to_the_gists_root(self, gh_context):
        """Gists are account-level, not repo-scoped."""
        gh_context.fetch.return_value = {"id": "abc123", "html_url": "https://gist.github.com/x"}

        await CreateGist().execute({"files": {"a.py": {"content": "x"}}}, gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/gists"
        assert call.kwargs["method"] == "POST"

    @pytest.mark.asyncio
    async def test_public_defaults_to_true(self, gh_context):
        """A public gist is world-readable and indexed, so this default is worth
        asserting explicitly."""
        gh_context.fetch.return_value = {"id": "abc123"}

        await CreateGist().execute({"files": {"a.py": {"content": "x"}}}, gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["public"] is True

    @pytest.mark.asyncio
    async def test_public_false_creates_a_secret_gist(self, gh_context):
        gh_context.fetch.return_value = {"id": "abc123"}

        await CreateGist().execute(
            {"files": {"a.py": {"content": "x"}}, "public": False}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["json"]["public"] is False

    @pytest.mark.asyncio
    async def test_description_defaults_to_empty_string(self, gh_context):
        gh_context.fetch.return_value = {"id": "abc123"}

        await CreateGist().execute({"files": {"a.py": {"content": "x"}}}, gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["description"] == ""

    @pytest.mark.asyncio
    async def test_files_map_is_passed_through_verbatim(self, gh_context):
        """Gist content is not base64-encoded, unlike repository files."""
        gh_context.fetch.return_value = {"id": "abc123"}
        files = {"a.py": {"content": "print(1)"}, "b.txt": {"content": "notes"}}

        await CreateGist().execute({"files": files}, gh_context)

        assert gh_context.fetch.call_args.kwargs["json"]["files"] == files

    @pytest.mark.asyncio
    async def test_missing_files_is_captured(self, gh_context):
        result = await CreateGist().execute({"description": "d"}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- get_user / list_organization_members ----


class TestGetUser:
    @pytest.mark.asyncio
    async def test_named_user_targets_users_path(self, gh_context):
        gh_context.fetch.return_value = {"login": "torvalds", "id": 1}

        await GetUser().execute({"username": "torvalds"}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/users/torvalds"

    @pytest.mark.asyncio
    async def test_omitted_username_returns_the_authenticated_user(self, gh_context):
        """`/user` includes private profile fields that `/users/{name}` omits."""
        gh_context.fetch.return_value = {"login": OWNER, "id": 1}

        await GetUser().execute({}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/user"

    @pytest.mark.asyncio
    async def test_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Not Found")

        result = await GetUser().execute({"username": "nope"}, gh_context)

        assert result.data["result"] is False


class TestListOrganizationMembers:
    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = []

        await ListOrganizationMembers().execute({"org": ORG}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/orgs/{ORG}/members"

    @pytest.mark.asyncio
    async def test_role_defaults_to_all(self, gh_context):
        gh_context.fetch.return_value = []

        await ListOrganizationMembers().execute({"org": ORG}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["role"] == "all"

    @pytest.mark.asyncio
    async def test_admin_role_filter_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListOrganizationMembers().execute({"org": ORG, "role": "admin"}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["role"] == "admin"

    @pytest.mark.asyncio
    async def test_pagination_is_applied(self, gh_context):
        gh_context.fetch.return_value = []

        await ListOrganizationMembers().execute({"org": ORG}, gh_context)

        assert gh_context.fetch.call_args.kwargs["params"]["per_page"] == 100

    @pytest.mark.asyncio
    async def test_missing_org_is_captured(self, gh_context):
        result = await ListOrganizationMembers().execute({}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- Workflows and rate limit ----


class TestListWorkflows:
    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = {"workflows": []}

        await ListWorkflows().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/actions/workflows"
        )

    @pytest.mark.asyncio
    async def test_unwraps_the_workflows_envelope(self, gh_context):
        """The Actions API wraps results, so paginated_fetch is given a data_key."""
        gh_context.fetch.return_value = {"total_count": 1, "workflows": [WORKFLOW]}

        result = await ListWorkflows().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_empty_envelope_yields_empty_list(self, gh_context):
        gh_context.fetch.return_value = {"total_count": 0, "workflows": []}

        result = await ListWorkflows().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data == []


class TestGetWorkflowRuns:
    @pytest.mark.asyncio
    async def test_request_url_includes_workflow_id(self, gh_context):
        gh_context.fetch.return_value = {"workflow_runs": []}

        await GetWorkflowRuns().execute(
            {"owner": OWNER, "repo": REPO, "workflow_id": 161335}, gh_context
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/actions/workflows/161335/runs"
        )

    @pytest.mark.asyncio
    async def test_unwraps_the_workflow_runs_envelope(self, gh_context):
        gh_context.fetch.return_value = {"total_count": 1, "workflow_runs": [WORKFLOW_RUN]}

        result = await GetWorkflowRuns().execute(
            {"owner": OWNER, "repo": REPO, "workflow_id": 161335}, gh_context
        )

        assert len(result.data) == 1

    @pytest.mark.asyncio
    async def test_status_and_branch_filters_forwarded(self, gh_context):
        gh_context.fetch.return_value = {"workflow_runs": []}

        await GetWorkflowRuns().execute(
            {
                "owner": OWNER,
                "repo": REPO,
                "workflow_id": 161335,
                "status": "failure",
                "branch": "main",
            },
            gh_context,
        )

        params = gh_context.fetch.call_args.kwargs["params"]
        assert params["status"] == "failure"
        assert params["branch"] == "main"

    @pytest.mark.asyncio
    async def test_filters_are_omitted_when_absent(self, gh_context):
        gh_context.fetch.return_value = {"workflow_runs": []}

        await GetWorkflowRuns().execute(
            {"owner": OWNER, "repo": REPO, "workflow_id": 161335}, gh_context
        )

        params = gh_context.fetch.call_args.kwargs["params"]
        assert "status" not in params
        assert "branch" not in params

    @pytest.mark.parametrize("missing", ["owner", "repo", "workflow_id"])
    @pytest.mark.asyncio
    async def test_required_inputs_are_captured(self, gh_context, missing):
        inputs = {"owner": OWNER, "repo": REPO, "workflow_id": 161335}
        del inputs[missing]

        result = await GetWorkflowRuns().execute(inputs, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


class TestGetRateLimit:
    @pytest.mark.asyncio
    async def test_request_url(self, gh_context):
        gh_context.fetch.return_value = {
            "resources": {"core": {"limit": 5000, "remaining": 4999, "reset": 1700000000}},
            "rate": {"limit": 5000, "remaining": 4999, "reset": 1700000000},
        }

        await GetRateLimit().execute({}, gh_context)

        assert gh_context.fetch.call_args.args[0] == f"{BASE}/rate_limit"

    @pytest.mark.asyncio
    async def test_takes_no_inputs(self, gh_context):
        gh_context.fetch.return_value = {"resources": {}, "rate": {}}

        await GetRateLimit().execute({"ignored": True}, gh_context)

        assert "params" not in gh_context.fetch.call_args.kwargs

    @pytest.mark.asyncio
    async def test_requires_a_token_like_every_other_action(self, gh_context):
        """Even though /rate_limit is callable unauthenticated, the decorator still
        gates on a token -- so this action can't be used to probe anonymously."""
        gh_context.auth = {"credentials": {}}

        result = await GetRateLimit().execute({}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- Tags and releases ----


class TestListTags:
    @pytest.mark.asyncio
    async def test_request_url_and_default_pagination(self, gh_context):
        gh_context.fetch.return_value = [tag_payload("v1.0.0")]

        await ListTags().execute({"owner": OWNER, "repo": REPO}, gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/tags"
        assert call.kwargs["params"] == {"per_page": 30, "page": 1}

    @pytest.mark.asyncio
    async def test_single_page_fetch_not_paginated(self, gh_context):
        """Unlike list_branches, tags do a single-page fetch and default to 30 --
        so a repo with more tags silently returns a partial list unless the caller
        pages explicitly."""
        gh_context.fetch.return_value = [tag_payload(f"v{i}") for i in range(30)]

        result = await ListTags().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert gh_context.fetch.await_count == 1
        assert len(result.data) == 30

    @pytest.mark.asyncio
    async def test_explicit_pagination_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListTags().execute(
            {"owner": OWNER, "repo": REPO, "per_page": 100, "page": 3}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"] == {"per_page": 100, "page": 3}


class TestListReleases:
    @pytest.mark.asyncio
    async def test_request_url_and_default_pagination(self, gh_context):
        gh_context.fetch.return_value = [RELEASE]

        await ListReleases().execute({"owner": OWNER, "repo": REPO}, gh_context)

        call = gh_context.fetch.call_args
        assert call.args[0] == f"{BASE}/repos/{OWNER}/{REPO}/releases"
        assert call.kwargs["params"] == {"per_page": 30, "page": 1}

    @pytest.mark.asyncio
    async def test_releases_are_not_tags(self, gh_context):
        """The releases endpoint omits plain git tags, so the two actions return
        different sets."""
        gh_context.fetch.return_value = [RELEASE]

        result = await ListReleases().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data[0]["tag_name"] == "v1.0.0"

    @pytest.mark.asyncio
    async def test_explicit_pagination_forwarded(self, gh_context):
        gh_context.fetch.return_value = []

        await ListReleases().execute(
            {"owner": OWNER, "repo": REPO, "per_page": 50, "page": 2}, gh_context
        )

        assert gh_context.fetch.call_args.kwargs["params"] == {"per_page": 50, "page": 2}


class TestGetRelease:
    @pytest.mark.asyncio
    async def test_request_url_includes_release_id(self, gh_context):
        gh_context.fetch.return_value = RELEASE

        await GetRelease().execute(
            {"owner": OWNER, "repo": REPO, "release_id": RELEASE_ID}, gh_context
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/releases/{RELEASE_ID}"
        )

    @pytest.mark.asyncio
    async def test_missing_release_id_is_captured(self, gh_context):
        result = await GetRelease().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


class TestGetLatestRelease:
    @pytest.mark.asyncio
    async def test_uses_the_latest_sentinel_path(self, gh_context):
        """`/releases/latest` excludes drafts and prereleases, so it is not simply
        the newest entry from list_releases."""
        gh_context.fetch.return_value = RELEASE

        await GetLatestRelease().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/releases/latest"
        )

    @pytest.mark.asyncio
    async def test_no_releases_error_is_captured(self, gh_context):
        gh_context.fetch.side_effect = Exception("HTTP 404: Not Found")

        result = await GetLatestRelease().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False


class TestGetReleaseByTag:
    @pytest.mark.asyncio
    async def test_request_url_includes_the_tag(self, gh_context):
        gh_context.fetch.return_value = RELEASE

        await GetReleaseByTag().execute(
            {"owner": OWNER, "repo": REPO, "tag": "v1.0.0"}, gh_context
        )

        assert (
            gh_context.fetch.call_args.args[0]
            == f"{BASE}/repos/{OWNER}/{REPO}/releases/tags/v1.0.0"
        )

    @pytest.mark.asyncio
    async def test_slashes_in_tags_are_percent_encoded(self, gh_context):
        """`quote(tag, safe="")` encodes the slash, so `release/2026-01` doesn't
        split into extra path segments and 404."""
        gh_context.fetch.return_value = RELEASE

        await GetReleaseByTag().execute(
            {"owner": OWNER, "repo": REPO, "tag": "release/2026-01"}, gh_context
        )

        url = gh_context.fetch.call_args.args[0]
        assert url.endswith("releases/tags/release%2F2026-01")

    @pytest.mark.asyncio
    async def test_spaces_and_specials_are_encoded(self, gh_context):
        gh_context.fetch.return_value = RELEASE

        await GetReleaseByTag().execute(
            {"owner": OWNER, "repo": REPO, "tag": "v1 beta+1"}, gh_context
        )

        url = gh_context.fetch.call_args.args[0]
        assert " " not in url
        assert "%20" in url

    @pytest.mark.asyncio
    async def test_missing_tag_is_captured(self, gh_context):
        result = await GetReleaseByTag().execute({"owner": OWNER, "repo": REPO}, gh_context)

        assert result.data["result"] is False
        gh_context.fetch.assert_not_called()


# ---- Config ----


class TestGithubFileReleaseConfig:
    @pytest.fixture(scope="class")
    def config(self):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)

    @pytest.mark.parametrize(
        "action", ["get_file_content", "create_file", "update_file", "delete_file"]
    )
    def test_file_actions_require_owner_repo_and_path(self, config, action):
        required = config["actions"][action]["input_schema"]["required"]
        assert "owner" in required
        assert "repo" in required
        assert "path" in required

    @pytest.mark.parametrize("action", ["update_file", "delete_file"])
    def test_mutating_file_actions_require_sha(self, config, action):
        """The sha is the concurrency guard, so the schema must demand it."""
        assert "sha" in config["actions"][action]["input_schema"]["required"]

    def test_create_file_does_not_require_sha(self, config):
        assert "sha" not in config["actions"]["create_file"]["input_schema"]["required"]

    def test_create_gist_requires_files(self, config):
        assert "files" in config["actions"]["create_gist"]["input_schema"]["required"]

    def test_get_user_requires_nothing(self, config):
        """It falls back to the authenticated user."""
        assert not config["actions"]["get_user"]["input_schema"].get("required")

    def test_get_rate_limit_requires_nothing(self, config):
        assert not config["actions"]["get_rate_limit"]["input_schema"].get("required")

    def test_get_release_by_tag_requires_tag(self, config):
        assert "tag" in config["actions"]["get_release_by_tag"]["input_schema"]["required"]
