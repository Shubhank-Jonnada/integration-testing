"""Repo-wide pytest configuration for the integration test suites.

Two jobs:

1. Make ``Integration.load()`` resolve ``config.json`` correctly under a
   pip-installed SDK. Every integration in this repo calls a bare
   ``Integration.load()``, and the SDK resolves that relative to its *own*
   package location -- which only works when the SDK is vendored into
   ``<integration>/dependencies/`` by the Lambda packaging step. Locally the
   SDK lives in site-packages, so the bare call looks for
   ``.../Lib/config.json`` and raises ConfigurationError at import time.
   We patch it to fall back to the calling module's own directory.

2. Provide the shared ``mock_context`` / ``make_context`` / ``env_credentials``
   fixtures that the integration test suites build on.
"""

import inspect
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

REPO_ROOT = Path(__file__).parent

# Let test modules import their integration as a top-level package
# (e.g. ``from notion.notion import ...``).
sys.path.insert(0, str(REPO_ROOT))


def _patch_integration_load():
    """Resolve a bare ``Integration.load()`` against the caller's directory."""
    from autohive_integrations_sdk import Integration

    if getattr(Integration.load, "_autohive_conftest_patched", False):
        return

    original = Integration.load.__func__

    def load(cls, config_path=None):
        if config_path is None:
            # Walk out of the SDK and locate the integration module that called us.
            for frame in inspect.stack()[1:]:
                caller = Path(frame.filename).resolve()
                candidate = caller.parent / "config.json"
                if candidate.exists():
                    config_path = candidate
                    break
        return original(cls, config_path)

    load._autohive_conftest_patched = True
    Integration.load = classmethod(load)


_patch_integration_load()


@pytest.fixture
def mock_context():
    """Minimal ExecutionContext double with an awaitable ``fetch``."""
    ctx = MagicMock(name="ExecutionContext")
    ctx.fetch = AsyncMock(name="fetch")
    ctx.auth = {}
    return ctx


@pytest.fixture
def make_context():
    """Factory for a context with an arbitrary ``auth`` payload."""

    def _make(auth=None):
        ctx = MagicMock(name="ExecutionContext")
        ctx.fetch = AsyncMock(name="fetch")
        ctx.auth = auth or {}
        return ctx

    return _make


@pytest.fixture
def env_credentials():
    """Read credentials from the environment; return None when absent.

    Used by the live integration suites so they can skip cleanly rather than
    fail when no credentials are configured.
    """

    def _read(*names):
        values = {name: os.environ.get(name) for name in names}
        if any(v is None for v in values.values()):
            return None
        return values

    return _read
