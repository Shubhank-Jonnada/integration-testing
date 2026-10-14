"""Test configuration for the Stripe integration suite.

`stripe.py` calls a bare `Integration.load()`, which the SDK resolves relative to
its own package location -- correct only when the SDK has been vendored into
`dependencies/` by the Lambda packaging step. Under a normally installed SDK the
lookup lands in site-packages and raises ConfigurationError at import time, so
the module can't be imported by tests at all.

Patch it to fall back to the calling module's own directory. This is a no-op
once the repo-wide conftest.py provides the same shim.
"""

import inspect
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))


def _patch_integration_load():
    from autohive_integrations_sdk import Integration

    if getattr(Integration.load, "_autohive_conftest_patched", False):
        return

    original = Integration.load.__func__

    def load(cls, config_path=None):
        if config_path is None:
            for frame in inspect.stack()[1:]:
                candidate = Path(frame.filename).resolve().parent / "config.json"
                if candidate.exists():
                    config_path = candidate
                    break
        return original(cls, config_path)

    load._autohive_conftest_patched = True
    Integration.load = classmethod(load)


_patch_integration_load()
