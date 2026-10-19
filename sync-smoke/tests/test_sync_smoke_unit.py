import importlib.util
import json

import pytest

from .context import ROOT

spec = importlib.util.spec_from_file_location("sync_smoke", ROOT / "sync_smoke.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_echo_reports_packaged_version_without_network():
    result = await module.EchoAction().execute({"message": "hello"}, None)
    assert result.data == {
        "message": "hello",
        "version": json.loads((ROOT / "config.json").read_text())["version"],
    }
