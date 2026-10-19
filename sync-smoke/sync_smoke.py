import json
from pathlib import Path

from autohive_integrations_sdk import ActionHandler, ActionResult, Integration

CONFIG_PATH = Path(__file__).with_name("config.json")
sync_smoke = Integration.load(str(CONFIG_PATH))


@sync_smoke.action("echo")
class EchoAction(ActionHandler):
    async def execute(self, inputs, context):
        version = json.loads(CONFIG_PATH.read_text())["version"]
        return ActionResult(data={"message": inputs["message"], "version": version}, cost_usd=0.0)
