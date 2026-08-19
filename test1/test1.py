from autohive_integrations_sdk import (
    Integration,
    ExecutionContext,
    ActionHandler,
    ActionResult,
    ActionError,
)
from typing import Dict, Any

# Create the integration using the config.json
test1 = Integration.load()

# Base URL for the ipinfo API
IPINFO_API_BASE_URL = "https://ipinfo.io"

# Fields returned by ipinfo that are surfaced directly in the action output
IPINFO_FIELDS = (
    "ip",
    "hostname",
    "city",
    "region",
    "country",
    "loc",
    "org",
    "postal",
    "timezone",
)


# ---- Helper Functions ----


def get_auth_headers(context: ExecutionContext) -> Dict[str, str]:
    """
    Build authentication headers for ipinfo API requests.

    Args:
        context: ExecutionContext containing auth credentials

    Returns:
        Dictionary with Authorization and Accept headers
    """
    credentials = context.auth.get("credentials", {})
    api_token = credentials.get("api_token", "")

    return {"Authorization": f"Bearer {api_token}", "Accept": "application/json"}


# ---- Action Handlers ----


@test1.action("lookup_ip")
class LookupIpAction(ActionHandler):
    """
    Looks up geolocation and network ownership details for an IP address.
    Omitting the address returns details for the caller's own IP.
    """

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext):
        try:
            ip_address = (inputs.get("ip_address") or "").strip()
            path = f"/{ip_address}/json" if ip_address else "/json"

            response = await context.fetch(
                f"{IPINFO_API_BASE_URL}{path}",
                method="GET",
                headers=get_auth_headers(context),
            )

            data = response.data or {}
            output = {field: data[field] for field in IPINFO_FIELDS if data.get(field)}
            output["result"] = True

            return ActionResult(data=output, cost_usd=0.0)

        except Exception as e:
            return ActionError(message=str(e))
