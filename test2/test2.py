from autohive_integrations_sdk import (
    Integration,
    ExecutionContext,
    ActionHandler,
    ActionResult,
    ActionError,
)
from typing import Dict, Any

# Create the integration using the config.json
test2 = Integration.load()

# Base URL for the OpenWeatherMap API
OPENWEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5"

# Unit system used when the caller does not specify one
DEFAULT_UNITS = "metric"


# ---- Helper Functions ----


def get_api_key(context: ExecutionContext) -> str:
    """
    Read the OpenWeatherMap API key from the execution context.

    Args:
        context: ExecutionContext containing auth credentials

    Returns:
        The configured API key, or an empty string if none is set
    """
    credentials = context.auth.get("credentials", {})

    return credentials.get("api_key", "")


# ---- Action Handlers ----


@test2.action("get_current_weather")
class GetCurrentWeatherAction(ActionHandler):
    """
    Retrieves the current weather conditions for a city, including
    temperature, apparent temperature, humidity and wind speed.
    """

    async def execute(self, inputs: Dict[str, Any], context: ExecutionContext):
        try:
            params = {
                "q": inputs.get("city"),
                "units": inputs.get("units") or DEFAULT_UNITS,
                "appid": get_api_key(context),
            }

            response = await context.fetch(
                f"{OPENWEATHER_API_BASE_URL}/weather",
                method="GET",
                params=params,
            )

            data = response.data or {}
            main = data.get("main") or {}
            wind = data.get("wind") or {}
            weather = (data.get("weather") or [{}])[0]

            output = {
                "city": data.get("name"),
                "country": (data.get("sys") or {}).get("country"),
                "conditions": weather.get("description"),
                "temperature": main.get("temp"),
                "feels_like": main.get("feels_like"),
                "humidity": main.get("humidity"),
                "wind_speed": wind.get("speed"),
            }

            # Drop fields OpenWeatherMap did not report for this location
            output = {key: value for key, value in output.items() if value is not None}
            output["result"] = True

            return ActionResult(data=output, cost_usd=0.0)

        except Exception as e:
            return ActionError(message=str(e))
