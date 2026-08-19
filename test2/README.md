# Test2

Retrieves current weather conditions for a city using the [OpenWeatherMap](https://openweathermap.org) API.

## Authentication

Test2 uses an OpenWeatherMap API key.

1. Create a free account at [openweathermap.org](https://home.openweathermap.org/users/sign_up).
2. Open [API keys](https://home.openweathermap.org/api_keys).
3. Copy the default key, or generate a new one.
4. Paste it into the **API Key** field when connecting the integration.

New keys can take up to a couple of hours to activate; until then the API returns HTTP 401.

## Actions

| Action | Description | Key inputs | Key outputs |
|---|---|---|---|
| `get_current_weather` | Get current weather conditions for a city | `city` (required), `units` (optional) | `city`, `country`, `conditions`, `temperature`, `feels_like`, `humidity`, `wind_speed`, `result` |

`city` accepts a plain city name (`Wellington`) or a qualified one (`Wellington,NZ`, `Austin,TX,US`).
Qualifying the name avoids ambiguity where several cities share a name.

`units` controls the unit system: `metric` (°C, m/s — the default), `imperial` (°F, mph), or `standard` (K, m/s).

## API

- Base URL: `https://api.openweathermap.org/data/2.5`
- Docs: https://openweathermap.org/current
- Rate limits: the free tier allows 60 calls per minute and 1,000,000 per month. Exceeding it returns HTTP 429.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Key is invalid, or newly created and not yet active | Verify the key, and allow up to two hours after creating it |
| `404 Not Found` | City name not recognised | Check the spelling, or qualify it with a country code such as `Wellington,NZ` |
| `429 Too Many Requests` | Call rate or monthly quota exceeded | Reduce call frequency or upgrade the OpenWeatherMap plan |
