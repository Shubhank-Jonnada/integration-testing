# Test1

Looks up geolocation and network ownership details for an IP address using the [ipinfo.io](https://ipinfo.io) API.

## Authentication

Test1 uses an ipinfo access token.

1. Sign up or sign in at [ipinfo.io](https://ipinfo.io).
2. Open [Account → Token](https://ipinfo.io/account/token).
3. Copy the access token.
4. Paste it into the **Access Token** field when connecting the integration.

The token is sent as a bearer token on every request.

## Actions

| Action | Description | Key inputs | Key outputs |
|---|---|---|---|
| `lookup_ip` | Look up geolocation and network ownership details for an IP address | `ip_address` (optional) | `ip`, `city`, `region`, `country`, `loc`, `org`, `postal`, `timezone`, `result` |

Omitting `ip_address` returns details for the IP address the request originates from.

Fields that ipinfo does not return for a given address are omitted from the output, so
callers should treat everything other than `ip` and `result` as optional.

## API

- Base URL: `https://ipinfo.io`
- Docs: https://ipinfo.io/developers
- Rate limits: the free tier allows 50,000 requests per month; paid plans raise this. Exceeding the quota returns HTTP 429.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Missing or invalid access token | Re-copy the token from the ipinfo dashboard and reconnect |
| `404 Not Found` | The value in `ip_address` is not a valid IP address | Pass a bare IPv4 or IPv6 address, not a hostname or URL |
| `429 Too Many Requests` | Monthly request quota exhausted | Wait for the quota to reset or upgrade the ipinfo plan |
