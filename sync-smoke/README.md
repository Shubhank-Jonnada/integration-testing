# Sync Smoke

A local echo action for testing release packaging, first upload, version bumps, and rollback review. No authentication or external API calls are needed.

Call `echo` with `{"message": "hello"}`. The response contains the message and the version from the packaged `config.json`, making older and newer deployments distinguishable.

Run tests with `pytest sync-smoke/tests`.
