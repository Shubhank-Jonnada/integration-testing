# Integration testing fixtures

This repository contains small integration fixtures used to validate Autohive connectors and review workflows.

Each integration lives in its own directory with its implementation, configuration, and focused tests. Changes should remain isolated to the integration being exercised so failures are easy to diagnose.

## Contributing

1. Keep credentials and local environment files out of the repository.
2. Add or update focused tests with behavior changes.
3. Run the affected integration's tests before opening a pull request.
