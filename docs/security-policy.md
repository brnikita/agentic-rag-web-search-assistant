# Security Policy

## Secrets

Secrets live in AWS Secrets Manager and are mounted at runtime by the External
Secrets Operator. Never commit secrets, and never pass them as build args —
build args are visible in image history.

## Access

Production access is granted just-in-time through the access request bot and
expires after four hours. There are no standing production credentials for
engineers.

## Data retention

Raw events are retained for 90 days. Aggregated rollups are retained for seven
years. Deletion requests under GDPR are processed within 30 days and cascade to
rollups on the next nightly run.

## Dependency policy

Critical and high CVEs must be patched within seven days. The weekly scan opens
tickets automatically; a ticket older than seven days blocks the release train.
