# Platform Architecture

The platform is split into three services.

## Ingestion service

Consumes events from the `raw-events` Kafka topic, validates them against the
Avro schema registry, and writes normalised records to the `events` table in
Postgres. It is stateless and scales horizontally; ordering is guaranteed per
partition key only.

## Query service

A read-only gRPC API in front of the `events` table. Every query must supply a
tenant id; cross-tenant reads are rejected at the interceptor layer, not in
business logic.

## Scheduler

Runs nightly rollups via Temporal workflows. Rollups are idempotent and keyed by
`(tenant_id, date)`, so a replayed workflow overwrites rather than duplicates.

## Why Postgres and not a warehouse

We evaluated Snowflake in 2024 and rejected it: our access pattern is
point-lookup dominated, and the latency budget for the query service is 150 ms
p99, which a warehouse cannot meet.
