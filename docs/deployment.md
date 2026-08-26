# Deployment Runbook

## Environments

| Environment | Cluster            | Deploys from |
| ----------- | ------------------ | ------------ |
| dev         | `eks-dev-euc1`     | every merge to `main` |
| staging     | `eks-staging-euc1` | tagged `v*-rc*` |
| production  | `eks-prod-euc1`    | tagged `v*` after sign-off |

## Release process

1. Cut a release branch from `main`.
2. Tag `vX.Y.Z-rc1`; CI deploys to staging automatically.
3. Run the smoke suite against staging.
4. Get sign-off from the on-call engineer, then tag `vX.Y.Z`.

Production deploys are blocked between Friday 15:00 UTC and Monday 08:00 UTC
unless the change is tagged `hotfix`.

## Rollback

`make rollback ENV=production` redeploys the previous image tag. It does not
revert database migrations — forward-fix those instead.
