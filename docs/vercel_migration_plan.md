# Streamlit to Vercel Migration Plan

## Objective

Move from the current monolithic Streamlit admin console to a Vercel-hosted web experience with:

- faster navigation
- predictable API contracts
- async job execution for heavy lineup workflows
- safer production deployment and observability

## Current Baseline

- UI entrypoint: `dashboard/admin_app.py` (~10,762 lines)
- Core logic already exists in reusable Python modules under `src/college_basketball_dfs`
- Main functional areas in Streamlit tabs:
  - game pipeline and props ingestion
  - season backfill
  - DK slate and registry management
  - RotoWire ingestion and mapping
  - lineup generation and exports
  - projection/tournament review analytics

## Target Architecture

- Frontend: Next.js App Router on Vercel (`apps/web`)
- API: FastAPI (`apps/api`) deployed separately (Cloud Run or equivalent)
- Data stores:
  - existing GCS bucket for pipeline data and artifacts
  - Postgres for job state, run metadata, and user audit events
- Async execution:
  - queue worker for lineup generation, backfills, and postmortem packet creation

## Why Not Run Heavy Workloads in Vercel Functions

Lineup generation and backfills are long-running and data-heavy. Vercel serverless functions are not the right place for multi-minute tasks and persistent job state. Vercel should host UI and lightweight orchestration calls.

## API Contract Phases

### Phase 1 (Implemented in this repo)

- `GET /health`
- `GET /v1/rotowire/slates`
- `GET /v1/registry/coverage`
- `POST /v1/registry/import-dk-slate`

Backed by `src/college_basketball_dfs/cbb_api_service.py` to avoid Streamlit imports.

### Phase 2

- `POST /v1/lineups/generate` -> returns job id
- `GET /v1/lineups/jobs/{job_id}`
- `GET /v1/lineups/jobs/{job_id}/artifacts`
- `POST /v1/review/postmortem/build`

### Phase 3

- Auth-protected endpoints
- per-user settings/profile defaults
- revisioned model and configuration snapshots

## UI Cutover Plan

### Slice A: Slate + Registry (Week 1)

- Build slate selection, coverage diagnostics, unresolved/conflict resolver UI
- Keep Streamlit as fallback

### Slice B: Lineup Generator (Weeks 2-3)

- Move all generator controls to React forms
- submit generation as async jobs
- stream status and expose downloads

### Slice C: Tournament Review + Diagnostics (Week 4)

- Port ownership calibration, false chalk, focus-game diagnostics
- keep packet export parity with Streamlit

### Slice D: Remove Streamlit Runtime Dependency (Week 5)

- freeze Streamlit UI to read-only mode
- switch user traffic to Vercel
- deprecate Streamlit deployment after parity signoff

## Parity Gates

Release is blocked unless all pass:

- For 3 consecutive slates:
  - same resolved DK ID count as Streamlit path
  - same lineup count and CSV schema
  - no regression in ownership diagnostic tables
- All critical APIs have:
  - request/response validation
  - structured logs
  - error budget monitoring

## Risks and Mitigations

- Risk: Name/team mismatches create bad DK ID resolutions
  - Mitigation: keep manual override workflow and audit each override
- Risk: Async lineup jobs fail silently
  - Mitigation: durable queue + run table + retry state + alerts
- Risk: Diverging business logic between UI and backend
  - Mitigation: only backend owns lineup/business logic; UI is presentation and orchestration

## Implementation Added in This Commit

- API service abstraction: `src/college_basketball_dfs/cbb_api_service.py`
- FastAPI migration starter: `apps/api`
- Next.js migration starter: `apps/web`

This is the starting point for replacing Streamlit tab by tab while preserving current modeling logic.
