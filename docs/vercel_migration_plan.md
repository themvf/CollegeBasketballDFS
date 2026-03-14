# Streamlit to Vercel Migration Plan

## Goal

Replace the Streamlit admin console with a production frontend on Vercel while keeping lineup generation, ingestion, postmortems, and data processing in Python.

The desired split is:

- `apps/web`: Next.js App Router UI deployed to Vercel
- `apps/api`: FastAPI orchestration layer deployed to Cloud Run
- `src/college_basketball_dfs`: shared Python domain logic and workers
- GCS: source-of-truth object storage for slate files, injuries, projections, lineup artifacts, and postmortem packets
- Firestore or Postgres: durable job state and run metadata

## Current Repo Baseline

### Existing frontend

The repo already has a migration starter in [apps/web](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/web) with:

- an app shell and canonical navigation in [navigation.ts](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/web/lib/navigation.ts)
- a lineup job UI in [lineup-generator-view.tsx](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/web/components/lineup-generator-view.tsx)
- page routes under [apps/web/app](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/web/app)

### Existing backend

The repo already has a FastAPI starter in [apps/api/main.py](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/api/main.py) with:

- review/read endpoints
- file upload endpoints
- async lineup job endpoints

### Existing Python domain logic

The production logic already lives in reusable modules:

- optimizer and pool building in [cbb_dk_optimizer.py](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/src/college_basketball_dfs/cbb_dk_optimizer.py)
- async lineup job orchestration in [cbb_lineup_jobs.py](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/src/college_basketball_dfs/cbb_lineup_jobs.py)
- review payload builders in [cbb_api_service.py](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/src/college_basketball_dfs/cbb_api_service.py)

## Architecture Principles

### 1. Backend owns business logic

The web app must not reimplement lineup rules, projection logic, or review math. It only:

- collects user input
- renders system state
- starts backend jobs
- polls status
- displays artifacts and diagnostics

### 2. Uploaded DK slate is the primary active slate

The product direction is now DK-slate-first. RotoWire can remain optional enrichment, not a hard readiness dependency.

### 3. Long-running work stays out of Vercel

Do not run these inside Vercel functions:

- lineup generation
- postmortem building
- tournament review scoring
- large CSV processing
- backfills

These remain Python jobs behind the API.

### 4. The UI should be workflow-first

The current Streamlit pain comes from mixing:

- admin repair tools
- ingestion
- modeling controls
- job execution
- analytics review

The new UI should treat these as guided flows with explicit state and blockers.

## Target Product Surface

### Primary workflow: Generate Lineup

This should remain the main user path and map to the existing nav model in [navigation.ts](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/web/lib/navigation.ts):

1. `Game Data`
2. `Prop Data`
3. `Backfill`
4. `DK Slate`
5. `Injuries`
6. `Slate + Vegas`
7. `Lineup Generator`
8. `Projection Review`
9. `Tournament Review`

### Secondary workflows

- `Player and Team Review`
- `Agentic Review`
- `Vegas Review`

### Internal-only workflows

These should move behind an admin mode instead of sitting in the main user path:

- alias repair
- RotoWire diagnostics
- registry mismatch tables
- manual override editors
- backfill operations

## Page-by-Page UX Model

### `DK Slate`

Purpose:

- upload DK slate for `date + slate_key`
- display active-slate readiness
- show counts, source metadata, and any optional enrichment status

Must show:

- selected date
- selected slate key
- DK slate upload status
- active rows
- active source
- last upload timestamp
- optional enrichment status

Should not show by default:

- raw registry mismatch dumps
- repair editors
- RotoWire-specific configuration

### `Injuries`

Purpose:

- upload date-scoped injury feed
- manage manual overrides

Must show:

- feed rows
- manual rows
- effective rows
- last feed upload status
- last manual override save status

### `Slate + Vegas`

Purpose:

- preview the active player pool
- inspect game environments
- verify projection inputs

Must show:

- player pool count
- slate games
- injury removals
- bookmaker source
- optional LineupStarter coverage

### `Lineup Generator`

Purpose:

- collect lineup intent
- launch async lineup jobs
- track run status
- download artifacts

Must show:

- selected date and slate
- lineup model or all-models mode
- contest profile summary
- player pool readiness
- job status timeline
- artifacts list
- generated summary table

Controls should stay limited to:

- contest type
- field size
- entry limit
- lineup count
- model/version selection
- seed
- max salary left
- global max exposure
- include teams
- player locks, excludes, exposure caps

Backend-managed policy should stay out of the UI:

- projection recency
- calibration knobs
- ownership guardrail thresholds
- uncertainty shrink tuning
- low-own bucket formula
- salary-left policy
- spike overlap
- phantom promotion lookback

## Backend Domain Boundaries

### API layer responsibilities

The API service should be thin and do only:

- request validation
- auth and permission checks
- storage calls
- job creation
- payload reads for the UI

### Worker responsibilities

The worker layer should do:

- slate loading
- pool building
- lineup generation
- phantom review
- postmortem packet creation
- artifact writes

### Storage responsibilities

GCS remains the blob/object store for:

- DK slates
- injuries
- LineupStarter priors
- supplements
- projections snapshots
- lineup run artifacts
- phantom review outputs
- tournament review uploads
- postmortem packets

Firestore or Postgres should hold durable metadata for:

- jobs
- lineup runs
- artifact manifests
- audit history
- user defaults

## API Contract Plan

### Existing endpoints already in repo

Current FastAPI surface in [apps/api/main.py](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/api/main.py):

- `GET /health`
- `GET /v1/rotowire/slates`
- `GET /v1/registry/coverage`
- `GET /v1/vegas/game-lines`
- `GET /v1/vegas/market-context`
- `GET /v1/vegas/prop-data`
- `GET /v1/ops/cache-coverage`
- `GET /v1/injuries/review`
- `POST /v1/injuries/feed/upload`
- `POST /v1/injuries/manual/upload`
- `GET /v1/reviews/projection`
- `POST /v1/reviews/projection/ownership/upload`
- `GET /v1/reviews/tournament`
- `POST /v1/reviews/tournament/standings/upload`
- `POST /v1/lineups/generate`
- `GET /v1/lineups/jobs/{job_id}`
- `GET /v1/lineups/jobs/{job_id}/artifacts`
- `GET /v1/lineups/jobs/{job_id}/artifacts/{artifact_name}`

### Recommended endpoint groups

#### Slate context

- `GET /v1/slates/{date}/{slate_key}/status`
- `GET /v1/slates/{date}/{slate_key}/pool-summary`
- `POST /v1/slates/{date}/{slate_key}/dk-slate/upload`
- `GET /v1/slates/{date}/{slate_key}/dk-slate`

#### Injuries

- `GET /v1/slates/{date}/injuries`
- `POST /v1/slates/{date}/injuries/feed/upload`
- `POST /v1/slates/{date}/injuries/manual/upload`

#### Priors and supplements

- `GET /v1/slates/{date}/{slate_key}/lineupstarter`
- `POST /v1/slates/{date}/{slate_key}/lineupstarter/upload`
- `GET /v1/slates/{date}/{slate_key}/supplements`
- `POST /v1/slates/{date}/{slate_key}/supplements/{slot}/upload`

#### Lineup jobs

- `POST /v1/lineup-jobs`
- `GET /v1/lineup-jobs/{job_id}`
- `GET /v1/lineup-jobs/{job_id}/artifacts`
- `GET /v1/lineup-runs/{run_id}`
- `GET /v1/lineup-runs?date=YYYY-MM-DD&slate_key=main`

#### Phantom review and postmortems

- `POST /v1/phantom-jobs`
- `GET /v1/phantom-jobs/{job_id}`
- `GET /v1/postmortems/{date}`
- `GET /v1/postmortems/{date}/{contest_id}`

### Contract design rules

- every endpoint must be date/slate explicit
- avoid hidden state from session variables
- uploads return storage metadata and parsed row counts
- job endpoints return a consistent status envelope
- artifact endpoints always return manifest-style metadata

## Job Model

### Required job schema

Each async job should store:

- `job_id`
- `job_type`
- `status`
- `created_at`
- `started_at`
- `completed_at`
- `requested_by`
- `selected_date`
- `slate_key`
- `request_payload`
- `progress_pct`
- `message`
- `error_code`
- `error_message`
- `artifact_manifest`
- `run_id`

### Status values

Use a small fixed state machine:

- `queued`
- `running`
- `succeeded`
- `failed`
- `cancelled`

### Job types

- `lineup_generate`
- `phantom_review`
- `postmortem_build`
- `props_import`
- `game_import`
- `backfill`

## Artifact Model

Each completed run should expose a manifest like:

- `run_id`
- `selected_date`
- `slate_key`
- `version_keys`
- `settings_snapshot`
- `artifact_items[]`

Each artifact item should include:

- `name`
- `filename`
- `content_type`
- `size_bytes`
- `storage_path`
- `download_url`
- `created_at`

For lineup runs, standard artifacts should be:

- `summary.csv`
- `slots.csv`
- `dk_upload.csv`
- `warnings.json`
- `run_meta.json`
- `all_versions.csv`

## Deployment Layout

### Vercel

Deploy [apps/web](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/web) to Vercel.

Environment variables:

- `NEXT_PUBLIC_API_BASE_URL`
- auth provider settings if added later

### Cloud Run API

Deploy [apps/api](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/api) as the public backend.

Environment variables:

- `CBB_GCS_BUCKET`
- `GCP_PROJECT`
- `GCP_SERVICE_ACCOUNT_JSON` or workload identity
- any model/review provider keys

### Cloud Run Jobs or worker service

Use a dedicated worker deployment for:

- lineup generation
- postmortem creation
- phantom review

This can start as:

- API enqueue -> worker process

Then harden to:

- API enqueue -> Pub/Sub or task queue -> worker

### Storage and metadata

- GCS for blobs
- Firestore for job state if you want low-friction GCP-native metadata
- Postgres if you want richer querying, user state, and audit history

## Auth Recommendation

For the frontend:

- Vercel Auth, Clerk, or Auth0 are all viable

For the API:

- validate bearer tokens in FastAPI
- map users to roles

Minimum roles:

- `admin`
- `operator`
- `viewer`

## Recommended Rollout

### Phase 1: finish the foundation

Implement first:

- `slate status` endpoint
- `lineup runs` list/read endpoints
- durable job metadata store
- standardized artifact manifest

### Phase 2: make the web lineup flow production-usable

Upgrade the existing lineup page in [lineup-generator-view.tsx](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/web/components/lineup-generator-view.tsx) to support:

- date and slate context
- team include filter
- locks/excludes/exposure caps
- saved-run list
- artifact summary cards
- explicit blockers from slate readiness

### Phase 3: port DK Slate and Injuries fully

These should be the next full web replacements because they are prerequisites for lineup generation.

### Phase 4: move review workflows

Port:

- projection review
- tournament review
- phantom review
- postmortem read views

### Phase 5: retire Streamlit

Keep Streamlit only as internal repair tooling until:

- web flow covers daily ops
- job runner is durable
- upload and artifact workflows are stable

## Concrete Backlog for This Repo

### Backend

1. Add `GET /v1/slates/{date}/{slate_key}/status` backed by `resolve_active_slate_context`.
2. Add `GET /v1/lineup-runs` and `GET /v1/lineup-runs/{run_id}`.
3. Move `LineupJobManager` off local-only metadata storage to Firestore or Postgres.
4. Standardize job responses so all async jobs share the same shape.
5. Remove remaining implicit RotoWire assumptions from public API request models.

### Frontend

1. Build a shared `SlateContextBar` component.
2. Build a `ReadinessCard` component that renders blockers and prerequisites.
3. Replace the current lineup request form with a richer job launcher using the real control surface.
4. Add `Saved Runs` and `Artifacts` screens to the `Generate Lineup` workflow.
5. Hide internal repair tools behind an admin-only section.

### Data and ops

1. Define lifecycle rules for large CSV artifacts in GCS.
2. Add structured logs for API requests and job transitions.
3. Add error notifications for failed lineup jobs.
4. Snapshot effective lineup settings into every run manifest.

## Definition of Done

The migration is ready for primary use when all of these are true:

- lineup generation is fully usable from Next.js
- DK Slate and Injuries are fully usable from Next.js
- saved runs and artifact downloads work from the web app
- async jobs are durable across restarts
- date/slate context is explicit in all major flows
- Streamlit is no longer required for normal slate prep and lineup generation

## Immediate Recommendation

The next implementation work should be:

1. build `slate status` and `saved runs` API endpoints
2. upgrade the existing web lineup page to consume them
3. port DK Slate upload/status into web
4. port Injuries into web

That is the shortest path to replacing the most painful Streamlit workflow while preserving the existing Python optimizer stack.
