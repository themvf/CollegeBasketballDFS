# Cron Automation Plan

## Goal
Automate daily data imports for College Basketball DFS so manual terminal runs are not required.

## Target Schedule
- 6:00 AM America/New_York: game/backfill refresh (optional next step)
- 5:30 PM America/New_York: props import (current priority)

## What We Are Implementing
1. Cloud Run Job named `cbb-props-import`
2. Cloud Scheduler trigger at `30 17 * * *` in timezone `America/New_York`
3. Secret Manager storage for `THE_ODDS_API_KEY`
4. GCS writes to `collegebasketballdfs` bucket

## Current Status
- APIs enabled: Cloud Run, Cloud Scheduler, Cloud Build, Secret Manager
- Secret created: `the-odds-api-key`
- Cloud Run Job deployed from source at least once
- Execution still failing; detailed runtime log investigation is in progress

## Known Failure Context
Recent executions fail at runtime after container starts. We need to verify:
- deployed command/args are shell-expanded correctly
- `PYTHONPATH=src` is present for `python -m college_basketball_dfs.cbb_props_pipeline`
- markets are passed as one CSV string: `player_points,player_rebounds,player_assists`

## Diagnostic Commands
Use these to inspect a failed execution:

```bash
gcloud run jobs executions describe <execution_id> --region us-central1
```

```bash
gcloud logging read \
'resource.type="cloud_run_job" AND resource.labels.job_name="cbb-props-import" AND labels."run.googleapis.com/execution_name"="<execution_id>"' \
--project college-basketball-487322 \
--limit 100 \
--format='value(textPayload)'
```

```bash
gcloud run jobs describe cbb-props-import --region us-central1 \
  --format='yaml(template.template.containers)'
```

## Once Job Is Healthy
Create the scheduler:

```bash
gcloud scheduler jobs create http cbb-props-530pm-et \
  --location us-central1 \
  --schedule "30 17 * * *" \
  --time-zone "America/New_York" \
  --uri "https://run.googleapis.com/v2/projects/college-basketball-487322/locations/us-central1/jobs/cbb-props-import:run" \
  --http-method POST \
  --oauth-service-account-email cbb-scheduler-invoker@college-basketball-487322.iam.gserviceaccount.com \
  --oauth-token-scope "https://www.googleapis.com/auth/cloud-platform"
```

## Next Action
Fix Cloud Run job runtime failure first, then enable the daily 5:30 PM ET scheduler.
