# Cron Automation Runbook

## Goal
Run the props import automatically every day and have Streamlit read the new data from GCS.

## Current Schedule
- Job: `cbb-props-import`
- Trigger: `cbb-props-530pm-et`
- Cron: `30 17 * * *`
- Timezone: `America/New_York`

## How It Works (End-to-End)
1. Cloud Scheduler sends an authenticated HTTP POST to:
   `https://run.googleapis.com/v2/projects/college-basketball-487322/locations/us-central1/jobs/cbb-props-import:run`
2. The Scheduler service account `cbb-scheduler-invoker@college-basketball-487322.iam.gserviceaccount.com` is used for OAuth and has `roles/run.invoker` on the job.
3. Cloud Run Job starts one execution using service account `cbb-runner@college-basketball-487322.iam.gserviceaccount.com`.
4. Inside the container, `/bin/sh` runs:
   - `DATE=$(TZ=America/New_York date +%F)`
   - `/layers/google.python.uv/uv-dependencies/.venv/bin/python -m college_basketball_dfs.cbb_props_pipeline --date $DATE --bucket $CBB_GCS_BUCKET --bookmakers $CBB_ODDS_BOOKMAKERS --gcp-project $GCP_PROJECT`
5. `THE_ODDS_API_KEY` is injected from Secret Manager secret `the-odds-api-key`.
6. Pipeline writes to GCS bucket `collegebasketballdfs`:
   - raw props JSON: `cbb/props/YYYY-MM-DD.json`
   - normalized props CSV: `cbb/props_lines/YYYY-MM-DD_props.csv`
7. Streamlit reads those files from GCS and renders tables/charts.

## How Streamlit Cloud Gets Updated
Important: This cron pipeline updates data, not app code.

- Data update path:
  - Cloud Run writes new files to GCS.
  - Streamlit reads from GCS on demand.
  - Props loader uses `@st.cache_data(ttl=600)`, so UI can lag by up to ~10 minutes for the same inputs.
- Code update path:
  - Streamlit code changes only appear after pushing new code to the branch your Streamlit app is deployed from.
  - Cloud Run job code changes only appear after redeploying the Cloud Run job from source.

So, cron does not redeploy Streamlit Cloud. It refreshes the data that Streamlit reads.

## Streamlit Secrets Needed
For Streamlit Cloud to read the same data, set these app secrets (or env vars):
- `cbb_gcs_bucket` (or `CBB_GCS_BUCKET`)
- `gcp_project` (or `GCP_PROJECT`)
- `gcp_service_account_json` or `gcp_service_account_json_b64` (if not using ADC)
- `the_odds_api_key` (only needed if running props import from Streamlit UI)

## Verify Health (Quick Checks)
```powershell
$GCLOUD="C:\Users\joshb\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

& $GCLOUD run jobs describe cbb-props-import `
  --project college-basketball-487322 `
  --region us-central1 `
  --format='yaml(status.latestCreatedExecution,spec.template.spec.template.spec.containers[0].command,spec.template.spec.template.spec.containers[0].args)'

& $GCLOUD scheduler jobs describe cbb-props-530pm-et `
  --project college-basketball-487322 `
  --location us-central1 `
  --format='yaml(state,schedule,timeZone,httpTarget.oauthToken.serviceAccountEmail,httpTarget.uri)'
```

## Run a Smoke Test
```powershell
$GCLOUD="C:\Users\joshb\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

& $GCLOUD scheduler jobs run cbb-props-530pm-et `
  --project college-basketball-487322 `
  --location us-central1

& $GCLOUD run jobs describe cbb-props-import `
  --project college-basketball-487322 `
  --region us-central1 `
  --format='yaml(status.latestCreatedExecution)'
```

When `completionStatus` is `EXECUTION_SUCCESS`, data refresh succeeded.

## Deploy/Update Job Code (After Local Code Changes)
Use this when you change Python code and want Cloud Run to use the new version:

```powershell
$GCLOUD="C:\Users\joshb\AppData\Local\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

& $GCLOUD run jobs deploy cbb-props-import `
  --project college-basketball-487322 `
  --region us-central1 `
  --source . `
  --command /bin/sh `
  --args='-c,DATE=$(TZ=America/New_York date +%F); /layers/google.python.uv/uv-dependencies/.venv/bin/python -m college_basketball_dfs.cbb_props_pipeline --date $DATE --bucket $CBB_GCS_BUCKET --bookmakers $CBB_ODDS_BOOKMAKERS --gcp-project $GCP_PROJECT' `
  --set-env-vars 'PYTHONPATH=src,CBB_GCS_BUCKET=collegebasketballdfs,GCP_PROJECT=college-basketball-487322,CBB_ODDS_BOOKMAKERS=fanduel' `
  --set-secrets 'THE_ODDS_API_KEY=the-odds-api-key:latest' `
  --task-timeout=1800s `
  --max-retries=1
```

## Common Failures
- `python: not found` or `python3: not found`
  - Cause: interpreter not on PATH in this Cloud Run image.
  - Fix: use `/layers/google.python.uv/uv-dependencies/.venv/bin/python`.
- `ModuleNotFoundError: requests`
  - Cause: using interpreter outside uv venv where deps are installed.
  - Fix: use the same uv venv python path above.
- `Service account ... does not exist`
  - Cause: scheduler invoker SA missing.
  - Fix: create SA, then bind `roles/run.invoker` on job.
