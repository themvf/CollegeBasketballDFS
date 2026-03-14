# Cloud Run API Deploy

This deploys the FastAPI backend in [apps/api/main.py](/abs/path/c:/Docs/_AI%20Python%20Projects/CollegeBasketballDFS/apps/api/main.py) so the Vercel frontend can talk to a real API.

## What This Creates

- a public Cloud Run service for the API
- environment variables:
  - `CBB_GCS_BUCKET`
  - `GCP_PROJECT`
- optional Secret Manager-backed `GCP_SERVICE_ACCOUNT_JSON`

After deploy, use the Cloud Run URL as:

- `NEXT_PUBLIC_API_BASE_URL` in Vercel

## Prerequisites

Install and authenticate `gcloud`:

```powershell
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Enable required services once:

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com
```

## One-Command Deploy

From the repo root, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy_api_cloud_run.ps1 `
  -ProjectId YOUR_PROJECT_ID `
  -Region us-central1 `
  -ServiceName college-basketball-dfs-api `
  -BucketName YOUR_GCS_BUCKET `
  -ServiceAccountJsonPath .\college-basketball-487322-384e69b56597.json
```

If you already use workload identity on Cloud Run, omit `-ServiceAccountJsonPath`.

## What The Script Does

1. resolves your GCP project
2. uploads the service-account JSON to Secret Manager if you pass `-ServiceAccountJsonPath`
3. grants the Cloud Run runtime service account access to that secret
4. builds the API container from the repo root
5. deploys Cloud Run with the required env vars
6. prints:
   - the service URL
   - `/health`
   - the slate-status URL
7. runs smoke checks against both endpoints

## Expected Smoke Test URLs

After deploy, these should work:

```text
https://YOUR_CLOUD_RUN_URL/health
https://YOUR_CLOUD_RUN_URL/v1/slates/2026-03-14/main/status?contest_type=Classic&slate_name=All
```

## Vercel

In your Vercel project settings, add:

```text
NEXT_PUBLIC_API_BASE_URL=https://YOUR_CLOUD_RUN_URL
```

Then redeploy Vercel.

## Notes

- The repo root `Dockerfile` is for the Cloud Run API deployment path.
- `.gcloudignore` and `.dockerignore` intentionally exclude your large local CSV/data workspace files so Cloud Build does not upload them.
- If the `health` endpoint works but slate status fails, the deploy is fine and the remaining issue is backend config or missing GCS data.
