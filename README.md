# CollegeBasketballDFS

College Basketball DFS data workflow with:
- NCAA open API fetch (`ncaa-api`)
- Google Cloud Storage cache-first pipeline
- Player-level CSV transform
- Streamlit admin app

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and set bucket/credentials as needed.

## Fetch Prior-Day Games + Box Scores

```bash
python -m college_basketball_dfs.cbb_ncaa --summary-only
python -m college_basketball_dfs.cbb_ncaa --json-out data/cbb_prior_day.json
```

## Flatten to Player CSV

```bash
python -m college_basketball_dfs.cbb_transform --input-json data/cbb_prior_day.json --output-csv data/cbb_prior_day_players.csv
```

## GCS Cache Pipeline

Cache-first flow:
1. Read `cbb/raw/YYYY-MM-DD.json` from GCS
2. If missing, fetch from NCAA API and write raw JSON
3. Write `cbb/players/YYYY-MM-DD_players.csv`

```bash
python -m college_basketball_dfs.cbb_pipeline --date 2026-02-12 --bucket your-gcs-bucket
python -m college_basketball_dfs.cbb_pipeline --date 2026-02-12 --bucket your-gcs-bucket --force-refresh
```

## Season Backfill To GCS

Backfill an entire season date range into your bucket:

```bash
python -m college_basketball_dfs.cbb_backfill --start-date 2025-11-01 --end-date 2026-03-31 --bucket your-gcs-bucket
```

Optional throttling and fail-fast:

```bash
python -m college_basketball_dfs.cbb_backfill --start-date 2025-11-01 --end-date 2026-03-31 --bucket your-gcs-bucket --sleep-seconds 0.2 --stop-on-error
```

## Streamlit Admin App

Run locally:

```bash
streamlit run dashboard/admin_app.py
```

Use sidebar controls to:
- run the cache pipeline
- run season backfill for a date range
- preview cached raw JSON and player CSV from GCS

For Streamlit Community Cloud, set entrypoint:
- `streamlit run dashboard/admin_app.py`

Secrets template:
- `.streamlit/secrets.toml.example`

## GCS Paths

- Raw JSON: `cbb/raw/YYYY-MM-DD.json`
- Players CSV: `cbb/players/YYYY-MM-DD_players.csv`
