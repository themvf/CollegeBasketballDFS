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

## Odds Import (Totals, Spread, Moneyline)

Import game-level odds from The Odds API and cache to GCS:

```bash
python -m college_basketball_dfs.cbb_odds_pipeline --date 2026-02-12 --bucket your-gcs-bucket --odds-api-key your_key
```

The importer pulls `h2h,spreads,totals` for `basketball_ncaab` and writes:
- raw odds JSON
- normalized game-level odds CSV
- By default you can filter to FanDuel with bookmaker key `fanduel` (sidebar `Bookmakers Filter`).

Backfill odds across a season range:

```bash
python -m college_basketball_dfs.cbb_odds_backfill --start-date 2025-11-01 --end-date 2026-03-31 --bucket your-gcs-bucket --odds-api-key your_key
```

Notes:
- Odds season backfill uses The Odds API historical endpoint by default.
- This requires a paid plan with historical access.

## Player Props Import

Import player prop lines (default markets: points/rebounds/assists):

```bash
python -m college_basketball_dfs.cbb_props_pipeline --date 2026-02-12 --bucket your-gcs-bucket --odds-api-key your_key --historical-mode
```

Writes:
- raw props JSON: `cbb/props/YYYY-MM-DD.json`
- normalized props lines CSV: `cbb/props_lines/YYYY-MM-DD_props.csv`

Daily pregame workflow in Streamlit:
- Set `Props Date Preset` to `Tomorrow`
- Set `Props Fetch Mode` to `Pregame Live`
- Click `Run Props Import` before tip-off

## Streamlit Admin App

Run locally:

```bash
streamlit run dashboard/admin_app.py
```

Use sidebar controls to:
- run the cache pipeline
- run The Odds API import for selected date
- run player props import for selected date
- run odds season backfill for a date range
- run season backfill for a date range
- preview cached raw JSON and player CSV from GCS
- lookup a team and view game-level results table:
  - `Game Date`, `Venue`, `Home/Away`, `Team Score`, `Opponent`, `Opponent Score`, `W/L`
- view a game-level odds table for selected date:
  - spreads, totals, and moneylines

For Streamlit Community Cloud, set entrypoint:
- `streamlit run dashboard/admin_app.py`

Secrets template:
- `.streamlit/secrets.toml.example`

## GCS Paths

- Raw JSON: `cbb/raw/YYYY-MM-DD.json`
- Players CSV: `cbb/players/YYYY-MM-DD_players.csv`
- Odds Raw JSON: `cbb/odds/YYYY-MM-DD.json`
- Odds Games CSV: `cbb/odds_games/YYYY-MM-DD_odds.csv`
- Props Raw JSON: `cbb/props/YYYY-MM-DD.json`
- Props Lines CSV: `cbb/props_lines/YYYY-MM-DD_props.csv`
