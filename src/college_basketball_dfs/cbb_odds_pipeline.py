from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_ncaa import parse_iso_date, prior_day
from college_basketball_dfs.cbb_odds import OddsApiClient, flatten_odds_payload
from college_basketball_dfs.cbb_transform import rows_to_csv_text


def run_cbb_odds_pipeline(
    game_date: date,
    bucket_name: str | None = None,
    odds_api_key: str | None = None,
    odds_base_url: str = "https://api.the-odds-api.com/v4",
    sport_key: str = "basketball_ncaab",
    regions: str = "us",
    markets: str = "h2h,spreads,totals",
    historical_mode: bool = False,
    historical_snapshot_time: str | None = None,
    force_refresh: bool = False,
    local_output_dir: str | None = None,
    gcp_project: str | None = None,
    gcp_service_account_json: str | None = None,
    gcp_service_account_json_b64: str | None = None,
    store: Any | None = None,
) -> dict[str, Any]:
    if store is None:
        bucket = bucket_name or os.getenv("CBB_GCS_BUCKET", "").strip()
        if not bucket:
            raise ValueError("GCS bucket is required. Set --bucket or CBB_GCS_BUCKET.")
        client = build_storage_client(
            service_account_json=gcp_service_account_json,
            service_account_json_b64=gcp_service_account_json_b64,
            project=gcp_project,
        )
        store = CbbGcsStore(bucket_name=bucket, client=client)

    payload: dict[str, Any] | None = None
    odds_cache_hit = False

    if not force_refresh:
        payload = store.read_odds_json(game_date)
        odds_cache_hit = payload is not None

    if payload is None:
        key = odds_api_key or os.getenv("THE_ODDS_API_KEY", "").strip()
        if not key:
            raise ValueError("The Odds API key is required. Set --odds-api-key or THE_ODDS_API_KEY.")

        client = OddsApiClient(api_key=key, base_url=odds_base_url)
        try:
            events = client.fetch_game_odds(
                game_date=game_date,
                sport_key=sport_key,
                regions=regions,
                markets=markets,
                historical=historical_mode,
                historical_snapshot_time=historical_snapshot_time,
            )
        finally:
            client.close()

        payload = {
            "game_date": game_date.isoformat(),
            "sport_key": sport_key,
            "regions": regions,
            "markets": markets,
            "historical_mode": historical_mode,
            "historical_snapshot_time": historical_snapshot_time,
            "events": events,
        }
        odds_blob = store.write_odds_json(game_date, payload)
    else:
        odds_blob = store.odds_blob_name(game_date)

    rows = flatten_odds_payload(payload)
    csv_text = rows_to_csv_text(rows)
    odds_games_blob = store.write_odds_games_csv(game_date, csv_text)

    local_json_path: str | None = None
    local_csv_path: str | None = None
    if local_output_dir:
        out_dir = Path(local_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"cbb_odds_{game_date.isoformat()}.json"
        csv_path = out_dir / f"cbb_odds_{game_date.isoformat()}_games.csv"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        csv_path.write_text(csv_text, encoding="utf-8")
        local_json_path = str(json_path)
        local_csv_path = str(csv_path)

    return {
        "game_date": game_date.isoformat(),
        "odds_cache_hit": odds_cache_hit,
        "bucket_name": store.bucket_name,
        "odds_blob": odds_blob,
        "odds_games_blob": odds_games_blob,
        "event_count": len(payload.get("events", [])) if isinstance(payload.get("events"), list) else 0,
        "odds_game_rows": len(rows),
        "local_json_path": local_json_path,
        "local_csv_path": local_csv_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import The Odds API game odds to GCS.")
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD. Defaults to prior day.")
    parser.add_argument("--bucket", type=str, default=os.getenv("CBB_GCS_BUCKET", ""), help="GCS bucket name.")
    parser.add_argument(
        "--odds-api-key",
        type=str,
        default=os.getenv("THE_ODDS_API_KEY", ""),
        help="The Odds API key.",
    )
    parser.add_argument("--odds-base-url", type=str, default="https://api.the-odds-api.com/v4")
    parser.add_argument("--sport-key", type=str, default="basketball_ncaab")
    parser.add_argument("--regions", type=str, default="us")
    parser.add_argument("--markets", type=str, default="h2h,spreads,totals")
    parser.add_argument("--historical-mode", action="store_true", help="Use /historical endpoint with date snapshot.")
    parser.add_argument("--historical-snapshot-time", type=str, default=None, help="ISO UTC time for historical date param.")
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--local-output-dir", type=str, default=None)
    parser.add_argument("--gcp-project", type=str, default=os.getenv("GCP_PROJECT", ""))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    requested_date = parse_iso_date(args.date)
    game_date = requested_date or prior_day()
    summary = run_cbb_odds_pipeline(
        game_date=game_date,
        bucket_name=args.bucket,
        odds_api_key=args.odds_api_key,
        odds_base_url=args.odds_base_url,
        sport_key=args.sport_key,
        regions=args.regions,
        markets=args.markets,
        historical_mode=args.historical_mode,
        historical_snapshot_time=args.historical_snapshot_time,
        force_refresh=args.force_refresh,
        local_output_dir=args.local_output_dir,
        gcp_project=args.gcp_project or None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
