from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from typing import Any

from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_ncaa import NcaaApiClient, fetch_games_with_boxscores, parse_iso_date, prior_day
from college_basketball_dfs.cbb_transform import flatten_games_payload, rows_to_csv_text


def run_cbb_cache_pipeline(
    game_date: date,
    bucket_name: str | None = None,
    ncaa_base_url: str = "https://ncaa-api.henrygd.me",
    sport: str = "basketball-men",
    division: str = "d1",
    conference: str = "all-conf",
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
    raw_cache_hit = False

    if not force_refresh:
        payload = store.read_raw_json(game_date)
        raw_cache_hit = payload is not None

    if payload is None:
        client = NcaaApiClient(base_url=ncaa_base_url)
        try:
            payload = fetch_games_with_boxscores(
                client=client,
                game_date=game_date,
                sport=sport,
                division=division,
                conference=conference,
            )
        finally:
            client.close()
        raw_blob = store.write_raw_json(game_date, payload)
    else:
        raw_blob = store.raw_blob_name(game_date)

    rows = flatten_games_payload(payload)
    csv_text = rows_to_csv_text(rows)
    players_blob = store.write_players_csv(game_date, csv_text)

    local_json_path: str | None = None
    local_csv_path: str | None = None
    if local_output_dir:
        out_dir = Path(local_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"cbb_{game_date.isoformat()}.json"
        csv_path = out_dir / f"cbb_{game_date.isoformat()}_players.csv"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        csv_path.write_text(csv_text, encoding="utf-8")
        local_json_path = str(json_path)
        local_csv_path = str(csv_path)

    return {
        "game_date": game_date.isoformat(),
        "raw_cache_hit": raw_cache_hit,
        "bucket_name": store.bucket_name,
        "raw_blob": raw_blob,
        "players_blob": players_blob,
        "game_count": int(payload.get("game_count", 0)),
        "boxscore_success_count": int(payload.get("boxscore_success_count", 0)),
        "boxscore_failure_count": int(payload.get("boxscore_failure_count", 0)),
        "player_row_count": len(rows),
        "local_json_path": local_json_path,
        "local_csv_path": local_csv_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NCAA CBB cache pipeline: GCS cache-first load, API fallback fetch, and player CSV export."
    )
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD. Defaults to prior day.")
    parser.add_argument("--bucket", type=str, default=os.getenv("CBB_GCS_BUCKET", ""), help="GCS bucket name.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("NCAA_API_BASE_URL", "https://ncaa-api.henrygd.me"),
        help="ncaa-api base URL.",
    )
    parser.add_argument("--sport", type=str, default="basketball-men")
    parser.add_argument("--division", type=str, default="d1")
    parser.add_argument("--conference", type=str, default="all-conf")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached raw JSON and refetch from API.")
    parser.add_argument(
        "--local-output-dir",
        type=str,
        default=None,
        help="Optional local directory for writing JSON+CSV artifacts.",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=os.getenv("GCP_PROJECT", ""),
        help="Optional Google Cloud project override.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    requested_date = parse_iso_date(args.date)
    game_date = requested_date or prior_day()
    summary = run_cbb_cache_pipeline(
        game_date=game_date,
        bucket_name=args.bucket,
        ncaa_base_url=args.base_url,
        sport=args.sport,
        division=args.division,
        conference=args.conference,
        force_refresh=args.force_refresh,
        local_output_dir=args.local_output_dir,
        gcp_project=args.gcp_project or None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
