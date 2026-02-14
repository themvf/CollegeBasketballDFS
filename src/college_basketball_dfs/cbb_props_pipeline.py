from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_ncaa import parse_iso_date, prior_day
from college_basketball_dfs.cbb_odds import OddsApiClient, flatten_player_props_payload
from college_basketball_dfs.cbb_transform import rows_to_csv_text


def _extract_event_summaries(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for event in events:
        event_id = event.get("id")
        if not event_id:
            continue
        summaries.append(
            {
                "id": event_id,
                "commence_time": event.get("commence_time"),
                "home_team": event.get("home_team"),
                "away_team": event.get("away_team"),
            }
        )
    return summaries


def _resolve_event_snapshot_time(
    event_summary: dict[str, Any],
    game_date: date,
    explicit_snapshot_time: str | None,
) -> str:
    if explicit_snapshot_time:
        return explicit_snapshot_time

    commence_raw = str(event_summary.get("commence_time") or "").strip()
    if commence_raw:
        try:
            parsed = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            # Player props are often posted closer to tipoff; sample just before commence.
            snapshot = parsed.astimezone(timezone.utc) - timedelta(minutes=15)
            return snapshot.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass

    return f"{game_date.isoformat()}T23:45:00Z"


def _event_body(event: dict[str, Any]) -> dict[str, Any]:
    wrapped = event.get("data")
    if isinstance(wrapped, dict):
        return wrapped
    return event


def _props_diagnostics(payload: dict[str, Any], requested_markets_csv: str) -> dict[str, int]:
    events = payload.get("events")
    if not isinstance(events, list):
        return {
            "events_with_bookmakers": 0,
            "events_with_requested_markets": 0,
            "total_bookmakers": 0,
            "total_requested_markets": 0,
            "total_outcomes": 0,
        }

    requested = {x.strip().lower() for x in requested_markets_csv.split(",") if x.strip()}
    events_with_bookmakers = 0
    events_with_requested_markets = 0
    total_bookmakers = 0
    total_requested_markets = 0
    total_outcomes = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        body = _event_body(event)
        bookmakers = body.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue
        if bookmakers:
            events_with_bookmakers += 1
        total_bookmakers += len(bookmakers)

        event_has_requested = False
        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue
            markets = bookmaker.get("markets")
            if not isinstance(markets, list):
                continue
            for market in markets:
                if not isinstance(market, dict):
                    continue
                market_key = str(market.get("key") or "").strip().lower()
                if requested and market_key not in requested:
                    continue
                event_has_requested = True
                total_requested_markets += 1
                outcomes = market.get("outcomes")
                if isinstance(outcomes, list):
                    total_outcomes += len(outcomes)

        if event_has_requested:
            events_with_requested_markets += 1

    return {
        "events_with_bookmakers": events_with_bookmakers,
        "events_with_requested_markets": events_with_requested_markets,
        "total_bookmakers": total_bookmakers,
        "total_requested_markets": total_requested_markets,
        "total_outcomes": total_outcomes,
    }


def run_cbb_props_pipeline(
    game_date: date,
    bucket_name: str | None = None,
    odds_api_key: str | None = None,
    odds_base_url: str = "https://api.the-odds-api.com/v4",
    sport_key: str = "basketball_ncaab",
    regions: str = "us",
    markets: str = "player_points,player_rebounds,player_assists",
    bookmakers: str | None = None,
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
    props_cache_hit = False

    if not force_refresh:
        payload = store.read_props_json(game_date)
        props_cache_hit = payload is not None

    key = odds_api_key or os.getenv("THE_ODDS_API_KEY", "").strip()
    if payload is None and not key:
        raise ValueError("The Odds API key is required. Set --odds-api-key or THE_ODDS_API_KEY.")

    if payload is None:
        client = OddsApiClient(api_key=key, base_url=odds_base_url)
        try:
            cached_game_odds = store.read_odds_json(game_date)
            events_seed: list[dict[str, Any]] = []
            if cached_game_odds and isinstance(cached_game_odds.get("events"), list):
                events_seed = [x for x in cached_game_odds["events"] if isinstance(x, dict)]
            else:
                events_seed = client.fetch_game_odds(
                    game_date=game_date,
                    sport_key=sport_key,
                    regions=regions,
                    markets="h2h",
                    bookmakers=bookmakers,
                    historical=historical_mode,
                    historical_snapshot_time=historical_snapshot_time,
                )

            event_summaries = _extract_event_summaries(events_seed)
            event_payloads: list[dict[str, Any]] = []
            for event_summary in event_summaries:
                event_id = str(event_summary["id"])
                event_snapshot_time = (
                    _resolve_event_snapshot_time(event_summary, game_date, historical_snapshot_time)
                    if historical_mode
                    else None
                )
                props_event = client.fetch_event_odds(
                    event_id=event_id,
                    sport_key=sport_key,
                    regions=regions,
                    markets=markets,
                    bookmakers=bookmakers,
                    historical=historical_mode,
                    historical_snapshot_time=event_snapshot_time,
                )
                props_event.setdefault("id", event_summary.get("id"))
                props_event.setdefault("commence_time", event_summary.get("commence_time"))
                props_event.setdefault("home_team", event_summary.get("home_team"))
                props_event.setdefault("away_team", event_summary.get("away_team"))
                event_payloads.append(props_event)
        finally:
            client.close()

        payload = {
            "game_date": game_date.isoformat(),
            "sport_key": sport_key,
            "regions": regions,
            "markets": markets,
            "bookmakers": bookmakers,
            "historical_mode": historical_mode,
            "historical_snapshot_time": historical_snapshot_time,
            "events": event_payloads,
        }
        props_blob = store.write_props_json(game_date, payload)
    else:
        props_blob = store.props_blob_name(game_date)

    rows = flatten_player_props_payload(payload)
    csv_text = rows_to_csv_text(rows)
    props_lines_blob = store.write_props_lines_csv(game_date, csv_text)
    diagnostics = _props_diagnostics(payload, markets)

    local_json_path: str | None = None
    local_csv_path: str | None = None
    if local_output_dir:
        out_dir = Path(local_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"cbb_props_{game_date.isoformat()}.json"
        csv_path = out_dir / f"cbb_props_{game_date.isoformat()}_lines.csv"
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        csv_path.write_text(csv_text, encoding="utf-8")
        local_json_path = str(json_path)
        local_csv_path = str(csv_path)

    return {
        "game_date": game_date.isoformat(),
        "props_cache_hit": props_cache_hit,
        "historical_mode": historical_mode,
        "markets": markets,
        "bookmakers": bookmakers,
        "bucket_name": store.bucket_name,
        "props_blob": props_blob,
        "props_lines_blob": props_lines_blob,
        "event_count": len(payload.get("events", [])) if isinstance(payload.get("events"), list) else 0,
        "prop_rows": len(rows),
        "events_with_bookmakers": diagnostics["events_with_bookmakers"],
        "events_with_requested_markets": diagnostics["events_with_requested_markets"],
        "total_bookmakers": diagnostics["total_bookmakers"],
        "total_requested_markets": diagnostics["total_requested_markets"],
        "total_outcomes": diagnostics["total_outcomes"],
        "local_json_path": local_json_path,
        "local_csv_path": local_csv_path,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import The Odds API player props to GCS.")
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
    parser.add_argument("--markets", type=str, default="player_points,player_rebounds,player_assists")
    parser.add_argument(
        "--bookmakers",
        type=str,
        default=os.getenv("CBB_ODDS_BOOKMAKERS", ""),
        help="Optional bookmaker key filter (example: fanduel).",
    )
    parser.add_argument("--historical-mode", action="store_true")
    parser.add_argument("--historical-snapshot-time", type=str, default=None)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--local-output-dir", type=str, default=None)
    parser.add_argument("--gcp-project", type=str, default=os.getenv("GCP_PROJECT", ""))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    requested_date = parse_iso_date(args.date)
    game_date = requested_date or prior_day()
    summary = run_cbb_props_pipeline(
        game_date=game_date,
        bucket_name=args.bucket,
        odds_api_key=args.odds_api_key,
        odds_base_url=args.odds_base_url,
        sport_key=args.sport_key,
        regions=args.regions,
        markets=args.markets,
        bookmakers=(args.bookmakers or None),
        historical_mode=args.historical_mode,
        historical_snapshot_time=args.historical_snapshot_time,
        force_refresh=args.force_refresh,
        local_output_dir=args.local_output_dir,
        gcp_project=args.gcp_project or None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
