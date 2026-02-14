from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

from college_basketball_dfs.cbb_backfill import iter_dates, parse_iso_date, season_start_for_date
from college_basketball_dfs.cbb_odds_pipeline import run_cbb_odds_pipeline


@dataclass
class OddsBackfillResult:
    start_date: str
    end_date: str
    total_dates: int
    success_dates: int
    failed_dates: int
    total_events: int
    total_odds_game_rows: int
    odds_cache_hits: int
    errors: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_dates": self.total_dates,
            "success_dates": self.success_dates,
            "failed_dates": self.failed_dates,
            "total_events": self.total_events,
            "total_odds_game_rows": self.total_odds_game_rows,
            "odds_cache_hits": self.odds_cache_hits,
            "errors": self.errors,
        }


def run_odds_season_backfill(
    start_date: date,
    end_date: date,
    runner: Callable[..., dict[str, Any]] = run_cbb_odds_pipeline,
    sleep_seconds: float = 0.0,
    stop_on_error: bool = False,
    **pipeline_kwargs: Any,
) -> OddsBackfillResult:
    dates = iter_dates(start_date, end_date)
    success_dates = 0
    failed_dates = 0
    total_events = 0
    total_odds_game_rows = 0
    odds_cache_hits = 0
    errors: list[dict[str, str]] = []

    for idx, game_date in enumerate(dates):
        try:
            summary = runner(game_date=game_date, **pipeline_kwargs)
            success_dates += 1
            total_events += int(summary.get("event_count", 0))
            total_odds_game_rows += int(summary.get("odds_game_rows", 0))
            if bool(summary.get("odds_cache_hit")):
                odds_cache_hits += 1
        except Exception as exc:
            failed_dates += 1
            errors.append({"date": game_date.isoformat(), "error": str(exc)})
            if stop_on_error:
                break

        if sleep_seconds > 0 and idx < len(dates) - 1:
            time.sleep(sleep_seconds)

    return OddsBackfillResult(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        total_dates=len(dates),
        success_dates=success_dates,
        failed_dates=failed_dates,
        total_events=total_events,
        total_odds_game_rows=total_odds_game_rows,
        odds_cache_hits=odds_cache_hits,
        errors=errors,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill The Odds API data across a date range to GCS.")
    parser.add_argument("--start-date", type=str, required=False, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=False, help="YYYY-MM-DD")
    parser.add_argument("--bucket", type=str, required=True, help="GCS bucket")
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
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--gcp-project", type=str, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    today = date.today()
    start = parse_iso_date(args.start_date) if args.start_date else season_start_for_date(today)
    end = parse_iso_date(args.end_date) if args.end_date else today

    result = run_odds_season_backfill(
        start_date=start,
        end_date=end,
        bucket_name=args.bucket,
        odds_api_key=args.odds_api_key,
        odds_base_url=args.odds_base_url,
        sport_key=args.sport_key,
        regions=args.regions,
        markets=args.markets,
        force_refresh=args.force_refresh,
        gcp_project=args.gcp_project,
        sleep_seconds=args.sleep_seconds,
        stop_on_error=args.stop_on_error,
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
