from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Callable

from college_basketball_dfs.cbb_pipeline import run_cbb_cache_pipeline


def parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def season_start_for_date(today: date) -> date:
    if today.month >= 11:
        return date(today.year, 11, 1)
    return date(today.year - 1, 11, 1)


def iter_dates(start_date: date, end_date: date) -> list[date]:
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")
    days = (end_date - start_date).days
    return [start_date.fromordinal(start_date.toordinal() + offset) for offset in range(days + 1)]


@dataclass
class BackfillResult:
    start_date: str
    end_date: str
    total_dates: int
    success_dates: int
    failed_dates: int
    total_games: int
    total_player_rows: int
    raw_cache_hits: int
    errors: list[dict[str, str]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "total_dates": self.total_dates,
            "success_dates": self.success_dates,
            "failed_dates": self.failed_dates,
            "total_games": self.total_games,
            "total_player_rows": self.total_player_rows,
            "raw_cache_hits": self.raw_cache_hits,
            "errors": self.errors,
        }


def run_season_backfill(
    start_date: date,
    end_date: date,
    runner: Callable[..., dict[str, Any]] = run_cbb_cache_pipeline,
    sleep_seconds: float = 0.0,
    stop_on_error: bool = False,
    **pipeline_kwargs: Any,
) -> BackfillResult:
    dates = iter_dates(start_date, end_date)
    success_dates = 0
    failed_dates = 0
    total_games = 0
    total_player_rows = 0
    raw_cache_hits = 0
    errors: list[dict[str, str]] = []

    for idx, game_date in enumerate(dates):
        try:
            summary = runner(game_date=game_date, **pipeline_kwargs)
            success_dates += 1
            total_games += int(summary.get("game_count", 0))
            total_player_rows += int(summary.get("player_row_count", 0))
            if bool(summary.get("raw_cache_hit")):
                raw_cache_hits += 1
        except Exception as exc:
            failed_dates += 1
            errors.append({"date": game_date.isoformat(), "error": str(exc)})
            if stop_on_error:
                break

        if sleep_seconds > 0 and idx < len(dates) - 1:
            time.sleep(sleep_seconds)

    return BackfillResult(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        total_dates=len(dates),
        success_dates=success_dates,
        failed_dates=failed_dates,
        total_games=total_games,
        total_player_rows=total_player_rows,
        raw_cache_hits=raw_cache_hits,
        errors=errors,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill CBB season dates to GCS cache.")
    parser.add_argument("--start-date", type=str, required=False, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, required=False, help="YYYY-MM-DD")
    parser.add_argument("--bucket", type=str, required=True, help="GCS bucket")
    parser.add_argument("--base-url", type=str, default="https://ncaa-api.henrygd.me")
    parser.add_argument("--sport", type=str, default="basketball-men")
    parser.add_argument("--division", type=str, default="d1")
    parser.add_argument("--conference", type=str, default="all-conf")
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

    result = run_season_backfill(
        start_date=start,
        end_date=end,
        bucket_name=args.bucket,
        ncaa_base_url=args.base_url,
        sport=args.sport,
        division=args.division,
        conference=args.conference,
        force_refresh=args.force_refresh,
        gcp_project=args.gcp_project,
        sleep_seconds=args.sleep_seconds,
        stop_on_error=args.stop_on_error,
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
