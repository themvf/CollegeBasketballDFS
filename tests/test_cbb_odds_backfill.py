from datetime import date

from college_basketball_dfs.cbb_odds_backfill import run_odds_season_backfill


def test_odds_backfill_aggregates_success() -> None:
    calls = []

    def stub_runner(*, game_date, **kwargs):
        calls.append((game_date, kwargs))
        return {
            "event_count": 5,
            "odds_game_rows": 5,
            "odds_cache_hit": game_date.day % 2 == 0,
        }

    result = run_odds_season_backfill(
        start_date=date(2026, 2, 10),
        end_date=date(2026, 2, 12),
        runner=stub_runner,
    ).as_dict()

    assert len(calls) == 3
    assert calls[0][1]["historical_mode"] is True
    assert calls[0][1]["historical_snapshot_time"].startswith("2026-02-10T23:59:59")
    assert result["success_dates"] == 3
    assert result["failed_dates"] == 0
    assert result["total_events"] == 15
    assert result["total_odds_game_rows"] == 15
    assert result["odds_cache_hits"] == 2


def test_odds_backfill_collects_errors() -> None:
    def stub_runner(*, game_date, **kwargs):
        if game_date.day == 11:
            raise RuntimeError("bad day")
        return {"event_count": 2, "odds_game_rows": 2, "odds_cache_hit": False}

    result = run_odds_season_backfill(
        start_date=date(2026, 2, 10),
        end_date=date(2026, 2, 12),
        runner=stub_runner,
    ).as_dict()

    assert result["success_dates"] == 2
    assert result["failed_dates"] == 1
    assert result["total_events"] == 4
    assert result["total_odds_game_rows"] == 4
    assert result["errors"][0]["date"] == "2026-02-11"
