from datetime import date

from college_basketball_dfs.cbb_backfill import iter_dates, run_season_backfill, season_start_for_date


def test_iter_dates_inclusive() -> None:
    values = iter_dates(date(2026, 2, 10), date(2026, 2, 12))
    assert values == [date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12)]


def test_season_start_for_date() -> None:
    assert season_start_for_date(date(2026, 11, 15)) == date(2026, 11, 1)
    assert season_start_for_date(date(2026, 2, 15)) == date(2025, 11, 1)


def test_backfill_aggregates_success() -> None:
    calls = []

    def stub_runner(*, game_date, **kwargs):
        calls.append(game_date)
        return {
            "game_count": 2,
            "player_row_count": 20,
            "raw_cache_hit": game_date.day % 2 == 0,
        }

    result = run_season_backfill(
        start_date=date(2026, 2, 10),
        end_date=date(2026, 2, 12),
        runner=stub_runner,
    )
    payload = result.as_dict()
    assert len(calls) == 3
    assert payload["success_dates"] == 3
    assert payload["failed_dates"] == 0
    assert payload["total_games"] == 6
    assert payload["total_player_rows"] == 60
    assert payload["raw_cache_hits"] == 2


def test_backfill_collects_errors_and_continues() -> None:
    def stub_runner(*, game_date, **kwargs):
        if game_date.day == 11:
            raise RuntimeError("boom")
        return {"game_count": 1, "player_row_count": 10, "raw_cache_hit": False}

    result = run_season_backfill(
        start_date=date(2026, 2, 10),
        end_date=date(2026, 2, 12),
        runner=stub_runner,
        stop_on_error=False,
    )
    payload = result.as_dict()
    assert payload["success_dates"] == 2
    assert payload["failed_dates"] == 1
    assert payload["total_games"] == 2
    assert payload["total_player_rows"] == 20
    assert payload["errors"][0]["date"] == "2026-02-11"
