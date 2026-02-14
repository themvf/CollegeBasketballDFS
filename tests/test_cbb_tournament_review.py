import pandas as pd

from college_basketball_dfs.cbb_tournament_review import (
    build_field_entries_and_players,
    build_player_exposure_comparison,
    build_user_strategy_summary,
    parse_lineup_players,
)


def _sample_slate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Name": "Alpha One", "Salary": 7000, "TeamAbbrev": "AAA", "Game Info": "AAA@BBB 02/14/2026 04:00PM ET"},
            {"Name": "Bravo Two", "Salary": 6800, "TeamAbbrev": "BBB", "Game Info": "AAA@BBB 02/14/2026 04:00PM ET"},
            {"Name": "Charlie Three", "Salary": 6600, "TeamAbbrev": "CCC", "Game Info": "CCC@DDD 02/14/2026 05:00PM ET"},
            {"Name": "Delta Four", "Salary": 6400, "TeamAbbrev": "DDD", "Game Info": "CCC@DDD 02/14/2026 05:00PM ET"},
            {"Name": "Echo Five", "Salary": 6200, "TeamAbbrev": "AAA", "Game Info": "AAA@BBB 02/14/2026 04:00PM ET"},
            {"Name": "Foxtrot Six", "Salary": 6000, "TeamAbbrev": "CCC", "Game Info": "CCC@DDD 02/14/2026 05:00PM ET"},
            {"Name": "Gamma Seven", "Salary": 5800, "TeamAbbrev": "DDD", "Game Info": "CCC@DDD 02/14/2026 05:00PM ET"},
            {"Name": "Hotel Eight", "Salary": 5600, "TeamAbbrev": "AAA", "Game Info": "AAA@BBB 02/14/2026 04:00PM ET"},
        ]
    )


def _sample_standings() -> pd.DataFrame:
    lineup = "F Alpha One F Bravo Two F Charlie Three G Delta Four G Echo Five G Foxtrot Six UTIL Gamma Seven UTIL Hotel Eight"
    return pd.DataFrame(
        [
            {
                "Rank": 1,
                "EntryId": "1001",
                "EntryName": "user1 (1/2)",
                "Points": 200.5,
                "Lineup": lineup,
                "Player": "Alpha One",
                "%Drafted": "25.0%",
            },
            {
                "Rank": 2,
                "EntryId": "1002",
                "EntryName": "user2",
                "Points": 190.0,
                "Lineup": lineup,
                "Player": "Bravo Two",
                "%Drafted": "20.0%",
            },
        ]
    )


def test_parse_lineup_players_parses_eight_slots() -> None:
    lineup = "F Alpha One F Bravo Two F Charlie Three G Delta Four G Echo Five G Foxtrot Six UTIL Gamma Seven UTIL Hotel Eight"
    parsed = parse_lineup_players(lineup)
    assert len(parsed) == 8
    assert parsed[0]["slot"] == "F"
    assert parsed[-1]["slot"] == "UTIL"


def test_build_field_entries_and_players_builds_salary_left() -> None:
    entries, expanded = build_field_entries_and_players(_sample_standings(), _sample_slate())
    assert len(entries) == 2
    assert len(expanded) == 16
    assert "salary_left" in entries.columns
    assert entries["salary_left"].iloc[0] == -400.0


def test_player_exposure_and_user_summary() -> None:
    entries, expanded = build_field_entries_and_players(_sample_standings(), _sample_slate())
    proj = pd.DataFrame(
        [
            {"Name": "Alpha One", "TeamAbbrev": "AAA", "projected_ownership": 18.0, "blended_projection": 30.0},
            {"Name": "Bravo Two", "TeamAbbrev": "BBB", "projected_ownership": 15.0, "blended_projection": 28.0},
        ]
    )
    exposure = build_player_exposure_comparison(expanded, entry_count=2, projection_df=proj)
    assert len(exposure) >= 2
    assert "field_ownership_pct" in exposure.columns
    assert "projected_ownership" in exposure.columns

    users = build_user_strategy_summary(entries)
    assert len(users) == 2
    assert "entries" in users.columns
