import pandas as pd

from college_basketball_dfs.cbb_tournament_review import (
    build_entry_actual_points_comparison,
    build_field_entries_and_players,
    build_player_exposure_comparison,
    compare_phantom_entries_to_field,
    build_user_strategy_summary,
    normalize_contest_standings_frame,
    parse_lineup_players,
    score_generated_lineups_against_actuals,
    summarize_phantom_entries,
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


def test_normalize_contest_standings_imputes_rank_from_points() -> None:
    raw = pd.DataFrame(
        [
            {
                "Rank": 99,
                "EntryId": "a",
                "EntryName": "A",
                "Points": 180.0,
                "Lineup": "F Alpha One",
                "Player": "Alpha One",
                "%Drafted": "10%",
            },
            {
                "Rank": 1,
                "EntryId": "b",
                "EntryName": "B",
                "Points": 210.0,
                "Lineup": "F Bravo Two",
                "Player": "Bravo Two",
                "%Drafted": "20%",
            },
        ]
    )
    out = normalize_contest_standings_frame(raw)
    by_entry = out.set_index("EntryId")
    assert float(by_entry.loc["b", "Rank"]) == 1.0
    assert float(by_entry.loc["a", "Rank"]) == 2.0
    assert float(by_entry.loc["a", "rank_from_file"]) == 99.0
    assert float(by_entry.loc["a", "rank_from_points"]) == 2.0


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
    actual = pd.DataFrame(
        [
            {"Name": "Alpha One Jr.", "actual_dk_points": 31.5},
            {"Name": "Bravo Two", "actual_dk_points": 28.25},
        ]
    )
    exposure = build_player_exposure_comparison(
        expanded,
        entry_count=2,
        projection_df=proj,
        actual_results_df=actual,
    )
    assert len(exposure) >= 2
    assert "field_ownership_pct" in exposure.columns
    assert "projected_ownership" in exposure.columns
    assert "final_dk_points" in exposure.columns
    alpha = exposure.loc[exposure["Name"] == "Alpha One", "final_dk_points"]
    assert not alpha.empty
    assert round(float(alpha.iloc[0]), 2) == 31.5

    users = build_user_strategy_summary(entries)
    assert len(users) == 2
    assert "entries" in users.columns
    assert "most_points" in users.columns
    assert round(float(users["most_points"].max()), 2) == 200.5


def test_build_entry_actual_points_comparison_adds_computed_points() -> None:
    entries, expanded = build_field_entries_and_players(_sample_standings(), _sample_slate())
    actual_df = pd.DataFrame(
        [
            {"Name": "Alpha One Jr.", "actual_dk_points": 30.0},
            {"Name": "Bravo Two", "actual_dk_points": 28.0},
            {"Name": "Charlie Three", "actual_dk_points": 26.0},
            {"Name": "Delta Four", "actual_dk_points": 24.0},
            {"Name": "Echo Five", "actual_dk_points": 22.0},
            {"Name": "Foxtrot Six", "actual_dk_points": 20.0},
            {"Name": "Gamma Seven", "actual_dk_points": 18.0},
            {"Name": "Hotel Eight", "actual_dk_points": 16.0},
        ]
    )
    compared = build_entry_actual_points_comparison(entries, expanded, actual_df)
    assert len(compared) == 2
    assert "computed_actual_points" in compared.columns
    assert "computed_coverage_pct" in compared.columns
    assert round(float(compared.iloc[0]["computed_actual_points"]), 2) == 184.0
    assert round(float(compared.iloc[0]["computed_coverage_pct"]), 2) == 100.0
    assert round(float(compared.iloc[0]["Points"]), 2) == 184.0
    assert "points_from_file" in compared.columns
    assert round(float(compared.iloc[0]["points_from_file"]), 2) == 200.5
    assert "rank_from_computed_points" in compared.columns
    assert round(float(compared.iloc[0]["rank_from_computed_points"]), 2) == 1.0


def test_score_generated_lineups_against_actuals_loose_name_match() -> None:
    generated_lineups = [
        {
            "lineup_number": 1,
            "lineup_strategy": "standard",
            "salary": 5000,
            "projected_points": 20.0,
            "players": [{"ID": "1", "Name": "Alpha One"}],
        }
    ]
    actual_df = pd.DataFrame([{"ID": "99", "Name": "Alpha One Jr.", "actual_dk_points": 30.0}])
    phantom = score_generated_lineups_against_actuals(
        generated_lineups=generated_lineups,
        actual_results_df=actual_df,
        version_key="standard_v1",
        version_label="Standard v1",
    )
    assert len(phantom) == 1
    assert float(phantom.iloc[0]["actual_points"]) == 30.0
    assert int(phantom.iloc[0]["matched_players"]) == 1
    assert float(phantom.iloc[0]["coverage_pct"]) == 100.0


def test_build_user_strategy_summary_uses_computed_actual_points_when_available() -> None:
    df = pd.DataFrame(
        [
            {
                "EntryId": "1",
                "EntryName": "userx (1/2)",
                "Points": 250.0,
                "computed_actual_points": 190.0,
                "Rank": 2,
                "salary_left": 100.0,
                "max_team_stack": 3,
                "max_game_stack": 3,
            },
            {
                "EntryId": "2",
                "EntryName": "userx (2/2)",
                "Points": 100.0,
                "computed_actual_points": 220.0,
                "Rank": 1,
                "salary_left": 0.0,
                "max_team_stack": 2,
                "max_game_stack": 2,
            },
        ]
    )
    users = build_user_strategy_summary(df)
    assert len(users) == 1
    assert round(float(users.iloc[0]["most_points"]), 2) == 220.0
    assert round(float(users.iloc[0]["avg_points"]), 2) == 205.0


def test_score_generated_lineups_against_actuals_and_field_compare() -> None:
    generated_lineups = [
        {
            "lineup_number": 1,
            "lineup_strategy": "spike",
            "pair_id": 1,
            "pair_role": "A",
            "salary": 49600,
            "projected_points": 210.0,
            "projected_ownership_sum": 160.0,
            "players": [
                {"ID": "1", "Name": "Alpha One"},
                {"ID": "2", "Name": "Bravo Two"},
                {"ID": "3", "Name": "Charlie Three"},
                {"ID": "4", "Name": "Delta Four"},
                {"ID": "5", "Name": "Echo Five"},
                {"ID": "6", "Name": "Foxtrot Six"},
                {"ID": "7", "Name": "Gamma Seven"},
                {"ID": "8", "Name": "Hotel Eight"},
            ],
        }
    ]
    actual_df = pd.DataFrame(
        [
            {"ID": "1", "Name": "Alpha One", "actual_dk_points": 30.0},
            {"ID": "2", "Name": "Bravo Two", "actual_dk_points": 28.0},
            {"ID": "3", "Name": "Charlie Three", "actual_dk_points": 26.0},
            {"ID": "4", "Name": "Delta Four", "actual_dk_points": 24.0},
            {"ID": "5", "Name": "Echo Five", "actual_dk_points": 22.0},
            {"ID": "6", "Name": "Foxtrot Six", "actual_dk_points": 20.0},
            {"ID": "7", "Name": "Gamma Seven", "actual_dk_points": 18.0},
            {"ID": "8", "Name": "Hotel Eight", "actual_dk_points": 16.0},
        ]
    )

    phantom = score_generated_lineups_against_actuals(
        generated_lineups=generated_lineups,
        actual_results_df=actual_df,
        version_key="spike_v2_tail",
        version_label="Spike v2",
    )
    assert len(phantom) == 1
    row = phantom.iloc[0]
    assert float(row["actual_points"]) == 184.0
    assert int(row["matched_players"]) == 8
    assert float(row["coverage_pct"]) == 100.0

    field_entries = pd.DataFrame(
        [
            {"EntryId": "a", "Points": 170.0},
            {"EntryId": "b", "Points": 190.0},
            {"EntryId": "c", "Points": 150.0},
        ]
    )
    compared = compare_phantom_entries_to_field(phantom, field_entries)
    assert len(compared) == 1
    assert int(compared.iloc[0]["field_size"]) == 3
    assert int(compared.iloc[0]["would_rank"]) == 2
    assert round(float(compared.iloc[0]["would_beat_pct"]), 2) == round((2 / 3) * 100.0, 2)

    summary = summarize_phantom_entries(compared)
    assert len(summary) == 1
    assert int(summary.iloc[0]["lineups"]) == 1
    assert float(summary.iloc[0]["best_actual_points"]) == 184.0
