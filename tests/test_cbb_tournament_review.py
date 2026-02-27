import pandas as pd

from college_basketball_dfs.cbb_tournament_review import (
    build_entry_actual_points_comparison,
    build_field_entries_and_players,
    build_player_exposure_comparison,
    build_top10_winner_gap_analysis,
    compare_phantom_entries_to_field,
    build_user_strategy_summary,
    detect_contest_standings_upload,
    extract_actual_ownership_from_standings,
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


def test_parse_lineup_players_supports_extended_slot_labels() -> None:
    lineup = "PG Alpha One SG Bravo Two SF Charlie Three PF Delta Four C Echo Five G Foxtrot Six F Gamma Seven UTIL Hotel Eight"
    parsed = parse_lineup_players(lineup)
    assert len(parsed) == 8
    assert parsed[0]["slot"] == "PG"
    assert parsed[-1]["slot"] == "UTIL"


def test_parse_lineup_players_falls_back_to_comma_names() -> None:
    lineup = "Alpha One, Bravo Two, Charlie Three, Delta Four, Echo Five, Foxtrot Six, Gamma Seven, Hotel Eight"
    parsed = parse_lineup_players(lineup)
    assert len(parsed) == 8
    assert all(p["slot"] == "UTIL" for p in parsed)


def test_detect_contest_standings_upload_flags_dk_entries_template() -> None:
    df = pd.DataFrame(
        [
            {
                "Entry ID": "123",
                "Contest Name": "CBB $1",
                "Contest ID": "999",
                "Entry Fee": "$1",
                "G": "",
                "F": "",
            }
        ]
    )
    meta = detect_contest_standings_upload(df)
    assert meta["kind"] == "dk_entries_template"
    assert bool(meta["is_usable"]) is False


def test_detect_contest_standings_upload_accepts_standings() -> None:
    meta = detect_contest_standings_upload(_sample_standings())
    assert meta["kind"] == "contest_standings"
    assert bool(meta["is_usable"]) is True
    assert int(meta["lineup_nonempty_rows"]) == 2


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


def test_normalize_contest_standings_maps_player_name_and_percent_drafted_columns() -> None:
    raw = pd.DataFrame(
        [
            {
                "rank": 1,
                "entryid": "x1",
                "entryname": "user1",
                "points": 150.0,
                "lineup": "F Alpha One",
                "Player Name": "Alpha One",
                "% Drafted": "12.5%",
            },
        ]
    )
    out = normalize_contest_standings_frame(raw)
    assert "Player" in out.columns
    assert "%Drafted" in out.columns
    assert str(out.iloc[0]["Player"]) == "Alpha One"
    assert round(float(out.iloc[0]["%Drafted"]), 2) == 12.5

    own = extract_actual_ownership_from_standings(out)
    assert len(own) == 1
    assert str(own.iloc[0]["player_name"]) == "Alpha One"
    assert round(float(own.iloc[0]["actual_ownership"]), 2) == 12.5


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
            {"Name": "Alpha One Jr.", "actual_dk_points": 40.0},
            {"Name": "Bravo Two", "actual_dk_points": 28.25},
        ]
    )
    exposure = build_player_exposure_comparison(
        expanded,
        entry_count=20,
        projection_df=proj,
        actual_results_df=actual,
    )
    assert len(exposure) >= 2
    assert "field_ownership_pct" in exposure.columns
    assert "projected_ownership" in exposure.columns
    assert "final_dk_points" in exposure.columns
    assert "high_points_low_own_flag" in exposure.columns
    alpha = exposure.loc[exposure["Name"] == "Alpha One", "final_dk_points"]
    assert not alpha.empty
    assert round(float(alpha.iloc[0]), 2) == 40.0
    alpha_flag = exposure.loc[exposure["Name"] == "Alpha One", "high_points_low_own_flag"]
    assert not alpha_flag.empty
    assert bool(alpha_flag.iloc[0]) is True

    users = build_user_strategy_summary(entries)
    assert len(users) == 2
    assert "entries" in users.columns
    assert "most_points" in users.columns
    assert round(float(users["most_points"].max()), 2) == 200.5


def test_player_exposure_ownership_alias_and_name_fallback() -> None:
    expanded = pd.DataFrame(
        [
            {"EntryId": "1", "resolved_name": "Alpha One", "TeamAbbrev": ""},
            {"EntryId": "2", "resolved_name": "Alpha One", "TeamAbbrev": ""},
        ]
    )
    projection_df = pd.DataFrame(
        [
            {"Name": "Alpha One", "TeamAbbrev": "AAA", "Ownership": 11.5, "blended_projection": 30.0},
        ]
    )
    actual_ownership_df = pd.DataFrame([{"player_name": "Alpha One", "actual_ownership": 18.0}])
    expo = build_player_exposure_comparison(
        expanded_players_df=expanded,
        entry_count=10,
        projection_df=projection_df,
        actual_ownership_df=actual_ownership_df,
        actual_results_df=None,
    )
    assert len(expo) == 1
    row = expo.iloc[0]
    assert round(float(row["projected_ownership"]), 2) == 11.5
    assert round(float(row["actual_ownership_from_file"]), 2) == 18.0
    assert round(float(row["field_ownership_pct"]), 2) == 20.0


def test_player_exposure_keeps_rows_when_team_unmapped() -> None:
    expanded = pd.DataFrame(
        [
            {"EntryId": "1", "resolved_name": "Alpha One", "TeamAbbrev": pd.NA},
            {"EntryId": "2", "resolved_name": "Bravo Two", "TeamAbbrev": pd.NA},
        ]
    )
    expo = build_player_exposure_comparison(
        expanded_players_df=expanded,
        entry_count=10,
        projection_df=pd.DataFrame(),
        actual_ownership_df=pd.DataFrame(),
        actual_results_df=None,
    )
    assert len(expo) == 2
    assert "field_ownership_pct" in expo.columns
    assert round(float(expo["field_ownership_pct"].sum()), 2) == 20.0


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
    assert float(compared.iloc[0]["field_best_points"]) == 190.0
    assert round(float(compared.iloc[0]["winner_gap"]), 2) == 6.0
    assert round(float(compared.iloc[0]["pct_of_winner"]), 2) == round((184.0 / 190.0) * 100.0, 2)

    summary = summarize_phantom_entries(compared)
    assert len(summary) == 1
    assert int(summary.iloc[0]["lineups"]) == 1
    assert float(summary.iloc[0]["best_actual_points"]) == 184.0
    assert float(summary.iloc[0]["winner_points"]) == 190.0
    assert round(float(summary.iloc[0]["winner_gap"]), 2) == 6.0
    assert round(float(summary.iloc[0]["pct_of_winner"]), 2) == round((184.0 / 190.0) * 100.0, 2)


def test_top10_winner_gap_analysis_identifies_missing_top_scorer() -> None:
    entries, expanded = build_field_entries_and_players(_sample_standings(), _sample_slate())
    proj_compare = pd.DataFrame(
        [
            {"Name": "Alpha One", "TeamAbbrev": "AAA", "actual_dk_points": 50.0, "blended_projection": 30.0, "blend_error": 20.0},
            {"Name": "Bravo Two", "TeamAbbrev": "BBB", "actual_dk_points": 45.0, "blended_projection": 31.0, "blend_error": 14.0},
            {"Name": "Charlie Three", "TeamAbbrev": "CCC", "actual_dk_points": 40.0, "blended_projection": 28.0, "blend_error": 12.0},
            {"Name": "Delta Four", "TeamAbbrev": "DDD", "actual_dk_points": 38.0, "blended_projection": 26.0, "blend_error": 12.0},
            {"Name": "Echo Five", "TeamAbbrev": "AAA", "actual_dk_points": 36.0, "blended_projection": 24.0, "blend_error": 12.0},
            {"Name": "Foxtrot Six", "TeamAbbrev": "CCC", "actual_dk_points": 34.0, "blended_projection": 22.0, "blend_error": 12.0},
            {"Name": "Gamma Seven", "TeamAbbrev": "DDD", "actual_dk_points": 32.0, "blended_projection": 21.0, "blend_error": 11.0},
            {"Name": "Hotel Eight", "TeamAbbrev": "AAA", "actual_dk_points": 30.0, "blended_projection": 20.0, "blend_error": 10.0},
        ]
    )
    generated_lineups = [
        {
            "lineup_number": 1,
            "players": [
                {"Name": "Bravo Two"},
                {"Name": "Charlie Three"},
                {"Name": "Delta Four"},
                {"Name": "Echo Five"},
                {"Name": "Foxtrot Six"},
                {"Name": "Gamma Seven"},
                {"Name": "Hotel Eight"},
                {"Name": "Bench Name"},
            ],
        }
    ]

    packet = build_top10_winner_gap_analysis(
        entries_df=entries,
        expanded_players_df=expanded,
        projection_comparison_df=proj_compare,
        generated_lineups=generated_lineups,
        top_n_winners=10,
        top_points_focus=5,
    )
    summary = packet["summary"]
    assert int(summary["top10_entries_count"]) == 2
    assert bool(summary["our_lineups_available"]) is True
    assert int(summary["our_lineups_count"]) == 1
    assert str(summary["top_scorer_name"]) == "Alpha One"
    assert bool(summary["top_scorer_in_our_lineups"]) is False
    assert int(summary["top3_target_count"]) == 3
    assert int(summary["top3_covered_count"]) == 2
    assert bool(summary["top3_all_in_single_lineup"]) is False
    assert "Alpha One" in list(summary["top3_missing_names"])

    missing_focus = packet["missing_focus_players_df"]
    assert not missing_focus.empty
    assert "Alpha One" in missing_focus["Name"].tolist()


def test_top10_winner_gap_analysis_detects_top3_combo_presence() -> None:
    entries, expanded = build_field_entries_and_players(_sample_standings(), _sample_slate())
    proj_compare = pd.DataFrame(
        [
            {"Name": "Alpha One", "actual_dk_points": 50.0, "blended_projection": 30.0, "blend_error": 20.0},
            {"Name": "Bravo Two", "actual_dk_points": 45.0, "blended_projection": 31.0, "blend_error": 14.0},
            {"Name": "Charlie Three", "actual_dk_points": 40.0, "blended_projection": 28.0, "blend_error": 12.0},
            {"Name": "Delta Four", "actual_dk_points": 38.0, "blended_projection": 26.0, "blend_error": 12.0},
        ]
    )
    generated_lineups = [
        {
            "lineup_number": 11,
            "players": [
                {"Name": "Alpha One"},
                {"Name": "Bravo Two"},
                {"Name": "Charlie Three"},
                {"Name": "Delta Four"},
                {"Name": "Echo Five"},
                {"Name": "Foxtrot Six"},
                {"Name": "Gamma Seven"},
                {"Name": "Hotel Eight"},
            ],
        }
    ]

    packet = build_top10_winner_gap_analysis(
        entries_df=entries,
        expanded_players_df=expanded,
        projection_comparison_df=proj_compare,
        generated_lineups=generated_lineups,
        top_n_winners=10,
        top_points_focus=5,
    )
    summary = packet["summary"]
    assert bool(summary["top_scorer_in_our_lineups"]) is True
    assert int(summary["top3_covered_count"]) == 3
    assert bool(summary["top3_all_in_single_lineup"]) is True

    hit_dist = packet["lineup_top3_hit_distribution_df"]
    assert not hit_dist.empty
    assert 3 in hit_dist["top3_hits"].tolist()


def test_top10_winner_gap_analysis_uses_lineup_uid_for_distinct_rows() -> None:
    entries, expanded = build_field_entries_and_players(_sample_standings(), _sample_slate())
    proj_compare = pd.DataFrame(
        [
            {"Name": "Alpha One", "actual_dk_points": 50.0, "blended_projection": 30.0, "blend_error": 20.0},
            {"Name": "Bravo Two", "actual_dk_points": 45.0, "blended_projection": 31.0, "blend_error": 14.0},
            {"Name": "Charlie Three", "actual_dk_points": 40.0, "blended_projection": 28.0, "blend_error": 12.0},
        ]
    )
    generated_lineups = [
        {
            "lineup_uid": "v1:1",
            "lineup_number": 1,
            "players": [{"Name": "Alpha One"}, {"Name": "Bravo Two"}, {"Name": "Charlie Three"}],
        },
        {
            "lineup_uid": "v2:1",
            "lineup_number": 1,
            "players": [{"Name": "Alpha One"}, {"Name": "Bravo Two"}, {"Name": "Charlie Three"}],
        },
    ]

    packet = build_top10_winner_gap_analysis(
        entries_df=entries,
        expanded_players_df=expanded,
        projection_comparison_df=proj_compare,
        generated_lineups=generated_lineups,
        top_n_winners=10,
        top_points_focus=5,
    )
    summary = packet["summary"]
    assert int(summary["our_lineups_count"]) == 2
    hit_dist = packet["lineup_top3_hit_distribution_df"]
    assert not hit_dist.empty
    row = hit_dist.loc[hit_dist["top3_hits"] == 3]
    assert not row.empty
    assert int(row.iloc[0]["lineups"]) == 2
