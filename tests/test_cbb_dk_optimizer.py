from collections import Counter

import pandas as pd

from college_basketball_dfs.cbb_dk_optimizer import (
    apply_focus_game_chalk_guardrail,
    apply_model_profile_adjustments,
    apply_projection_uncertainty_adjustment,
    apply_projection_calibration,
    apply_false_chalk_discount,
    apply_minutes_shock_override,
    apply_chalk_ceiling_guardrail,
    apply_ownership_surprise_guardrails,
    apply_contest_objective,
    build_dk_upload_csv,
    build_player_pool,
    enrich_lineups_minutes_from_pool,
    generate_lineups,
    lineups_slots_frame,
    lineups_summary_frame,
    normalize_projected_ownership_total,
    normalize_injuries_frame,
    ownership_salary_bucket_key,
    projection_salary_bucket_key,
    remove_injured_players,
)


def _sample_slate() -> pd.DataFrame:
    rows = []
    for i in range(1, 7):
        rows.append(
            {
                "Position": "G",
                "Name + ID": f"Guard {i} ({1000 + i})",
                "Name": f"Guard {i}",
                "ID": str(1000 + i),
                "Roster Position": "G/UTIL",
                "Salary": 6200 - (i * 120),
                "Game Info": "AAA@BBB 02/14/2026 04:00PM ET" if i <= 2 else "CCC@DDD 02/14/2026 05:00PM ET",
                "TeamAbbrev": "AAA" if i % 2 == 0 else "CCC",
                "AvgPointsPerGame": 25 + i,
            }
        )
    for i in range(1, 7):
        rows.append(
            {
                "Position": "F",
                "Name + ID": f"Forward {i} ({2000 + i})",
                "Name": f"Forward {i}",
                "ID": str(2000 + i),
                "Roster Position": "F/UTIL",
                "Salary": 6100 - (i * 140),
                "Game Info": "EEE@FFF 02/14/2026 06:00PM ET" if i <= 2 else "CCC@DDD 02/14/2026 05:00PM ET",
                "TeamAbbrev": "EEE" if i % 2 == 0 else "DDD",
                "AvgPointsPerGame": 24 + i,
            }
        )
    return pd.DataFrame(rows)


def _sample_props() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_name": "Guard 1", "bookmaker": "fanduel", "market": "player_points", "line": 18.5},
            {"player_name": "Guard 1", "bookmaker": "fanduel", "market": "player_assists", "line": 5.5},
            {"player_name": "Forward 1", "bookmaker": "fanduel", "market": "player_points", "line": 16.5},
            {"player_name": "Forward 1", "bookmaker": "fanduel", "market": "player_rebounds", "line": 9.5},
            {"player_name": "Forward 1", "bookmaker": "fanduel", "market": "player_threes_made", "line": 1.5},
        ]
    )


def _sample_season_stats() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_date": "2026-02-01",
                "player_name": "Guard 1",
                "team_name": "CCC",
                "minutes_played": 34,
                "points": 20,
                "rebounds": 4,
                "assists": 6,
                "steals": 2,
                "blocks": 0,
                "turnovers": 3,
                "tpm": 2,
                "fga": 14,
                "fta": 5,
                "dk_fpts": 36.5,
            },
            {
                "game_date": "2026-02-03",
                "player_name": "Forward 1",
                "team_name": "DDD",
                "minutes_played": 32,
                "points": 16,
                "rebounds": 9,
                "assists": 2,
                "steals": 1,
                "blocks": 1,
                "turnovers": 2,
                "tpm": 1,
                "fga": 12,
                "fta": 4,
                "dk_fpts": 31.25,
            },
        ]
    )


def test_remove_injured_players_filters_out_and_doubtful() -> None:
    slate = _sample_slate()
    injuries = pd.DataFrame(
        [
            {"player_name": "Guard 1", "team": "CCC", "status": "Out", "active": True},
            {"player_name": "Forward 2", "team": "EEE", "status": "Doubtful", "active": True},
            {"player_name": "Guard 2", "team": "AAA", "status": "Questionable", "active": True},
        ]
    )

    filtered, removed = remove_injured_players(slate, injuries)
    assert "Guard 1" in removed["Name"].tolist()
    assert "Forward 2" in removed["Name"].tolist()
    assert "Guard 2" not in removed["Name"].tolist()
    assert len(filtered) == len(slate) - 2


def test_remove_injured_players_defaults_feed_rows_active_when_active_missing() -> None:
    slate = _sample_slate()
    injuries = pd.DataFrame(
        [
            {"player_name": "Guard 1", "team": "CCC", "status": "Out"},
        ]
    )

    filtered, removed = remove_injured_players(slate, injuries)
    assert "Guard 1" in removed["Name"].tolist()
    assert len(filtered) == len(slate) - 1


def test_remove_injured_players_uses_unique_name_fallback_for_team_mismatch() -> None:
    slate = _sample_slate()
    injuries = pd.DataFrame(
        [
            # Team label does not match slate abbreviation, but player name is unique in slate.
            {"player_name": "Guard 1", "team": "SOMETEAM", "status": "Out", "active": True},
        ]
    )

    filtered, removed = remove_injured_players(slate, injuries)
    assert "Guard 1" in removed["Name"].tolist()
    assert len(filtered) == len(slate) - 1


def test_normalize_injuries_frame_handles_title_case_slate_feed_csv() -> None:
    raw = pd.DataFrame(
        [
            {
                "Player": "Joe Hurlburt",
                "Team": "Davidson",
                "Pos": "C",
                "Injury": "Undisclosed",
                "Status": "Out",
                "Est. Return": "Subscribers Only",
            },
            {
                "Player": "Lucas Langarita",
                "Team": "Utah",
                "Pos": "G",
                "Injury": "Lower Leg",
                "Status": "Game Time Decision",
                "Est. Return": "Subscribers Only",
            },
            {
                "Player": "Xavier Brown",
                "Team": "South Florida",
                "Pos": "G",
                "Injury": "Undisclosed",
                "Status": "Out For Season",
                "Est. Return": "Subscribers Only",
            },
        ]
    )

    normalized = normalize_injuries_frame(raw)
    assert len(normalized) == 3
    assert set(normalized["status"].tolist()) == {"out", "questionable"}
    assert normalized.loc[normalized["player_name"] == "Joe Hurlburt", "team"].iloc[0] == "DAVIDSON"
    assert "Undisclosed" in normalized.loc[normalized["player_name"] == "Joe Hurlburt", "notes"].iloc[0]


def test_build_player_pool_merges_props() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    g1 = pool.loc[pool["Name"] == "Guard 1"].iloc[0]
    f1 = pool.loc[pool["Name"] == "Forward 1"].iloc[0]
    assert g1["vegas_points_line"] == 18.5
    assert g1["vegas_assists_line"] == 5.5
    assert f1["vegas_threes_line"] == 1.5
    assert g1["projected_dk_points"] > 0
    assert f1["projected_ownership"] > 0


def test_build_player_pool_ownership_v2_has_slate_controls() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    assert not pool.empty
    assert "projected_ownership_v1" in pool.columns
    assert "ownership_model" in pool.columns
    assert "ownership_temperature" in pool.columns
    assert "ownership_target_total" in pool.columns
    assert set(pool["ownership_model"].astype(str).unique().tolist()) == {"v3_tiered_softmax"}
    own = pd.to_numeric(pool["projected_ownership"], errors="coerce")
    assert own.notna().all()
    assert (own >= 0).all()
    assert (own <= 100).all()
    # Ownership target is 800% across players before clipping.
    assert abs(float(own.sum()) - 800.0) <= 5.0


def test_build_player_pool_uses_historical_ownership_prior_when_current_field_is_missing() -> None:
    slate = _sample_slate()
    slate.index = pd.Index(range(10, 10 + len(slate)))
    history = pd.DataFrame(
        [
            {"Name": "Guard 1", "TeamAbbrev": "CCC", "actual_ownership": 26.0, "review_date": "2026-02-07"},
            {"Name": "Guard 1", "TeamAbbrev": "CCC", "actual_ownership": 28.0, "review_date": "2026-02-10"},
            {"Name": "Guard 1", "TeamAbbrev": "CCC", "actual_ownership": 30.0, "review_date": "2026-02-12"},
            {"Name": "Forward 1", "TeamAbbrev": "DDD", "actual_ownership": 5.0, "review_date": "2026-02-08"},
            {"Name": "Forward 1", "TeamAbbrev": "DDD", "actual_ownership": 6.0, "review_date": "2026-02-11"},
            {"Name": "Forward 1", "TeamAbbrev": "DDD", "actual_ownership": 4.0, "review_date": "2026-02-13"},
        ]
    )

    base_pool = build_player_pool(slate, _sample_props(), bookmaker_filter="fanduel")
    hist_pool = build_player_pool(
        slate,
        _sample_props(),
        ownership_history_df=history,
        bookmaker_filter="fanduel",
    )

    hist_g1 = hist_pool.loc[hist_pool["Name"] == "Guard 1"].iloc[0]
    hist_f1 = hist_pool.loc[hist_pool["Name"] == "Forward 1"].iloc[0]

    assert round(float(hist_g1["historical_ownership_avg"]), 2) == 28.0
    assert round(float(hist_g1["historical_ownership_last5"]), 2) == 28.0
    assert int(hist_g1["historical_ownership_samples"]) == 3
    assert round(float(hist_g1["field_ownership_pct"]), 2) == 28.0
    assert hist_g1["ownership_prior_source"] == "historical_last5"
    assert bool(hist_g1["historical_ownership_used_in_prior"]) is True
    assert round(float(hist_f1["field_ownership_pct"]), 2) == 5.0
    projected_shift = (
        pd.to_numeric(hist_pool["projected_ownership"], errors="coerce")
        - pd.to_numeric(base_pool["projected_ownership"], errors="coerce")
    ).abs().sum()
    assert projected_shift > 0.0


def test_generate_lineups_respects_locks_and_excludes() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    locked = ["1001"]
    excluded = ["2006"]
    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=5,
        contest_type="Small GPP",
        locked_ids=locked,
        excluded_ids=excluded,
        random_seed=13,
    )

    assert warnings == []
    assert len(lineups) == 5
    for lineup in lineups:
        ids = set(lineup["player_ids"])
        assert "1001" in ids
        assert "2006" not in ids
        assert lineup["salary"] <= 50000
        assert len(ids) == 8

    upload_csv = build_dk_upload_csv(lineups)
    assert upload_csv.startswith("G,G,G,F,F,F,UTIL,UTIL")


def test_lineup_minutes_summary_columns_present() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy()
    pool["our_minutes_avg"] = 28.0
    pool["our_minutes_last7"] = 29.0
    pool["our_minutes_last3"] = 30.0
    pool["our_minutes_recent"] = 31.0

    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=3,
        contest_type="Small GPP",
        random_seed=13,
    )

    assert warnings == []
    assert len(lineups) == 3
    assert all(abs(float(lineup["expected_minutes_sum"]) - 248.0) < 1e-6 for lineup in lineups)
    assert all(abs(float(lineup["avg_minutes_last3"]) - 30.0) < 1e-6 for lineup in lineups)
    assert all(float(lineup["ceiling_projection"]) > float(lineup["projected_points"]) for lineup in lineups)
    for lineup in lineups:
        lineup["lineup_model_label"] = "Standard v1"

    summary_df = lineups_summary_frame(lineups)
    assert "Expected Minutes Sum" in summary_df.columns
    assert "Avg Minutes (Past 3 Games)" in summary_df.columns
    assert "Ceiling Projection" in summary_df.columns
    assert "Lineup Model" in summary_df.columns
    assert abs(float(pd.to_numeric(summary_df["Expected Minutes Sum"], errors="coerce").iloc[0]) - 248.0) < 1e-6
    assert abs(float(pd.to_numeric(summary_df["Avg Minutes (Past 3 Games)"], errors="coerce").iloc[0]) - 30.0) < 1e-6
    assert float(pd.to_numeric(summary_df["Ceiling Projection"], errors="coerce").iloc[0]) > float(
        pd.to_numeric(summary_df["Projected Points"], errors="coerce").iloc[0]
    )
    assert str(summary_df["Lineup Model"].iloc[0]) == "Standard v1"

    slots_df = lineups_slots_frame(lineups)
    assert "G1 Team" in slots_df.columns
    assert "G1 Salary" in slots_df.columns
    assert "G1 Projected Points" in slots_df.columns
    assert "G1 Ceiling Projection" in slots_df.columns
    assert "G1 Projected Ownership" in slots_df.columns
    assert pd.to_numeric(slots_df["G1 Salary"], errors="coerce").iloc[0] > 0


def test_enrich_lineups_minutes_from_pool_backfills_legacy_lineups() -> None:
    legacy_lineups = [
        {
            "lineup_number": 1,
            "players": [
                {"ID": "1001", "Name": "Guard 1"},
                {"ID": "1002", "Name": "Guard 2"},
            ],
            "salary": 12000,
            "projected_points": 40.0,
            "projected_ownership_sum": 20.0,
        }
    ]
    pool_df = pd.DataFrame(
        [
            {"ID": "1001", "our_minutes_recent": 31.0, "our_minutes_last7": 30.0, "our_minutes_last3": 32.0, "our_minutes_avg": 29.0},
            {"ID": "1002", "our_minutes_recent": 29.0, "our_minutes_last7": 28.0, "our_minutes_last3": 27.0, "our_minutes_avg": 26.0},
        ]
    )

    enriched = enrich_lineups_minutes_from_pool(legacy_lineups, pool_df)
    assert len(enriched) == 1
    lineup = enriched[0]
    assert abs(float(lineup["expected_minutes_sum"]) - 60.0) < 1e-6
    assert abs(float(lineup["avg_minutes_last3"]) - 29.5) < 1e-6
    assert abs(float(enriched[0]["players"][0]["our_minutes_recent"]) - 31.0) < 1e-6
    assert abs(float(enriched[0]["players"][1]["our_minutes_last3"]) - 27.0) < 1e-6


def test_generate_lineups_respects_max_salary_left() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=5,
        contest_type="Small GPP",
        max_salary_left=5000,
        random_seed=13,
    )

    assert warnings == []
    assert len(lineups) == 5
    assert all(int(lineup["salary"]) >= 45000 for lineup in lineups)


def test_generate_lineups_respects_global_max_exposure() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    requested_lineups = 10
    max_exposure_pct = 40.0
    lineups, _ = generate_lineups(
        pool_df=pool,
        num_lineups=requested_lineups,
        contest_type="Small GPP",
        global_max_exposure_pct=max_exposure_pct,
        random_seed=13,
    )

    exposure_counts: Counter[str] = Counter()
    for lineup in lineups:
        exposure_counts.update(lineup["player_ids"])

    max_allowed_count = int((max_exposure_pct / 100.0) * requested_lineups)
    assert exposure_counts
    assert max(exposure_counts.values()) <= max_allowed_count


def test_generate_lineups_spike_mode_decorrelates_pairs() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=4,
        contest_type="Small GPP",
        lineup_strategy="spike",
        spike_max_pair_overlap=4,
        random_seed=13,
    )

    assert warnings == []
    assert len(lineups) == 4
    assert lineups[0]["pair_id"] == 1 and lineups[0]["pair_role"] == "A"
    assert lineups[1]["pair_id"] == 1 and lineups[1]["pair_role"] == "B"
    assert lineups[2]["pair_id"] == 2 and lineups[2]["pair_role"] == "A"
    assert lineups[3]["pair_id"] == 2 and lineups[3]["pair_role"] == "B"
    assert all(str(l.get("lineup_strategy")) == "spike" for l in lineups)

    for idx in (0, 2):
        a_ids = set(lineups[idx]["player_ids"])
        b_ids = set(lineups[idx + 1]["player_ids"])
        assert len(a_ids & b_ids) <= 4


def test_generate_lineups_objective_adjustments_increase_target_exposure() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    target_id = "2006"

    base_lineups, base_warnings = generate_lineups(
        pool_df=pool,
        num_lineups=20,
        contest_type="Small GPP",
        random_seed=13,
    )
    boosted_lineups, boosted_warnings = generate_lineups(
        pool_df=pool,
        num_lineups=20,
        contest_type="Small GPP",
        objective_score_adjustments={target_id: 30.0},
        random_seed=13,
    )

    assert base_warnings == []
    assert boosted_warnings == []
    base_count = sum(1 for lineup in base_lineups if target_id in set(lineup["player_ids"]))
    boosted_count = sum(1 for lineup in boosted_lineups if target_id in set(lineup["player_ids"]))
    assert boosted_count >= base_count
    assert boosted_count > 0


def test_build_player_pool_blends_our_and_vegas_stats() -> None:
    pool = build_player_pool(
        _sample_slate(),
        _sample_props(),
        season_stats_df=_sample_season_stats(),
        bookmaker_filter="fanduel",
    )
    g1 = pool.loc[pool["Name"] == "Guard 1"].iloc[0]
    # Our points=20, Vegas points=18.5, vegas_weight=0.35 (2 markets) => weighted blend=19.475
    assert round(float(g1["blend_points_proj"]), 3) == 19.475
    # Rebounds missing in vegas for Guard 1, so blended should keep our rebounds=4
    assert round(float(g1["blend_rebounds_proj"]), 2) == 4.0
    assert int(g1["vegas_markets_found"]) == 2
    assert bool(g1["vegas_projection_usable"]) is True
    assert round(float(g1["vegas_blend_weight"]), 2) == 0.35
    assert float(g1["our_minutes_avg"]) > 0
    assert float(g1["our_usage_proxy"]) > 0
    assert round(float(g1["blended_projection"]), 3) == round(float(g1["projected_dk_points"]), 3)
    assert float(g1["projection_per_dollar"]) > 0

    # Forward 1 has vegas line support and should produce vegas delta + potential signal columns
    f1 = pool.loc[pool["Name"] == "Forward 1"].iloc[0]
    assert "vegas_over_our_flag" in f1.index
    assert "low_own_ceiling_flag" in f1.index
    assert int(f1["vegas_markets_found"]) == 3
    assert bool(f1["vegas_projection_usable"]) is True
    assert round(float(f1["vegas_blend_weight"]), 2) == 0.55

    # Guard 2 has no vegas markets, so vegas projection should be NA and blend falls back to our projection.
    g2 = pool.loc[pool["Name"] == "Guard 2"].iloc[0]
    assert int(g2["vegas_markets_found"]) == 0
    assert bool(g2["vegas_projection_usable"]) is False
    assert round(float(g2["vegas_blend_weight"]), 2) == 0.0
    assert pd.isna(g2["vegas_dk_projection"])
    assert round(float(g2["blend_points_proj"]), 3) == round(float(g2["our_points_proj"]), 3)


def test_build_player_pool_recent_form_window_and_points_blend() -> None:
    season_rows = []
    recent_points = [10, 12, 14, 16, 18, 20, 30, 40]
    recent_minutes = [20, 22, 24, 26, 28, 30, 32, 34]
    for idx, (pts, mins) in enumerate(zip(recent_points, recent_minutes), start=1):
        season_rows.append(
            {
                "game_date": f"2026-02-{idx:02d}",
                "player_name": "Guard 1",
                "team_name": "CCC",
                "minutes_played": mins,
                "points": pts,
                "rebounds": 4,
                "assists": 5,
                "steals": 1,
                "blocks": 0,
                "turnovers": 2,
                "tpm": 2,
                "fga": 12,
                "fta": 3,
                "dk_fpts": float(pts + 10),
            }
        )
    season_rows.append(
        {
            "game_date": "2026-02-01",
            "player_name": "Forward 1",
            "team_name": "DDD",
            "minutes_played": 32,
            "points": 16,
            "rebounds": 9,
            "assists": 2,
            "steals": 1,
            "blocks": 1,
            "turnovers": 2,
            "tpm": 1,
            "fga": 12,
            "fta": 4,
            "dk_fpts": 31.25,
        }
    )
    season_stats = pd.DataFrame(season_rows)

    pool = build_player_pool(
        _sample_slate(),
        _sample_props(),
        season_stats_df=season_stats,
        bookmaker_filter="fanduel",
        recent_form_games=3,
        recent_points_weight=0.5,
    )
    g1 = pool.loc[pool["Name"] == "Guard 1"].iloc[0]
    assert abs(float(g1["our_minutes_last7"]) - 28.0) < 1e-6
    assert abs(float(g1["our_minutes_recent"]) - 32.0) < 1e-6
    assert abs(float(g1["our_minutes_last3"]) - 32.0) < 1e-6
    assert abs(float(g1["our_points_recent"]) - 30.0) < 1e-6
    # our_points_avg=20.0 blended 50/50 with recent=30.0
    assert abs(float(g1["our_points_proj"]) - 25.0) < 1e-6
    assert int(g1["recent_form_games_window"]) == 3
    assert abs(float(g1["recent_points_weight"]) - 0.5) < 1e-6


def test_build_player_pool_attaches_tail_metrics_from_game_odds() -> None:
    odds_scored_df = pd.DataFrame(
        [
            {
                "event_id": "evt_aaa_bbb",
                "home_team": "BBB",
                "away_team": "AAA",
                "total_points": 148.5,
                "spread_home": -3.5,
                "tail_residual_mu": 1.0,
                "tail_sigma": 10.5,
                "p_plus_8": 0.22,
                "p_plus_12": 0.11,
                "volatility_score": 10.5,
            },
            {
                "event_id": "evt_ccc_ddd",
                "home_team": "DDD",
                "away_team": "CCC",
                "total_points": 152.0,
                "spread_home": -1.5,
                "tail_residual_mu": 2.0,
                "tail_sigma": 12.2,
                "p_plus_8": 0.31,
                "p_plus_12": 0.18,
                "volatility_score": 12.2,
            },
        ]
    )
    pool = build_player_pool(
        _sample_slate(),
        _sample_props(),
        bookmaker_filter="fanduel",
        odds_games_df=odds_scored_df,
    )
    assert "game_p_plus_8" in pool.columns
    assert "game_p_plus_12" in pool.columns
    assert "game_tail_score" in pool.columns
    game_rows = pool.loc[pool["game_key"] == "AAA@BBB"]
    assert not game_rows.empty
    assert game_rows["game_p_plus_12"].notna().any()
    assert game_rows["game_tail_score"].notna().any()


def test_apply_contest_objective_can_toggle_tail_signals() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    pool = pool.copy()
    pool["game_tail_score"] = 80.0
    pool["game_tail_to_ownership_pct"] = 1.0

    legacy = apply_contest_objective(pool, contest_type="Large GPP", include_tail_signals=False)
    tail = apply_contest_objective(pool, contest_type="Large GPP", include_tail_signals=True)

    assert "objective_score" in legacy.columns
    assert "objective_score" in tail.columns
    assert float(tail["objective_score"].mean()) > float(legacy["objective_score"].mean())


def test_apply_projection_calibration_scales_projection_columns() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    base = pool.loc[:, ["projected_dk_points", "blended_projection", "our_dk_projection"]].copy()
    scaled = apply_projection_calibration(pool, projection_scale=0.9)

    assert "projection_calibration_scale" in scaled.columns
    assert float(scaled["projection_calibration_scale"].iloc[0]) == 0.9
    for col in ["projected_dk_points", "blended_projection", "our_dk_projection"]:
        b = pd.to_numeric(base[col], errors="coerce")
        s = pd.to_numeric(scaled[col], errors="coerce")
        diff = (s - (b * 0.9)).abs().fillna(0.0)
        assert float(diff.max()) < 1e-6


def test_projection_salary_bucket_key_segments_expected_ranges() -> None:
    assert projection_salary_bucket_key(4300) == "lt4500"
    assert projection_salary_bucket_key(4500) == "4500_6999"
    assert projection_salary_bucket_key(6999) == "4500_6999"
    assert projection_salary_bucket_key(7000) == "7000_9999"
    assert projection_salary_bucket_key(10500) == "gte10000"


def test_apply_projection_calibration_applies_salary_bucket_scales() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy()
    pool = pool.reset_index(drop=True)
    pool.loc[0, "Salary"] = 4300
    pool.loc[1, "Salary"] = 7600

    base = pool.loc[:, ["Salary", "projected_dk_points", "blended_projection", "our_dk_projection"]].copy()
    scaled = apply_projection_calibration(
        pool,
        projection_scale=1.0,
        projection_salary_bucket_scales={"lt4500": 0.8, "7000_9999": 1.1},
    )

    assert "projection_bucket_scale" in scaled.columns
    assert "projection_total_scale" in scaled.columns
    assert "projection_salary_bucket" in scaled.columns

    lt_row = scaled.iloc[0]
    mid_row = scaled.iloc[1]
    assert lt_row["projection_salary_bucket"] == "lt4500"
    assert float(lt_row["projection_bucket_scale"]) == 0.8
    assert mid_row["projection_salary_bucket"] == "7000_9999"
    assert float(mid_row["projection_bucket_scale"]) == 1.1

    for idx, expected_scale in [(0, 0.8), (1, 1.1)]:
        for col in ["projected_dk_points", "blended_projection", "our_dk_projection"]:
            before = pd.to_numeric(base.loc[idx, col], errors="coerce")
            after = pd.to_numeric(scaled.loc[idx, col], errors="coerce")
            assert abs(float(after) - (float(before) * expected_scale)) < 1e-6


def test_apply_projection_calibration_applies_role_bucket_scales() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy()
    base = pool.loc[:, ["PositionBase", "projected_dk_points", "blended_projection", "our_dk_projection"]].copy()
    scaled = apply_projection_calibration(
        pool,
        projection_scale=1.0,
        projection_role_bucket_scales={"guard": 0.9, "forward": 1.1},
    )

    assert "projection_role_bucket" in scaled.columns
    assert "projection_role_scale" in scaled.columns

    g_idx = int(scaled.loc[scaled["PositionBase"] == "G"].index[0])
    f_idx = int(scaled.loc[scaled["PositionBase"] == "F"].index[0])
    assert scaled.loc[g_idx, "projection_role_bucket"] == "guard"
    assert scaled.loc[f_idx, "projection_role_bucket"] == "forward"
    assert abs(float(scaled.loc[g_idx, "projection_role_scale"]) - 0.9) < 1e-6
    assert abs(float(scaled.loc[f_idx, "projection_role_scale"]) - 1.1) < 1e-6

    for idx, expected_scale in [(g_idx, 0.9), (f_idx, 1.1)]:
        for col in ["projected_dk_points", "blended_projection", "our_dk_projection"]:
            before = pd.to_numeric(base.loc[idx, col], errors="coerce")
            after = pd.to_numeric(scaled.loc[idx, col], errors="coerce")
            assert abs(float(after) - (float(before) * expected_scale)) < 1e-6


def test_apply_projection_uncertainty_adjustment_shrinks_high_risk_rows() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy()
    pool = pool.reset_index(drop=True)
    pool["projection_uncertainty_score"] = 0.0
    pool["dnp_risk_score"] = 0.0
    pool.loc[0, "projection_uncertainty_score"] = 1.0
    pool.loc[0, "dnp_risk_score"] = 0.9

    before = pd.to_numeric(pool["projected_dk_points"], errors="coerce").copy()
    adjusted = apply_projection_uncertainty_adjustment(
        pool,
        uncertainty_weight=0.18,
        high_risk_extra_shrink=0.10,
        dnp_risk_threshold=0.30,
        min_multiplier=0.68,
    )
    after = pd.to_numeric(adjusted["projected_dk_points"], errors="coerce")
    mult = pd.to_numeric(adjusted["projection_uncertainty_multiplier"], errors="coerce")

    assert "projection_uncertainty_multiplier" in adjusted.columns
    assert float(mult.iloc[0]) < 1.0
    assert abs(float(after.iloc[0]) - (float(before.iloc[0]) * float(mult.iloc[0]))) < 1e-3
    # Low-risk rows should remain effectively unchanged.
    assert abs(float(mult.iloc[1]) - 1.0) < 1e-6


def test_apply_minutes_shock_override_boosts_positive_minute_deltas() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    pool["our_minutes_avg"] = 20.0
    pool["our_minutes_last7"] = 20.0
    pool["our_minutes_recent"] = 20.0
    pool["projection_uncertainty_score"] = 0.05
    pool["dnp_risk_score"] = 0.05
    pool["ownership_chalk_surge_score"] = 70.0

    # First row has a clear positive minutes shock signal.
    pool.loc[0, "our_minutes_last7"] = 28.0
    pool.loc[0, "our_minutes_recent"] = 29.0
    before = pd.to_numeric(pool["projected_dk_points"], errors="coerce")
    adjusted = apply_minutes_shock_override(pool)
    after = pd.to_numeric(adjusted["projected_dk_points"], errors="coerce")

    assert "minutes_shock_boost_pct" in adjusted.columns
    assert float(adjusted.loc[0, "minutes_shock_boost_pct"]) > 0.0
    assert float(after.iloc[0]) > float(before.iloc[0])
    assert float(adjusted.loc[1, "minutes_shock_boost_pct"]) <= float(adjusted.loc[0, "minutes_shock_boost_pct"])


def test_apply_chalk_ceiling_guardrail_softly_lifts_top_candidates() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    pool["projected_ownership"] = 10.0
    pool["ownership_chalk_surge_score"] = 45.0
    pool["value_per_1k"] = pd.to_numeric(pool["value_per_1k"], errors="coerce").fillna(0.0)
    pool["minutes_shock_boost_pct"] = 0.0

    # Two strong chalk-ceiling candidates with low projected ownership.
    pool.loc[0, "projected_ownership"] = 8.0
    pool.loc[0, "ownership_chalk_surge_score"] = 92.0
    pool.loc[0, "minutes_shock_boost_pct"] = 12.0
    pool.loc[1, "projected_ownership"] = 7.0
    pool.loc[1, "ownership_chalk_surge_score"] = 90.0
    pool.loc[1, "minutes_shock_boost_pct"] = 10.0

    adjusted = apply_chalk_ceiling_guardrail(pool, max_players=2, min_floor=18.0, max_floor=24.0, blend_weight=0.5)
    lifted = adjusted.loc[adjusted["chalk_ceiling_guardrail_flag"] == True]  # noqa: E712

    assert "chalk_ceiling_guardrail_flag" in adjusted.columns
    assert len(lifted) <= 2
    assert len(lifted) >= 1
    assert float(adjusted.loc[0, "projected_ownership"]) > 8.0


def test_build_player_pool_ownership_surge_columns_present() -> None:
    slate = _sample_slate().copy()
    slate["attention_index"] = [1200, 980, 120, 80, 45, 30, 410, 390, 60, 55, 20, 10]
    slate["field_ownership_pct"] = [42, 39, 16, 13, 8, 6, 28, 24, 10, 9, 5, 3]
    pool = build_player_pool(slate, _sample_props(), bookmaker_filter="fanduel")

    assert "ownership_chalk_surge_score" in pool.columns
    assert "ownership_confidence" in pool.columns
    assert "ownership_leverage_shrink" in pool.columns
    assert "game_stack_focus_score" in pool.columns
    assert "game_stack_focus_flag" in pool.columns
    surge = pd.to_numeric(pool["ownership_chalk_surge_score"], errors="coerce")
    conf = pd.to_numeric(pool["ownership_confidence"], errors="coerce")
    shrink = pd.to_numeric(pool["ownership_leverage_shrink"], errors="coerce")
    assert surge.notna().all()
    assert conf.between(0, 1).all()
    assert shrink.between(0, 0.45).all()
    assert pd.to_numeric(pool["game_stack_focus_score"], errors="coerce").ge(0.0).all()
    assert bool(pool["game_stack_focus_flag"].astype(bool).any()) is True


def test_apply_ownership_surprise_guardrails_raises_triggered_rows() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    pool["projected_ownership"] = 18.0
    pool["ownership_chalk_surge_score"] = 50.0
    pool["projected_dk_points"] = pd.to_numeric(pool["projected_dk_points"], errors="coerce").fillna(20.0)

    pool.loc[0, "projected_ownership"] = 6.0
    pool.loc[0, "ownership_chalk_surge_score"] = 92.0
    pool.loc[0, "projected_dk_points"] = float(pool["projected_dk_points"].max()) + 5.0

    adjusted = apply_ownership_surprise_guardrails(
        pool,
        projected_ownership_threshold=10.0,
        surge_score_threshold=72.0,
        projection_rank_threshold=0.6,
        ownership_floor_base=10.0,
        ownership_floor_cap=24.0,
    )
    assert "ownership_guardrail_flag" in adjusted.columns
    assert "ownership_guardrail_delta" in adjusted.columns
    assert bool(adjusted.loc[0, "ownership_guardrail_flag"]) is True
    assert float(adjusted.loc[0, "projected_ownership"]) > 6.0
    assert float(adjusted.loc[0, "ownership_guardrail_delta"]) > 0.0


def test_apply_focus_game_chalk_guardrail_lifts_focus_game_candidates() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    pool["projected_ownership"] = 10.0
    pool["historical_ownership_baseline"] = 6.0
    pool["team_stack_popularity_score"] = 30.0
    pool["ownership_chalk_surge_score"] = 45.0
    pool["minutes_shock_boost_pct"] = 0.0
    pool["game_stack_focus_score"] = 20.0
    pool["game_stack_focus_flag"] = False

    pool.loc[0, "projected_ownership"] = 9.0
    pool.loc[0, "historical_ownership_baseline"] = 24.0
    pool.loc[0, "team_stack_popularity_score"] = 90.0
    pool.loc[0, "ownership_chalk_surge_score"] = 88.0
    pool.loc[0, "minutes_shock_boost_pct"] = 12.0
    pool.loc[0, "game_stack_focus_score"] = 95.0
    pool.loc[0, "game_stack_focus_flag"] = True

    adjusted = apply_focus_game_chalk_guardrail(
        pool,
        projected_ownership_threshold=18.0,
        projection_rank_threshold=0.0,
        floor_base=14.0,
        floor_cap=30.0,
    )
    assert "focus_game_chalk_guardrail_flag" in adjusted.columns
    assert bool(adjusted.loc[0, "focus_game_chalk_guardrail_flag"]) is True
    assert float(adjusted.loc[0, "projected_ownership"]) > 9.0
    assert float(adjusted.loc[0, "focus_game_chalk_guardrail_delta"]) > 0.0
    assert bool(adjusted.loc[1, "focus_game_chalk_guardrail_flag"]) is False


def test_apply_false_chalk_discount_shrinks_unsupported_high_ownership_and_normalizes_total() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    pool["projected_ownership"] = 12.0
    pool["ownership_chalk_surge_score"] = 35.0
    pool["ownership_confidence"] = 0.25
    pool["field_ownership_pct"] = 6.0
    pool["minutes_shock_boost_pct"] = 0.0
    pool["value_per_1k"] = pd.to_numeric(pool.get("value_per_1k"), errors="coerce").fillna(4.0)

    pool.loc[0, "projected_ownership"] = 28.0
    pool.loc[0, "ownership_chalk_surge_score"] = 24.0
    pool.loc[0, "ownership_confidence"] = 0.10
    pool.loc[0, "field_ownership_pct"] = 3.0

    pool.loc[1, "projected_ownership"] = 28.0
    pool.loc[1, "ownership_chalk_surge_score"] = 95.0
    pool.loc[1, "ownership_confidence"] = 0.95
    pool.loc[1, "field_ownership_pct"] = 42.0
    pool.loc[1, "projected_dk_points"] = float(pd.to_numeric(pool["projected_dk_points"], errors="coerce").max()) + 5.0

    starting_total = float(pd.to_numeric(pool["projected_ownership"], errors="coerce").sum())
    pool["ownership_target_total"] = starting_total
    discounted = apply_false_chalk_discount(pool)
    normalized = normalize_projected_ownership_total(discounted, target_total=starting_total)

    assert bool(normalized.loc[0, "false_chalk_discount_flag"]) is True
    assert float(normalized.loc[0, "false_chalk_discount_delta"]) > 0.0
    assert float(normalized.loc[0, "projected_ownership"]) < 28.0
    assert float(normalized.loc[0, "false_chalk_discount_delta"]) > float(normalized.loc[1, "false_chalk_discount_delta"])
    assert abs(float(pd.to_numeric(normalized["projected_ownership"], errors="coerce").sum()) - starting_total) <= 1.0


def test_apply_model_profile_adjustments_standout_capture_boosts_focus_candidates() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    scored = apply_contest_objective(pool, contest_type="Large GPP", include_tail_signals=True)
    scored["projected_ownership"] = 18.0
    scored["ownership_chalk_surge_score"] = 50.0
    scored["our_minutes_recent"] = pd.to_numeric(scored.get("our_minutes_last7"), errors="coerce").fillna(24.0)
    scored["our_points_recent"] = pd.to_numeric(scored.get("our_points_avg"), errors="coerce").fillna(18.0)
    scored["projection_uncertainty_score"] = 0.10
    scored["dnp_risk_score"] = 0.10

    # Candidate profile: under-owned, strong surge, good form.
    scored.loc[0, "projected_ownership"] = 9.0
    scored.loc[0, "ownership_chalk_surge_score"] = 93.0
    scored.loc[0, "our_minutes_recent"] = float(scored.loc[0, "our_minutes_recent"]) + 4.0
    scored.loc[0, "our_points_recent"] = float(scored.loc[0, "our_points_recent"]) + 6.0
    scored.loc[0, "projection_uncertainty_score"] = 0.05
    scored.loc[0, "dnp_risk_score"] = 0.05

    adjusted = apply_model_profile_adjustments(scored, model_profile="standout_capture_v1")
    assert "model_profile_bonus" in adjusted.columns
    assert "model_profile_focus_flag" in adjusted.columns
    assert float(adjusted.loc[0, "model_profile_bonus"]) > 0.0
    assert bool(adjusted.loc[0, "model_profile_focus_flag"]) is True
    assert float(adjusted.loc[0, "objective_score"]) > float(scored.loc[0, "objective_score"])


def test_apply_model_profile_adjustments_tail_spike_pairs_boosts_variance_candidates() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    scored = apply_contest_objective(pool, contest_type="Large GPP", include_tail_signals=True)
    scored["projected_ownership"] = 16.0
    scored["game_tail_score"] = 48.0
    scored["game_tail_to_ownership_pct"] = 0.35
    scored["leverage_score"] = pd.to_numeric(scored.get("leverage_score"), errors="coerce").fillna(8.0)
    scored["game_volatility_score"] = 0.28
    scored["stack_anchor_focus_flag"] = False
    scored["minutes_shock_boost_pct"] = 3.0
    scored["projection_uncertainty_score"] = 0.12
    scored["dnp_risk_score"] = 0.10

    scored.loc[0, "projected_ownership"] = 8.0
    scored.loc[0, "game_tail_score"] = 97.0
    scored.loc[0, "game_tail_to_ownership_pct"] = 0.93
    scored.loc[0, "leverage_score"] = float(pd.to_numeric(scored["leverage_score"], errors="coerce").fillna(0.0).max()) + 5.0
    scored.loc[0, "game_volatility_score"] = 0.88
    scored.loc[0, "stack_anchor_focus_flag"] = True
    scored.loc[0, "minutes_shock_boost_pct"] = 11.0
    scored.loc[0, "projection_uncertainty_score"] = 0.05
    scored.loc[0, "dnp_risk_score"] = 0.05

    adjusted = apply_model_profile_adjustments(scored, model_profile="tail_spike_pairs")
    assert float(adjusted.loc[0, "model_profile_bonus"]) > 0.0
    assert bool(adjusted.loc[0, "model_profile_focus_flag"]) is True
    assert float(adjusted.loc[0, "objective_score"]) > float(scored.loc[0, "objective_score"])


def test_apply_model_profile_adjustments_chalk_value_capture_boosts_value_chalk() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    scored = apply_contest_objective(pool, contest_type="Large GPP", include_tail_signals=True)
    scored["projected_ownership"] = 14.0
    scored["ownership_chalk_surge_score"] = 55.0
    scored["team_stack_popularity_score"] = 52.0
    scored["ownership_value_tier_z"] = 0.0
    scored["projection_uncertainty_score"] = 0.12
    scored["dnp_risk_score"] = 0.12
    scored["our_minutes_recent"] = pd.to_numeric(scored.get("our_minutes_last7"), errors="coerce").fillna(24.0)
    scored["value_per_1k"] = pd.to_numeric(scored.get("value_per_1k"), errors="coerce").fillna(4.0)
    scored["ownership_salary_bucket"] = scored["Salary"].map(ownership_salary_bucket_key)

    scored.loc[0, "Salary"] = 5000
    scored.loc[0, "ownership_salary_bucket"] = "lt5500"
    scored.loc[0, "projected_ownership"] = 21.0
    scored.loc[0, "ownership_chalk_surge_score"] = 92.0
    scored.loc[0, "team_stack_popularity_score"] = 91.0
    scored.loc[0, "ownership_value_tier_z"] = 2.8
    scored.loc[0, "value_per_1k"] = 6.9
    scored.loc[0, "projection_uncertainty_score"] = 0.05
    scored.loc[0, "dnp_risk_score"] = 0.05
    scored.loc[0, "our_minutes_recent"] = float(scored.loc[0, "our_minutes_recent"]) + 5.0

    adjusted = apply_model_profile_adjustments(scored, model_profile="chalk_value_capture_v1")
    assert float(adjusted.loc[0, "model_profile_bonus"]) > 0.0
    assert bool(adjusted.loc[0, "model_profile_focus_flag"]) is True
    assert float(adjusted.loc[0, "objective_score"]) > float(scored.loc[0, "objective_score"])


def test_apply_model_profile_adjustments_salary_efficiency_ceiling_boosts_ceiling_candidates() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    scored = apply_contest_objective(pool, contest_type="Large GPP", include_tail_signals=True)
    scored["projected_ownership"] = 16.0
    scored["game_tail_score"] = 45.0
    scored["ownership_value_tier_z"] = 0.0
    scored["minutes_shock_boost_pct"] = 2.0
    scored["projection_uncertainty_score"] = 0.12
    scored["dnp_risk_score"] = 0.12
    scored["our_minutes_avg"] = pd.to_numeric(scored.get("our_minutes_avg"), errors="coerce").fillna(24.0)
    scored["our_minutes_recent"] = pd.to_numeric(scored.get("our_minutes_recent"), errors="coerce")
    scored["our_minutes_recent"] = scored["our_minutes_recent"].where(scored["our_minutes_recent"].notna(), scored["our_minutes_avg"])
    scored["value_per_1k"] = pd.to_numeric(scored.get("value_per_1k"), errors="coerce").fillna(4.0)

    scored.loc[0, "Salary"] = 6700
    scored.loc[0, "projected_ownership"] = 11.0
    scored.loc[0, "game_tail_score"] = 96.0
    scored.loc[0, "ownership_value_tier_z"] = 2.4
    scored.loc[0, "minutes_shock_boost_pct"] = 13.0
    scored.loc[0, "value_per_1k"] = 6.8
    scored.loc[0, "projection_uncertainty_score"] = 0.05
    scored.loc[0, "dnp_risk_score"] = 0.04
    scored.loc[0, "our_minutes_recent"] = float(scored.loc[0, "our_minutes_avg"]) + 4.0
    scored.loc[0, "projected_dk_points"] = float(pd.to_numeric(scored["projected_dk_points"], errors="coerce").fillna(0.0).max()) + 8.0

    adjusted = apply_model_profile_adjustments(scored, model_profile="salary_efficiency_ceiling_v1")
    assert float(adjusted.loc[0, "model_profile_bonus"]) > 0.0
    assert bool(adjusted.loc[0, "model_profile_focus_flag"]) is True
    assert float(adjusted.loc[0, "objective_score"]) > float(scored.loc[0, "objective_score"])


def test_generate_lineups_records_model_profile_in_payload() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=6,
        contest_type="Large GPP",
        include_tail_signals=True,
        model_profile="standout_capture_v1",
        random_seed=13,
    )
    assert warnings == []
    assert len(lineups) == 6
    assert all(str(lineup.get("model_profile")) == "standout_capture_v1" for lineup in lineups)
    assert all(float(lineup.get("ceiling_projection") or 0.0) > float(lineup.get("projected_points") or 0.0) for lineup in lineups)


def test_generate_lineups_low_own_bucket_enforces_required_share() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    pool["projected_ownership"] = 28.0
    pool["ownership_confidence"] = 0.0
    pool["game_tail_score"] = 40.0
    pool["leverage_score"] = pd.to_numeric(pool["projected_dk_points"], errors="coerce").fillna(0.0)
    low_own_ids = pool.head(4)["ID"].astype(str).tolist()
    pool.loc[pool["ID"].astype(str).isin(low_own_ids), "projected_ownership"] = 7.0
    pool.loc[pool["ID"].astype(str).isin(low_own_ids), "game_tail_score"] = 85.0
    pool["ownership_target_total"] = float(pd.to_numeric(pool["projected_ownership"], errors="coerce").sum())

    requested_lineups = 12
    required_share = 60.0
    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=requested_lineups,
        contest_type="Large GPP",
        low_own_bucket_exposure_pct=required_share,
        low_own_bucket_min_per_lineup=1,
        low_own_bucket_max_projected_ownership=10.0,
        low_own_bucket_min_projection=10.0,
        low_own_bucket_min_tail_score=55.0,
        random_seed=13,
    )

    assert len(lineups) == requested_lineups
    assert any("Ownership reliability mitigation reduced low-owned forcing" in warning for warning in warnings)
    lineups_with_low_own = sum(1 for lineup in lineups if int(lineup.get("low_own_upside_count") or 0) >= 1)
    scaled_share = required_share * 0.35
    min_expected = int(round((scaled_share / 100.0) * requested_lineups))
    assert lineups_with_low_own >= min_expected


def test_generate_lineups_preferred_game_bonus_increases_preferred_exposure() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    preferred_game = "AAA@BBB"

    base_lineups, base_warnings = generate_lineups(
        pool_df=pool,
        num_lineups=20,
        contest_type="Small GPP",
        random_seed=13,
    )
    boosted_lineups, boosted_warnings = generate_lineups(
        pool_df=pool,
        num_lineups=20,
        contest_type="Small GPP",
        preferred_game_keys=[preferred_game],
        preferred_game_bonus=2.0,
        random_seed=13,
    )

    assert base_warnings == []
    assert boosted_warnings == []
    base_avg = float(
        sum(sum(1 for p in lineup["players"] if str(p.get("game_key") or "").strip().upper() == preferred_game) for lineup in base_lineups)
        / max(1, len(base_lineups))
    )
    boosted_avg = float(
        sum(sum(1 for p in lineup["players"] if str(p.get("game_key") or "").strip().upper() == preferred_game) for lineup in boosted_lineups)
        / max(1, len(boosted_lineups))
    )
    assert boosted_avg >= base_avg


def test_generate_lineups_preferred_game_stack_requirement_enforces_share() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    preferred_game = "CCC@DDD"
    requested_lineups = 12
    required_pct = 50.0

    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=requested_lineups,
        contest_type="Large GPP",
        preferred_game_keys=[preferred_game],
        preferred_game_bonus=1.5,
        preferred_game_stack_lineup_pct=required_pct,
        preferred_game_stack_min_players=3,
        auto_preferred_game_count=0,
        random_seed=13,
    )

    assert len(lineups) == requested_lineups
    assert any("Focus-game stack guardrails active" in warning for warning in warnings)
    met_count = sum(1 for lineup in lineups if bool(lineup.get("preferred_game_stack_met")))
    assert met_count >= int(round((required_pct / 100.0) * requested_lineups))


def test_generate_lineups_ceiling_boost_marks_expected_count() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    requested_lineups = 10
    ceiling_pct = 50.0
    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=requested_lineups,
        contest_type="Large GPP",
        ceiling_boost_lineup_pct=ceiling_pct,
        ceiling_boost_stack_bonus=2.0,
        ceiling_boost_salary_left_target=120,
        random_seed=13,
    )
    assert warnings == []
    assert len(lineups) == requested_lineups
    boosted = sum(1 for lineup in lineups if bool(lineup.get("ceiling_boost_active")))
    assert boosted == int(round((ceiling_pct / 100.0) * requested_lineups))
