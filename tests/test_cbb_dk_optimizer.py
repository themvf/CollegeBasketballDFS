from collections import Counter

import pandas as pd

from college_basketball_dfs.cbb_dk_optimizer import (
    apply_projection_uncertainty_adjustment,
    apply_projection_calibration,
    apply_ownership_surprise_guardrails,
    apply_contest_objective,
    build_dk_upload_csv,
    build_player_pool,
    generate_lineups,
    normalize_injuries_frame,
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
    assert set(pool["ownership_model"].astype(str).unique().tolist()) == {"v2_softmax"}
    own = pd.to_numeric(pool["projected_ownership"], errors="coerce")
    assert own.notna().all()
    assert (own >= 0).all()
    assert (own <= 100).all()
    # Ownership target is 800% across players before clipping.
    assert abs(float(own.sum()) - 800.0) <= 5.0


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


def test_build_player_pool_ownership_surge_columns_present() -> None:
    slate = _sample_slate().copy()
    slate["attention_index"] = [1200, 980, 120, 80, 45, 30, 410, 390, 60, 55, 20, 10]
    slate["field_ownership_pct"] = [42, 39, 16, 13, 8, 6, 28, 24, 10, 9, 5, 3]
    pool = build_player_pool(slate, _sample_props(), bookmaker_filter="fanduel")

    assert "ownership_chalk_surge_score" in pool.columns
    assert "ownership_confidence" in pool.columns
    assert "ownership_leverage_shrink" in pool.columns
    surge = pd.to_numeric(pool["ownership_chalk_surge_score"], errors="coerce")
    conf = pd.to_numeric(pool["ownership_confidence"], errors="coerce")
    shrink = pd.to_numeric(pool["ownership_leverage_shrink"], errors="coerce")
    assert surge.notna().all()
    assert conf.between(0, 1).all()
    assert shrink.between(0, 0.45).all()


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


def test_generate_lineups_low_own_bucket_enforces_required_share() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel").copy().reset_index(drop=True)
    pool["projected_ownership"] = 28.0
    pool["game_tail_score"] = 40.0
    pool["leverage_score"] = pd.to_numeric(pool["projected_dk_points"], errors="coerce").fillna(0.0)
    low_own_ids = pool.head(4)["ID"].astype(str).tolist()
    pool.loc[pool["ID"].astype(str).isin(low_own_ids), "projected_ownership"] = 7.0
    pool.loc[pool["ID"].astype(str).isin(low_own_ids), "game_tail_score"] = 85.0

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
    assert warnings == []
    lineups_with_low_own = sum(1 for lineup in lineups if int(lineup.get("low_own_upside_count") or 0) >= 1)
    min_expected = int(round((required_share / 100.0) * requested_lineups))
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
