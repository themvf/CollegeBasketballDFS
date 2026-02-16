from collections import Counter

import pandas as pd

from college_basketball_dfs.cbb_dk_optimizer import (
    apply_contest_objective,
    build_dk_upload_csv,
    build_player_pool,
    generate_lineups,
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


def test_build_player_pool_merges_props() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    g1 = pool.loc[pool["Name"] == "Guard 1"].iloc[0]
    f1 = pool.loc[pool["Name"] == "Forward 1"].iloc[0]
    assert g1["vegas_points_line"] == 18.5
    assert g1["vegas_assists_line"] == 5.5
    assert f1["vegas_threes_line"] == 1.5
    assert g1["projected_dk_points"] > 0
    assert f1["projected_ownership"] > 0


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
