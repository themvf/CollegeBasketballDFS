from __future__ import annotations

import pandas as pd

from college_basketball_dfs.cbb_tail_model import (
    fit_total_tail_model,
    map_slate_games_to_tail_features,
    score_odds_games_for_tail,
)


def test_fit_and_score_total_tail_model() -> None:
    history = pd.DataFrame(
        [
            {"has_total_line": True, "total_error": 6.0, "total_points": 148.5, "vegas_home_margin": 3.0},
            {"has_total_line": True, "total_error": -4.0, "total_points": 142.0, "vegas_home_margin": -2.5},
            {"has_total_line": True, "total_error": 10.0, "total_points": 156.0, "vegas_home_margin": 8.0},
            {"has_total_line": True, "total_error": -8.0, "total_points": 134.0, "vegas_home_margin": -11.0},
            {"has_total_line": True, "total_error": 2.0, "total_points": 150.0, "vegas_home_margin": 1.0},
            {"has_total_line": True, "total_error": -1.5, "total_points": 145.0, "vegas_home_margin": -5.5},
        ]
    )
    model = fit_total_tail_model(history)
    assert int(model["samples"]) == 6
    assert len(model["mu_beta"]) == 3
    assert len(model["sigma_beta"]) == 3

    odds = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "home_team": "Purdue Boilermakers",
                "away_team": "Iowa Hawkeyes",
                "total_points": 149.5,
                "spread_home": -4.5,
                "spread_away": 4.5,
            },
            {
                "event_id": "evt2",
                "home_team": "Baylor Bears",
                "away_team": "Kansas Jayhawks",
                "total_points": 137.5,
                "spread_home": -1.5,
                "spread_away": 1.5,
            },
        ]
    )
    scored = score_odds_games_for_tail(odds, model)
    assert "p_plus_8" in scored.columns
    assert "p_plus_12" in scored.columns
    assert "tail_sigma" in scored.columns
    assert scored["p_plus_8"].between(0, 1).all()
    assert scored["p_plus_12"].between(0, 1).all()
    assert (scored["tail_sigma"] > 0).all()


def test_map_slate_games_to_tail_features() -> None:
    odds_scored = pd.DataFrame(
        [
            {
                "event_id": "evt1",
                "home_team": "Purdue Boilermakers",
                "away_team": "Iowa Hawkeyes",
                "total_points": 149.5,
                "spread_home": -4.5,
                "tail_residual_mu": 0.7,
                "tail_sigma": 11.2,
                "p_plus_8": 0.26,
                "p_plus_12": 0.14,
                "volatility_score": 11.2,
            }
        ]
    )
    mapped = map_slate_games_to_tail_features(
        slate_game_keys=["IOWA@PURDUE", "BAY@UL"],
        odds_tail_df=odds_scored,
    )
    assert len(mapped) == 1
    row = mapped.iloc[0]
    assert row["game_key"] == "IOWA@PURDUE"
    assert row["game_tail_event_id"] == "evt1"
    assert float(row["game_p_plus_8"]) == 0.26
    assert float(row["game_p_plus_12"]) == 0.14
