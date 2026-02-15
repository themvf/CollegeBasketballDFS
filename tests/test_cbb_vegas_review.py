from __future__ import annotations

from college_basketball_dfs.cbb_vegas_review import (
    build_calibration_models_frame,
    build_spread_buckets_frame,
    build_total_buckets_frame,
    build_vegas_review_games_frame,
    summarize_vegas_accuracy,
)


def _event_payload(
    event_id: str,
    home_team: str,
    away_team: str,
    home_ml: int,
    away_ml: int,
    home_spread: float,
    away_spread: float,
    total: float,
) -> dict:
    return {
        "id": event_id,
        "home_team": home_team,
        "away_team": away_team,
        "bookmakers": [
            {
                "key": "fanduel",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home_team, "price": home_ml},
                            {"name": away_team, "price": away_ml},
                        ],
                    },
                    {
                        "key": "spreads",
                        "outcomes": [
                            {"name": home_team, "point": home_spread, "price": -110},
                            {"name": away_team, "point": away_spread, "price": -110},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "point": total, "price": -110},
                            {"name": "Under", "point": total, "price": -110},
                        ],
                    },
                ],
            }
        ],
    }


def test_build_vegas_review_games_frame_matches_and_scores() -> None:
    raw_payloads = [
        {
            "game_date": "2026-01-10",
            "games": [
                {"home_team": "Alpha", "away_team": "Beta", "home_score": 80, "away_score": 70, "status": "final"},
                {"home_team": "Gamma", "away_team": "Delta", "home_score": 65, "away_score": 60, "status": "final"},
            ],
        }
    ]
    odds_payloads = [
        {
            "game_date": "2026-01-10",
            "events": [
                _event_payload(
                    event_id="e1",
                    home_team="Alpha",
                    away_team="Beta",
                    home_ml=-250,
                    away_ml=210,
                    home_spread=-6.0,
                    away_spread=6.0,
                    total=146.0,
                ),
                # Reversed home/away vs raw payload to verify orientation handling.
                _event_payload(
                    event_id="e2",
                    home_team="Delta",
                    away_team="Gamma",
                    home_ml=-170,
                    away_ml=145,
                    home_spread=-3.0,
                    away_spread=3.0,
                    total=128.0,
                ),
            ],
        }
    ]

    games_df = build_vegas_review_games_frame(raw_payloads=raw_payloads, odds_payloads=odds_payloads)
    assert len(games_df) == 2

    alpha_row = games_df.loc[(games_df["home_team"] == "Alpha") & (games_df["away_team"] == "Beta")].iloc[0]
    assert float(alpha_row["actual_total"]) == 150.0
    assert float(alpha_row["total_points"]) == 146.0
    assert float(alpha_row["vegas_home_margin"]) == 6.0
    assert bool(alpha_row["winner_pick_correct"]) is True

    gamma_row = games_df.loc[(games_df["home_team"] == "Gamma") & (games_df["away_team"] == "Delta")].iloc[0]
    assert float(gamma_row["actual_total"]) == 125.0
    assert float(gamma_row["total_points"]) == 128.0
    assert float(gamma_row["vegas_home_margin"]) == -3.0
    assert bool(gamma_row["winner_pick_correct"]) is False


def test_summarize_vegas_accuracy_and_models() -> None:
    raw_payloads = [
        {
            "game_date": "2026-01-10",
            "games": [
                {"home_team": "Alpha", "away_team": "Beta", "home_score": 80, "away_score": 70, "status": "final"},
                {"home_team": "Gamma", "away_team": "Delta", "home_score": 65, "away_score": 60, "status": "final"},
            ],
        }
    ]
    odds_payloads = [
        {
            "game_date": "2026-01-10",
            "events": [
                _event_payload(
                    event_id="e1",
                    home_team="Alpha",
                    away_team="Beta",
                    home_ml=-250,
                    away_ml=210,
                    home_spread=-6.0,
                    away_spread=6.0,
                    total=146.0,
                ),
                _event_payload(
                    event_id="e2",
                    home_team="Delta",
                    away_team="Gamma",
                    home_ml=-170,
                    away_ml=145,
                    home_spread=-3.0,
                    away_spread=3.0,
                    total=128.0,
                ),
            ],
        }
    ]
    games_df = build_vegas_review_games_frame(raw_payloads=raw_payloads, odds_payloads=odds_payloads)

    summary = summarize_vegas_accuracy(games_df)
    assert int(summary["total_games"]) == 2
    assert int(summary["odds_matched_games"]) == 2
    assert round(float(summary["total_mae"]), 3) == 3.5
    assert round(float(summary["spread_mae"]), 3) == 6.0
    assert round(float(summary["winner_pick_accuracy_pct"]), 1) == 50.0
    assert round(float(summary["total_within_5_pct"]), 1) == 100.0

    models = build_calibration_models_frame(games_df)
    assert len(models) == 2
    assert set(models["model"].tolist()) == {"Total Points Calibration", "Spread Margin Calibration"}
    assert all(int(x) == 2 for x in models["samples"].tolist())
    assert all(float(x) <= 0.0 for x in models["mae_delta"].tolist())

    total_buckets = build_total_buckets_frame(games_df)
    spread_buckets = build_spread_buckets_frame(games_df)
    assert not total_buckets.empty
    assert not spread_buckets.empty


def test_build_vegas_review_games_frame_fuzzy_name_match() -> None:
    raw_payloads = [
        {
            "game_date": "2026-01-11",
            "games": [
                {"home_team": "Duke", "away_team": "North Carolina", "home_score": 82, "away_score": 78, "status": "final"},
            ],
        }
    ]
    odds_payloads = [
        {
            "game_date": "2026-01-11",
            "events": [
                _event_payload(
                    event_id="e3",
                    home_team="Duke Blue Devils",
                    away_team="North Carolina Tar Heels",
                    home_ml=-140,
                    away_ml=120,
                    home_spread=-2.0,
                    away_spread=2.0,
                    total=154.0,
                )
            ],
        }
    ]
    games_df = build_vegas_review_games_frame(raw_payloads=raw_payloads, odds_payloads=odds_payloads)
    assert len(games_df) == 1
    row = games_df.iloc[0]
    assert str(row["event_id"]) == "e3"
    assert str(row["odds_match_type"]) == "fuzzy"
    assert float(row["odds_match_score"]) >= 0.72
