from college_basketball_dfs.cbb_team_lookup import rows_from_payloads, rows_from_raw_payload


def test_rows_from_raw_payload_returns_home_and_away_rows() -> None:
    payload = {
        "game_date": "2026-02-12",
        "games": [
            {
                "home_team": "UMass Lowell",
                "away_team": "Bryant",
                "home_score": "88",
                "away_score": "69",
            }
        ],
    }
    rows = rows_from_raw_payload(payload)
    assert len(rows) == 2

    home_row = rows[0]
    away_row = rows[1]

    assert home_row["Team"] == "UMass Lowell"
    assert home_row["Home/Away"] == "Home"
    assert home_row["Team Score"] == 88
    assert home_row["Opponent"] == "Bryant"
    assert home_row["Opponent Score"] == 69
    assert home_row["W/L"] == "W"
    assert home_row["Venue"] == "UMass Lowell"

    assert away_row["Team"] == "Bryant"
    assert away_row["Home/Away"] == "Away"
    assert away_row["Team Score"] == 69
    assert away_row["Opponent"] == "UMass Lowell"
    assert away_row["Opponent Score"] == 88
    assert away_row["W/L"] == "L"
    assert away_row["Venue"] == "UMass Lowell"


def test_rows_from_payloads_combines_multiple_days() -> None:
    payloads = [
        {"game_date": "2026-02-12", "games": [{"home_team": "A", "away_team": "B", "home_score": "70", "away_score": "60"}]},
        {"game_date": "2026-02-13", "games": [{"home_team": "C", "away_team": "D", "home_score": "61", "away_score": "65"}]},
    ]
    rows = rows_from_payloads(payloads)
    assert len(rows) == 4
