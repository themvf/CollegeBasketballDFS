from college_basketball_dfs.cbb_odds import flatten_odds_payload


def test_flatten_odds_payload_builds_consensus_row() -> None:
    payload = {
        "game_date": "2026-02-12",
        "events": [
            {
                "id": "evt-1",
                "commence_time": "2026-02-12T23:00:00Z",
                "home_team": "Home U",
                "away_team": "Away U",
                "bookmakers": [
                    {
                        "key": "book1",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Home U", "price": -130},
                                    {"name": "Away U", "price": 110},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Home U", "point": -3.5, "price": -110},
                                    {"name": "Away U", "point": 3.5, "price": -110},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 149.5, "price": -105},
                                    {"name": "Under", "point": 149.5, "price": -115},
                                ],
                            },
                        ],
                    },
                    {
                        "key": "book2",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Home U", "price": -125},
                                    {"name": "Away U", "price": 108},
                                ],
                            },
                            {
                                "key": "spreads",
                                "outcomes": [
                                    {"name": "Home U", "point": -3.0, "price": -112},
                                    {"name": "Away U", "point": 3.0, "price": -108},
                                ],
                            },
                            {
                                "key": "totals",
                                "outcomes": [
                                    {"name": "Over", "point": 150.0, "price": -110},
                                    {"name": "Under", "point": 150.0, "price": -110},
                                ],
                            },
                        ],
                    },
                ],
            }
        ],
    }

    rows = flatten_odds_payload(payload)
    assert len(rows) == 1
    row = rows[0]
    assert row["game_date"] == "2026-02-12"
    assert row["home_team"] == "Home U"
    assert row["away_team"] == "Away U"
    assert row["bookmakers_count"] == 2
    assert row["moneyline_home"] == -127.5
    assert row["moneyline_away"] == 109.0
    assert row["spread_home"] == -3.25
    assert row["spread_away"] == 3.25
    assert row["total_points"] == 149.75
    assert row["moneyline_samples"] == 2
    assert row["spread_samples"] == 2
    assert row["total_samples"] == 2
