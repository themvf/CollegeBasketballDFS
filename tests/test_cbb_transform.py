from college_basketball_dfs.cbb_transform import flatten_games_payload, rows_to_csv_text


def test_flatten_games_payload_builds_player_rows_with_fantasy_points() -> None:
    payload = {
        "game_date": "2026-02-12",
        "game_count": 1,
        "games": [
            {
                "game_id": "1234",
                "status": "final",
                "start_date": "02/12/2026",
                "start_time": "7:00 PM ET",
                "home_team": "Home U",
                "away_team": "Away U",
                "home_score": "80",
                "away_score": "72",
                "boxscore": {
                    "teams": [
                        {"teamId": "1", "isHome": True, "nameShort": "Home U"},
                        {"teamId": "2", "isHome": False, "nameShort": "Away U"},
                    ],
                    "teamBoxscore": [
                        {
                            "teamId": 1,
                            "teamStats": {"points": "80", "totalRebounds": "35", "assists": "18", "turnovers": "12"},
                            "playerStats": [
                                {
                                    "id": 10,
                                    "number": 3,
                                    "firstName": "Jane",
                                    "lastName": "Doe",
                                    "position": "G",
                                    "starter": True,
                                    "minutesPlayed": "32.5",
                                    "points": "20",
                                    "totalRebounds": "8",
                                    "assists": "4",
                                    "steals": "2",
                                    "blockedShots": "1",
                                    "turnovers": "3",
                                    "personalFouls": "2",
                                    "fieldGoalsMade": "7",
                                    "fieldGoalsAttempted": "14",
                                    "freeThrowsMade": "3",
                                    "freeThrowsAttempted": "4",
                                    "threePointsMade": "3",
                                    "threePointsAttempted": "6",
                                    "offensiveRebounds": "1",
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    }

    rows = flatten_games_payload(payload)
    assert len(rows) == 1
    row = rows[0]
    assert row["player_name"] == "Jane Doe"
    assert row["team_name"] == "Home U"
    assert row["opponent_team"] == "Away U"
    assert row["minutes_played"] == 32.5
    assert row["points"] == 20
    assert row["dk_fpts"] == 42.0
    assert row["fd_fpts"] == 41.6


def test_flatten_games_payload_skips_games_without_boxscore() -> None:
    payload = {"game_date": "2026-02-12", "games": [{"game_id": "1234", "boxscore": None}]}
    assert flatten_games_payload(payload) == []


def test_rows_to_csv_text_has_header_and_row() -> None:
    csv_text = rows_to_csv_text([{"a": 1, "b": "x"}])
    lines = [line.strip() for line in csv_text.strip().splitlines()]
    assert lines[0] == "a,b"
    assert lines[1] == "1,x"
