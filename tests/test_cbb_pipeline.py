from datetime import date

from college_basketball_dfs.cbb_pipeline import run_cbb_cache_pipeline


def _sample_payload() -> dict:
    return {
        "game_date": "2026-02-12",
        "game_count": 1,
        "boxscore_success_count": 1,
        "boxscore_failure_count": 0,
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


class FakeStore:
    def __init__(self, cached_payload=None):
        self.bucket_name = "test-bucket"
        self.cached_payload = cached_payload
        self.raw_writes = []
        self.players_writes = []

    def raw_blob_name(self, game_date: date) -> str:
        return f"cbb/raw/{game_date.isoformat()}.json"

    def players_blob_name(self, game_date: date) -> str:
        return f"cbb/players/{game_date.isoformat()}_players.csv"

    def read_raw_json(self, game_date: date):
        return self.cached_payload

    def write_raw_json(self, game_date: date, payload: dict) -> str:
        self.raw_writes.append((game_date, payload))
        return self.raw_blob_name(game_date)

    def write_players_csv(self, game_date: date, csv_text: str) -> str:
        self.players_writes.append((game_date, csv_text))
        return self.players_blob_name(game_date)


def test_pipeline_uses_cache_when_available() -> None:
    store = FakeStore(cached_payload=_sample_payload())
    summary = run_cbb_cache_pipeline(game_date=date(2026, 2, 12), store=store)

    assert summary["raw_cache_hit"] is True
    assert summary["game_count"] == 1
    assert summary["player_row_count"] == 1
    assert len(store.raw_writes) == 0
    assert len(store.players_writes) == 1


def test_pipeline_fetches_when_cache_missing(monkeypatch) -> None:
    store = FakeStore(cached_payload=None)

    class StubClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

        def close(self) -> None:
            return

    monkeypatch.setattr("college_basketball_dfs.cbb_pipeline.NcaaApiClient", StubClient)
    monkeypatch.setattr("college_basketball_dfs.cbb_pipeline.fetch_games_with_boxscores", lambda **kwargs: _sample_payload())

    summary = run_cbb_cache_pipeline(
        game_date=date(2026, 2, 12),
        ncaa_base_url="http://example",
        store=store,
    )

    assert summary["raw_cache_hit"] is False
    assert summary["game_count"] == 1
    assert summary["player_row_count"] == 1
    assert len(store.raw_writes) == 1
    assert len(store.players_writes) == 1
