from datetime import date

from college_basketball_dfs.cbb_odds_pipeline import run_cbb_odds_pipeline


def _sample_odds_payload() -> dict:
    return {
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
                                    {"name": "Home U", "point": -3.5},
                                    {"name": "Away U", "point": 3.5},
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
                    }
                ],
            }
        ],
    }


class FakeStore:
    def __init__(self, cached_payload=None):
        self.bucket_name = "test-bucket"
        self.cached_payload = cached_payload
        self.raw_writes = []
        self.csv_writes = []

    def odds_blob_name(self, game_date: date) -> str:
        return f"cbb/odds/{game_date.isoformat()}.json"

    def odds_games_blob_name(self, game_date: date) -> str:
        return f"cbb/odds_games/{game_date.isoformat()}_odds.csv"

    def read_odds_json(self, game_date: date):
        return self.cached_payload

    def write_odds_json(self, game_date: date, payload: dict) -> str:
        self.raw_writes.append((game_date, payload))
        return self.odds_blob_name(game_date)

    def write_odds_games_csv(self, game_date: date, csv_text: str) -> str:
        self.csv_writes.append((game_date, csv_text))
        return self.odds_games_blob_name(game_date)


def test_odds_pipeline_uses_cache_when_available() -> None:
    store = FakeStore(cached_payload=_sample_odds_payload())
    summary = run_cbb_odds_pipeline(game_date=date(2026, 2, 12), store=store)
    assert summary["odds_cache_hit"] is True
    assert summary["event_count"] == 1
    assert summary["odds_game_rows"] == 1
    assert len(store.raw_writes) == 0
    assert len(store.csv_writes) == 1


def test_odds_pipeline_fetches_when_cache_missing(monkeypatch) -> None:
    store = FakeStore(cached_payload=None)

    class StubOddsClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url

        def close(self) -> None:
            return

        def fetch_game_odds(self, **kwargs):
            return _sample_odds_payload()["events"]

    monkeypatch.setattr("college_basketball_dfs.cbb_odds_pipeline.OddsApiClient", StubOddsClient)
    summary = run_cbb_odds_pipeline(
        game_date=date(2026, 2, 12),
        odds_api_key="key",
        odds_base_url="http://example",
        store=store,
    )
    assert summary["odds_cache_hit"] is False
    assert summary["event_count"] == 1
    assert summary["odds_game_rows"] == 1
    assert len(store.raw_writes) == 1
    assert len(store.csv_writes) == 1
