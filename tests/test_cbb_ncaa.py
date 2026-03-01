from datetime import date

from college_basketball_dfs.cbb_ncaa import (
    NcaaApiClient,
    NcaaApiClientError,
    extract_game_id,
    fetch_games_with_boxscores,
    prior_day,
)


def test_prior_day_uses_reference_date() -> None:
    assert prior_day(date(2026, 2, 13)) == date(2026, 2, 12)


def test_extract_game_id_from_nested_game_id() -> None:
    payload = {"game": {"gameID": "6503390", "url": "/game/6503390"}}
    assert extract_game_id(payload) == "6503390"


def test_extract_game_id_from_url_when_needed() -> None:
    payload = {"game": {"url": "/game/7000001"}}
    assert extract_game_id(payload) == "7000001"


def test_fetch_games_with_boxscores_collects_one_row() -> None:
    class StubClient:
        def fetch_scoreboard(self, game_date, sport, division, conference):
            assert game_date == date(2026, 2, 12)
            assert sport == "basketball-men"
            assert division == "d1"
            assert conference == "all-conf"
            return {
                "games": [
                    {
                        "game": {
                            "gameID": "12345",
                            "gameState": "final",
                            "startDate": "02/12/2026",
                            "startTime": "7:00 PM ET",
                            "home": {"score": "80", "names": {"short": "Home Team"}},
                            "away": {"score": "72", "names": {"short": "Away Team"}},
                        }
                    }
                ]
            }

        def fetch_boxscore(self, game_id):
            assert game_id == "12345"
            return {"contestId": 12345, "status": "F"}

    result = fetch_games_with_boxscores(
        client=StubClient(),
        game_date=date(2026, 2, 12),
    )
    assert result["game_count"] == 1
    assert result["boxscore_success_count"] == 1
    assert result["boxscore_failure_count"] == 0
    assert result["games"][0]["home_team"] == "Home Team"
    assert result["games"][0]["away_team"] == "Away Team"
    assert result["games"][0]["boxscore"] == {"contestId": 12345, "status": "F"}


def test_fetch_scoreboard_falls_back_to_direct_ncaa(monkeypatch) -> None:
    client = NcaaApiClient(base_url="https://ncaa-api.henrygd.me")

    def fail_get(path, params=None):
        raise NcaaApiClientError("public wrapper down")

    def fake_direct_scoreboard(session, timeout_seconds, *, game_date, sport, division):
        assert game_date == date(2026, 2, 27)
        assert sport == "basketball-men"
        assert division == "d1"
        return {"games": [{"game": {"gameID": "6501730"}}]}

    monkeypatch.setattr(client, "get", fail_get)
    monkeypatch.setattr("college_basketball_dfs.cbb_ncaa._fetch_direct_scoreboard", fake_direct_scoreboard)

    try:
        payload = client.fetch_scoreboard(date(2026, 2, 27))
    finally:
        client.close()

    assert payload == {"games": [{"game": {"gameID": "6501730"}}]}


def test_fetch_boxscore_falls_back_to_direct_ncaa(monkeypatch) -> None:
    client = NcaaApiClient(base_url="https://ncaa-api.henrygd.me")

    def fail_get(path, params=None):
        raise NcaaApiClientError("public wrapper down")

    def fake_direct_boxscore(session, timeout_seconds, *, game_id):
        assert game_id == "6501730"
        return {"contestId": 6501730, "teamBoxscore": []}

    monkeypatch.setattr(client, "get", fail_get)
    monkeypatch.setattr("college_basketball_dfs.cbb_ncaa._fetch_direct_boxscore", fake_direct_boxscore)

    try:
        payload = client.fetch_boxscore("6501730")
    finally:
        client.close()

    assert payload == {"contestId": 6501730, "teamBoxscore": []}
