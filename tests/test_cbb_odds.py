from datetime import date

from college_basketball_dfs.cbb_odds import OddsApiClient, flatten_odds_payload


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


def test_flatten_odds_payload_handles_historical_wrapper_data() -> None:
    payload = {
        "game_date": "2026-02-12",
        "events": [
            {
                "id": "evt-2",
                "commence_time": "2026-02-12T22:00:00Z",
                "home_team": "Team A",
                "away_team": "Team B",
                "bookmakers": [],
            }
        ],
    }
    rows = flatten_odds_payload(payload)
    assert len(rows) == 1
    assert rows[0]["home_team"] == "Team A"


def test_fetch_game_odds_historical_reads_data_field(monkeypatch) -> None:
    captured = {}

    def fake_get(self, path, params):
        captured["path"] = path
        captured["params"] = params
        return {"data": [{"id": "evt"}]}

    monkeypatch.setattr("college_basketball_dfs.cbb_odds.OddsApiClient.get", fake_get)
    client = OddsApiClient(api_key="x")
    try:
        rows = client.fetch_game_odds(game_date=date(2026, 2, 12), historical=True)
    finally:
        client.close()

    assert rows == [{"id": "evt"}]
    assert captured["path"].endswith("/historical/sports/basketball_ncaab/odds")
    assert "date" in captured["params"]


def test_fetch_game_odds_includes_bookmakers_param(monkeypatch) -> None:
    captured = {}

    def fake_get(self, path, params):
        captured["params"] = params
        return []

    monkeypatch.setattr("college_basketball_dfs.cbb_odds.OddsApiClient.get", fake_get)
    client = OddsApiClient(api_key="x")
    try:
        client.fetch_game_odds(game_date=date(2026, 2, 12), bookmakers="fanduel")
    finally:
        client.close()

    assert captured["params"]["bookmakers"] == "fanduel"


def test_fetch_event_odds_historical_reads_data_field(monkeypatch) -> None:
    captured = {}

    def fake_get(self, path, params):
        captured["path"] = path
        captured["params"] = params
        return {"timestamp": "x", "data": {"id": "evt-1", "bookmakers": []}}

    monkeypatch.setattr("college_basketball_dfs.cbb_odds.OddsApiClient.get", fake_get)
    client = OddsApiClient(api_key="x")
    try:
        event = client.fetch_event_odds(
            event_id="evt-1",
            historical=True,
            historical_snapshot_time="2026-02-12T23:59:59Z",
        )
    finally:
        client.close()

    assert event == {"id": "evt-1", "bookmakers": []}
    assert captured["path"].endswith("/historical/sports/basketball_ncaab/events/evt-1/odds")
    assert captured["params"]["date"] == "2026-02-12T23:59:59Z"


def test_get_retries_429_then_succeeds(monkeypatch) -> None:
    class StubResponse:
        def __init__(self, status_code: int, payload, text: str = "", headers: dict | None = None) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = text
            self.headers = headers or {}

        def json(self):
            return self._payload

    responses = [
        StubResponse(429, {"message": "rate limited"}, text='{"message":"rate limited"}', headers={"Retry-After": "0"}),
        StubResponse(200, [{"id": "evt-1"}]),
    ]

    sleep_calls: list[float] = []

    client = OddsApiClient(
        api_key="x",
        min_interval_seconds=0.0,
        max_retries=2,
        retry_backoff_seconds=0.01,
        max_retry_backoff_seconds=0.01,
    )
    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: responses.pop(0))
    monkeypatch.setattr("college_basketball_dfs.cbb_odds.time.sleep", lambda s: sleep_calls.append(float(s)))
    try:
        payload = client.get("/sports/basketball_ncaab/odds", {})
    finally:
        client.close()

    assert payload == [{"id": "evt-1"}]
    assert len(sleep_calls) >= 1
