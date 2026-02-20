from datetime import date

import pytest

from college_basketball_dfs.cbb_odds import flatten_player_props_payload
from college_basketball_dfs.cbb_props_pipeline import (
    _resolve_markets_argument,
    build_arg_parser,
    run_cbb_props_pipeline,
)


def _sample_props_payload() -> dict:
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
                                "key": "player_points",
                                "outcomes": [
                                    {"name": "Over", "description": "John Doe", "point": 15.5, "price": -110},
                                    {"name": "Under", "description": "John Doe", "point": 15.5, "price": -110},
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


class FakeStore:
    def __init__(self, cached_props=None, cached_odds=None):
        self.bucket_name = "test-bucket"
        self.cached_props = cached_props
        self.cached_odds = cached_odds
        self.props_writes = []
        self.props_csv_writes = []

    def props_blob_name(self, game_date: date) -> str:
        return f"cbb/props/{game_date.isoformat()}.json"

    def props_lines_blob_name(self, game_date: date) -> str:
        return f"cbb/props_lines/{game_date.isoformat()}_props.csv"

    def read_props_json(self, game_date: date):
        return self.cached_props

    def read_odds_json(self, game_date: date):
        return self.cached_odds

    def write_props_json(self, game_date: date, payload: dict) -> str:
        self.props_writes.append((game_date, payload))
        return self.props_blob_name(game_date)

    def write_props_lines_csv(self, game_date: date, csv_text: str) -> str:
        self.props_csv_writes.append((game_date, csv_text))
        return self.props_lines_blob_name(game_date)


def test_flatten_player_props_payload() -> None:
    rows = flatten_player_props_payload(_sample_props_payload())
    assert len(rows) == 1
    row = rows[0]
    assert row["market"] == "player_points"
    assert row["player_name"] == "John Doe"
    assert row["line"] == 15.5
    assert row["over_price"] == -110.0
    assert row["under_price"] == -110.0


def test_flatten_player_props_payload_handles_wrapped_event_data() -> None:
    wrapped = {
        "game_date": "2026-02-12",
        "events": [{"id": "evt-1", "home_team": "Home U", "away_team": "Away U", "data": _sample_props_payload()["events"][0]}],
    }
    rows = flatten_player_props_payload(wrapped)
    assert len(rows) == 1
    assert rows[0]["player_name"] == "John Doe"


def test_props_pipeline_uses_cache_when_available() -> None:
    store = FakeStore(cached_props=_sample_props_payload())
    summary = run_cbb_props_pipeline(game_date=date(2026, 2, 12), store=store)
    assert summary["props_cache_hit"] is True
    assert summary["event_count"] == 1
    assert summary["prop_rows"] == 1
    assert len(store.props_writes) == 0
    assert len(store.props_csv_writes) == 1


def test_props_pipeline_fetches_when_cache_missing(monkeypatch) -> None:
    store = FakeStore(cached_props=None, cached_odds={"events": [{"id": "evt-1"}]})

    class StubOddsClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url

        def close(self) -> None:
            return

        def fetch_game_odds(self, **kwargs):
            return [{"id": "evt-1"}]

        def fetch_event_odds(self, **kwargs):
            return _sample_props_payload()["events"][0]

    monkeypatch.setattr("college_basketball_dfs.cbb_props_pipeline.OddsApiClient", StubOddsClient)
    summary = run_cbb_props_pipeline(
        game_date=date(2026, 2, 12),
        odds_api_key="key",
        store=store,
    )
    assert summary["props_cache_hit"] is False
    assert summary["event_count"] == 1
    assert summary["prop_rows"] == 1
    assert len(store.props_writes) == 1
    assert len(store.props_csv_writes) == 1


def test_props_pipeline_historical_uses_pregame_snapshot(monkeypatch) -> None:
    store = FakeStore(
        cached_props=None,
        cached_odds={"events": [{"id": "evt-1", "commence_time": "2026-02-12T23:00:00Z"}]},
    )
    seen = {}

    class StubOddsClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url

        def close(self) -> None:
            return

        def fetch_game_odds(self, **kwargs):
            return [{"id": "evt-1", "commence_time": "2026-02-12T23:00:00Z"}]

        def fetch_event_odds(self, **kwargs):
            seen["snapshot"] = kwargs.get("historical_snapshot_time")
            return _sample_props_payload()["events"][0]

    monkeypatch.setattr("college_basketball_dfs.cbb_props_pipeline.OddsApiClient", StubOddsClient)
    run_cbb_props_pipeline(
        game_date=date(2026, 2, 12),
        odds_api_key="key",
        historical_mode=True,
        historical_snapshot_time=None,
        store=store,
    )
    assert seen["snapshot"] == "2026-02-12T22:45:00Z"


def test_props_pipeline_returns_diagnostics_fields() -> None:
    store = FakeStore(cached_props=_sample_props_payload())
    summary = run_cbb_props_pipeline(game_date=date(2026, 2, 12), store=store)
    assert "events_with_bookmakers" in summary
    assert "events_with_requested_markets" in summary
    assert "total_bookmakers" in summary
    assert "total_requested_markets" in summary
    assert "total_outcomes" in summary


def test_props_pipeline_applies_inter_event_sleep(monkeypatch) -> None:
    store = FakeStore(
        cached_props=None,
        cached_odds={"events": [{"id": "evt-1"}, {"id": "evt-2"}]},
    )
    sleep_calls: list[float] = []

    class StubOddsClient:
        def __init__(self, api_key: str, base_url: str) -> None:
            self.api_key = api_key
            self.base_url = base_url

        def close(self) -> None:
            return

        def fetch_game_odds(self, **kwargs):
            return [{"id": "evt-1"}, {"id": "evt-2"}]

        def fetch_event_odds(self, **kwargs):
            payload = _sample_props_payload()["events"][0].copy()
            payload["id"] = kwargs["event_id"]
            return payload

    monkeypatch.setattr("college_basketball_dfs.cbb_props_pipeline.OddsApiClient", StubOddsClient)
    monkeypatch.setattr("college_basketball_dfs.cbb_props_pipeline.time.sleep", lambda s: sleep_calls.append(float(s)))
    run_cbb_props_pipeline(
        game_date=date(2026, 2, 12),
        odds_api_key="key",
        inter_event_sleep_seconds=0.25,
        store=store,
    )
    assert sleep_calls == [0.25]


def test_resolve_markets_argument_handles_split_cloud_run_args() -> None:
    parser = build_arg_parser()
    resolved = _resolve_markets_argument(
        parser=parser,
        markets_csv="player_points",
        unknown_args=["player_rebounds", "player_assists"],
    )
    assert resolved == "player_points,player_rebounds,player_assists"


def test_resolve_markets_argument_rejects_unknown_flags() -> None:
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        _resolve_markets_argument(
            parser=parser,
            markets_csv="player_points",
            unknown_args=["--unexpected-flag"],
        )
