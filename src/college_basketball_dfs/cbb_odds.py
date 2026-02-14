from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping

import requests


class OddsApiError(RuntimeError):
    """Raised for The Odds API failures."""


def _day_window_utc(game_date: date) -> tuple[str, str]:
    start = datetime(game_date.year, game_date.month, game_date.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace("+00:00", "Z")


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _median(values: list[float | None]) -> float | None:
    cleaned = [x for x in values if x is not None]
    if not cleaned:
        return None
    return float(statistics.median(cleaned))


def _outcome_for_team(outcomes: list[Mapping[str, Any]], team_name: str) -> Mapping[str, Any] | None:
    for outcome in outcomes:
        if str(outcome.get("name", "")).strip().lower() == team_name.strip().lower():
            return outcome
    return None


def _outcome_for_name(outcomes: list[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    for outcome in outcomes:
        if str(outcome.get("name", "")).strip().lower() == name.strip().lower():
            return outcome
    return None


@dataclass
class OddsApiClient:
    api_key: str
    base_url: str = "https://api.the-odds-api.com/v4"
    timeout_seconds: int = 20

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def get(self, path: str, params: dict[str, Any]) -> Any:
        full_params = {"apiKey": self.api_key, **params}
        try:
            response = self.session.get(self._url(path), params=full_params, timeout=self.timeout_seconds)
        except requests.RequestException as exc:
            raise OddsApiError(f"GET {path} failed: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:500]
            raise OddsApiError(f"GET {path} failed ({response.status_code}): {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise OddsApiError(f"GET {path} returned invalid JSON") from exc

    def fetch_game_odds(
        self,
        game_date: date,
        sport_key: str = "basketball_ncaab",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        odds_format: str = "american",
        date_format: str = "iso",
        historical: bool = False,
        historical_snapshot_time: str | None = None,
    ) -> list[dict[str, Any]]:
        commence_from, commence_to = _day_window_utc(game_date)
        if historical:
            snapshot_time = historical_snapshot_time or f"{game_date.isoformat()}T23:59:59Z"
            payload = self.get(
                path=f"/historical/sports/{sport_key}/odds",
                params={
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                    "dateFormat": date_format,
                    "date": snapshot_time,
                    "commenceTimeFrom": commence_from,
                    "commenceTimeTo": commence_to,
                },
            )
            if not isinstance(payload, dict):
                raise OddsApiError(f"Unexpected historical odds payload type: {type(payload).__name__}")
            data = payload.get("data")
            if not isinstance(data, list):
                raise OddsApiError("Historical odds payload missing list field `data`.")
            return [x for x in data if isinstance(x, dict)]

        payload = self.get(
            path=f"/sports/{sport_key}/odds",
            params={
                "regions": regions,
                "markets": markets,
                "oddsFormat": odds_format,
                "dateFormat": date_format,
                "commenceTimeFrom": commence_from,
                "commenceTimeTo": commence_to,
            },
        )
        if not isinstance(payload, list):
            raise OddsApiError(f"Unexpected odds payload type: {type(payload).__name__}")
        return [x for x in payload if isinstance(x, dict)]

    def fetch_event_odds(
        self,
        event_id: str,
        sport_key: str = "basketball_ncaab",
        regions: str = "us",
        markets: str = "player_points,player_rebounds,player_assists",
        odds_format: str = "american",
        date_format: str = "iso",
        historical: bool = False,
        historical_snapshot_time: str | None = None,
    ) -> dict[str, Any]:
        if historical:
            if not historical_snapshot_time:
                raise OddsApiError("historical_snapshot_time is required for historical event odds.")
            payload = self.get(
                path=f"/historical/sports/{sport_key}/events/{event_id}/odds",
                params={
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                    "dateFormat": date_format,
                    "date": historical_snapshot_time,
                },
            )
        else:
            payload = self.get(
                path=f"/sports/{sport_key}/events/{event_id}/odds",
                params={
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": odds_format,
                    "dateFormat": date_format,
                },
            )

        if not isinstance(payload, dict):
            raise OddsApiError(f"Unexpected event odds payload type: {type(payload).__name__}")
        return payload


def flatten_odds_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events")
    if not isinstance(events, list):
        return []

    game_date = str(payload.get("game_date", ""))
    rows: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, Mapping):
            continue
        home_team = str(event.get("home_team") or "")
        away_team = str(event.get("away_team") or "")
        if not home_team or not away_team:
            continue

        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            bookmakers = []

        ml_home_values: list[float | None] = []
        ml_away_values: list[float | None] = []
        spread_home_values: list[float | None] = []
        spread_away_values: list[float | None] = []
        total_values: list[float | None] = []
        over_price_values: list[float | None] = []
        under_price_values: list[float | None] = []

        for bookmaker in bookmakers:
            if not isinstance(bookmaker, Mapping):
                continue
            markets = bookmaker.get("markets")
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, Mapping):
                    continue
                outcomes = market.get("outcomes")
                if not isinstance(outcomes, list):
                    continue
                outcomes = [x for x in outcomes if isinstance(x, Mapping)]
                market_key = str(market.get("key") or "").strip().lower()

                if market_key == "h2h":
                    home_outcome = _outcome_for_team(outcomes, home_team)
                    away_outcome = _outcome_for_team(outcomes, away_team)
                    ml_home_values.append(_to_float((home_outcome or {}).get("price")))
                    ml_away_values.append(_to_float((away_outcome or {}).get("price")))

                if market_key == "spreads":
                    home_outcome = _outcome_for_team(outcomes, home_team)
                    away_outcome = _outcome_for_team(outcomes, away_team)
                    spread_home_values.append(_to_float((home_outcome or {}).get("point")))
                    spread_away_values.append(_to_float((away_outcome or {}).get("point")))

                if market_key == "totals":
                    over_outcome = _outcome_for_name(outcomes, "Over")
                    under_outcome = _outcome_for_name(outcomes, "Under")
                    over_point = _to_float((over_outcome or {}).get("point"))
                    under_point = _to_float((under_outcome or {}).get("point"))
                    total_values.append(over_point if over_point is not None else under_point)
                    over_price_values.append(_to_float((over_outcome or {}).get("price")))
                    under_price_values.append(_to_float((under_outcome or {}).get("price")))

        row = {
            "game_date": game_date,
            "event_id": event.get("id"),
            "commence_time": event.get("commence_time"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers_count": len(bookmakers),
            "moneyline_home": _median(ml_home_values),
            "moneyline_away": _median(ml_away_values),
            "spread_home": _median(spread_home_values),
            "spread_away": _median(spread_away_values),
            "total_points": _median(total_values),
            "over_price": _median(over_price_values),
            "under_price": _median(under_price_values),
            "moneyline_samples": len([x for x in ml_home_values if x is not None]),
            "spread_samples": len([x for x in spread_home_values if x is not None]),
            "total_samples": len([x for x in total_values if x is not None]),
        }
        rows.append(row)

    return rows


def flatten_player_props_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    events = payload.get("events")
    if not isinstance(events, list):
        return []

    game_date = str(payload.get("game_date", ""))
    rows: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, Mapping):
            continue
        event_id = event.get("id")
        home_team = event.get("home_team")
        away_team = event.get("away_team")
        commence_time = event.get("commence_time")

        bookmakers = event.get("bookmakers")
        if not isinstance(bookmakers, list):
            continue

        for bookmaker in bookmakers:
            if not isinstance(bookmaker, Mapping):
                continue
            bookmaker_key = bookmaker.get("key")
            markets = bookmaker.get("markets")
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, Mapping):
                    continue
                market_key = str(market.get("key") or "").strip().lower()
                if not market_key.startswith("player_"):
                    continue

                outcomes = market.get("outcomes")
                if not isinstance(outcomes, list):
                    continue
                typed_outcomes = [x for x in outcomes if isinstance(x, Mapping)]

                by_player: dict[str, dict[str, Mapping[str, Any]]] = {}
                for outcome in typed_outcomes:
                    name = str(outcome.get("name") or "").strip()
                    outcome_type = name.lower()
                    player_name = str(outcome.get("description") or "").strip()
                    if not player_name and outcome_type not in ("over", "under"):
                        player_name = name
                    if not player_name:
                        continue
                    by_player.setdefault(player_name, {})
                    if outcome_type in ("over", "under"):
                        by_player[player_name][outcome_type] = outcome
                    else:
                        by_player[player_name]["single"] = outcome

                for player_name, player_outcomes in by_player.items():
                    over_outcome = player_outcomes.get("over")
                    under_outcome = player_outcomes.get("under")
                    single_outcome = player_outcomes.get("single")

                    line = _to_float((over_outcome or {}).get("point"))
                    if line is None:
                        line = _to_float((under_outcome or {}).get("point"))
                    if line is None:
                        line = _to_float((single_outcome or {}).get("point"))

                    over_price = _to_float((over_outcome or {}).get("price"))
                    under_price = _to_float((under_outcome or {}).get("price"))
                    if over_price is None and single_outcome is not None:
                        over_price = _to_float(single_outcome.get("price"))

                    rows.append(
                        {
                            "game_date": game_date,
                            "event_id": event_id,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker_key,
                            "market": market_key,
                            "player_name": player_name,
                            "line": line,
                            "over_price": over_price,
                            "under_price": under_price,
                        }
                    )

    return rows
