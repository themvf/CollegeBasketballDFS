from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


class OddsApiError(RuntimeError):
    """Raised for The Odds API failures."""


try:
    # Slate date semantics are ET-local in DK workflow; convert ET bounds to UTC for API filters.
    _SLATE_LOCAL_TZ = ZoneInfo("America/New_York")
except ZoneInfoNotFoundError:
    _SLATE_LOCAL_TZ = timezone.utc


def _day_window_utc(game_date: date) -> tuple[str, str]:
    start_local = datetime(game_date.year, game_date.month, game_date.day, tzinfo=_SLATE_LOCAL_TZ)
    end_local = start_local + timedelta(days=1)
    start_utc = start_local.astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)
    return start_utc.isoformat().replace("+00:00", "Z"), end_utc.isoformat().replace("+00:00", "Z")


def _historical_snapshot_default(game_date: date) -> str:
    # Use end-of-local-day (ET) snapshot, expressed in UTC.
    _, end_utc_iso = _day_window_utc(game_date)
    end_utc = datetime.fromisoformat(end_utc_iso.replace("Z", "+00:00"))
    snapshot = end_utc - timedelta(seconds=1)
    return snapshot.replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
    max_retries: int = 5
    retry_backoff_seconds: float = 1.0
    max_retry_backoff_seconds: float = 16.0
    min_interval_seconds: float = 0.35

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self._last_request_monotonic = 0.0

    def close(self) -> None:
        self.session.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def _enforce_min_interval(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_request_monotonic
        wait_seconds = self.min_interval_seconds - elapsed
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _mark_request(self) -> None:
        self._last_request_monotonic = time.monotonic()

    def _retry_delay_seconds(self, attempt: int, retry_after_raw: str | None = None) -> float:
        backoff = min(self.max_retry_backoff_seconds, self.retry_backoff_seconds * (2**attempt))
        if not retry_after_raw:
            return backoff
        try:
            retry_after = float(retry_after_raw)
        except (TypeError, ValueError):
            retry_after = 0.0
        return max(backoff, retry_after)

    def get(self, path: str, params: dict[str, Any]) -> Any:
        full_params = {"apiKey": self.api_key, **params}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            self._enforce_min_interval()
            try:
                response = self.session.get(self._url(path), params=full_params, timeout=self.timeout_seconds)
                self._mark_request()
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise OddsApiError(f"GET {path} failed after retries: {exc}") from exc
                time.sleep(self._retry_delay_seconds(attempt))
                continue

            if response.status_code == 429:
                detail = response.text[:500]
                if attempt >= self.max_retries:
                    raise OddsApiError(f"GET {path} failed ({response.status_code}): {detail}")
                delay = self._retry_delay_seconds(attempt, response.headers.get("Retry-After"))
                time.sleep(delay)
                continue

            if response.status_code >= 500:
                detail = response.text[:500]
                if attempt >= self.max_retries:
                    raise OddsApiError(f"GET {path} failed ({response.status_code}): {detail}")
                time.sleep(self._retry_delay_seconds(attempt))
                continue

            if response.status_code >= 400:
                detail = response.text[:500]
                raise OddsApiError(f"GET {path} failed ({response.status_code}): {detail}")

            try:
                return response.json()
            except ValueError as exc:
                raise OddsApiError(f"GET {path} returned invalid JSON") from exc

        if last_error is not None:
            raise OddsApiError(f"GET {path} failed after retries: {last_error}") from last_error
        raise OddsApiError(f"GET {path} failed unexpectedly after retries.")

    def fetch_game_odds(
        self,
        game_date: date,
        sport_key: str = "basketball_ncaab",
        regions: str = "us",
        markets: str = "h2h,spreads,totals",
        bookmakers: str | None = None,
        odds_format: str = "american",
        date_format: str = "iso",
        historical: bool = False,
        historical_snapshot_time: str | None = None,
    ) -> list[dict[str, Any]]:
        commence_from, commence_to = _day_window_utc(game_date)
        common_params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            common_params["bookmakers"] = bookmakers

        if historical:
            snapshot_time = historical_snapshot_time or _historical_snapshot_default(game_date)
            payload = self.get(
                path=f"/historical/sports/{sport_key}/odds",
                params={
                    **common_params,
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
                **common_params,
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
        bookmakers: str | None = None,
        odds_format: str = "american",
        date_format: str = "iso",
        historical: bool = False,
        historical_snapshot_time: str | None = None,
    ) -> dict[str, Any]:
        common_params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        }
        if bookmakers:
            common_params["bookmakers"] = bookmakers

        if historical:
            if not historical_snapshot_time:
                raise OddsApiError("historical_snapshot_time is required for historical event odds.")
            payload = self.get(
                path=f"/historical/sports/{sport_key}/events/{event_id}/odds",
                params={
                    **common_params,
                    "date": historical_snapshot_time,
                },
            )
            if not isinstance(payload, dict):
                raise OddsApiError(f"Unexpected historical event odds payload type: {type(payload).__name__}")
            data = payload.get("data")
            if isinstance(data, dict):
                return data
            raise OddsApiError("Historical event odds payload missing object field `data`.")
        else:
            payload = self.get(
                path=f"/sports/{sport_key}/events/{event_id}/odds",
                params=common_params,
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
        event_body: Mapping[str, Any] = event
        wrapped_data = event.get("data")
        if isinstance(wrapped_data, Mapping):
            event_body = wrapped_data

        event_id = event_body.get("id") or event.get("id")
        home_team = event_body.get("home_team") or event.get("home_team")
        away_team = event_body.get("away_team") or event.get("away_team")
        commence_time = event_body.get("commence_time") or event.get("commence_time")

        bookmakers = event_body.get("bookmakers")
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
