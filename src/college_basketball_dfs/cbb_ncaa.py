from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import requests


class NcaaApiClientError(RuntimeError):
    """Raised for NCAA API request failures."""


@dataclass
class NcaaApiClient:
    base_url: str
    timeout_seconds: int = 20
    max_retries: int = 3
    retry_backoff_seconds: float = 0.75

    def __post_init__(self) -> None:
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(self._url(path), params=params, timeout=self.timeout_seconds)
                if response.status_code >= 400:
                    detail = response.text[:500]
                    raise NcaaApiClientError(f"GET {path} failed ({response.status_code}): {detail}")
                return response.json()
            except (requests.RequestException, ValueError, NcaaApiClientError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)

        raise NcaaApiClientError(f"GET {path} failed after retries: {last_error}")

    def fetch_scoreboard(
        self,
        game_date: date,
        sport: str = "basketball-men",
        division: str = "d1",
        conference: str = "all-conf",
    ) -> dict[str, Any]:
        path = f"/scoreboard/{sport}/{division}/{game_date:%Y/%m/%d}/{conference}"
        payload = self.get(path)
        if not isinstance(payload, dict):
            raise NcaaApiClientError(f"Unexpected scoreboard payload type: {type(payload).__name__}")
        return payload

    def fetch_boxscore(self, game_id: str) -> dict[str, Any]:
        payload = self.get(f"/game/{game_id}/boxscore")
        if not isinstance(payload, dict):
            raise NcaaApiClientError(f"Unexpected boxscore payload type for game {game_id}: {type(payload).__name__}")
        return payload


GAME_ID_PATTERN = re.compile(r"/game/(\d+)")


def prior_day(reference_date: date | None = None) -> date:
    reference = reference_date or date.today()
    return reference - timedelta(days=1)


def parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    return date.fromisoformat(value)


def extract_game_id(game_entry: Mapping[str, Any]) -> str:
    nested_game = game_entry.get("game")
    if isinstance(nested_game, Mapping):
        direct_game_id = nested_game.get("gameID")
        if direct_game_id:
            return str(direct_game_id)

        contest_id = nested_game.get("contestId")
        if contest_id:
            return str(contest_id)

        game_url = nested_game.get("url")
        if isinstance(game_url, str):
            match = GAME_ID_PATTERN.search(game_url)
            if match:
                return match.group(1)

    direct = game_entry.get("gameID")
    if direct:
        return str(direct)

    raise ValueError(f"Unable to extract game ID from game payload: {game_entry}")


def fetch_games_with_boxscores(
    client: NcaaApiClient,
    game_date: date,
    sport: str = "basketball-men",
    division: str = "d1",
    conference: str = "all-conf",
) -> dict[str, Any]:
    scoreboard = client.fetch_scoreboard(
        game_date=game_date,
        sport=sport,
        division=division,
        conference=conference,
    )
    raw_games = scoreboard.get("games")
    if not isinstance(raw_games, list):
        raw_games = []

    games_with_boxscores: list[dict[str, Any]] = []
    for entry in raw_games:
        if not isinstance(entry, Mapping):
            continue
        game: Mapping[str, Any] = entry.get("game") if isinstance(entry.get("game"), Mapping) else {}
        try:
            game_id = extract_game_id(entry)
        except ValueError as exc:
            games_with_boxscores.append(
                {
                    "game_id": None,
                    "status": game.get("gameState"),
                    "home_team": _nested_name(game, "home"),
                    "away_team": _nested_name(game, "away"),
                    "error": str(exc),
                    "boxscore": None,
                }
            )
            continue

        boxscore: dict[str, Any] | None
        error: str | None = None
        try:
            boxscore = client.fetch_boxscore(game_id)
        except NcaaApiClientError as exc:
            boxscore = None
            error = str(exc)

        games_with_boxscores.append(
            {
                "game_id": game_id,
                "status": game.get("gameState"),
                "start_date": game.get("startDate"),
                "start_time": game.get("startTime"),
                "home_team": _nested_name(game, "home"),
                "away_team": _nested_name(game, "away"),
                "home_score": _nested_score(game, "home"),
                "away_score": _nested_score(game, "away"),
                "error": error,
                "boxscore": boxscore,
            }
        )

    success_count = sum(1 for game in games_with_boxscores if game["boxscore"] is not None)
    failure_count = sum(1 for game in games_with_boxscores if game["boxscore"] is None)
    return {
        "game_date": game_date.isoformat(),
        "sport": sport,
        "division": division,
        "conference": conference,
        "game_count": len(games_with_boxscores),
        "boxscore_success_count": success_count,
        "boxscore_failure_count": failure_count,
        "games": games_with_boxscores,
    }


def _nested_name(game: Mapping[str, Any], side: str) -> str | None:
    side_data = game.get(side)
    if not isinstance(side_data, Mapping):
        return None
    names = side_data.get("names")
    if not isinstance(names, Mapping):
        return None
    short_name = names.get("short")
    return str(short_name) if short_name else None


def _nested_score(game: Mapping[str, Any], side: str) -> str | None:
    side_data = game.get(side)
    if not isinstance(side_data, Mapping):
        return None
    score = side_data.get("score")
    return str(score) if score not in (None, "") else None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch prior-day NCAA basketball games and box scores from the ncaa-api service."
    )
    parser.add_argument("--date", type=str, default=None, help="Date in YYYY-MM-DD. Defaults to prior day.")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.getenv("NCAA_API_BASE_URL", "https://ncaa-api.henrygd.me"),
        help="ncaa-api base URL.",
    )
    parser.add_argument("--sport", type=str, default="basketball-men", help="Sport path segment.")
    parser.add_argument("--division", type=str, default="d1", help="Division path segment.")
    parser.add_argument("--conference", type=str, default="all-conf", help="Conference path segment.")
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional output file path. If omitted, output is printed to stdout.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary only (no full JSON payload).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    requested_date = parse_iso_date(args.date)
    game_date = requested_date or prior_day()

    client = NcaaApiClient(base_url=args.base_url)
    try:
        result = fetch_games_with_boxscores(
            client=client,
            game_date=game_date,
            sport=args.sport,
            division=args.division,
            conference=args.conference,
        )
    finally:
        client.close()

    summary = (
        f"{result['game_date']}: games={result['game_count']}, "
        f"boxscores_ok={result['boxscore_success_count']}, "
        f"boxscores_failed={result['boxscore_failure_count']}"
    )
    print(summary)

    if args.summary_only:
        return

    payload = json.dumps(result, indent=2)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        print(f"Wrote payload to {path}")
        return

    print(payload)


if __name__ == "__main__":
    main()
