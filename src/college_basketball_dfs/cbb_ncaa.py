from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import requests


class NcaaApiClientError(RuntimeError):
    """Raised for NCAA API request failures."""


DIRECT_SCOREBOARD_HASH = "7287cda610a9326931931080cb3a604828febe6fe3c9016a7e4a36db99efdb7c"
DIRECT_BOXSCORE_HASHES = (
    "4a7fa26398db33de3ff51402a90eb5f25acef001cca28d239fe5361315d1419a",
    "5fcf84602d59c003f37ddd1185da542578080e04fe854e935cbcaee590a0e8a2",
)
DIRECT_SPORT_CODES = {
    "basketball-men": "MBB",
    "basketball-women": "WBB",
}
DIRECT_DIVISION_CODES = {
    "d1": 1,
    "d2": 2,
    "d3": 3,
}


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
        try:
            payload = self.get(path)
        except NcaaApiClientError:
            if not _supports_direct_scoreboard_fallback(sport=sport, division=division, conference=conference):
                raise
            payload = _fetch_direct_scoreboard(
                session=self.session,
                timeout_seconds=self.timeout_seconds,
                game_date=game_date,
                sport=sport,
                division=division,
            )
        if not isinstance(payload, dict):
            raise NcaaApiClientError(f"Unexpected scoreboard payload type: {type(payload).__name__}")
        return payload

    def fetch_boxscore(self, game_id: str) -> dict[str, Any]:
        try:
            payload = self.get(f"/game/{game_id}/boxscore")
        except NcaaApiClientError:
            payload = _fetch_direct_boxscore(
                session=self.session,
                timeout_seconds=self.timeout_seconds,
                game_id=game_id,
            )
        if not isinstance(payload, dict):
            raise NcaaApiClientError(f"Unexpected boxscore payload type for game {game_id}: {type(payload).__name__}")
        return payload


GAME_ID_PATTERN = re.compile(r"/game/(\d+)")


def _supports_direct_scoreboard_fallback(sport: str, division: str, conference: str) -> bool:
    return (
        sport in DIRECT_SPORT_CODES
        and division in DIRECT_DIVISION_CODES
        and str(conference or "").strip().lower() in {"", "all-conf"}
    )


def _season_year_for_date(game_date: date) -> int:
    return game_date.year - 1 if game_date.month < 8 else game_date.year


def _direct_request_json(
    session: requests.Session,
    timeout_seconds: int,
    *,
    params: dict[str, Any],
) -> Any:
    try:
        response = session.get("https://sdataprod.ncaa.com/", params=params, timeout=timeout_seconds)
    except requests.RequestException as exc:
        raise NcaaApiClientError(f"Direct NCAA request failed: {exc}") from exc
    if response.status_code >= 400:
        detail = response.text[:500]
        raise NcaaApiClientError(f"Direct NCAA request failed ({response.status_code}): {detail}")
    try:
        return response.json()
    except ValueError as exc:
        raise NcaaApiClientError("Direct NCAA request returned invalid JSON") from exc


def _normalize_direct_game_state(value: object) -> str:
    text = str(value or "").strip().upper()
    if text == "F":
        return "final"
    if text == "I":
        return "live"
    return "pre"


def _format_direct_start_time(start_date_raw: object, start_time_raw: object) -> str:
    start_time = str(start_time_raw or "").strip()
    if not start_time:
        return ""
    if start_time.upper() == "TBA":
        return "TBA"
    start_date = str(start_date_raw or "").strip()
    try:
        parsed = datetime.strptime(f"{start_date} {start_time}", "%m/%d/%Y %H:%M")
    except ValueError:
        return start_time
    return parsed.strftime("%I:%M %p").lstrip("0") + " ET"


def _build_direct_team_payload(team: Mapping[str, Any]) -> dict[str, Any]:
    score = team.get("score")
    seed = team.get("seed")
    rank = team.get("teamRank")
    conference_seo = str(team.get("conferenceSeo") or "").strip()
    return {
        "score": "" if score is None else str(score),
        "names": {
            "char6": str(team.get("name6Char") or ""),
            "short": str(team.get("nameShort") or ""),
            "seo": str(team.get("seoname") or ""),
            "full": str(team.get("nameShort") or ""),
        },
        "winner": bool(team.get("isWinner")),
        "seed": "" if seed in (None, "") else str(seed),
        "description": "",
        "rank": "" if rank in (None, "") else str(rank),
        "conferences": [
            {
                "conferenceName": "",
                "conferenceSeo": conference_seo,
            }
        ],
    }


def _fetch_direct_scoreboard(
    session: requests.Session,
    timeout_seconds: int,
    *,
    game_date: date,
    sport: str,
    division: str,
) -> dict[str, Any]:
    sport_code = DIRECT_SPORT_CODES.get(sport)
    division_code = DIRECT_DIVISION_CODES.get(division)
    if not sport_code or division_code is None:
        raise NcaaApiClientError(f"Direct NCAA scoreboard fallback is not supported for {sport} {division}.")

    payload = _direct_request_json(
        session,
        timeout_seconds,
        params={
            "extensions": json.dumps(
                {
                    "persistedQuery": {
                        "version": 1,
                        "sha256Hash": DIRECT_SCOREBOARD_HASH,
                    }
                },
                separators=(",", ":"),
            ),
            "variables": json.dumps(
                {
                    "sportCode": sport_code,
                    "division": division_code,
                    "seasonYear": _season_year_for_date(game_date),
                    "contestDate": game_date.strftime("%Y/%m/%d"),
                },
                separators=(",", ":"),
            ),
        },
    )

    contests = payload.get("data", {}).get("contests")
    if not isinstance(contests, list):
        raise NcaaApiClientError("Direct NCAA scoreboard fallback payload missing `data.contests`.")

    games: list[dict[str, Any]] = []
    for contest in contests:
        if not isinstance(contest, Mapping):
            continue
        teams = contest.get("teams")
        if not isinstance(teams, list):
            continue
        home_team = next((team for team in teams if isinstance(team, Mapping) and bool(team.get("isHome"))), None)
        away_team = next((team for team in teams if isinstance(team, Mapping) and not bool(team.get("isHome"))), None)
        if not isinstance(home_team, Mapping) or not isinstance(away_team, Mapping):
            continue
        start_date_raw = contest.get("startDate")
        start_time_raw = contest.get("startTime")
        games.append(
            {
                "game": {
                    "gameID": str(contest.get("contestId") or ""),
                    "away": _build_direct_team_payload(away_team),
                    "finalMessage": str(contest.get("finalMessage") or ""),
                    "bracketRound": "",
                    "title": f"{away_team.get('nameShort') or ''} {home_team.get('nameShort') or ''}".strip(),
                    "contestName": "",
                    "url": str(contest.get("url") or ""),
                    "network": str(contest.get("broadcasterName") or ""),
                    "home": _build_direct_team_payload(home_team),
                    "liveVideoEnabled": bool(contest.get("liveVideos")),
                    "startTime": _format_direct_start_time(start_date_raw, start_time_raw),
                    "startTimeEpoch": "" if contest.get("startTimeEpoch") is None else str(contest.get("startTimeEpoch")),
                    "bracketId": "",
                    "gameState": _normalize_direct_game_state(contest.get("gameState")),
                    "startDate": str(start_date_raw or ""),
                    "currentPeriod": str(contest.get("currentPeriod") or ""),
                    "videoState": "",
                    "bracketRegion": "",
                    "contestClock": str(contest.get("contestClock") or ""),
                }
            }
        )

    return {"games": games}


def _fetch_direct_boxscore(
    session: requests.Session,
    timeout_seconds: int,
    *,
    game_id: str,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for hash_value in DIRECT_BOXSCORE_HASHES:
        try:
            payload = _direct_request_json(
                session,
                timeout_seconds,
                params={
                    "extensions": json.dumps(
                        {
                            "persistedQuery": {
                                "version": 1,
                                "sha256Hash": hash_value,
                            }
                        },
                        separators=(",", ":"),
                    ),
                    "variables": json.dumps(
                        {
                            "contestId": str(game_id),
                            "staticTestEnv": None,
                        },
                        separators=(",", ":"),
                    ),
                },
            )
        except NcaaApiClientError as exc:
            last_error = exc
            continue
        boxscore = payload.get("data", {}).get("boxscore")
        if isinstance(boxscore, Mapping):
            return dict(boxscore)
        last_error = NcaaApiClientError("Direct NCAA boxscore fallback payload missing `data.boxscore`.")
    if last_error is not None:
        raise NcaaApiClientError(f"Direct NCAA boxscore fallback failed for game {game_id}: {last_error}") from last_error
    raise NcaaApiClientError(f"Direct NCAA boxscore fallback failed for game {game_id}.")


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
