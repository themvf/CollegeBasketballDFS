from __future__ import annotations

from typing import Any, Mapping


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def _result_label(team_score: int | None, opp_score: int | None) -> str:
    if team_score is None or opp_score is None:
        return "N/A"
    if team_score > opp_score:
        return "W"
    if team_score < opp_score:
        return "L"
    return "T"


def _team_row(
    game_date: str,
    venue: str | None,
    home_away: str,
    team: str | None,
    team_score: int | None,
    opponent: str | None,
    opponent_score: int | None,
) -> dict[str, Any]:
    return {
        "Team": team,
        "Game Date": game_date,
        "Venue": venue,
        "Home/Away": home_away,
        "Team Score": team_score,
        "Opponent": opponent,
        "Opponent Score": opponent_score,
        "W/L": _result_label(team_score, opponent_score),
    }


def rows_from_raw_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    game_date = str(payload.get("game_date", ""))
    games = payload.get("games")
    if not isinstance(games, list):
        return []

    rows: list[dict[str, Any]] = []
    for game in games:
        if not isinstance(game, Mapping):
            continue

        home_team = game.get("home_team")
        away_team = game.get("away_team")
        home_score = _to_int(game.get("home_score"))
        away_score = _to_int(game.get("away_score"))
        venue = str(home_team) if home_team else None

        rows.append(
            _team_row(
                game_date=game_date,
                venue=venue,
                home_away="Home",
                team=str(home_team) if home_team else None,
                team_score=home_score,
                opponent=str(away_team) if away_team else None,
                opponent_score=away_score,
            )
        )
        rows.append(
            _team_row(
                game_date=game_date,
                venue=venue,
                home_away="Away",
                team=str(away_team) if away_team else None,
                team_score=away_score,
                opponent=str(home_team) if home_team else None,
                opponent_score=home_score,
            )
        )
    return rows


def rows_from_payloads(payloads: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(rows_from_raw_payload(payload))
    return rows
