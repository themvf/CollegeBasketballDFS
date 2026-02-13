from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Mapping


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace("%", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int | None:
    as_float = _to_float(value)
    if as_float is None:
        return None
    return int(as_float)


def _build_player_name(first_name: Any, last_name: Any) -> str:
    first = str(first_name or "").strip()
    last = str(last_name or "").strip()
    return f"{first} {last}".strip()


def _calc_dk_fpts(player: Mapping[str, Any]) -> float:
    points = _to_float(player.get("points")) or 0.0
    rebounds = _to_float(player.get("totalRebounds")) or 0.0
    assists = _to_float(player.get("assists")) or 0.0
    steals = _to_float(player.get("steals")) or 0.0
    blocks = _to_float(player.get("blockedShots")) or 0.0
    turnovers = _to_float(player.get("turnovers")) or 0.0
    threes = _to_float(player.get("threePointsMade")) or 0.0
    return (
        points
        + (1.25 * rebounds)
        + (1.5 * assists)
        + (2.0 * steals)
        + (2.0 * blocks)
        - (0.5 * turnovers)
        + (0.5 * threes)
    )


def _calc_fd_fpts(player: Mapping[str, Any]) -> float:
    points = _to_float(player.get("points")) or 0.0
    rebounds = _to_float(player.get("totalRebounds")) or 0.0
    assists = _to_float(player.get("assists")) or 0.0
    steals = _to_float(player.get("steals")) or 0.0
    blocks = _to_float(player.get("blockedShots")) or 0.0
    turnovers = _to_float(player.get("turnovers")) or 0.0
    return (
        points
        + (1.2 * rebounds)
        + (1.5 * assists)
        + (3.0 * steals)
        + (3.0 * blocks)
        - turnovers
    )


def _team_label(is_home: bool | None, home_team: Any, away_team: Any) -> str | None:
    if is_home is True:
        return str(home_team) if home_team else None
    if is_home is False:
        return str(away_team) if away_team else None
    return None


def _opponent_label(is_home: bool | None, home_team: Any, away_team: Any) -> str | None:
    if is_home is True:
        return str(away_team) if away_team else None
    if is_home is False:
        return str(home_team) if home_team else None
    return None


def flatten_games_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    games = payload.get("games")
    if not isinstance(games, list):
        return []

    rows: list[dict[str, Any]] = []
    game_date = payload.get("game_date")

    for game_item in games:
        if not isinstance(game_item, Mapping):
            continue

        boxscore = game_item.get("boxscore")
        if not isinstance(boxscore, Mapping):
            continue

        home_team = game_item.get("home_team")
        away_team = game_item.get("away_team")
        teams_meta = boxscore.get("teams")
        team_meta_by_id: dict[str, Mapping[str, Any]] = {}
        if isinstance(teams_meta, list):
            for team in teams_meta:
                if isinstance(team, Mapping) and team.get("teamId") is not None:
                    team_meta_by_id[str(team["teamId"])] = team

        team_boxscore = boxscore.get("teamBoxscore")
        if not isinstance(team_boxscore, list):
            continue

        for team_entry in team_boxscore:
            if not isinstance(team_entry, Mapping):
                continue

            team_id = str(team_entry.get("teamId")) if team_entry.get("teamId") is not None else None
            team_meta = team_meta_by_id.get(team_id or "", {})
            is_home = team_meta.get("isHome") if isinstance(team_meta, Mapping) else None

            team_name = None
            if isinstance(team_meta, Mapping):
                team_name = team_meta.get("nameShort") or team_meta.get("teamName")
            if not team_name:
                team_name = _team_label(is_home, home_team, away_team)

            opponent = _opponent_label(is_home, home_team, away_team)

            team_stats = team_entry.get("teamStats") if isinstance(team_entry.get("teamStats"), Mapping) else {}
            team_points = _to_int(team_stats.get("points"))
            team_rebounds = _to_int(team_stats.get("totalRebounds"))
            team_assists = _to_int(team_stats.get("assists"))
            team_turnovers = _to_int(team_stats.get("turnovers"))

            player_stats = team_entry.get("playerStats")
            if not isinstance(player_stats, list):
                continue

            for player in player_stats:
                if not isinstance(player, Mapping):
                    continue

                row = {
                    "game_date": game_date,
                    "game_id": game_item.get("game_id"),
                    "game_status": game_item.get("status"),
                    "start_date": game_item.get("start_date"),
                    "start_time": game_item.get("start_time"),
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": _to_int(game_item.get("home_score")),
                    "away_score": _to_int(game_item.get("away_score")),
                    "team_id": team_id,
                    "team_name": team_name,
                    "opponent_team": opponent,
                    "is_home": is_home,
                    "player_id": _to_int(player.get("id")),
                    "player_number": _to_int(player.get("number")),
                    "player_first_name": player.get("firstName"),
                    "player_last_name": player.get("lastName"),
                    "player_name": _build_player_name(player.get("firstName"), player.get("lastName")),
                    "position": player.get("position"),
                    "starter": player.get("starter"),
                    "minutes_played": _to_float(player.get("minutesPlayed")),
                    "points": _to_int(player.get("points")),
                    "rebounds": _to_int(player.get("totalRebounds")),
                    "assists": _to_int(player.get("assists")),
                    "steals": _to_int(player.get("steals")),
                    "blocks": _to_int(player.get("blockedShots")),
                    "turnovers": _to_int(player.get("turnovers")),
                    "fouls": _to_int(player.get("personalFouls")),
                    "fgm": _to_int(player.get("fieldGoalsMade")),
                    "fga": _to_int(player.get("fieldGoalsAttempted")),
                    "ftm": _to_int(player.get("freeThrowsMade")),
                    "fta": _to_int(player.get("freeThrowsAttempted")),
                    "tpm": _to_int(player.get("threePointsMade")),
                    "tpa": _to_int(player.get("threePointsAttempted")),
                    "oreb": _to_int(player.get("offensiveRebounds")),
                    "dk_fpts": round(_calc_dk_fpts(player), 2),
                    "fd_fpts": round(_calc_fd_fpts(player), 2),
                    "team_points": team_points,
                    "team_rebounds": team_rebounds,
                    "team_assists": team_assists,
                    "team_turnovers": team_turnovers,
                }
                rows.append(row)

    return rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rows_to_csv_text(rows), encoding="utf-8")


def rows_to_csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Flatten ncaa-api prior-day games JSON to player-level CSV rows."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="data/cbb_prior_day.json",
        help="Path to input JSON from college_basketball_dfs.cbb_ncaa.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/cbb_prior_day_players.csv",
        help="Path to output player-level CSV.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_csv)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    rows = flatten_games_payload(payload)
    write_csv(rows, output_path)
    print(f"games={payload.get('game_count', 0)}, player_rows={len(rows)}, output={output_path}")


if __name__ == "__main__":
    main()
