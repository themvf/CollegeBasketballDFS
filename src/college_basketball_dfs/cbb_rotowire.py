from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests


class RotoWireClientError(RuntimeError):
    """Raised for RotoWire optimizer request failures."""


SITE_ID_BY_NAME = {
    "draftkings": 1,
    "dk": 1,
}


@dataclass
class RotoWireClient:
    base_url: str = "https://www.rotowire.com"
    timeout_seconds: int = 20
    max_retries: int = 3
    retry_backoff_seconds: float = 0.75
    cookie_header: str | None = None
    user_agent: str = "Mozilla/5.0 (compatible; CollegeBasketballDFS/1.0)"

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json, text/plain, */*",
                "User-Agent": self.user_agent,
                "Referer": f"{self.base_url.rstrip('/')}/daily/ncaab/optimizer.php",
            }
        )
        if self.cookie_header:
            self.session.headers["Cookie"] = self.cookie_header

    def close(self) -> None:
        self.session.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(self._url(path), params=dict(params or {}), timeout=self.timeout_seconds)
                if response.status_code >= 400:
                    detail = response.text[:500]
                    raise RotoWireClientError(f"GET {path} failed ({response.status_code}): {detail}")
                return response.json()
            except (requests.RequestException, ValueError, RotoWireClientError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        raise RotoWireClientError(f"GET {path} failed after retries: {last_error}")

    def fetch_slate_catalog(self, site_id: int = 1) -> dict[str, Any]:
        payload = self.get_json("/daily/ncaab/api/slate-list.php", params={"siteID": site_id})
        if not isinstance(payload, Mapping):
            raise RotoWireClientError(f"Unexpected slate catalog payload type: {type(payload).__name__}")
        slates = payload.get("slates")
        if not isinstance(slates, list):
            raise RotoWireClientError("Slate catalog payload missing list field `slates`.")
        return {"slates": slates, "games": payload.get("games")}

    def fetch_players(self, slate_id: int) -> list[dict[str, Any]]:
        payload = self.get_json("/daily/ncaab/api/players.php", params={"slateID": slate_id})
        if not isinstance(payload, list):
            raise RotoWireClientError(f"Unexpected players payload type: {type(payload).__name__}")
        return [row for row in payload if isinstance(row, Mapping)]


def parse_site_id(site: str | None, site_id: int | None) -> int:
    if site_id is not None:
        return int(site_id)
    if not site:
        return 1
    normalized = site.strip().lower()
    if normalized in SITE_ID_BY_NAME:
        return SITE_ID_BY_NAME[normalized]
    try:
        return int(normalized)
    except ValueError as exc:
        raise ValueError(f"Unsupported site value: {site}") from exc


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("%", "")
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _to_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        if not cleaned:
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None
    return None


def flatten_slates(catalog: Mapping[str, Any], site_id: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for slate in catalog.get("slates", []):
        if not isinstance(slate, Mapping):
            continue
        rows.append(
            {
                "site_id": site_id,
                "slate_id": _to_int(slate.get("slateID")),
                "contest_type": slate.get("contestType"),
                "slate_name": slate.get("slateName"),
                "salary_cap": _to_int(slate.get("salaryCap")),
                "start_datetime": slate.get("startDate"),
                "end_datetime": slate.get("endDate"),
                "slate_date": slate.get("startDateOnly"),
                "start_time": slate.get("timeOnly"),
                "default_slate": bool(slate.get("defaultSlate", False)),
                "game_count": len(slate.get("games", [])) if isinstance(slate.get("games"), list) else 0,
                "game_ids": ",".join(str(x) for x in slate.get("games", []) if x is not None),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "slate_id",
                "contest_type",
                "slate_name",
                "salary_cap",
                "start_datetime",
                "end_datetime",
                "slate_date",
                "start_time",
                "default_slate",
                "game_count",
                "game_ids",
            ]
        )
    return out.sort_values(["slate_date", "start_datetime", "contest_type", "slate_name"], kind="stable").reset_index(
        drop=True
    )


def select_slate(
    slates_df: pd.DataFrame,
    slate_id: int | None = None,
    slate_date: str | None = None,
    contest_type: str | None = None,
    slate_name: str | None = None,
    first_match: bool = False,
) -> pd.Series:
    if slates_df.empty:
        raise ValueError("No slates returned from RotoWire.")

    filtered = slates_df.copy()
    if slate_id is not None:
        filtered = filtered.loc[filtered["slate_id"] == int(slate_id)]
    if slate_date:
        filtered = filtered.loc[filtered["slate_date"].astype(str) == str(slate_date)]
    if contest_type:
        contest_norm = contest_type.strip().lower()
        filtered = filtered.loc[filtered["contest_type"].astype(str).str.lower() == contest_norm]
    if slate_name:
        name_norm = slate_name.strip().lower()
        filtered = filtered.loc[filtered["slate_name"].astype(str).str.lower() == name_norm]

    if filtered.empty:
        raise ValueError("No slate matched the provided filters.")
    if len(filtered) > 1 and not first_match:
        summary = filtered[["slate_id", "slate_date", "contest_type", "slate_name", "start_time"]].to_dict("records")
        raise ValueError(f"Multiple slates matched the provided filters: {summary}")
    return filtered.iloc[0]


def normalize_players(raw_players: list[Mapping[str, Any]], slate_row: Mapping[str, Any] | None = None) -> pd.DataFrame:
    slate_meta = slate_row or {}
    rows: list[dict[str, Any]] = []
    for player in raw_players:
        team = player.get("team") if isinstance(player.get("team"), Mapping) else {}
        opponent = player.get("opponent") if isinstance(player.get("opponent"), Mapping) else {}
        odds = player.get("odds") if isinstance(player.get("odds"), Mapping) else {}
        stats = player.get("stats") if isinstance(player.get("stats"), Mapping) else {}
        avg_fpts = stats.get("avgFpts") if isinstance(stats.get("avgFpts"), Mapping) else {}
        advanced = stats.get("advanced") if isinstance(stats.get("advanced"), Mapping) else {}
        game = player.get("game") if isinstance(player.get("game"), Mapping) else {}

        first_name = str(player.get("firstName") or "").strip()
        last_name = str(player.get("lastName") or "").strip()
        full_name = " ".join(part for part in [first_name, last_name] if part).strip()
        salary = _to_int(player.get("salary"))
        proj_fpts = _to_float(player.get("pts"))

        rows.append(
            {
                "site_id": slate_meta.get("site_id"),
                "slate_id": _to_int(player.get("slateID")) or slate_meta.get("slate_id"),
                "slate_date": slate_meta.get("slate_date"),
                "contest_type": slate_meta.get("contest_type"),
                "slate_name": slate_meta.get("slate_name"),
                "rw_id": _to_int(player.get("rwID")),
                "player_name": full_name or None,
                "first_name": first_name or None,
                "last_name": last_name or None,
                "roto_position": player.get("rotoPos"),
                "site_positions": "/".join(str(pos) for pos in player.get("pos", []) if pos) if isinstance(player.get("pos"), list) else None,
                "injury_status": player.get("injuryStatus"),
                "is_home": bool(player.get("isHome", False)),
                "team_abbr": team.get("abbr"),
                "team_name": team.get("city"),
                "team_nickname": team.get("nickname"),
                "opp_abbr": opponent.get("team"),
                "game_datetime": game.get("dateTime"),
                "salary": salary,
                "proj_fantasy_points": proj_fpts,
                "proj_minutes": _to_float(player.get("minutes")),
                "proj_value_per_1k": (proj_fpts / salary * 1000.0) if proj_fpts is not None and salary not in (None, 0) else None,
                "moneyline": _to_float(odds.get("moneyline")),
                "over_under": _to_float(odds.get("overUnder")),
                "spread": _to_float(odds.get("spread")),
                "implied_points": _to_float(odds.get("impliedPts")),
                "implied_win_prob": _to_float(odds.get("impliedWinProb")),
                "season_games": _to_int((stats.get("season") or {}).get("games") if isinstance(stats.get("season"), Mapping) else None),
                "season_minutes": _to_float((stats.get("season") or {}).get("minutes") if isinstance(stats.get("season"), Mapping) else None),
                "avg_fpts_last3": _to_float(avg_fpts.get("last3")),
                "avg_fpts_last5": _to_float(avg_fpts.get("last5")),
                "avg_fpts_last7": _to_float(avg_fpts.get("last7")),
                "avg_fpts_last14": _to_float(avg_fpts.get("last14")),
                "avg_fpts_season": _to_float(avg_fpts.get("season")),
                "usage_rate": _to_float(advanced.get("usage")),
                "player_link": player.get("link"),
                "team_link": player.get("teamLink"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "site_id",
                "slate_id",
                "slate_date",
                "contest_type",
                "slate_name",
                "rw_id",
                "player_name",
                "first_name",
                "last_name",
                "roto_position",
                "site_positions",
                "injury_status",
                "is_home",
                "team_abbr",
                "team_name",
                "team_nickname",
                "opp_abbr",
                "game_datetime",
                "salary",
                "proj_fantasy_points",
                "proj_minutes",
                "proj_value_per_1k",
                "moneyline",
                "over_under",
                "spread",
                "implied_points",
                "implied_win_prob",
                "season_games",
                "season_minutes",
                "avg_fpts_last3",
                "avg_fpts_last5",
                "avg_fpts_last7",
                "avg_fpts_last14",
                "avg_fpts_season",
                "usage_rate",
                "player_link",
                "team_link",
            ]
        )
    return out.sort_values(["proj_fantasy_points", "salary", "player_name"], ascending=[False, False, True], kind="stable").reset_index(drop=True)


def export_dataframe(df: pd.DataFrame, output_csv: str | None = None, output_json: str | None = None) -> None:
    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    if output_json:
        json_path = Path(output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch RotoWire College Basketball optimizer slates and player projections.")
    parser.add_argument("--site", type=str, default="draftkings", help="Site alias or numeric site ID. Default: draftkings")
    parser.add_argument("--site-id", type=int, default=None, help="Explicit numeric site ID. Overrides --site.")
    parser.add_argument("--cookie-header", type=str, default=os.getenv("ROTOWIRE_COOKIE"), help="Optional Cookie header for member-authenticated requests.")
    parser.add_argument("--list-slates", action="store_true", help="List slates and exit.")
    parser.add_argument("--slate-id", type=int, default=None, help="Specific slate ID to export.")
    parser.add_argument("--date", type=str, default=None, help="Filter slates by YYYY-MM-DD.")
    parser.add_argument("--contest-type", type=str, default=None, help="Filter slates by contest type, e.g. Classic or Showdown.")
    parser.add_argument("--slate-name", type=str, default=None, help="Filter slates by exact slate name, e.g. Night.")
    parser.add_argument("--first-match", action="store_true", help="Use the first matching slate when filters are ambiguous.")
    parser.add_argument("--csv-out", type=str, default=None, help="Write normalized player projections to CSV.")
    parser.add_argument("--json-out", type=str, default=None, help="Write normalized player projections to JSON.")
    parser.add_argument("--slates-csv-out", type=str, default=None, help="Write available slate metadata to CSV.")
    parser.add_argument("--summary-only", action="store_true", help="Print only summary rows instead of full player data.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    site_id = parse_site_id(args.site, args.site_id)
    client = RotoWireClient(cookie_header=args.cookie_header)
    try:
        catalog = client.fetch_slate_catalog(site_id=site_id)
        slates_df = flatten_slates(catalog, site_id=site_id)
        if args.slates_csv_out:
            export_dataframe(slates_df, output_csv=args.slates_csv_out)

        if args.list_slates:
            if slates_df.empty:
                print("No slates returned.")
            else:
                print(slates_df.to_string(index=False))
            return 0

        selected = select_slate(
            slates_df,
            slate_id=args.slate_id,
            slate_date=args.date,
            contest_type=args.contest_type,
            slate_name=args.slate_name,
            first_match=args.first_match,
        )
        raw_players = client.fetch_players(int(selected["slate_id"]))
        players_df = normalize_players(raw_players, slate_row=selected.to_dict())
        export_dataframe(players_df, output_csv=args.csv_out, output_json=args.json_out)

        if args.summary_only:
            summary = players_df[["player_name", "team_abbr", "salary", "proj_fantasy_points", "proj_minutes"]].head(20)
            print(summary.to_string(index=False))
        else:
            print(players_df.to_string(index=False))
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
