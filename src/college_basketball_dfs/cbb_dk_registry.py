from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


REGISTRY_HISTORY_COLUMNS = [
    "dk_id",
    "name_plus_id",
    "player_name",
    "player_name_norm",
    "team_abbr",
    "team_norm",
    "position",
    "position_base",
    "roster_position",
    "salary",
    "opp_abbr",
    "game_key",
    "slate_date",
    "slate_key",
    "source_name",
]

REGISTRY_COLUMNS = [
    "dk_id",
    "name_plus_id",
    "player_name",
    "player_name_norm",
    "team_abbr",
    "team_norm",
    "position",
    "position_base",
    "roster_position",
    "latest_salary",
    "salary_min",
    "salary_max",
    "salary_median",
    "last_opp_abbr",
    "last_game_key",
    "first_seen_date",
    "last_seen_date",
    "seen_count",
    "source_name_last",
]

RESOLUTION_COLUMNS = [
    "player_name",
    "team_abbr",
    "opp_abbr",
    "salary",
    "position",
    "dk_resolution_status",
    "dk_id",
    "name_plus_id",
    "dk_match_confidence",
    "dk_match_reason",
    "candidate_count",
    "matched_team_abbr",
    "matched_salary",
    "matched_last_seen_date",
]


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip().replace("$", "").replace(",", "")
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    if number is None:
        return None
    return int(round(number))


def _position_base(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    if raw.startswith("G"):
        return "G"
    if raw.startswith("F"):
        return "F"
    if raw.startswith("C"):
        return "C"
    return raw[:1]


def _extract_game_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(" ")[0].upper()


def _extract_opp_abbr(game_info: Any, team_abbr: Any) -> str:
    game_key = _extract_game_key(game_info)
    team = str(team_abbr or "").strip().upper()
    if "@" not in game_key or not team:
        return ""
    away, home = (part.strip().upper() for part in game_key.split("@", 1))
    if team == away:
        return home
    if team == home:
        return away
    return ""


def _empty_history_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REGISTRY_HISTORY_COLUMNS)


def _empty_registry_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REGISTRY_COLUMNS)


def _empty_resolution_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=RESOLUTION_COLUMNS)


def extract_registry_rows_from_dk_slate(
    slate_df: pd.DataFrame,
    slate_date: date | str | None = None,
    slate_key: str | None = None,
    source_name: str | None = None,
) -> pd.DataFrame:
    if slate_df is None or slate_df.empty:
        return _empty_history_frame()

    work = slate_df.copy()
    empty_series = pd.Series([""] * len(work), index=work.index, dtype="object")
    work["Name"] = work.get("Name", empty_series).astype(str).str.strip()
    work["ID"] = work.get("ID", empty_series).astype(str).str.strip()
    work["Name + ID"] = work.get("Name + ID", empty_series).astype(str).str.strip()
    work["TeamAbbrev"] = work.get("TeamAbbrev", empty_series).astype(str).str.strip().str.upper()
    work["Position"] = work.get("Position", empty_series).astype(str).str.strip().str.upper()
    work["Roster Position"] = work.get("Roster Position", empty_series).astype(str).str.strip().str.upper()
    work["Salary"] = pd.to_numeric(work.get("Salary"), errors="coerce")
    work["Game Info"] = work.get("Game Info", empty_series).astype(str).str.strip()
    work = work.loc[(work["Name"] != "") & (work["ID"] != "")]
    if work.empty:
        return _empty_history_frame()

    slate_ts = pd.to_datetime(slate_date, errors="coerce")
    work["dk_id"] = work["ID"]
    work["name_plus_id"] = work["Name + ID"].where(work["Name + ID"] != "", work["Name"] + " (" + work["ID"] + ")")
    work["player_name"] = work["Name"]
    work["player_name_norm"] = work["Name"].map(_normalize_text)
    work["team_abbr"] = work["TeamAbbrev"]
    work["team_norm"] = work["TeamAbbrev"].map(_normalize_text)
    work["position"] = work["Position"]
    work["position_base"] = work["Position"].map(_position_base)
    work["roster_position"] = work["Roster Position"]
    work["salary"] = work["Salary"]
    work["game_key"] = work["Game Info"].map(_extract_game_key)
    work["opp_abbr"] = [
        _extract_opp_abbr(game_info, team_abbr)
        for game_info, team_abbr in zip(work["Game Info"].tolist(), work["TeamAbbrev"].tolist())
    ]
    work["slate_date"] = slate_ts
    work["slate_key"] = str(slate_key or "").strip().lower()
    work["source_name"] = str(source_name or "").strip()
    return work[REGISTRY_HISTORY_COLUMNS].reset_index(drop=True)


def build_dk_identity_registry(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df is None or history_df.empty:
        return _empty_registry_frame()

    work = history_df.copy()
    empty_series = pd.Series([""] * len(work), index=work.index, dtype="object")
    work["dk_id"] = work.get("dk_id", empty_series).astype(str).str.strip()
    work["player_name"] = work.get("player_name", empty_series).astype(str).str.strip()
    work["player_name_norm"] = work.get("player_name_norm", empty_series).astype(str).str.strip()
    work["team_abbr"] = work.get("team_abbr", empty_series).astype(str).str.strip().str.upper()
    work["team_norm"] = work.get("team_norm", empty_series).astype(str).str.strip()
    work["position"] = work.get("position", empty_series).astype(str).str.strip().str.upper()
    work["position_base"] = work.get("position_base", empty_series).astype(str).str.strip().str.upper()
    work["roster_position"] = work.get("roster_position", empty_series).astype(str).str.strip().str.upper()
    work["salary"] = pd.to_numeric(work.get("salary"), errors="coerce")
    work["slate_date"] = pd.to_datetime(work.get("slate_date"), errors="coerce")
    work["seen_order"] = range(len(work))
    work = work.loc[(work["dk_id"] != "") & (work["player_name_norm"] != "")]
    if work.empty:
        return _empty_registry_frame()

    rows: list[dict[str, Any]] = []
    for dk_id, one in work.groupby("dk_id", sort=False):
        ordered = one.sort_values(["slate_date", "seen_order"], ascending=[True, True], kind="stable")
        latest = ordered.iloc[-1]
        salary_values = pd.to_numeric(ordered["salary"], errors="coerce").dropna()
        rows.append(
            {
                "dk_id": str(dk_id),
                "name_plus_id": str(latest.get("name_plus_id") or ""),
                "player_name": str(latest.get("player_name") or ""),
                "player_name_norm": str(latest.get("player_name_norm") or ""),
                "team_abbr": str(latest.get("team_abbr") or "").upper(),
                "team_norm": str(latest.get("team_norm") or ""),
                "position": str(latest.get("position") or "").upper(),
                "position_base": str(latest.get("position_base") or "").upper(),
                "roster_position": str(latest.get("roster_position") or "").upper(),
                "latest_salary": float(salary_values.iloc[-1]) if not salary_values.empty else pd.NA,
                "salary_min": float(salary_values.min()) if not salary_values.empty else pd.NA,
                "salary_max": float(salary_values.max()) if not salary_values.empty else pd.NA,
                "salary_median": float(salary_values.median()) if not salary_values.empty else pd.NA,
                "last_opp_abbr": str(latest.get("opp_abbr") or "").upper(),
                "last_game_key": str(latest.get("game_key") or "").upper(),
                "first_seen_date": ordered["slate_date"].min(),
                "last_seen_date": ordered["slate_date"].max(),
                "seen_count": int(len(ordered)),
                "source_name_last": str(latest.get("source_name") or ""),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return _empty_registry_frame()
    out["last_seen_date"] = pd.to_datetime(out.get("last_seen_date"), errors="coerce")
    out["first_seen_date"] = pd.to_datetime(out.get("first_seen_date"), errors="coerce")
    return out.sort_values(
        ["player_name", "team_abbr", "last_seen_date", "seen_count"],
        ascending=[True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)


def build_registry_history_from_local_directory(directory: str | Path) -> pd.DataFrame:
    base = Path(directory)
    if not base.exists() or not base.is_dir():
        return _empty_history_frame()

    frames: list[pd.DataFrame] = []
    for csv_path in sorted(base.glob("DKSalaries*.csv")):
        try:
            slate_df = pd.read_csv(csv_path)
        except Exception:
            continue
        metadata = parse_local_dk_slate_filename(csv_path.name)
        frames.append(
            extract_registry_rows_from_dk_slate(
                slate_df,
                slate_date=metadata.get("slate_date"),
                slate_key=metadata.get("slate_key"),
                source_name=str(csv_path),
            )
        )
    if not frames:
        return _empty_history_frame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return _empty_history_frame()
    return combined.drop_duplicates(
        subset=["dk_id", "team_abbr", "salary", "slate_date", "source_name"],
        keep="last",
    ).reset_index(drop=True)


def parse_local_dk_slate_filename(filename: str) -> dict[str, Any]:
    text = str(filename or "").strip()
    match = re.search(r"(\d{1,2})_(\d{1,2})_(\d{4})(?:_([A-Za-z0-9-]+))?", text)
    if not match:
        return {"slate_date": pd.NaT, "slate_key": ""}
    month = int(match.group(1))
    day = int(match.group(2))
    year = int(match.group(3))
    slate_key = str(match.group(4) or "").strip().lower()
    try:
        slate_ts = pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        slate_ts = pd.NaT
    return {"slate_date": slate_ts, "slate_key": slate_key}


def build_rotowire_dk_slate(
    rotowire_df: pd.DataFrame,
    registry_df: pd.DataFrame,
    slate_date: date | str | None = None,
    slate_key: str | None = None,
    min_confidence_score: float = 55.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if rotowire_df is None or rotowire_df.empty:
        meta = {
            "source": "rotowire_registry",
            "players_total": 0,
            "resolved_players": 0,
            "unresolved_players": 0,
            "conflict_players": 0,
            "coverage_pct": 0.0,
            "fully_resolved": False,
        }
        return pd.DataFrame(), _empty_resolution_frame(), meta

    registry = registry_df.copy() if registry_df is not None else _empty_registry_frame()
    if registry.empty:
        meta = {
            "source": "rotowire_registry",
            "players_total": int(len(rotowire_df)),
            "resolved_players": 0,
            "unresolved_players": int(len(rotowire_df)),
            "conflict_players": 0,
            "coverage_pct": 0.0,
            "fully_resolved": False,
            "failure_reason": "registry_empty",
        }
        return pd.DataFrame(), _empty_resolution_frame(), meta

    slate_ts = pd.to_datetime(slate_date, errors="coerce")
    rotowire = rotowire_df.copy()
    empty_series = pd.Series([""] * len(rotowire), index=rotowire.index, dtype="object")
    rotowire["player_name"] = rotowire.get("player_name", empty_series).astype(str).str.strip()
    rotowire["team_abbr"] = rotowire.get("team_abbr", empty_series).astype(str).str.strip().str.upper()
    rotowire["opp_abbr"] = rotowire.get("opp_abbr", empty_series).astype(str).str.strip().str.upper()
    rotowire["salary"] = pd.to_numeric(rotowire.get("salary"), errors="coerce")
    rotowire["site_positions"] = rotowire.get("site_positions", empty_series).astype(str).str.strip().str.upper()
    rotowire["roto_position"] = rotowire.get("roto_position", empty_series).astype(str).str.strip().str.upper()
    rotowire["player_name_norm"] = rotowire["player_name"].map(_normalize_text)
    registry_empty_series = pd.Series([""] * len(registry), index=registry.index, dtype="object")
    registry["dk_id"] = registry.get("dk_id", registry_empty_series).astype(str).str.strip()
    registry["player_name_norm"] = registry.get("player_name_norm", registry_empty_series).astype(str).str.strip()
    registry["team_abbr"] = registry.get("team_abbr", registry_empty_series).astype(str).str.strip().str.upper()
    registry["position_base"] = registry.get("position_base", registry_empty_series).astype(str).str.strip().str.upper()
    registry["last_opp_abbr"] = registry.get("last_opp_abbr", registry_empty_series).astype(str).str.strip().str.upper()
    registry["last_seen_date"] = pd.to_datetime(registry.get("last_seen_date"), errors="coerce")
    registry["latest_salary"] = pd.to_numeric(registry.get("latest_salary"), errors="coerce")
    registry["seen_count"] = pd.to_numeric(registry.get("seen_count"), errors="coerce").fillna(0).astype(int)

    resolved_rows: list[dict[str, Any]] = []
    resolution_rows: list[dict[str, Any]] = []

    for row in rotowire.to_dict(orient="records"):
        player_name = str(row.get("player_name") or "").strip()
        team_abbr = str(row.get("team_abbr") or "").strip().upper()
        opp_abbr = str(row.get("opp_abbr") or "").strip().upper()
        salary = _safe_int(row.get("salary"))
        position_text = str(row.get("roto_position") or row.get("site_positions") or "").strip().upper()
        position_base = _position_base(position_text)
        game_info = _build_rotowire_game_info(row)

        candidates = registry.loc[registry["player_name_norm"] == _normalize_text(player_name)].copy()
        if not candidates.empty:
            team_matches = candidates.loc[candidates["team_abbr"] == team_abbr].copy()
            if not team_matches.empty:
                candidates = team_matches

        if candidates.empty:
            resolution_rows.append(
                _resolution_row(
                    player_name=player_name,
                    team_abbr=team_abbr,
                    opp_abbr=opp_abbr,
                    salary=salary,
                    position=position_text,
                    status="unresolved",
                    reason="no_registry_name_match",
                )
            )
            resolved_rows.append(
                _resolved_slate_row(
                    row=row,
                    dk_id="",
                    name_plus_id="",
                    position=position_base,
                    roster_position=_derive_roster_position(row, fallback_position=position_base),
                    game_info=game_info,
                    status="unresolved",
                    confidence=0.0,
                )
            )
            continue

        scored = candidates.copy()
        scored["_score"] = scored.apply(
            lambda candidate: _score_candidate(
                candidate=candidate,
                team_abbr=team_abbr,
                opp_abbr=opp_abbr,
                salary=salary,
                position_base=position_base,
                selected_date=slate_ts,
            ),
            axis=1,
        )
        scored = scored.sort_values(
            ["_score", "seen_count", "last_seen_date"],
            ascending=[False, False, False],
            kind="stable",
        ).reset_index(drop=True)
        best = scored.iloc[0]
        best_score = float(best["_score"])
        second_score = float(scored.iloc[1]["_score"]) if len(scored) > 1 else float("-inf")
        if best_score < float(min_confidence_score):
            status = "unresolved"
            reason = "low_confidence"
        elif len(scored) > 1 and abs(best_score - second_score) < 3.0:
            status = "conflict"
            reason = "ambiguous_registry_match"
        else:
            status = "resolved"
            reason = "matched"

        matched_id = str(best.get("dk_id") or "") if status == "resolved" else ""
        matched_name_plus_id = str(best.get("name_plus_id") or "") if status == "resolved" else ""
        resolved_rows.append(
            _resolved_slate_row(
                row=row,
                dk_id=matched_id,
                name_plus_id=matched_name_plus_id,
                position=str(best.get("position") or position_base).upper(),
                roster_position=str(best.get("roster_position") or _derive_roster_position(row, fallback_position=position_base)).upper(),
                game_info=game_info,
                status=status,
                confidence=best_score,
            )
        )
        resolution_rows.append(
            _resolution_row(
                player_name=player_name,
                team_abbr=team_abbr,
                opp_abbr=opp_abbr,
                salary=salary,
                position=position_text,
                status=status,
                reason=reason,
                dk_id=matched_id,
                name_plus_id=matched_name_plus_id,
                confidence=best_score,
                candidate_count=int(len(scored)),
                matched_team_abbr=str(best.get("team_abbr") or "").upper(),
                matched_salary=_safe_int(best.get("latest_salary")),
                matched_last_seen_date=best.get("last_seen_date"),
            )
        )

    resolution_df = pd.DataFrame(resolution_rows)
    if resolution_df.empty:
        resolution_df = _empty_resolution_frame()
    resolved_slate = pd.DataFrame(resolved_rows)
    if resolved_slate.empty:
        resolved_slate = pd.DataFrame(
            columns=[
                "Position",
                "Name + ID",
                "Name",
                "ID",
                "Roster Position",
                "Salary",
                "Game Info",
                "TeamAbbrev",
                "AvgPointsPerGame",
                "dk_resolution_status",
                "dk_match_confidence",
                "rw_id",
            ]
        )

    resolved_count = int((resolution_df.get("dk_resolution_status") == "resolved").sum()) if not resolution_df.empty else 0
    unresolved_count = int((resolution_df.get("dk_resolution_status") == "unresolved").sum()) if not resolution_df.empty else 0
    conflict_count = int((resolution_df.get("dk_resolution_status") == "conflict").sum()) if not resolution_df.empty else 0
    total_players = int(len(rotowire))
    meta = {
        "source": "rotowire_registry",
        "players_total": total_players,
        "resolved_players": resolved_count,
        "unresolved_players": unresolved_count,
        "conflict_players": conflict_count,
        "coverage_pct": round((resolved_count / total_players) * 100.0, 2) if total_players else 0.0,
        "fully_resolved": bool(total_players > 0 and resolved_count == total_players and conflict_count == 0),
        "slate_key": str(slate_key or "").strip().lower(),
    }
    return resolved_slate, resolution_df, meta


def _score_candidate(
    candidate: pd.Series,
    team_abbr: str,
    opp_abbr: str,
    salary: int | None,
    position_base: str,
    selected_date: pd.Timestamp | pd.NaT,
) -> float:
    score = 0.0
    candidate_team = str(candidate.get("team_abbr") or "").strip().upper()
    if candidate_team and team_abbr and candidate_team == team_abbr:
        score += 45.0
    elif candidate_team and team_abbr:
        score -= 10.0

    candidate_position = str(candidate.get("position_base") or "").strip().upper()
    if candidate_position and position_base and candidate_position == position_base:
        score += 12.0
    elif candidate_position and position_base:
        score -= 5.0

    candidate_salary = _safe_int(candidate.get("latest_salary"))
    if candidate_salary is not None and salary is not None:
        salary_gap = abs(candidate_salary - salary)
        if salary_gap == 0:
            score += 18.0
        elif salary_gap <= 200:
            score += 14.0
        elif salary_gap <= 500:
            score += 9.0
        elif salary_gap <= 1000:
            score += 4.0
        else:
            score -= 6.0

    candidate_opp = str(candidate.get("last_opp_abbr") or "").strip().upper()
    if candidate_opp and opp_abbr and candidate_opp == opp_abbr:
        score += 7.0

    last_seen = pd.to_datetime(candidate.get("last_seen_date"), errors="coerce")
    if pd.notna(last_seen) and pd.notna(selected_date):
        day_gap = abs((selected_date.normalize() - last_seen.normalize()).days)
        if day_gap <= 7:
            score += 6.0
        elif day_gap <= 30:
            score += 4.0
        elif day_gap <= 90:
            score += 2.0

    seen_count = _safe_int(candidate.get("seen_count")) or 0
    if seen_count >= 5:
        score += 4.0
    elif seen_count >= 2:
        score += 2.0
    return float(score)


def _derive_roster_position(row: dict[str, Any], fallback_position: str) -> str:
    site_positions = str(row.get("site_positions") or "").strip().upper()
    if site_positions:
        return site_positions if "UTIL" in site_positions else f"{site_positions}/UTIL"
    base = str(fallback_position or "").strip().upper()
    return f"{base}/UTIL" if base else "UTIL"


def _build_rotowire_game_info(row: dict[str, Any]) -> str:
    team_abbr = str(row.get("team_abbr") or "").strip().upper()
    opp_abbr = str(row.get("opp_abbr") or "").strip().upper()
    if not team_abbr or not opp_abbr:
        return ""
    is_home = bool(row.get("is_home"))
    game_key = f"{opp_abbr}@{team_abbr}" if is_home else f"{team_abbr}@{opp_abbr}"
    game_dt = pd.to_datetime(row.get("game_datetime"), errors="coerce")
    if pd.notna(game_dt):
        return f"{game_key} {game_dt.strftime('%m/%d/%Y %I:%M%p ET')}"
    return game_key


def _resolved_slate_row(
    row: dict[str, Any],
    dk_id: str,
    name_plus_id: str,
    position: str,
    roster_position: str,
    game_info: str,
    status: str,
    confidence: float,
) -> dict[str, Any]:
    player_name = str(row.get("player_name") or "").strip()
    avg_points = _safe_float(row.get("avg_fpts_season"))
    salary = _safe_int(row.get("salary")) or 0
    return {
        "Position": str(position or "").strip().upper(),
        "Name + ID": str(name_plus_id or "").strip(),
        "Name": player_name,
        "ID": str(dk_id or "").strip(),
        "Roster Position": str(roster_position or "").strip().upper(),
        "Salary": salary,
        "Game Info": game_info,
        "TeamAbbrev": str(row.get("team_abbr") or "").strip().upper(),
        "AvgPointsPerGame": round(float(avg_points), 2) if avg_points is not None else pd.NA,
        "dk_resolution_status": str(status or "").strip().lower(),
        "dk_match_confidence": round(float(confidence), 2),
        "rw_id": _safe_int(row.get("rw_id")),
    }


def _resolution_row(
    player_name: str,
    team_abbr: str,
    opp_abbr: str,
    salary: int | None,
    position: str,
    status: str,
    reason: str,
    dk_id: str = "",
    name_plus_id: str = "",
    confidence: float = 0.0,
    candidate_count: int = 0,
    matched_team_abbr: str = "",
    matched_salary: int | None = None,
    matched_last_seen_date: Any = None,
) -> dict[str, Any]:
    return {
        "player_name": player_name,
        "team_abbr": team_abbr,
        "opp_abbr": opp_abbr,
        "salary": salary,
        "position": position,
        "dk_resolution_status": str(status or "").strip().lower(),
        "dk_id": str(dk_id or "").strip(),
        "name_plus_id": str(name_plus_id or "").strip(),
        "dk_match_confidence": round(float(confidence), 2),
        "dk_match_reason": str(reason or "").strip().lower(),
        "candidate_count": int(candidate_count),
        "matched_team_abbr": str(matched_team_abbr or "").strip().upper(),
        "matched_salary": matched_salary,
        "matched_last_seen_date": pd.to_datetime(matched_last_seen_date, errors="coerce"),
    }
