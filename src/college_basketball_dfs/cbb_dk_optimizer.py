from __future__ import annotations

import csv
import io
import math
import random
import re
from typing import Any, Callable

import pandas as pd

SALARY_CAP = 50000
ROSTER_SIZE = 8
MIN_G = 3
MIN_F = 3

INJURY_STATUSES_REMOVE_DEFAULT = ("out", "doubtful")


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]", "", text)


def _player_team_key(player_name: Any, team: Any) -> str:
    return f"{_normalize_text(player_name)}|{_normalize_text(team)}"


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    as_float = _safe_float(value)
    if as_float is None:
        return None
    return int(as_float)


def _position_base(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if not raw:
        return ""
    if raw.startswith("G"):
        return "G"
    if raw.startswith("F"):
        return "F"
    return raw[:1]


def _game_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text.split(" ")[0].upper()


def normalize_injuries_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["player_name", "team", "status", "notes", "active", "updated_at"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    out = df.copy()
    rename_map = {
        "player": "player_name",
        "name": "player_name",
        "teamabbrev": "team",
        "team_abbrev": "team",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    for col in cols:
        if col not in out.columns:
            out[col] = ""

    out["player_name"] = out["player_name"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip()
    out["status"] = out["status"].astype(str).str.strip()
    out["notes"] = out["notes"].astype(str).str.strip()
    out["updated_at"] = out["updated_at"].astype(str).str.strip()
    out["active"] = out["active"].map(lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"} if pd.notna(x) else False)

    out = out.loc[(out["player_name"] != "") & (out["team"] != "")]
    out = out[cols].reset_index(drop=True)
    return out


def remove_injured_players(
    slate_df: pd.DataFrame,
    injuries_df: pd.DataFrame | None,
    statuses_to_remove: tuple[str, ...] = INJURY_STATUSES_REMOVE_DEFAULT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if slate_df.empty:
        return slate_df.copy(), pd.DataFrame(columns=slate_df.columns)

    working = slate_df.copy()
    working["_injury_key"] = working.apply(
        lambda r: _player_team_key(r.get("Name"), r.get("TeamAbbrev")),
        axis=1,
    )

    injuries = normalize_injuries_frame(injuries_df)
    if injuries.empty:
        return working.drop(columns=["_injury_key"]), pd.DataFrame(columns=slate_df.columns)

    statuses = {s.strip().lower() for s in statuses_to_remove if s.strip()}
    active = injuries.loc[injuries["active"]].copy()
    active["status_norm"] = active["status"].astype(str).str.strip().str.lower()
    active = active.loc[active["status_norm"].isin(statuses)]
    if active.empty:
        return working.drop(columns=["_injury_key"]), pd.DataFrame(columns=slate_df.columns)

    active["_injury_key"] = active.apply(lambda r: _player_team_key(r["player_name"], r["team"]), axis=1)
    blocked = set(active["_injury_key"].unique().tolist())
    removed_mask = working["_injury_key"].isin(blocked)

    removed = working.loc[removed_mask].drop(columns=["_injury_key"]).copy()
    filtered = working.loc[~removed_mask].drop(columns=["_injury_key"]).copy()
    return filtered, removed


def _market_alias(market_key: str) -> str:
    mk = market_key.strip().lower()
    if mk in {"player_points"}:
        return "vegas_points_line"
    if mk in {"player_rebounds"}:
        return "vegas_rebounds_line"
    if mk in {"player_assists"}:
        return "vegas_assists_line"
    if mk in {"player_threes", "player_threes_made", "player_3pt_made"}:
        return "vegas_threes_line"
    return f"vegas_{mk}_line"


def build_player_pool(
    slate_df: pd.DataFrame,
    props_df: pd.DataFrame | None,
    bookmaker_filter: str | None = None,
) -> pd.DataFrame:
    if slate_df.empty:
        return pd.DataFrame()

    out = slate_df.copy()
    out["ID"] = out["ID"].astype(str)
    out["Name"] = out["Name"].astype(str).str.strip()
    out["TeamAbbrev"] = out["TeamAbbrev"].astype(str).str.strip().str.upper()
    out["Position"] = out["Position"].astype(str).str.strip().str.upper()
    out["PositionBase"] = out["Position"].map(_position_base)
    out["Salary"] = out["Salary"].map(_safe_int).fillna(0).astype(int)
    out["AvgPointsPerGame"] = pd.to_numeric(out.get("AvgPointsPerGame"), errors="coerce")
    out["game_key"] = out.get("Game Info", "").map(_game_key)
    out["_name_norm"] = out["Name"].map(_normalize_text)

    if props_df is not None and not props_df.empty:
        p = props_df.copy()
        p["player_name"] = p["player_name"].astype(str).str.strip()
        p["bookmaker"] = p.get("bookmaker", "").astype(str).str.strip().str.lower()
        p["market"] = p["market"].astype(str).str.strip().str.lower()
        p["line"] = pd.to_numeric(p.get("line"), errors="coerce")
        p = p.loc[p["market"].str.startswith("player_")]
        if bookmaker_filter:
            p = p.loc[p["bookmaker"] == bookmaker_filter.strip().lower()]

        if not p.empty:
            p["_name_norm"] = p["player_name"].map(_normalize_text)
            p["market_alias"] = p["market"].map(_market_alias)
            agg = (
                p.groupby(["_name_norm", "market_alias"], as_index=False)["line"]
                .median()
                .pivot(index="_name_norm", columns="market_alias", values="line")
                .reset_index()
            )
            out = out.merge(agg, on="_name_norm", how="left")

    for col in ["vegas_points_line", "vegas_rebounds_line", "vegas_assists_line", "vegas_threes_line"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    vegas_core = (
        out["vegas_points_line"].fillna(0)
        + (1.25 * out["vegas_rebounds_line"].fillna(0))
        + (1.5 * out["vegas_assists_line"].fillna(0))
        + (0.5 * out["vegas_threes_line"].fillna(0))
    )
    dd_count = (
        (out["vegas_points_line"].fillna(-1) >= 9.5).astype(int)
        + (out["vegas_rebounds_line"].fillna(-1) >= 9.5).astype(int)
        + (out["vegas_assists_line"].fillna(-1) >= 9.5).astype(int)
    )
    bonus = (dd_count >= 2).astype(int) * 1.5 + (dd_count >= 3).astype(int) * 3.0
    has_vegas = out[["vegas_points_line", "vegas_rebounds_line", "vegas_assists_line", "vegas_threes_line"]].notna().any(axis=1)
    out["vegas_dk_projection"] = (vegas_core + bonus).where(has_vegas, pd.NA)

    out["model_projection"] = pd.to_numeric(out["AvgPointsPerGame"], errors="coerce")
    out["projected_dk_points"] = out["model_projection"].astype(float)
    both_mask = out["vegas_dk_projection"].notna() & out["model_projection"].notna()
    vegas_only_mask = out["vegas_dk_projection"].notna() & ~out["model_projection"].notna()
    out.loc[both_mask, "projected_dk_points"] = (
        (0.65 * out.loc[both_mask, "vegas_dk_projection"]) + (0.35 * out.loc[both_mask, "model_projection"])
    )
    out.loc[vegas_only_mask, "projected_dk_points"] = out.loc[vegas_only_mask, "vegas_dk_projection"]
    out["projected_dk_points"] = out["projected_dk_points"].fillna(0.0)
    out["value_per_1k"] = out["projected_dk_points"] / (out["Salary"].replace(0, pd.NA) / 1000.0)

    proj_pct = out["projected_dk_points"].rank(method="average", pct=True).fillna(0.0)
    sal_pct = out["Salary"].rank(method="average", pct=True).fillna(0.0)
    val_pct = out["value_per_1k"].rank(method="average", pct=True).fillna(0.0)
    out["projected_ownership"] = (2.0 + 38.0 * ((0.5 * proj_pct) + (0.3 * sal_pct) + (0.2 * val_pct))).round(2)
    out["leverage_score"] = (out["projected_dk_points"] - (0.15 * out["projected_ownership"])).round(3)

    return out.drop(columns=["_name_norm"])


def apply_contest_objective(pool_df: pd.DataFrame, contest_type: str) -> pd.DataFrame:
    out = pool_df.copy()
    ct = str(contest_type or "").strip().lower()
    base = out["projected_dk_points"].fillna(0.0)
    own = out["projected_ownership"].fillna(0.0)
    leverage = base - (0.15 * own)
    ceiling = base * 1.18

    if ct == "cash":
        out["objective_score"] = base
    elif ct == "small gpp":
        out["objective_score"] = base + (0.25 * leverage)
    else:  # large gpp default
        out["objective_score"] = base + (0.45 * leverage) + (0.1 * ceiling)
    return out


def _is_feasible_partial(
    selected: list[dict[str, Any]],
    cap: int,
    remaining_player_count_after_pick: int,
) -> bool:
    g_count = sum(1 for p in selected if p["PositionBase"] == "G")
    f_count = sum(1 for p in selected if p["PositionBase"] == "F")
    if len(selected) > cap:
        return False
    if g_count > cap - MIN_F:
        return False
    if f_count > cap - MIN_G:
        return False
    if g_count + remaining_player_count_after_pick < MIN_G:
        return False
    if f_count + remaining_player_count_after_pick < MIN_F:
        return False
    return True


def _lineup_valid(players: list[dict[str, Any]]) -> bool:
    if len(players) != ROSTER_SIZE:
        return False
    g_count = sum(1 for p in players if p["PositionBase"] == "G")
    f_count = sum(1 for p in players if p["PositionBase"] == "F")
    if g_count < MIN_G or f_count < MIN_F:
        return False
    salary = sum(int(p["Salary"]) for p in players)
    if salary > SALARY_CAP:
        return False
    games = {str(p.get("game_key") or "") for p in players if str(p.get("game_key") or "")}
    if len(games) < 2:
        return False
    return True


def _assign_dk_slots(players: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    guards = [p for p in players if p["PositionBase"] == "G"]
    forwards = [p for p in players if p["PositionBase"] == "F"]
    if len(guards) < 3 or len(forwards) < 3:
        return None
    guards = sorted(guards, key=lambda p: float(p.get("objective_score", 0.0)), reverse=True)
    forwards = sorted(forwards, key=lambda p: float(p.get("objective_score", 0.0)), reverse=True)

    g_slots = guards[:3]
    f_slots = forwards[:3]
    used_ids = {str(p["ID"]) for p in g_slots + f_slots}
    util = [p for p in players if str(p["ID"]) not in used_ids]
    if len(util) != 2:
        return None
    return g_slots + f_slots + util


def generate_lineups(
    pool_df: pd.DataFrame,
    num_lineups: int,
    contest_type: str,
    locked_ids: list[str] | None = None,
    excluded_ids: list[str] | None = None,
    exposure_caps_pct: dict[str, float] | None = None,
    random_seed: int = 7,
    max_attempts_per_lineup: int = 1200,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if pool_df.empty:
        return [], ["Player pool is empty."]
    if num_lineups <= 0:
        return [], ["Number of lineups must be > 0."]

    if progress_callback is not None:
        progress_callback(0, num_lineups, "Starting lineup generation...")

    scored = apply_contest_objective(pool_df, contest_type)
    scored = scored.loc[scored["Salary"] > 0].copy()

    locked_set = {str(x) for x in (locked_ids or []) if str(x).strip()}
    excluded_set = {str(x) for x in (excluded_ids or []) if str(x).strip()}
    scored = scored.loc[~scored["ID"].astype(str).isin(excluded_set)].copy()
    if scored.empty:
        return [], ["All players were excluded."]

    if locked_set & excluded_set:
        return [], ["A player cannot be both locked and excluded."]

    players = scored.to_dict(orient="records")
    by_id = {str(p["ID"]): p for p in players}
    missing_locks = [pid for pid in locked_set if pid not in by_id]
    if missing_locks:
        return [], [f"Locked players not found in pool: {', '.join(missing_locks)}"]

    lock_players = [by_id[pid] for pid in locked_set]
    if len(lock_players) > ROSTER_SIZE:
        return [], ["Too many locked players for roster size."]

    lock_salary = sum(int(p["Salary"]) for p in lock_players)
    if lock_salary > SALARY_CAP:
        return [], ["Locked players exceed salary cap."]
    if not _is_feasible_partial(lock_players, ROSTER_SIZE, ROSTER_SIZE - len(lock_players)):
        return [], ["Locked players make roster constraints impossible."]

    cap_counts: dict[str, int] = {str(p["ID"]): num_lineups for p in players}
    exposure_caps_pct = exposure_caps_pct or {}
    for pid, pct in exposure_caps_pct.items():
        player_id = str(pid)
        if player_id not in cap_counts:
            continue
        clamped_pct = max(0.0, min(100.0, float(pct)))
        cap_counts[player_id] = max(0, int(math.floor((clamped_pct / 100.0) * num_lineups)))
    for lock in locked_set:
        cap_counts[lock] = num_lineups

    exposure_counts: dict[str, int] = {str(p["ID"]): 0 for p in players}
    rng = random.Random(random_seed)
    lineups: list[dict[str, Any]] = []
    warnings: list[str] = []

    min_salary_any = int(scored["Salary"].min())
    player_cols = ["ID", "Name", "Name + ID", "TeamAbbrev", "PositionBase", "Salary", "game_key", "objective_score", "projected_dk_points", "projected_ownership"]

    for lineup_idx in range(num_lineups):
        if progress_callback is not None:
            progress_callback(lineup_idx, num_lineups, f"Generating lineup {lineup_idx + 1} of {num_lineups}...")
        best_lineup: list[dict[str, Any]] | None = None
        best_score = -10**12

        for _ in range(max_attempts_per_lineup):
            selected = [dict(x) for x in lock_players]
            selected_ids = {str(p["ID"]) for p in selected}
            salary = sum(int(p["Salary"]) for p in selected)

            while len(selected) < ROSTER_SIZE:
                remaining_slots = ROSTER_SIZE - len(selected)
                candidates: list[dict[str, Any]] = []
                weights: list[float] = []

                for p in players:
                    pid = str(p["ID"])
                    if pid in selected_ids:
                        continue
                    if exposure_counts[pid] >= cap_counts.get(pid, num_lineups):
                        continue
                    next_salary = salary + int(p["Salary"])
                    if next_salary > SALARY_CAP:
                        continue
                    rem_after_pick = remaining_slots - 1
                    if next_salary + (rem_after_pick * min_salary_any) > SALARY_CAP:
                        continue

                    partial = selected + [p]
                    if not _is_feasible_partial(partial, ROSTER_SIZE, rem_after_pick):
                        continue

                    candidates.append(p)
                    base_weight = max(0.01, float(p.get("objective_score", 0.0)))
                    weights.append(base_weight * rng.uniform(0.85, 1.15))

                if not candidates:
                    selected = []
                    break

                pick = rng.choices(candidates, weights=weights, k=1)[0]
                selected.append(pick)
                selected_ids.add(str(pick["ID"]))
                salary += int(pick["Salary"])

            if not selected:
                continue
            if not _lineup_valid(selected):
                continue

            selected_score = sum(float(p.get("objective_score", 0.0)) for p in selected)
            if lineups:
                current_ids = {str(p["ID"]) for p in selected}
                max_overlap = max(len(current_ids & set(l["player_ids"])) for l in lineups)
                selected_score -= max(0, max_overlap - 6) * 2.0

            if selected_score > best_score:
                best_score = selected_score
                best_lineup = [{k: p.get(k) for k in player_cols} for p in selected]

        if best_lineup is None:
            warnings.append(
                f"Stopped early at lineup {lineup_idx + 1}: could not satisfy constraints/exposure with current pool."
            )
            if progress_callback is not None:
                progress_callback(len(lineups), num_lineups, "Stopped early due to constraints.")
            break

        for p in best_lineup:
            exposure_counts[str(p["ID"])] += 1

        lineup_salary = int(sum(int(p["Salary"]) for p in best_lineup))
        lineup_proj = float(sum(float(p.get("projected_dk_points") or 0.0) for p in best_lineup))
        lineup_own = float(sum(float(p.get("projected_ownership") or 0.0) for p in best_lineup))

        lineups.append(
            {
                "lineup_number": len(lineups) + 1,
                "players": best_lineup,
                "player_ids": [str(p["ID"]) for p in best_lineup],
                "salary": lineup_salary,
                "projected_points": round(lineup_proj, 2),
                "projected_ownership_sum": round(lineup_own, 2),
            }
        )

        if progress_callback is not None:
            progress_callback(len(lineups), num_lineups, f"Generated {len(lineups)} of {num_lineups} lineups.")

    return lineups, warnings


def lineups_summary_frame(lineups: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lineup in lineups:
        players = lineup["players"]
        rows.append(
            {
                "Lineup": lineup["lineup_number"],
                "Salary": lineup["salary"],
                "Projected Points": lineup["projected_points"],
                "Projected Ownership Sum": lineup["projected_ownership_sum"],
                "Players": " | ".join(str(p.get("Name + ID") or p.get("Name")) for p in players),
            }
        )
    return pd.DataFrame(rows)


def build_dk_upload_csv(lineups: list[dict[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    header = ["G", "G", "G", "F", "F", "F", "UTIL", "UTIL"]
    writer.writerow(header)
    for lineup in lineups:
        slots = _assign_dk_slots(lineup["players"])
        if slots is None:
            continue
        row = [str(p.get("Name + ID") or p.get("ID") or "") for p in slots]
        writer.writerow(row)
    return output.getvalue()
