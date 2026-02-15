from __future__ import annotations

import csv
import io
import math
import random
import re
from collections import Counter
from typing import Any, Callable

import pandas as pd
from college_basketball_dfs.cbb_tail_model import map_slate_games_to_tail_features

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


def _blend_stat(our_series: pd.Series, vegas_series: pd.Series) -> pd.Series:
    our = pd.to_numeric(our_series, errors="coerce")
    vegas = pd.to_numeric(vegas_series, errors="coerce")
    both = our.notna() & vegas.notna()
    only_our = our.notna() & ~vegas.notna()
    only_vegas = ~our.notna() & vegas.notna()
    out = pd.Series([0.0] * len(our), index=our.index, dtype="float64")
    out.loc[both] = (our.loc[both] + vegas.loc[both]) / 2.0
    out.loc[only_our] = our.loc[only_our]
    out.loc[only_vegas] = vegas.loc[only_vegas]
    return out


def build_player_pool(
    slate_df: pd.DataFrame,
    props_df: pd.DataFrame | None,
    season_stats_df: pd.DataFrame | None = None,
    bookmaker_filter: str | None = None,
    odds_games_df: pd.DataFrame | None = None,
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
    out["_team_norm"] = out["TeamAbbrev"].map(_normalize_text)

    # Attach game-level Vegas tail metrics (p+8 / p+12 / volatility) by matching slate game keys to odds events.
    if odds_games_df is not None and not odds_games_df.empty and "game_key" in out.columns:
        game_keys = sorted({str(x or "").strip().upper() for x in out["game_key"].tolist() if str(x or "").strip()})
        mapped = map_slate_games_to_tail_features(slate_game_keys=game_keys, odds_tail_df=odds_games_df)
        if not mapped.empty:
            out = out.merge(mapped, on="game_key", how="left")

    # Our V1 projection source: season averages from cached NCAA player stats.
    if season_stats_df is not None and not season_stats_df.empty:
        s = season_stats_df.copy()
        s = s.rename(columns={"team_name": "team_name_hist"})
        s["player_name"] = s.get("player_name", "").astype(str).str.strip()
        s["team_name_hist"] = s.get("team_name_hist", "").astype(str).str.strip()
        s["_name_norm"] = s["player_name"].map(_normalize_text)
        s["_team_norm"] = s["team_name_hist"].map(_normalize_text)

        numeric_cols = [
            "points",
            "rebounds",
            "assists",
            "tpm",
            "steals",
            "blocks",
            "turnovers",
            "minutes_played",
            "fga",
            "fta",
            "dk_fpts",
        ]
        for col in numeric_cols:
            if col in s.columns:
                s[col] = pd.to_numeric(s[col], errors="coerce")
            else:
                s[col] = pd.NA

        # Primary key: player+team (if team labels align). Fallback: player-only.
        agg_team = (
            s.groupby(["_name_norm", "_team_norm"], as_index=False)[numeric_cols]
            .mean(numeric_only=True)
            .rename(
                columns={
                    "points": "our_points_avg_team",
                    "rebounds": "our_rebounds_avg_team",
                    "assists": "our_assists_avg_team",
                    "tpm": "our_threes_avg_team",
                    "steals": "our_steals_avg_team",
                    "blocks": "our_blocks_avg_team",
                    "turnovers": "our_turnovers_avg_team",
                    "minutes_played": "our_minutes_avg_team",
                    "fga": "our_fga_avg_team",
                    "fta": "our_fta_avg_team",
                    "dk_fpts": "our_dk_fpts_avg_team",
                }
            )
        )
        agg_name = (
            s.groupby(["_name_norm"], as_index=False)[numeric_cols]
            .mean(numeric_only=True)
            .rename(
                columns={
                    "points": "our_points_avg_name",
                    "rebounds": "our_rebounds_avg_name",
                    "assists": "our_assists_avg_name",
                    "tpm": "our_threes_avg_name",
                    "steals": "our_steals_avg_name",
                    "blocks": "our_blocks_avg_name",
                    "turnovers": "our_turnovers_avg_name",
                    "minutes_played": "our_minutes_avg_name",
                    "fga": "our_fga_avg_name",
                    "fta": "our_fta_avg_name",
                    "dk_fpts": "our_dk_fpts_avg_name",
                }
            )
        )

        out = out.merge(agg_team, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(agg_name, on=["_name_norm"], how="left")

        for base in [
            "our_points_avg",
            "our_rebounds_avg",
            "our_assists_avg",
            "our_threes_avg",
            "our_steals_avg",
            "our_blocks_avg",
            "our_turnovers_avg",
            "our_minutes_avg",
            "our_fga_avg",
            "our_fta_avg",
            "our_dk_fpts_avg",
        ]:
            team_col = f"{base}_team"
            name_col = f"{base}_name"
            out[base] = pd.to_numeric(out.get(team_col), errors="coerce")
            out[base] = out[base].where(out[base].notna(), pd.to_numeric(out.get(name_col), errors="coerce"))

        # Usage proxy: possession involvement estimate per minute.
        out["our_usage_proxy"] = (
            (out["our_fga_avg"].fillna(0) + (0.44 * out["our_fta_avg"].fillna(0)) + out["our_turnovers_avg"].fillna(0))
            / out["our_minutes_avg"].replace(0, pd.NA)
        )
    else:
        for col in [
            "our_points_avg",
            "our_rebounds_avg",
            "our_assists_avg",
            "our_threes_avg",
            "our_steals_avg",
            "our_blocks_avg",
            "our_turnovers_avg",
            "our_minutes_avg",
            "our_fga_avg",
            "our_fta_avg",
            "our_dk_fpts_avg",
            "our_usage_proxy",
        ]:
            out[col] = pd.NA

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

    # Our projection stat line (season averages), with DK Avg fallback for points only.
    out["our_points_proj"] = pd.to_numeric(out["our_points_avg"], errors="coerce")
    out["our_points_proj"] = out["our_points_proj"].where(out["our_points_proj"].notna(), out["AvgPointsPerGame"])
    out["our_rebounds_proj"] = pd.to_numeric(out["our_rebounds_avg"], errors="coerce")
    out["our_assists_proj"] = pd.to_numeric(out["our_assists_avg"], errors="coerce")
    out["our_threes_proj"] = pd.to_numeric(out["our_threes_avg"], errors="coerce")
    out["our_steals_proj"] = pd.to_numeric(out["our_steals_avg"], errors="coerce")
    out["our_blocks_proj"] = pd.to_numeric(out["our_blocks_avg"], errors="coerce")
    out["our_turnovers_proj"] = pd.to_numeric(out["our_turnovers_avg"], errors="coerce")

    # Blended guard-rail stat line. If Vegas is missing a stat, keep our projection.
    out["blend_points_proj"] = _blend_stat(out["our_points_proj"], out["vegas_points_line"])
    out["blend_rebounds_proj"] = _blend_stat(out["our_rebounds_proj"], out["vegas_rebounds_line"])
    out["blend_assists_proj"] = _blend_stat(out["our_assists_proj"], out["vegas_assists_line"])
    out["blend_threes_proj"] = _blend_stat(out["our_threes_proj"], out["vegas_threes_line"])

    # Reference projections: pure "our" and pure "vegas-ish"
    out["our_dk_projection"] = (
        out["our_points_proj"].fillna(0)
        + (1.25 * out["our_rebounds_proj"].fillna(0))
        + (1.5 * out["our_assists_proj"].fillna(0))
        + (2.0 * out["our_steals_proj"].fillna(0))
        + (2.0 * out["our_blocks_proj"].fillna(0))
        - (0.5 * out["our_turnovers_proj"].fillna(0))
        + (0.5 * out["our_threes_proj"].fillna(0))
    )
    out["vegas_dk_projection"] = (
        out["vegas_points_line"].fillna(0)
        + (1.25 * out["vegas_rebounds_line"].fillna(0))
        + (1.5 * out["vegas_assists_line"].fillna(0))
        + (0.5 * out["vegas_threes_line"].fillna(0))
        + (2.0 * out["our_steals_proj"].fillna(0))
        + (2.0 * out["our_blocks_proj"].fillna(0))
        - (0.5 * out["our_turnovers_proj"].fillna(0))
    )

    # Final V1 projected fantasy points: DK scoring from blended stat line + our defensive/TO estimates.
    dd_count = (
        (out["blend_points_proj"] >= 9.5).astype(int)
        + (out["blend_rebounds_proj"] >= 9.5).astype(int)
        + (out["blend_assists_proj"] >= 9.5).astype(int)
        + (out["our_blocks_proj"].fillna(-1) >= 9.5).astype(int)
        + (out["our_steals_proj"].fillna(-1) >= 9.5).astype(int)
    )
    bonus = (dd_count >= 2).astype(int) * 1.5 + (dd_count >= 3).astype(int) * 3.0
    out["projected_dk_points"] = (
        out["blend_points_proj"].fillna(0)
        + (1.25 * out["blend_rebounds_proj"].fillna(0))
        + (1.5 * out["blend_assists_proj"].fillna(0))
        + (0.5 * out["blend_threes_proj"].fillna(0))
        + (2.0 * out["our_steals_proj"].fillna(0))
        + (2.0 * out["our_blocks_proj"].fillna(0))
        - (0.5 * out["our_turnovers_proj"].fillna(0))
        + bonus
    ).astype(float)

    out["blended_projection"] = out["projected_dk_points"]
    out["projection_per_dollar"] = out["projected_dk_points"] / out["Salary"].replace(0, pd.NA)
    out["value_per_1k"] = out["projection_per_dollar"] * 1000.0

    our = pd.to_numeric(out["our_dk_projection"], errors="coerce")
    vegas = pd.to_numeric(out["vegas_dk_projection"], errors="coerce")
    out["vegas_vs_our_delta_pct"] = ((vegas - our) / our.replace(0, pd.NA)) * 100.0
    out["vegas_over_our_flag"] = (
        ((our > 0) & (vegas >= (our * 1.10)))
        .map(lambda x: "📈" if bool(x) else "")
        .astype(str)
    )

    proj_pct = out["projected_dk_points"].rank(method="average", pct=True).fillna(0.0)
    sal_pct = out["Salary"].rank(method="average", pct=True).fillna(0.0)
    val_pct = out["value_per_1k"].rank(method="average", pct=True).fillna(0.0)
    out["projected_ownership"] = (2.0 + 38.0 * ((0.5 * proj_pct) + (0.3 * sal_pct) + (0.2 * val_pct))).round(2)

    for col in ["game_p_plus_8", "game_p_plus_12", "game_volatility_score", "game_tail_residual_mu", "game_tail_sigma", "game_tail_match_score"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    game_own_proxy = out.groupby("game_key")["projected_ownership"].mean().rename("game_avg_projected_ownership")
    out = out.merge(game_own_proxy.reset_index(), on="game_key", how="left")
    out["game_avg_projected_ownership"] = pd.to_numeric(out["game_avg_projected_ownership"], errors="coerce").fillna(0.0)
    out["game_tail_to_ownership"] = out["game_p_plus_12"] / (out["game_avg_projected_ownership"] + 0.5)

    p8 = out["game_p_plus_8"].fillna(0.0).clip(lower=0.0, upper=1.0)
    p12 = out["game_p_plus_12"].fillna(0.0).clip(lower=0.0, upper=1.0)
    vol_pct = out["game_volatility_score"].rank(method="average", pct=True).fillna(0.0)
    ratio_pct = out["game_tail_to_ownership"].rank(method="average", pct=True).fillna(0.0)
    out["game_volatility_pct"] = vol_pct
    out["game_tail_to_ownership_pct"] = ratio_pct
    out["game_tail_score"] = (
        100.0 * ((0.45 * p12) + (0.25 * p8) + (0.15 * vol_pct) + (0.15 * ratio_pct))
    ).round(3)

    out["low_own_ceiling_flag"] = (
        (
            (pd.to_numeric(out["projected_ownership"], errors="coerce") < 10.0)
            & (our >= 20.0)
            & (vegas >= 20.0)
        )
        .map(lambda x: "🔥" if bool(x) else "")
        .astype(str)
    )
    out["leverage_score"] = (out["projected_dk_points"] - (0.15 * out["projected_ownership"])).round(3)

    drop_cols = [
        "_name_norm",
        "_team_norm",
        "our_points_avg_team",
        "our_rebounds_avg_team",
        "our_assists_avg_team",
        "our_threes_avg_team",
        "our_steals_avg_team",
        "our_blocks_avg_team",
        "our_turnovers_avg_team",
        "our_minutes_avg_team",
        "our_fga_avg_team",
        "our_fta_avg_team",
        "our_dk_fpts_avg_team",
        "our_points_avg_name",
        "our_rebounds_avg_name",
        "our_assists_avg_name",
        "our_threes_avg_name",
        "our_steals_avg_name",
        "our_blocks_avg_name",
        "our_turnovers_avg_name",
        "our_minutes_avg_name",
        "our_fga_avg_name",
        "our_fta_avg_name",
        "our_dk_fpts_avg_name",
    ]
    existing_drop = [c for c in drop_cols if c in out.columns]
    return out.drop(columns=existing_drop)


def apply_contest_objective(pool_df: pd.DataFrame, contest_type: str) -> pd.DataFrame:
    out = pool_df.copy()
    ct = str(contest_type or "").strip().lower()
    base = out["projected_dk_points"].fillna(0.0)
    own = out["projected_ownership"].fillna(0.0)
    leverage = base - (0.15 * own)
    ceiling = base * 1.18
    tail_score = pd.to_numeric(out.get("game_tail_score"), errors="coerce").fillna(0.0) / 100.0
    tail_to_own = pd.to_numeric(out.get("game_tail_to_ownership_pct"), errors="coerce").fillna(0.0)

    if ct == "cash":
        out["objective_score"] = base
    elif ct == "small gpp":
        out["objective_score"] = base + (0.25 * leverage) + (3.0 * tail_score) + (1.6 * tail_to_own)
    else:  # large gpp default
        out["objective_score"] = base + (0.45 * leverage) + (0.1 * ceiling) + (5.0 * tail_score) + (2.3 * tail_to_own)
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


def _lineup_game_counts(players: list[dict[str, Any]]) -> Counter[str]:
    keys = [str(p.get("game_key") or "") for p in players if str(p.get("game_key") or "")]
    return Counter(keys)


def _primary_game_from_counts(game_counts: Counter[str]) -> str:
    if not game_counts:
        return ""
    return game_counts.most_common(1)[0][0]


def _stack_bonus_from_counts(game_counts: Counter[str]) -> float:
    # Reward concentrated lineup stories for GPP ceiling.
    if not game_counts:
        return 0.0
    return float(sum(max(0, n - 1) for n in game_counts.values()))


def generate_lineups(
    pool_df: pd.DataFrame,
    num_lineups: int,
    contest_type: str,
    locked_ids: list[str] | None = None,
    excluded_ids: list[str] | None = None,
    exposure_caps_pct: dict[str, float] | None = None,
    global_max_exposure_pct: float = 100.0,
    max_salary_left: int | None = None,
    lineup_strategy: str = "standard",
    spike_max_pair_overlap: int = 4,
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
    min_salary_used = SALARY_CAP - int(max(0, max_salary_left if max_salary_left is not None else SALARY_CAP))

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
    max_salary_any = int(scored["Salary"].max())
    if lock_salary + ((ROSTER_SIZE - len(lock_players)) * max_salary_any) < min_salary_used:
        return [], ["Locked players make minimum-salary-used constraint impossible."]
    if not _is_feasible_partial(lock_players, ROSTER_SIZE, ROSTER_SIZE - len(lock_players)):
        return [], ["Locked players make roster constraints impossible."]

    global_pct = max(0.0, min(100.0, float(global_max_exposure_pct)))
    global_cap = max(0, int(math.floor((global_pct / 100.0) * num_lineups)))
    cap_counts: dict[str, int] = {str(p["ID"]): global_cap for p in players}
    exposure_caps_pct = exposure_caps_pct or {}
    for pid, pct in exposure_caps_pct.items():
        player_id = str(pid)
        if player_id not in cap_counts:
            continue
        clamped_pct = max(0.0, min(100.0, float(pct)))
        cap_counts[player_id] = min(
            cap_counts[player_id],
            max(0, int(math.floor((clamped_pct / 100.0) * num_lineups))),
        )
    for lock in locked_set:
        cap_counts[lock] = num_lineups

    exposure_counts: dict[str, int] = {str(p["ID"]): 0 for p in players}
    rng = random.Random(random_seed)
    lineups: list[dict[str, Any]] = []
    warnings: list[str] = []

    min_salary_any = int(scored["Salary"].min())
    player_cols = [
        "ID",
        "Name",
        "Name + ID",
        "TeamAbbrev",
        "PositionBase",
        "Salary",
        "game_key",
        "objective_score",
        "projected_dk_points",
        "projected_ownership",
        "game_tail_event_id",
        "game_tail_match_score",
        "game_p_plus_8",
        "game_p_plus_12",
        "game_volatility_score",
        "game_tail_to_ownership",
        "game_tail_score",
    ]
    strategy_norm = str(lineup_strategy or "standard").strip().lower()
    spike_mode = strategy_norm in {"spike", "lineup spike", "spike pairs", "lineup_spike", "lineup_spike_pairs"}
    spike_pair_overlap_cap = max(0, min(ROSTER_SIZE, int(spike_max_pair_overlap)))

    for lineup_idx in range(num_lineups):
        anchor_ids: set[str] = set()
        anchor_games: set[str] = set()
        anchor_primary_game = ""
        pair_overlap_limit = ROSTER_SIZE
        if spike_mode and (lineup_idx % 2 == 1) and lineups:
            anchor = lineups[lineup_idx - 1]
            anchor_ids = {str(pid) for pid in anchor.get("player_ids", [])}
            anchor_games = {
                str(p.get("game_key") or "")
                for p in anchor.get("players", [])
                if str(p.get("game_key") or "")
            }
            anchor_primary_game = _primary_game_from_counts(_lineup_game_counts(anchor.get("players", [])))
            minimum_overlap_required = len(anchor_ids & locked_set)
            pair_overlap_limit = max(spike_pair_overlap_cap, minimum_overlap_required)

        if progress_callback is not None:
            if spike_mode:
                pair_number = (lineup_idx // 2) + 1
                pair_role = "A" if lineup_idx % 2 == 0 else "B"
                status = f"Generating lineup {lineup_idx + 1} of {num_lineups} (Pair {pair_number}-{pair_role})..."
            else:
                status = f"Generating lineup {lineup_idx + 1} of {num_lineups}..."
            progress_callback(lineup_idx, num_lineups, status)
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
                    if next_salary + (rem_after_pick * max_salary_any) < min_salary_used:
                        continue
                    if anchor_ids:
                        next_selected_ids = selected_ids | {pid}
                        if len(next_selected_ids & anchor_ids) > pair_overlap_limit:
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
            final_salary = int(sum(int(p["Salary"]) for p in selected))
            if final_salary < min_salary_used:
                continue
            current_ids = {str(p["ID"]) for p in selected}
            if anchor_ids and len(current_ids & anchor_ids) > pair_overlap_limit:
                continue

            selected_score = sum(float(p.get("objective_score", 0.0)) for p in selected)
            game_counts = _lineup_game_counts(selected)
            if spike_mode:
                def _safe_num(value: Any) -> float:
                    num = _safe_float(value)
                    if num is None or math.isnan(num):
                        return 0.0
                    return float(num)

                selected_score += 1.25 * _stack_bonus_from_counts(game_counts)
                avg_tail = sum(_safe_num(p.get("game_tail_score")) for p in selected) / max(1, len(selected))
                avg_vol = sum(_safe_num(p.get("game_volatility_score")) for p in selected) / max(1, len(selected))
                selected_score += (0.20 * avg_tail) + (0.02 * avg_vol)

            if lineups:
                max_overlap = max(len(current_ids & set(l["player_ids"])) for l in lineups)
                selected_score -= max(0, max_overlap - 6) * 2.0
            if anchor_ids:
                overlap_count = len(current_ids & anchor_ids)
                selected_score -= 1.4 * float(overlap_count * overlap_count)
                current_games = set(game_counts.keys())
                overlap_games = len(current_games & anchor_games)
                selected_score -= 0.9 * float(overlap_games)
                primary_game = _primary_game_from_counts(game_counts)
                if anchor_primary_game and primary_game and primary_game != anchor_primary_game:
                    selected_score += 2.5

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
        pair_id = ((lineup_idx // 2) + 1) if spike_mode else None
        pair_role = ("A" if (lineup_idx % 2 == 0) else "B") if spike_mode else None

        lineups.append(
            {
                "lineup_number": len(lineups) + 1,
                "players": best_lineup,
                "player_ids": [str(p["ID"]) for p in best_lineup],
                "salary": lineup_salary,
                "projected_points": round(lineup_proj, 2),
                "projected_ownership_sum": round(lineup_own, 2),
                "lineup_strategy": "spike" if spike_mode else "standard",
                "pair_id": pair_id,
                "pair_role": pair_role,
            }
        )

        if progress_callback is not None:
            progress_callback(len(lineups), num_lineups, f"Generated {len(lineups)} of {num_lineups} lineups.")

    return lineups, warnings


def lineups_summary_frame(lineups: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lineup in lineups:
        players = lineup["players"]
        row = {
            "Lineup": lineup["lineup_number"],
            "Salary": lineup["salary"],
            "Projected Points": lineup["projected_points"],
            "Projected Ownership Sum": lineup["projected_ownership_sum"],
            "Players": " | ".join(str(p.get("Name + ID") or p.get("Name")) for p in players),
        }
        if lineup.get("pair_id") is not None and lineup.get("pair_role"):
            row["Pair"] = f"{lineup['pair_id']}{lineup['pair_role']}"
        if lineup.get("lineup_strategy"):
            row["Strategy"] = str(lineup.get("lineup_strategy"))
        rows.append(row)
    return pd.DataFrame(rows)


def lineups_slots_frame(lineups: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lineup in lineups:
        slots = _assign_dk_slots(lineup["players"])
        if slots is None:
            continue
        row = {
            "Lineup": lineup["lineup_number"],
            "Salary": lineup["salary"],
            "Projected Points": lineup["projected_points"],
            "Projected Ownership Sum": lineup["projected_ownership_sum"],
            "G1": slots[0].get("Name + ID") or slots[0].get("Name"),
            "G2": slots[1].get("Name + ID") or slots[1].get("Name"),
            "G3": slots[2].get("Name + ID") or slots[2].get("Name"),
            "F1": slots[3].get("Name + ID") or slots[3].get("Name"),
            "F2": slots[4].get("Name + ID") or slots[4].get("Name"),
            "F3": slots[5].get("Name + ID") or slots[5].get("Name"),
            "UTIL1": slots[6].get("Name + ID") or slots[6].get("Name"),
            "UTIL2": slots[7].get("Name + ID") or slots[7].get("Name"),
        }
        if lineup.get("pair_id") is not None and lineup.get("pair_role"):
            row["Pair"] = f"{lineup['pair_id']}{lineup['pair_role']}"
        if lineup.get("lineup_strategy"):
            row["Strategy"] = str(lineup.get("lineup_strategy"))
        rows.append(row)
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
