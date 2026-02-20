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


def _safe_num(value: Any, default: float = 0.0) -> float:
    num = _safe_float(value)
    if num is None or math.isnan(num):
        return float(default)
    return float(num)


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


def _normalize_col_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def normalize_injuries_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["player_name", "team", "status", "notes", "active", "updated_at"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    out = df.copy()
    normalized_columns = {_normalize_col_name(col): col for col in out.columns}
    rename_map = {
        "player": "player_name",
        "name": "player_name",
        "playername": "player_name",
        "athlete": "player_name",
        "playerfullname": "player_name",
        "injuredplayer": "player_name",
        "team": "team",
        "teamabbrev": "team",
        "teamabbreviation": "team",
        "teamabbr": "team",
        "teamname": "team",
        "school": "team",
        "status": "status",
        "injurystatus": "status",
        "designation": "status",
        "reportstatus": "status",
        "notes": "notes",
        "injury": "injury_detail",
        "comment": "notes",
        "injurynotes": "notes",
        "estreturn": "est_return",
        "isactive": "active",
        "active": "active",
        "enabled": "active",
        "updatedat": "updated_at",
    }
    resolved_rename: dict[str, str] = {}
    for source_key, dest_col in rename_map.items():
        source_col = normalized_columns.get(source_key)
        if source_col:
            resolved_rename[source_col] = dest_col
    if resolved_rename:
        out = out.rename(columns=resolved_rename)
    had_active_col = "active" in out.columns
    for col in cols:
        if col not in out.columns:
            out[col] = ""

    injury_detail = out.get("injury_detail")
    est_return = out.get("est_return")
    if injury_detail is not None:
        injury_text = injury_detail.astype(str).str.strip()
        notes_blank = out["notes"].astype(str).str.strip() == ""
        out.loc[notes_blank, "notes"] = injury_text.loc[notes_blank]
        if est_return is not None:
            est_text = est_return.astype(str).str.strip()
            has_est = est_text != ""
            use_est = notes_blank & has_est
            out.loc[use_est, "notes"] = (
                injury_text.loc[use_est] + " | Est Return: " + est_text.loc[use_est]
            )

    out["player_name"] = out["player_name"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["status"] = out["status"].astype(str).str.strip().str.lower()
    status_map = {
        "o": "out",
        "out": "out",
        "outforseason": "out",
        "season": "out",
        "d": "doubtful",
        "doubtful": "doubtful",
        "q": "questionable",
        "questionable": "questionable",
        "gametimedecision": "questionable",
        "daytoday": "questionable",
        "gtd": "questionable",
        "p": "probable",
        "probable": "probable",
        "a": "available",
        "available": "available",
        "active": "available",
    }
    out["status"] = out["status"].map(
        lambda x: status_map.get(_normalize_col_name(x), str(x).strip().lower())
    )
    out["notes"] = out["notes"].astype(str).str.strip()
    out["updated_at"] = out["updated_at"].astype(str).str.strip()
    if had_active_col:
        out["active"] = out["active"].map(
            lambda x: str(x).strip().lower() in {"1", "true", "yes", "y"}
            if pd.notna(x)
            else False
        )
    else:
        # Feed CSV rows are treated as active unless explicitly set otherwise.
        out["active"] = True
    out["updated_at"] = out["updated_at"].where(
        out["updated_at"].astype(str).str.strip() != "",
        pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
    )

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
    if "Name" in working.columns:
        working["_injury_name"] = working["Name"].map(_normalize_text)
    else:
        working["_injury_name"] = ""

    injuries = normalize_injuries_frame(injuries_df)
    if injuries.empty:
        return working.drop(columns=["_injury_key", "_injury_name"]), pd.DataFrame(columns=slate_df.columns)

    statuses = {s.strip().lower() for s in statuses_to_remove if s.strip()}
    active = injuries.loc[injuries["active"]].copy()
    active["status_norm"] = active["status"].astype(str).str.strip().str.lower()
    active = active.loc[active["status_norm"].isin(statuses)]
    if active.empty:
        return working.drop(columns=["_injury_key", "_injury_name"]), pd.DataFrame(columns=slate_df.columns)

    active["_injury_key"] = active.apply(lambda r: _player_team_key(r["player_name"], r["team"]), axis=1)
    if "player_name" in active.columns:
        active["_injury_name"] = active["player_name"].map(_normalize_text)
    else:
        active["_injury_name"] = ""
    blocked = set(active["_injury_key"].unique().tolist())
    blocked_names = set(active["_injury_name"].unique().tolist())

    # Fallback to name-only matching when the slate name is unique to avoid team-label mismatch leakage.
    name_counts = working["_injury_name"].value_counts(dropna=False)
    unique_name_keys = set(name_counts.loc[name_counts == 1].index.tolist())
    blocked_unique_names = blocked_names & unique_name_keys

    removed_mask = working["_injury_key"].isin(blocked) | working["_injury_name"].isin(blocked_unique_names)

    removed = working.loc[removed_mask].drop(columns=["_injury_key", "_injury_name"]).copy()
    filtered = working.loc[~removed_mask].drop(columns=["_injury_key", "_injury_name"]).copy()
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


def _blend_stat_weighted(
    our_series: pd.Series,
    vegas_series: pd.Series,
    vegas_weight_series: pd.Series,
) -> pd.Series:
    our = pd.to_numeric(our_series, errors="coerce")
    vegas = pd.to_numeric(vegas_series, errors="coerce")
    weight = pd.to_numeric(vegas_weight_series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    both = our.notna() & vegas.notna()
    only_our = our.notna() & ~vegas.notna()
    only_vegas = ~our.notna() & vegas.notna()
    out = pd.Series(index=our.index, dtype="float64")
    out.loc[both] = (our.loc[both] * (1.0 - weight.loc[both])) + (vegas.loc[both] * weight.loc[both])
    out.loc[only_our] = our.loc[only_our]
    out.loc[only_vegas] = vegas.loc[only_vegas]
    return out


def _rank_pct_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").rank(method="average", pct=True).fillna(0.0)


def _ownership_temperature_for_games(num_games: int) -> float:
    # Fewer games -> lower temperature (more concentrated ownership); more games -> flatter.
    n = max(1, int(num_games))
    temp = 0.60 + (0.09 * max(0, n - 2))
    return float(min(1.80, max(0.55, temp)))


def _softmax(values: pd.Series, temperature: float) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    temp = max(0.01, float(temperature))
    centered = (v - float(v.mean())) / float(v.std(ddof=0) or 1.0)
    scaled = centered / temp
    shifted = scaled - float(scaled.max())
    exp_vals = shifted.map(math.exp)
    denom = float(exp_vals.sum())
    if denom <= 0:
        return pd.Series([1.0 / max(1, len(v))] * len(v), index=v.index, dtype="float64")
    return exp_vals / denom


def _allocate_ownership_with_cap(
    probs: pd.Series,
    target_total: float,
    cap_per_player: float = 100.0,
) -> pd.Series:
    p = pd.to_numeric(probs, errors="coerce").fillna(0.0).clip(lower=0.0)
    if p.empty:
        return pd.Series(dtype="float64")
    if float(p.sum()) <= 0:
        p = pd.Series([1.0 / float(len(p))] * len(p), index=p.index, dtype="float64")

    total_target = max(0.0, float(target_total))
    cap = max(0.0, float(cap_per_player))
    alloc = pd.Series([0.0] * len(p), index=p.index, dtype="float64")
    remaining_mask = pd.Series([True] * len(p), index=p.index)

    # Water-filling with cap: preserve total mass while respecting per-player max where feasible.
    for _ in range(len(p)):
        remaining_total = total_target - float(alloc.sum())
        if remaining_total <= 0:
            break

        work = p.where(remaining_mask, 0.0)
        denom = float(work.sum())
        if denom <= 0:
            break
        tentative = remaining_total * (work / denom)
        hit_cap = remaining_mask & (tentative >= (cap - 1e-9))

        if not bool(hit_cap.any()):
            alloc.loc[remaining_mask] = alloc.loc[remaining_mask] + tentative.loc[remaining_mask]
            break

        alloc.loc[hit_cap] = cap
        remaining_mask.loc[hit_cap] = False
        if not bool(remaining_mask.any()):
            break

    return alloc


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
        s["game_date"] = pd.to_datetime(s.get("game_date"), errors="coerce")
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

        # Rolling form signal: average minutes over each player's most recent 7 games.
        recent_team = (
            s.sort_values(["_name_norm", "_team_norm", "game_date"])
            .groupby(["_name_norm", "_team_norm"], as_index=False, group_keys=False)
            .tail(7)
            .groupby(["_name_norm", "_team_norm"], as_index=False)["minutes_played"]
            .mean(numeric_only=True)
            .rename(columns={"minutes_played": "our_minutes_last7_team"})
        )
        recent_name = (
            s.sort_values(["_name_norm", "game_date"])
            .groupby(["_name_norm"], as_index=False, group_keys=False)
            .tail(7)
            .groupby(["_name_norm"], as_index=False)["minutes_played"]
            .mean(numeric_only=True)
            .rename(columns={"minutes_played": "our_minutes_last7_name"})
        )

        out = out.merge(agg_team, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(agg_name, on=["_name_norm"], how="left")
        out = out.merge(recent_team, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(recent_name, on=["_name_norm"], how="left")

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

        out["our_minutes_last7"] = pd.to_numeric(out.get("our_minutes_last7_team"), errors="coerce")
        out["our_minutes_last7"] = out["our_minutes_last7"].where(
            out["our_minutes_last7"].notna(),
            pd.to_numeric(out.get("our_minutes_last7_name"), errors="coerce"),
        )
        out["our_minutes_last7"] = out["our_minutes_last7"].where(
            out["our_minutes_last7"].notna(),
            pd.to_numeric(out.get("our_minutes_avg"), errors="coerce"),
        )

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
            "our_minutes_last7",
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

    vegas_market_cols = ["vegas_points_line", "vegas_rebounds_line", "vegas_assists_line", "vegas_threes_line"]
    out["vegas_markets_found"] = out[vegas_market_cols].notna().sum(axis=1).astype(int)
    out["vegas_points_available"] = out["vegas_points_line"].notna()
    out["vegas_projection_usable"] = out["vegas_points_available"] & (out["vegas_markets_found"] >= 2)
    vegas_weight = pd.Series(0.0, index=out.index, dtype="float64")
    vegas_weight.loc[out["vegas_points_available"] & (out["vegas_markets_found"] >= 3)] = 0.55
    vegas_weight.loc[out["vegas_points_available"] & (out["vegas_markets_found"] == 2)] = 0.35
    out["vegas_blend_weight"] = vegas_weight.where(out["vegas_projection_usable"], 0.0)

    # Blended guard-rail stat line. If Vegas is missing a stat, keep our projection.
    out["blend_points_proj"] = _blend_stat_weighted(
        out["our_points_proj"],
        out["vegas_points_line"],
        out["vegas_blend_weight"],
    )
    out["blend_rebounds_proj"] = _blend_stat_weighted(
        out["our_rebounds_proj"],
        out["vegas_rebounds_line"],
        out["vegas_blend_weight"],
    )
    out["blend_assists_proj"] = _blend_stat_weighted(
        out["our_assists_proj"],
        out["vegas_assists_line"],
        out["vegas_blend_weight"],
    )
    out["blend_threes_proj"] = _blend_stat_weighted(
        out["our_threes_proj"],
        out["vegas_threes_line"],
        out["vegas_blend_weight"],
    )

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
    vegas_dk_projection_raw = (
        out["vegas_points_line"].fillna(0)
        + (1.25 * out["vegas_rebounds_line"].fillna(0))
        + (1.5 * out["vegas_assists_line"].fillna(0))
        + (0.5 * out["vegas_threes_line"].fillna(0))
        + (2.0 * out["our_steals_proj"].fillna(0))
        + (2.0 * out["our_blocks_proj"].fillna(0))
        - (0.5 * out["our_turnovers_proj"].fillna(0))
    )
    out["vegas_dk_projection"] = vegas_dk_projection_raw.where(out["vegas_projection_usable"], pd.NA)

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

    proj_pct = _rank_pct_series(out["projected_dk_points"])
    sal_pct = _rank_pct_series(out["Salary"])
    val_pct = _rank_pct_series(out["value_per_1k"])
    mins_signal = pd.to_numeric(out.get("our_minutes_last7"), errors="coerce")
    mins_signal = mins_signal.where(mins_signal.notna(), pd.to_numeric(out.get("our_minutes_avg"), errors="coerce"))
    mins_pct = _rank_pct_series(mins_signal)
    vegas_pts_pct = _rank_pct_series(pd.to_numeric(out.get("vegas_dk_projection"), errors="coerce"))

    # Keep legacy ownership estimate for diagnostics.
    out["projected_ownership_v1"] = (2.0 + 38.0 * ((0.5 * proj_pct) + (0.3 * sal_pct) + (0.2 * val_pct))).round(2)

    # V2 ownership model: slate-size-aware softmax normalized to 8 roster slots (=800% total ownership mass).
    own_score = (
        (0.45 * proj_pct)
        + (0.18 * sal_pct)
        + (0.12 * val_pct)
        + (0.15 * mins_pct)
        + (0.10 * vegas_pts_pct)
    )
    unique_game_keys = {
        str(x or "").strip().upper()
        for x in out.get("game_key", pd.Series(dtype=str)).tolist()
        if str(x or "").strip()
    }
    slate_game_count = max(1, len(unique_game_keys))
    ownership_temperature = _ownership_temperature_for_games(slate_game_count)
    ownership_probs = _softmax(own_score, temperature=ownership_temperature)
    ownership_target_total = 800.0
    ownership_alloc = _allocate_ownership_with_cap(
        ownership_probs,
        target_total=ownership_target_total,
        cap_per_player=100.0,
    )
    out["ownership_model"] = "v2_softmax"
    out["ownership_temperature"] = ownership_temperature
    out["ownership_target_total"] = ownership_target_total
    out["ownership_total_projected"] = float(ownership_alloc.sum())
    out["projected_ownership"] = ownership_alloc.round(2)

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
        "our_minutes_last7_team",
        "our_minutes_last7_name",
    ]
    existing_drop = [c for c in drop_cols if c in out.columns]
    return out.drop(columns=existing_drop)


def apply_contest_objective(
    pool_df: pd.DataFrame,
    contest_type: str,
    include_tail_signals: bool = False,
) -> pd.DataFrame:
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
        out["objective_score"] = base + (0.25 * leverage)
        if include_tail_signals:
            out["objective_score"] = out["objective_score"] + (3.0 * tail_score) + (1.6 * tail_to_own)
    else:  # large gpp default
        out["objective_score"] = base + (0.45 * leverage) + (0.1 * ceiling)
        if include_tail_signals:
            out["objective_score"] = out["objective_score"] + (5.0 * tail_score) + (2.3 * tail_to_own)
    return out


def apply_projection_calibration(
    pool_df: pd.DataFrame,
    projection_scale: float = 1.0,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    scale = _safe_num(projection_scale, 1.0)
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    for col in ["projected_dk_points", "blended_projection", "our_dk_projection", "vegas_dk_projection"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * float(scale)

    if "Salary" in out.columns and "projected_dk_points" in out.columns:
        salary = pd.to_numeric(out["Salary"], errors="coerce")
        proj = pd.to_numeric(out["projected_dk_points"], errors="coerce")
        out["projection_per_dollar"] = proj / salary.replace(0, pd.NA)
        out["value_per_1k"] = out["projection_per_dollar"] * 1000.0

    if "projected_dk_points" in out.columns and "projected_ownership" in out.columns:
        proj = pd.to_numeric(out["projected_dk_points"], errors="coerce").fillna(0.0)
        own = pd.to_numeric(out["projected_ownership"], errors="coerce").fillna(0.0)
        out["leverage_score"] = (proj - (0.15 * own)).round(3)

    out["projection_calibration_scale"] = float(scale)
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


def _salary_texture_bucket(salary_left: int) -> str:
    left = max(0, int(salary_left))
    if left <= 100:
        return "0-100"
    if left <= 300:
        return "101-300"
    if left <= 500:
        return "301-500"
    if left <= 900:
        return "501-900"
    return "901+"


def _lineup_stack_signature(players: list[dict[str, Any]]) -> str:
    counts = _lineup_game_counts(players)
    if not counts:
        return "no_game_key"
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "|".join(f"{game}:{count}" for game, count in ordered)


def _cluster_mutation_type(variant_idx: int) -> str:
    if variant_idx <= 0:
        return "seed"
    mutations = [
        "same_role_swap",
        "bring_back_rotation",
        "stack_size_shift",
        "chalk_pivot",
        "value_risk_swap",
        "salary_texture_shift",
    ]
    return mutations[(variant_idx - 1) % len(mutations)]


def _build_cluster_specs(
    players: list[dict[str, Any]],
    num_lineups: int,
    target_clusters: int = 15,
    variants_per_cluster: int = 10,
) -> list[dict[str, Any]]:
    if num_lineups <= 0:
        return []

    game_rows: dict[str, dict[str, list[float]]] = {}
    for p in players:
        game_key = str(p.get("game_key") or "").strip().upper()
        if not game_key:
            continue
        bucket = game_rows.setdefault(
            game_key,
            {
                "projected": [],
                "ownership": [],
                "tail": [],
                "total": [],
                "spread_abs": [],
            },
        )
        bucket["projected"].append(_safe_num(p.get("projected_dk_points")))
        bucket["ownership"].append(_safe_num(p.get("projected_ownership")))
        bucket["tail"].append(_safe_num(p.get("game_tail_score")))
        total_line = _safe_float(p.get("game_total_line"))
        spread_line = _safe_float(p.get("game_spread_line"))
        if total_line is not None and not math.isnan(total_line):
            bucket["total"].append(float(total_line))
        if spread_line is not None and not math.isnan(spread_line):
            bucket["spread_abs"].append(abs(float(spread_line)))

    if not game_rows:
        return [
            {
                "cluster_id": "C01",
                "cluster_script": "balanced",
                "anchor_game_key": "",
                "target_lineups": num_lineups,
            }
        ]

    metric_rows: list[dict[str, Any]] = []
    for game_key, values in game_rows.items():
        projected = values["projected"] or [0.0]
        ownership = values["ownership"] or [0.0]
        tail = values["tail"] or [0.0]
        totals = values["total"]
        spreads = values["spread_abs"]
        metric_rows.append(
            {
                "game_key": game_key,
                "projected_mean": float(sum(projected) / len(projected)),
                "ownership_mean": float(sum(ownership) / len(ownership)),
                "tail_mean": float(sum(tail) / len(tail)),
                "total_mean": float(sum(totals) / len(totals)) if totals else 0.0,
                "spread_abs_mean": float(sum(spreads) / len(spreads)) if spreads else 999.0,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics["leverage_score"] = metrics["tail_mean"] - (0.35 * metrics["ownership_mean"])
    metrics["contrarian_score"] = metrics["projected_mean"] - (0.5 * metrics["ownership_mean"])

    unique_games = int(len(metrics))
    desired_clusters = max(1, int(math.ceil(float(num_lineups) / max(1, int(variants_per_cluster)))))
    cluster_count = max(1, min(int(target_clusters), unique_games, int(num_lineups), desired_clusters))

    pools: dict[str, list[str]] = {
        "high_total": (
            metrics.sort_values(["total_mean", "projected_mean"], ascending=[False, False])["game_key"].tolist()
        ),
        "tight_spread": (
            metrics.sort_values(["spread_abs_mean", "total_mean"], ascending=[True, False])["game_key"].tolist()
        ),
        "tail_leverage": (
            metrics.sort_values(["leverage_score", "tail_mean"], ascending=[False, False])["game_key"].tolist()
        ),
        "contrarian": (
            metrics.sort_values(["ownership_mean", "contrarian_score"], ascending=[True, False])["game_key"].tolist()
        ),
        "balanced": (
            metrics.sort_values(["projected_mean", "tail_mean"], ascending=[False, False])["game_key"].tolist()
        ),
    }
    script_cycle = ["high_total", "tight_spread", "tail_leverage", "contrarian", "balanced"]
    used_games: set[str] = set()
    specs: list[dict[str, Any]] = []

    for idx in range(cluster_count):
        script = script_cycle[idx % len(script_cycle)]
        pool = pools.get(script) or []
        anchor = ""
        for game_key in pool:
            if game_key not in used_games:
                anchor = game_key
                break
        if not anchor:
            if pool:
                anchor = pool[idx % len(pool)]
            else:
                fallback = metrics["game_key"].tolist()
                anchor = fallback[idx % len(fallback)] if fallback else ""
        if anchor:
            used_games.add(anchor)
        specs.append(
            {
                "cluster_id": f"C{idx + 1:02d}",
                "cluster_script": script,
                "anchor_game_key": anchor,
            }
        )

    base = int(num_lineups // cluster_count)
    remainder = int(num_lineups % cluster_count)
    for idx, spec in enumerate(specs):
        spec["target_lineups"] = base + (1 if idx < remainder else 0)
    return specs


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
    include_tail_signals: bool = False,
    spike_max_pair_overlap: int = 4,
    cluster_target_count: int = 15,
    cluster_variants_per_cluster: int = 10,
    projection_scale: float = 1.0,
    salary_left_target: int | None = 50,
    salary_left_penalty_divisor: float = 75.0,
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

    calibrated = apply_projection_calibration(pool_df, projection_scale=projection_scale)
    scored = apply_contest_objective(calibrated, contest_type, include_tail_signals=include_tail_signals)
    scored = scored.loc[scored["Salary"] > 0].copy()
    min_salary_used = SALARY_CAP - int(max(0, max_salary_left if max_salary_left is not None else SALARY_CAP))
    salary_target = None if salary_left_target is None else max(0, min(SALARY_CAP, int(salary_left_target)))
    salary_penalty_divisor = max(1.0, float(salary_left_penalty_divisor))

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
        "projection_calibration_scale",
    ]
    strategy_norm = str(lineup_strategy or "standard").strip().lower()
    spike_mode = strategy_norm in {"spike", "lineup spike", "spike pairs", "lineup_spike", "lineup_spike_pairs"}
    cluster_mode = strategy_norm in {
        "cluster",
        "cluster_v1",
        "cluster_v1_experimental",
        "cluster_mutation",
        "cluster_mutation_v1",
    }
    spike_pair_overlap_cap = max(0, min(ROSTER_SIZE, int(spike_max_pair_overlap)))

    if cluster_mode:
        return _generate_lineups_cluster_mode(
            players=players,
            num_lineups=num_lineups,
            lock_players=lock_players,
            cap_counts=cap_counts,
            exposure_counts=exposure_counts,
            min_salary_used=min_salary_used,
            min_salary_any=min_salary_any,
            max_salary_any=max_salary_any,
            random_seed=random_seed,
            max_attempts_per_lineup=max_attempts_per_lineup,
            player_cols=player_cols,
            progress_callback=progress_callback,
            cluster_target_count=cluster_target_count,
            cluster_variants_per_cluster=cluster_variants_per_cluster,
            salary_left_target=salary_target,
            salary_left_penalty_divisor=salary_penalty_divisor,
        )

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
            salary_left = SALARY_CAP - final_salary
            if salary_target is not None:
                selected_score -= abs(float(salary_left - salary_target)) / salary_penalty_divisor
            if spike_mode:
                def _safe_num(value: Any) -> float:
                    num = _safe_float(value)
                    if num is None or math.isnan(num):
                        return 0.0
                    return float(num)

                selected_score += 1.25 * _stack_bonus_from_counts(game_counts)
                if include_tail_signals:
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
                "salary_left": SALARY_CAP - lineup_salary,
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


def _generate_lineups_cluster_mode(
    players: list[dict[str, Any]],
    num_lineups: int,
    lock_players: list[dict[str, Any]],
    cap_counts: dict[str, int],
    exposure_counts: dict[str, int],
    min_salary_used: int,
    min_salary_any: int,
    max_salary_any: int,
    random_seed: int,
    max_attempts_per_lineup: int,
    player_cols: list[str],
    progress_callback: Callable[[int, int, str], None] | None = None,
    cluster_target_count: int = 15,
    cluster_variants_per_cluster: int = 10,
    salary_left_target: int | None = 50,
    salary_left_penalty_divisor: float = 75.0,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not players:
        return [], ["No players available for cluster generation."]

    warnings: list[str] = []
    lineups: list[dict[str, Any]] = []
    rng = random.Random(random_seed)
    salary_target = None if salary_left_target is None else max(0, min(SALARY_CAP, int(salary_left_target)))
    salary_penalty_divisor = max(1.0, float(salary_left_penalty_divisor))

    total_baseline = pd.to_numeric(
        pd.Series([_safe_float(p.get("game_total_line")) for p in players]),
        errors="coerce",
    ).dropna()
    spread_baseline = pd.to_numeric(
        pd.Series([abs(_safe_num(p.get("game_spread_line"))) for p in players]),
        errors="coerce",
    ).dropna()
    own_baseline = pd.to_numeric(
        pd.Series([_safe_num(p.get("projected_ownership")) for p in players]),
        errors="coerce",
    ).dropna()
    tail_baseline = pd.to_numeric(
        pd.Series([_safe_num(p.get("game_tail_score")) for p in players]),
        errors="coerce",
    ).dropna()
    base_total = float(total_baseline.median()) if not total_baseline.empty else 145.0
    base_spread = float(spread_baseline.median()) if not spread_baseline.empty else 8.0
    base_own = float(own_baseline.median()) if not own_baseline.empty else 14.0
    base_tail = float(tail_baseline.median()) if not tail_baseline.empty else 50.0

    cluster_specs = _build_cluster_specs(
        players=players,
        num_lineups=num_lineups,
        target_clusters=cluster_target_count,
        variants_per_cluster=cluster_variants_per_cluster,
    )
    if not cluster_specs:
        return [], ["No cluster specs available for this slate."]

    for cluster_idx, cluster in enumerate(cluster_specs):
        cluster_id = str(cluster.get("cluster_id") or f"C{cluster_idx + 1:02d}")
        cluster_script = str(cluster.get("cluster_script") or "balanced")
        anchor_game_key = str(cluster.get("anchor_game_key") or "").strip().upper()
        target_lineups = int(cluster.get("target_lineups") or 0)
        if target_lineups <= 0:
            continue

        cluster_lineups: list[dict[str, Any]] = []
        seed_player_ids: set[str] = set()
        seed_lineup_id: int | None = None

        for variant_idx in range(target_lineups):
            mutation_type = _cluster_mutation_type(variant_idx)
            required_anchor_count = 0
            if anchor_game_key:
                if mutation_type == "seed":
                    required_anchor_count = 2
                elif mutation_type == "stack_size_shift":
                    required_anchor_count = 2 if (variant_idx % 2 == 0) else 1
                else:
                    required_anchor_count = 1

            if progress_callback is not None:
                progress_callback(
                    len(lineups),
                    num_lineups,
                    (
                        f"Generating lineup {len(lineups) + 1} of {num_lineups} "
                        f"({cluster_id} {cluster_script} | {mutation_type})..."
                    ),
                )

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
                    current_anchor_count = (
                        sum(1 for p in selected if str(p.get("game_key") or "").strip().upper() == anchor_game_key)
                        if anchor_game_key
                        else 0
                    )

                    for p in players:
                        pid = str(p["ID"])
                        if pid in selected_ids:
                            continue
                        if exposure_counts.get(pid, 0) >= cap_counts.get(pid, num_lineups):
                            continue

                        next_salary = salary + int(p["Salary"])
                        if next_salary > SALARY_CAP:
                            continue
                        rem_after_pick = remaining_slots - 1
                        if next_salary + (rem_after_pick * min_salary_any) > SALARY_CAP:
                            continue
                        if next_salary + (rem_after_pick * max_salary_any) < min_salary_used:
                            continue

                        pick_game_key = str(p.get("game_key") or "").strip().upper()
                        if anchor_game_key and required_anchor_count > 0:
                            is_anchor_pick = pick_game_key == anchor_game_key
                            projected_anchor = current_anchor_count + (1 if is_anchor_pick else 0)
                            needed_after_pick = max(0, required_anchor_count - projected_anchor)
                            if needed_after_pick > rem_after_pick:
                                continue

                        partial = selected + [p]
                        if not _is_feasible_partial(partial, ROSTER_SIZE, rem_after_pick):
                            continue

                        base_weight = max(0.01, float(p.get("objective_score", 0.0)))
                        weight_mult = 1.0
                        if anchor_game_key:
                            if pick_game_key == anchor_game_key:
                                weight_mult *= 1.30
                            elif pick_game_key:
                                weight_mult *= 0.94

                        total_line = _safe_num(p.get("game_total_line"), base_total)
                        spread_abs = abs(_safe_num(p.get("game_spread_line"), base_spread))
                        own_pct = _safe_num(p.get("projected_ownership"), base_own)
                        tail_score = _safe_num(p.get("game_tail_score"), base_tail)

                        if cluster_script == "high_total":
                            weight_mult *= 1.0 + max(0.0, (total_line - base_total) / 180.0)
                        elif cluster_script == "tight_spread":
                            weight_mult *= 1.0 + max(0.0, (base_spread - spread_abs) / 35.0)
                        elif cluster_script == "tail_leverage":
                            weight_mult *= 1.0 + (tail_score / 400.0) + max(0.0, (base_own - own_pct) / 200.0)
                        elif cluster_script == "contrarian":
                            weight_mult *= 1.0 + max(0.0, (base_own - own_pct) / 120.0)
                        else:
                            weight_mult *= 1.0 + max(0.0, (tail_score - base_tail) / 600.0)

                        candidates.append(p)
                        weights.append(base_weight * max(0.05, weight_mult) * rng.uniform(0.85, 1.15))

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

                game_counts = _lineup_game_counts(selected)
                anchor_count = int(game_counts.get(anchor_game_key, 0)) if anchor_game_key else 0
                if anchor_game_key and anchor_count < required_anchor_count:
                    continue

                lineup_own = float(sum(float(p.get("projected_ownership") or 0.0) for p in selected))
                selected_score = sum(float(p.get("objective_score", 0.0)) for p in selected)
                selected_score += 1.10 * _stack_bonus_from_counts(game_counts)
                salary_left = SALARY_CAP - final_salary
                if salary_target is not None:
                    selected_score -= abs(float(salary_left - salary_target)) / salary_penalty_divisor
                if anchor_game_key:
                    selected_score += 1.8 * float(anchor_count)

                current_ids = {str(p["ID"]) for p in selected}
                if lineups:
                    max_overlap = max(len(current_ids & set(l["player_ids"])) for l in lineups)
                    selected_score -= max(0, max_overlap - 5) * 2.5
                if cluster_lineups:
                    max_cluster_overlap = max(len(current_ids & set(l["player_ids"])) for l in cluster_lineups)
                    selected_score -= max(0, max_cluster_overlap - 6) * 1.4

                if mutation_type == "same_role_swap" and seed_player_ids:
                    overlap_seed = len(current_ids & seed_player_ids)
                    selected_score -= max(0, overlap_seed - 5) * 1.6
                    selected_score += max(0.0, 3.0 - abs(float(overlap_seed) - 4.0))
                elif mutation_type == "bring_back_rotation" and anchor_game_key:
                    anchor_teams = {
                        str(p.get("TeamAbbrev") or "").strip().upper()
                        for p in selected
                        if str(p.get("game_key") or "").strip().upper() == anchor_game_key
                    }
                    if len(anchor_teams) >= 2:
                        selected_score += 2.8
                    else:
                        selected_score -= 2.0
                elif mutation_type == "stack_size_shift" and anchor_game_key:
                    target_anchor = 3 if (variant_idx % 2 == 0) else 2
                    selected_score -= abs(anchor_count - target_anchor) * 1.4
                elif mutation_type == "chalk_pivot":
                    selected_score -= 0.22 * lineup_own
                elif mutation_type == "value_risk_swap":
                    value_players = [p for p in selected if int(p.get("Salary") or 0) <= 5200]
                    selected_score += 0.35 * float(len(value_players))
                    if value_players:
                        value_minutes = [
                            _safe_num(p.get("our_minutes_last7"), _safe_num(p.get("our_minutes_avg"), 0.0))
                            for p in value_players
                        ]
                        selected_score += (sum(value_minutes) / max(1.0, float(len(value_minutes)))) / 25.0
                elif mutation_type == "salary_texture_shift":
                    salary_left = SALARY_CAP - final_salary
                    target_salary_left = [100, 300, 500, 700, 900][variant_idx % 5]
                    selected_score -= abs(float(salary_left - target_salary_left)) / 95.0

                if selected_score > best_score:
                    best_score = selected_score
                    best_lineup = [{k: p.get(k) for k in player_cols} for p in selected]

            if best_lineup is None:
                warnings.append(
                    f"Cluster {cluster_id} ({cluster_script}) stopped early at variant {variant_idx + 1}."
                )
                break

            for p in best_lineup:
                exposure_counts[str(p["ID"])] = exposure_counts.get(str(p["ID"]), 0) + 1

            lineup_salary = int(sum(int(p["Salary"]) for p in best_lineup))
            lineup_proj = float(sum(float(p.get("projected_dk_points") or 0.0) for p in best_lineup))
            lineup_own = float(sum(float(p.get("projected_ownership") or 0.0) for p in best_lineup))
            lineup_number = len(lineups) + 1
            if seed_lineup_id is None:
                seed_lineup_id = lineup_number
                mutation_label = "seed"
                seed_player_ids = {str(p["ID"]) for p in best_lineup}
            else:
                mutation_label = mutation_type

            lineup_payload = {
                "lineup_number": lineup_number,
                "players": best_lineup,
                "player_ids": [str(p["ID"]) for p in best_lineup],
                "salary": lineup_salary,
                "salary_left": SALARY_CAP - lineup_salary,
                "projected_points": round(lineup_proj, 2),
                "projected_ownership_sum": round(lineup_own, 2),
                "lineup_strategy": "cluster",
                "pair_id": None,
                "pair_role": None,
                "cluster_id": cluster_id,
                "cluster_script": cluster_script,
                "anchor_game_key": anchor_game_key,
                "seed_lineup_id": seed_lineup_id,
                "mutation_type": mutation_label,
                "stack_signature": _lineup_stack_signature(best_lineup),
                "salary_texture_bucket": _salary_texture_bucket(SALARY_CAP - lineup_salary),
            }
            lineups.append(lineup_payload)
            cluster_lineups.append(lineup_payload)

            if progress_callback is not None:
                progress_callback(
                    len(lineups),
                    num_lineups,
                    f"Generated {len(lineups)} of {num_lineups} lineups ({cluster_id}).",
                )
            if len(lineups) >= num_lineups:
                break

        if len(lineups) >= num_lineups:
            break

    if len(lineups) < num_lineups:
        warnings.append(
            f"Cluster generator returned {len(lineups)} of {num_lineups} requested lineups."
        )
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
        if lineup.get("cluster_id"):
            row["Cluster"] = str(lineup.get("cluster_id"))
        if lineup.get("mutation_type"):
            row["Mutation"] = str(lineup.get("mutation_type"))
        if lineup.get("stack_signature"):
            row["Stack Signature"] = str(lineup.get("stack_signature"))
        if lineup.get("salary_texture_bucket"):
            row["Salary Texture"] = str(lineup.get("salary_texture_bucket"))
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
        if lineup.get("cluster_id"):
            row["Cluster"] = str(lineup.get("cluster_id"))
        if lineup.get("mutation_type"):
            row["Mutation"] = str(lineup.get("mutation_type"))
        if lineup.get("stack_signature"):
            row["Stack Signature"] = str(lineup.get("stack_signature"))
        if lineup.get("salary_texture_bucket"):
            row["Salary Texture"] = str(lineup.get("salary_texture_bucket"))
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
