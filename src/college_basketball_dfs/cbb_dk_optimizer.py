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
PROJECTION_SALARY_BUCKETS = ("lt4500", "4500_6999", "7000_9999", "gte10000")
PROJECTION_ROLE_BUCKETS = ("guard", "forward", "center", "other")
OWNERSHIP_SALARY_BUCKETS = ("lt5500", "5500_7499", "gte7500")


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


def _player_expected_minutes(player: dict[str, Any]) -> float:
    return max(
        0.0,
        _safe_num(
            player.get("our_minutes_recent"),
            _safe_num(
                player.get("our_minutes_last7"),
                _safe_num(player.get("our_minutes_last3"), _safe_num(player.get("our_minutes_avg"), 0.0)),
            ),
        ),
    )


def _lineup_minutes_metrics(players: list[dict[str, Any]]) -> tuple[float, float]:
    if not players:
        return 0.0, 0.0

    expected_minutes_values: list[float] = []
    last3_minutes_values: list[float] = []
    for player in players:
        expected_minutes = _player_expected_minutes(player)
        last3_minutes = max(0.0, _safe_num(player.get("our_minutes_last3"), expected_minutes))
        expected_minutes_values.append(expected_minutes)
        last3_minutes_values.append(last3_minutes)

    expected_minutes_sum = float(sum(expected_minutes_values))
    avg_minutes_last3 = 0.0 if not last3_minutes_values else (
        float(sum(last3_minutes_values)) / float(len(last3_minutes_values))
    )
    return round(expected_minutes_sum, 2), round(avg_minutes_last3, 2)


def _lineup_ceiling_projection_value(lineup: dict[str, Any]) -> float:
    explicit = _safe_float(lineup.get("ceiling_projection"))
    if explicit is not None and not math.isnan(explicit):
        return round(float(explicit), 2)
    projected = _safe_float(lineup.get("projected_points"))
    if projected is None or math.isnan(projected):
        return 0.0
    return round(float(projected) * 1.18, 2)


def _lineup_model_label_value(lineup: dict[str, Any]) -> str:
    for key in ("lineup_model_label", "version_label", "lineup_model_key", "version_key", "model_profile"):
        value = str(lineup.get(key) or "").strip()
        if value:
            return value
    return ""


def _player_ceiling_projection_value(player: dict[str, Any]) -> float:
    explicit = _safe_float(player.get("ceiling_projection"))
    if explicit is not None and not math.isnan(explicit):
        return round(float(explicit), 2)
    projected = _safe_float(player.get("projected_dk_points"))
    if projected is None or math.isnan(projected):
        return 0.0
    return round(float(projected) * 1.18, 2)


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


def projection_salary_bucket_key(salary: Any) -> str:
    sal = _safe_float(salary)
    if sal is None or math.isnan(sal):
        return "4500_6999"
    if sal < 4500:
        return "lt4500"
    if sal < 7000:
        return "4500_6999"
    if sal < 10000:
        return "7000_9999"
    return "gte10000"


def projection_role_bucket_key(position: Any) -> str:
    pos = str(position or "").strip().upper()
    if not pos:
        return "other"
    if pos.startswith("G"):
        return "guard"
    if pos.startswith("F"):
        return "forward"
    if pos.startswith("C"):
        return "center"
    return "other"


def ownership_salary_bucket_key(salary: Any) -> str:
    sal = _safe_float(salary)
    if sal is None or math.isnan(sal):
        return "5500_7499"
    if sal < 5500:
        return "lt5500"
    if sal < 7500:
        return "5500_7499"
    return "gte7500"


def _normalize_col_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _attach_historical_ownership_priors(
    pool_df: pd.DataFrame,
    ownership_history_df: pd.DataFrame | None,
) -> pd.DataFrame:
    out = pool_df.copy()
    for col in [
        "historical_ownership_avg",
        "historical_ownership_last5",
        "historical_ownership_samples",
        "historical_ownership_baseline",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    if "ownership_prior_source" not in out.columns:
        out["ownership_prior_source"] = ""
    if "historical_ownership_used_in_prior" not in out.columns:
        out["historical_ownership_used_in_prior"] = False
    if "field_ownership_pct" not in out.columns:
        out["field_ownership_pct"] = pd.NA

    existing_field = pd.to_numeric(out.get("field_ownership_pct"), errors="coerce")
    if ownership_history_df is None or ownership_history_df.empty:
        out["ownership_prior_source"] = existing_field.map(lambda x: "current_field" if pd.notna(x) else "")
        return out

    hist = ownership_history_df.copy()
    rename_map: dict[str, str] = {}
    normalized_cols = {_normalize_col_name(col): col for col in hist.columns}
    for alias, dest in {
        "playername": "Name",
        "player": "Name",
        "name": "Name",
        "teamabbrev": "TeamAbbrev",
        "teamabbr": "TeamAbbrev",
        "team": "TeamAbbrev",
        "actualownership": "actual_ownership",
        "actualown": "actual_ownership",
        "fieldownership": "actual_ownership",
        "ownership": "actual_ownership",
        "own": "actual_ownership",
        "pctdrafted": "actual_ownership",
    }.items():
        source_col = normalized_cols.get(alias)
        if source_col and source_col not in rename_map:
            rename_map[source_col] = dest
    if rename_map:
        hist = hist.rename(columns=rename_map)

    if "Name" not in hist.columns or "actual_ownership" not in hist.columns:
        out["ownership_prior_source"] = existing_field.map(lambda x: "current_field" if pd.notna(x) else "")
        return out

    if "TeamAbbrev" not in hist.columns:
        hist["TeamAbbrev"] = ""
    if "review_date" not in hist.columns:
        hist["review_date"] = pd.NaT

    hist["Name"] = hist["Name"].astype(str).str.strip()
    hist["TeamAbbrev"] = hist["TeamAbbrev"].astype(str).str.strip().str.upper()
    hist["actual_ownership"] = pd.to_numeric(hist.get("actual_ownership"), errors="coerce").clip(lower=0.0, upper=100.0)
    hist["review_date"] = pd.to_datetime(hist.get("review_date"), errors="coerce")
    hist["_name_norm"] = hist["Name"].map(_normalize_text)
    hist["_team_norm"] = hist["TeamAbbrev"].map(_normalize_text)
    hist = hist.loc[(hist["_name_norm"] != "") & hist["actual_ownership"].notna()].copy()
    if hist.empty:
        out["ownership_prior_source"] = existing_field.map(lambda x: "current_field" if pd.notna(x) else "")
        return out

    team_hist = hist.loc[hist["_team_norm"] != ""].copy()
    if not team_hist.empty:
        team_hist = team_hist.sort_values(["_name_norm", "_team_norm", "review_date"], kind="stable")
        team_avg = (
            team_hist.groupby(["_name_norm", "_team_norm"], as_index=False)
            .agg(
                historical_ownership_avg_team=("actual_ownership", "mean"),
                historical_ownership_samples_team=("actual_ownership", "count"),
            )
        )
        team_last5 = (
            team_hist.groupby(["_name_norm", "_team_norm"], as_index=False, group_keys=False)
            .tail(5)
            .groupby(["_name_norm", "_team_norm"], as_index=False)["actual_ownership"]
            .mean()
            .rename(columns={"actual_ownership": "historical_ownership_last5_team"})
        )
        out = out.merge(team_avg, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(team_last5, on=["_name_norm", "_team_norm"], how="left")
    else:
        out["historical_ownership_avg_team"] = pd.NA
        out["historical_ownership_last5_team"] = pd.NA
        out["historical_ownership_samples_team"] = pd.NA

    current_name_counts = out["_name_norm"].value_counts(dropna=False)
    unique_name_keys = set(current_name_counts.loc[current_name_counts == 1].index.tolist())
    name_hist = hist.loc[hist["_name_norm"].isin(unique_name_keys)].copy()
    if not name_hist.empty:
        name_hist = name_hist.sort_values(["_name_norm", "review_date"], kind="stable")
        name_avg = (
            name_hist.groupby("_name_norm", as_index=False)
            .agg(
                historical_ownership_avg_name=("actual_ownership", "mean"),
                historical_ownership_samples_name=("actual_ownership", "count"),
            )
        )
        name_last5 = (
            name_hist.groupby("_name_norm", as_index=False, group_keys=False)
            .tail(5)
            .groupby("_name_norm", as_index=False)["actual_ownership"]
            .mean()
            .rename(columns={"actual_ownership": "historical_ownership_last5_name"})
        )
        out = out.merge(name_avg, on="_name_norm", how="left")
        out = out.merge(name_last5, on="_name_norm", how="left")
    else:
        out["historical_ownership_avg_name"] = pd.NA
        out["historical_ownership_last5_name"] = pd.NA
        out["historical_ownership_samples_name"] = pd.NA

    out["historical_ownership_avg"] = pd.to_numeric(out.get("historical_ownership_avg_team"), errors="coerce")
    out["historical_ownership_avg"] = out["historical_ownership_avg"].where(
        out["historical_ownership_avg"].notna(),
        pd.to_numeric(out.get("historical_ownership_avg_name"), errors="coerce"),
    )
    out["historical_ownership_last5"] = pd.to_numeric(out.get("historical_ownership_last5_team"), errors="coerce")
    out["historical_ownership_last5"] = out["historical_ownership_last5"].where(
        out["historical_ownership_last5"].notna(),
        pd.to_numeric(out.get("historical_ownership_last5_name"), errors="coerce"),
    )
    out["historical_ownership_samples"] = pd.to_numeric(out.get("historical_ownership_samples_team"), errors="coerce")
    out["historical_ownership_samples"] = out["historical_ownership_samples"].where(
        out["historical_ownership_samples"].notna(),
        pd.to_numeric(out.get("historical_ownership_samples_name"), errors="coerce"),
    )
    out["historical_ownership_baseline"] = pd.to_numeric(out["historical_ownership_last5"], errors="coerce")
    out["historical_ownership_baseline"] = out["historical_ownership_baseline"].where(
        out["historical_ownership_baseline"].notna(),
        pd.to_numeric(out["historical_ownership_avg"], errors="coerce"),
    )

    historical_baseline = pd.to_numeric(out["historical_ownership_baseline"], errors="coerce")
    current_field = pd.to_numeric(out.get("field_ownership_pct"), errors="coerce")
    out["field_ownership_pct"] = current_field.where(current_field.notna(), historical_baseline).round(2)
    out["historical_ownership_used_in_prior"] = current_field.isna() & historical_baseline.notna()
    source = pd.Series([""] * len(out), index=out.index, dtype="object")
    source.loc[current_field.notna()] = "current_field"
    source.loc[current_field.isna() & pd.to_numeric(out["historical_ownership_last5"], errors="coerce").notna()] = "historical_last5"
    source.loc[
        current_field.isna()
        & pd.to_numeric(out["historical_ownership_last5"], errors="coerce").isna()
        & pd.to_numeric(out["historical_ownership_avg"], errors="coerce").notna()
    ] = "historical_avg"
    out["ownership_prior_source"] = source
    return out


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


def _sigmoid_series(values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    return vals.map(lambda x: 1.0 / (1.0 + math.exp(-float(x))))


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


def _ownership_target_total_from_pool(pool_df: pd.DataFrame, fallback_total: float = 800.0) -> float:
    raw = pool_df.get("ownership_target_total")
    if isinstance(raw, pd.Series):
        numeric = pd.to_numeric(raw, errors="coerce").dropna()
        if not numeric.empty:
            return max(0.0, float(numeric.iloc[0]))
    else:
        numeric = _safe_float(raw)
        if numeric is not None:
            return max(0.0, float(numeric))
    return max(0.0, float(fallback_total))


def _recompute_leverage_score(pool_df: pd.DataFrame) -> pd.DataFrame:
    out = pool_df.copy()
    if "projected_dk_points" in out.columns and "projected_ownership" in out.columns:
        proj = pd.to_numeric(out["projected_dk_points"], errors="coerce").fillna(0.0)
        own = pd.to_numeric(out["projected_ownership"], errors="coerce").fillna(0.0)
        out["leverage_score"] = (proj - (0.15 * own)).round(3)
    return out


def normalize_projected_ownership_total(
    pool_df: pd.DataFrame,
    target_total: float | None = None,
    cap_per_player: float = 100.0,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty or "projected_ownership" not in out.columns:
        return out

    current = pd.to_numeric(out.get("projected_ownership"), errors="coerce").fillna(0.0).clip(lower=0.0)
    current_total = float(current.sum())
    desired_total = _ownership_target_total_from_pool(out) if target_total is None else max(0.0, float(target_total))
    out["ownership_total_pre_normalize"] = current_total
    out["ownership_total_target"] = desired_total
    if current_total <= 0.0 or desired_total <= 0.0:
        out["ownership_normalization_scale"] = 1.0
        return _recompute_leverage_score(out)

    probs = current / current_total
    normalized = _allocate_ownership_with_cap(
        probs,
        target_total=desired_total,
        cap_per_player=cap_per_player,
    )
    norm_total = float(normalized.sum())
    if norm_total > 0.0 and abs(norm_total - desired_total) > 1e-6:
        normalized = normalized * (desired_total / norm_total)

    out["ownership_normalization_scale"] = desired_total / current_total
    out["ownership_total_post_normalize"] = float(normalized.sum())
    out["projected_ownership"] = normalized.round(2)
    return _recompute_leverage_score(out)


def _ownership_reliability_for_pool(pool_df: pd.DataFrame, default_reliability: float = 0.50) -> float:
    confidence = pd.to_numeric(pool_df.get("ownership_confidence"), errors="coerce")
    if confidence is None or confidence.dropna().empty:
        return max(0.35, min(0.60, float(default_reliability)))
    median_conf = float(confidence.dropna().median())
    reliability = 0.20 + (0.60 * median_conf)
    return max(0.35, min(0.60, reliability))


def build_game_focus_summary(pool_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "game_key",
        "player_count",
        "team_count",
        "projected_points_sum",
        "projected_points_mean",
        "projected_ownership_mean",
        "historical_ownership_mean",
        "game_tail_score_mean",
        "game_total_line",
        "game_spread_abs",
        "value_per_1k_mean",
        "team_stack_popularity_mean",
        "minutes_shock_boost_mean",
        "cheap_value_rate",
        "game_stack_focus_score",
        "game_stack_focus_rank",
        "game_stack_focus_flag",
        "recommended_stack_size",
    ]
    if pool_df.empty or "game_key" not in pool_df.columns:
        return pd.DataFrame(columns=cols)

    work = pool_df.copy()
    work["game_key"] = work["game_key"].astype(str).str.strip().str.upper()
    work = work.loc[work["game_key"] != ""].copy()
    if work.empty:
        return pd.DataFrame(columns=cols)

    work["projected_dk_points"] = pd.to_numeric(work.get("projected_dk_points"), errors="coerce")
    work["projected_ownership"] = pd.to_numeric(work.get("projected_ownership"), errors="coerce")
    work["historical_ownership_baseline"] = pd.to_numeric(work.get("historical_ownership_baseline"), errors="coerce")
    work["game_tail_score"] = pd.to_numeric(work.get("game_tail_score"), errors="coerce")
    work["game_total_line"] = pd.to_numeric(work.get("game_total_line"), errors="coerce")
    work["game_spread_abs"] = pd.to_numeric(
        work.get("game_spread_line", pd.Series([pd.NA] * len(work), index=work.index)),
        errors="coerce",
    ).abs()
    work["value_per_1k"] = pd.to_numeric(work.get("value_per_1k"), errors="coerce")
    work["team_stack_popularity_score"] = pd.to_numeric(work.get("team_stack_popularity_score"), errors="coerce")
    work["minutes_shock_boost_pct"] = pd.to_numeric(work.get("minutes_shock_boost_pct"), errors="coerce")
    work["Salary"] = pd.to_numeric(work.get("Salary"), errors="coerce")
    work["_cheap_value_flag"] = (
        (work["Salary"].fillna(99999.0) < 5500.0)
        & (work["value_per_1k"].fillna(0.0) >= 4.6)
    ).astype(float)
    work["_player_id"] = work.get("ID", pd.Series(work.index, index=work.index)).astype(str)
    work["_team_norm"] = work.get("TeamAbbrev", pd.Series([""] * len(work), index=work.index)).astype(str).str.strip().str.upper()

    summary = (
        work.groupby("game_key", as_index=False)
        .agg(
            player_count=("_player_id", "nunique"),
            team_count=("_team_norm", lambda s: int(s.loc[s.astype(str).str.strip() != ""].nunique())),
            projected_points_sum=("projected_dk_points", "sum"),
            projected_points_mean=("projected_dk_points", "mean"),
            projected_ownership_mean=("projected_ownership", "mean"),
            historical_ownership_mean=("historical_ownership_baseline", "mean"),
            game_tail_score_mean=("game_tail_score", "mean"),
            game_total_line=("game_total_line", "median"),
            game_spread_abs=("game_spread_abs", "median"),
            value_per_1k_mean=("value_per_1k", "mean"),
            team_stack_popularity_mean=("team_stack_popularity_score", "mean"),
            minutes_shock_boost_mean=("minutes_shock_boost_pct", "mean"),
            cheap_value_rate=("_cheap_value_flag", "mean"),
        )
    )
    if summary.empty:
        return pd.DataFrame(columns=cols)

    projected_sum_pct = _rank_pct_series(summary["projected_points_sum"])
    tail_pct = _rank_pct_series(summary["game_tail_score_mean"])
    historical_pct = _rank_pct_series(summary["historical_ownership_mean"])
    projected_own_pct = _rank_pct_series(summary["projected_ownership_mean"])
    value_pct = _rank_pct_series(summary["value_per_1k_mean"])
    team_stack_pct = _rank_pct_series(summary["team_stack_popularity_mean"])
    cheap_value_pct = _rank_pct_series(summary["cheap_value_rate"])
    minutes_pct = _rank_pct_series(summary["minutes_shock_boost_mean"])
    total_pct = _rank_pct_series(summary["game_total_line"])
    tight_spread_pct = pd.to_numeric(summary["game_spread_abs"], errors="coerce").rank(
        method="average",
        pct=True,
        ascending=False,
    ).fillna(0.0)
    focus_score = (
        (0.24 * projected_sum_pct)
        + (0.17 * tail_pct)
        + (0.14 * historical_pct)
        + (0.11 * projected_own_pct)
        + (0.10 * team_stack_pct)
        + (0.09 * value_pct)
        + (0.05 * cheap_value_pct)
        + (0.04 * minutes_pct)
        + (0.04 * total_pct)
        + (0.02 * tight_spread_pct)
    ).clip(lower=0.0, upper=1.0)
    summary["game_stack_focus_score"] = (100.0 * focus_score).round(3)
    summary = summary.sort_values(
        ["game_stack_focus_score", "projected_points_sum", "game_tail_score_mean"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)
    summary["game_stack_focus_rank"] = summary.index + 1
    summary["game_stack_focus_flag"] = summary["game_stack_focus_rank"] <= min(2, len(summary))
    summary["recommended_stack_size"] = summary["team_count"].map(
        lambda x: 3 if int(_safe_num(x, 0.0)) >= 2 else 2
    ).astype(int)
    return summary[cols]


def recommended_focus_stack_settings(
    pool_df: pd.DataFrame,
    contest_type: str,
    focus_game_count: int = 2,
) -> dict[str, Any]:
    summary = build_game_focus_summary(pool_df)
    focus_count = max(0, int(focus_game_count))
    if focus_count <= 0 or summary.empty:
        return {
            "preferred_game_keys": [],
            "stack_lineup_pct": 0.0,
            "min_players": 0,
            "slate_game_count": 0,
            "focus_summary": summary,
        }

    preferred_game_keys = summary.head(focus_count)["game_key"].astype(str).tolist()
    slate_game_count = int(summary["game_key"].nunique())
    contest_norm = str(contest_type or "").strip().lower()
    if contest_norm == "cash":
        stack_lineup_pct = 0.0
    elif contest_norm == "small gpp":
        stack_lineup_pct = 50.0
    else:
        stack_lineup_pct = 60.0
    if slate_game_count > 0 and slate_game_count <= 4 and stack_lineup_pct > 0.0:
        stack_lineup_pct = min(80.0, stack_lineup_pct + 10.0)
    min_players = 0 if stack_lineup_pct <= 0.0 else (3 if slate_game_count <= 4 else 2)
    return {
        "preferred_game_keys": preferred_game_keys,
        "stack_lineup_pct": float(stack_lineup_pct),
        "min_players": int(min_players),
        "slate_game_count": slate_game_count,
        "focus_summary": summary,
    }


def build_player_pool(
    slate_df: pd.DataFrame,
    props_df: pd.DataFrame | None,
    season_stats_df: pd.DataFrame | None = None,
    ownership_history_df: pd.DataFrame | None = None,
    bookmaker_filter: str | None = None,
    odds_games_df: pd.DataFrame | None = None,
    recent_form_games: int = 7,
    recent_points_weight: float = 0.0,
) -> pd.DataFrame:
    if slate_df.empty:
        return pd.DataFrame()

    recent_window = max(1, min(15, int(_safe_num(recent_form_games, 7))))
    points_recent_weight = max(0.0, min(1.0, float(_safe_num(recent_points_weight, 0.0))))

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
    out = _attach_historical_ownership_priors(out, ownership_history_df)

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

        recent_numeric_cols = ["minutes_played", "points", "dk_fpts"]

        def _recent_rollup(
            frame: pd.DataFrame,
            group_cols: list[str],
            sort_cols: list[str],
            window: int,
            suffix: str,
        ) -> pd.DataFrame:
            return (
                frame.sort_values(sort_cols)
                .groupby(group_cols, as_index=False, group_keys=False)
                .tail(int(window))
                .groupby(group_cols, as_index=False)[recent_numeric_cols]
                .mean(numeric_only=True)
                .rename(
                    columns={
                        "minutes_played": f"our_minutes_{suffix}",
                        "points": f"our_points_{suffix}",
                        "dk_fpts": f"our_dk_fpts_{suffix}",
                    }
                )
            )

        recent_last7_team = _recent_rollup(
            s,
            group_cols=["_name_norm", "_team_norm"],
            sort_cols=["_name_norm", "_team_norm", "game_date"],
            window=7,
            suffix="last7_team",
        )
        recent_last7_name = _recent_rollup(
            s,
            group_cols=["_name_norm"],
            sort_cols=["_name_norm", "game_date"],
            window=7,
            suffix="last7_name",
        )
        recent_last3_team = _recent_rollup(
            s,
            group_cols=["_name_norm", "_team_norm"],
            sort_cols=["_name_norm", "_team_norm", "game_date"],
            window=3,
            suffix="last3_team",
        )
        recent_last3_name = _recent_rollup(
            s,
            group_cols=["_name_norm"],
            sort_cols=["_name_norm", "game_date"],
            window=3,
            suffix="last3_name",
        )
        recent_dynamic_team = _recent_rollup(
            s,
            group_cols=["_name_norm", "_team_norm"],
            sort_cols=["_name_norm", "_team_norm", "game_date"],
            window=recent_window,
            suffix="recent_team",
        )
        recent_dynamic_name = _recent_rollup(
            s,
            group_cols=["_name_norm"],
            sort_cols=["_name_norm", "game_date"],
            window=recent_window,
            suffix="recent_name",
        )

        out = out.merge(agg_team, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(agg_name, on=["_name_norm"], how="left")
        out = out.merge(recent_last7_team, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(recent_last7_name, on=["_name_norm"], how="left")
        out = out.merge(recent_last3_team, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(recent_last3_name, on=["_name_norm"], how="left")
        out = out.merge(recent_dynamic_team, on=["_name_norm", "_team_norm"], how="left")
        out = out.merge(recent_dynamic_name, on=["_name_norm"], how="left")

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
        out["our_minutes_recent"] = pd.to_numeric(out.get("our_minutes_recent_team"), errors="coerce")
        out["our_minutes_recent"] = out["our_minutes_recent"].where(
            out["our_minutes_recent"].notna(),
            pd.to_numeric(out.get("our_minutes_recent_name"), errors="coerce"),
        )
        out["our_minutes_recent"] = out["our_minutes_recent"].where(
            out["our_minutes_recent"].notna(),
            pd.to_numeric(out.get("our_minutes_last7"), errors="coerce"),
        )
        out["our_minutes_last3"] = pd.to_numeric(out.get("our_minutes_last3_team"), errors="coerce")
        out["our_minutes_last3"] = out["our_minutes_last3"].where(
            out["our_minutes_last3"].notna(),
            pd.to_numeric(out.get("our_minutes_last3_name"), errors="coerce"),
        )
        out["our_minutes_last3"] = out["our_minutes_last3"].where(
            out["our_minutes_last3"].notna(),
            pd.to_numeric(out.get("our_minutes_recent"), errors="coerce"),
        )
        out["our_minutes_last3"] = out["our_minutes_last3"].where(
            out["our_minutes_last3"].notna(),
            pd.to_numeric(out.get("our_minutes_last7"), errors="coerce"),
        )
        out["our_minutes_last3"] = out["our_minutes_last3"].where(
            out["our_minutes_last3"].notna(),
            pd.to_numeric(out.get("our_minutes_avg"), errors="coerce"),
        )
        out["our_points_recent"] = pd.to_numeric(out.get("our_points_recent_team"), errors="coerce")
        out["our_points_recent"] = out["our_points_recent"].where(
            out["our_points_recent"].notna(),
            pd.to_numeric(out.get("our_points_recent_name"), errors="coerce"),
        )
        out["our_points_recent"] = out["our_points_recent"].where(
            out["our_points_recent"].notna(),
            pd.to_numeric(out.get("our_points_avg"), errors="coerce"),
        )
        out["our_points_recent"] = out["our_points_recent"].where(
            out["our_points_recent"].notna(),
            pd.to_numeric(out.get("AvgPointsPerGame"), errors="coerce"),
        )
        out["our_dk_fpts_recent"] = pd.to_numeric(out.get("our_dk_fpts_recent_team"), errors="coerce")
        out["our_dk_fpts_recent"] = out["our_dk_fpts_recent"].where(
            out["our_dk_fpts_recent"].notna(),
            pd.to_numeric(out.get("our_dk_fpts_recent_name"), errors="coerce"),
        )
        out["our_dk_fpts_recent"] = out["our_dk_fpts_recent"].where(
            out["our_dk_fpts_recent"].notna(),
            pd.to_numeric(out.get("our_dk_fpts_avg"), errors="coerce"),
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
            "our_minutes_last3",
            "our_minutes_recent",
            "our_points_recent",
            "our_dk_fpts_recent",
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

    # Our projection stat line (season averages), with optional recent-form blend for points.
    points_base = pd.to_numeric(out["our_points_avg"], errors="coerce")
    points_base = points_base.where(points_base.notna(), out["AvgPointsPerGame"])
    points_recent = pd.to_numeric(out.get("our_points_recent"), errors="coerce")
    out["our_points_proj"] = points_base
    if points_recent_weight > 0.0:
        points_base_fill = points_base.where(points_base.notna(), points_recent)
        points_recent_fill = points_recent.where(points_recent.notna(), points_base)
        out["our_points_proj"] = ((1.0 - points_recent_weight) * points_base_fill) + (
            points_recent_weight * points_recent_fill
        )
    out["our_points_proj"] = pd.to_numeric(out["our_points_proj"], errors="coerce")
    out["our_points_proj"] = out["our_points_proj"].where(out["our_points_proj"].notna(), points_base)
    out["our_rebounds_proj"] = pd.to_numeric(out["our_rebounds_avg"], errors="coerce")
    out["our_assists_proj"] = pd.to_numeric(out["our_assists_avg"], errors="coerce")
    out["our_threes_proj"] = pd.to_numeric(out["our_threes_avg"], errors="coerce")
    out["our_steals_proj"] = pd.to_numeric(out["our_steals_avg"], errors="coerce")
    out["our_blocks_proj"] = pd.to_numeric(out["our_blocks_avg"], errors="coerce")
    out["our_turnovers_proj"] = pd.to_numeric(out["our_turnovers_avg"], errors="coerce")
    out["recent_form_games_window"] = int(recent_window)
    out["recent_points_weight"] = float(round(points_recent_weight, 4))

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
    salary_num = pd.to_numeric(out.get("Salary"), errors="coerce")
    value_per_1k = pd.to_numeric(out["value_per_1k"], errors="coerce")
    val_pct = _rank_pct_series(value_per_1k)
    ownership_salary_bucket = salary_num.map(ownership_salary_bucket_key)
    out["ownership_salary_bucket"] = ownership_salary_bucket
    value_tier_median = value_per_1k.groupby(ownership_salary_bucket).transform("median")
    value_tier_std = value_per_1k.groupby(ownership_salary_bucket).transform("std").replace(0.0, pd.NA)
    value_tier_z = ((value_per_1k - value_tier_median) / value_tier_std)
    value_tier_z = value_tier_z.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0).clip(lower=-3.5, upper=3.5)
    value_shock = (value_tier_z.clip(lower=0.0) / 2.5).clip(lower=0.0, upper=1.0)
    tier_value_signal = ((0.55 * val_pct) + (0.45 * value_shock)).clip(lower=0.0, upper=1.0)
    salary_tier_bias = ownership_salary_bucket.map(
        {
            "lt5500": 0.06,
            "5500_7499": 0.02,
            "gte7500": -0.04,
        }
    ).fillna(0.0)
    mins_signal = pd.to_numeric(out.get("our_minutes_recent"), errors="coerce")
    mins_signal = mins_signal.where(mins_signal.notna(), pd.to_numeric(out.get("our_minutes_last7"), errors="coerce"))
    mins_signal = mins_signal.where(mins_signal.notna(), pd.to_numeric(out.get("our_minutes_avg"), errors="coerce"))
    mins_pct = _rank_pct_series(mins_signal)
    vegas_pts_pct = _rank_pct_series(pd.to_numeric(out.get("vegas_dk_projection"), errors="coerce"))
    game_p12_raw = pd.to_numeric(
        out.get("game_p_plus_12", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    game_vol_raw = pd.to_numeric(
        out.get("game_volatility_score", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    game_p12_pct = _rank_pct_series(game_p12_raw)
    game_vol_pct = _rank_pct_series(game_vol_raw)

    # Lightweight minutes/DNP uncertainty model from trend + usage + slate volatility.
    mins_avg = pd.to_numeric(out.get("our_minutes_avg"), errors="coerce")
    mins_recent_signal = pd.to_numeric(out.get("our_minutes_recent"), errors="coerce")
    mins_recent_signal = mins_recent_signal.where(
        mins_recent_signal.notna(),
        pd.to_numeric(out.get("our_minutes_last7"), errors="coerce"),
    )
    mins_ref = mins_avg.where(mins_avg.notna(), mins_recent_signal)
    mins_ref = mins_ref.where(mins_ref.notna(), pd.Series([28.0] * len(out), index=out.index, dtype="float64"))
    mins_recent = mins_recent_signal.where(mins_recent_signal.notna(), mins_avg)
    mins_recent = mins_recent.where(mins_recent.notna(), mins_ref)
    minutes_drop_ratio = ((mins_ref - mins_recent).clip(lower=0.0) / mins_ref.replace(0.0, pd.NA)).fillna(0.0)
    minutes_drop_ratio = minutes_drop_ratio.clip(lower=0.0, upper=1.0)
    low_minutes_risk = ((24.0 - mins_recent).clip(lower=0.0) / 24.0).fillna(0.0).clip(lower=0.0, upper=1.0)
    usage_proxy = pd.to_numeric(out.get("our_usage_proxy"), errors="coerce")
    usage_risk = (1.0 - _rank_pct_series(usage_proxy)).clip(lower=0.0, upper=1.0)
    uncertainty_score = (
        (0.42 * minutes_drop_ratio)
        + (0.23 * low_minutes_risk)
        + (0.15 * usage_risk)
        + (0.20 * game_vol_pct)
    ).clip(lower=0.0, upper=1.0)
    dnp_logit = (
        -2.2
        + (2.6 * minutes_drop_ratio)
        + (1.35 * low_minutes_risk)
        + (0.85 * usage_risk)
        + (0.65 * game_vol_pct)
    )
    dnp_risk_score = _sigmoid_series(dnp_logit).clip(lower=0.0, upper=1.0)
    out["minutes_drop_ratio"] = minutes_drop_ratio.round(4)
    out["minutes_low_floor_risk"] = low_minutes_risk.round(4)
    out["projection_uncertainty_score"] = uncertainty_score.round(4)
    out["dnp_risk_score"] = dnp_risk_score.round(4)
    out["high_uncertainty_flag"] = (
        ((uncertainty_score >= 0.58) | (dnp_risk_score >= 0.30))
        .map(lambda x: bool(x))
        .astype(bool)
    )

    # Ownership surge features from attention + field priors + game tail context.
    attention_signal = pd.to_numeric(
        out.get("attention_index", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    if attention_signal.notna().any():
        attention_pct = _rank_pct_series(attention_signal.fillna(0.0).clip(lower=0.0).map(lambda x: math.log1p(float(x))))
    else:
        attention_pct = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    field_own_signal = pd.to_numeric(
        out.get("field_ownership_pct", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    field_own_pct = _rank_pct_series(field_own_signal)
    surge_score = ((0.55 * attention_pct) + (0.25 * field_own_pct) + (0.20 * game_p12_pct)).clip(lower=0.0, upper=1.0)
    if "game_key" in out.columns:
        game_surge = surge_score.groupby(out["game_key"]).transform("mean").fillna(surge_score)
    else:
        game_surge = surge_score
    surge_uncertainty = (
        (0.55 * game_vol_pct) + (0.45 * (surge_score - game_surge).abs().clip(lower=0.0, upper=1.0))
    ).clip(lower=0.0, upper=1.0)
    surge_flag = ((surge_score >= 0.78) | (game_surge >= 0.72)).astype(float)

    if "TeamAbbrev" in out.columns:
        team_key = out["TeamAbbrev"].astype(str).str.strip().str.upper()
    else:
        team_key = pd.Series(["UNKNOWN"] * len(out), index=out.index, dtype="object")
    team_key = team_key.where(team_key != "", "UNKNOWN")
    cheap_salary_flag = (salary_num < 5500.0).astype(float)
    team_value_mean = value_per_1k.groupby(team_key).transform("mean")
    team_projection_mass = pd.to_numeric(out["projected_dk_points"], errors="coerce").groupby(team_key).transform("sum")
    team_cheap_value_rate = cheap_salary_flag.groupby(team_key).transform("mean").fillna(0.0)
    team_game_surge = game_surge.groupby(team_key).transform("mean").fillna(game_surge)
    team_value_rank = _rank_pct_series(team_value_mean)
    team_projection_mass_rank = _rank_pct_series(team_projection_mass)
    team_stack_popularity = (
        (0.32 * team_value_rank)
        + (0.26 * team_cheap_value_rate)
        + (0.24 * team_projection_mass_rank)
        + (0.18 * team_game_surge)
    ).clip(lower=0.0, upper=1.0)
    out["ownership_value_tier_z"] = value_tier_z.round(4)
    out["ownership_tier_value_signal"] = tier_value_signal.round(4)
    out["ownership_salary_tier_bias"] = salary_tier_bias.round(4)
    out["team_stack_popularity_score"] = team_stack_popularity.round(4)

    # Keep legacy ownership estimate for diagnostics.
    out["projected_ownership_v1"] = (2.0 + 38.0 * ((0.5 * proj_pct) + (0.3 * sal_pct) + (0.2 * val_pct))).round(2)

    # V3 ownership model: salary-tier specific softmax normalized to 8 roster slots (=800% total ownership mass).
    tier_coefficients: dict[str, dict[str, float]] = {
        "lt5500": {
            "proj": 0.33,
            "sal": 0.05,
            "val": 0.18,
            "mins": 0.12,
            "vegas": 0.04,
            "attention": 0.10,
            "field": 0.07,
            "game": 0.07,
            "surge": 0.05,
            "tier_value": 0.24,
            "team_stack": 0.22,
            "tier_bias": 1.00,
        },
        "5500_7499": {
            "proj": 0.43,
            "sal": 0.10,
            "val": 0.14,
            "mins": 0.13,
            "vegas": 0.08,
            "attention": 0.11,
            "field": 0.09,
            "game": 0.08,
            "surge": 0.06,
            "tier_value": 0.12,
            "team_stack": 0.14,
            "tier_bias": 1.00,
        },
        "gte7500": {
            "proj": 0.55,
            "sal": 0.08,
            "val": 0.08,
            "mins": 0.15,
            "vegas": 0.11,
            "attention": 0.10,
            "field": 0.10,
            "game": 0.08,
            "surge": 0.06,
            "tier_value": 0.05,
            "team_stack": 0.08,
            "tier_bias": 1.00,
        },
    }
    own_score = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    for tier_key, weights in tier_coefficients.items():
        tier_mask = ownership_salary_bucket == tier_key
        if not bool(tier_mask.any()):
            continue
        own_score.loc[tier_mask] = (
            (weights["proj"] * proj_pct.loc[tier_mask])
            + (weights["sal"] * sal_pct.loc[tier_mask])
            + (weights["val"] * val_pct.loc[tier_mask])
            + (weights["mins"] * mins_pct.loc[tier_mask])
            + (weights["vegas"] * vegas_pts_pct.loc[tier_mask])
            + (weights["attention"] * attention_pct.loc[tier_mask])
            + (weights["field"] * field_own_pct.loc[tier_mask])
            + (weights["game"] * game_surge.loc[tier_mask])
            + (weights["surge"] * surge_flag.loc[tier_mask])
            + (weights["tier_value"] * tier_value_signal.loc[tier_mask])
            + (weights["team_stack"] * team_stack_popularity.loc[tier_mask])
            + (weights["tier_bias"] * salary_tier_bias.loc[tier_mask])
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

    # Cap leverage when surge context is unstable by shrinking toward slate-average ownership.
    if len(out) > 0:
        baseline_own = ownership_target_total / float(len(out))
        leverage_shrink = ((0.30 * surge_uncertainty) + (0.20 * surge_flag)).clip(lower=0.0, upper=0.45)
        ownership_alloc = (ownership_alloc * (1.0 - leverage_shrink)) + (baseline_own * leverage_shrink)
        ownership_alloc = ownership_alloc.clip(lower=0.0, upper=100.0)
        total_after = float(ownership_alloc.sum())
        if total_after > 0.0:
            ownership_alloc = ownership_alloc * (ownership_target_total / total_after)
        ownership_probs = ownership_alloc / max(1e-9, float(ownership_alloc.sum()))
        ownership_alloc = _allocate_ownership_with_cap(
            ownership_probs,
            target_total=ownership_target_total,
            cap_per_player=100.0,
        )
    else:
        leverage_shrink = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    out["ownership_model"] = "v3_tiered_softmax"
    out["ownership_temperature"] = ownership_temperature
    out["ownership_target_total"] = ownership_target_total
    out["ownership_total_projected"] = float(ownership_alloc.sum())
    out["projected_ownership"] = ownership_alloc.round(2)
    out["ownership_chalk_surge_score"] = (100.0 * surge_score).round(3)
    out["ownership_chalk_surge_flag"] = surge_flag.astype(bool)
    out["ownership_confidence"] = (1.0 - surge_uncertainty).clip(lower=0.0, upper=1.0).round(4)
    out["ownership_leverage_shrink"] = pd.to_numeric(leverage_shrink, errors="coerce").fillna(0.0).round(4)

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
    game_focus_summary = build_game_focus_summary(out)
    if not game_focus_summary.empty:
        out = out.merge(
            game_focus_summary[
                [
                    "game_key",
                    "game_stack_focus_score",
                    "game_stack_focus_rank",
                    "game_stack_focus_flag",
                    "recommended_stack_size",
                ]
            ],
            on="game_key",
            how="left",
        )
    else:
        out["game_stack_focus_score"] = 0.0
        out["game_stack_focus_rank"] = pd.NA
        out["game_stack_focus_flag"] = False
        out["recommended_stack_size"] = 2

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
        "our_minutes_recent_team",
        "our_minutes_recent_name",
        "our_points_recent_team",
        "our_points_recent_name",
        "our_dk_fpts_recent_team",
        "our_dk_fpts_recent_name",
        "our_minutes_last7_team",
        "our_minutes_last7_name",
        "our_minutes_last3_team",
        "our_minutes_last3_name",
        "our_points_last7_team",
        "our_points_last7_name",
        "our_points_last3_team",
        "our_points_last3_name",
        "our_dk_fpts_last7_team",
        "our_dk_fpts_last7_name",
        "our_dk_fpts_last3_team",
        "our_dk_fpts_last3_name",
        "historical_ownership_avg_team",
        "historical_ownership_last5_team",
        "historical_ownership_samples_team",
        "historical_ownership_avg_name",
        "historical_ownership_last5_name",
        "historical_ownership_samples_name",
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


def apply_model_profile_adjustments(
    pool_df: pd.DataFrame,
    model_profile: str = "legacy_baseline",
) -> pd.DataFrame:
    out = pool_df.copy()
    profile_key = str(model_profile or "legacy_baseline").strip().lower()
    out["model_profile"] = str(model_profile or "legacy_baseline")
    out["model_profile_bonus"] = 0.0
    out["model_profile_focus_flag"] = False

    if out.empty:
        return out

    def _numeric_col(col_name: str) -> pd.Series:
        raw = out.get(col_name, pd.Series([pd.NA] * len(out), index=out.index))
        series = pd.to_numeric(raw, errors="coerce")
        if isinstance(series, pd.Series):
            return series
        return pd.Series([series] * len(out), index=out.index, dtype="float64")

    objective = _numeric_col("objective_score").fillna(0.0)
    projection = _numeric_col("projected_dk_points").fillna(0.0)
    ownership = _numeric_col("projected_ownership").fillna(0.0).clip(lower=0.0)
    salary = _numeric_col("Salary")
    value = _numeric_col("value_per_1k")
    surge = (
        _numeric_col("ownership_chalk_surge_score")
        .fillna(0.0)
        .clip(lower=0.0, upper=100.0)
    )
    tail_score = _numeric_col("game_tail_score").fillna(0.0)
    team_stack_popularity = _numeric_col("team_stack_popularity_score").fillna(0.0).clip(0.0, 100.0)
    tier_value_z = _numeric_col("ownership_value_tier_z").fillna(0.0).clip(-3.5, 3.5)
    minutes_boost = _numeric_col("minutes_shock_boost_pct").fillna(0.0).clip(lower=0.0)
    mins_recent = _numeric_col("our_minutes_recent")
    mins_recent = mins_recent.where(mins_recent.notna(), _numeric_col("our_minutes_last7"))
    mins_recent = mins_recent.where(mins_recent.notna(), _numeric_col("our_minutes_avg"))
    mins_avg = _numeric_col("our_minutes_avg")
    mins_avg = mins_avg.where(mins_avg.notna(), mins_recent)
    points_recent = _numeric_col("our_points_recent")
    points_recent = points_recent.where(points_recent.notna(), _numeric_col("our_points_avg"))
    points_recent = points_recent.where(points_recent.notna(), _numeric_col("AvgPointsPerGame"))
    uncertainty = _numeric_col("projection_uncertainty_score").fillna(0.0).clip(0.0, 1.0)
    dnp_risk = _numeric_col("dnp_risk_score").fillna(0.0).clip(0.0, 1.0)

    proj_pct = _rank_pct_series(projection)
    value_pct = _rank_pct_series(value)
    tail_pct = _rank_pct_series(tail_score)
    surge_pct = _rank_pct_series(surge)
    recent_pts_pct = _rank_pct_series(points_recent)
    minutes_recent_pct = _rank_pct_series(mins_recent)
    minutes_boost_pct = _rank_pct_series(minutes_boost)

    minute_trend = ((mins_recent - mins_avg) / mins_avg.replace(0.0, pd.NA)).fillna(0.0).clip(lower=-0.35, upper=0.60)
    minute_riser = minute_trend.clip(lower=0.0)
    low_own_edge = ((14.0 - ownership).clip(lower=0.0) / 14.0).clip(0.0, 1.0)
    risk_penalty = ((0.60 * dnp_risk) + (0.40 * uncertainty)).clip(0.0, 1.0)

    salary_bucket = out.get("ownership_salary_bucket")
    if not isinstance(salary_bucket, pd.Series):
        salary_bucket = salary.map(ownership_salary_bucket_key)
    salary_bucket = salary_bucket.astype(str)

    if profile_key == "standout_capture_v1":
        chalk_shock_score = (
            (((surge - 55.0).clip(lower=0.0) / 45.0).clip(0.0, 1.0))
            * (((20.0 - ownership).clip(lower=0.0) / 20.0).clip(0.0, 1.0))
        )
        standout_signal = (
            (0.40 * proj_pct)
            + (0.18 * tail_pct)
            + (0.16 * recent_pts_pct)
            + (0.12 * minute_riser)
            + (0.08 * low_own_edge)
            + (0.06 * surge_pct)
        ).clip(0.0, 1.0)
        profile_bonus = (
            (4.2 * standout_signal)
            + (3.0 * chalk_shock_score)
            - (2.0 * risk_penalty)
        ).clip(lower=-1.5, upper=6.5)
        focus_mask = (
            ((chalk_shock_score >= 0.35) | (standout_signal >= 0.60))
            & (risk_penalty <= 0.70)
        )
        out["objective_score"] = (objective + profile_bonus).clip(lower=0.01)
        out["model_profile_bonus"] = profile_bonus.round(4)
        out["model_profile_focus_flag"] = focus_mask.astype(bool)
        return out

    if profile_key == "tail_spike_pairs":
        tail_to_own = _numeric_col("game_tail_to_ownership_pct").fillna(0.0).clip(0.0, 1.0)
        leverage = _numeric_col("leverage_score").fillna(0.0)
        leverage_pct = _rank_pct_series(leverage)
        volatility = _numeric_col("game_volatility_score").fillna(0.0).clip(0.0, 1.0)
        stack_focus = (
            out.get("stack_anchor_focus_flag", pd.Series([False] * len(out), index=out.index))
            .map(lambda x: 1.0 if bool(x) else 0.0)
            .astype(float)
        )
        ownership_contrarian_edge = ((18.0 - ownership).clip(lower=0.0) / 18.0).clip(0.0, 1.0)
        spike_signal = (
            (0.28 * tail_pct)
            + (0.18 * tail_to_own)
            + (0.18 * leverage_pct)
            + (0.14 * ownership_contrarian_edge)
            + (0.10 * minutes_boost_pct)
            + (0.07 * volatility)
            + (0.05 * stack_focus)
        ).clip(0.0, 1.0)
        tail_norm = (tail_score / 100.0).clip(0.0, 1.0)
        contrarian_shock = (
            (0.55 * ownership_contrarian_edge * tail_norm)
            + (0.45 * ((surge / 100.0) * tail_to_own))
        ).clip(0.0, 1.0)
        profile_bonus = (
            (4.4 * spike_signal)
            + (2.5 * contrarian_shock)
            - (2.3 * risk_penalty)
        ).clip(lower=-1.8, upper=7.0)
        focus_mask = (
            ((spike_signal >= 0.62) | (contrarian_shock >= 0.36))
            & (risk_penalty <= 0.75)
        )
        out["objective_score"] = (objective + profile_bonus).clip(lower=0.01)
        out["model_profile_bonus"] = profile_bonus.round(4)
        out["model_profile_focus_flag"] = focus_mask.astype(bool)
        return out

    if profile_key == "chalk_value_capture_v1":
        stack_pop_norm = (team_stack_popularity / 100.0).clip(0.0, 1.0)
        tier_value_norm = ((tier_value_z + 3.5) / 7.0).clip(0.0, 1.0)
        value_spike = ((tier_value_z - 0.15).clip(lower=0.0) / 2.85).clip(0.0, 1.0)
        cheap_bias = salary_bucket.map(
            {
                "lt5500": 1.0,
                "5500_7499": 0.58,
                "gte7500": 0.18,
            }
        ).fillna(0.45)
        ownership_chalk_norm = ((ownership - 8.0).clip(lower=0.0, upper=25.0) / 25.0).clip(0.0, 1.0)
        chalk_alignment = (
            (0.50 * (surge / 100.0))
            + (0.30 * stack_pop_norm)
            + (0.20 * ownership_chalk_norm)
        ).clip(0.0, 1.0)
        chalk_value_signal = (
            (0.30 * value_pct)
            + (0.18 * value_spike)
            + (0.14 * cheap_bias)
            + (0.14 * chalk_alignment)
            + (0.10 * proj_pct)
            + (0.08 * minutes_recent_pct)
            + (0.06 * tier_value_norm)
        ).clip(0.0, 1.0)
        profile_bonus = (
            (3.8 * chalk_value_signal)
            + (2.4 * (cheap_bias * chalk_alignment))
            - (2.1 * risk_penalty)
        ).clip(lower=-1.5, upper=6.0)
        focus_mask = (
            (cheap_bias >= 0.55)
            & (value_spike >= 0.45)
            & (chalk_alignment >= 0.48)
            & (risk_penalty <= 0.72)
        )
        out["objective_score"] = (objective + profile_bonus).clip(lower=0.01)
        out["model_profile_bonus"] = profile_bonus.round(4)
        out["model_profile_focus_flag"] = focus_mask.astype(bool)
        return out

    if profile_key == "salary_efficiency_ceiling_v1":
        tail_norm = (tail_score / 100.0).clip(0.0, 1.0)
        tier_value_norm = ((tier_value_z + 3.5) / 7.0).clip(0.0, 1.0)
        mid_salary_bias = (1.0 - ((salary - 6700.0).abs() / 4200.0)).clip(lower=0.0, upper=1.0).fillna(0.0)
        contrarian_edge = ((20.0 - ownership).clip(lower=0.0) / 20.0).clip(0.0, 1.0)
        ceiling_signal = (
            (0.30 * proj_pct)
            + (0.24 * value_pct)
            + (0.17 * tail_pct)
            + (0.10 * minutes_boost_pct)
            + (0.09 * tier_value_norm)
            + (0.10 * mid_salary_bias)
        ).clip(0.0, 1.0)
        upside_signal = (
            (0.55 * tail_norm)
            + (0.25 * contrarian_edge)
            + (0.20 * minute_riser)
        ).clip(0.0, 1.0)
        profile_bonus = (
            (4.1 * ceiling_signal)
            + (2.4 * upside_signal)
            - (2.2 * risk_penalty)
        ).clip(lower=-1.5, upper=6.2)
        focus_mask = (
            ((ceiling_signal >= 0.62) | ((tail_norm >= 0.72) & (value_pct >= 0.65)))
            & (risk_penalty <= 0.72)
        )
        out["objective_score"] = (objective + profile_bonus).clip(lower=0.01)
        out["model_profile_bonus"] = profile_bonus.round(4)
        out["model_profile_focus_flag"] = focus_mask.astype(bool)
        return out

    return out


def apply_projection_calibration(
    pool_df: pd.DataFrame,
    projection_scale: float = 1.0,
    projection_salary_bucket_scales: dict[str, float] | None = None,
    projection_role_bucket_scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    scale = _safe_num(projection_scale, 1.0)
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    bucket_scales: dict[str, float] = {}
    for bucket in PROJECTION_SALARY_BUCKETS:
        raw_scale = None if projection_salary_bucket_scales is None else projection_salary_bucket_scales.get(bucket)
        bucket_scale = _safe_num(raw_scale, 1.0)
        if not math.isfinite(bucket_scale) or bucket_scale <= 0.0:
            bucket_scale = 1.0
        bucket_scales[bucket] = float(bucket_scale)

    role_scales: dict[str, float] = {}
    for role_bucket in PROJECTION_ROLE_BUCKETS:
        raw_scale = None if projection_role_bucket_scales is None else projection_role_bucket_scales.get(role_bucket)
        role_scale = _safe_num(raw_scale, 1.0)
        if not math.isfinite(role_scale) or role_scale <= 0.0:
            role_scale = 1.0
        role_scales[role_bucket] = float(role_scale)

    row_bucket_scales = pd.Series(1.0, index=out.index, dtype="float64")
    if "Salary" in out.columns:
        salary_num = pd.to_numeric(out["Salary"], errors="coerce")
        salary_bucket = salary_num.map(projection_salary_bucket_key)
        row_bucket_scales = salary_bucket.map(lambda key: float(bucket_scales.get(str(key), 1.0))).fillna(1.0)
        out["projection_salary_bucket"] = salary_bucket

    row_role_scales = pd.Series(1.0, index=out.index, dtype="float64")
    role_source = "PositionBase" if "PositionBase" in out.columns else ("Position" if "Position" in out.columns else "")
    if role_source:
        role_bucket = out[role_source].map(projection_role_bucket_key)
        row_role_scales = role_bucket.map(lambda key: float(role_scales.get(str(key), 1.0))).fillna(1.0)
        out["projection_role_bucket"] = role_bucket

    total_row_scale = row_bucket_scales * row_role_scales * float(scale)
    for col in ["projected_dk_points", "blended_projection", "our_dk_projection", "vegas_dk_projection"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * total_row_scale

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
    out["projection_bucket_scale"] = row_bucket_scales
    out["projection_role_scale"] = row_role_scales
    out["projection_total_scale"] = total_row_scale
    return out


def apply_ownership_surprise_guardrails(
    pool_df: pd.DataFrame,
    projected_ownership_threshold: float = 10.0,
    surge_score_threshold: float = 72.0,
    projection_rank_threshold: float = 0.60,
    ownership_floor_base: float = 10.0,
    ownership_floor_cap: float = 24.0,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    proj_own = pd.to_numeric(out.get("projected_ownership"), errors="coerce").fillna(0.0)
    surge_score = pd.to_numeric(out.get("ownership_chalk_surge_score"), errors="coerce").fillna(0.0)
    projection = pd.to_numeric(out.get("projected_dk_points"), errors="coerce")
    projection_rank = projection.rank(method="average", pct=True).fillna(0.0)
    focus_score = pd.to_numeric(out.get("game_stack_focus_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    focus_flag = (
        out.get("game_stack_focus_flag", pd.Series([False] * len(out), index=out.index))
        .map(lambda x: bool(x))
        .astype(bool)
    )
    historical_baseline = pd.to_numeric(out.get("historical_ownership_baseline"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)

    own_threshold = max(0.0, float(projected_ownership_threshold))
    surge_threshold = max(0.0, min(100.0, float(surge_score_threshold)))
    proj_rank_threshold = max(0.0, min(1.0, float(projection_rank_threshold)))
    floor_base = max(0.0, float(ownership_floor_base))
    floor_cap = max(floor_base, float(ownership_floor_cap))

    base_candidate_mask = (
        (proj_own <= own_threshold)
        & (surge_score >= surge_threshold)
        & (projection_rank >= proj_rank_threshold)
    )
    focus_candidate_mask = (
        focus_flag
        & (projection_rank >= max(0.35, proj_rank_threshold - 0.18))
        & (focus_score >= 70.0)
        & (historical_baseline >= max(floor_base, own_threshold))
        & (proj_own <= max(own_threshold + 8.0, floor_base + 4.0))
    )
    candidate_mask = base_candidate_mask | focus_candidate_mask
    normalized_surge = ((surge_score - surge_threshold).clip(lower=0.0) / max(1.0, (100.0 - surge_threshold))).clip(0.0, 1.0)
    focus_norm = (focus_score / 100.0).clip(lower=0.0, upper=1.0)
    hist_norm = (historical_baseline / max(1.0, floor_cap)).clip(lower=0.0, upper=1.0)
    guardrail_strength = ((0.60 * normalized_surge) + (0.25 * focus_norm) + (0.15 * hist_norm)).clip(lower=0.0, upper=1.0)
    implied_floor = (floor_base + (guardrail_strength * (floor_cap - floor_base))).clip(lower=floor_base, upper=floor_cap)
    historical_floor = (historical_baseline * (0.88 + (0.12 * focus_norm))).clip(lower=0.0, upper=floor_cap)
    guardrail_target = pd.concat([implied_floor, historical_floor], axis=1).max(axis=1)
    guardrail_ownership = proj_own.where(~candidate_mask, proj_own.combine(guardrail_target, max))

    out["ownership_guardrail_flag"] = candidate_mask.map(lambda x: bool(x))
    out["ownership_guardrail_floor"] = implied_floor.round(3)
    out["ownership_guardrail_delta"] = (guardrail_ownership - proj_own).round(3)
    out["projected_ownership"] = guardrail_ownership.round(2)
    return _recompute_leverage_score(out)


def apply_focus_game_chalk_guardrail(
    pool_df: pd.DataFrame,
    projected_ownership_threshold: float = 18.0,
    projection_rank_threshold: float = 0.45,
    floor_base: float = 14.0,
    floor_cap: float = 32.0,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    projected_ownership = pd.to_numeric(out.get("projected_ownership"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    projection = pd.to_numeric(out.get("projected_dk_points"), errors="coerce")
    projection_rank = projection.rank(method="average", pct=True).fillna(0.0)
    focus_score = pd.to_numeric(out.get("game_stack_focus_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    focus_flag = (
        out.get("game_stack_focus_flag", pd.Series([False] * len(out), index=out.index))
        .map(lambda x: bool(x))
        .astype(bool)
    )
    historical_baseline = pd.to_numeric(out.get("historical_ownership_baseline"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    team_stack = pd.to_numeric(out.get("team_stack_popularity_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    surge = pd.to_numeric(out.get("ownership_chalk_surge_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    minutes_boost = pd.to_numeric(out.get("minutes_shock_boost_pct"), errors="coerce").fillna(0.0).clip(lower=0.0)
    if float(minutes_boost.max()) > 0.0:
        minutes_norm = (minutes_boost / float(minutes_boost.max())).clip(lower=0.0, upper=1.0)
    else:
        minutes_norm = pd.Series([0.0] * len(out), index=out.index, dtype="float64")

    focus_norm = (focus_score / 100.0).clip(lower=0.0, upper=1.0)
    historical_norm = (historical_baseline / max(1.0, float(floor_cap))).clip(lower=0.0, upper=1.0)
    team_stack_norm = (team_stack / 100.0).clip(lower=0.0, upper=1.0)
    surge_norm = (surge / 100.0).clip(lower=0.0, upper=1.0)

    support_score = (
        (0.36 * focus_norm)
        + (0.24 * historical_norm)
        + (0.18 * team_stack_norm)
        + (0.12 * surge_norm)
        + (0.10 * minutes_norm)
    ).clip(lower=0.0, upper=1.0)
    candidate_mask = (
        focus_flag
        & (projection_rank >= max(0.0, min(1.0, float(projection_rank_threshold))))
        & (historical_baseline >= floor_base)
        & (projected_ownership <= max(0.0, float(projected_ownership_threshold)))
    )
    target_floor = (floor_base + (support_score * (max(floor_base, float(floor_cap)) - floor_base))).clip(lower=floor_base, upper=floor_cap)
    historical_floor = (historical_baseline * (0.92 + (0.08 * focus_norm))).clip(lower=0.0, upper=floor_cap)
    adjusted_ownership = projected_ownership.copy()
    adjusted_ownership.loc[candidate_mask] = (
        pd.concat(
            [
                projected_ownership.loc[candidate_mask],
                target_floor.loc[candidate_mask],
                historical_floor.loc[candidate_mask],
            ],
            axis=1,
        )
        .max(axis=1)
        .clip(lower=0.0, upper=floor_cap)
    )

    out["focus_game_chalk_guardrail_flag"] = candidate_mask.astype(bool)
    out["focus_game_chalk_guardrail_target"] = target_floor.round(3)
    out["focus_game_chalk_guardrail_delta"] = (adjusted_ownership - projected_ownership).round(3)
    out["projected_ownership"] = adjusted_ownership.round(2)
    return _recompute_leverage_score(out)


def apply_false_chalk_discount(
    pool_df: pd.DataFrame,
    projected_ownership_floor: float = 14.0,
    max_discount_pct: float = 0.42,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    projected_ownership = pd.to_numeric(out.get("projected_ownership"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    projection = pd.to_numeric(out.get("projected_dk_points"), errors="coerce")
    salary = pd.to_numeric(out.get("Salary"), errors="coerce")
    value = pd.to_numeric(out.get("value_per_1k"), errors="coerce")
    surge = pd.to_numeric(out.get("ownership_chalk_surge_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    field_own = pd.to_numeric(
        out.get("field_ownership_pct", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    confidence = pd.to_numeric(
        out.get("ownership_confidence", pd.Series([0.50] * len(out), index=out.index)),
        errors="coerce",
    ).fillna(0.50).clip(lower=0.0, upper=1.0)
    minutes_boost = pd.to_numeric(out.get("minutes_shock_boost_pct"), errors="coerce").fillna(0.0).clip(lower=0.0)
    historical_baseline = pd.to_numeric(out.get("historical_ownership_baseline"), errors="coerce").fillna(0.0)
    focus_score = pd.to_numeric(out.get("game_stack_focus_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)

    proj_pct = _rank_pct_series(projection)
    value_pct = _rank_pct_series(value)
    field_pct = _rank_pct_series(field_own)
    hist_pct = _rank_pct_series(historical_baseline)
    if float(minutes_boost.max()) > 0.0:
        minutes_pct = (minutes_boost / float(minutes_boost.max())).clip(lower=0.0, upper=1.0)
    else:
        minutes_pct = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    surge_norm = (surge / 100.0).clip(lower=0.0, upper=1.0)
    mid_salary_bias = (1.0 - ((salary - 6700.0).abs() / 4200.0)).clip(lower=0.0, upper=1.0).fillna(0.0)
    focus_norm = (focus_score / 100.0).clip(lower=0.0, upper=1.0)

    support_score = (
        (0.30 * proj_pct)
        + (0.16 * surge_norm)
        + (0.12 * field_pct)
        + (0.10 * value_pct)
        + (0.08 * minutes_pct)
        + (0.06 * mid_salary_bias)
        + (0.10 * hist_pct)
        + (0.08 * focus_norm)
    ).clip(lower=0.0, upper=1.0)

    own_floor = max(0.0, float(projected_ownership_floor))
    max_discount = max(0.0, min(0.80, float(max_discount_pct)))
    high_own_pressure = ((projected_ownership - own_floor).clip(lower=0.0) / max(1.0, (32.0 - own_floor))).clip(0.0, 1.0)
    support_gap = (1.0 - support_score).clip(lower=0.0, upper=1.0)
    confidence_gap = (1.0 - confidence).clip(lower=0.0, upper=1.0)
    false_chalk_risk = (
        high_own_pressure
        * support_gap
        * (0.65 + (0.35 * confidence_gap))
    ).clip(lower=0.0, upper=1.0)

    baseline_ownership = float(projected_ownership.sum()) / float(max(1, len(out)))
    adjusted_ownership = baseline_ownership + (
        (projected_ownership - baseline_ownership)
        * (1.0 - (max_discount * false_chalk_risk))
    )
    adjusted_ownership = adjusted_ownership.clip(lower=0.0, upper=100.0)

    out["false_chalk_discount_score"] = false_chalk_risk.round(4)
    out["false_chalk_discount_flag"] = (
        (projected_ownership >= own_floor) & (false_chalk_risk >= 0.18)
    ).astype(bool)
    out["false_chalk_discount_delta"] = (projected_ownership - adjusted_ownership).round(3)
    out["projected_ownership"] = adjusted_ownership.round(2)
    return _recompute_leverage_score(out)


def _annotate_unsupported_false_chalk(
    pool_df: pd.DataFrame,
    preferred_games: set[str] | None = None,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        out["unsupported_false_chalk_flag"] = pd.Series(dtype="bool")
        out["unsupported_false_chalk_score"] = pd.Series(dtype="float64")
        return out

    preferred = {
        str(key or "").strip().upper().split(" ")[0]
        for key in (preferred_games or set())
        if str(key or "").strip()
    }
    false_chalk_flag = (
        out.get("false_chalk_discount_flag", pd.Series([False] * len(out), index=out.index))
        .map(lambda x: bool(x))
        .astype(bool)
    )
    focus_game_flag = (
        out.get("game_stack_focus_flag", pd.Series([False] * len(out), index=out.index))
        .map(lambda x: bool(x))
        .astype(bool)
    )
    focus_game_chalk_flag = (
        out.get("focus_game_chalk_guardrail_flag", pd.Series([False] * len(out), index=out.index))
        .map(lambda x: bool(x))
        .astype(bool)
    )
    if "game_key_norm" in out.columns:
        game_key_norm = out["game_key_norm"].astype(str).str.strip().str.upper().str.split().str[0]
    else:
        game_key_norm = (
            out.get("game_key", pd.Series([""] * len(out), index=out.index))
            .astype(str)
            .str.strip()
            .str.upper()
            .str.split()
            .str[0]
        )
    preferred_mask = game_key_norm.isin(preferred) if preferred else pd.Series([False] * len(out), index=out.index)
    supported_mask = (preferred_mask | focus_game_flag | focus_game_chalk_flag).astype(bool)
    false_chalk_score = pd.to_numeric(out.get("false_chalk_discount_score"), errors="coerce").fillna(0.0).clip(lower=0.0)
    unsupported_mask = (false_chalk_flag & ~supported_mask).astype(bool)

    out["unsupported_false_chalk_flag"] = unsupported_mask
    out["unsupported_false_chalk_score"] = false_chalk_score.where(unsupported_mask, 0.0).round(4)
    return out


def apply_projection_uncertainty_adjustment(
    pool_df: pd.DataFrame,
    uncertainty_weight: float = 0.18,
    high_risk_extra_shrink: float = 0.10,
    dnp_risk_threshold: float = 0.30,
    min_multiplier: float = 0.68,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    weight = max(0.0, min(0.60, _safe_num(uncertainty_weight, 0.18)))
    extra = max(0.0, min(0.40, _safe_num(high_risk_extra_shrink, 0.10)))
    risk_threshold = max(0.0, min(0.95, _safe_num(dnp_risk_threshold, 0.30)))
    floor_mult = max(0.50, min(1.0, _safe_num(min_multiplier, 0.68)))

    uncertainty = pd.to_numeric(out.get("projection_uncertainty_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    dnp_risk = pd.to_numeric(out.get("dnp_risk_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    if risk_threshold >= 1.0:
        high_risk_component = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    else:
        high_risk_component = ((dnp_risk - risk_threshold).clip(lower=0.0) / (1.0 - risk_threshold)).clip(0.0, 1.0)

    multiplier = 1.0 - (weight * uncertainty) - (extra * high_risk_component)
    multiplier = multiplier.clip(lower=floor_mult, upper=1.0)
    out["projection_uncertainty_multiplier"] = multiplier.round(4)

    for col in ["projected_dk_points", "blended_projection", "our_dk_projection", "vegas_dk_projection"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * multiplier

    if "Salary" in out.columns and "projected_dk_points" in out.columns:
        salary = pd.to_numeric(out["Salary"], errors="coerce")
        proj = pd.to_numeric(out["projected_dk_points"], errors="coerce")
        out["projection_per_dollar"] = proj / salary.replace(0, pd.NA)
        out["value_per_1k"] = out["projection_per_dollar"] * 1000.0

    if "projected_dk_points" in out.columns and "projected_ownership" in out.columns:
        proj = pd.to_numeric(out["projected_dk_points"], errors="coerce").fillna(0.0)
        own = pd.to_numeric(out["projected_ownership"], errors="coerce").fillna(0.0)
        out["leverage_score"] = (proj - (0.15 * own)).round(3)
    return out


def apply_minutes_shock_override(
    pool_df: pd.DataFrame,
    max_boost_pct: float = 0.18,
    boost_per_minute: float = 0.025,
    expected_rotation_bonus_minutes: float = 2.0,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    boost_cap = max(0.0, min(0.30, _safe_num(max_boost_pct, 0.18)))
    per_minute = max(0.0, min(0.05, _safe_num(boost_per_minute, 0.025)))
    rotation_bonus = max(0.0, min(4.0, _safe_num(expected_rotation_bonus_minutes, 2.0)))

    mins_avg = pd.to_numeric(out.get("our_minutes_avg"), errors="coerce")
    mins_last7 = pd.to_numeric(out.get("our_minutes_last7"), errors="coerce")
    mins_recent = pd.to_numeric(out.get("our_minutes_recent"), errors="coerce")
    mins_recent = mins_recent.where(mins_recent.notna(), mins_last7)
    mins_recent = mins_recent.where(mins_recent.notna(), mins_avg)
    mins_avg = mins_avg.where(mins_avg.notna(), mins_last7)
    mins_avg = mins_avg.where(mins_avg.notna(), mins_recent)
    mins_avg = mins_avg.fillna(0.0)

    uncertainty = pd.to_numeric(out.get("projection_uncertainty_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    dnp_risk = pd.to_numeric(out.get("dnp_risk_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    surge = pd.to_numeric(out.get("ownership_chalk_surge_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)

    minutes_drop = pd.to_numeric(out.get("minutes_drop_ratio"), errors="coerce")
    if minutes_drop.isna().all():
        minutes_drop = ((mins_avg - mins_recent).clip(lower=0.0) / mins_avg.replace(0.0, pd.NA)).fillna(0.0)
    minutes_drop = minutes_drop.clip(lower=0.0, upper=1.0)

    expected_rotation_flag = (
        (mins_recent >= 22.0)
        & (dnp_risk <= 0.42)
        & ((minutes_drop <= 0.25) | (surge >= 68.0))
    )
    expected_rotation_minutes = mins_avg + (expected_rotation_flag.astype(float) * rotation_bonus)

    minutes_anchor = pd.concat([mins_recent, mins_last7, expected_rotation_minutes], axis=1).max(axis=1, skipna=True)
    minutes_anchor = minutes_anchor.fillna(mins_avg)
    minutes_delta = (minutes_anchor - mins_avg).clip(lower=0.0, upper=10.0)
    raw_boost = (minutes_delta * per_minute).clip(lower=0.0, upper=boost_cap)
    reliability = (1.0 - (0.55 * dnp_risk) - (0.35 * uncertainty)).clip(lower=0.35, upper=1.0)
    minutes_boost = (raw_boost * reliability).clip(lower=0.0, upper=boost_cap)
    multiplier = (1.0 + minutes_boost).clip(lower=1.0, upper=1.0 + boost_cap)

    for col in ["projected_dk_points", "blended_projection", "our_dk_projection", "vegas_dk_projection"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * multiplier

    if "Salary" in out.columns and "projected_dk_points" in out.columns:
        salary = pd.to_numeric(out["Salary"], errors="coerce")
        proj = pd.to_numeric(out["projected_dk_points"], errors="coerce")
        out["projection_per_dollar"] = proj / salary.replace(0, pd.NA)
        out["value_per_1k"] = out["projection_per_dollar"] * 1000.0

    if "projected_dk_points" in out.columns and "projected_ownership" in out.columns:
        proj = pd.to_numeric(out["projected_dk_points"], errors="coerce").fillna(0.0)
        own = pd.to_numeric(out["projected_ownership"], errors="coerce").fillna(0.0)
        out["leverage_score"] = (proj - (0.15 * own)).round(3)

    out["expected_rotation_flag"] = expected_rotation_flag.astype(bool)
    out["minutes_shock_delta"] = minutes_delta.round(3)
    out["minutes_shock_boost_pct"] = (minutes_boost * 100.0).round(3)
    out["minutes_shock_multiplier"] = multiplier.round(4)
    return out


def apply_chalk_ceiling_guardrail(
    pool_df: pd.DataFrame,
    max_players: int = 3,
    min_floor: float = 18.0,
    max_floor: float = 24.0,
    blend_weight: float = 0.45,
    score_threshold: float = 0.58,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    projected_ownership = pd.to_numeric(out.get("projected_ownership"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    projection = pd.to_numeric(out.get("projected_dk_points"), errors="coerce")
    salary = pd.to_numeric(out.get("Salary"), errors="coerce")
    surge = pd.to_numeric(out.get("ownership_chalk_surge_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)
    value = pd.to_numeric(out.get("value_per_1k"), errors="coerce")
    field_own = pd.to_numeric(
        out.get("field_ownership_pct", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    minutes_boost = pd.to_numeric(out.get("minutes_shock_boost_pct"), errors="coerce").fillna(0.0).clip(lower=0.0)
    historical_baseline = pd.to_numeric(out.get("historical_ownership_baseline"), errors="coerce").fillna(0.0)
    focus_score = pd.to_numeric(out.get("game_stack_focus_score"), errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0)

    proj_pct = _rank_pct_series(projection)
    value_pct = _rank_pct_series(value)
    field_pct = _rank_pct_series(field_own)
    hist_pct = _rank_pct_series(historical_baseline)
    surge_norm = (surge / 100.0).clip(lower=0.0, upper=1.0)
    if float(minutes_boost.max()) > 0.0:
        minutes_norm = (minutes_boost / float(minutes_boost.max())).clip(lower=0.0, upper=1.0)
    else:
        minutes_norm = pd.Series([0.0] * len(out), index=out.index, dtype="float64")
    mid_salary_score = (1.0 - ((salary - 6500.0).abs() / 4000.0)).clip(lower=0.0, upper=1.0).fillna(0.0)
    focus_norm = (focus_score / 100.0).clip(lower=0.0, upper=1.0)

    archetype_score = (
        (0.30 * proj_pct)
        + (0.18 * value_pct)
        + (0.14 * surge_norm)
        + (0.12 * minutes_norm)
        + (0.08 * mid_salary_score)
        + (0.08 * field_pct)
        + (0.06 * hist_pct)
        + (0.04 * focus_norm)
    ).clip(lower=0.0, upper=1.0)

    floor_min = max(0.0, float(min_floor))
    floor_max = max(floor_min, float(max_floor))
    blend = max(0.0, min(1.0, float(blend_weight)))
    threshold = max(0.0, min(1.0, float(score_threshold)))
    projected_floor = (floor_min + ((floor_max - floor_min) * archetype_score)).clip(lower=floor_min, upper=floor_max)
    projection_cut = float(projection.quantile(0.55)) if projection.notna().any() else 0.0

    candidate_mask = (
        (projected_ownership < projected_floor)
        & (archetype_score >= threshold)
        & (projection.fillna(0.0) >= projection_cut)
    )
    limit = max(0, min(int(max_players), len(out)))
    selected_mask = pd.Series(False, index=out.index, dtype="bool")
    if limit > 0:
        candidate_df = pd.DataFrame(
            {
                "score": archetype_score,
            },
            index=out.index,
        )
        chosen_idx = (
            candidate_df.loc[candidate_mask]
            .sort_values("score", ascending=False)
            .head(limit)
            .index
        )
        selected_mask.loc[chosen_idx] = True

    ownership_delta = (projected_floor - projected_ownership).clip(lower=0.0)
    adjusted_ownership = projected_ownership.copy()
    adjusted_ownership.loc[selected_mask] = (
        projected_ownership.loc[selected_mask]
        + (blend * ownership_delta.loc[selected_mask])
    ).clip(lower=0.0, upper=100.0)

    out["chalk_ceiling_guardrail_flag"] = selected_mask.astype(bool)
    out["chalk_ceiling_guardrail_target"] = projected_floor.round(3)
    out["chalk_ceiling_guardrail_delta"] = (adjusted_ownership - projected_ownership).round(3)
    out["projected_ownership"] = adjusted_ownership.round(2)
    return _recompute_leverage_score(out)


def apply_stack_anchor_bias(
    pool_df: pd.DataFrame,
    top_game_count: int = 2,
    objective_bonus: float = 0.85,
) -> pd.DataFrame:
    out = pool_df.copy()
    if out.empty:
        return out

    if "objective_score" not in out.columns:
        out["stack_anchor_focus_flag"] = False
        return out

    game_key = out.get("game_key")
    if game_key is None:
        out["stack_anchor_focus_flag"] = False
        return out

    game_norm = game_key.map(lambda x: str(x or "").strip().upper())
    focus_score = pd.to_numeric(
        out.get("game_stack_focus_score", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    totals = pd.to_numeric(
        out.get("game_total_line", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    )
    tail = pd.to_numeric(
        out.get("game_tail_score", pd.Series([pd.NA] * len(out), index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    if focus_score.notna().any():
        game_metric = focus_score.groupby(game_norm).mean()
    elif totals.notna().any():
        game_metric = totals.groupby(game_norm).median()
    elif tail.notna().any():
        game_metric = tail.groupby(game_norm).mean()
    else:
        out["stack_anchor_focus_flag"] = False
        return out

    game_metric = game_metric.loc[game_metric.index != ""].sort_values(ascending=False)
    if game_metric.empty:
        out["stack_anchor_focus_flag"] = False
        return out

    top_n = max(1, min(int(top_game_count), len(game_metric)))
    focus_games = set(game_metric.head(top_n).index.tolist())
    focus_mask = game_norm.isin(focus_games)
    if not bool(focus_mask.any()):
        out["stack_anchor_focus_flag"] = False
        return out

    bonus_base = max(0.0, float(objective_bonus))
    tail_bonus = (tail / 100.0).clip(lower=0.0, upper=1.0) * 0.30
    bonus = focus_mask.map(lambda x: bonus_base if bool(x) else 0.0).astype(float) + (
        focus_mask.astype(float) * tail_bonus
    )
    out["objective_score"] = (
        pd.to_numeric(out.get("objective_score"), errors="coerce").fillna(0.0)
        + bonus
    ).clip(lower=0.01)
    out["stack_anchor_focus_flag"] = focus_mask.astype(bool)
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


def _preferred_game_stack_size(players: list[dict[str, Any]], preferred_games: set[str]) -> int:
    if not players or not preferred_games:
        return 0
    counts: Counter[str] = Counter(
        str(p.get("game_key") or "").strip().upper()
        for p in players
        if str(p.get("game_key") or "").strip().upper() in preferred_games
    )
    return int(max(counts.values(), default=0))


def _preferred_game_stack_total(players: list[dict[str, Any]], preferred_games: set[str]) -> int:
    if not players or not preferred_games:
        return 0
    return int(
        sum(1 for p in players if str(p.get("game_key") or "").strip().upper() in preferred_games)
    )


def _can_still_hit_preferred_stack(
    selected: list[dict[str, Any]],
    candidate_game_key: str,
    preferred_games: set[str],
    required_count: int,
    remaining_player_count_after_pick: int,
) -> bool:
    required = max(0, int(required_count))
    if required <= 0 or not preferred_games:
        return True
    counts: Counter[str] = Counter(
        str(p.get("game_key") or "").strip().upper()
        for p in selected
        if str(p.get("game_key") or "").strip().upper() in preferred_games
    )
    normalized_candidate = str(candidate_game_key or "").strip().upper()
    if normalized_candidate in preferred_games:
        counts[normalized_candidate] += 1
    current_best = max(counts.values(), default=0)
    possible_best = current_best + max(0, int(remaining_player_count_after_pick))
    return bool(possible_best >= required)


def _distributed_lineup_indices(
    total_lineups: int,
    target_count: int,
    rng: random.Random,
    buckets: int = 10,
) -> set[int]:
    total = max(0, int(total_lineups))
    target = max(0, min(total, int(target_count)))
    if total <= 0 or target <= 0:
        return set()
    bucket_count = max(1, min(int(buckets), total))
    bucket_size = float(total) / float(bucket_count)
    bucket_slices: list[list[int]] = []
    for bucket_idx in range(bucket_count):
        start = int(math.floor(bucket_idx * bucket_size))
        end = int(math.floor((bucket_idx + 1) * bucket_size))
        if bucket_idx == (bucket_count - 1):
            end = total
        if end <= start:
            end = min(total, start + 1)
        indices = list(range(start, end))
        if indices:
            bucket_slices.append(indices)

    selected: set[int] = set()
    passes = max(1, int(math.ceil(float(target) / max(1, len(bucket_slices)))))
    for _ in range(passes):
        for bucket in bucket_slices:
            if len(selected) >= target:
                break
            choices = [idx for idx in bucket if idx not in selected]
            if choices:
                selected.add(rng.choice(choices))
        if len(selected) >= target:
            break

    if len(selected) < target:
        remaining = [idx for idx in range(total) if idx not in selected]
        if remaining:
            selected.update(rng.sample(remaining, k=min(len(remaining), target - len(selected))))
    return selected


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
                "focus": [],
                "total": [],
                "spread_abs": [],
            },
        )
        bucket["projected"].append(_safe_num(p.get("projected_dk_points")))
        bucket["ownership"].append(_safe_num(p.get("projected_ownership")))
        bucket["tail"].append(_safe_num(p.get("game_tail_score")))
        bucket["focus"].append(_safe_num(p.get("game_stack_focus_score")))
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
        focus = values["focus"] or [0.0]
        totals = values["total"]
        spreads = values["spread_abs"]
        metric_rows.append(
            {
                "game_key": game_key,
                "projected_mean": float(sum(projected) / len(projected)),
                "ownership_mean": float(sum(ownership) / len(ownership)),
                "tail_mean": float(sum(tail) / len(tail)),
                "focus_mean": float(sum(focus) / len(focus)),
                "total_mean": float(sum(totals) / len(totals)) if totals else 0.0,
                "spread_abs_mean": float(sum(spreads) / len(spreads)) if spreads else 999.0,
            }
        )

    metrics = pd.DataFrame(metric_rows)
    metrics["leverage_score"] = metrics["tail_mean"] + (0.15 * metrics["focus_mean"]) - (0.35 * metrics["ownership_mean"])
    metrics["contrarian_score"] = metrics["projected_mean"] + (0.10 * metrics["focus_mean"]) - (0.5 * metrics["ownership_mean"])

    unique_games = int(len(metrics))
    desired_clusters = max(1, int(math.ceil(float(num_lineups) / max(1, int(variants_per_cluster)))))
    cluster_count = max(1, min(int(target_clusters), unique_games, int(num_lineups), desired_clusters))

    pools: dict[str, list[str]] = {
        "high_total": (
            metrics.sort_values(["total_mean", "focus_mean", "projected_mean"], ascending=[False, False, False])["game_key"].tolist()
        ),
        "tight_spread": (
            metrics.sort_values(["spread_abs_mean", "focus_mean", "total_mean"], ascending=[True, False, False])["game_key"].tolist()
        ),
        "tail_leverage": (
            metrics.sort_values(["leverage_score", "focus_mean", "tail_mean"], ascending=[False, False, False])["game_key"].tolist()
        ),
        "contrarian": (
            metrics.sort_values(["ownership_mean", "focus_mean", "contrarian_score"], ascending=[True, False, False])["game_key"].tolist()
        ),
        "balanced": (
            metrics.sort_values(["focus_mean", "projected_mean", "tail_mean"], ascending=[False, False, False])["game_key"].tolist()
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
    cluster_target_count: int = 15,
    cluster_variants_per_cluster: int = 10,
    projection_scale: float = 1.0,
    projection_salary_bucket_scales: dict[str, float] | None = None,
    projection_role_bucket_scales: dict[str, float] | None = None,
    apply_ownership_guardrails: bool = False,
    ownership_guardrail_projected_threshold: float = 10.0,
    ownership_guardrail_surge_threshold: float = 72.0,
    ownership_guardrail_projection_rank_threshold: float = 0.60,
    ownership_guardrail_floor_base: float = 10.0,
    ownership_guardrail_floor_cap: float = 24.0,
    apply_uncertainty_shrink: bool = False,
    uncertainty_weight: float = 0.18,
    high_risk_extra_shrink: float = 0.10,
    dnp_risk_threshold: float = 0.30,
    uncertainty_min_multiplier: float = 0.68,
    low_own_bucket_exposure_pct: float = 0.0,
    low_own_bucket_min_per_lineup: int = 1,
    low_own_bucket_max_projected_ownership: float = 10.0,
    low_own_bucket_min_projection: float = 24.0,
    low_own_bucket_min_tail_score: float = 55.0,
    low_own_bucket_objective_bonus: float = 0.0,
    preferred_game_keys: list[str] | None = None,
    preferred_game_bonus: float = 0.0,
    preferred_game_stack_lineup_pct: float = 0.0,
    preferred_game_stack_min_players: int = 0,
    auto_preferred_game_count: int = 2,
    max_unsupported_false_chalk_per_lineup: int | None = None,
    ceiling_boost_lineup_pct: float = 0.0,
    ceiling_boost_stack_bonus: float = 0.0,
    ceiling_boost_salary_left_target: int | None = None,
    objective_score_adjustments: dict[str, float] | None = None,
    salary_left_target: int | None = 50,
    salary_left_penalty_divisor: float = 75.0,
    random_seed: int = 7,
    max_attempts_per_lineup: int = 1200,
    model_profile: str = "legacy_baseline",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    if pool_df.empty:
        return [], ["Player pool is empty."]
    if num_lineups <= 0:
        return [], ["Number of lineups must be > 0."]

    if progress_callback is not None:
        progress_callback(0, num_lineups, "Starting lineup generation...")

    calibrated = apply_projection_calibration(
        pool_df,
        projection_scale=projection_scale,
        projection_salary_bucket_scales=projection_salary_bucket_scales,
        projection_role_bucket_scales=projection_role_bucket_scales,
    )
    calibrated = apply_minutes_shock_override(calibrated)
    if apply_uncertainty_shrink:
        calibrated = apply_projection_uncertainty_adjustment(
            calibrated,
            uncertainty_weight=uncertainty_weight,
            high_risk_extra_shrink=high_risk_extra_shrink,
            dnp_risk_threshold=dnp_risk_threshold,
            min_multiplier=uncertainty_min_multiplier,
        )
    if apply_ownership_guardrails:
        calibrated = apply_ownership_surprise_guardrails(
            calibrated,
            projected_ownership_threshold=ownership_guardrail_projected_threshold,
            surge_score_threshold=ownership_guardrail_surge_threshold,
            projection_rank_threshold=ownership_guardrail_projection_rank_threshold,
            ownership_floor_base=ownership_guardrail_floor_base,
            ownership_floor_cap=ownership_guardrail_floor_cap,
        )
    calibrated = apply_focus_game_chalk_guardrail(calibrated)
    calibrated = apply_chalk_ceiling_guardrail(calibrated)
    calibrated = apply_false_chalk_discount(calibrated)
    calibrated = normalize_projected_ownership_total(calibrated)
    scored = apply_contest_objective(calibrated, contest_type, include_tail_signals=include_tail_signals)
    scored = apply_stack_anchor_bias(scored)
    scored = apply_model_profile_adjustments(scored, model_profile=model_profile)
    scored = scored.loc[scored["Salary"] > 0].copy()
    if objective_score_adjustments:
        objective_bonus = scored["ID"].astype(str).map(
            lambda pid: _safe_num(objective_score_adjustments.get(str(pid), 0.0), 0.0)
        )
        scored["objective_score"] = (
            pd.to_numeric(scored.get("objective_score"), errors="coerce").fillna(0.0)
            + pd.to_numeric(objective_bonus, errors="coerce").fillna(0.0)
        ).clip(lower=0.01)
        scored["objective_score_adjustment"] = pd.to_numeric(objective_bonus, errors="coerce").fillna(0.0)

    preferred_games = {
        str(key or "").strip().upper().split(" ")[0]
        for key in (preferred_game_keys or [])
        if str(key or "").strip()
    }
    preferred_game_bonus_value = max(0.0, float(preferred_game_bonus))
    preferred_game_stack_lineup_pct_value = max(0.0, min(100.0, float(preferred_game_stack_lineup_pct)))
    preferred_game_stack_min_players_value = max(0, min(ROSTER_SIZE, int(preferred_game_stack_min_players)))
    focus_settings = recommended_focus_stack_settings(
        scored,
        contest_type,
        focus_game_count=max(0, int(auto_preferred_game_count)),
    )
    if (
        not preferred_games
        and max(0, int(auto_preferred_game_count)) > 0
        and (
            preferred_game_bonus_value > 0.0
            or preferred_game_stack_lineup_pct_value > 0.0
            or preferred_game_stack_min_players_value > 0
            or max_unsupported_false_chalk_per_lineup is not None
        )
    ):
        preferred_games = {
            str(key or "").strip().upper()
            for key in (focus_settings.get("preferred_game_keys") or [])
            if str(key or "").strip()
        }
    if preferred_games and preferred_game_stack_lineup_pct_value > 0.0 and preferred_game_stack_min_players_value <= 0:
        preferred_game_stack_min_players_value = int(focus_settings.get("min_players") or 0)
    if "game_key" in scored.columns:
        scored["game_key_norm"] = scored["game_key"].map(lambda x: str(x or "").strip().upper().split(" ")[0])
    if preferred_games and preferred_game_bonus_value > 0.0 and "game_key_norm" in scored.columns:
        preferred_mask = scored["game_key_norm"].isin(preferred_games)
        scored["objective_score"] = (
            pd.to_numeric(scored.get("objective_score"), errors="coerce").fillna(0.0)
            + preferred_mask.map(lambda x: preferred_game_bonus_value if bool(x) else 0.0)
        ).clip(lower=0.01)
        scored["preferred_game_flag"] = preferred_mask.map(lambda x: bool(x))
    else:
        scored["preferred_game_flag"] = False
    scored = _annotate_unsupported_false_chalk(scored, preferred_games)

    warnings: list[str] = []
    if preferred_games and preferred_game_stack_lineup_pct_value > 0.0 and preferred_game_stack_min_players_value > 0:
        warnings.append(
            "Focus-game stack guardrails active: "
            f"{preferred_game_stack_lineup_pct_value:.0f}% of lineups require >= {preferred_game_stack_min_players_value} players "
            f"from one of {', '.join(sorted(preferred_games))}."
        )
    unsupported_false_chalk_cap = None
    if max_unsupported_false_chalk_per_lineup is not None:
        unsupported_false_chalk_cap = max(0, min(ROSTER_SIZE, int(max_unsupported_false_chalk_per_lineup)))
        if unsupported_false_chalk_cap >= ROSTER_SIZE:
            unsupported_false_chalk_cap = None
    ownership_reliability = _ownership_reliability_for_pool(scored)
    low_own_candidate_ids: set[str] = set()
    requested_low_own_exposure = max(0.0, min(100.0, float(low_own_bucket_exposure_pct)))
    low_own_exposure = requested_low_own_exposure
    low_own_min_required = max(0, min(ROSTER_SIZE, int(low_own_bucket_min_per_lineup)))
    requested_low_own_min_required = low_own_min_required
    low_own_ownership_cap_input = max(0.0, float(low_own_bucket_max_projected_ownership))
    low_own_projection_floor_input = max(0.0, float(low_own_bucket_min_projection))
    low_own_tail_floor_input = max(0.0, float(low_own_bucket_min_tail_score))
    low_own_bonus = max(0.0, float(low_own_bucket_objective_bonus))
    if requested_low_own_exposure > 0.0:
        low_own_exposure = max(0.0, min(100.0, requested_low_own_exposure * ownership_reliability))
        if ownership_reliability < 0.55:
            low_own_min_required = min(low_own_min_required, 1)
        low_own_bonus = low_own_bonus * ownership_reliability
        if (
            abs(low_own_exposure - requested_low_own_exposure) > 1e-6
            or low_own_min_required != requested_low_own_min_required
        ):
            warnings.append(
                "Ownership reliability mitigation reduced low-owned forcing from "
                f"{requested_low_own_exposure:.0f}% to {low_own_exposure:.0f}% "
                f"(min per lineup {requested_low_own_min_required} -> {low_own_min_required}, "
                f"slate reliability={ownership_reliability:.2f})."
            )
    if low_own_exposure > 0.0:
        proj = pd.to_numeric(scored.get("projected_dk_points"), errors="coerce").fillna(0.0)
        own = pd.to_numeric(scored.get("projected_ownership"), errors="coerce").fillna(100.0)
        leverage = pd.to_numeric(scored.get("leverage_score"), errors="coerce")
        tail = pd.to_numeric(scored.get("game_tail_score"), errors="coerce").fillna(0.0)
        minutes_boost = pd.to_numeric(scored.get("minutes_shock_boost_pct"), errors="coerce").fillna(0.0).clip(lower=0.0)
        leverage_rank = leverage.rank(method="average", pct=True).fillna(0.0)
        tail_rank = tail.rank(method="average", pct=True).fillna(0.0)
        minutes_rank = minutes_boost.rank(method="average", pct=True).fillna(0.0)
        low_own_upside_score = (
            (0.46 * tail_rank)
            + (0.34 * leverage_rank)
            + (0.20 * minutes_rank)
        ).clip(lower=0.0, upper=1.0)
        dynamic_projection_floor = float(proj.quantile(0.42)) if proj.notna().any() else low_own_projection_floor_input
        effective_projection_floor = max(
            8.0,
            min(low_own_projection_floor_input, dynamic_projection_floor if dynamic_projection_floor > 0.0 else low_own_projection_floor_input),
        )
        effective_ownership_cap = max(low_own_ownership_cap_input, 14.0 + (3.0 * (1.0 - ownership_reliability)))
        effective_tail_floor = min(low_own_tail_floor_input, 50.0)
        low_own_mask = (
            (own <= effective_ownership_cap)
            & (
                (proj >= effective_projection_floor)
                | (low_own_upside_score >= 0.70)
            )
            & (
                (tail >= effective_tail_floor)
                | (leverage_rank >= 0.62)
                | (minutes_boost >= 6.0)
            )
        )
        scored["low_own_upside_v2_score"] = low_own_upside_score.round(4)
        scored["low_own_upside_flag"] = low_own_mask.map(lambda x: bool(x))
        low_own_candidate_ids = set(scored.loc[low_own_mask, "ID"].astype(str).tolist())
        if low_own_bonus > 0.0 and low_own_candidate_ids:
            low_own_score_by_id = (
                scored.loc[scored["ID"].astype(str).isin(low_own_candidate_ids), ["ID", "low_own_upside_v2_score"]]
                .drop_duplicates(subset=["ID"], keep="first")
                .set_index("ID")["low_own_upside_v2_score"]
                .to_dict()
            )
            candidate_bonus = scored["ID"].astype(str).map(
                lambda pid: low_own_bonus * (0.75 + float(low_own_score_by_id.get(pid, 0.0)))
                if pid in low_own_candidate_ids
                else 0.0
            )
            scored["objective_score"] = (
                pd.to_numeric(scored.get("objective_score"), errors="coerce").fillna(0.0)
                + pd.to_numeric(candidate_bonus, errors="coerce").fillna(0.0)
            ).clip(lower=0.01)
    else:
        scored["low_own_upside_flag"] = False
        scored["low_own_upside_v2_score"] = 0.0

    unsupported_false_chalk_ids: set[str] = set(
        scored.loc[scored["unsupported_false_chalk_flag"].astype(bool), "ID"].astype(str).tolist()
    )
    if unsupported_false_chalk_cap is not None:
        if unsupported_false_chalk_ids:
            warnings.append(
                "Non-focus false-chalk cap active: "
                f"max {unsupported_false_chalk_cap} unsupported high-owned plays per lineup "
                f"across {len(unsupported_false_chalk_ids)} tagged candidates."
            )
        else:
            warnings.append(
                "Non-focus false-chalk cap was enabled, but no unsupported false-chalk candidates were tagged on this slate."
            )

    if preferred_games and preferred_game_stack_lineup_pct_value > 0.0 and preferred_game_stack_min_players_value <= 0:
        preferred_game_stack_min_players_value = 2

    min_salary_used = SALARY_CAP - int(max(0, max_salary_left if max_salary_left is not None else SALARY_CAP))
    salary_target = None if salary_left_target is None else max(0, min(SALARY_CAP, int(salary_left_target)))
    ceiling_salary_target = (
        salary_target
        if ceiling_boost_salary_left_target is None
        else max(0, min(SALARY_CAP, int(ceiling_boost_salary_left_target)))
    )
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
    if unsupported_false_chalk_cap is not None:
        locked_unsupported_false_chalk = sum(
            1 for p in lock_players if str(p.get("ID")) in unsupported_false_chalk_ids
        )
        if locked_unsupported_false_chalk > unsupported_false_chalk_cap:
            return [], [
                "Locked players exceed the non-focus false-chalk cap "
                f"({locked_unsupported_false_chalk} > {unsupported_false_chalk_cap})."
            ]

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

    if low_own_exposure > 0.0 and low_own_min_required > 0 and not low_own_candidate_ids:
        warnings.append(
            "Low-owned upside bucket was enabled, but no candidates met the filters; generation continued without bucket constraints."
        )

    target_low_own_lineups = int(round((low_own_exposure / 100.0) * float(num_lineups)))
    target_low_own_lineups = max(0, min(num_lineups, target_low_own_lineups))
    low_own_required_indices: set[int] = set()
    if target_low_own_lineups > 0 and low_own_min_required > 0 and low_own_candidate_ids:
        low_own_required_indices = _distributed_lineup_indices(
            total_lineups=num_lineups,
            target_count=target_low_own_lineups,
            rng=rng,
            buckets=10,
        )

    target_preferred_stack_lineups = int(round((preferred_game_stack_lineup_pct_value / 100.0) * float(num_lineups)))
    target_preferred_stack_lineups = max(0, min(num_lineups, target_preferred_stack_lineups))
    preferred_stack_required_indices: set[int] = set()
    if (
        target_preferred_stack_lineups > 0
        and preferred_game_stack_min_players_value > 0
        and preferred_games
    ):
        preferred_stack_required_indices = _distributed_lineup_indices(
            total_lineups=num_lineups,
            target_count=target_preferred_stack_lineups,
            rng=rng,
            buckets=10,
        )

    ceiling_pct = max(0.0, min(100.0, float(ceiling_boost_lineup_pct)))
    target_ceiling_lineups = int(round((ceiling_pct / 100.0) * float(num_lineups)))
    target_ceiling_lineups = max(0, min(num_lineups, target_ceiling_lineups))
    ceiling_boost_indices: set[int] = set()
    if target_ceiling_lineups > 0:
        ceiling_boost_indices = set(rng.sample(list(range(num_lineups)), k=target_ceiling_lineups))

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
        "our_minutes_avg",
        "our_minutes_last7",
        "our_minutes_last3",
        "our_minutes_recent",
        "game_tail_event_id",
        "game_tail_match_score",
        "game_p_plus_8",
        "game_p_plus_12",
        "game_volatility_score",
        "game_tail_to_ownership",
        "game_tail_score",
        "projection_calibration_scale",
        "projection_bucket_scale",
        "projection_role_scale",
        "projection_total_scale",
        "projection_salary_bucket",
        "projection_role_bucket",
        "projection_uncertainty_score",
        "dnp_risk_score",
        "projection_uncertainty_multiplier",
        "field_ownership_pct",
        "historical_ownership_avg",
        "historical_ownership_last5",
        "historical_ownership_samples",
        "historical_ownership_baseline",
        "ownership_prior_source",
        "historical_ownership_used_in_prior",
        "ownership_chalk_surge_score",
        "ownership_chalk_surge_flag",
        "ownership_confidence",
        "game_stack_focus_score",
        "game_stack_focus_rank",
        "game_stack_focus_flag",
        "recommended_stack_size",
        "ownership_guardrail_flag",
        "ownership_guardrail_floor",
        "ownership_guardrail_delta",
        "focus_game_chalk_guardrail_flag",
        "focus_game_chalk_guardrail_target",
        "focus_game_chalk_guardrail_delta",
        "chalk_ceiling_guardrail_flag",
        "chalk_ceiling_guardrail_target",
        "chalk_ceiling_guardrail_delta",
        "false_chalk_discount_score",
        "false_chalk_discount_flag",
        "false_chalk_discount_delta",
        "unsupported_false_chalk_flag",
        "unsupported_false_chalk_score",
        "low_own_upside_flag",
        "low_own_upside_v2_score",
        "preferred_game_flag",
        "stack_anchor_focus_flag",
        "expected_rotation_flag",
        "minutes_shock_delta",
        "minutes_shock_boost_pct",
        "minutes_shock_multiplier",
        "model_profile_bonus",
        "model_profile_focus_flag",
    ]
    model_profile_value = str(model_profile or "legacy_baseline")
    strategy_norm = str(lineup_strategy or "standard").strip().lower()
    spike_mode = strategy_norm in {"spike", "lineup spike", "spike pairs", "lineup_spike", "lineup_spike_pairs"}
    cluster_mode = strategy_norm in {
        "cluster",
        "cluster_v1",
        "cluster_v1_experimental",
        "cluster_mutation",
        "cluster_mutation_v1",
    }

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
            low_own_candidate_ids=low_own_candidate_ids,
            low_own_min_per_lineup=low_own_min_required,
            low_own_required_indices=low_own_required_indices,
            unsupported_false_chalk_ids=unsupported_false_chalk_ids,
            max_unsupported_false_chalk_per_lineup=unsupported_false_chalk_cap,
            preferred_game_keys=preferred_games,
            preferred_game_bonus=preferred_game_bonus_value,
            preferred_game_stack_min_players=preferred_game_stack_min_players_value,
            preferred_stack_required_indices=preferred_stack_required_indices,
            ceiling_boost_indices=ceiling_boost_indices,
            ceiling_boost_stack_bonus=max(0.0, float(ceiling_boost_stack_bonus)),
            ceiling_boost_salary_left_target=ceiling_salary_target,
            model_profile=model_profile_value,
        )

    for lineup_idx in range(num_lineups):
        require_low_own = (
            lineup_idx in low_own_required_indices
            and low_own_min_required > 0
            and bool(low_own_candidate_ids)
        )
        require_preferred_stack = (
            lineup_idx in preferred_stack_required_indices
            and preferred_game_stack_min_players_value > 0
            and bool(preferred_games)
        )
        ceiling_boost_active = lineup_idx in ceiling_boost_indices
        active_salary_target = ceiling_salary_target if ceiling_boost_active else salary_target

        if progress_callback is not None:
            status = f"Generating lineup {lineup_idx + 1} of {num_lineups}..."
            progress_callback(lineup_idx, num_lineups, status)
        best_lineup: list[dict[str, Any]] | None = None
        best_score = -10**12

        for _ in range(max_attempts_per_lineup):
            selected = [dict(x) for x in lock_players]
            selected_ids = {str(p["ID"]) for p in selected}
            salary = sum(int(p["Salary"]) for p in selected)
            current_low_own_count = sum(1 for p in selected if str(p.get("ID")) in low_own_candidate_ids)
            current_unsupported_false_chalk_count = sum(
                1 for p in selected if str(p.get("ID")) in unsupported_false_chalk_ids
            )

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
                    if require_low_own:
                        is_low_own_pick = pid in low_own_candidate_ids
                        projected_low_own = current_low_own_count + (1 if is_low_own_pick else 0)
                        min_needed_after_pick = max(0, low_own_min_required - projected_low_own)
                        if min_needed_after_pick > rem_after_pick:
                            continue
                    if unsupported_false_chalk_cap is not None:
                        is_unsupported_false_chalk_pick = pid in unsupported_false_chalk_ids
                        projected_unsupported_false_chalk = (
                            current_unsupported_false_chalk_count + (1 if is_unsupported_false_chalk_pick else 0)
                        )
                        if projected_unsupported_false_chalk > unsupported_false_chalk_cap:
                            continue
                    pick_game_key = str(p.get("game_key") or "").strip().upper()
                    if require_preferred_stack and not _can_still_hit_preferred_stack(
                        selected,
                        pick_game_key,
                        preferred_games,
                        preferred_game_stack_min_players_value,
                        rem_after_pick,
                    ):
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
                if str(pick.get("ID")) in low_own_candidate_ids:
                    current_low_own_count += 1
                if str(pick.get("ID")) in unsupported_false_chalk_ids:
                    current_unsupported_false_chalk_count += 1

            if not selected:
                continue
            if not _lineup_valid(selected):
                continue
            final_salary = int(sum(int(p["Salary"]) for p in selected))
            if final_salary < min_salary_used:
                continue
            current_ids = {str(p["ID"]) for p in selected}
            lineup_low_own_count = sum(1 for pid in current_ids if pid in low_own_candidate_ids)
            if require_low_own and lineup_low_own_count < low_own_min_required:
                continue
            lineup_unsupported_false_chalk_count = sum(1 for pid in current_ids if pid in unsupported_false_chalk_ids)
            if (
                unsupported_false_chalk_cap is not None
                and lineup_unsupported_false_chalk_count > unsupported_false_chalk_cap
            ):
                continue
            preferred_stack_size = _preferred_game_stack_size(selected, preferred_games)
            if require_preferred_stack and preferred_stack_size < preferred_game_stack_min_players_value:
                continue

            selected_score = sum(float(p.get("objective_score", 0.0)) for p in selected)
            game_counts = _lineup_game_counts(selected)
            salary_left = SALARY_CAP - final_salary
            if active_salary_target is not None:
                selected_score -= abs(float(salary_left - active_salary_target)) / salary_penalty_divisor
            if preferred_games and preferred_game_bonus_value > 0.0:
                preferred_count = float(
                    sum(cnt for game_key, cnt in game_counts.items() if str(game_key or "").strip().upper() in preferred_games)
                )
                selected_score += preferred_game_bonus_value * preferred_count
            if ceiling_boost_active and ceiling_boost_stack_bonus > 0.0:
                selected_score += float(ceiling_boost_stack_bonus) * _stack_bonus_from_counts(game_counts)
            if spike_mode:
                def _safe_metric_num(value: Any) -> float:
                    num = _safe_float(value)
                    if num is None or math.isnan(num):
                        return 0.0
                    return float(num)

                selected_score += 1.25 * _stack_bonus_from_counts(game_counts)
                if include_tail_signals:
                    avg_tail = sum(_safe_metric_num(p.get("game_tail_score")) for p in selected) / max(1, len(selected))
                    avg_vol = sum(_safe_metric_num(p.get("game_volatility_score")) for p in selected) / max(1, len(selected))
                    selected_score += (0.20 * avg_tail) + (0.02 * avg_vol)

            if lineups:
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
        expected_minutes_sum, avg_minutes_last3 = _lineup_minutes_metrics(best_lineup)
        preferred_game_player_count = _preferred_game_stack_total(best_lineup, preferred_games)
        preferred_game_stack_size = _preferred_game_stack_size(best_lineup, preferred_games)
        unsupported_false_chalk_count = int(
            sum(1 for p in best_lineup if str(p.get("ID")) in unsupported_false_chalk_ids)
        )

        lineups.append(
            {
                "lineup_number": len(lineups) + 1,
                "players": best_lineup,
                "player_ids": [str(p["ID"]) for p in best_lineup],
                "salary": lineup_salary,
                "salary_left": SALARY_CAP - lineup_salary,
                "projected_points": round(lineup_proj, 2),
                "ceiling_projection": round(lineup_proj * 1.18, 2),
                "projected_ownership_sum": round(lineup_own, 2),
                "expected_minutes_sum": expected_minutes_sum,
                "avg_minutes_last3": avg_minutes_last3,
                "lineup_strategy": "spike" if spike_mode else "standard",
                "model_profile": model_profile_value,
                "low_own_upside_count": int(sum(1 for p in best_lineup if str(p.get("ID")) in low_own_candidate_ids)),
                "preferred_game_player_count": int(preferred_game_player_count),
                "preferred_game_stack_size": int(preferred_game_stack_size),
                "preferred_game_stack_met": bool(
                    preferred_game_stack_min_players_value > 0 and preferred_game_stack_size >= preferred_game_stack_min_players_value
                ),
                "unsupported_false_chalk_count": unsupported_false_chalk_count,
                "ceiling_boost_active": bool(ceiling_boost_active),
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
    low_own_candidate_ids: set[str] | None = None,
    low_own_min_per_lineup: int = 0,
    low_own_required_indices: set[int] | None = None,
    unsupported_false_chalk_ids: set[str] | None = None,
    max_unsupported_false_chalk_per_lineup: int | None = None,
    preferred_game_keys: set[str] | None = None,
    preferred_game_bonus: float = 0.0,
    preferred_game_stack_min_players: int = 0,
    preferred_stack_required_indices: set[int] | None = None,
    ceiling_boost_indices: set[int] | None = None,
    ceiling_boost_stack_bonus: float = 0.0,
    ceiling_boost_salary_left_target: int | None = None,
    model_profile: str = "legacy_baseline",
) -> tuple[list[dict[str, Any]], list[str]]:
    if not players:
        return [], ["No players available for cluster generation."]

    warnings: list[str] = []
    lineups: list[dict[str, Any]] = []
    rng = random.Random(random_seed)
    salary_target = None if salary_left_target is None else max(0, min(SALARY_CAP, int(salary_left_target)))
    ceiling_salary_target = (
        salary_target
        if ceiling_boost_salary_left_target is None
        else max(0, min(SALARY_CAP, int(ceiling_boost_salary_left_target)))
    )
    salary_penalty_divisor = max(1.0, float(salary_left_penalty_divisor))
    low_own_ids = {str(pid) for pid in (low_own_candidate_ids or set()) if str(pid).strip()}
    low_own_required = max(0, min(ROSTER_SIZE, int(low_own_min_per_lineup)))
    low_own_required_idx = {int(x) for x in (low_own_required_indices or set())}
    unsupported_false_chalk_ids_set = {
        str(pid) for pid in (unsupported_false_chalk_ids or set()) if str(pid).strip()
    }
    unsupported_false_chalk_cap = (
        None
        if max_unsupported_false_chalk_per_lineup is None
        else max(0, min(ROSTER_SIZE, int(max_unsupported_false_chalk_per_lineup)))
    )
    if unsupported_false_chalk_cap is not None and unsupported_false_chalk_cap >= ROSTER_SIZE:
        unsupported_false_chalk_cap = None
    preferred_games = {str(x or "").strip().upper() for x in (preferred_game_keys or set()) if str(x or "").strip()}
    preferred_bonus = max(0.0, float(preferred_game_bonus))
    preferred_stack_required = {int(x) for x in (preferred_stack_required_indices or set())}
    preferred_stack_min = max(0, min(ROSTER_SIZE, int(preferred_game_stack_min_players)))
    ceiling_idx = {int(x) for x in (ceiling_boost_indices or set())}
    ceiling_stack_bonus = max(0.0, float(ceiling_boost_stack_bonus))

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
            lineup_global_idx = int(len(lineups))
            require_low_own = (
                lineup_global_idx in low_own_required_idx
                and low_own_required > 0
                and bool(low_own_ids)
            )
            require_preferred_stack = (
                lineup_global_idx in preferred_stack_required
                and preferred_stack_min > 0
                and bool(preferred_games)
            )
            ceiling_boost_active = lineup_global_idx in ceiling_idx
            active_salary_target = ceiling_salary_target if ceiling_boost_active else salary_target
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
                current_low_own_count = sum(1 for p in selected if str(p.get("ID")) in low_own_ids)
                current_unsupported_false_chalk_count = sum(
                    1 for p in selected if str(p.get("ID")) in unsupported_false_chalk_ids_set
                )

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
                        if require_low_own:
                            is_low_own_pick = pid in low_own_ids
                            projected_low_own = current_low_own_count + (1 if is_low_own_pick else 0)
                            low_own_needed_after_pick = max(0, low_own_required - projected_low_own)
                            if low_own_needed_after_pick > rem_after_pick:
                                continue
                        if unsupported_false_chalk_cap is not None:
                            projected_unsupported_false_chalk = current_unsupported_false_chalk_count + (
                                1 if pid in unsupported_false_chalk_ids_set else 0
                            )
                            if projected_unsupported_false_chalk > unsupported_false_chalk_cap:
                                continue
                        if require_preferred_stack and not _can_still_hit_preferred_stack(
                            selected,
                            pick_game_key,
                            preferred_games,
                            preferred_stack_min,
                            rem_after_pick,
                        ):
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
                    if str(pick.get("ID")) in low_own_ids:
                        current_low_own_count += 1
                    if str(pick.get("ID")) in unsupported_false_chalk_ids_set:
                        current_unsupported_false_chalk_count += 1

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
                lineup_low_own_count = sum(1 for p in selected if str(p.get("ID")) in low_own_ids)
                if require_low_own and lineup_low_own_count < low_own_required:
                    continue
                lineup_unsupported_false_chalk_count = sum(
                    1 for p in selected if str(p.get("ID")) in unsupported_false_chalk_ids_set
                )
                if (
                    unsupported_false_chalk_cap is not None
                    and lineup_unsupported_false_chalk_count > unsupported_false_chalk_cap
                ):
                    continue
                preferred_stack_size = _preferred_game_stack_size(selected, preferred_games)
                if require_preferred_stack and preferred_stack_size < preferred_stack_min:
                    continue

                lineup_own = float(sum(float(p.get("projected_ownership") or 0.0) for p in selected))
                selected_score = sum(float(p.get("objective_score", 0.0)) for p in selected)
                selected_score += 1.10 * _stack_bonus_from_counts(game_counts)
                salary_left = SALARY_CAP - final_salary
                if active_salary_target is not None:
                    selected_score -= abs(float(salary_left - active_salary_target)) / salary_penalty_divisor
                if anchor_game_key:
                    selected_score += 1.8 * float(anchor_count)
                if preferred_games and preferred_bonus > 0.0:
                    preferred_count = float(
                        sum(cnt for game_key, cnt in game_counts.items() if str(game_key or "").strip().upper() in preferred_games)
                    )
                    selected_score += preferred_bonus * preferred_count
                if ceiling_boost_active and ceiling_stack_bonus > 0.0:
                    selected_score += ceiling_stack_bonus * _stack_bonus_from_counts(game_counts)

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
                            _safe_num(
                                p.get("our_minutes_recent"),
                                _safe_num(p.get("our_minutes_last7"), _safe_num(p.get("our_minutes_avg"), 0.0)),
                            )
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
            expected_minutes_sum, avg_minutes_last3 = _lineup_minutes_metrics(best_lineup)
            preferred_game_player_count = _preferred_game_stack_total(best_lineup, preferred_games)
            preferred_game_stack_size = _preferred_game_stack_size(best_lineup, preferred_games)
            unsupported_false_chalk_count = int(
                sum(1 for p in best_lineup if str(p.get("ID")) in unsupported_false_chalk_ids_set)
            )
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
                "ceiling_projection": round(lineup_proj * 1.18, 2),
                "projected_ownership_sum": round(lineup_own, 2),
                "expected_minutes_sum": expected_minutes_sum,
                "avg_minutes_last3": avg_minutes_last3,
                "lineup_strategy": "cluster",
                "model_profile": str(model_profile or "legacy_baseline"),
                "cluster_id": cluster_id,
                "cluster_script": cluster_script,
                "anchor_game_key": anchor_game_key,
                "seed_lineup_id": seed_lineup_id,
                "mutation_type": mutation_label,
                "stack_signature": _lineup_stack_signature(best_lineup),
                "salary_texture_bucket": _salary_texture_bucket(SALARY_CAP - lineup_salary),
                "low_own_upside_count": int(sum(1 for p in best_lineup if str(p.get("ID")) in low_own_ids)),
                "preferred_game_player_count": int(preferred_game_player_count),
                "preferred_game_stack_size": int(preferred_game_stack_size),
                "preferred_game_stack_met": bool(preferred_stack_min > 0 and preferred_game_stack_size >= preferred_stack_min),
                "unsupported_false_chalk_count": unsupported_false_chalk_count,
                "ceiling_boost_active": bool(ceiling_boost_active),
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
        expected_minutes_sum_calc, avg_minutes_last3_calc = _lineup_minutes_metrics(players)
        expected_minutes_sum_value = _safe_float(lineup.get("expected_minutes_sum"))
        avg_minutes_last3_value = _safe_float(lineup.get("avg_minutes_last3"))
        if expected_minutes_sum_value is None or math.isnan(expected_minutes_sum_value):
            expected_minutes_sum_value = expected_minutes_sum_calc
        if avg_minutes_last3_value is None or math.isnan(avg_minutes_last3_value):
            avg_minutes_last3_value = avg_minutes_last3_calc
        row = {
            "Lineup": lineup["lineup_number"],
            "Salary": lineup["salary"],
            "Projected Points": lineup["projected_points"],
            "Ceiling Projection": _lineup_ceiling_projection_value(lineup),
            "Projected Ownership Sum": lineup["projected_ownership_sum"],
            "Expected Minutes Sum": round(float(expected_minutes_sum_value), 2),
            "Avg Minutes (Past 3 Games)": round(float(avg_minutes_last3_value), 2),
            "Players": " | ".join(str(p.get("Name + ID") or p.get("Name")) for p in players),
        }
        lineup_model_label = _lineup_model_label_value(lineup)
        if lineup_model_label:
            row["Lineup Model"] = lineup_model_label
        if lineup.get("lineup_strategy"):
            row["Strategy"] = str(lineup.get("lineup_strategy"))
        if lineup.get("model_profile"):
            row["Model Profile"] = str(lineup.get("model_profile"))
        if lineup.get("cluster_id"):
            row["Cluster"] = str(lineup.get("cluster_id"))
        if lineup.get("mutation_type"):
            row["Mutation"] = str(lineup.get("mutation_type"))
        if lineup.get("stack_signature"):
            row["Stack Signature"] = str(lineup.get("stack_signature"))
        if lineup.get("salary_texture_bucket"):
            row["Salary Texture"] = str(lineup.get("salary_texture_bucket"))
        if lineup.get("low_own_upside_count") is not None:
            row["Low-Own Upside Count"] = int(lineup.get("low_own_upside_count") or 0)
        if lineup.get("preferred_game_player_count") is not None:
            row["Preferred Game Players"] = int(lineup.get("preferred_game_player_count") or 0)
        if lineup.get("preferred_game_stack_size") is not None:
            row["Preferred Game Stack Size"] = int(lineup.get("preferred_game_stack_size") or 0)
        if lineup.get("preferred_game_stack_met") is not None:
            row["Preferred Game Stack Met"] = bool(lineup.get("preferred_game_stack_met"))
        if lineup.get("unsupported_false_chalk_count") is not None:
            row["Unsupported False Chalk Count"] = int(lineup.get("unsupported_false_chalk_count") or 0)
        if lineup.get("ceiling_boost_active") is not None:
            row["Ceiling Boost"] = bool(lineup.get("ceiling_boost_active"))
        rows.append(row)
    return pd.DataFrame(rows)


def enrich_lineups_minutes_from_pool(
    lineups: list[dict[str, Any]],
    pool_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    if not lineups:
        return []

    minute_cols = ["our_minutes_recent", "our_minutes_last7", "our_minutes_last3", "our_minutes_avg"]
    minutes_by_id: dict[str, dict[str, float]] = {}
    if isinstance(pool_df, pd.DataFrame) and (not pool_df.empty) and ("ID" in pool_df.columns):
        pool_work = pool_df.copy()
        pool_work["ID"] = pool_work["ID"].astype(str).str.strip()
        for col in minute_cols:
            if col not in pool_work.columns:
                pool_work[col] = pd.NA
            pool_work[col] = pd.to_numeric(pool_work[col], errors="coerce")
        pool_work = pool_work.drop_duplicates(subset=["ID"], keep="first")
        for row in pool_work.loc[:, ["ID"] + minute_cols].to_dict(orient="records"):
            pid = str(row.get("ID") or "").strip()
            if not pid:
                continue
            row_minutes: dict[str, float] = {}
            for col in minute_cols:
                val = _safe_float(row.get(col))
                if val is None or math.isnan(val):
                    continue
                row_minutes[col] = float(val)
            if row_minutes:
                minutes_by_id[pid] = row_minutes

    out: list[dict[str, Any]] = []
    for lineup in lineups:
        lineup_copy = dict(lineup)
        players_in = lineup.get("players") or []
        players_out: list[dict[str, Any]] = []
        for player in players_in:
            p = dict(player)
            player_id = str(p.get("ID") or "").strip()
            mapped = minutes_by_id.get(player_id) or {}
            for col in minute_cols:
                current_val = _safe_float(p.get(col))
                if current_val is None or math.isnan(current_val):
                    mapped_val = _safe_float(mapped.get(col))
                    if mapped_val is not None and not math.isnan(mapped_val):
                        p[col] = float(mapped_val)
            players_out.append(p)

        expected_minutes_sum, avg_minutes_last3 = _lineup_minutes_metrics(players_out)
        lineup_copy["players"] = players_out
        lineup_copy["expected_minutes_sum"] = expected_minutes_sum
        lineup_copy["avg_minutes_last3"] = avg_minutes_last3
        out.append(lineup_copy)

    return out


def lineups_slots_frame(lineups: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    slot_labels = ("G1", "G2", "G3", "F1", "F2", "F3", "UTIL1", "UTIL2")
    for lineup in lineups:
        slots = _assign_dk_slots(lineup["players"])
        if slots is None:
            continue
        expected_minutes_sum_calc, avg_minutes_last3_calc = _lineup_minutes_metrics(lineup["players"])
        expected_minutes_sum_value = _safe_float(lineup.get("expected_minutes_sum"))
        avg_minutes_last3_value = _safe_float(lineup.get("avg_minutes_last3"))
        if expected_minutes_sum_value is None or math.isnan(expected_minutes_sum_value):
            expected_minutes_sum_value = expected_minutes_sum_calc
        if avg_minutes_last3_value is None or math.isnan(avg_minutes_last3_value):
            avg_minutes_last3_value = avg_minutes_last3_calc
        row = {
            "Lineup": lineup["lineup_number"],
            "Salary": lineup["salary"],
            "Projected Points": lineup["projected_points"],
            "Ceiling Projection": _lineup_ceiling_projection_value(lineup),
            "Projected Ownership Sum": lineup["projected_ownership_sum"],
            "Expected Minutes Sum": round(float(expected_minutes_sum_value), 2),
            "Avg Minutes (Past 3 Games)": round(float(avg_minutes_last3_value), 2),
        }
        for slot_label, player in zip(slot_labels, slots, strict=False):
            row[slot_label] = player.get("Name + ID") or player.get("Name")
            row[f"{slot_label} ID"] = str(player.get("ID") or "")
            row[f"{slot_label} Team"] = str(player.get("TeamAbbrev") or "")
            row[f"{slot_label} Position"] = str(player.get("Position") or player.get("PositionBase") or "")
            row[f"{slot_label} Salary"] = _safe_num(player.get("Salary"), 0.0)
            row[f"{slot_label} Projected Points"] = round(_safe_num(player.get("projected_dk_points"), 0.0), 2)
            row[f"{slot_label} Ceiling Projection"] = _player_ceiling_projection_value(player)
            row[f"{slot_label} Projected Ownership"] = round(_safe_num(player.get("projected_ownership"), 0.0), 2)
        lineup_model_label = _lineup_model_label_value(lineup)
        if lineup_model_label:
            row["Lineup Model"] = lineup_model_label
        if lineup.get("lineup_strategy"):
            row["Strategy"] = str(lineup.get("lineup_strategy"))
        if lineup.get("model_profile"):
            row["Model Profile"] = str(lineup.get("model_profile"))
        if lineup.get("cluster_id"):
            row["Cluster"] = str(lineup.get("cluster_id"))
        if lineup.get("mutation_type"):
            row["Mutation"] = str(lineup.get("mutation_type"))
        if lineup.get("stack_signature"):
            row["Stack Signature"] = str(lineup.get("stack_signature"))
        if lineup.get("salary_texture_bucket"):
            row["Salary Texture"] = str(lineup.get("salary_texture_bucket"))
        if lineup.get("low_own_upside_count") is not None:
            row["Low-Own Upside Count"] = int(lineup.get("low_own_upside_count") or 0)
        if lineup.get("preferred_game_player_count") is not None:
            row["Preferred Game Players"] = int(lineup.get("preferred_game_player_count") or 0)
        if lineup.get("preferred_game_stack_size") is not None:
            row["Preferred Game Stack Size"] = int(lineup.get("preferred_game_stack_size") or 0)
        if lineup.get("preferred_game_stack_met") is not None:
            row["Preferred Game Stack Met"] = bool(lineup.get("preferred_game_stack_met"))
        if lineup.get("unsupported_false_chalk_count") is not None:
            row["Unsupported False Chalk Count"] = int(lineup.get("unsupported_false_chalk_count") or 0)
        if lineup.get("ceiling_boost_active") is not None:
            row["Ceiling Boost"] = bool(lineup.get("ceiling_boost_active"))
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
