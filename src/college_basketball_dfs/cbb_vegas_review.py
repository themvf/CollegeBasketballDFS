from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping

import pandas as pd

from college_basketball_dfs.cbb_odds import flatten_odds_payload


def _normalize_team_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    as_float = _to_float(value)
    if as_float is None:
        return None
    return int(round(as_float))


def _is_final_status(status: Any) -> bool:
    text = str(status or "").strip().lower()
    if not text:
        return True
    return text in {"f", "final"} or text.startswith("final")


def _winner_from_margin(home_margin: float | int | None) -> str:
    if home_margin is None or pd.isna(home_margin):
        return ""
    if float(home_margin) > 0:
        return "home"
    if float(home_margin) < 0:
        return "away"
    return "push"


def build_results_frame(raw_payloads: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in raw_payloads:
        game_date = str(payload.get("game_date") or "").strip()
        games = payload.get("games")
        if not isinstance(games, list):
            continue
        for game in games:
            if not isinstance(game, Mapping):
                continue
            home_team = str(game.get("home_team") or "").strip()
            away_team = str(game.get("away_team") or "").strip()
            if not home_team or not away_team:
                continue
            if not _is_final_status(game.get("status")):
                continue
            home_score = _to_int(game.get("home_score"))
            away_score = _to_int(game.get("away_score"))
            if home_score is None or away_score is None:
                continue
            actual_total = home_score + away_score
            actual_home_margin = home_score - away_score
            rows.append(
                {
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_score": home_score,
                    "away_score": away_score,
                    "actual_total": actual_total,
                    "actual_home_margin": actual_home_margin,
                    "actual_winner_side": _winner_from_margin(actual_home_margin),
                    "_home_norm": _normalize_team_name(home_team),
                    "_away_norm": _normalize_team_name(away_team),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "game_date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "actual_total",
                "actual_home_margin",
                "actual_winner_side",
                "_home_norm",
                "_away_norm",
            ]
        )
    out = pd.DataFrame(rows)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["game_date"]).reset_index(drop=True)
    return out


def build_odds_frame(odds_payloads: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in odds_payloads:
        rows.extend(flatten_odds_payload(payload))

    if not rows:
        return pd.DataFrame(
            columns=[
                "game_date",
                "event_id",
                "home_team",
                "away_team",
                "bookmakers_count",
                "moneyline_home",
                "moneyline_away",
                "spread_home",
                "spread_away",
                "total_points",
                "total_samples",
                "spread_samples",
                "moneyline_samples",
                "vegas_home_margin",
                "_home_norm",
                "_away_norm",
            ]
        )

    out = pd.DataFrame(rows)
    out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["home_team"] = out.get("home_team", "").astype(str).str.strip()
    out["away_team"] = out.get("away_team", "").astype(str).str.strip()
    out["_home_norm"] = out["home_team"].map(_normalize_team_name)
    out["_away_norm"] = out["away_team"].map(_normalize_team_name)
    for col in [
        "bookmakers_count",
        "moneyline_home",
        "moneyline_away",
        "spread_home",
        "spread_away",
        "total_points",
        "total_samples",
        "spread_samples",
        "moneyline_samples",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    vegas_margin = -out["spread_home"]
    vegas_margin = vegas_margin.where(out["spread_home"].notna(), out["spread_away"])
    out["vegas_home_margin"] = pd.to_numeric(vegas_margin, errors="coerce")
    return out


def _oriented_odds_frame(odds_df: pd.DataFrame) -> pd.DataFrame:
    if odds_df.empty:
        return odds_df.copy()

    base_cols = [
        "game_date",
        "event_id",
        "bookmakers_count",
        "moneyline_home",
        "moneyline_away",
        "spread_home",
        "spread_away",
        "total_points",
        "over_price",
        "under_price",
        "moneyline_samples",
        "spread_samples",
        "total_samples",
        "vegas_home_margin",
    ]
    cols = [c for c in base_cols if c in odds_df.columns]
    keep_cols = cols + ["home_team", "away_team", "_home_norm", "_away_norm"]
    primary = odds_df[keep_cols].copy()

    flipped = primary.copy()
    flipped["home_team"] = primary["away_team"]
    flipped["away_team"] = primary["home_team"]
    flipped["_home_norm"] = primary["_away_norm"]
    flipped["_away_norm"] = primary["_home_norm"]

    if "moneyline_home" in flipped.columns and "moneyline_away" in flipped.columns:
        flipped["moneyline_home"] = primary["moneyline_away"]
        flipped["moneyline_away"] = primary["moneyline_home"]
    if "spread_home" in flipped.columns and "spread_away" in flipped.columns:
        flipped["spread_home"] = primary["spread_away"]
        flipped["spread_away"] = primary["spread_home"]
    if "vegas_home_margin" in flipped.columns:
        flipped["vegas_home_margin"] = -pd.to_numeric(primary["vegas_home_margin"], errors="coerce")

    out = pd.concat([primary, flipped], ignore_index=True)
    out["_join_key"] = out["game_date"].astype(str) + "|" + out["_home_norm"] + "|" + out["_away_norm"]
    quality_cols = [c for c in ["total_samples", "spread_samples", "bookmakers_count"] if c in out.columns]
    if quality_cols:
        out["_quality"] = 0.0
        for col in quality_cols:
            out["_quality"] = out["_quality"] + pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        out = out.sort_values(["_join_key", "_quality"], ascending=[True, False])
    out = out.drop_duplicates(subset=["_join_key"], keep="first")
    return out.drop(columns=[c for c in ["_join_key", "_quality"] if c in out.columns]).reset_index(drop=True)


def build_vegas_review_games_frame(
    raw_payloads: Iterable[Mapping[str, Any]],
    odds_payloads: Iterable[Mapping[str, Any]],
) -> pd.DataFrame:
    results_df = build_results_frame(raw_payloads)
    if results_df.empty:
        return results_df

    odds_df = build_odds_frame(odds_payloads)
    oriented_odds = _oriented_odds_frame(odds_df)
    if oriented_odds.empty:
        out = results_df.copy()
        for col in [
            "event_id",
            "bookmakers_count",
            "moneyline_home",
            "moneyline_away",
            "spread_home",
            "spread_away",
            "total_points",
            "over_price",
            "under_price",
            "moneyline_samples",
            "spread_samples",
            "total_samples",
            "vegas_home_margin",
        ]:
            out[col] = pd.NA
        return compute_vegas_accuracy_columns(out)

    merged = results_df.merge(
        oriented_odds,
        on=["game_date", "_home_norm", "_away_norm"],
        how="left",
        suffixes=("", "_odds"),
    )
    return compute_vegas_accuracy_columns(merged)


def compute_vegas_accuracy_columns(games_df: pd.DataFrame) -> pd.DataFrame:
    if games_df.empty:
        return games_df.copy()

    out = games_df.copy()
    out["actual_total"] = pd.to_numeric(out["actual_total"], errors="coerce")
    out["actual_home_margin"] = pd.to_numeric(out["actual_home_margin"], errors="coerce")
    out["total_points"] = pd.to_numeric(out.get("total_points"), errors="coerce")
    out["vegas_home_margin"] = pd.to_numeric(out.get("vegas_home_margin"), errors="coerce")

    out["has_odds_match"] = out["event_id"].notna()
    out["has_total_line"] = out["total_points"].notna()
    out["has_spread_line"] = out["vegas_home_margin"].notna()

    out["total_error"] = out["actual_total"] - out["total_points"]
    out["total_abs_error"] = out["total_error"].abs()
    out["spread_error"] = out["actual_home_margin"] - out["vegas_home_margin"]
    out["spread_abs_error"] = out["spread_error"].abs()

    out["total_result"] = pd.NA
    over_mask = out["has_total_line"] & (out["actual_total"] > out["total_points"])
    under_mask = out["has_total_line"] & (out["actual_total"] < out["total_points"])
    push_mask = out["has_total_line"] & (out["actual_total"] == out["total_points"])
    out.loc[over_mask, "total_result"] = "Over"
    out.loc[under_mask, "total_result"] = "Under"
    out.loc[push_mask, "total_result"] = "Push"

    out["predicted_winner_side"] = out["vegas_home_margin"].map(_winner_from_margin)
    out["winner_pick_correct"] = (
        out["has_spread_line"]
        & out["actual_winner_side"].isin(["home", "away"])
        & out["predicted_winner_side"].isin(["home", "away"])
        & (out["actual_winner_side"] == out["predicted_winner_side"])
    )

    out["spread_within_3"] = out["has_spread_line"] & (out["spread_abs_error"] <= 3.0)
    out["spread_within_5"] = out["has_spread_line"] & (out["spread_abs_error"] <= 5.0)
    out["spread_within_10"] = out["has_spread_line"] & (out["spread_abs_error"] <= 10.0)
    out["total_within_5"] = out["has_total_line"] & (out["total_abs_error"] <= 5.0)
    out["total_within_10"] = out["has_total_line"] & (out["total_abs_error"] <= 10.0)
    return out


def _safe_pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _safe_rmse(errors: pd.Series) -> float:
    if errors.empty:
        return 0.0
    return float(math.sqrt(float((errors.pow(2).mean()))))


def summarize_vegas_accuracy(games_df: pd.DataFrame) -> dict[str, float | int]:
    if games_df.empty:
        return {
            "total_games": 0,
            "odds_matched_games": 0,
            "total_line_games": 0,
            "spread_line_games": 0,
            "total_mae": 0.0,
            "total_rmse": 0.0,
            "total_within_5_pct": 0.0,
            "total_within_10_pct": 0.0,
            "over_rate_pct": 0.0,
            "under_rate_pct": 0.0,
            "push_rate_pct": 0.0,
            "spread_mae": 0.0,
            "spread_rmse": 0.0,
            "spread_within_3_pct": 0.0,
            "spread_within_5_pct": 0.0,
            "winner_pick_accuracy_pct": 0.0,
        }

    total_line_games = int(games_df["has_total_line"].sum())
    spread_line_games = int(games_df["has_spread_line"].sum())

    total_errors = games_df.loc[games_df["has_total_line"], "total_error"].dropna()
    spread_errors = games_df.loc[games_df["has_spread_line"], "spread_error"].dropna()
    total_results = games_df.loc[games_df["has_total_line"], "total_result"].astype(str)
    pick_mask = (
        games_df["has_spread_line"]
        & games_df["actual_winner_side"].isin(["home", "away"])
        & games_df["predicted_winner_side"].isin(["home", "away"])
    )

    return {
        "total_games": int(len(games_df)),
        "odds_matched_games": int(games_df["has_odds_match"].sum()),
        "total_line_games": total_line_games,
        "spread_line_games": spread_line_games,
        "total_mae": float(total_errors.abs().mean()) if not total_errors.empty else 0.0,
        "total_rmse": _safe_rmse(total_errors) if not total_errors.empty else 0.0,
        "total_within_5_pct": _safe_pct(float(games_df["total_within_5"].sum()), float(total_line_games)),
        "total_within_10_pct": _safe_pct(float(games_df["total_within_10"].sum()), float(total_line_games)),
        "over_rate_pct": _safe_pct(float((total_results == "Over").sum()), float(total_line_games)),
        "under_rate_pct": _safe_pct(float((total_results == "Under").sum()), float(total_line_games)),
        "push_rate_pct": _safe_pct(float((total_results == "Push").sum()), float(total_line_games)),
        "spread_mae": float(spread_errors.abs().mean()) if not spread_errors.empty else 0.0,
        "spread_rmse": _safe_rmse(spread_errors) if not spread_errors.empty else 0.0,
        "spread_within_3_pct": _safe_pct(float(games_df["spread_within_3"].sum()), float(spread_line_games)),
        "spread_within_5_pct": _safe_pct(float(games_df["spread_within_5"].sum()), float(spread_line_games)),
        "winner_pick_accuracy_pct": _safe_pct(
            float(games_df.loc[pick_mask, "winner_pick_correct"].sum()),
            float(int(pick_mask.sum())),
        ),
    }


def fit_linear_calibration(games_df: pd.DataFrame, x_col: str, y_col: str, label: str) -> dict[str, float | str | int]:
    valid = games_df[[x_col, y_col]].copy()
    valid[x_col] = pd.to_numeric(valid[x_col], errors="coerce")
    valid[y_col] = pd.to_numeric(valid[y_col], errors="coerce")
    valid = valid.dropna(subset=[x_col, y_col]).reset_index(drop=True)

    if valid.empty:
        return {
            "model": label,
            "samples": 0,
            "slope": 0.0,
            "intercept": 0.0,
            "r2": 0.0,
            "baseline_mae": 0.0,
            "calibrated_mae": 0.0,
            "mae_delta": 0.0,
        }

    x = valid[x_col]
    y = valid[y_col]
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    centered_x = x - x_mean
    centered_y = y - y_mean
    variance_x = float((centered_x * centered_x).sum())
    covariance_xy = float((centered_x * centered_y).sum())
    slope = (covariance_xy / variance_x) if variance_x > 0 else 0.0
    intercept = y_mean - (slope * x_mean)

    baseline_pred = x
    calibrated_pred = intercept + (slope * x)
    baseline_mae = float((y - baseline_pred).abs().mean())
    calibrated_mae = float((y - calibrated_pred).abs().mean())
    corr = float(x.corr(y)) if len(valid) > 1 else 0.0
    if pd.isna(corr):
        corr = 0.0
    r2 = corr * corr

    return {
        "model": label,
        "samples": int(len(valid)),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "baseline_mae": baseline_mae,
        "calibrated_mae": calibrated_mae,
        "mae_delta": calibrated_mae - baseline_mae,
    }


def build_calibration_models_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        fit_linear_calibration(
            games_df.loc[games_df["has_total_line"]].copy(),
            x_col="total_points",
            y_col="actual_total",
            label="Total Points Calibration",
        ),
        fit_linear_calibration(
            games_df.loc[games_df["has_spread_line"]].copy(),
            x_col="vegas_home_margin",
            y_col="actual_home_margin",
            label="Spread Margin Calibration",
        ),
    ]
    return pd.DataFrame(rows)


def build_total_buckets_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.loc[games_df["has_total_line"]].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "vegas_total_bucket",
                "games",
                "avg_vegas_total",
                "avg_actual_total",
                "mae_total",
                "over_rate_pct",
                "under_rate_pct",
            ]
        )

    bins = [0, 130, 140, 150, 160, 170, 300]
    labels = ["<130", "130-139", "140-149", "150-159", "160-169", "170+"]
    df["vegas_total_bucket"] = pd.cut(df["total_points"], bins=bins, labels=labels, include_lowest=True, right=False)
    grouped = (
        df.groupby("vegas_total_bucket", dropna=False, observed=False)
        .agg(
            games=("event_id", "count"),
            avg_vegas_total=("total_points", "mean"),
            avg_actual_total=("actual_total", "mean"),
            mae_total=("total_abs_error", "mean"),
            over_rate_pct=("total_result", lambda s: _safe_pct(float((s == "Over").sum()), float(len(s)))),
            under_rate_pct=("total_result", lambda s: _safe_pct(float((s == "Under").sum()), float(len(s)))),
        )
        .reset_index()
    )
    return grouped


def build_spread_buckets_frame(games_df: pd.DataFrame) -> pd.DataFrame:
    df = games_df.loc[games_df["has_spread_line"]].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "vegas_spread_bucket",
                "games",
                "avg_abs_vegas_margin",
                "avg_abs_actual_margin",
                "mae_spread",
                "winner_pick_accuracy_pct",
            ]
        )

    df["abs_vegas_margin"] = df["vegas_home_margin"].abs()
    df["abs_actual_margin"] = df["actual_home_margin"].abs()
    bins = [0, 2, 4, 6, 8, 12, 100]
    labels = ["<2", "2-3.9", "4-5.9", "6-7.9", "8-11.9", "12+"]
    df["vegas_spread_bucket"] = pd.cut(df["abs_vegas_margin"], bins=bins, labels=labels, include_lowest=True, right=False)

    grouped = (
        df.groupby("vegas_spread_bucket", dropna=False, observed=False)
        .agg(
            games=("event_id", "count"),
            avg_abs_vegas_margin=("abs_vegas_margin", "mean"),
            avg_abs_actual_margin=("abs_actual_margin", "mean"),
            mae_spread=("spread_abs_error", "mean"),
            winner_pick_accuracy_pct=(
                "winner_pick_correct",
                lambda s: _safe_pct(float(pd.to_numeric(s, errors="coerce").fillna(0).sum()), float(len(s))),
            ),
        )
        .reset_index()
    )
    return grouped
