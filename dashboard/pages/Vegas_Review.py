from __future__ import annotations

import io
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from college_basketball_dfs.cbb_backfill import iter_dates, season_start_for_date
from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_ncaa import prior_day
from college_basketball_dfs.cbb_vegas_review import (
    build_calibration_models_frame,
    build_spread_buckets_frame,
    build_total_buckets_frame,
    build_vegas_review_games_frame,
    summarize_vegas_accuracy,
)


def _secret(name: str) -> str | None:
    try:
        if name in st.secrets:
            value = st.secrets[name]
            return str(value) if value else None
    except Exception:
        return None
    return None


def _resolve_credential_json() -> str | None:
    return os.getenv("GCP_SERVICE_ACCOUNT_JSON") or _secret("gcp_service_account_json")


def _resolve_credential_json_b64() -> str | None:
    return os.getenv("GCP_SERVICE_ACCOUNT_JSON_B64") or _secret("gcp_service_account_json_b64")


@st.cache_data(ttl=900, show_spinner=False)
def load_vegas_review_dataset(
    bucket_name: str,
    start_date: date,
    end_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)

    raw_payloads: list[dict[str, Any]] = []
    odds_payloads: list[dict[str, Any]] = []
    dates = iter_dates(start_date, end_date)
    raw_days = 0
    odds_days = 0

    for d in dates:
        raw_payload = store.read_raw_json(d)
        if raw_payload is not None:
            raw_days += 1
            raw_payloads.append(raw_payload)
        odds_payload = store.read_odds_json(d)
        if odds_payload is not None:
            odds_days += 1
            odds_payloads.append(odds_payload)

    games_df = build_vegas_review_games_frame(raw_payloads=raw_payloads, odds_payloads=odds_payloads)
    meta = {
        "dates_scanned": len(dates),
        "raw_days_found": raw_days,
        "odds_days_found": odds_days,
    }
    return games_df, meta


@st.cache_data(ttl=900, show_spinner=False)
def load_projection_player_totals_dataset(
    bucket_name: str,
    start_date: date,
    end_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)

    frames: list[pd.DataFrame] = []
    dates = iter_dates(start_date, end_date)
    projection_days = 0

    for d in dates:
        csv_text = store.read_projections_csv(d)
        if not csv_text or not csv_text.strip():
            continue
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            continue
        if df.empty:
            continue
        projection_days += 1
        df = df.copy()
        df["review_date"] = d.isoformat()
        frames.append(df)

    if not frames:
        return pd.DataFrame(), {"projection_days_found": 0, "projection_rows": 0}

    out = pd.concat(frames, ignore_index=True)
    return out, {
        "projection_days_found": int(projection_days),
        "projection_rows": int(len(out)),
    }


def _safe_corr(series_x: pd.Series, series_y: pd.Series, method: str) -> float | None:
    valid = pd.DataFrame({"x": series_x, "y": series_y}).dropna()
    if len(valid) < 2:
        return None
    if str(method).strip().lower() == "spearman":
        # Avoid pandas' scipy dependency for spearman by ranking then using pearson.
        value = valid["x"].rank(method="average").corr(valid["y"].rank(method="average"))
    else:
        value = valid["x"].corr(valid["y"])
    if pd.isna(value):
        return None
    return float(value)


def _build_join_audit_outputs(
    player_df: pd.DataFrame,
    *,
    low_match_threshold: float = 0.70,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    summary: dict[str, Any] = {
        "rows_total": 0,
        "rows_with_game_total_line": 0,
        "rows_with_game_tail_match_score": 0,
        "rows_with_vegas_points_line": 0,
        "rows_with_total_and_points_line": 0,
        "rows_with_low_match_score": 0,
        "low_match_threshold": float(low_match_threshold),
        "pct_with_game_total_line": 0.0,
        "pct_with_game_tail_match_score": 0.0,
        "pct_with_vegas_points_line": 0.0,
        "pct_with_total_and_points_line": 0.0,
        "pct_low_match_within_scored": 0.0,
        "match_score_median": None,
        "match_score_p10": None,
        "match_score_min": None,
        "game_group_source": "unavailable",
        "approx_unique_game_groups": 0,
    }
    if player_df is None or player_df.empty:
        return summary, pd.DataFrame(), pd.DataFrame()

    work = player_df.copy()
    summary["rows_total"] = int(len(work))

    total_line = pd.to_numeric(work.get("game_total_line"), errors="coerce")
    match_score = pd.to_numeric(work.get("game_tail_match_score"), errors="coerce")
    vegas_points_line = pd.to_numeric(work.get("vegas_points_line"), errors="coerce")

    has_total = total_line.notna()
    has_match = match_score.notna()
    has_points = vegas_points_line.notna()
    has_total_and_points = has_total & has_points
    low_match = has_match & (match_score < float(low_match_threshold))

    rows_total = max(1, int(len(work)))
    summary["rows_with_game_total_line"] = int(has_total.sum())
    summary["rows_with_game_tail_match_score"] = int(has_match.sum())
    summary["rows_with_vegas_points_line"] = int(has_points.sum())
    summary["rows_with_total_and_points_line"] = int(has_total_and_points.sum())
    summary["rows_with_low_match_score"] = int(low_match.sum())
    summary["pct_with_game_total_line"] = float(100.0 * summary["rows_with_game_total_line"] / rows_total)
    summary["pct_with_game_tail_match_score"] = float(100.0 * summary["rows_with_game_tail_match_score"] / rows_total)
    summary["pct_with_vegas_points_line"] = float(100.0 * summary["rows_with_vegas_points_line"] / rows_total)
    summary["pct_with_total_and_points_line"] = float(100.0 * summary["rows_with_total_and_points_line"] / rows_total)
    scored_rows = max(1, summary["rows_with_game_tail_match_score"])
    summary["pct_low_match_within_scored"] = float(100.0 * summary["rows_with_low_match_score"] / scored_rows)

    match_nonnull = match_score.dropna()
    if not match_nonnull.empty:
        summary["match_score_median"] = float(match_nonnull.median())
        summary["match_score_p10"] = float(match_nonnull.quantile(0.10))
        summary["match_score_min"] = float(match_nonnull.min())

    group_key, group_source = _resolve_game_group_key(work)
    summary["game_group_source"] = str(group_source)
    summary["approx_unique_game_groups"] = int(group_key.dropna().nunique())

    per_date_df = pd.DataFrame()
    if "review_date" in work.columns:
        work["review_date"] = work["review_date"].astype(str).str.strip()
        work = work.loc[work["review_date"] != ""].copy()
    if not work.empty and "review_date" in work.columns:
        work["_total_line"] = total_line
        work["_match_score"] = match_score
        work["_points_line"] = vegas_points_line
        work["_has_total"] = work["_total_line"].notna()
        work["_has_match"] = work["_match_score"].notna()
        work["_has_points"] = work["_points_line"].notna()
        work["_has_total_and_points"] = work["_has_total"] & work["_has_points"]
        work["_low_match"] = work["_has_match"] & (work["_match_score"] < float(low_match_threshold))

        rows: list[dict[str, Any]] = []
        for review_date, day in work.groupby("review_date", as_index=False):
            day = day.copy()
            n = int(len(day))
            pair = day.loc[day["_has_total_and_points"], ["_total_line", "_points_line"]].dropna()
            rows.append(
                {
                    "review_date": str(review_date),
                    "rows": n,
                    "rows_with_total_line": int(day["_has_total"].sum()),
                    "rows_with_match_score": int(day["_has_match"].sum()),
                    "rows_with_points_line": int(day["_has_points"].sum()),
                    "rows_with_total_and_points_line": int(day["_has_total_and_points"].sum()),
                    "rows_with_low_match_score": int(day["_low_match"].sum()),
                    "pct_total_line": round(float(100.0 * day["_has_total"].mean()), 2) if n else 0.0,
                    "pct_points_line": round(float(100.0 * day["_has_points"].mean()), 2) if n else 0.0,
                    "pct_low_match_within_scored": (
                        round(float(100.0 * day["_low_match"].sum() / max(1, int(day["_has_match"].sum()))), 2)
                    ),
                    "match_score_median": (
                        round(float(pd.to_numeric(day["_match_score"], errors="coerce").dropna().median()), 4)
                        if day["_has_match"].any()
                        else None
                    ),
                    "points_line_pair_samples": int(len(pair)),
                    "points_line_pair_pearson": (
                        _safe_corr(pair["_total_line"], pair["_points_line"], method="pearson") if len(pair) >= 2 else None
                    ),
                    "points_line_pair_spearman": (
                        _safe_corr(pair["_total_line"], pair["_points_line"], method="spearman") if len(pair) >= 2 else None
                    ),
                }
            )
        per_date_df = pd.DataFrame(rows)
        if not per_date_df.empty:
            per_date_df = per_date_df.sort_values("review_date", ascending=False).reset_index(drop=True)

    preview_cols = [
        "review_date",
        "Name",
        "TeamAbbrev",
        "game_total_line",
        "game_spread_line",
        "game_tail_match_score",
        "game_tail_event_id",
        "vegas_points_line",
        "blend_points_proj",
        "our_points_proj",
    ]
    existing_preview_cols = [c for c in preview_cols if c in player_df.columns]
    low_match_df = player_df.copy()
    if "game_tail_match_score" in low_match_df.columns:
        low_match_df["_game_tail_match_score"] = pd.to_numeric(low_match_df["game_tail_match_score"], errors="coerce")
        low_match_df = low_match_df.loc[low_match_df["_game_tail_match_score"].notna()]
        low_match_df = low_match_df.loc[low_match_df["_game_tail_match_score"] < float(low_match_threshold)].copy()
        low_match_df = low_match_df.sort_values("_game_tail_match_score", ascending=True)
    else:
        low_match_df = pd.DataFrame(columns=existing_preview_cols)
    if existing_preview_cols:
        low_match_df = low_match_df[existing_preview_cols].head(200).copy()
    else:
        low_match_df = pd.DataFrame()

    return summary, per_date_df, low_match_df


def _safe_weighted_corr(
    series_x: pd.Series,
    series_y: pd.Series,
    series_w: pd.Series,
    method: str,
) -> float | None:
    valid = pd.DataFrame({"x": series_x, "y": series_y, "w": series_w}).dropna()
    if len(valid) < 2:
        return None

    weights = pd.to_numeric(valid["w"], errors="coerce")
    valid = valid.loc[weights.notna()].copy()
    if valid.empty:
        return None
    valid["w"] = pd.to_numeric(valid["w"], errors="coerce")
    valid = valid.loc[valid["w"] > 0].copy()
    if len(valid) < 2:
        return None

    x = pd.to_numeric(valid["x"], errors="coerce")
    y = pd.to_numeric(valid["y"], errors="coerce")
    w = pd.to_numeric(valid["w"], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y, "w": w}).dropna()
    valid = valid.loc[valid["w"] > 0].copy()
    if len(valid) < 2:
        return None

    x = valid["x"]
    y = valid["y"]
    w = valid["w"]

    if str(method).strip().lower() == "spearman":
        x = x.rank(method="average")
        y = y.rank(method="average")

    w_sum = float(w.sum())
    if w_sum <= 0.0:
        return None

    x_mean = float((w * x).sum() / w_sum)
    y_mean = float((w * y).sum() / w_sum)

    x_centered = x - x_mean
    y_centered = y - y_mean
    cov_xy = float((w * x_centered * y_centered).sum() / w_sum)
    var_x = float((w * (x_centered ** 2)).sum() / w_sum)
    var_y = float((w * (y_centered ** 2)).sum() / w_sum)

    if var_x <= 0.0 or var_y <= 0.0:
        return None

    corr = cov_xy / ((var_x ** 0.5) * (var_y ** 0.5))
    if pd.isna(corr):
        return None
    return float(max(-1.0, min(1.0, corr)))


def _pick_weight_col(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if int(values.notna().sum()) < 2:
            continue
        if int((values > 0).sum()) < 2:
            continue
        return col
    return None


def _resolve_game_group_key(frame: pd.DataFrame) -> tuple[pd.Series, str]:
    direct_key_candidates = [
        "game_key",
        "Game Info",
        "GameInfo",
        "game_info",
        "event_id",
        "odds_event_id",
        "game_id",
        "matchup",
        "matchup_key",
    ]
    null_tokens = {"", "nan", "none", "null"}

    for col in direct_key_candidates:
        if col not in frame.columns:
            continue
        key = frame[col].astype(str).str.strip()
        key = key.where(~key.str.lower().isin(null_tokens), pd.NA)
        coverage = float(key.notna().mean()) if len(key) else 0.0
        if coverage >= 0.5:
            return key, col

    parts: list[pd.Series] = []
    if "review_date" in frame.columns:
        review_date = frame["review_date"].astype(str).str.strip()
        review_date = review_date.where(~review_date.str.lower().isin(null_tokens), "")
        parts.append(review_date)
    for col in [
        "game_total_line",
        "game_spread_line",
        "game_tail_residual_mu",
        "game_tail_sigma",
        "game_p_plus_8",
        "game_p_plus_12",
    ]:
        if col not in frame.columns:
            continue
        nums = pd.to_numeric(frame[col], errors="coerce").round(3)
        parts.append(nums.map(lambda v: f"{col}:{v:.3f}" if pd.notna(v) else ""))

    if not parts:
        return pd.Series(pd.NA, index=frame.index, dtype="object"), "unavailable"

    out = pd.Series([""] * len(frame), index=frame.index, dtype="object")
    for part in parts:
        out = out + "|" + part.astype(str).str.strip()
    out = out.str.strip("|")
    out = out.where(out != "", pd.NA)
    return out, "synthetic_game_bucket"


def _build_game_level_view(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    top_n_players: int,
) -> tuple[pd.DataFrame, str]:
    group_key, key_source = _resolve_game_group_key(frame)
    x = pd.to_numeric(frame.get(x_col), errors="coerce")
    y = pd.to_numeric(frame.get(y_col), errors="coerce")
    work = pd.DataFrame({"group_key": group_key, "x": x, "y": y}).dropna()
    if work.empty:
        return pd.DataFrame(), key_source

    grouped = work.groupby("group_key", dropna=True)
    agg = grouped.agg(
        x=("x", "mean"),
        y_game_avg=("y", "mean"),
    ).reset_index()
    n = max(1, int(top_n_players))
    top_col = f"y_game_top{n}_avg"
    top_df = grouped["y"].apply(lambda s: float(pd.to_numeric(s, errors="coerce").nlargest(n).mean())).reset_index()
    top_df = top_df.rename(columns={"y": top_col})
    agg = agg.merge(top_df, on="group_key", how="left")
    return agg, key_source


def build_game_total_player_total_correlation_frame(
    player_df: pd.DataFrame,
    *,
    top_n_players: int = 3,
) -> pd.DataFrame:
    columns = [
        "metric",
        "x_col",
        "y_col",
        "samples",
        "pearson",
        "spearman",
        "level",
        "aggregation",
        "weighting",
        "weight_col",
        "game_group_source",
    ]
    if player_df is None or player_df.empty:
        return pd.DataFrame(columns=columns)

    work = player_df.copy()
    specs = [
        ("game_total_line", "vegas_points_line", "game_total_line_vs_vegas_points_line"),
        ("game_total_line", "blend_points_proj", "game_total_line_vs_blend_points_proj"),
        ("game_total_line", "our_points_proj", "game_total_line_vs_our_points_proj"),
        ("game_total_line", "vegas_dk_projection", "game_total_line_vs_vegas_dk_projection"),
        ("game_total_line", "blended_projection", "game_total_line_vs_blended_projection"),
    ]

    rows: list[dict[str, Any]] = []
    salary_weight_col = _pick_weight_col(work, ["Salary", "salary"])
    minutes_weight_col = _pick_weight_col(
        work,
        ["our_minutes_recent", "our_minutes_last7", "our_minutes_avg", "actual_minutes", "minutes_played"],
    )
    if minutes_weight_col == salary_weight_col:
        minutes_weight_col = None

    for x_col, y_col, metric in specs:
        if x_col not in work.columns or y_col not in work.columns:
            rows.append(
                {
                    "metric": metric,
                    "x_col": x_col,
                    "y_col": y_col,
                    "samples": 0,
                    "pearson": None,
                    "spearman": None,
                    "level": "player",
                    "aggregation": "raw",
                    "weighting": "none",
                    "weight_col": None,
                    "game_group_source": None,
                }
            )
            continue
        x = pd.to_numeric(work[x_col], errors="coerce")
        y = pd.to_numeric(work[y_col], errors="coerce")
        valid = pd.DataFrame({"x": x, "y": y}).dropna()
        rows.append(
            {
                "metric": metric,
                "x_col": x_col,
                "y_col": y_col,
                "samples": int(len(valid)),
                "pearson": _safe_corr(x, y, method="pearson"),
                "spearman": _safe_corr(x, y, method="spearman"),
                "level": "player",
                "aggregation": "raw",
                "weighting": "none",
                "weight_col": None,
                "game_group_source": None,
            }
        )
        for weighting, weight_col in [
            ("salary", salary_weight_col),
            ("minutes", minutes_weight_col),
        ]:
            if not weight_col or weight_col not in work.columns:
                continue
            w = pd.to_numeric(work[weight_col], errors="coerce")
            valid_weighted = pd.DataFrame({"x": x, "y": y, "w": w}).dropna()
            valid_weighted = valid_weighted.loc[valid_weighted["w"] > 0]
            rows.append(
                {
                    "metric": f"{metric}_{weighting}_weighted",
                    "x_col": x_col,
                    "y_col": y_col,
                    "samples": int(len(valid_weighted)),
                    "pearson": _safe_weighted_corr(x, y, w, method="pearson"),
                    "spearman": _safe_weighted_corr(x, y, w, method="spearman"),
                    "level": "player",
                    "aggregation": "raw",
                    "weighting": weighting,
                    "weight_col": weight_col,
                    "game_group_source": None,
                }
            )

        game_level_view, game_group_source = _build_game_level_view(
            work,
            x_col=x_col,
            y_col=y_col,
            top_n_players=top_n_players,
        )
        if not game_level_view.empty:
            x_game = pd.to_numeric(game_level_view["x"], errors="coerce")
            y_game_avg = pd.to_numeric(game_level_view["y_game_avg"], errors="coerce")
            valid_game_avg = pd.DataFrame({"x": x_game, "y": y_game_avg}).dropna()
            rows.append(
                {
                    "metric": f"{metric}_game_avg",
                    "x_col": x_col,
                    "y_col": y_col,
                    "samples": int(len(valid_game_avg)),
                    "pearson": _safe_corr(x_game, y_game_avg, method="pearson"),
                    "spearman": _safe_corr(x_game, y_game_avg, method="spearman"),
                    "level": "game",
                    "aggregation": "avg",
                    "weighting": "none",
                    "weight_col": None,
                    "game_group_source": game_group_source,
                }
            )

            n = max(1, int(top_n_players))
            top_col = f"y_game_top{n}_avg"
            if top_col in game_level_view.columns:
                y_game_top = pd.to_numeric(game_level_view[top_col], errors="coerce")
                valid_game_top = pd.DataFrame({"x": x_game, "y": y_game_top}).dropna()
                rows.append(
                    {
                        "metric": f"{metric}_game_top{n}_avg",
                        "x_col": x_col,
                        "y_col": y_col,
                        "samples": int(len(valid_game_top)),
                        "pearson": _safe_corr(x_game, y_game_top, method="pearson"),
                        "spearman": _safe_corr(x_game, y_game_top, method="spearman"),
                        "level": "game",
                        "aggregation": f"top{n}_avg",
                        "weighting": "none",
                        "weight_col": None,
                        "game_group_source": game_group_source,
                    }
                )
    out = pd.DataFrame(rows, columns=columns)
    if out.empty:
        return out
    out["samples"] = pd.to_numeric(out["samples"], errors="coerce").fillna(0).astype(int)
    out = out.sort_values(
        by=["level", "aggregation", "weighting", "samples", "metric"],
        ascending=[True, True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    return out


st.set_page_config(page_title="Vegas Review", layout="wide")
st.title("Vegas Review")
st.caption(
    "Evaluate how often Vegas totals and spreads matched actual results, then fit calibration models "
    "to improve projection guard rails."
)

default_bucket = os.getenv("CBB_GCS_BUCKET", "").strip() or (_secret("cbb_gcs_bucket") or "")
default_project = os.getenv("GCP_PROJECT", "").strip() or (_secret("gcp_project") or "")
default_start = season_start_for_date(date.today())
default_end = prior_day()

with st.sidebar:
    st.header("Vegas Review Settings")
    start_date = st.date_input("Start Date", value=default_start, key="vegas_review_start_date")
    end_date = st.date_input("End Date", value=default_end, key="vegas_review_end_date")
    bucket_name = st.text_input("GCS Bucket", value=default_bucket, key="vegas_review_bucket")
    gcp_project = st.text_input("GCP Project (optional)", value=default_project, key="vegas_review_project")
    run_review_clicked = st.button("Run Vegas Review", key="run_vegas_review")
    if st.button("Clear Cached Results", key="clear_vegas_review_cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

if start_date > end_date:
    st.error("Start Date must be on or before End Date.")
    st.stop()

if not run_review_clicked:
    st.info("Select your date range and click `Run Vegas Review`.")
    st.stop()

if not bucket_name.strip():
    st.error("Set a GCS bucket to run Vegas Review.")
    st.stop()

cred_json = _resolve_credential_json()
cred_json_b64 = _resolve_credential_json_b64()

with st.spinner("Loading game and odds history from GCS..."):
    games_df, meta = load_vegas_review_dataset(
        bucket_name=bucket_name.strip(),
        start_date=start_date,
        end_date=end_date,
        gcp_project=gcp_project.strip() or None,
        service_account_json=cred_json,
        service_account_json_b64=cred_json_b64,
    )
    player_totals_df, player_totals_meta = load_projection_player_totals_dataset(
        bucket_name=bucket_name.strip(),
        start_date=start_date,
        end_date=end_date,
        gcp_project=gcp_project.strip() or None,
        service_account_json=cred_json,
        service_account_json_b64=cred_json_b64,
    )

if games_df.empty:
    st.warning("No merged game-level results found for the selected range.")
    st.json(meta)
    st.stop()

summary = summarize_vegas_accuracy(games_df)
model_df = build_calibration_models_frame(games_df)
total_bucket_df = build_total_buckets_frame(games_df)
spread_bucket_df = build_spread_buckets_frame(games_df)

meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)
meta_col1.metric("Dates Scanned", int(meta["dates_scanned"]))
meta_col2.metric("Raw Days Found", int(meta["raw_days_found"]))
meta_col3.metric("Odds Days Found", int(meta["odds_days_found"]))
meta_col4.metric("Projection Days Found", int(player_totals_meta.get("projection_days_found") or 0))

m1, m2, m3, m4 = st.columns(4)
m1.metric("Games (Final)", int(summary["total_games"]))
m2.metric("Games Matched to Odds", int(summary["odds_matched_games"]))
m3.metric("Total MAE", f"{float(summary['total_mae']):.2f}")
m4.metric("Spread MAE", f"{float(summary['spread_mae']):.2f}")

m5, m6, m7, m8 = st.columns(4)
m5.metric("Winner Pick Accuracy", f"{float(summary['winner_pick_accuracy_pct']):.1f}%")
m6.metric("Totals Within 5", f"{float(summary['total_within_5_pct']):.1f}%")
m7.metric("Spreads Within 3", f"{float(summary['spread_within_3_pct']):.1f}%")
m8.metric("Over Rate", f"{float(summary['over_rate_pct']):.1f}%")

st.subheader("Calibration Models")
st.caption(
    "Use these linear fits as a first-pass correction model. Negative `mae_delta` means calibrated "
    "predictions beat raw Vegas lines in this sample."
)
st.dataframe(model_df, hide_index=True, use_container_width=True)

if not model_df.empty:
    for _, row in model_df.iterrows():
        label = str(row.get("model") or "")
        slope = float(row.get("slope") or 0.0)
        intercept = float(row.get("intercept") or 0.0)
        if "Total" in label:
            st.code(f"Adjusted Total = {intercept:.3f} + ({slope:.3f} * Vegas Total)")
        elif "Spread" in label:
            st.code(f"Adjusted Home Margin = {intercept:.3f} + ({slope:.3f} * Vegas Home Margin)")

st.subheader("Totals by Vegas Bucket")
st.dataframe(total_bucket_df, hide_index=True, use_container_width=True)

st.subheader("Spreads by Vegas Bucket")
st.dataframe(spread_bucket_df, hide_index=True, use_container_width=True)

st.subheader("Game Total -> Player Total Correlation")
st.caption(
    "Correlation between game total lines and player-level totals from saved `Slate + Vegas` snapshots "
    "(higher positive values mean high-total games are associated with higher player totals)."
)
cfg1, cfg2, cfg3 = st.columns(3)
top_n_players = int(
    cfg1.slider(
        "Top-N Players Per Game",
        min_value=2,
        max_value=8,
        value=3,
        step=1,
        help="Used for the game-level top-N aggregate correlations.",
    )
)
min_samples_gate = int(
    cfg2.number_input(
        "Min Samples Gate",
        min_value=10,
        max_value=5000,
        value=100,
        step=10,
        help="Rows under this sample size are hidden unless explicitly shown.",
    )
)
show_below_gate = bool(
    cfg3.checkbox(
        "Show Below-Gate Rows",
        value=False,
        help="Display rows with sample counts below the minimum gate.",
    )
)

player_total_corr_df = build_game_total_player_total_correlation_frame(
    player_totals_df,
    top_n_players=top_n_players,
)
if player_total_corr_df.empty:
    st.info("No projection snapshots found in this range, so player-total correlations are unavailable.")
else:
    primary = player_total_corr_df.loc[
        (player_total_corr_df["metric"] == "game_total_line_vs_vegas_points_line")
        & (player_total_corr_df["level"] == "player")
        & (player_total_corr_df["weighting"] == "none")
    ]
    primary_samples = int(primary["samples"].iloc[0]) if not primary.empty else 0
    primary_spearman = (
        float(primary["spearman"].iloc[0]) if (not primary.empty and pd.notna(primary["spearman"].iloc[0])) else None
    )
    primary_pearson = (
        float(primary["pearson"].iloc[0]) if (not primary.empty and pd.notna(primary["pearson"].iloc[0])) else None
    )
    pc1, pc2, pc3, pc4 = st.columns(4)
    pc1.metric("Samples (Points Line)", primary_samples)
    pc2.metric(
        "Spearman (Points Line)",
        f"{primary_spearman:.3f}" if primary_spearman is not None else "n/a",
    )
    pc3.metric(
        "Pearson (Points Line)",
        f"{primary_pearson:.3f}" if primary_pearson is not None else "n/a",
    )
    pc4.metric("Sample Gate", "PASS" if primary_samples >= min_samples_gate else f"LOW (<{min_samples_gate})")

    corr_display_df = player_total_corr_df.copy()
    corr_display_df["samples"] = pd.to_numeric(corr_display_df["samples"], errors="coerce").fillna(0).astype(int)
    corr_display_df["sample_gate"] = corr_display_df["samples"].map(
        lambda n: "pass" if int(n) >= min_samples_gate else "below"
    )
    if not show_below_gate:
        corr_display_df = corr_display_df.loc[corr_display_df["samples"] >= min_samples_gate].copy()

    if corr_display_df.empty:
        st.warning(
            "All correlation rows are below the current minimum sample gate. "
            "Lower `Min Samples Gate` or enable `Show Below-Gate Rows`."
        )
    else:
        st.dataframe(corr_display_df, hide_index=True, use_container_width=True)

st.subheader("Game/Odds Join Audit")
st.caption(
    "Audit mapping quality and data availability used to connect slate rows to game-level odds/tail features. "
    "Use this before interpreting total-line correlations."
)
audit_threshold = float(
    st.slider(
        "Low Match Score Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.01,
        help=(
            "Rows below this `game_tail_match_score` are flagged as weak joins. "
            "Tail mapping currently accepts matches around 0.57+, so values near that floor are higher risk."
        ),
    )
)
audit_summary, audit_by_date_df, low_match_rows_df = _build_join_audit_outputs(
    player_totals_df,
    low_match_threshold=audit_threshold,
)
as1, as2, as3, as4 = st.columns(4)
as1.metric("Rows (Projection Snapshots)", int(audit_summary.get("rows_total") or 0))
as2.metric(
    "With Game Total Line",
    f"{int(audit_summary.get('rows_with_game_total_line') or 0)} ({float(audit_summary.get('pct_with_game_total_line') or 0.0):.1f}%)",
)
as3.metric(
    "With Match Score",
    f"{int(audit_summary.get('rows_with_game_tail_match_score') or 0)} ({float(audit_summary.get('pct_with_game_tail_match_score') or 0.0):.1f}%)",
)
as4.metric(
    "Low-Match Rows",
    f"{int(audit_summary.get('rows_with_low_match_score') or 0)} ({float(audit_summary.get('pct_low_match_within_scored') or 0.0):.1f}% of scored)",
)
bs1, bs2, bs3, bs4 = st.columns(4)
bs1.metric(
    "With Vegas Points Line",
    f"{int(audit_summary.get('rows_with_vegas_points_line') or 0)} ({float(audit_summary.get('pct_with_vegas_points_line') or 0.0):.1f}%)",
)
bs2.metric(
    "With Total + Points Line",
    f"{int(audit_summary.get('rows_with_total_and_points_line') or 0)} ({float(audit_summary.get('pct_with_total_and_points_line') or 0.0):.1f}%)",
)
bs3.metric(
    "Match Score Median",
    (
        f"{float(audit_summary.get('match_score_median')):.3f}"
        if audit_summary.get("match_score_median") is not None
        else "n/a"
    ),
)
bs4.metric(
    "Approx Unique Game Groups",
    f"{int(audit_summary.get('approx_unique_game_groups') or 0)} ({str(audit_summary.get('game_group_source') or 'n/a')})",
)
if not audit_by_date_df.empty:
    st.caption("Per-date join coverage and total-to-points-line diagnostics.")
    st.dataframe(audit_by_date_df, hide_index=True, use_container_width=True)
else:
    st.info("No per-date join audit rows available in this window.")

if not low_match_rows_df.empty:
    st.caption("Sample low-match rows (potentially risky joins).")
    st.dataframe(low_match_rows_df, hide_index=True, use_container_width=True)
else:
    st.caption("No low-match rows under current threshold.")

st.subheader("Game-Level Vegas Review")
show_cols = [
    "game_date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "actual_total",
    "total_points",
    "total_error",
    "total_abs_error",
    "actual_home_margin",
    "vegas_home_margin",
    "spread_error",
    "spread_abs_error",
    "actual_winner_side",
    "predicted_winner_side",
    "winner_pick_correct",
    "bookmakers_count",
    "event_id",
]
show_cols = [c for c in show_cols if c in games_df.columns]
view_df = games_df[show_cols].copy()
if "game_date" in view_df.columns:
    view_df = view_df.sort_values(["game_date", "home_team", "away_team"], ascending=[False, True, True])
st.dataframe(view_df, hide_index=True, use_container_width=True)
st.download_button(
    "Download Vegas Review CSV",
    data=view_df.to_csv(index=False),
    file_name=f"vegas_review_{start_date.isoformat()}_{end_date.isoformat()}.csv",
    mime="text/csv",
    key="download_vegas_review_csv",
)
