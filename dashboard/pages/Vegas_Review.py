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


def build_game_total_player_total_correlation_frame(player_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["metric", "x_col", "y_col", "samples", "pearson", "spearman"]
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
            }
        )

    return pd.DataFrame(rows, columns=columns)


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
player_total_corr_df = build_game_total_player_total_correlation_frame(player_totals_df)

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
if player_total_corr_df.empty:
    st.info("No projection snapshots found in this range, so player-total correlations are unavailable.")
else:
    primary = player_total_corr_df.loc[
        player_total_corr_df["metric"] == "game_total_line_vs_vegas_points_line"
    ]
    primary_samples = int(primary["samples"].iloc[0]) if not primary.empty else 0
    primary_spearman = (
        float(primary["spearman"].iloc[0]) if (not primary.empty and pd.notna(primary["spearman"].iloc[0])) else None
    )
    primary_pearson = (
        float(primary["pearson"].iloc[0]) if (not primary.empty and pd.notna(primary["pearson"].iloc[0])) else None
    )
    pc1, pc2, pc3 = st.columns(3)
    pc1.metric("Samples (Points Line)", primary_samples)
    pc2.metric(
        "Spearman (Points Line)",
        f"{primary_spearman:.3f}" if primary_spearman is not None else "n/a",
    )
    pc3.metric(
        "Pearson (Points Line)",
        f"{primary_pearson:.3f}" if primary_pearson is not None else "n/a",
    )
    st.dataframe(player_total_corr_df, hide_index=True, use_container_width=True)

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
