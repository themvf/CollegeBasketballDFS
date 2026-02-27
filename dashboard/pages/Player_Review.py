from __future__ import annotations

import io
import os
import re
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
from college_basketball_dfs.cbb_tournament_review import build_projection_actual_comparison


PROJECTED_OWNERSHIP_ALIASES = (
    "projected_ownership",
    "projected ownership",
    "projectedownership",
    "proj_ownership",
    "projownership",
    "ownership_projection",
    "ownershipprojection",
    "ownership_proj",
    "ownershipproj",
    "own_pct",
    "own%",
    "ownership",
    "projected_ownership_v1",
)
NULLISH_TEXT_VALUES = {"", "nan", "none", "null"}


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


def _build_actual_results_from_players_frame(players_df: pd.DataFrame) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    needed = [
        "player_id",
        "player_name",
        "team_name",
        "minutes_played",
        "points",
        "rebounds",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "tpm",
    ]
    cols = [c for c in needed if c in players_df.columns]
    if not cols:
        return pd.DataFrame()

    out = players_df[cols].copy().rename(
        columns={
            "player_id": "ID",
            "player_name": "Name",
            "team_name": "team_name",
            "minutes_played": "actual_minutes",
            "points": "actual_points",
            "rebounds": "actual_rebounds",
            "assists": "actual_assists",
            "steals": "actual_steals",
            "blocks": "actual_blocks",
            "turnovers": "actual_turnovers",
            "tpm": "actual_threes",
        }
    )
    for col in [
        "actual_minutes",
        "actual_points",
        "actual_rebounds",
        "actual_assists",
        "actual_steals",
        "actual_blocks",
        "actual_turnovers",
        "actual_threes",
    ]:
        out[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(0.0)

    dd_count = (
        (out["actual_points"] >= 10).astype(int)
        + (out["actual_rebounds"] >= 10).astype(int)
        + (out["actual_assists"] >= 10).astype(int)
        + (out["actual_steals"] >= 10).astype(int)
        + (out["actual_blocks"] >= 10).astype(int)
    )
    bonus = (dd_count >= 2).astype(int) * 1.5 + (dd_count >= 3).astype(int) * 3.0
    out["actual_dk_points"] = (
        out["actual_points"]
        + (1.25 * out["actual_rebounds"])
        + (1.5 * out["actual_assists"])
        + (2.0 * out["actual_steals"])
        + (2.0 * out["actual_blocks"])
        - (0.5 * out["actual_turnovers"])
        + (0.5 * out["actual_threes"])
        + bonus
    )
    out["ID"] = out["ID"].astype(str).str.strip()
    out["Name"] = out["Name"].astype(str).str.strip()
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_projection_actual_comparison_dataset(
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
    actual_days = 0
    comparison_days = 0

    for d in dates:
        proj_csv = store.read_projections_csv(d)
        if not proj_csv or not proj_csv.strip():
            continue
        try:
            proj_df = pd.read_csv(io.StringIO(proj_csv))
        except Exception:
            continue
        if proj_df.empty:
            continue
        projection_days += 1

        try:
            players_blob = store.players_blob_name(d)
            players_csv = store.read_players_csv_blob(players_blob)
        except Exception:
            players_csv = ""
        if not players_csv or not players_csv.strip():
            continue
        try:
            players_df = pd.read_csv(io.StringIO(players_csv))
        except Exception:
            continue
        if players_df.empty:
            continue
        actual_days += 1

        actual_df = _build_actual_results_from_players_frame(players_df)
        if actual_df.empty:
            continue
        comparison_df = build_projection_actual_comparison(
            projection_df=proj_df,
            actual_results_df=actual_df,
        )
        if comparison_df.empty:
            continue
        comparison_days += 1
        comparison_df = comparison_df.copy()
        comparison_df["review_date"] = d.isoformat()
        frames.append(comparison_df)

    if not frames:
        return pd.DataFrame(), {
            "projection_days_found": int(projection_days),
            "actual_days_found": int(actual_days),
            "comparison_days_found": int(comparison_days),
            "comparison_rows": 0,
        }

    out = pd.concat(frames, ignore_index=True)
    return out, {
        "projection_days_found": int(projection_days),
        "actual_days_found": int(actual_days),
        "comparison_days_found": int(comparison_days),
        "comparison_rows": int(len(out)),
    }


def _norm_col_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _resolve_column_alias(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    if df is None or df.empty:
        return None
    normalized = {_norm_col_key(c): str(c) for c in df.columns}
    for alias in aliases:
        alias_key = _norm_col_key(alias)
        if alias_key in normalized:
            return normalized[alias_key]
    return None


def _norm_text_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _first_nonempty(series: pd.Series) -> str:
    for value in series:
        text = str(value or "").strip()
        if text and text.lower() not in NULLISH_TEXT_VALUES:
            return text
    return ""


def _coerce_ownership_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace("%", "", regex=False)
    text = text.mask(text.str.lower().isin(NULLISH_TEXT_VALUES), pd.NA)
    out = pd.to_numeric(text, errors="coerce")
    valid = out.dropna()
    if not valid.empty and float(valid.quantile(0.95)) <= 1.0:
        out = out * 100.0
    return out


def _build_player_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "player_key",
        "player_name",
        "team",
        "position",
        "review_date",
        "row_order",
        "actual_dk_points",
        "projected_ownership",
        "salary",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    id_col = _resolve_column_alias(work, ("ID", "id", "player_id", "playerid"))
    name_col = _resolve_column_alias(work, ("Name", "name", "player_name", "player"))
    team_col = _resolve_column_alias(work, ("TeamAbbrev", "team_abbrev", "team_name", "team", "Team"))
    position_col = _resolve_column_alias(work, ("Position", "position", "Roster Position", "roster_position", "Pos", "pos"))
    review_date_col = _resolve_column_alias(work, ("review_date", "review date", "slate_date", "game_date", "date"))
    salary_col = _resolve_column_alias(work, ("Salary", "salary", "dk_salary", "dk salary"))
    points_col = _resolve_column_alias(work, ("actual_dk_points", "actual dk points", "final_dk_points", "dk_points"))
    ownership_col = _resolve_column_alias(work, PROJECTED_OWNERSHIP_ALIASES)

    out = pd.DataFrame(index=work.index)
    out["player_id"] = work[id_col] if id_col else ""
    out["player_name"] = work[name_col] if name_col else ""
    out["team"] = work[team_col] if team_col else ""
    out["position"] = work[position_col] if position_col else ""
    out["review_date"] = work[review_date_col] if review_date_col else pd.NA
    out["salary"] = pd.to_numeric(work[salary_col], errors="coerce") if salary_col else pd.NA
    out["actual_dk_points"] = pd.to_numeric(work[points_col], errors="coerce") if points_col else pd.NA
    out["projected_ownership"] = _coerce_ownership_series(work[ownership_col]) if ownership_col else pd.NA
    out["row_order"] = pd.RangeIndex(start=0, stop=len(out), step=1)

    out["player_id"] = out["player_id"].astype(str).str.strip()
    out["player_name"] = out["player_name"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["position"] = out["position"].astype(str).str.strip().str.upper()
    out["player_name_key"] = out["player_name"].map(_norm_text_key)
    out["player_id_key"] = out["player_id"].map(_norm_text_key)

    out["player_key"] = ""
    has_id_key = out["player_id_key"] != ""
    out.loc[has_id_key, "player_key"] = "id:" + out.loc[has_id_key, "player_id_key"]
    missing_key = out["player_key"] == ""
    has_name_key = out["player_name_key"] != ""
    use_name_key = missing_key & has_name_key
    out.loc[use_name_key, "player_key"] = (
        "name:" + out.loc[use_name_key, "player_name_key"] + "|team:" + out.loc[use_name_key, "team"]
    )
    out = out.loc[out["player_key"] != ""].copy()
    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce")
    out["team"] = out["team"].replace({"NAN": "", "NONE": "", "NULL": ""})
    out["position"] = out["position"].replace({"NAN": "", "NONE": "", "NULL": ""})

    return out[cols].copy()


def _aggregate_player_metric(
    history_df: pd.DataFrame,
    *,
    value_col: str,
    season_col: str,
    last5_col: str,
    agg_mode: str,
) -> pd.DataFrame:
    if history_df is None or history_df.empty or value_col not in history_df.columns:
        return pd.DataFrame(columns=["player_key", season_col, last5_col])

    work = history_df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[work["player_key"].astype(str).str.strip() != ""].copy()
    work = work.loc[work[value_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["player_key", season_col, last5_col])

    work["_sort_date"] = pd.to_datetime(work.get("review_date"), errors="coerce")
    work["_sort_ord"] = pd.to_numeric(work.get("row_order"), errors="coerce").fillna(0.0)
    work = work.sort_values(
        ["player_key", "_sort_date", "_sort_ord"],
        ascending=[True, False, False],
        kind="stable",
    )
    last5 = work.groupby("player_key", sort=False, as_index=False).head(5).copy()

    if agg_mode == "mean":
        season = (
            work.groupby("player_key", as_index=False)[value_col]
            .mean(numeric_only=True)
            .rename(columns={value_col: season_col})
        )
        recent = (
            last5.groupby("player_key", as_index=False)[value_col]
            .mean(numeric_only=True)
            .rename(columns={value_col: last5_col})
        )
    else:
        season = (
            work.groupby("player_key", as_index=False)[value_col]
            .sum(numeric_only=True)
            .rename(columns={value_col: season_col})
        )
        recent = (
            last5.groupby("player_key", as_index=False)[value_col]
            .sum(numeric_only=True)
            .rename(columns={value_col: last5_col})
        )

    return season.merge(recent, on="player_key", how="outer")


def build_player_review_table(
    projection_actual_df: pd.DataFrame,
    player_totals_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], dict[str, bool]]:
    actual_history = _build_player_history_frame(projection_actual_df)
    projection_history = _build_player_history_frame(player_totals_df)

    identity_frames: list[pd.DataFrame] = []
    for frame in [actual_history, projection_history]:
        if frame.empty:
            continue
        identity_frames.append(frame[["player_key", "player_name", "team", "position", "review_date", "row_order"]].copy())

    if not identity_frames:
        meta = {"has_points": False, "has_ownership": False, "has_salary": False}
        return pd.DataFrame(), [], meta

    identity = pd.concat(identity_frames, ignore_index=True)
    identity["review_date"] = pd.to_datetime(identity.get("review_date"), errors="coerce")
    identity["_sort_date"] = identity["review_date"]
    identity["_sort_ord"] = pd.to_numeric(identity.get("row_order"), errors="coerce").fillna(0.0)
    identity = identity.sort_values(
        ["player_key", "_sort_date", "_sort_ord"],
        ascending=[True, False, False],
        kind="stable",
    )
    identity_summary = (
        identity.groupby("player_key", as_index=False)
        .agg(
            player_name=("player_name", _first_nonempty),
            team=("team", _first_nonempty),
            position=("position", _first_nonempty),
        )
        .copy()
    )

    points_summary = _aggregate_player_metric(
        actual_history,
        value_col="actual_dk_points",
        season_col="Total Fantasy Points Season",
        last5_col="Total Fantasy Points Last 5 Games",
        agg_mode="sum",
    )
    ownership_summary = _aggregate_player_metric(
        projection_history,
        value_col="projected_ownership",
        season_col="Average Ownership Season",
        last5_col="Average Ownership Last 5 Games",
        agg_mode="mean",
    )

    salary_source = actual_history if actual_history["salary"].notna().any() else projection_history
    salary_summary = _aggregate_player_metric(
        salary_source,
        value_col="salary",
        season_col="Average DK Salary This Season",
        last5_col="_unused_last5_salary",
        agg_mode="mean",
    )
    if "_unused_last5_salary" in salary_summary.columns:
        salary_summary = salary_summary.drop(columns=["_unused_last5_salary"])

    out = identity_summary.merge(points_summary, on="player_key", how="left")
    out = out.merge(ownership_summary, on="player_key", how="left")
    out = out.merge(salary_summary, on="player_key", how="left")
    out = out.rename(columns={"player_name": "Player Name", "position": "Position", "team": "Team"})

    for col in [
        "Total Fantasy Points Season",
        "Total Fantasy Points Last 5 Games",
        "Average Ownership Season",
        "Average Ownership Last 5 Games",
        "Average DK Salary This Season",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in ["Total Fantasy Points Season", "Total Fantasy Points Last 5 Games"]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    for col in [
        "Total Fantasy Points Season",
        "Total Fantasy Points Last 5 Games",
        "Average Ownership Season",
        "Average Ownership Last 5 Games",
        "Average DK Salary This Season",
    ]:
        if col in out.columns:
            out[col] = out[col].round(2)

    out["Player Name"] = out["Player Name"].astype(str).str.strip()
    out["Team"] = out["Team"].astype(str).str.strip().str.upper()
    out["Position"] = out["Position"].astype(str).str.strip().str.upper()
    out = out.loc[out["Player Name"] != ""].copy()
    out = out.sort_values(
        ["Team", "Total Fantasy Points Season", "Player Name"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    team_options = sorted([str(x).strip().upper() for x in out["Team"].dropna().unique().tolist() if str(x).strip()])
    meta = {
        "has_points": bool(points_summary["Total Fantasy Points Season"].notna().any()) if not points_summary.empty else False,
        "has_ownership": bool(ownership_summary["Average Ownership Season"].notna().any())
        if not ownership_summary.empty
        else False,
        "has_salary": bool(salary_summary["Average DK Salary This Season"].notna().any()) if not salary_summary.empty else False,
    }
    return out, team_options, meta


st.set_page_config(page_title="Player Review", layout="wide")
st.title("Player Review")
st.caption("Review player fantasy totals, ownership trends, and average salary by team.")

default_bucket = os.getenv("CBB_GCS_BUCKET", "").strip() or (_secret("cbb_gcs_bucket") or "")
default_project = os.getenv("GCP_PROJECT", "").strip() or (_secret("gcp_project") or "")
default_start = season_start_for_date(date.today())
default_end = prior_day()

with st.sidebar:
    st.header("Player Review Settings")
    start_date = st.date_input("Start Date", value=default_start, key="player_review_start_date")
    end_date = st.date_input("End Date", value=default_end, key="player_review_end_date")
    bucket_name = st.text_input("GCS Bucket", value=default_bucket, key="player_review_bucket")
    gcp_project = st.text_input("GCP Project (optional)", value=default_project, key="player_review_project")
    run_review_clicked = st.button("Run Player Review", key="run_player_review")
    if st.button("Clear Cached Results", key="clear_player_review_cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

if start_date > end_date:
    st.error("Start Date must be on or before End Date.")
    st.stop()

if not run_review_clicked:
    st.info("Select your date range and click `Run Player Review`.")
    st.stop()

if not bucket_name.strip():
    st.error("Set a GCS bucket to run Player Review.")
    st.stop()

cred_json = _resolve_credential_json()
cred_json_b64 = _resolve_credential_json_b64()

with st.spinner("Loading projection snapshots and actual results from GCS..."):
    player_totals_df, player_totals_meta = load_projection_player_totals_dataset(
        bucket_name=bucket_name.strip(),
        start_date=start_date,
        end_date=end_date,
        gcp_project=gcp_project.strip() or None,
        service_account_json=cred_json,
        service_account_json_b64=cred_json_b64,
    )
    projection_actual_df, projection_actual_meta = load_projection_actual_comparison_dataset(
        bucket_name=bucket_name.strip(),
        start_date=start_date,
        end_date=end_date,
        gcp_project=gcp_project.strip() or None,
        service_account_json=cred_json,
        service_account_json_b64=cred_json_b64,
    )

m1, m2, m3, m4 = st.columns(4)
m1.metric("Projection Days", int(player_totals_meta.get("projection_days_found") or 0))
m2.metric("Projection Rows", int(player_totals_meta.get("projection_rows") or 0))
m3.metric("Proj-Actual Days", int(projection_actual_meta.get("comparison_days_found") or 0))
m4.metric("Proj-Actual Rows", int(projection_actual_meta.get("comparison_rows") or 0))

player_review_df, player_review_teams, player_review_meta = build_player_review_table(
    projection_actual_df=projection_actual_df,
    player_totals_df=player_totals_df,
)

if player_review_df.empty or not player_review_teams:
    st.warning(
        "No player review rows were found for this range. "
        "This page needs projection snapshots and at least one valid player mapping."
    )
    st.stop()

selected_team = st.selectbox("Team", options=player_review_teams, index=0, key="player_review_team_select")
team_df = player_review_df.loc[
    player_review_df["Team"].astype(str).str.strip().str.upper() == str(selected_team).strip().upper()
].copy()

show_cols = [
    "Player Name",
    "Position",
    "Total Fantasy Points Season",
    "Total Fantasy Points Last 5 Games",
    "Average Ownership Season",
    "Average Ownership Last 5 Games",
    "Average DK Salary This Season",
]
show_cols = [c for c in show_cols if c in team_df.columns]

if team_df.empty:
    st.info("No player rows found for the selected team in this date range.")
else:
    team_df = team_df.sort_values(
        ["Total Fantasy Points Season", "Player Name"],
        ascending=[False, True],
        kind="stable",
    )
    st.dataframe(team_df[show_cols], hide_index=True, use_container_width=True)
    st.download_button(
        "Download Team Player Review CSV",
        data=team_df[show_cols].to_csv(index=False),
        file_name=f"player_review_{selected_team}_{start_date.isoformat()}_{end_date.isoformat()}.csv",
        mime="text/csv",
        key="download_player_review_csv",
    )

if not bool(player_review_meta.get("has_points")):
    st.caption("Note: Total fantasy points are unavailable because no matched actual DK points were found.")
if not bool(player_review_meta.get("has_ownership")):
    st.caption("Note: Ownership columns are unavailable because projection ownership values were not found.")
if not bool(player_review_meta.get("has_salary")):
    st.caption("Note: Average DK salary is unavailable because salary values were not found.")
