from __future__ import annotations

import io
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from college_basketball_dfs.cbb_ai_review import (
    AI_REVIEW_SYSTEM_PROMPT,
    DEFAULT_OPENAI_REVIEW_MODEL,
    OPENAI_REVIEW_MODEL_FALLBACKS,
    build_daily_ai_review_packet,
    build_global_ai_review_packet,
    build_global_ai_review_user_prompt,
    request_openai_review,
)
from college_basketball_dfs.cbb_backfill import iter_dates
from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_ncaa import prior_day
from college_basketball_dfs.cbb_tournament_review import build_projection_actual_comparison


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


def _to_ownership_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_ownership_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership"])
    out = df.copy()
    rename_map = {
        "id": "ID",
        "player_id": "ID",
        "name": "Name",
        "player_name": "Name",
        "team": "TeamAbbrev",
        "team_abbrev": "TeamAbbrev",
        "teamabbrev": "TeamAbbrev",
        "ownership": "actual_ownership",
        "own%": "actual_ownership",
        "own_pct": "actual_ownership",
        "actual_own": "actual_ownership",
    }
    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    for col in ["ID", "Name", "TeamAbbrev", "actual_ownership"]:
        if col not in out.columns:
            out[col] = ""
    out["ID"] = out["ID"].astype(str).str.strip()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["TeamAbbrev"] = out["TeamAbbrev"].astype(str).str.strip().str.upper()
    out["actual_ownership"] = out["actual_ownership"].map(_to_ownership_float)
    out = out.loc[(out["ID"] != "") | (out["Name"] != "")]
    return out[["ID", "Name", "TeamAbbrev", "actual_ownership"]].reset_index(drop=True)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: object, default: int = 0) -> int:
    return int(_safe_float(value, float(default)))


@st.cache_data(ttl=600, show_spinner=False)
def load_saved_lineup_run_dates(
    bucket_name: str,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> list[date]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    return store.list_lineup_run_dates()


@st.cache_data(ttl=600, show_spinner=False)
def load_projection_snapshot_frame(
    bucket_name: str,
    selected_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> pd.DataFrame:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    csv_text = store.read_projections_csv(selected_date)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


@st.cache_data(ttl=600, show_spinner=False)
def load_ownership_frame_for_date(
    bucket_name: str,
    selected_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> pd.DataFrame:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    csv_text = store.read_ownership_csv(selected_date)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership"])
    raw_df = pd.read_csv(io.StringIO(csv_text))
    return _normalize_ownership_frame(raw_df)


@st.cache_data(ttl=600, show_spinner=False)
def load_actual_results_frame_for_date(
    bucket_name: str,
    selected_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> pd.DataFrame:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    blob_name = store.players_blob_name(selected_date)
    try:
        csv_text = store.read_players_csv_blob(blob_name)
    except Exception:
        return pd.DataFrame()
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(csv_text))
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
    cols = [c for c in needed if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = df[cols].copy().rename(
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


@st.cache_data(ttl=600, show_spinner=False)
def load_first_phantom_summary_for_date(
    bucket_name: str,
    selected_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> pd.DataFrame:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    run_ids = store.list_lineup_run_ids(selected_date)
    for run_id in run_ids:
        payload = store.read_phantom_review_summary_json(selected_date, run_id)
        if not isinstance(payload, dict):
            continue
        summary_rows = payload.get("summary_rows")
        if isinstance(summary_rows, list) and summary_rows:
            frame = pd.DataFrame(summary_rows)
            if not frame.empty:
                return frame
    return pd.DataFrame()


def _build_exposure_frame(projection_df: pd.DataFrame, ownership_df: pd.DataFrame) -> pd.DataFrame:
    if projection_df.empty:
        return pd.DataFrame()
    proj = projection_df.copy()
    if "ID" in proj.columns:
        proj["ID"] = proj["ID"].astype(str).str.strip()
    if "Name" in proj.columns:
        proj["Name"] = proj["Name"].astype(str).str.strip()

    own = ownership_df.copy() if isinstance(ownership_df, pd.DataFrame) else pd.DataFrame()
    if "ID" in own.columns:
        own["ID"] = own["ID"].astype(str).str.strip()
    if "Name" in own.columns:
        own["Name"] = own["Name"].astype(str).str.strip()

    if not own.empty and "ID" in proj.columns and "ID" in own.columns:
        expo = proj.merge(own[["ID", "actual_ownership"]], on="ID", how="left")
    elif not own.empty and "Name" in proj.columns and "Name" in own.columns:
        expo = proj.merge(own[["Name", "actual_ownership"]], on="Name", how="left")
    else:
        expo = proj.copy()
        expo["actual_ownership"] = pd.NA

    expo["projected_ownership"] = pd.to_numeric(expo.get("projected_ownership"), errors="coerce")
    expo["actual_ownership_from_file"] = pd.to_numeric(expo.get("actual_ownership"), errors="coerce")
    expo["ownership_diff_vs_proj"] = expo["actual_ownership_from_file"] - expo["projected_ownership"]
    expo["field_ownership_pct"] = expo["actual_ownership_from_file"]
    return expo


st.set_page_config(page_title="Global Agentic Review", layout="wide")
st.title("Global Agentic Review")
st.caption(
    "Multi-slate AI diagnostics and recommendations to improve projection quality, ownership quality, "
    "and lineup construction globally."
)

default_bucket = os.getenv("CBB_GCS_BUCKET", "").strip() or (_secret("cbb_gcs_bucket") or "")
default_project = os.getenv("GCP_PROJECT", "").strip() or (_secret("gcp_project") or "")
default_end_date = prior_day()
model_options = list(dict.fromkeys([DEFAULT_OPENAI_REVIEW_MODEL, *OPENAI_REVIEW_MODEL_FALLBACKS]))

with st.sidebar:
    st.header("Global Agentic Settings")
    bucket_name = st.text_input("GCS Bucket", value=default_bucket, key="global_agentic_bucket")
    gcp_project = st.text_input("GCP Project (optional)", value=default_project, key="global_agentic_project")
    end_date = st.date_input("Review End Date", value=default_end_date, key="global_agentic_end_date")
    lookback_days = int(
        st.slider("Lookback Days", min_value=7, max_value=180, value=30, step=1, key="global_agentic_lookback_days")
    )
    focus_limit = int(
        st.slider(
            "Global Focus Players",
            min_value=5,
            max_value=60,
            value=25,
            step=1,
            key="global_agentic_focus_limit",
        )
    )
    use_saved_run_dates = bool(
        st.checkbox(
            "Use Saved Run Dates Only",
            value=True,
            key="global_agentic_use_saved_dates",
            help="If enabled, only dates with saved lineup runs are scanned.",
        )
    )
    selected_model = st.selectbox(
        "OpenAI Model",
        options=model_options,
        index=0,
        key="global_agentic_model",
        help="Choose model for final recommendation generation.",
    )
    max_output_tokens = int(
        st.number_input(
            "Max Output Tokens",
            min_value=200,
            max_value=8000,
            value=1800,
            step=100,
            key="global_agentic_max_output_tokens",
        )
    )
    build_packet_clicked = st.button("Build Global Packet", key="global_agentic_build_packet")
    run_openai_clicked = st.button("Run OpenAI Review", key="global_agentic_run_openai")

if not bucket_name.strip():
    st.info("Set a GCS bucket to build a global review.")
    st.stop()

if end_date > date.today():
    st.error("Review End Date cannot be in the future.")
    st.stop()

cred_json = _resolve_credential_json()
cred_json_b64 = _resolve_credential_json_b64()
openai_key = (os.getenv("OPENAI_API_KEY", "").strip() or (_secret("openai_api_key") or "").strip())
st.caption(
    "OpenAI key: "
    + ("loaded from secrets/env (`OPENAI_API_KEY` or `openai_api_key`)" if openai_key else "not set")
)

if build_packet_clicked:
    with st.spinner("Building global packet from historical slates..."):
        cutoff = end_date - timedelta(days=max(1, lookback_days) - 1)
        if use_saved_run_dates:
            candidate_dates = load_saved_lineup_run_dates(
                bucket_name=bucket_name.strip(),
                gcp_project=gcp_project.strip() or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
        else:
            candidate_dates = [d for d in iter_dates(cutoff, end_date)]
        candidate_dates = [
            d for d in candidate_dates if isinstance(d, date) and cutoff <= d <= end_date
        ]
        candidate_dates = sorted(set(candidate_dates))

        daily_packets: list[dict[str, Any]] = []
        scanned_dates = 0
        used_dates = 0
        for review_day in candidate_dates:
            scanned_dates += 1
            proj_df = load_projection_snapshot_frame(
                bucket_name=bucket_name.strip(),
                selected_date=review_day,
                gcp_project=gcp_project.strip() or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            actual_df = load_actual_results_frame_for_date(
                bucket_name=bucket_name.strip(),
                selected_date=review_day,
                gcp_project=gcp_project.strip() or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if proj_df.empty or actual_df.empty:
                continue

            proj_compare_df = build_projection_actual_comparison(
                projection_df=proj_df,
                actual_results_df=actual_df,
            )
            if proj_compare_df.empty:
                continue

            own_df = load_ownership_frame_for_date(
                bucket_name=bucket_name.strip(),
                selected_date=review_day,
                gcp_project=gcp_project.strip() or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            exposure_df = _build_exposure_frame(proj_df, own_df)
            phantom_summary_df = load_first_phantom_summary_for_date(
                bucket_name=bucket_name.strip(),
                selected_date=review_day,
                gcp_project=gcp_project.strip() or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            day_packet = build_daily_ai_review_packet(
                review_date=review_day.isoformat(),
                contest_id="global-review",
                projection_comparison_df=proj_compare_df,
                entries_df=pd.DataFrame(),
                exposure_df=exposure_df,
                phantom_summary_df=phantom_summary_df,
                phantom_lineups_df=pd.DataFrame(),
                adjustment_factors_df=pd.DataFrame(),
                focus_limit=focus_limit,
            )
            daily_packets.append(day_packet)
            used_dates += 1

        global_packet = build_global_ai_review_packet(
            daily_packets=daily_packets,
            focus_limit=focus_limit,
        )
        global_user_prompt = build_global_ai_review_user_prompt(global_packet)
        st.session_state["global_agentic_packet"] = global_packet
        st.session_state["global_agentic_prompt"] = global_user_prompt
        st.session_state["global_agentic_meta"] = {
            "scanned_dates": scanned_dates,
            "used_dates": used_dates,
            "window_start": cutoff.isoformat(),
            "window_end": end_date.isoformat(),
            "lookback_days": lookback_days,
            "use_saved_run_dates": use_saved_run_dates,
        }
        st.success(f"Built global packet using {used_dates} of {scanned_dates} scanned dates.")

global_packet_state = st.session_state.get("global_agentic_packet")
global_user_prompt = str(st.session_state.get("global_agentic_prompt") or "").strip()
global_meta = st.session_state.get("global_agentic_meta") or {}

if isinstance(global_packet_state, dict) and global_packet_state:
    summary = global_packet_state.get("window_summary") or {}
    scorecards = global_packet_state.get("global_scorecards") or {}
    projection = scorecards.get("projection") or {}
    ownership = scorecards.get("ownership") or {}
    lineup = scorecards.get("lineup") or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Slates Used", _safe_int(summary.get("slate_count"), default=0))
    m2.metric("Weighted Blend MAE", f"{_safe_float(projection.get('weighted_blend_mae')):.2f}")
    m3.metric("Weighted Ownership MAE", f"{_safe_float(ownership.get('weighted_ownership_mae')):.2f}")
    m4.metric("Lineups Scored", _safe_int(lineup.get("total_lineups_scored"), default=0))

    st.caption(
        "Packet build summary: "
        f"scanned_dates={_safe_int(global_meta.get('scanned_dates'), default=0)}, "
        f"used_dates={_safe_int(global_meta.get('used_dates'), default=0)}, "
        f"window={global_meta.get('window_start', '')} to {global_meta.get('window_end', '')}"
    )

    packet_json = json.dumps(global_packet_state, indent=2, ensure_ascii=True)
    st.download_button(
        "Download Global Packet JSON",
        data=packet_json,
        file_name=f"global_agentic_packet_{end_date.isoformat()}.json",
        mime="application/json",
        key="download_global_agentic_packet_json",
    )
    st.download_button(
        "Download Global User Prompt TXT",
        data=global_user_prompt,
        file_name=f"global_agentic_prompt_{end_date.isoformat()}.txt",
        mime="text/plain",
        key="download_global_agentic_prompt_txt",
    )

    with st.expander("Global Packet Preview"):
        st.json(global_packet_state)
    with st.expander("Global Prompt Preview"):
        st.text_area(
            "Prompt",
            value=global_user_prompt,
            height=280,
            key="global_agentic_prompt_preview",
        )
else:
    st.info("Click `Build Global Packet` in the sidebar to start.")

if run_openai_clicked:
    if not openai_key:
        st.error("Set `OPENAI_API_KEY` or Streamlit secret `openai_api_key` first.")
    elif not global_user_prompt:
        st.error("Build a global packet first.")
    else:
        with st.spinner("Generating global recommendations..."):
            try:
                output = request_openai_review(
                    api_key=openai_key,
                    user_prompt=global_user_prompt,
                    system_prompt=AI_REVIEW_SYSTEM_PROMPT,
                    model=selected_model,
                    max_output_tokens=max_output_tokens,
                )
                st.session_state["global_agentic_output"] = output
            except Exception as exc:
                st.exception(exc)

global_output = str(st.session_state.get("global_agentic_output") or "").strip()
if global_output:
    st.subheader("Global AI Recommendations")
    st.text_area(
        "Model Output",
        value=global_output,
        height=380,
        key="global_agentic_output_preview",
    )
    st.download_button(
        "Download Recommendations TXT",
        data=global_output,
        file_name=f"global_agentic_recommendations_{end_date.isoformat()}.txt",
        mime="text/plain",
        key="download_global_agentic_recommendations_txt",
    )
