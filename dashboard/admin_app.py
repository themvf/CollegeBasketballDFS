from __future__ import annotations

import io
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_backfill import run_season_backfill, season_start_for_date
from college_basketball_dfs.cbb_ncaa import prior_day
from college_basketball_dfs.cbb_odds_backfill import run_odds_season_backfill
from college_basketball_dfs.cbb_odds import flatten_odds_payload
from college_basketball_dfs.cbb_odds_pipeline import run_cbb_odds_pipeline
from college_basketball_dfs.cbb_pipeline import run_cbb_cache_pipeline
from college_basketball_dfs.cbb_team_lookup import rows_from_payloads


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


def _resolve_odds_api_key() -> str | None:
    return os.getenv("THE_ODDS_API_KEY") or _secret("the_odds_api_key")


@st.cache_data(ttl=600, show_spinner=False)
def load_team_lookup_frame(
    bucket_name: str,
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
    blob_names = store.list_raw_blob_names()
    payloads = [store.read_raw_json_blob(name) for name in blob_names]
    rows = rows_from_payloads(payloads)
    if not rows:
        return pd.DataFrame(
            columns=[
                "Team",
                "Game Date",
                "Venue",
                "Home/Away",
                "Team Score",
                "Opponent",
                "Opponent Score",
                "W/L",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["Game Date"] = pd.to_datetime(frame["Game Date"], errors="coerce")
    frame = frame.dropna(subset=["Team", "Game Date"]).sort_values(["Game Date", "Team"], ascending=[False, True])
    return frame


@st.cache_data(ttl=600, show_spinner=False)
def load_odds_frame_for_date(
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
    payload = store.read_odds_json(selected_date)
    if payload is None:
        return pd.DataFrame()

    rows = flatten_odds_payload(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame


st.set_page_config(page_title="CBB Admin Cache", layout="wide")
st.title("College Basketball Admin Cache")
st.caption("Cache-first data pipeline backed by Google Cloud Storage.")

default_bucket = os.getenv("CBB_GCS_BUCKET", "").strip() or (_secret("cbb_gcs_bucket") or "")
default_base_url = os.getenv("NCAA_API_BASE_URL", "https://ncaa-api.henrygd.me").strip()
default_project = os.getenv("GCP_PROJECT", "").strip() or (_secret("gcp_project") or "")
odds_api_key = (_resolve_odds_api_key() or "").strip()

with st.sidebar:
    st.header("Pipeline Settings")
    selected_date = st.date_input("Slate Date", value=prior_day())
    default_season_start = season_start_for_date(date.today())
    backfill_start = st.date_input("Backfill Start", value=default_season_start)
    backfill_end = st.date_input("Backfill End", value=prior_day())
    bucket_name = st.text_input("GCS Bucket", value=default_bucket)
    base_url = st.text_input("NCAA API Base URL", value=default_base_url)
    gcp_project = st.text_input("GCP Project (optional)", value=default_project)
    st.caption(
        "The Odds API key source: "
        + ("loaded from secrets/env" if odds_api_key else "missing (`the_odds_api_key`)")
    )
    force_refresh = st.checkbox("Force API refresh (ignore cached raw JSON)", value=False)
    backfill_sleep = st.number_input("Backfill Sleep Seconds", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    stop_on_error = st.checkbox("Stop Backfill On Error", value=False)
    run_clicked = st.button("Run Cache Pipeline")
    run_odds_clicked = st.button("Run Odds Import")
    run_backfill_clicked = st.button("Run Season Backfill")
    run_odds_backfill_clicked = st.button("Run Odds Season Backfill")
    preview_clicked = st.button("Load Cached Preview")

cred_json = _resolve_credential_json()
cred_json_b64 = _resolve_credential_json_b64()

if not cred_json and not cred_json_b64:
    st.info(
        "No inline GCP service-account secret detected. "
        "Using default Google credentials if available."
    )

if run_clicked:
    if not bucket_name:
        st.error("Set a GCS bucket before running.")
    else:
        with st.spinner("Running CBB cache pipeline..."):
            try:
                summary = run_cbb_cache_pipeline(
                    game_date=selected_date,
                    bucket_name=bucket_name,
                    ncaa_base_url=base_url,
                    force_refresh=force_refresh,
                    gcp_project=gcp_project or None,
                    gcp_service_account_json=cred_json,
                    gcp_service_account_json_b64=cred_json_b64,
                )
                load_team_lookup_frame.clear()
                st.session_state["cbb_last_summary"] = summary
                st.success("Pipeline completed.")
            except Exception as exc:
                st.exception(exc)

if run_odds_clicked:
    if not bucket_name:
        st.error("Set a GCS bucket before importing odds.")
    elif not odds_api_key:
        st.error("Set The Odds API key in sidebar or Streamlit secrets (`the_odds_api_key`).")
    else:
        with st.spinner("Importing game odds from The Odds API..."):
            try:
                summary = run_cbb_odds_pipeline(
                    game_date=selected_date,
                    bucket_name=bucket_name,
                    odds_api_key=odds_api_key,
                    historical_mode=(selected_date < date.today()),
                    historical_snapshot_time=f"{selected_date.isoformat()}T23:59:59Z" if selected_date < date.today() else None,
                    force_refresh=force_refresh,
                    gcp_project=gcp_project or None,
                    gcp_service_account_json=cred_json,
                    gcp_service_account_json_b64=cred_json_b64,
                )
                load_odds_frame_for_date.clear()
                st.session_state["cbb_odds_summary"] = summary
                st.success("Odds import completed.")
            except Exception as exc:
                st.exception(exc)

if run_backfill_clicked:
    if not bucket_name:
        st.error("Set a GCS bucket before running backfill.")
    elif backfill_start > backfill_end:
        st.error("Backfill start date must be before or equal to end date.")
    else:
        with st.spinner("Running season backfill... this can take several minutes."):
            try:
                result = run_season_backfill(
                    start_date=backfill_start,
                    end_date=backfill_end,
                    bucket_name=bucket_name,
                    ncaa_base_url=base_url,
                    force_refresh=force_refresh,
                    gcp_project=gcp_project or None,
                    gcp_service_account_json=cred_json,
                    gcp_service_account_json_b64=cred_json_b64,
                    sleep_seconds=float(backfill_sleep),
                    stop_on_error=stop_on_error,
                )
                load_team_lookup_frame.clear()
                payload = result.as_dict()
                st.session_state["cbb_backfill_summary"] = payload
                st.success("Season backfill completed.")
            except Exception as exc:
                st.exception(exc)

if run_odds_backfill_clicked:
    if not bucket_name:
        st.error("Set a GCS bucket before running odds backfill.")
    elif not odds_api_key:
        st.error("Set The Odds API key in Streamlit secrets (`the_odds_api_key`).")
    elif backfill_start > backfill_end:
        st.error("Backfill start date must be before or equal to end date.")
    else:
        with st.spinner("Running odds season backfill..."):
            try:
                result = run_odds_season_backfill(
                    start_date=backfill_start,
                    end_date=backfill_end,
                    bucket_name=bucket_name,
                    odds_api_key=odds_api_key,
                    historical_mode=True,
                    force_refresh=force_refresh,
                    gcp_project=gcp_project or None,
                    gcp_service_account_json=cred_json,
                    gcp_service_account_json_b64=cred_json_b64,
                    sleep_seconds=float(backfill_sleep),
                    stop_on_error=stop_on_error,
                )
                load_odds_frame_for_date.clear()
                st.session_state["cbb_odds_backfill_summary"] = result.as_dict()
                st.success("Odds season backfill completed.")
            except Exception as exc:
                st.exception(exc)

summary = st.session_state.get("cbb_last_summary")
if summary:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Games", summary["game_count"])
    c2.metric("Boxscores OK", summary["boxscore_success_count"])
    c3.metric("Boxscores Failed", summary["boxscore_failure_count"])
    c4.metric("Player Rows", summary["player_row_count"])
    st.json(summary)

odds_summary = st.session_state.get("cbb_odds_summary")
if odds_summary:
    st.subheader("Odds Import Summary")
    o1, o2, o3 = st.columns(3)
    o1.metric("Events", odds_summary["event_count"])
    o2.metric("Game Rows", odds_summary["odds_game_rows"])
    o3.metric("Cache Hit", "Yes" if odds_summary["odds_cache_hit"] else "No")
    st.json(odds_summary)

odds_backfill_summary = st.session_state.get("cbb_odds_backfill_summary")
if odds_backfill_summary:
    st.subheader("Odds Season Backfill Summary")
    ob1, ob2, ob3, ob4 = st.columns(4)
    ob1.metric("Total Dates", odds_backfill_summary["total_dates"])
    ob2.metric("Success Dates", odds_backfill_summary["success_dates"])
    ob3.metric("Failed Dates", odds_backfill_summary["failed_dates"])
    ob4.metric("Cache Hits", odds_backfill_summary["odds_cache_hits"])
    st.json(odds_backfill_summary)

backfill_summary = st.session_state.get("cbb_backfill_summary")
if backfill_summary:
    st.subheader("Season Backfill Summary")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Total Dates", backfill_summary["total_dates"])
    b2.metric("Success Dates", backfill_summary["success_dates"])
    b3.metric("Failed Dates", backfill_summary["failed_dates"])
    b4.metric("Cache Hits", backfill_summary["raw_cache_hits"])
    st.json(backfill_summary)

st.subheader("Game Odds")
if not bucket_name:
    st.info("Set a GCS bucket in sidebar to load game odds.")
else:
    try:
        odds_df = load_odds_frame_for_date(
            bucket_name=bucket_name,
            selected_date=selected_date,
            gcp_project=gcp_project or None,
            service_account_json=cred_json,
            service_account_json_b64=cred_json_b64,
        )
        if odds_df.empty:
            st.warning("No cached odds found for selected date. Run `Run Odds Import` first.")
        else:
            display = odds_df.rename(
                columns={
                    "game_date": "Game Date",
                    "home_team": "Home Team",
                    "away_team": "Away Team",
                    "spread_home": "Spread (Home)",
                    "spread_away": "Spread (Away)",
                    "total_points": "Total",
                    "moneyline_home": "Moneyline (Home)",
                    "moneyline_away": "Moneyline (Away)",
                }
            )
            table_cols = [
                "Game Date",
                "Home Team",
                "Away Team",
                "Spread (Home)",
                "Spread (Away)",
                "Total",
                "Moneyline (Home)",
                "Moneyline (Away)",
            ]
            st.dataframe(display[table_cols], hide_index=True, use_container_width=True)
    except Exception as exc:
        st.exception(exc)

st.subheader("Team Lookup")
if not bucket_name:
    st.info("Set a GCS bucket in sidebar to load team lookup data.")
else:
    try:
        team_df = load_team_lookup_frame(
            bucket_name=bucket_name,
            gcp_project=gcp_project or None,
            service_account_json=cred_json,
            service_account_json_b64=cred_json_b64,
        )
        if team_df.empty:
            st.warning("No cached raw game files found in bucket for team lookup.")
        else:
            min_date = team_df["Game Date"].min().date()
            max_date = team_df["Game Date"].max().date()
            c1, c2, c3 = st.columns([2, 1, 1])
            teams = sorted(team_df["Team"].dropna().unique().tolist())
            selected_team = c1.selectbox("Team", options=teams, index=0)
            date_start = c2.date_input("From", value=min_date, min_value=min_date, max_value=max_date, key="team_from")
            date_end = c3.date_input("To", value=max_date, min_value=min_date, max_value=max_date, key="team_to")

            filtered = team_df.loc[team_df["Team"] == selected_team].copy()
            filtered = filtered.loc[
                (filtered["Game Date"].dt.date >= date_start) & (filtered["Game Date"].dt.date <= date_end)
            ]
            filtered = filtered.sort_values("Game Date", ascending=False)
            filtered["Game Date"] = filtered["Game Date"].dt.strftime("%Y-%m-%d")
            table_cols = [
                "Game Date",
                "Venue",
                "Home/Away",
                "Team Score",
                "Opponent",
                "Opponent Score",
                "W/L",
            ]
            st.dataframe(filtered[table_cols], hide_index=True, use_container_width=True)
    except Exception as exc:
        st.exception(exc)

if preview_clicked:
    if not bucket_name:
        st.error("Set a GCS bucket before previewing.")
    else:
        with st.spinner("Loading cached files from GCS..."):
            try:
                client = build_storage_client(
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                    project=gcp_project or None,
                )
                store = CbbGcsStore(bucket_name=bucket_name, client=client)

                raw_payload = store.read_raw_json(selected_date)
                if raw_payload is None:
                    st.warning("No raw JSON cache found for this date.")
                else:
                    st.subheader("Raw JSON Snapshot")
                    st.json(
                        {
                            "game_date": raw_payload.get("game_date"),
                            "game_count": raw_payload.get("game_count"),
                            "boxscore_success_count": raw_payload.get("boxscore_success_count"),
                            "boxscore_failure_count": raw_payload.get("boxscore_failure_count"),
                        }
                    )

                players_blob = store.players_blob_name(selected_date)
                blob = store.bucket.blob(players_blob)
                if not blob.exists():
                    st.warning("No player CSV cache found for this date.")
                else:
                    csv_text = blob.download_as_text(encoding="utf-8")
                    if not csv_text.strip():
                        st.warning("Cached player CSV is empty.")
                    else:
                        df = pd.read_csv(io.StringIO(csv_text))
                        st.subheader("Player CSV Preview")
                        st.dataframe(df.head(200), hide_index=True, use_container_width=True)
            except Exception as exc:
                st.exception(exc)
