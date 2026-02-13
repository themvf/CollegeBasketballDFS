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
from college_basketball_dfs.cbb_ncaa import prior_day
from college_basketball_dfs.cbb_pipeline import run_cbb_cache_pipeline


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


st.set_page_config(page_title="CBB Admin Cache", layout="wide")
st.title("College Basketball Admin Cache")
st.caption("Cache-first data pipeline backed by Google Cloud Storage.")

default_bucket = os.getenv("CBB_GCS_BUCKET", "").strip() or (_secret("cbb_gcs_bucket") or "")
default_base_url = os.getenv("NCAA_API_BASE_URL", "https://ncaa-api.henrygd.me").strip()
default_project = os.getenv("GCP_PROJECT", "").strip() or (_secret("gcp_project") or "")

with st.sidebar:
    st.header("Pipeline Settings")
    selected_date = st.date_input("Slate Date", value=prior_day())
    bucket_name = st.text_input("GCS Bucket", value=default_bucket)
    base_url = st.text_input("NCAA API Base URL", value=default_base_url)
    gcp_project = st.text_input("GCP Project (optional)", value=default_project)
    force_refresh = st.checkbox("Force API refresh (ignore cached raw JSON)", value=False)
    run_clicked = st.button("Run Cache Pipeline")
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
                st.session_state["cbb_last_summary"] = summary
                st.success("Pipeline completed.")
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
