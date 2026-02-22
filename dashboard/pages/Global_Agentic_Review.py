from __future__ import annotations

import io
import json
import os
import re
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
    MARKET_CORRELATION_AI_REVIEW_SYSTEM_PROMPT,
    OPENAI_REVIEW_MODEL_FALLBACKS,
    build_daily_ai_review_packet,
    build_global_ai_review_packet,
    build_global_ai_review_user_prompt,
    build_market_correlation_ai_review_packet,
    build_market_correlation_ai_review_user_prompt,
    request_openai_review,
)
from college_basketball_dfs.cbb_backfill import iter_dates
from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_ncaa import prior_day
from college_basketball_dfs.cbb_tournament_review import (
    build_field_entries_and_players,
    build_player_exposure_comparison,
    build_projection_actual_comparison,
    extract_actual_ownership_from_standings,
    normalize_contest_standings_frame,
)

NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
SLATE_PRESET_OPTIONS = ["Main", "Afternoon", "Full Day", "Night", "Custom"]


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


def _norm_name_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _norm_name_key_loose(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in NAME_SUFFIX_TOKENS]
    return "".join(tokens)


def _normalize_player_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    text = text.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.0+)?", text):
        try:
            return str(int(float(text)))
        except (TypeError, ValueError):
            return text
    return re.sub(r"\s+", "", text)


def _normalize_slate_label(value: object, default: str = "Main") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text or default


def _slate_key_from_label(value: object, default: str = "main") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or default


def _contest_id_from_blob_name(blob_name: str, selected_date: date, selected_slate_key: str | None = None) -> str:
    filename = str(blob_name or "").split("/")[-1]
    # Legacy: YYYY-MM-DD_<contest>.csv
    prefix = f"{selected_date.isoformat()}_"
    if filename.startswith(prefix) and filename.endswith(".csv"):
        cid = filename[len(prefix) : -4]
        return cid or "contest"
    if selected_slate_key is None and filename.endswith(".csv"):
        stem = filename[:-4]
        if "_" in stem:
            _, rest = stem.split("_", 1)
            if rest:
                return rest
    # Slate-scoped: <slate_key>_<contest>.csv
    safe_slate = _slate_key_from_label(selected_slate_key or "main")
    slate_prefix = f"{safe_slate}_"
    if filename.startswith(slate_prefix) and filename.endswith(".csv"):
        cid = filename[len(slate_prefix) : -4]
        return cid or "contest"
    return "contest"


def _normalize_ownership_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"])
    out = df.copy()
    normalized_columns = {re.sub(r"[^a-z0-9]", "", str(c).strip().lower()): c for c in out.columns}
    rename_aliases = {
        "id": "ID",
        "playerid": "ID",
        "dkid": "ID",
        "draftkingsid": "ID",
        "nameid": "ID",
        "name": "Name",
        "playername": "Name",
        "player": "Name",
        "athlete": "Name",
        "team": "TeamAbbrev",
        "teamabbrev": "TeamAbbrev",
        "teamabbr": "TeamAbbrev",
        "teamabbreviation": "TeamAbbrev",
        "ownership": "actual_ownership",
        "own": "actual_ownership",
        "ownpct": "actual_ownership",
        "actualown": "actual_ownership",
        "actualownership": "actual_ownership",
        "drafted": "actual_ownership",
        "pctdrafted": "actual_ownership",
        "fieldownership": "actual_ownership",
    }
    resolved_rename: dict[str, str] = {}
    for alias, dest in rename_aliases.items():
        source = normalized_columns.get(alias)
        if source:
            resolved_rename[source] = dest
    if resolved_rename:
        out = out.rename(columns=resolved_rename)
    for col in ["ID", "Name", "TeamAbbrev", "actual_ownership"]:
        if col not in out.columns:
            out[col] = ""
    out["ID"] = out["ID"].map(_normalize_player_id)
    out["Name"] = out["Name"].astype(str).str.strip()
    out["TeamAbbrev"] = out["TeamAbbrev"].astype(str).str.strip().str.upper()
    out["actual_ownership"] = out["actual_ownership"].map(_to_ownership_float)
    out["name_key"] = out["Name"].map(_norm_name_key)
    out["name_key_loose"] = out["Name"].map(_norm_name_key_loose)
    out = out.loc[(out["ID"] != "") | (out["Name"] != "")]
    out = out.sort_values(["ID", "Name"], ascending=[True, True]).drop_duplicates(
        subset=["ID", "name_key", "TeamAbbrev"], keep="last"
    )
    return out[["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"]].reset_index(drop=True)


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
    selected_slate_key: str | None,
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
    try:
        return store.list_lineup_run_dates(selected_slate_key)
    except TypeError:
        return store.list_lineup_run_dates()


@st.cache_data(ttl=600, show_spinner=False)
def load_projection_snapshot_frame(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
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
    try:
        csv_text = store.read_projections_csv(selected_date, selected_slate_key)
    except TypeError:
        csv_text = store.read_projections_csv(selected_date)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


@st.cache_data(ttl=600, show_spinner=False)
def load_dk_slate_frame_for_date(
    bucket_name: str,
    selected_date: date,
    slate_key: str | None,
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
    try:
        csv_text = store.read_dk_slate_csv(selected_date, slate_key)
    except TypeError:
        csv_text = store.read_dk_slate_csv(selected_date)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


@st.cache_data(ttl=600, show_spinner=False)
def load_first_contest_standings_for_date(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> tuple[pd.DataFrame, str]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    legacy_prefix = f"{store.contest_standings_prefix}/{selected_date.isoformat()}_"
    blobs = list(store.bucket.list_blobs(prefix=legacy_prefix))
    if selected_slate_key is None:
        scoped_prefix = f"{store.contest_standings_prefix}/{selected_date.isoformat()}/"
        blobs.extend(list(store.bucket.list_blobs(prefix=scoped_prefix)))
    else:
        safe_slate = _slate_key_from_label(selected_slate_key or "main")
        scoped_prefix = f"{store.contest_standings_prefix}/{selected_date.isoformat()}/{safe_slate}_"
        blobs.extend(list(store.bucket.list_blobs(prefix=scoped_prefix)))
    if not blobs:
        return pd.DataFrame(), ""
    unique_blobs = {str(getattr(b, "name", "") or ""): b for b in blobs}
    blobs = sorted(unique_blobs.values(), key=lambda b: str(getattr(b, "name", "") or ""), reverse=True)
    for blob in blobs:
        try:
            csv_text = blob.download_as_text(encoding="utf-8")
        except Exception:
            continue
        if not csv_text or not csv_text.strip():
            continue
        try:
            frame = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            continue
        if not frame.empty:
            return frame, str(blob.name or "")
    return pd.DataFrame(), ""


@st.cache_data(ttl=600, show_spinner=False)
def load_ownership_frame_for_date(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
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
    try:
        csv_text = store.read_ownership_csv(selected_date, selected_slate_key)
    except TypeError:
        csv_text = store.read_ownership_csv(selected_date)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"])
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
    selected_slate_key: str | None,
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
    try:
        run_ids = store.list_lineup_run_ids(selected_date, selected_slate_key)
    except TypeError:
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
        proj["ID"] = proj["ID"].map(_normalize_player_id)
    if "Name" in proj.columns:
        proj["Name"] = proj["Name"].astype(str).str.strip()
        proj["name_key"] = proj["Name"].map(_norm_name_key)
        proj["name_key_loose"] = proj["Name"].map(_norm_name_key_loose)

    own = ownership_df.copy() if isinstance(ownership_df, pd.DataFrame) else pd.DataFrame()
    if "ID" in own.columns:
        own["ID"] = own["ID"].map(_normalize_player_id)
    if "Name" in own.columns:
        own["Name"] = own["Name"].astype(str).str.strip()
        own["name_key"] = own["Name"].map(_norm_name_key)
        own["name_key_loose"] = own["Name"].map(_norm_name_key_loose)

    expo = proj.copy()
    expo["actual_ownership"] = pd.NA
    if not own.empty:
        if "ID" in expo.columns and "ID" in own.columns:
            by_id = (
                own.loc[own["ID"] != "", ["ID", "actual_ownership"]]
                .dropna(subset=["actual_ownership"])
                .drop_duplicates("ID")
                .rename(columns={"actual_ownership": "actual_ownership_id"})
            )
            if not by_id.empty:
                expo = expo.merge(by_id, on="ID", how="left")
                expo["actual_ownership"] = pd.to_numeric(expo["actual_ownership_id"], errors="coerce")
        if "name_key" in expo.columns and "name_key" in own.columns:
            by_name = (
                own.loc[own["name_key"] != "", ["name_key", "actual_ownership"]]
                .dropna(subset=["actual_ownership"])
                .drop_duplicates("name_key")
                .rename(columns={"actual_ownership": "actual_ownership_name"})
            )
            if not by_name.empty:
                expo = expo.merge(by_name, on="name_key", how="left")
                expo["actual_ownership"] = pd.to_numeric(expo.get("actual_ownership"), errors="coerce").where(
                    pd.to_numeric(expo.get("actual_ownership"), errors="coerce").notna(),
                    pd.to_numeric(expo.get("actual_ownership_name"), errors="coerce"),
                )
        if "name_key_loose" in expo.columns and "name_key_loose" in own.columns:
            by_name_loose = (
                own.loc[own["name_key_loose"] != "", ["name_key_loose", "actual_ownership"]]
                .dropna(subset=["actual_ownership"])
                .drop_duplicates("name_key_loose")
                .rename(columns={"actual_ownership": "actual_ownership_name_loose"})
            )
            if not by_name_loose.empty:
                expo = expo.merge(by_name_loose, on="name_key_loose", how="left")
                expo["actual_ownership"] = pd.to_numeric(expo.get("actual_ownership"), errors="coerce").where(
                    pd.to_numeric(expo.get("actual_ownership"), errors="coerce").notna(),
                    pd.to_numeric(expo.get("actual_ownership_name_loose"), errors="coerce"),
                )
        expo = expo.drop(
            columns=[
                "actual_ownership_id",
                "actual_ownership_name",
                "actual_ownership_name_loose",
            ],
            errors="ignore",
        )

    expo["projected_ownership"] = pd.to_numeric(expo.get("projected_ownership"), errors="coerce")
    expo["actual_ownership_from_file"] = pd.to_numeric(expo.get("actual_ownership"), errors="coerce")
    expo["ownership_diff_vs_proj"] = expo["actual_ownership_from_file"] - expo["projected_ownership"]
    expo["field_ownership_pct"] = expo["actual_ownership_from_file"]
    return expo


def _build_market_review_rows(
    projection_df: pd.DataFrame,
    projection_compare_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    review_day: date,
) -> pd.DataFrame:
    if projection_df is None or projection_df.empty:
        return pd.DataFrame()
    out = projection_df.copy()
    if "ID" in out.columns:
        out["ID"] = out["ID"].map(_normalize_player_id)
    if "Name" in out.columns:
        out["Name"] = out["Name"].astype(str).str.strip()
    if "TeamAbbrev" in out.columns:
        out["TeamAbbrev"] = out["TeamAbbrev"].astype(str).str.strip().str.upper()

    cmp_df = projection_compare_df.copy() if isinstance(projection_compare_df, pd.DataFrame) else pd.DataFrame()
    cmp_cols = [c for c in ["actual_dk_points", "blend_error", "our_error", "vegas_error", "actual_minutes"] if c in cmp_df.columns]
    if not cmp_df.empty and cmp_cols:
        if "ID" in cmp_df.columns:
            cmp_df["ID"] = cmp_df["ID"].map(_normalize_player_id)
        if "Name" in cmp_df.columns:
            cmp_df["Name"] = cmp_df["Name"].astype(str).str.strip()
        if "ID" in out.columns and "ID" in cmp_df.columns:
            cmp_by_id = cmp_df.loc[cmp_df["ID"] != "", ["ID"] + cmp_cols].drop_duplicates("ID")
            out = out.merge(cmp_by_id, on="ID", how="left")
        elif "Name" in out.columns and "Name" in cmp_df.columns:
            cmp_by_name = cmp_df.loc[cmp_df["Name"] != "", ["Name"] + cmp_cols].drop_duplicates("Name")
            out = out.merge(cmp_by_name, on="Name", how="left")

    expo_df = exposure_df.copy() if isinstance(exposure_df, pd.DataFrame) else pd.DataFrame()
    if not expo_df.empty:
        own_cols = [c for c in ["projected_ownership", "actual_ownership_from_file", "ownership_diff_vs_proj"] if c in expo_df.columns]
        if own_cols:
            if "ID" in expo_df.columns:
                expo_df["ID"] = expo_df["ID"].map(_normalize_player_id)
            if "Name" in expo_df.columns:
                expo_df["Name"] = expo_df["Name"].astype(str).str.strip()
            if "ID" in out.columns and "ID" in expo_df.columns:
                own_by_id = expo_df.loc[expo_df["ID"] != "", ["ID"] + own_cols].drop_duplicates("ID")
                out = out.merge(own_by_id, on="ID", how="left", suffixes=("", "_own"))
            elif "Name" in out.columns and "Name" in expo_df.columns:
                own_by_name = expo_df.loc[expo_df["Name"] != "", ["Name"] + own_cols].drop_duplicates("Name")
                out = out.merge(own_by_name, on="Name", how="left", suffixes=("", "_own"))

    for col in [
        "blended_projection",
        "our_dk_projection",
        "vegas_dk_projection",
        "actual_dk_points",
        "blend_error",
        "game_total_line",
        "game_spread_line",
        "game_tail_score",
        "vegas_blend_weight",
        "projected_ownership",
        "actual_ownership_from_file",
        "ownership_diff_vs_proj",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    missing_blend = pd.to_numeric(out["blend_error"], errors="coerce").isna()
    out.loc[missing_blend, "blend_error"] = (
        pd.to_numeric(out.loc[missing_blend, "actual_dk_points"], errors="coerce")
        - pd.to_numeric(out.loc[missing_blend, "blended_projection"], errors="coerce")
    )
    out["ownership_error"] = pd.to_numeric(out["ownership_diff_vs_proj"], errors="coerce")
    missing_own = out["ownership_error"].isna()
    out.loc[missing_own, "ownership_error"] = (
        pd.to_numeric(out.loc[missing_own, "actual_ownership_from_file"], errors="coerce")
        - pd.to_numeric(out.loc[missing_own, "projected_ownership"], errors="coerce")
    )

    out["review_date"] = review_day.isoformat()
    keep_cols = [
        "review_date",
        "ID",
        "Name",
        "TeamAbbrev",
        "Position",
        "Salary",
        "blended_projection",
        "our_dk_projection",
        "vegas_dk_projection",
        "actual_dk_points",
        "blend_error",
        "projected_ownership",
        "actual_ownership_from_file",
        "ownership_error",
        "game_total_line",
        "game_spread_line",
        "game_tail_score",
        "vegas_blend_weight",
    ]
    return out[[c for c in keep_cols if c in out.columns]].copy()


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

st.subheader("Global Agentic Settings")
g1, g2 = st.columns(2)
bucket_name = g1.text_input("GCS Bucket", value=default_bucket, key="global_agentic_bucket")
gcp_project = g2.text_input("GCP Project (optional)", value=default_project, key="global_agentic_project")

g3, g4 = st.columns(2)
include_all_slates = bool(
    g3.checkbox(
        "Include All Slates",
        value=False,
        key="global_agentic_include_all_slates",
        help="When enabled, scans all slate labels instead of only the active slate label.",
    )
)
shared_slate_preset = g4.selectbox(
    "Active Slate Label",
    options=SLATE_PRESET_OPTIONS,
    index=0,
    key="shared_slate_preset",
    help="Shared slate context used when loading DK slates and lineup runs.",
    disabled=include_all_slates,
)
shared_slate_custom = ""
if (not include_all_slates) and shared_slate_preset == "Custom":
    shared_slate_custom = st.text_input(
        "Custom Active Slate Label",
        value="Main",
        key="shared_slate_custom_label",
    )
shared_slate_label = _normalize_slate_label(
    shared_slate_custom if shared_slate_preset == "Custom" else shared_slate_preset
)
shared_slate_key = _slate_key_from_label(shared_slate_label)
effective_slate_key = None if include_all_slates else shared_slate_key
if include_all_slates:
    st.caption("Active slate: `All slates`")
else:
    st.caption(f"Active slate: `{shared_slate_label}` (key: `{shared_slate_key}`)")

g5, g6, g7 = st.columns(3)
end_date = g5.date_input("Review End Date", value=default_end_date, key="global_agentic_end_date")
lookback_days = int(
    g6.slider("Lookback Days", min_value=7, max_value=180, value=30, step=1, key="global_agentic_lookback_days")
)
focus_limit = int(
    g7.slider(
        "Global Focus Players",
        min_value=5,
        max_value=60,
        value=25,
        step=1,
        key="global_agentic_focus_limit",
    )
)

g8, g9, g10 = st.columns(3)
market_focus_limit = int(
    g8.slider(
        "Market Focus Rows",
        min_value=5,
        max_value=80,
        value=25,
        step=1,
        key="global_agentic_market_focus_limit",
    )
)
market_min_bucket_samples = int(
    g9.slider(
        "Market Min Bucket Samples",
        min_value=5,
        max_value=120,
        value=20,
        step=1,
        key="global_agentic_market_min_bucket_samples",
    )
)
use_saved_run_dates = bool(
    g10.checkbox(
        "Use Saved Run Dates Only",
        value=True,
        key="global_agentic_use_saved_dates",
        help="If enabled, only dates with saved lineup runs are scanned.",
    )
)

g11, g12 = st.columns(2)
selected_model = g11.selectbox(
    "OpenAI Model",
    options=model_options,
    index=0,
    key="global_agentic_model",
    help="Choose model for final recommendation generation.",
)
max_output_tokens = int(
    g12.number_input(
        "Max Output Tokens",
        min_value=200,
        max_value=8000,
        value=1800,
        step=100,
        key="global_agentic_max_output_tokens",
    )
)

g13, g14 = st.columns(2)
build_packet_clicked = g13.button("Build Global Packet", key="global_agentic_build_packet")
run_openai_clicked = g14.button("Run OpenAI Review", key="global_agentic_run_openai")

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
                selected_slate_key=effective_slate_key,
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
        tournament_dates_used = 0
        market_frames: list[pd.DataFrame] = []
        for review_day in candidate_dates:
            scanned_dates += 1
            proj_df = load_projection_snapshot_frame(
                bucket_name=bucket_name.strip(),
                selected_date=review_day,
                selected_slate_key=effective_slate_key,
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

            contest_id = "global-review"
            entries_df = pd.DataFrame()
            exposure_df = pd.DataFrame()
            standings_df, standings_blob = load_first_contest_standings_for_date(
                bucket_name=bucket_name.strip(),
                selected_date=review_day,
                selected_slate_key=effective_slate_key,
                gcp_project=gcp_project.strip() or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if not standings_df.empty:
                normalized = normalize_contest_standings_frame(standings_df)
                slate_df = load_dk_slate_frame_for_date(
                    bucket_name=bucket_name.strip(),
                    selected_date=review_day,
                    slate_key=effective_slate_key,
                    gcp_project=gcp_project.strip() or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                if not normalized.empty and not slate_df.empty:
                    entries_df, expanded_df = build_field_entries_and_players(normalized, slate_df)
                    if not entries_df.empty:
                        tournament_dates_used += 1
                        contest_id = _contest_id_from_blob_name(
                            standings_blob,
                            review_day,
                            selected_slate_key=effective_slate_key,
                        )
                        actual_own_df = extract_actual_ownership_from_standings(normalized)
                        exposure_df = build_player_exposure_comparison(
                            expanded_players_df=expanded_df,
                            entry_count=int(len(entries_df)),
                            projection_df=proj_df,
                            actual_ownership_df=actual_own_df,
                            actual_results_df=actual_df,
                        )
            if exposure_df.empty:
                own_df = load_ownership_frame_for_date(
                    bucket_name=bucket_name.strip(),
                    selected_date=review_day,
                    selected_slate_key=effective_slate_key,
                    gcp_project=gcp_project.strip() or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                exposure_df = _build_exposure_frame(proj_df, own_df)
            market_day_df = _build_market_review_rows(
                projection_df=proj_df,
                projection_compare_df=proj_compare_df,
                exposure_df=exposure_df,
                review_day=review_day,
            )
            if not market_day_df.empty:
                market_frames.append(market_day_df)
            phantom_summary_df = load_first_phantom_summary_for_date(
                bucket_name=bucket_name.strip(),
                selected_date=review_day,
                selected_slate_key=effective_slate_key,
                gcp_project=gcp_project.strip() or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            day_packet = build_daily_ai_review_packet(
                review_date=review_day.isoformat(),
                contest_id=contest_id,
                projection_comparison_df=proj_compare_df,
                entries_df=entries_df,
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
        market_rows_df = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame()
        market_packet = build_market_correlation_ai_review_packet(
            review_rows_df=market_rows_df,
            focus_limit=market_focus_limit,
            min_bucket_samples=market_min_bucket_samples,
        )
        market_user_prompt = build_market_correlation_ai_review_user_prompt(market_packet)
        st.session_state["global_agentic_packet"] = global_packet
        st.session_state["global_agentic_prompt"] = global_user_prompt
        st.session_state["global_agentic_market_packet"] = market_packet
        st.session_state["global_agentic_market_prompt"] = market_user_prompt
        st.session_state.pop("global_agentic_output", None)
        st.session_state.pop("global_agentic_market_output", None)
        st.session_state["global_agentic_meta"] = {
            "scanned_dates": scanned_dates,
            "used_dates": used_dates,
            "window_start": cutoff.isoformat(),
            "window_end": end_date.isoformat(),
            "lookback_days": lookback_days,
            "use_saved_run_dates": use_saved_run_dates,
            "include_all_slates": include_all_slates,
            "tournament_dates_used": tournament_dates_used,
            "market_rows": int(len(market_rows_df)),
        }
        st.success(f"Built global packet using {used_dates} of {scanned_dates} scanned dates.")

global_packet_state = st.session_state.get("global_agentic_packet")
global_user_prompt = str(st.session_state.get("global_agentic_prompt") or "").strip()
global_meta = st.session_state.get("global_agentic_meta") or {}
market_packet_state = st.session_state.get("global_agentic_market_packet")
market_user_prompt = str(st.session_state.get("global_agentic_market_prompt") or "").strip()

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
        f"tournament_dates_used={_safe_int(global_meta.get('tournament_dates_used'), default=0)}, "
        f"market_rows={_safe_int(global_meta.get('market_rows'), default=0)}, "
        f"include_all_slates={bool(global_meta.get('include_all_slates', False))}, "
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
    st.info("Click `Build Global Packet` above to start.")

st.markdown("---")
st.subheader("Market Correlation Agent (Multi-Date)")
st.caption(
    "Correlates odds/market features with ownership and actual points across the selected date window, "
    "then proposes bucket-level calibration adjustments."
)
if isinstance(market_packet_state, dict) and market_packet_state:
    market_window = market_packet_state.get("window_summary") or {}
    market_quality = market_packet_state.get("global_quality") or {}

    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("Dates Used", _safe_int(market_window.get("dates_used"), default=0))
    mm2.metric("Rows", _safe_int(market_window.get("rows"), default=0))
    mm3.metric("Blend MAE", f"{_safe_float(market_quality.get('blend_mae')):.2f}")
    mm4.metric(
        "Total->Points Spearman",
        f"{_safe_float(market_quality.get('total_line_vs_actual_points_spearman')):.3f}",
    )

    corr_df = pd.DataFrame(market_packet_state.get("correlation_table") or [])
    if not corr_df.empty:
        st.caption("Correlation Table")
        st.dataframe(corr_df, hide_index=True, use_container_width=True)

    bucket_tables = market_packet_state.get("bucket_calibration") or {}
    total_bucket_df = pd.DataFrame(bucket_tables.get("total_line_buckets") or [])
    spread_bucket_df = pd.DataFrame(bucket_tables.get("abs_spread_buckets") or [])
    b1, b2 = st.columns(2)
    with b1:
        st.caption("Total-Line Buckets")
        if total_bucket_df.empty:
            st.info("No total-line bucket summary available.")
        else:
            st.dataframe(total_bucket_df, hide_index=True, use_container_width=True)
    with b2:
        st.caption("Abs-Spread Buckets")
        if spread_bucket_df.empty:
            st.info("No spread bucket summary available.")
        else:
            st.dataframe(spread_bucket_df, hide_index=True, use_container_width=True)

    rec_df = pd.DataFrame(market_packet_state.get("calibration_recommendations") or [])
    if not rec_df.empty:
        st.caption("Calibration Recommendations")
        st.dataframe(rec_df, hide_index=True, use_container_width=True)

    market_packet_json = json.dumps(market_packet_state, indent=2, ensure_ascii=True)
    st.download_button(
        "Download Market Packet JSON",
        data=market_packet_json,
        file_name=f"market_correlation_packet_{end_date.isoformat()}.json",
        mime="application/json",
        key="download_global_agentic_market_packet_json",
    )
    st.download_button(
        "Download Market User Prompt TXT",
        data=market_user_prompt,
        file_name=f"market_correlation_prompt_{end_date.isoformat()}.txt",
        mime="text/plain",
        key="download_global_agentic_market_prompt_txt",
    )
    with st.expander("Market Packet Preview"):
        st.json(market_packet_state)
    with st.expander("Market Prompt Preview"):
        st.text_area(
            "Market Prompt",
            value=market_user_prompt,
            height=260,
            key="global_agentic_market_prompt_preview",
        )
else:
    st.info("Build the global packet to generate market-correlation analysis.")

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
                exc_text = str(exc or "")
                if "max_output_tokens" in exc_text.lower():
                    retry_tokens = min(8000, max(max_output_tokens + 800, int(max_output_tokens * 1.8)))
                    try:
                        st.caption(
                            "Global review hit max-output truncation; retrying with "
                            f"`max_output_tokens={retry_tokens}`."
                        )
                        output = request_openai_review(
                            api_key=openai_key,
                            user_prompt=global_user_prompt,
                            system_prompt=AI_REVIEW_SYSTEM_PROMPT,
                            model=selected_model,
                            max_output_tokens=retry_tokens,
                        )
                        st.session_state["global_agentic_output"] = output
                    except Exception as retry_exc:
                        st.exception(retry_exc)
                else:
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

run_market_openai_clicked = st.button("Run Market OpenAI Review", key="global_agentic_run_market_openai")
if run_market_openai_clicked:
    if not openai_key:
        st.error("Set `OPENAI_API_KEY` or Streamlit secret `openai_api_key` first.")
    elif not market_user_prompt:
        st.error("Build a global packet first to generate the market-correlation prompt.")
    else:
        with st.spinner("Generating market-correlation recommendations..."):
            try:
                market_output = request_openai_review(
                    api_key=openai_key,
                    user_prompt=market_user_prompt,
                    system_prompt=MARKET_CORRELATION_AI_REVIEW_SYSTEM_PROMPT,
                    model=selected_model,
                    max_output_tokens=max_output_tokens,
                )
                st.session_state["global_agentic_market_output"] = market_output
            except Exception as exc:
                exc_text = str(exc or "")
                if "max_output_tokens" in exc_text.lower():
                    retry_tokens = min(8000, max(max_output_tokens + 800, int(max_output_tokens * 1.8)))
                    try:
                        st.caption(
                            "Market review hit max-output truncation; retrying with "
                            f"`max_output_tokens={retry_tokens}`."
                        )
                        market_output = request_openai_review(
                            api_key=openai_key,
                            user_prompt=market_user_prompt,
                            system_prompt=MARKET_CORRELATION_AI_REVIEW_SYSTEM_PROMPT,
                            model=selected_model,
                            max_output_tokens=retry_tokens,
                        )
                        st.session_state["global_agentic_market_output"] = market_output
                    except Exception as retry_exc:
                        st.exception(retry_exc)
                else:
                    st.exception(exc)

market_output_text = str(st.session_state.get("global_agentic_market_output") or "").strip()
if market_output_text:
    st.subheader("Market Correlation Recommendations")
    st.text_area(
        "Market Model Output",
        value=market_output_text,
        height=360,
        key="global_agentic_market_output_preview",
    )
    st.download_button(
        "Download Market Recommendations TXT",
        data=market_output_text,
        file_name=f"market_correlation_recommendations_{end_date.isoformat()}.txt",
        mime="text/plain",
        key="download_global_agentic_market_recommendations_txt",
    )
