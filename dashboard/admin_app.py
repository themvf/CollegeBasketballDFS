from __future__ import annotations

import io
import inspect
import json
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_backfill import iter_dates, run_season_backfill, season_start_for_date
from college_basketball_dfs.cbb_ncaa import prior_day
from college_basketball_dfs.cbb_odds_backfill import run_odds_season_backfill
from college_basketball_dfs.cbb_odds import flatten_odds_payload
from college_basketball_dfs.cbb_odds_pipeline import run_cbb_odds_pipeline
from college_basketball_dfs.cbb_pipeline import run_cbb_cache_pipeline
from college_basketball_dfs.cbb_props_pipeline import run_cbb_props_pipeline
from college_basketball_dfs.cbb_team_lookup import rows_from_payloads
from college_basketball_dfs.cbb_dk_optimizer import (
    build_dk_upload_csv,
    build_player_pool,
    generate_lineups,
    lineups_slots_frame,
    lineups_summary_frame,
    normalize_injuries_frame,
    projection_salary_bucket_key,
    remove_injured_players,
)
from college_basketball_dfs.cbb_tail_model import fit_total_tail_model, score_odds_games_for_tail
from college_basketball_dfs.cbb_tournament_review import (
    build_entry_actual_points_comparison,
    build_field_entries_and_players,
    build_player_exposure_comparison,
    build_projection_actual_comparison,
    build_projection_adjustment_factors,
    compare_phantom_entries_to_field,
    build_user_strategy_summary,
    score_generated_lineups_against_actuals,
    summarize_phantom_entries,
    summarize_generated_lineups,
    normalize_contest_standings_frame,
    extract_actual_ownership_from_standings,
)
from college_basketball_dfs.cbb_ai_review import (
    AI_REVIEW_SYSTEM_PROMPT,
    GAME_SLATE_AI_REVIEW_SYSTEM_PROMPT,
    build_ai_review_user_prompt,
    build_daily_ai_review_packet,
    build_game_slate_ai_review_packet,
    build_game_slate_ai_review_user_prompt,
    build_global_ai_review_packet,
    build_global_ai_review_user_prompt,
    request_openai_review,
)
from college_basketball_dfs.cbb_vegas_review import build_vegas_review_games_frame


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


ROLE_FILTER_OPTIONS = ["All", "Guard (G)", "Forward (F)"]
COMMON_ODDS_BOOKMAKER_KEYS = [
    "fanduel",
    "draftkings",
    "betmgm",
    "betrivers",
    "caesars",
    "espnbet",
    "hardrockbet",
    "pointsbetus",
    "lowvig",
    "betonlineag",
    "bovada",
]
NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
PROJECTION_SALARY_BUCKET_ORDER = ("lt4500", "4500_6999", "7000_9999", "gte10000")
PROJECTION_SALARY_BUCKET_LABELS = {
    "lt4500": "< 4500",
    "4500_6999": "4500-6999",
    "7000_9999": "7000-9999",
    "gte10000": "10000+",
}


def _csv_values(text: str | None) -> list[str]:
    if not text:
        return []
    parts = [str(x).strip().lower() for x in str(text).split(",")]
    return [x for x in parts if x]


def _safe_float_value(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        text = str(value).strip().replace("%", "")
        if text == "":
            return float(default)
        return float(text)
    except (TypeError, ValueError):
        return float(default)


def _safe_int_value(value: object, default: int = 0) -> int:
    return int(_safe_float_value(value, float(default)))


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


def _merged_bookmakers(selected: list[str] | None, custom_csv: str | None = None) -> list[str]:
    out: list[str] = []
    for item in (selected or []):
        key = str(item).strip().lower()
        if key and key not in out:
            out.append(key)
    for item in _csv_values(custom_csv):
        if item not in out:
            out.append(item)
    return out


def _filter_frame_by_role(
    df: pd.DataFrame,
    selected_role: str,
    position_col: str = "Position",
) -> pd.DataFrame:
    if df.empty or position_col not in df.columns:
        return df
    pos = df[position_col].astype(str).str.strip().str.upper()
    if selected_role == "Guard (G)":
        return df.loc[pos.str.startswith("G")].copy()
    if selected_role == "Forward (F)":
        return df.loc[pos.str.startswith("F")].copy()
    return df


def _dk_slate_blob_name(slate_date: date) -> str:
    return f"cbb/dk_slates/{slate_date.isoformat()}_dk_slate.csv"


def _read_dk_slate_csv(store: CbbGcsStore, slate_date: date) -> str | None:
    reader = getattr(store, "read_dk_slate_csv", None)
    if callable(reader):
        return reader(slate_date)
    blob = store.bucket.blob(_dk_slate_blob_name(slate_date))
    if not blob.exists():
        return None
    return blob.download_as_text(encoding="utf-8")


def _write_dk_slate_csv(store: CbbGcsStore, slate_date: date, csv_text: str) -> str:
    writer = getattr(store, "write_dk_slate_csv", None)
    if callable(writer):
        return writer(slate_date, csv_text)
    blob_name = _dk_slate_blob_name(slate_date)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _delete_dk_slate_csv(store: CbbGcsStore, slate_date: date) -> tuple[bool, str]:
    blob_name = _dk_slate_blob_name(slate_date)
    deleter = getattr(store, "delete_dk_slate_csv", None)
    if callable(deleter):
        return bool(deleter(slate_date)), blob_name
    blob = store.bucket.blob(blob_name)
    if not blob.exists():
        return False, blob_name
    blob.delete()
    return True, blob_name


def _injuries_blob_name() -> str:
    return "cbb/injuries/injuries_master.csv"


def _injuries_feed_blob_name(selected_date: date | None = None) -> str:
    if selected_date is None:
        return "cbb/injuries/injuries_feed.csv"
    return f"cbb/injuries/feed/{selected_date.isoformat()}_injuries_feed.csv"


def _injuries_manual_blob_name() -> str:
    return "cbb/injuries/injuries_manual.csv"


def _read_injuries_csv(store: CbbGcsStore) -> str | None:
    reader = getattr(store, "read_injuries_csv", None)
    if callable(reader):
        return reader()
    blob = store.bucket.blob(_injuries_blob_name())
    if not blob.exists():
        return None
    return blob.download_as_text(encoding="utf-8")


def _write_injuries_csv(store: CbbGcsStore, csv_text: str) -> str:
    writer = getattr(store, "write_injuries_csv", None)
    if callable(writer):
        return writer(csv_text)
    blob_name = _injuries_blob_name()
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _read_injuries_feed_csv(store: CbbGcsStore, selected_date: date | None = None) -> str | None:
    reader = getattr(store, "read_injuries_feed_csv", None)
    if callable(reader):
        try:
            params = inspect.signature(reader).parameters
            if "game_date" in params:
                return reader(game_date=selected_date)
            if selected_date is not None and len(params) >= 1:
                return reader(selected_date)
            return reader()
        except (TypeError, ValueError):
            return reader()
    blob = store.bucket.blob(_injuries_feed_blob_name(selected_date))
    if not blob.exists():
        if selected_date is not None:
            legacy_blob = store.bucket.blob(_injuries_feed_blob_name(None))
            if legacy_blob.exists():
                return legacy_blob.download_as_text(encoding="utf-8")
        return None
    return blob.download_as_text(encoding="utf-8")


def _write_injuries_feed_csv(store: CbbGcsStore, csv_text: str, selected_date: date | None = None) -> str:
    writer = getattr(store, "write_injuries_feed_csv", None)
    if callable(writer):
        try:
            params = inspect.signature(writer).parameters
            if "game_date" in params:
                return writer(csv_text=csv_text, game_date=selected_date)
            if selected_date is not None and len(params) >= 2:
                return writer(selected_date, csv_text)
            return writer(csv_text)
        except (TypeError, ValueError):
            return writer(csv_text)
    blob_name = _injuries_feed_blob_name(selected_date)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _read_injuries_manual_csv(store: CbbGcsStore) -> str | None:
    reader = getattr(store, "read_injuries_manual_csv", None)
    if callable(reader):
        return reader()
    blob = store.bucket.blob(_injuries_manual_blob_name())
    if not blob.exists():
        return None
    return blob.download_as_text(encoding="utf-8")


def _write_injuries_manual_csv(store: CbbGcsStore, csv_text: str) -> str:
    writer = getattr(store, "write_injuries_manual_csv", None)
    if callable(writer):
        return writer(csv_text)
    blob_name = _injuries_manual_blob_name()
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _delete_injuries_feed_csv(store: CbbGcsStore, selected_date: date | None = None) -> tuple[bool, str]:
    blob_name = _injuries_feed_blob_name(selected_date)
    deleter = getattr(store, "delete_injuries_feed_csv", None)
    if callable(deleter):
        try:
            params = inspect.signature(deleter).parameters
            if "game_date" in params:
                return bool(deleter(game_date=selected_date)), blob_name
            if selected_date is not None and len(params) >= 1:
                return bool(deleter(selected_date)), blob_name
            return bool(deleter()), blob_name
        except (TypeError, ValueError):
            return bool(deleter()), blob_name
    blob = store.bucket.blob(blob_name)
    if not blob.exists():
        return False, blob_name
    blob.delete()
    return True, blob_name


def _projections_blob_name(slate_date: date) -> str:
    return f"cbb/projections/{slate_date.isoformat()}_projections.csv"


def _write_projections_csv(store: CbbGcsStore, slate_date: date, csv_text: str) -> str:
    writer = getattr(store, "write_projections_csv", None)
    if callable(writer):
        return writer(slate_date, csv_text)
    blob_name = _projections_blob_name(slate_date)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _read_projections_csv(store: CbbGcsStore, slate_date: date) -> str | None:
    reader = getattr(store, "read_projections_csv", None)
    if callable(reader):
        return reader(slate_date)
    blob = store.bucket.blob(_projections_blob_name(slate_date))
    if not blob.exists():
        return None
    return blob.download_as_text(encoding="utf-8")


def _ownership_blob_name(slate_date: date) -> str:
    return f"cbb/ownership/{slate_date.isoformat()}_ownership.csv"


def _read_ownership_csv(store: CbbGcsStore, slate_date: date) -> str | None:
    reader = getattr(store, "read_ownership_csv", None)
    if callable(reader):
        return reader(slate_date)
    blob = store.bucket.blob(_ownership_blob_name(slate_date))
    if not blob.exists():
        return None
    return blob.download_as_text(encoding="utf-8")


def _write_ownership_csv(store: CbbGcsStore, slate_date: date, csv_text: str) -> str:
    writer = getattr(store, "write_ownership_csv", None)
    if callable(writer):
        return writer(slate_date, csv_text)
    blob_name = _ownership_blob_name(slate_date)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _contest_standings_blob_name(slate_date: date, contest_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(contest_id or "").strip())
    safe = safe or "contest"
    return f"cbb/contest_standings/{slate_date.isoformat()}_{safe}.csv"


def _read_contest_standings_csv(store: CbbGcsStore, slate_date: date, contest_id: str) -> str | None:
    reader = getattr(store, "read_contest_standings_csv", None)
    if callable(reader):
        return reader(slate_date, contest_id)
    blob = store.bucket.blob(_contest_standings_blob_name(slate_date, contest_id))
    if not blob.exists():
        return None
    return blob.download_as_text(encoding="utf-8")


def _write_contest_standings_csv(store: CbbGcsStore, slate_date: date, contest_id: str, csv_text: str) -> str:
    writer = getattr(store, "write_contest_standings_csv", None)
    if callable(writer):
        return writer(slate_date, contest_id, csv_text)
    blob_name = _contest_standings_blob_name(slate_date, contest_id)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _new_lineup_run_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"run_{stamp}_{uuid4().hex[:8]}"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(x) for x in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return str(value)


def persist_lineup_run_bundle(
    store: CbbGcsStore,
    slate_date: date,
    bundle: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(bundle.get("run_id") or _new_lineup_run_id())
    generated_at_utc = str(bundle.get("generated_at_utc") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    settings = _json_safe(bundle.get("settings") or {})
    versions = bundle.get("versions") or {}
    version_entries: list[dict[str, Any]] = []

    for version_key, version_data in versions.items():
        version_name = str(version_key)
        lineups = version_data.get("lineups") or []
        warnings = [str(x) for x in (version_data.get("warnings") or [])]
        upload_csv = str(version_data.get("upload_csv") or "")
        slots_df = lineups_slots_frame(lineups)
        if slots_df.empty:
            slots_df = lineups_summary_frame(lineups)
        lineups_csv = slots_df.to_csv(index=False)

        payload = {
            "run_id": run_id,
            "slate_date": slate_date.isoformat(),
            "generated_at_utc": generated_at_utc,
            "run_mode": str(bundle.get("run_mode") or "single"),
            "version_key": version_name,
            "version_label": str(version_data.get("version_label") or version_name),
            "lineup_strategy": str(version_data.get("lineup_strategy") or ""),
            "model_profile": str(version_data.get("model_profile") or ""),
            "include_tail_signals": bool(version_data.get("include_tail_signals", False)),
            "warnings": warnings,
            "settings": settings,
            "lineups": _json_safe(lineups),
            "dk_upload_csv": upload_csv,
        }
        json_blob = store.write_lineup_version_json(slate_date, run_id, version_name, payload)
        csv_blob = store.write_lineup_version_csv(slate_date, run_id, version_name, lineups_csv)
        upload_blob = store.write_lineup_version_upload_csv(slate_date, run_id, version_name, upload_csv)
        version_entries.append(
            {
                "version_key": version_name,
                "version_label": str(version_data.get("version_label") or version_name),
                "lineup_strategy": str(version_data.get("lineup_strategy") or ""),
                "model_profile": str(version_data.get("model_profile") or ""),
                "include_tail_signals": bool(version_data.get("include_tail_signals", False)),
                "lineup_count_generated": int(len(lineups)),
                "warning_count": int(len(warnings)),
                "json_blob": json_blob,
                "csv_blob": csv_blob,
                "dk_upload_blob": upload_blob,
            }
        )

    manifest = {
        "run_id": run_id,
        "slate_date": slate_date.isoformat(),
        "generated_at_utc": generated_at_utc,
        "run_mode": str(bundle.get("run_mode") or "single"),
        "settings": settings,
        "versions": version_entries,
    }
    manifest_blob = store.write_lineup_run_manifest_json(slate_date, run_id, manifest)
    return {
        "run_id": run_id,
        "manifest_blob": manifest_blob,
        "version_count": len(version_entries),
        "manifest": manifest,
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_saved_lineup_run_manifests(
    bucket_name: str,
    selected_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> list[dict[str, Any]]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    run_ids = store.list_lineup_run_ids(selected_date)
    manifests: list[dict[str, Any]] = []
    for run_id in run_ids:
        payload = store.read_lineup_run_manifest_json(selected_date, run_id)
        if isinstance(payload, dict):
            manifests.append(payload)
    manifests.sort(key=lambda x: str(x.get("generated_at_utc") or ""), reverse=True)
    return manifests


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
def compute_projection_calibration_from_phantom(
    bucket_name: str,
    selected_date: date,
    lookback_days: int,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> dict[str, Any]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    end_date = selected_date - timedelta(days=1)
    if lookback_days <= 0 or end_date < date(2000, 1, 1):
        return {
            "scale": 1.0,
            "raw_scale": 1.0,
            "used_dates": 0,
            "lineups": 0,
            "weighted_avg_actual": 0.0,
            "weighted_avg_projected": 0.0,
            "weighted_avg_delta": 0.0,
        }
    start_date = end_date - timedelta(days=max(1, int(lookback_days)) - 1)
    run_dates = store.list_lineup_run_dates()
    candidate_dates = [d for d in run_dates if isinstance(d, date) and start_date <= d <= end_date]
    if not candidate_dates:
        return {
            "scale": 1.0,
            "raw_scale": 1.0,
            "used_dates": 0,
            "lineups": 0,
            "weighted_avg_actual": 0.0,
            "weighted_avg_projected": 0.0,
            "weighted_avg_delta": 0.0,
        }

    total_actual = 0.0
    total_projected = 0.0
    total_lineups = 0.0
    used_dates = 0
    for run_date in sorted(candidate_dates, reverse=True):
        run_ids = store.list_lineup_run_ids(run_date)
        summary_df = pd.DataFrame()
        for run_id in run_ids:
            payload = store.read_phantom_review_summary_json(run_date, run_id)
            if not isinstance(payload, dict):
                continue
            rows = payload.get("summary_rows")
            if not isinstance(rows, list) or not rows:
                continue
            summary_df = pd.DataFrame(rows)
            if not summary_df.empty:
                break
        if summary_df.empty:
            continue

        lineups = pd.to_numeric(summary_df.get("lineups"), errors="coerce").fillna(0.0)
        avg_actual = pd.to_numeric(summary_df.get("avg_actual_points"), errors="coerce").fillna(0.0)
        avg_proj = pd.to_numeric(summary_df.get("avg_projected_points"), errors="coerce").fillna(0.0)
        use_mask = (lineups > 0.0) & (avg_proj > 0.0)
        if not bool(use_mask.any()):
            continue

        weighted_actual = float((lineups[use_mask] * avg_actual[use_mask]).sum())
        weighted_proj = float((lineups[use_mask] * avg_proj[use_mask]).sum())
        weighted_lineups = float(lineups[use_mask].sum())
        if weighted_proj <= 0.0 or weighted_lineups <= 0.0:
            continue
        total_actual += weighted_actual
        total_projected += weighted_proj
        total_lineups += weighted_lineups
        used_dates += 1

    if total_projected <= 0.0 or total_lineups <= 0.0:
        return {
            "scale": 1.0,
            "raw_scale": 1.0,
            "used_dates": used_dates,
            "lineups": int(total_lineups),
            "weighted_avg_actual": 0.0,
            "weighted_avg_projected": 0.0,
            "weighted_avg_delta": 0.0,
        }

    raw_scale = float(total_actual / total_projected)
    clipped_scale = float(min(1.05, max(0.75, raw_scale)))
    avg_actual = float(total_actual / total_lineups)
    avg_projected = float(total_projected / total_lineups)
    avg_delta = float(avg_actual - avg_projected)
    return {
        "scale": clipped_scale,
        "raw_scale": raw_scale,
        "used_dates": int(used_dates),
        "lineups": int(total_lineups),
        "weighted_avg_actual": avg_actual,
        "weighted_avg_projected": avg_projected,
        "weighted_avg_delta": avg_delta,
    }


def _default_salary_bucket_calibration(
    lookback_days: int,
    min_samples_per_bucket: int,
) -> dict[str, Any]:
    bucket_rows = [
        {
            "salary_bucket": key,
            "salary_bucket_label": PROJECTION_SALARY_BUCKET_LABELS.get(key, key),
            "samples": 0,
            "avg_actual": 0.0,
            "avg_projection": 0.0,
            "avg_error": 0.0,
            "mae": 0.0,
            "raw_scale": 1.0,
            "scale": 1.0,
            "used_for_adjustment": False,
        }
        for key in PROJECTION_SALARY_BUCKET_ORDER
    ]
    return {
        "lookback_days": int(max(0, lookback_days)),
        "min_samples_per_bucket": int(max(1, min_samples_per_bucket)),
        "used_dates": 0,
        "player_rows": 0,
        "scales": {key: 1.0 for key in PROJECTION_SALARY_BUCKET_ORDER},
        "bucket_rows": bucket_rows,
    }


@st.cache_data(ttl=300, show_spinner=False)
def compute_projection_salary_bucket_calibration(
    bucket_name: str,
    selected_date: date,
    lookback_days: int,
    min_samples_per_bucket: int,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> dict[str, Any]:
    lookback = int(max(0, lookback_days))
    min_samples = int(max(1, min_samples_per_bucket))
    default_result = _default_salary_bucket_calibration(lookback, min_samples)

    end_date = selected_date - timedelta(days=1)
    if lookback <= 0 or end_date < date(2000, 1, 1):
        return default_result

    start_date = end_date - timedelta(days=max(1, lookback) - 1)
    all_rows: list[pd.DataFrame] = []
    used_dates = 0
    cursor = end_date
    while cursor >= start_date:
        slate_df = load_dk_slate_frame_for_date(
            bucket_name=bucket_name,
            selected_date=cursor,
            gcp_project=gcp_project,
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
        )
        if slate_df.empty:
            cursor -= timedelta(days=1)
            continue
        actual_df = load_actual_results_frame_for_date(
            bucket_name=bucket_name,
            selected_date=cursor,
            gcp_project=gcp_project,
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
        )
        if actual_df.empty:
            cursor -= timedelta(days=1)
            continue

        comp_df = build_projection_actual_comparison(
            projection_df=slate_df,
            actual_results_df=actual_df,
        )
        if comp_df.empty:
            cursor -= timedelta(days=1)
            continue

        comp = comp_df.copy()
        comp["Salary"] = pd.to_numeric(comp.get("Salary"), errors="coerce")
        comp["blended_projection"] = pd.to_numeric(comp.get("blended_projection"), errors="coerce")
        comp["actual_dk_points"] = pd.to_numeric(comp.get("actual_dk_points"), errors="coerce")
        comp["blend_error"] = pd.to_numeric(comp.get("blend_error"), errors="coerce")
        comp = comp.loc[
            comp["Salary"].notna()
            & comp["blended_projection"].notna()
            & comp["actual_dk_points"].notna()
            & (comp["blended_projection"] > 0.0)
        ].copy()
        if comp.empty:
            cursor -= timedelta(days=1)
            continue

        comp["salary_bucket"] = comp["Salary"].map(projection_salary_bucket_key)
        all_rows.append(comp[["salary_bucket", "actual_dk_points", "blended_projection", "blend_error"]])
        used_dates += 1
        cursor -= timedelta(days=1)

    if not all_rows:
        return default_result

    full = pd.concat(all_rows, ignore_index=True)
    if full.empty:
        return default_result

    output_rows: list[dict[str, Any]] = []
    scales: dict[str, float] = {}
    for key in PROJECTION_SALARY_BUCKET_ORDER:
        seg = full.loc[full["salary_bucket"] == key].copy()
        samples = int(len(seg))
        avg_actual = float(pd.to_numeric(seg.get("actual_dk_points"), errors="coerce").mean()) if samples else 0.0
        avg_projection = float(pd.to_numeric(seg.get("blended_projection"), errors="coerce").mean()) if samples else 0.0
        avg_error = float(pd.to_numeric(seg.get("blend_error"), errors="coerce").mean()) if samples else 0.0
        mae = float(pd.to_numeric(seg.get("blend_error"), errors="coerce").abs().mean()) if samples else 0.0
        raw_scale = 1.0
        if avg_projection > 0.0 and samples > 0:
            raw_scale = float(avg_actual / avg_projection)
        clipped_scale = float(min(1.15, max(0.70, raw_scale)))
        use_scale = clipped_scale if samples >= min_samples else 1.0
        scales[key] = float(use_scale)
        output_rows.append(
            {
                "salary_bucket": key,
                "salary_bucket_label": PROJECTION_SALARY_BUCKET_LABELS.get(key, key),
                "samples": samples,
                "avg_actual": avg_actual,
                "avg_projection": avg_projection,
                "avg_error": avg_error,
                "mae": mae,
                "raw_scale": raw_scale,
                "scale": float(use_scale),
                "used_for_adjustment": bool(samples >= min_samples),
            }
        )

    return {
        "lookback_days": lookback,
        "min_samples_per_bucket": min_samples,
        "used_dates": int(used_dates),
        "player_rows": int(len(full)),
        "scales": scales,
        "bucket_rows": output_rows,
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_saved_lineup_version_payload(
    bucket_name: str,
    selected_date: date,
    run_id: str,
    version_key: str,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> dict[str, Any] | None:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    payload = store.read_lineup_version_json(selected_date, run_id, version_key)
    if not isinstance(payload, dict):
        return None
    return payload


def _list_players_blob_names(store: CbbGcsStore) -> list[str]:
    list_fn = getattr(store, "list_players_blob_names", None)
    if callable(list_fn):
        return list_fn()
    blobs = store.bucket.list_blobs(prefix="cbb/players/")
    names = [blob.name for blob in blobs if blob.name.endswith("_players.csv")]
    names.sort()
    return names


def _read_players_csv_blob(store: CbbGcsStore, blob_name: str) -> str:
    read_fn = getattr(store, "read_players_csv_blob", None)
    if callable(read_fn):
        return read_fn(blob_name)
    blob = store.bucket.blob(blob_name)
    return blob.download_as_text(encoding="utf-8")


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


@st.cache_data(ttl=600, show_spinner=False)
def load_vegas_review_frame_for_date(
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
    raw_payload = store.read_raw_json(selected_date)
    odds_payload = store.read_odds_json(selected_date)
    if not isinstance(raw_payload, dict) or not isinstance(odds_payload, dict):
        return pd.DataFrame()
    return build_vegas_review_games_frame(raw_payloads=[raw_payload], odds_payloads=[odds_payload])


@st.cache_data(ttl=600, show_spinner=False)
def load_props_frame_for_date(
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
    payload = store.read_props_json(selected_date)
    if payload is None:
        return pd.DataFrame()
    from college_basketball_dfs.cbb_odds import flatten_player_props_payload

    rows = flatten_player_props_payload(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame


@st.cache_data(ttl=600, show_spinner=False)
def load_dk_slate_frame_for_date(
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
    csv_text = _read_dk_slate_csv(store, selected_date)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


@st.cache_data(ttl=300, show_spinner=False)
def load_injuries_feed_frame(
    bucket_name: str,
    selected_date: date | None,
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
    csv_text = _read_injuries_feed_csv(store, selected_date=selected_date)
    if not csv_text or not csv_text.strip():
        return normalize_injuries_frame(None)
    frame = pd.read_csv(io.StringIO(csv_text))
    return normalize_injuries_frame(frame)


@st.cache_data(ttl=300, show_spinner=False)
def load_injuries_manual_frame(
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
    csv_text = _read_injuries_manual_csv(store)
    if not csv_text or not csv_text.strip():
        return normalize_injuries_frame(None)
    frame = pd.read_csv(io.StringIO(csv_text))
    return normalize_injuries_frame(frame)


@st.cache_data(ttl=300, show_spinner=False)
def load_injuries_frame(
    bucket_name: str,
    selected_date: date | None,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> pd.DataFrame:
    feed_df = load_injuries_feed_frame(
        bucket_name=bucket_name,
        selected_date=selected_date,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    manual_df = load_injuries_manual_frame(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if feed_df.empty and manual_df.empty:
        # Backward compatibility: fall back to older single-source injuries file only when new layers are absent.
        client = build_storage_client(
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
            project=gcp_project,
        )
        store = CbbGcsStore(bucket_name=bucket_name, client=client)
        legacy_csv = _read_injuries_csv(store)
        if not legacy_csv or not legacy_csv.strip():
            return normalize_injuries_frame(None)
        legacy_frame = pd.read_csv(io.StringIO(legacy_csv))
        return normalize_injuries_frame(legacy_frame)

    frames: list[pd.DataFrame] = []
    if not feed_df.empty:
        feed = feed_df.copy()
        feed["_source"] = "feed"
        frames.append(feed)
    if not manual_df.empty:
        manual = manual_df.copy()
        manual["_source"] = "manual"
        frames.append(manual)

    combined = pd.concat(frames, ignore_index=True)
    combined["_injury_key"] = combined.apply(
        lambda r: f"{str(r.get('player_name') or '').strip().lower()}|{str(r.get('team') or '').strip().lower()}",
        axis=1,
    )
    # Manual rows are appended last and override feed rows on duplicate player+team keys.
    combined = combined.drop_duplicates(subset=["_injury_key"], keep="last")
    combined = combined.drop(columns=["_injury_key", "_source"], errors="ignore")
    return normalize_injuries_frame(combined)


@st.cache_data(ttl=1800, show_spinner=False)
def load_season_player_history_frame(
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
    season_start = season_start_for_date(selected_date)
    selected_iso = selected_date.isoformat()
    blob_names = _list_players_blob_names(store)

    frames: list[pd.DataFrame] = []
    for blob_name in blob_names:
        match = re.search(r"(\d{4}-\d{2}-\d{2})_players\.csv$", blob_name)
        if not match:
            continue
        blob_date = match.group(1)
        if blob_date < season_start.isoformat() or blob_date > selected_iso:
            continue
        csv_text = _read_players_csv_blob(store, blob_name)
        if not csv_text.strip():
            continue
        df = pd.read_csv(io.StringIO(csv_text))
        needed = [
            "game_date",
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
            "fga",
            "fta",
            "dk_fpts",
        ]
        cols = [c for c in needed if c in df.columns]
        if not cols:
            continue
        frames.append(df[cols].copy())

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=1800, show_spinner=False)
def load_season_vegas_history_frame(
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

    season_start = season_start_for_date(selected_date)
    end_date = selected_date - timedelta(days=1)
    if end_date < season_start:
        return pd.DataFrame()

    raw_payloads: list[dict[str, Any]] = []
    odds_payloads: list[dict[str, Any]] = []
    for d in iter_dates(season_start, end_date):
        raw_payload = store.read_raw_json(d)
        if raw_payload is not None:
            raw_payloads.append(raw_payload)
        odds_payload = store.read_odds_json(d)
        if odds_payload is not None:
            odds_payloads.append(odds_payload)

    if not raw_payloads or not odds_payloads:
        return pd.DataFrame()
    return build_vegas_review_games_frame(raw_payloads=raw_payloads, odds_payloads=odds_payloads)


def _ownership_to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_ownership_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership"])
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
    out["actual_ownership"] = out["actual_ownership"].map(_ownership_to_float)
    out["name_key"] = out["Name"].map(_norm_name_key)
    out["name_key_loose"] = out["Name"].map(_norm_name_key_loose)
    out = out.loc[(out["ID"] != "") | (out["Name"] != "")]
    out = out.sort_values(["ID", "Name"], ascending=[True, True]).drop_duplicates(
        subset=["ID", "name_key", "TeamAbbrev"], keep="last"
    )
    return out[["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"]].reset_index(drop=True)


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
    csv_text = _read_projections_csv(store, selected_date)
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
    csv_text = _read_ownership_csv(store, selected_date)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"])
    df = pd.read_csv(io.StringIO(csv_text))
    return normalize_ownership_frame(df)


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
    blob_name = f"cbb/players/{selected_date.isoformat()}_players.csv"
    try:
        csv_text = _read_players_csv_blob(store, blob_name)
    except Exception:
        return pd.DataFrame()
    if not csv_text.strip():
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
    out = df[cols].copy()
    out = out.rename(
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
    for c in [
        "actual_minutes",
        "actual_points",
        "actual_rebounds",
        "actual_assists",
        "actual_steals",
        "actual_blocks",
        "actual_turnovers",
        "actual_threes",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = 0.0

    dd_count = (
        (out["actual_points"].fillna(-1) >= 10).astype(int)
        + (out["actual_rebounds"].fillna(-1) >= 10).astype(int)
        + (out["actual_assists"].fillna(-1) >= 10).astype(int)
        + (out["actual_steals"].fillna(-1) >= 10).astype(int)
        + (out["actual_blocks"].fillna(-1) >= 10).astype(int)
    )
    bonus = (dd_count >= 2).astype(int) * 1.5 + (dd_count >= 3).astype(int) * 3.0
    out["actual_dk_points"] = (
        out["actual_points"].fillna(0)
        + (1.25 * out["actual_rebounds"].fillna(0))
        + (1.5 * out["actual_assists"].fillna(0))
        + (2.0 * out["actual_steals"].fillna(0))
        + (2.0 * out["actual_blocks"].fillna(0))
        - (0.5 * out["actual_turnovers"].fillna(0))
        + (0.5 * out["actual_threes"].fillna(0))
        + bonus
    )
    out["ID"] = out["ID"].astype(str).str.strip()
    out["Name"] = out["Name"].astype(str).str.strip()
    out["team_name"] = out["team_name"].astype(str).str.strip()
    return out


@st.cache_data(ttl=600, show_spinner=False)
def load_contest_standings_frame(
    bucket_name: str,
    selected_date: date,
    contest_id: str,
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
    csv_text = _read_contest_standings_csv(store, selected_date, contest_id)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


def build_optimizer_pool_for_date(
    bucket_name: str,
    slate_date: date,
    bookmaker: str | None,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slate_df = load_dk_slate_frame_for_date(
        bucket_name=bucket_name,
        selected_date=slate_date,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    injuries_df = load_injuries_frame(
        bucket_name=bucket_name,
        selected_date=slate_date,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if slate_df.empty:
        return pd.DataFrame(), pd.DataFrame(), slate_df, injuries_df, pd.DataFrame()

    filtered_slate, removed_injured = remove_injured_players(slate_df, injuries_df)
    season_history_df = load_season_player_history_frame(
        bucket_name=bucket_name,
        selected_date=slate_date,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    odds_df = load_odds_frame_for_date(
        bucket_name=bucket_name,
        selected_date=slate_date,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    vegas_history_df = load_season_vegas_history_frame(
        bucket_name=bucket_name,
        selected_date=slate_date,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    tail_model = fit_total_tail_model(vegas_history_df)
    odds_scored_df = score_odds_games_for_tail(odds_df, tail_model) if not odds_df.empty else pd.DataFrame()

    props_df = load_props_frame_for_date(
        bucket_name=bucket_name,
        selected_date=slate_date,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    pool_df = build_player_pool(
        slate_df=filtered_slate,
        props_df=props_df,
        season_stats_df=season_history_df,
        bookmaker_filter=(bookmaker or None),
        odds_games_df=odds_scored_df,
    )
    return pool_df, removed_injured, slate_df, injuries_df, season_history_df


st.set_page_config(page_title="CBB Admin Cache", layout="wide")
st.title("College Basketball Admin Cache")
st.caption("Cache-first data pipeline backed by Google Cloud Storage.")

default_bucket = os.getenv("CBB_GCS_BUCKET", "").strip() or (_secret("cbb_gcs_bucket") or "")
default_base_url = os.getenv("NCAA_API_BASE_URL", "https://ncaa-api.henrygd.me").strip()
default_project = os.getenv("GCP_PROJECT", "").strip() or (_secret("gcp_project") or "")
odds_api_key = (_resolve_odds_api_key() or "").strip()
default_bookmakers = os.getenv("CBB_ODDS_BOOKMAKERS", "").strip() or (_secret("cbb_odds_bookmakers") or "fanduel")
default_season_start = season_start_for_date(date.today())
default_props_markets = "player_points,player_rebounds,player_assists"
default_props_event_sleep_seconds = float(os.getenv("CBB_ODDS_EVENT_SLEEP_SECONDS", "0.6"))

with st.sidebar:
    st.header("Global Settings")
    bucket_name = st.text_input("GCS Bucket", value=default_bucket)
    base_url = st.text_input("NCAA API Base URL", value=default_base_url)
    gcp_project = st.text_input("GCP Project (optional)", value=default_project)
    default_bookmakers_filter = st.text_input(
        "Default Bookmakers Filter",
        value=default_bookmakers,
        help="Comma-separated bookmaker keys (example: fanduel). Leave blank for all.",
    )
    st.caption(
        "The Odds API key source: "
        + ("loaded from secrets/env" if odds_api_key else "missing (`the_odds_api_key`)")
    )
    st.caption("Configure workflow-specific settings inside each tab.")

cred_json = _resolve_credential_json()
cred_json_b64 = _resolve_credential_json_b64()

if not cred_json and not cred_json_b64:
    st.info(
        "No inline GCP service-account secret detected. "
        "Using default Google credentials if available."
    )

tab_game, tab_props, tab_backfill, tab_dk, tab_injuries, tab_slate_vegas, tab_lineups, tab_projection_review, tab_tournament_review = st.tabs(
    [
        "Game Data",
        "Prop Data",
        "Backfill",
        "DK Slate",
        "Injuries",
        "Slate + Vegas",
        "Lineup Generator",
        "Projection Review",
        "Tournament Review",
    ]
)
# Render Agentic Review directly under Slate + Vegas (same tab container).
tab_agentic_review = tab_slate_vegas

with tab_game:
    st.subheader("Game Imports")
    game_selected_date = st.date_input("Slate Date", value=prior_day(), key="game_slate_date")
    game_force_refresh = st.checkbox(
        "Force API refresh (ignore cached raw JSON)",
        value=False,
        key="game_force_refresh",
    )
    game_bookmakers_filter = st.text_input(
        "Game Odds Bookmakers",
        value=(default_bookmakers_filter.strip() or "fanduel"),
        key="game_bookmakers_filter",
        help="Comma-separated bookmaker keys used by `Run Odds Import`.",
    )
    st.caption(
        "`Run Game Import (Cache/API)` may call NCAA API and writes/updates GCS cache. "
        "`Preview Cached GCS Data (Read Only)` only reads existing cache files."
    )
    c1, c2, c3 = st.columns(3)
    run_clicked = c1.button("Run Game Import (Cache/API)", key="run_cache_pipeline")
    run_odds_clicked = c2.button("Run Odds Import", key="run_odds_import")
    preview_clicked = c3.button("Preview Cached GCS Data (Read Only)", key="preview_cached_data")

with tab_props:
    st.subheader("Prop Imports")
    default_props_bookmakers = _csv_values(default_bookmakers_filter) or ["fanduel"]
    props_bookmaker_options: list[str] = []
    for key in COMMON_ODDS_BOOKMAKER_KEYS + default_props_bookmakers:
        k = str(key).strip().lower()
        if k and k not in props_bookmaker_options:
            props_bookmaker_options.append(k)
    props_date_preset = st.selectbox(
        "Props Date Preset",
        options=["Custom", "Today", "Tomorrow"],
        index=2,
        key="props_date_preset",
        help="Use Today/Tomorrow for pregame props pulls without manual date entry.",
    )
    if props_date_preset == "Today":
        props_selected_date = date.today()
        st.caption(f"Props Date: `{props_selected_date.isoformat()}`")
    elif props_date_preset == "Tomorrow":
        props_selected_date = date.today() + timedelta(days=1)
        st.caption(f"Props Date: `{props_selected_date.isoformat()}`")
    else:
        props_selected_date = st.date_input("Props Date", value=prior_day(), key="props_custom_date")
    props_markets = st.text_input(
        "Props Markets",
        value=default_props_markets,
        key="props_markets",
        help="Comma-separated The Odds API player prop market keys.",
    )
    props_bookmakers_selected = st.multiselect(
        "Props Bookmakers",
        options=props_bookmaker_options,
        default=[k for k in default_props_bookmakers if k in props_bookmaker_options],
        key="props_bookmakers_selected",
        help="Checkbox dropdown of bookmaker keys used for props import.",
    )
    props_bookmakers_custom = st.text_input(
        "Additional Props Bookmakers (optional)",
        value="",
        key="props_bookmakers_custom",
        help="Comma-separated extra keys not listed above.",
    )
    props_bookmakers_keys = _merged_bookmakers(props_bookmakers_selected, props_bookmakers_custom)
    props_bookmakers_filter = ",".join(props_bookmakers_keys)
    st.caption(
        "Resolved props bookmakers: "
        + (f"`{props_bookmakers_filter}`" if props_bookmakers_filter else "`all books (no filter)`")
    )
    props_fetch_mode = st.selectbox(
        "Props Fetch Mode",
        options=["Pregame Live", "Historical Snapshot"],
        index=0,
        key="props_fetch_mode",
        help="Use Pregame Live for today/tomorrow pulls prior to tip-off.",
    )
    props_import_mode = st.selectbox(
        "Props Import Mode",
        options=["Auto (Cache -> API)", "Cache Only", "Force API Refresh"],
        index=0,
        key="props_import_mode",
        help="Choose whether props import can call API or only load from cached GCS data.",
    )
    props_force_refresh_auto = st.checkbox(
        "Force API refresh in Auto mode",
        value=False,
        key="props_force_refresh_auto",
        help="Used only when Props Import Mode is `Auto (Cache -> API)`.",
    )
    props_event_sleep_seconds = st.number_input(
        "Props Event Sleep Seconds",
        min_value=0.0,
        max_value=5.0,
        value=default_props_event_sleep_seconds,
        step=0.1,
        key="props_event_sleep_seconds",
        help="Delay between event-level props requests to avoid Odds API frequency limits (429).",
    )
    run_props_clicked = st.button("Run Props Import", key="run_props_import")

with tab_backfill:
    st.subheader("Backfill Jobs")
    bcfg1, bcfg2 = st.columns(2)
    backfill_start = bcfg1.date_input("Backfill Start", value=default_season_start, key="backfill_start")
    backfill_end = bcfg2.date_input("Backfill End", value=prior_day(), key="backfill_end")
    bcfg3, bcfg4, bcfg5 = st.columns(3)
    backfill_force_refresh = bcfg3.checkbox(
        "Force API refresh",
        value=False,
        key="backfill_force_refresh",
        help="Ignore cached raw JSON/odds files and refetch API responses.",
    )
    backfill_sleep = bcfg4.number_input(
        "Sleep Seconds",
        min_value=0.0,
        max_value=5.0,
        value=0.0,
        step=0.1,
        key="backfill_sleep_seconds",
    )
    stop_on_error = bcfg5.checkbox("Stop On Error", value=False, key="backfill_stop_on_error")
    backfill_bookmakers_filter = st.text_input(
        "Odds Backfill Bookmakers",
        value=(default_bookmakers_filter.strip() or "fanduel"),
        key="backfill_bookmakers_filter",
        help="Comma-separated bookmaker keys for odds backfill.",
    )
    c4, c5 = st.columns(2)
    run_backfill_clicked = c4.button("Run Season Backfill", key="run_season_backfill")
    run_odds_backfill_clicked = c5.button("Run Odds Season Backfill", key="run_odds_season_backfill")

with tab_dk:
    st.subheader("DraftKings Slate Upload")
    dk_slate_date = st.date_input("DraftKings Slate Date", value=game_selected_date, key="dk_slate_date")
    st.caption("Each date stores one DK slate file. Uploading/replacing only affects the selected date.")
    uploaded_dk_slate = st.file_uploader(
        "Upload DraftKings Slate CSV",
        type=["csv"],
        key="dk_slate_upload",
        help="Upload the DraftKings player/salary slate CSV for this date.",
    )
    delete_dk_slate_confirm = st.checkbox(
        "Confirm delete for selected date",
        value=False,
        key="confirm_delete_dk_slate",
    )
    d1, d2, d3 = st.columns(3)
    upload_dk_slate_clicked = d1.button("Upload DK Slate to GCS", key="upload_dk_slate_to_gcs")
    load_dk_slate_clicked = d2.button("Refresh Cached Slate View", key="refresh_dk_slate_view")
    delete_dk_slate_clicked = d3.button("Delete Cached Slate for Date", key="delete_dk_slate_for_date")

with tab_game:
    if run_clicked:
        if not bucket_name:
            st.error("Set a GCS bucket before running.")
        else:
            with st.spinner("Running CBB cache pipeline..."):
                try:
                    summary = run_cbb_cache_pipeline(
                        game_date=game_selected_date,
                        bucket_name=bucket_name,
                        ncaa_base_url=base_url,
                        force_refresh=game_force_refresh,
                        gcp_project=gcp_project or None,
                        gcp_service_account_json=cred_json,
                        gcp_service_account_json_b64=cred_json_b64,
                    )
                    load_team_lookup_frame.clear()
                    st.session_state["cbb_last_summary"] = summary
                    st.success("Pipeline completed.")
                except Exception as exc:
                    st.exception(exc)

with tab_game:
    if run_odds_clicked:
        if not bucket_name:
            st.error("Set a GCS bucket before importing odds.")
        elif not odds_api_key:
            st.error("Set The Odds API key in sidebar or Streamlit secrets (`the_odds_api_key`).")
        else:
            with st.spinner("Importing game odds from The Odds API..."):
                try:
                    summary = run_cbb_odds_pipeline(
                        game_date=game_selected_date,
                        bucket_name=bucket_name,
                        odds_api_key=odds_api_key,
                        bookmakers=(game_bookmakers_filter.strip() or None),
                        historical_mode=(game_selected_date < date.today()),
                        historical_snapshot_time=f"{game_selected_date.isoformat()}T23:59:59Z"
                        if game_selected_date < date.today()
                        else None,
                        force_refresh=game_force_refresh,
                        gcp_project=gcp_project or None,
                        gcp_service_account_json=cred_json,
                        gcp_service_account_json_b64=cred_json_b64,
                    )
                    load_odds_frame_for_date.clear()
                    st.session_state["cbb_odds_summary"] = summary
                    st.success("Odds import completed.")
                except Exception as exc:
                    st.exception(exc)

with tab_props:
    if run_props_clicked:
        if not bucket_name:
            st.error("Set a GCS bucket before importing props.")
        elif props_import_mode != "Cache Only" and not odds_api_key:
            st.error("Set The Odds API key in Streamlit secrets (`the_odds_api_key`).")
        else:
            with st.spinner("Loading player props..."):
                try:
                    if props_import_mode == "Cache Only":
                        from college_basketball_dfs.cbb_odds import flatten_player_props_payload

                        client = build_storage_client(
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                            project=gcp_project or None,
                        )
                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                        payload = store.read_props_json(props_selected_date)
                        if payload is None:
                            summary = {
                                "game_date": props_selected_date.isoformat(),
                                "props_cache_hit": False,
                                "historical_mode": (props_fetch_mode == "Historical Snapshot"),
                                "markets": props_markets.strip(),
                                "bookmakers": (props_bookmakers_filter.strip() or None),
                                "bucket_name": bucket_name,
                                "props_blob": store.props_blob_name(props_selected_date),
                                "props_lines_blob": store.props_lines_blob_name(props_selected_date),
                                "event_count": 0,
                                "prop_rows": 0,
                                "cache_only": True,
                            }
                            st.warning("No cached props found for selected date.")
                        else:
                            rows = flatten_player_props_payload(payload)
                            summary = {
                                "game_date": props_selected_date.isoformat(),
                                "props_cache_hit": True,
                                "historical_mode": (props_fetch_mode == "Historical Snapshot"),
                                "markets": props_markets.strip(),
                                "bookmakers": (props_bookmakers_filter.strip() or None),
                                "bucket_name": bucket_name,
                                "props_blob": store.props_blob_name(props_selected_date),
                                "props_lines_blob": store.props_lines_blob_name(props_selected_date),
                                "event_count": len(payload.get("events", [])) if isinstance(payload.get("events"), list) else 0,
                                "prop_rows": len(rows),
                                "cache_only": True,
                            }
                            st.success("Props loaded from cache.")
                    else:
                        props_kwargs = {
                            "game_date": props_selected_date,
                            "bucket_name": bucket_name,
                            "odds_api_key": odds_api_key,
                            "markets": props_markets.strip(),
                            "bookmakers": (props_bookmakers_filter.strip() or None),
                            "historical_mode": (props_fetch_mode == "Historical Snapshot"),
                            "historical_snapshot_time": None,
                            "inter_event_sleep_seconds": float(props_event_sleep_seconds),
                            "force_refresh": (True if props_import_mode == "Force API Refresh" else props_force_refresh_auto),
                            "gcp_project": gcp_project or None,
                            "gcp_service_account_json": cred_json,
                            "gcp_service_account_json_b64": cred_json_b64,
                        }
                        # Backward-compat: if deployed pipeline is older, drop unknown kwargs.
                        allowed = set(inspect.signature(run_cbb_props_pipeline).parameters.keys())
                        filtered_props_kwargs = {k: v for k, v in props_kwargs.items() if k in allowed}
                        summary = run_cbb_props_pipeline(**filtered_props_kwargs)
                        st.success("Props import completed.")
                    load_props_frame_for_date.clear()
                    st.session_state["cbb_props_summary"] = summary
                except Exception as exc:
                    st.exception(exc)

with tab_dk:
    if delete_dk_slate_clicked:
        if not bucket_name:
            st.error("Set a GCS bucket before deleting DraftKings slate.")
        elif not delete_dk_slate_confirm:
            st.error("Check `Confirm delete for selected date` before deleting.")
        else:
            with st.spinner("Deleting DraftKings slate from GCS..."):
                try:
                    client = build_storage_client(
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                        project=gcp_project or None,
                    )
                    store = CbbGcsStore(bucket_name=bucket_name, client=client)
                    deleted, blob_name = _delete_dk_slate_csv(store, dk_slate_date)
                    load_dk_slate_frame_for_date.clear()
                    if deleted:
                        st.session_state.pop("cbb_dk_upload_summary", None)
                        st.success(f"Deleted `{blob_name}`")
                    else:
                        st.warning(f"No cached slate found for selected date. Expected `{blob_name}`")
                except Exception as exc:
                    st.exception(exc)

with tab_dk:
    if upload_dk_slate_clicked:
        if not bucket_name:
            st.error("Set a GCS bucket before uploading DraftKings slate.")
        elif uploaded_dk_slate is None:
            st.error("Choose a DraftKings slate CSV file before uploading.")
        else:
            with st.spinner("Uploading DraftKings slate CSV to GCS..."):
                try:
                    csv_bytes = uploaded_dk_slate.getvalue()
                    csv_text = csv_bytes.decode("utf-8-sig")
                    if not csv_text.strip():
                        st.error("Uploaded CSV is empty.")
                    else:
                        df = pd.read_csv(io.StringIO(csv_text))
                        client = build_storage_client(
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                            project=gcp_project or None,
                        )
                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                        blob_name = _write_dk_slate_csv(store, dk_slate_date, csv_text)
                        load_dk_slate_frame_for_date.clear()
                        st.session_state["cbb_dk_upload_summary"] = {
                            "slate_date": dk_slate_date.isoformat(),
                            "bucket_name": bucket_name,
                            "dk_slate_blob": blob_name,
                            "source_file_name": uploaded_dk_slate.name,
                            "row_count": int(len(df)),
                            "column_count": int(len(df.columns)),
                        }
                        st.success("DraftKings slate uploaded.")
                except Exception as exc:
                    st.exception(exc)

with tab_backfill:
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
                        force_refresh=backfill_force_refresh,
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

with tab_backfill:
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
                        bookmakers=(backfill_bookmakers_filter.strip() or None),
                        historical_mode=True,
                        force_refresh=backfill_force_refresh,
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

with tab_game:
    summary = st.session_state.get("cbb_last_summary")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games", summary["game_count"])
        c2.metric("Boxscores OK", summary["boxscore_success_count"])
        c3.metric("Boxscores Failed", summary["boxscore_failure_count"])
        c4.metric("Player Rows", summary["player_row_count"])
        st.json(summary)

with tab_game:
    odds_summary = st.session_state.get("cbb_odds_summary")
    if odds_summary:
        st.subheader("Odds Import Summary")
        o1, o2, o3 = st.columns(3)
        o1.metric("Events", odds_summary["event_count"])
        o2.metric("Game Rows", odds_summary["odds_game_rows"])
        o3.metric("Cache Hit", "Yes" if odds_summary["odds_cache_hit"] else "No")
        st.json(odds_summary)

with tab_props:
    props_summary = st.session_state.get("cbb_props_summary")
    if props_summary:
        st.subheader("Props Import Summary")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("Events", props_summary["event_count"])
        p2.metric("Prop Rows", props_summary["prop_rows"])
        p3.metric("Cache Hit", "Yes" if props_summary["props_cache_hit"] else "No")
        p4.metric("Events w/ Books", props_summary.get("events_with_bookmakers", 0))
        p5.metric("Events w/ Markets", props_summary.get("events_with_requested_markets", 0))
        p6.metric("Mode", "Historical" if props_summary.get("historical_mode") else "Live")
        st.json(props_summary)

with tab_backfill:
    odds_backfill_summary = st.session_state.get("cbb_odds_backfill_summary")
    if odds_backfill_summary:
        st.subheader("Odds Season Backfill Summary")
        ob1, ob2, ob3, ob4 = st.columns(4)
        ob1.metric("Total Dates", odds_backfill_summary["total_dates"])
        ob2.metric("Success Dates", odds_backfill_summary["success_dates"])
        ob3.metric("Failed Dates", odds_backfill_summary["failed_dates"])
        ob4.metric("Cache Hits", odds_backfill_summary["odds_cache_hits"])
        st.json(odds_backfill_summary)

with tab_backfill:
    backfill_summary = st.session_state.get("cbb_backfill_summary")
    if backfill_summary:
        st.subheader("Season Backfill Summary")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Total Dates", backfill_summary["total_dates"])
        b2.metric("Success Dates", backfill_summary["success_dates"])
        b3.metric("Failed Dates", backfill_summary["failed_dates"])
        b4.metric("Cache Hits", backfill_summary["raw_cache_hits"])
        st.json(backfill_summary)

with tab_dk:
    if load_dk_slate_clicked:
        load_dk_slate_frame_for_date.clear()

    dk_upload_summary = st.session_state.get("cbb_dk_upload_summary")
    if dk_upload_summary:
        st.subheader("DK Upload Summary")
        st.json(dk_upload_summary)

    st.subheader("Cached DraftKings Slate")
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to load DraftKings slate data.")
    else:
        try:
            dk_df = load_dk_slate_frame_for_date(
                bucket_name=bucket_name,
                selected_date=dk_slate_date,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if dk_df.empty:
                st.warning("No cached DraftKings slate found for selected date. Upload a CSV first.")
            else:
                st.caption(f"Rows: {len(dk_df):,} | Columns: {len(dk_df.columns):,}")
                st.dataframe(dk_df, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.exception(exc)

with tab_injuries:
    st.subheader("Injuries")
    st.caption(
        "Two-layer injury system: feed CSV is slate-specific by date, "
        "and manual entries persist as overrides for late news."
    )
    injury_feed_date = st.date_input(
        "Injury Feed Date",
        value=dk_slate_date,
        key="injury_feed_date",
        help="Uploaded injury feed applies only to this slate date.",
    )
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to manage injuries.")
    else:
        try:
            feed_df = load_injuries_feed_frame(
                bucket_name=bucket_name,
                selected_date=injury_feed_date,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            manual_df = load_injuries_manual_frame(
                bucket_name=bucket_name,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            effective_df = load_injuries_frame(
                bucket_name=bucket_name,
                selected_date=injury_feed_date,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Feed Rows", int(len(feed_df)))
            c2.metric("Manual Rows", int(len(manual_df)))
            c3.metric("Effective Rows", int(len(effective_df)))

            st.subheader("Injury Feed Upload (CSV Replace)")
            st.caption(
                "Upload your latest injury report CSV for the selected date. "
                "Replacing feed clears stale source injuries for that date only."
            )
            feed_upload = st.file_uploader(
                "Upload Injury Feed CSV",
                type=["csv"],
                key="injuries_feed_upload",
            )
            uf1, uf2, uf3 = st.columns(3)
            replace_feed_clicked = uf1.button("Replace Feed CSV", key="replace_injury_feed_csv")
            clear_feed_clicked = uf2.button("Clear Feed CSV", key="clear_injury_feed_csv")
            reload_all_clicked = uf3.button("Reload Injury Data", key="reload_injury_data")

            if replace_feed_clicked or clear_feed_clicked:
                client = build_storage_client(
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                    project=gcp_project or None,
                )
                store = CbbGcsStore(bucket_name=bucket_name, client=client)

                if replace_feed_clicked:
                    if feed_upload is None:
                        st.error("Choose an injury feed CSV file first.")
                    else:
                        csv_text = feed_upload.getvalue().decode("utf-8-sig")
                        raw_feed = pd.read_csv(io.StringIO(csv_text))
                        normalized_feed = normalize_injuries_frame(raw_feed)
                        blob_name = _write_injuries_feed_csv(
                            store,
                            normalized_feed.to_csv(index=False),
                            selected_date=injury_feed_date,
                        )
                        load_injuries_feed_frame.clear()
                        load_injuries_manual_frame.clear()
                        load_injuries_frame.clear()
                        st.success(
                            f"Replaced feed injuries for `{injury_feed_date.isoformat()}` in `{blob_name}` "
                            f"with {len(normalized_feed)} rows."
                        )
                if clear_feed_clicked:
                    deleted, blob_name = _delete_injuries_feed_csv(store, selected_date=injury_feed_date)
                    load_injuries_feed_frame.clear()
                    load_injuries_manual_frame.clear()
                    load_injuries_frame.clear()
                    if deleted:
                        st.success(
                            f"Cleared feed injuries for `{injury_feed_date.isoformat()}` by deleting `{blob_name}`."
                        )
                    else:
                        st.info(
                            f"No feed injury file found for `{injury_feed_date.isoformat()}` at `{blob_name}`."
                        )

                feed_df = load_injuries_feed_frame(
                    bucket_name=bucket_name,
                    selected_date=injury_feed_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                manual_df = load_injuries_manual_frame(
                    bucket_name=bucket_name,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                effective_df = load_injuries_frame(
                    bucket_name=bucket_name,
                    selected_date=injury_feed_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )

            if reload_all_clicked:
                load_injuries_feed_frame.clear()
                load_injuries_manual_frame.clear()
                load_injuries_frame.clear()
                st.info("Reloaded injury data from GCS.")
                feed_df = load_injuries_feed_frame(
                    bucket_name=bucket_name,
                    selected_date=injury_feed_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                manual_df = load_injuries_manual_frame(
                    bucket_name=bucket_name,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                effective_df = load_injuries_frame(
                    bucket_name=bucket_name,
                    selected_date=injury_feed_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )

            if feed_df.empty:
                st.caption("No feed injury CSV loaded yet.")
            else:
                st.dataframe(feed_df, hide_index=True, use_container_width=True)

            st.subheader("Manual Injury Overrides")
            add_col1, add_col2, add_col3 = st.columns([2, 1, 1])
            with add_col1:
                injury_player = st.text_input("Player Name", key="inj_player_name")
            with add_col2:
                injury_team = st.text_input("Team Abbrev", key="inj_team_abbrev")
            with add_col3:
                injury_status = st.selectbox(
                    "Status",
                    options=["Out", "Doubtful", "Questionable", "Probable", "Available"],
                    index=0,
                    key="inj_status",
                )
            injury_notes = st.text_input("Notes (optional)", key="inj_notes")
            add_injury_clicked = st.button("Add Manual Injury", key="add_injury_button")

            if add_injury_clicked:
                if not injury_player.strip() or not injury_team.strip():
                    st.error("Player Name and Team Abbrev are required.")
                else:
                    new_row = pd.DataFrame(
                        [
                            {
                                "player_name": injury_player.strip(),
                                "team": injury_team.strip().upper(),
                                "status": injury_status,
                                "notes": injury_notes.strip(),
                                "active": True,
                                "updated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ"),
                            }
                        ]
                    )
                    merged = pd.concat([manual_df, new_row], ignore_index=True)
                    merged = normalize_injuries_frame(merged)
                    client = build_storage_client(
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                        project=gcp_project or None,
                    )
                    store = CbbGcsStore(bucket_name=bucket_name, client=client)
                    _write_injuries_manual_csv(store, merged.to_csv(index=False))
                    load_injuries_feed_frame.clear()
                    load_injuries_manual_frame.clear()
                    load_injuries_frame.clear()
                    st.success("Manual injury added.")
                    manual_df = load_injuries_manual_frame(
                        bucket_name=bucket_name,
                        gcp_project=gcp_project or None,
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                    )

            st.subheader("Manual Injury List Editor")
            edited_injuries = st.data_editor(
                manual_df,
                num_rows="dynamic",
                use_container_width=True,
                key="manual_injuries_editor_df",
            )
            save_col, reload_col = st.columns(2)
            save_injuries_clicked = save_col.button("Save Manual Injury List", key="save_injury_list")
            reload_injuries_clicked = reload_col.button("Reload Manual List", key="reload_injury_list")

            if save_injuries_clicked:
                normalized = normalize_injuries_frame(edited_injuries)
                client = build_storage_client(
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                    project=gcp_project or None,
                )
                store = CbbGcsStore(bucket_name=bucket_name, client=client)
                blob_name = _write_injuries_manual_csv(store, normalized.to_csv(index=False))
                load_injuries_feed_frame.clear()
                load_injuries_manual_frame.clear()
                load_injuries_frame.clear()
                st.success(f"Saved manual injury list to `{blob_name}`")

            if reload_injuries_clicked:
                load_injuries_feed_frame.clear()
                load_injuries_manual_frame.clear()
                load_injuries_frame.clear()
                st.info("Reloaded manual injury list from GCS.")

            st.subheader("Effective Injury Filter (Feed + Manual)")
            st.caption("This combined list is applied when building Slate + Vegas and generating lineups.")
            if effective_df.empty:
                st.info("No effective injuries loaded.")
            else:
                st.dataframe(effective_df, hide_index=True, use_container_width=True)
        except Exception as exc:
            st.exception(exc)

with tab_slate_vegas:
    st.subheader("Slate + Vegas Player Pool")
    st.caption("Lineup Generator uses the `blended_projection` (Projected DK Points) from this table.")
    slate_vegas_date = st.date_input("DK/Optimizer Slate Date", value=game_selected_date, key="slate_vegas_date")
    vegas_bookmaker = st.text_input(
        "Vegas Bookmaker Source",
        value=(default_bookmakers_filter.strip() or "fanduel"),
        key="slate_vegas_bookmaker",
        help="Use the same bookmaker used for odds/props imports (example: fanduel).",
    )
    refresh_pool_clicked = st.button("Refresh Slate + Vegas", key="refresh_slate_vegas_pool")
    if refresh_pool_clicked:
        load_dk_slate_frame_for_date.clear()
        load_props_frame_for_date.clear()
        load_odds_frame_for_date.clear()
        load_injuries_frame.clear()
        load_season_player_history_frame.clear()
        load_season_vegas_history_frame.clear()

    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to build the slate player pool.")
    else:
        try:
            pool_df, removed_injured_df, raw_slate_df, _, season_history_df = build_optimizer_pool_for_date(
                bucket_name=bucket_name,
                slate_date=slate_vegas_date,
                bookmaker=(vegas_bookmaker.strip() or None),
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if raw_slate_df.empty:
                st.warning("No DraftKings slate found for selected optimizer date. Upload in `DK Slate` tab first.")
            else:
                m1, m2, m3 = st.columns(3)
                m1.metric("Raw Slate Players", int(len(raw_slate_df)))
                m2.metric("Removed (Out/Doubtful)", int(len(removed_injured_df)))
                m3.metric("Active Pool Players", int(len(pool_df)))
                mins_pct = pd.to_numeric(pool_df.get("our_minutes_avg", pd.Series(dtype=float)), errors="coerce")
                avg_mins = float(mins_pct.mean()) if len(mins_pct) and mins_pct.notna().any() else 0.0
                mins_last7 = pd.to_numeric(pool_df.get("our_minutes_last7", pd.Series(dtype=float)), errors="coerce")
                avg_mins_last7 = float(mins_last7.mean()) if len(mins_last7) and mins_last7.notna().any() else 0.0
                st.caption(
                    f"Season stats rows used: `{len(season_history_df):,}` | "
                    f"Average projected minutes: `{avg_mins:.1f}` | "
                    f"Average last-7 minutes: `{avg_mins_last7:.1f}`"
                )

                if not removed_injured_df.empty:
                    st.subheader("Removed Injured Players")
                    removed_cols = [c for c in ["Name", "TeamAbbrev", "ID"] if c in removed_injured_df.columns]
                    st.dataframe(removed_injured_df[removed_cols], hide_index=True, use_container_width=True)

                if pool_df.empty:
                    st.warning("Player pool is empty after injury filtering.")
                else:
                    show_cols = [
                        "ID",
                        "Name + ID",
                        "Name",
                        "TeamAbbrev",
                        "Position",
                        "Salary",
                        "projection_per_dollar",
                        "blended_projection",
                        "our_minutes_avg",
                        "our_minutes_last7",
                        "our_usage_proxy",
                        "our_points_proj",
                        "our_rebounds_proj",
                        "our_assists_proj",
                        "our_threes_proj",
                        "our_dk_projection",
                        "projected_dk_points",
                        "projected_ownership",
                        "leverage_score",
                        "game_tail_match_score",
                        "game_total_line",
                        "game_spread_line",
                        "game_tail_residual_mu",
                        "game_tail_sigma",
                        "game_p_plus_8",
                        "game_p_plus_12",
                        "game_volatility_score",
                        "game_avg_projected_ownership",
                        "game_tail_to_ownership",
                        "game_tail_score",
                        "vegas_over_our_flag",
                        "low_own_ceiling_flag",
                        "vegas_vs_our_delta_pct",
                        "blend_points_proj",
                        "blend_rebounds_proj",
                        "blend_assists_proj",
                        "blend_threes_proj",
                        "vegas_points_line",
                        "vegas_rebounds_line",
                        "vegas_assists_line",
                        "vegas_threes_line",
                        "vegas_markets_found",
                        "vegas_points_available",
                        "vegas_projection_usable",
                        "vegas_blend_weight",
                        "vegas_dk_projection",
                    ]
                    existing_cols = [c for c in show_cols if c in pool_df.columns]
                    display_pool = pool_df[existing_cols].sort_values("projected_dk_points", ascending=False)
                    numeric_cols = [
                        "Salary",
                        "projection_per_dollar",
                        "blended_projection",
                        "our_minutes_avg",
                        "our_minutes_last7",
                        "our_usage_proxy",
                        "our_points_proj",
                        "our_rebounds_proj",
                        "our_assists_proj",
                        "our_threes_proj",
                        "our_dk_projection",
                        "projected_dk_points",
                        "projected_ownership",
                        "leverage_score",
                        "game_tail_match_score",
                        "game_total_line",
                        "game_spread_line",
                        "game_tail_residual_mu",
                        "game_tail_sigma",
                        "game_p_plus_8",
                        "game_p_plus_12",
                        "game_volatility_score",
                        "game_avg_projected_ownership",
                        "game_tail_to_ownership",
                        "game_tail_score",
                        "vegas_vs_our_delta_pct",
                        "blend_points_proj",
                        "blend_rebounds_proj",
                        "blend_assists_proj",
                        "blend_threes_proj",
                        "vegas_points_line",
                        "vegas_rebounds_line",
                        "vegas_assists_line",
                        "vegas_threes_line",
                        "vegas_markets_found",
                        "vegas_blend_weight",
                        "vegas_dk_projection",
                    ]
                    for col in numeric_cols:
                        if col in display_pool.columns:
                            display_pool[col] = pd.to_numeric(display_pool[col], errors="coerce")
                    slate_role_filter = st.selectbox(
                        "Role Filter",
                        options=ROLE_FILTER_OPTIONS,
                        index=0,
                        key="slate_vegas_role_filter",
                        help="Filter the table by DraftKings role.",
                    )
                    display_pool_view = _filter_frame_by_role(
                        display_pool,
                        selected_role=slate_role_filter,
                        position_col="Position",
                    )
                    st.caption(f"Showing `{len(display_pool_view)}` of `{len(display_pool)}` players.")
                    st.dataframe(display_pool_view, hide_index=True, use_container_width=True)
                    save_proj_clicked = st.button("Save Projections Snapshot to GCS", key="save_proj_snapshot")
                    if save_proj_clicked:
                        client = build_storage_client(
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                            project=gcp_project or None,
                        )
                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                        blob_name = _write_projections_csv(
                            store,
                            slate_vegas_date,
                            display_pool.to_csv(index=False),
                        )
                        st.success(f"Saved projections to `{blob_name}` (same date overwrites).")
                    st.download_button(
                        "Download Active Pool CSV",
                        data=display_pool.to_csv(index=False),
                        file_name=f"cbb_active_pool_{slate_vegas_date.isoformat()}.csv",
                        mime="text/csv",
                        key="download_active_pool_csv",
                    )
        except Exception as exc:
            st.exception(exc)

with tab_lineups:
    st.subheader("DK Lineup Generator")
    if "auto_save_runs_to_gcs" not in st.session_state:
        st.session_state["auto_save_runs_to_gcs"] = True
    auto_save_runs_to_gcs = bool(st.session_state.get("auto_save_runs_to_gcs", True))
    lineup_slate_date = st.date_input("Lineup Slate Date", value=game_selected_date, key="lineup_slate_date")
    lineup_bookmaker = st.text_input(
        "Lineup Bookmaker Source",
        value=(default_bookmakers_filter.strip() or "fanduel"),
        key="lineup_bookmaker_source",
    )
    c1, c2, c3, c4 = st.columns(4)
    lineup_count = int(c1.slider("Lineups", min_value=1, max_value=150, value=20, step=1))
    contest_type = c2.selectbox("Contest Type", options=["Cash", "Small GPP", "Large GPP"], index=1)
    lineup_seed = int(c3.number_input("Random Seed", min_value=1, max_value=999999, value=7, step=1))
    run_mode_label = c4.selectbox(
        "Run Mode",
        options=["Single Version", "All Versions"],
        index=0,
        help=(
            "All Versions generates and saves all lineup models: "
            "standard_v1, spike_v1_legacy, spike_v2_tail, cluster_v1_experimental."
        ),
    )
    run_mode_key = "all" if run_mode_label == "All Versions" else "single"

    c5, c6, c7 = st.columns(3)
    if run_mode_key == "single":
        lineup_model_label = c5.selectbox(
            "Lineup Model",
            options=[
                "Standard v1",
                "Spike v1 (Legacy A/B)",
                "Spike v2 (Tail A/B)",
                "Cluster v1 (Experimental)",
            ],
            index=0,
            help=(
                "Run one lineup model. Use All Versions to save all models in a single run."
            ),
        )
        if lineup_model_label == "Spike v2 (Tail A/B)":
            selected_model_key = "spike_v2_tail"
            lineup_strategy = "spike"
            include_tail_signals = True
        elif lineup_model_label == "Cluster v1 (Experimental)":
            selected_model_key = "cluster_v1_experimental"
            lineup_strategy = "cluster"
            include_tail_signals = False
        elif lineup_model_label == "Spike v1 (Legacy A/B)":
            selected_model_key = "spike_v1_legacy"
            lineup_strategy = "spike"
            include_tail_signals = False
        else:
            selected_model_key = "standard_v1"
            lineup_strategy = "standard"
            include_tail_signals = False
    else:
        c5.caption("Lineup Models")
        c5.write("All Versions: `standard_v1`, `spike_v1_legacy`, `spike_v2_tail`, `cluster_v1_experimental`")
        selected_model_key = "standard_v1"
        lineup_strategy = "standard"
        include_tail_signals = False

    if run_mode_key == "all" or lineup_strategy == "cluster":
        st.caption(
            "Cluster v1 Phase 1 uses seed + mutation generation with target `15 clusters x 10 variants` "
            "(auto-adjusted for smaller lineup counts/slates)."
        )

    max_salary_left = int(
        c6.slider(
            "Max Salary Left Per Lineup",
            min_value=0,
            max_value=10000,
            value=500,
            step=50,
            help="Lineups must use at least 50000 - this value in salary.",
        )
    )
    global_max_exposure_pct = float(
        c7.slider(
            "Global Max Player Exposure %",
            min_value=0,
            max_value=100,
            value=60,
            step=1,
            help="Caps every player's max lineup rate across the run (locks override this cap).",
        )
    )
    s1, s2, s3 = st.columns(3)
    strict_salary_utilization = bool(
        s1.checkbox(
            "Strict Salary Utilization",
            value=False,
            help="When enabled, lineups are constrained to use at least 49,950 salary.",
        )
    )
    salary_left_target = int(
        s2.slider(
            "Salary Left Target",
            min_value=0,
            max_value=500,
            value=250,
            step=10,
            help="Scoring penalty targets this salary-left value.",
        )
    )
    auto_projection_calibration = bool(
        s3.checkbox(
            "Auto Projection Calibration",
            value=True,
            help="Scale projections from recent phantom review actual-vs-projected results.",
        )
    )
    calibration_lookback_days = 14
    if auto_projection_calibration:
        calibration_lookback_days = int(
            st.slider(
                "Calibration Lookback Days",
                min_value=3,
                max_value=60,
                value=14,
                step=1,
                help="Uses prior phantom review summaries to estimate projection scale.",
            )
        )
    b1, b2 = st.columns(2)
    auto_salary_bucket_calibration = bool(
        b1.checkbox(
            "Salary-Bucket Residual Calibration",
            value=True,
            help="Apply per-salary projection scales from recent projection-vs-actual errors.",
        )
    )
    bucket_calibration_min_samples = int(
        b2.slider(
            "Min Samples Per Salary Bucket",
            min_value=5,
            max_value=80,
            value=20,
            step=1,
            help="Buckets below this count stay at scale 1.0.",
        )
    )
    bucket_calibration_lookback_days = int(max(7, calibration_lookback_days))
    if auto_salary_bucket_calibration:
        bucket_calibration_lookback_days = int(
            st.slider(
                "Salary-Bucket Calibration Lookback Days",
                min_value=7,
                max_value=90,
                value=int(max(7, calibration_lookback_days)),
                step=1,
                help="Uses prior slates with both DK slate projections and final box-score results.",
            )
        )
    u1, u2, u3, u4 = st.columns(4)
    apply_uncertainty_shrink = bool(
        u1.checkbox(
            "Minutes/DNP Uncertainty Shrink",
            value=True,
            help="Shrinks projections for players with high minutes volatility and DNP risk.",
        )
    )
    uncertainty_shrink_pct = float(
        u2.slider(
            "Uncertainty Shrink %",
            min_value=0,
            max_value=35,
            value=18,
            step=1,
            help="Base shrink applied proportionally to projection uncertainty score.",
        )
    )
    dnp_risk_threshold_pct = float(
        u3.slider(
            "DNP Risk Threshold %",
            min_value=10,
            max_value=60,
            value=30,
            step=1,
            help="Extra shrink applies above this DNP-risk level.",
        )
    )
    high_risk_extra_shrink_pct = float(
        u4.slider(
            "High-Risk Extra Shrink %",
            min_value=0,
            max_value=30,
            value=10,
            step=1,
            help="Additional shrink for players above the DNP-risk threshold.",
        )
    )
    effective_max_salary_left = min(max_salary_left, 50) if strict_salary_utilization else max_salary_left
    spike_max_pair_overlap = 4
    if run_mode_key == "all" or lineup_strategy == "spike":
        spike_max_pair_overlap = int(
            st.slider(
                "Spike Max Shared Players (A vs B)",
                min_value=0,
                max_value=8,
                value=4,
                step=1,
                help=(
                    "Within each A/B pair, lineup B can share at most this many players with lineup A "
                    "(locks can force overlap)."
                ),
            )
        )
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to generate lineups.")
    else:
        try:
            pool_df, removed_injured_df, raw_slate_df, _, _ = build_optimizer_pool_for_date(
                bucket_name=bucket_name,
                slate_date=lineup_slate_date,
                bookmaker=(lineup_bookmaker.strip() or None),
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )

            if raw_slate_df.empty:
                st.warning("No DraftKings slate found for selected optimizer date. Upload in `DK Slate` tab first.")
            elif pool_df.empty:
                st.warning("No players available after injury filtering. Check `Injuries` tab or slate date.")
            else:
                pool_sorted = pool_df.sort_values("projected_dk_points", ascending=False).copy()
                player_labels = [
                    f"{row['Name']} ({row['TeamAbbrev']}) [{row['ID']}]"
                    for _, row in pool_sorted.iterrows()
                ]
                label_to_id = {
                    f"{row['Name']} ({row['TeamAbbrev']}) [{row['ID']}]": str(row["ID"])
                    for _, row in pool_sorted.iterrows()
                }

                l1, l2 = st.columns(2)
                locked_labels = l1.multiselect("Lock Players (in every lineup)", options=player_labels, default=[])
                excluded_labels = l2.multiselect("Exclude Players", options=player_labels, default=[])

                exposure_players = st.multiselect(
                    "Exposure Caps (max % by player)",
                    options=player_labels,
                    default=[],
                    help="Pick players to set max exposure percentage across generated lineups.",
                )
                exposure_caps: dict[str, float] = {}
                if exposure_players:
                    st.caption("Exposure Caps")
                    exp_cols = st.columns(3)
                    for idx, label in enumerate(exposure_players):
                        col = exp_cols[idx % 3]
                        with col:
                            pct = st.slider(
                                label=f"{label}",
                                min_value=0,
                                max_value=100,
                                value=100,
                                step=1,
                                key=f"exp_cap_{label}",
                            )
                            exposure_caps[label_to_id[label]] = float(pct)

                generate_lineups_clicked = st.button("Generate DK Lineups", key="generate_dk_lineups")
                if generate_lineups_clicked:
                    locked_ids = [label_to_id[x] for x in locked_labels]
                    excluded_ids = [label_to_id[x] for x in excluded_labels]
                    progress_text = st.empty()
                    progress_bar = st.progress(0, text="Starting lineup generation...")
                    projection_scale = 1.0
                    calibration_meta: dict[str, Any] = {}
                    projection_salary_bucket_scales: dict[str, float] = {}
                    salary_bucket_calibration_meta: dict[str, Any] = {}
                    uncertainty_weight = uncertainty_shrink_pct / 100.0
                    dnp_risk_threshold = dnp_risk_threshold_pct / 100.0
                    high_risk_extra_shrink = high_risk_extra_shrink_pct / 100.0
                    if auto_projection_calibration:
                        calibration_meta = compute_projection_calibration_from_phantom(
                            bucket_name=bucket_name,
                            selected_date=lineup_slate_date,
                            lookback_days=calibration_lookback_days,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        projection_scale = float(calibration_meta.get("scale") or 1.0)
                        st.caption(
                            "Projection calibration applied: "
                            f"scale={projection_scale:.4f} (raw={float(calibration_meta.get('raw_scale') or 1.0):.4f}, "
                            f"used_dates={int(calibration_meta.get('used_dates') or 0)}, "
                            f"lineups={int(calibration_meta.get('lineups') or 0)}, "
                            f"avg_delta={float(calibration_meta.get('weighted_avg_delta') or 0.0):.2f})"
                        )
                    if auto_salary_bucket_calibration:
                        salary_bucket_calibration_meta = compute_projection_salary_bucket_calibration(
                            bucket_name=bucket_name,
                            selected_date=lineup_slate_date,
                            lookback_days=bucket_calibration_lookback_days,
                            min_samples_per_bucket=bucket_calibration_min_samples,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        projection_salary_bucket_scales = {
                            str(k): float(v)
                            for k, v in (salary_bucket_calibration_meta.get("scales") or {}).items()
                            if str(k).strip()
                        }
                        bucket_rows = salary_bucket_calibration_meta.get("bucket_rows") or []
                        applied = [
                            r
                            for r in bucket_rows
                            if bool(r.get("used_for_adjustment")) and abs(float(r.get("scale", 1.0)) - 1.0) >= 1e-6
                        ]
                        if applied:
                            summary = ", ".join(
                                f"{PROJECTION_SALARY_BUCKET_LABELS.get(str(r.get('salary_bucket')), str(r.get('salary_bucket')))}: {float(r.get('scale')):.3f}"
                                for r in applied
                            )
                            st.caption(
                                "Salary-bucket calibration applied: "
                                f"{summary} "
                                f"(dates={int(salary_bucket_calibration_meta.get('used_dates') or 0)}, "
                                f"rows={int(salary_bucket_calibration_meta.get('player_rows') or 0)})."
                            )
                        else:
                            st.caption(
                                "Salary-bucket calibration found no strong adjustments "
                                f"(dates={int(salary_bucket_calibration_meta.get('used_dates') or 0)}, "
                                f"rows={int(salary_bucket_calibration_meta.get('player_rows') or 0)})."
                            )
                    if strict_salary_utilization and effective_max_salary_left < max_salary_left:
                        st.caption(
                            f"Strict salary utilization enabled: max salary left tightened from "
                            f"`{max_salary_left}` to `{effective_max_salary_left}`."
                        )
                    if apply_uncertainty_shrink and uncertainty_weight > 0.0:
                        st.caption(
                            "Minutes/DNP uncertainty shrink applied: "
                            f"base={uncertainty_weight:.2f}, "
                            f"dnp_threshold={dnp_risk_threshold:.2f}, "
                            f"high_risk_extra={high_risk_extra_shrink:.2f}."
                        )

                    if run_mode_key == "all":
                        version_plan = [
                            {
                                "version_key": "standard_v1",
                                "version_label": "Standard v1",
                                "lineup_strategy": "standard",
                                "include_tail_signals": False,
                                "model_profile": "legacy_baseline",
                                "spike_max_pair_overlap": spike_max_pair_overlap,
                            },
                            {
                                "version_key": "spike_v1_legacy",
                                "version_label": "Spike v1 (Legacy A/B)",
                                "lineup_strategy": "spike",
                                "include_tail_signals": False,
                                "model_profile": "legacy_spike_pairs",
                                "spike_max_pair_overlap": spike_max_pair_overlap,
                            },
                            {
                                "version_key": "spike_v2_tail",
                                "version_label": "Spike v2 (Tail A/B)",
                                "lineup_strategy": "spike",
                                "include_tail_signals": True,
                                "model_profile": "tail_spike_pairs",
                                "spike_max_pair_overlap": spike_max_pair_overlap,
                            },
                            {
                                "version_key": "cluster_v1_experimental",
                                "version_label": "Cluster v1 (Experimental)",
                                "lineup_strategy": "cluster",
                                "include_tail_signals": False,
                                "model_profile": "cluster_seed_mutation_v1",
                                "spike_max_pair_overlap": spike_max_pair_overlap,
                                "cluster_target_count": 15,
                                "cluster_variants_per_cluster": 10,
                            },
                        ]
                    else:
                        if selected_model_key == "spike_v2_tail":
                            version_plan = [
                                {
                                    "version_key": "spike_v2_tail",
                                    "version_label": "Spike v2 (Tail A/B)",
                                    "lineup_strategy": "spike",
                                    "include_tail_signals": True,
                                    "model_profile": "tail_spike_pairs",
                                    "spike_max_pair_overlap": spike_max_pair_overlap,
                                }
                            ]
                        elif selected_model_key == "cluster_v1_experimental":
                            version_plan = [
                                {
                                    "version_key": "cluster_v1_experimental",
                                    "version_label": "Cluster v1 (Experimental)",
                                    "lineup_strategy": "cluster",
                                    "include_tail_signals": False,
                                    "model_profile": "cluster_seed_mutation_v1",
                                    "spike_max_pair_overlap": spike_max_pair_overlap,
                                    "cluster_target_count": 15,
                                    "cluster_variants_per_cluster": 10,
                                }
                            ]
                        elif selected_model_key == "spike_v1_legacy":
                            version_plan = [
                                {
                                    "version_key": "spike_v1_legacy",
                                    "version_label": "Spike v1 (Legacy A/B)",
                                    "lineup_strategy": "spike",
                                    "include_tail_signals": False,
                                    "model_profile": "legacy_spike_pairs",
                                    "spike_max_pair_overlap": spike_max_pair_overlap,
                                }
                            ]
                        else:
                            version_plan = [
                                {
                                    "version_key": "standard_v1",
                                    "version_label": "Standard v1",
                                    "lineup_strategy": "standard",
                                    "include_tail_signals": False,
                                    "model_profile": "legacy_baseline",
                                    "spike_max_pair_overlap": spike_max_pair_overlap,
                                }
                            ]

                    total_units = max(1, lineup_count * len(version_plan))
                    generated_versions: dict[str, Any] = {}

                    for version_idx, version_cfg in enumerate(version_plan):
                        version_offset = version_idx * lineup_count

                        def _lineup_progress(done: int, total: int, status: str) -> None:
                            done_local = max(0, min(lineup_count, int(done)))
                            units_done = version_offset + done_local
                            pct = int((units_done / total_units) * 100)
                            pct = max(0, min(100, pct))
                            progress_bar.progress(pct, text=f"[{version_cfg['version_label']}] {status}")
                            progress_text.caption(f"{units_done}/{total_units}")

                        lineups, warnings = generate_lineups(
                            pool_df=pool_sorted,
                            num_lineups=lineup_count,
                            contest_type=contest_type,
                            locked_ids=locked_ids,
                            excluded_ids=excluded_ids,
                            exposure_caps_pct=exposure_caps,
                            global_max_exposure_pct=global_max_exposure_pct,
                            max_salary_left=effective_max_salary_left,
                            lineup_strategy=str(version_cfg["lineup_strategy"]),
                            include_tail_signals=bool(version_cfg.get("include_tail_signals", False)),
                            spike_max_pair_overlap=int(version_cfg["spike_max_pair_overlap"]),
                            cluster_target_count=int(version_cfg.get("cluster_target_count", 15)),
                            cluster_variants_per_cluster=int(version_cfg.get("cluster_variants_per_cluster", 10)),
                            projection_scale=projection_scale,
                            projection_salary_bucket_scales=projection_salary_bucket_scales,
                            apply_uncertainty_shrink=apply_uncertainty_shrink,
                            uncertainty_weight=uncertainty_weight,
                            high_risk_extra_shrink=high_risk_extra_shrink,
                            dnp_risk_threshold=dnp_risk_threshold,
                            salary_left_target=salary_left_target,
                            random_seed=lineup_seed + version_idx,
                            progress_callback=_lineup_progress,
                        )
                        generated_versions[str(version_cfg["version_key"])] = {
                            "version_key": str(version_cfg["version_key"]),
                            "version_label": str(version_cfg["version_label"]),
                            "lineup_strategy": str(version_cfg["lineup_strategy"]),
                            "include_tail_signals": bool(version_cfg.get("include_tail_signals", False)),
                            "model_profile": str(version_cfg.get("model_profile") or ""),
                            "lineups": lineups,
                            "warnings": warnings,
                            "upload_csv": build_dk_upload_csv(lineups) if lineups else "",
                        }

                    total_generated = sum(len(v.get("lineups") or []) for v in generated_versions.values())
                    progress_bar.progress(100, text=f"Finished: {total_generated} total lineups generated.")

                    run_bundle = {
                        "run_id": _new_lineup_run_id(),
                        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "slate_date": lineup_slate_date.isoformat(),
                        "run_mode": run_mode_key,
                        "settings": {
                            "selected_model_key": selected_model_key,
                            "lineup_count": lineup_count,
                            "contest_type": contest_type,
                            "lineup_seed": lineup_seed,
                            "max_salary_left": effective_max_salary_left,
                            "requested_max_salary_left": max_salary_left,
                            "strict_salary_utilization": strict_salary_utilization,
                            "salary_left_target": salary_left_target,
                            "global_max_exposure_pct": global_max_exposure_pct,
                            "spike_max_pair_overlap": spike_max_pair_overlap,
                            "cluster_target_count": 15,
                            "cluster_variants_per_cluster": 10,
                            "projection_scale": projection_scale,
                            "projection_salary_bucket_scales": projection_salary_bucket_scales,
                            "apply_uncertainty_shrink": apply_uncertainty_shrink,
                            "uncertainty_weight": uncertainty_weight,
                            "high_risk_extra_shrink": high_risk_extra_shrink,
                            "dnp_risk_threshold": dnp_risk_threshold,
                            "auto_projection_calibration": auto_projection_calibration,
                            "calibration_lookback_days": calibration_lookback_days,
                            "calibration_meta": calibration_meta,
                            "auto_salary_bucket_calibration": auto_salary_bucket_calibration,
                            "bucket_calibration_lookback_days": bucket_calibration_lookback_days,
                            "bucket_calibration_min_samples": bucket_calibration_min_samples,
                            "salary_bucket_calibration_meta": salary_bucket_calibration_meta,
                            "bookmaker": lineup_bookmaker.strip(),
                            "locked_ids": locked_ids,
                            "excluded_ids": excluded_ids,
                            "exposure_caps_pct": exposure_caps,
                        },
                        "versions": generated_versions,
                    }
                    st.session_state["cbb_generated_run_bundle"] = run_bundle
                    first_version_key = next(iter(generated_versions.keys()), "")
                    st.session_state["cbb_active_version_key"] = first_version_key

                    if auto_save_runs_to_gcs:
                        client = build_storage_client(
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                            project=gcp_project or None,
                        )
                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                        saved_meta = persist_lineup_run_bundle(store, lineup_slate_date, run_bundle)
                        st.session_state["cbb_generated_run_manifest"] = saved_meta.get("manifest")
                        load_saved_lineup_run_manifests.clear()
                        load_saved_lineup_version_payload.clear()
                        st.success(
                            f"Saved run `{saved_meta.get('run_id')}` with {saved_meta.get('version_count')} version(s) "
                            f"to `{saved_meta.get('manifest_blob')}`."
                        )

                run_bundle = st.session_state.get("cbb_generated_run_bundle") or {}
                generated_versions = run_bundle.get("versions") or {}
                if generated_versions:
                    version_keys = list(generated_versions.keys())
                    default_version = st.session_state.get("cbb_active_version_key")
                    if default_version not in version_keys:
                        default_version = version_keys[0]

                    def _generated_version_label(key: str) -> str:
                        version_data = generated_versions.get(key) or {}
                        label = str(version_data.get("version_label") or key)
                        strategy = str(version_data.get("lineup_strategy") or "")
                        profile = str(version_data.get("model_profile") or "")
                        tail_tag = "tail" if bool(version_data.get("include_tail_signals", False)) else "legacy"
                        profile_text = profile or tail_tag
                        return f"{label} [{strategy} | {profile_text}]"

                    active_version_key = st.selectbox(
                        "Generated Version",
                        options=version_keys,
                        index=version_keys.index(default_version),
                        format_func=_generated_version_label,
                        key="generated_version_picker",
                    )
                    st.session_state["cbb_active_version_key"] = active_version_key
                    active_version = generated_versions.get(active_version_key) or {}
                    generated = active_version.get("lineups") or []
                    warnings = [str(x) for x in (active_version.get("warnings") or [])]
                    upload_csv = str(active_version.get("upload_csv") or "")
                    st.session_state["cbb_generated_lineups"] = generated
                    st.session_state["cbb_generated_lineups_warnings"] = warnings
                    st.session_state["cbb_generated_upload_csv"] = upload_csv

                    if warnings:
                        for msg in warnings:
                            st.warning(msg)

                    g1, g2, g3, g4 = st.columns(4)
                    g1.metric("Generated Lineups", len(generated))
                    g2.metric("Injured Removed", int(len(removed_injured_df)))
                    g3.metric("Pool Size", int(len(pool_sorted)))
                    g4.metric("Run Versions", int(len(generated_versions)))
                    st.caption(
                        f"Run ID: `{run_bundle.get('run_id', '')}` | Mode: `{run_bundle.get('run_mode', 'single')}` | "
                        f"Version: `{active_version_key}`"
                    )

                    if generated:
                        summary_df = lineups_summary_frame(generated)
                        st.dataframe(summary_df, hide_index=True, use_container_width=True)

                        slots_df = lineups_slots_frame(generated)
                        if not slots_df.empty:
                            st.subheader("Generated Lineups (Slot View)")
                            st.dataframe(slots_df, hide_index=True, use_container_width=True)

                        st.download_button(
                            "Download DK Upload CSV",
                            data=upload_csv,
                            file_name=f"dk_lineups_{lineup_slate_date.isoformat()}_{active_version_key}.csv",
                            mime="text/csv",
                            key="download_dk_upload_csv",
                        )
                elif generate_lineups_clicked:
                    st.error("No lineups were generated. Adjust locks/exclusions/exposure settings and retry.")

                st.markdown("---")
                st.subheader("Saved Lineup Runs (GCS)")
                rr1, rr2, rr3 = st.columns(3)
                refresh_saved_runs_clicked = rr1.button("Refresh Saved Runs", key="refresh_saved_runs")
                auto_save_runs_to_gcs = rr2.checkbox(
                    "Auto-save generated lineup runs to GCS",
                    key="auto_save_runs_to_gcs",
                    help="Saves manifest + per-version lineup JSON/CSV + DK upload CSV.",
                )
                save_current_run_clicked = rr3.button("Save Current Run to GCS", key="save_current_run_to_gcs")
                if refresh_saved_runs_clicked:
                    load_saved_lineup_run_dates.clear()
                    load_saved_lineup_run_manifests.clear()
                    load_saved_lineup_version_payload.clear()
                if save_current_run_clicked:
                    current_run_bundle = st.session_state.get("cbb_generated_run_bundle") or {}
                    current_versions = current_run_bundle.get("versions") or {}
                    if not current_versions:
                        st.error("No generated run is loaded in session to save.")
                    else:
                        save_date = lineup_slate_date
                        bundle_slate = str(current_run_bundle.get("slate_date") or "").strip()
                        if bundle_slate:
                            try:
                                save_date = date.fromisoformat(bundle_slate)
                            except ValueError:
                                save_date = lineup_slate_date
                        client = build_storage_client(
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                            project=gcp_project or None,
                        )
                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                        saved_meta = persist_lineup_run_bundle(store, save_date, current_run_bundle)
                        st.session_state["cbb_generated_run_manifest"] = saved_meta.get("manifest")
                        load_saved_lineup_run_dates.clear()
                        load_saved_lineup_run_manifests.clear()
                        load_saved_lineup_version_payload.clear()
                        st.success(
                            f"Saved current run `{saved_meta.get('run_id')}` "
                            f"to `{saved_meta.get('manifest_blob')}`."
                        )

                saved_run_dates = load_saved_lineup_run_dates(
                    bucket_name=bucket_name,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                if not saved_run_dates:
                    st.info("No saved lineup runs found in GCS yet.")
                else:
                    date_options = ["All Dates"] + [d.isoformat() for d in saved_run_dates]
                    default_date_label = lineup_slate_date.isoformat()
                    if default_date_label not in date_options:
                        default_date_label = date_options[0]
                    selected_saved_date_label = st.selectbox(
                        "Saved Run Date",
                        options=date_options,
                        index=date_options.index(default_date_label),
                        key="saved_run_date_picker",
                    )
                    sr1, sr2 = st.columns([1, 2])
                    load_latest_clicked = sr1.button("Load Latest Run", key="load_latest_saved_run")
                    run_id_filter = sr2.text_input("Filter Run ID (optional)", key="saved_run_id_filter")

                    selected_dates: list[date]
                    if selected_saved_date_label == "All Dates":
                        selected_dates = saved_run_dates
                    else:
                        selected_dates = [date.fromisoformat(selected_saved_date_label)]

                    merged_manifests: list[dict[str, Any]] = []
                    for selected_saved_date in selected_dates:
                        manifests_for_date = load_saved_lineup_run_manifests(
                            bucket_name=bucket_name,
                            selected_date=selected_saved_date,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        for manifest in manifests_for_date:
                            entry = dict(manifest)
                            entry["_saved_date"] = selected_saved_date
                            merged_manifests.append(entry)
                    merged_manifests.sort(key=lambda x: str(x.get("generated_at_utc") or ""), reverse=True)

                    if run_id_filter.strip():
                        needle = run_id_filter.strip().lower()
                        merged_manifests = [
                            m for m in merged_manifests if needle in str(m.get("run_id") or "").strip().lower()
                        ]

                    if not merged_manifests:
                        st.info("No saved runs match the selected date/filter.")
                    else:
                        run_option_map: dict[str, dict[str, Any]] = {}
                        run_options: list[str] = []
                        for manifest in merged_manifests:
                            run_id = str(manifest.get("run_id") or "")
                            generated_at = str(manifest.get("generated_at_utc") or "")
                            run_mode = str(manifest.get("run_mode") or "single")
                            saved_date_obj = manifest.get("_saved_date")
                            saved_date_str = (
                                saved_date_obj.isoformat() if isinstance(saved_date_obj, date) else str(saved_date_obj or "")
                            )
                            label = f"{saved_date_str} | {generated_at} | {run_id} | {run_mode}"
                            run_options.append(label)
                            run_option_map[label] = manifest

                        if load_latest_clicked and run_options:
                            st.session_state["saved_run_picker"] = run_options[0]
                        if str(st.session_state.get("saved_run_picker", "")) not in run_options:
                            st.session_state.pop("saved_run_picker", None)

                        selected_run_label = st.selectbox(
                            "Saved Run",
                            options=run_options,
                            index=0,
                            key="saved_run_picker",
                        )
                        selected_manifest = run_option_map[selected_run_label]
                        selected_manifest_date = selected_manifest.get("_saved_date")
                        if not isinstance(selected_manifest_date, date):
                            selected_manifest_date = lineup_slate_date

                        versions_meta = selected_manifest.get("versions") or []
                        if versions_meta:
                            version_meta_map = {str(v.get("version_key") or ""): v for v in versions_meta}
                            saved_version_keys = [k for k in version_meta_map.keys() if k]
                            if saved_version_keys:
                                current_saved_version = str(st.session_state.get("saved_run_version_picker", "")).strip()
                                if current_saved_version not in saved_version_keys:
                                    st.session_state["saved_run_version_picker"] = saved_version_keys[0]
                                selected_saved_version = st.selectbox(
                                    "Saved Version",
                                    options=saved_version_keys,
                                    index=0,
                                    format_func=lambda k: (
                                        f"{version_meta_map[k].get('version_label', k)} "
                                        f"[{k} | "
                                        f"{(version_meta_map[k].get('model_profile') or ('tail' if bool(version_meta_map[k].get('include_tail_signals', False)) else 'legacy'))}]"
                                    ),
                                    key="saved_run_version_picker",
                                )
                                saved_payload = load_saved_lineup_version_payload(
                                    bucket_name=bucket_name,
                                    selected_date=selected_manifest_date,
                                    run_id=str(selected_manifest.get("run_id") or ""),
                                    version_key=selected_saved_version,
                                    gcp_project=gcp_project or None,
                                    service_account_json=cred_json,
                                    service_account_json_b64=cred_json_b64,
                                )
                                if isinstance(saved_payload, dict):
                                    saved_lineups = saved_payload.get("lineups") or []
                                    saved_warnings = [str(x) for x in (saved_payload.get("warnings") or [])]
                                    if saved_warnings:
                                        for msg in saved_warnings:
                                            st.warning(f"[Saved] {msg}")
                                    if saved_lineups:
                                        saved_summary_df = lineups_summary_frame(saved_lineups)
                                        st.dataframe(saved_summary_df, hide_index=True, use_container_width=True)
                                        saved_slots_df = lineups_slots_frame(saved_lineups)
                                        if not saved_slots_df.empty:
                                            st.dataframe(saved_slots_df, hide_index=True, use_container_width=True)
                                        saved_upload_csv = str(saved_payload.get("dk_upload_csv") or build_dk_upload_csv(saved_lineups))
                                        st.download_button(
                                            "Download Saved DK Upload CSV",
                                            data=saved_upload_csv,
                                            file_name=(
                                                f"dk_lineups_{selected_manifest_date.isoformat()}_"
                                                f"{selected_manifest.get('run_id', 'run')}_{selected_saved_version}.csv"
                                            ),
                                            mime="text/csv",
                                            key=f"download_saved_dk_upload_csv_{selected_saved_version}",
                                        )
        except Exception as exc:
            st.exception(exc)

with tab_projection_review:
    st.subheader("Projection Review")
    review_date = st.date_input("Review Date", value=lineup_slate_date, key="projection_review_date")
    refresh_review_clicked = st.button("Refresh Review Data", key="refresh_projection_review")
    if refresh_review_clicked:
        load_projection_snapshot_frame.clear()
        load_actual_results_frame_for_date.clear()
        load_ownership_frame_for_date.clear()

    ownership_upload = st.file_uploader(
        "Upload Actual Ownership CSV (optional)",
        type=["csv"],
        key="review_ownership_upload",
        help="Upload contest ownership export to compare projected ownership vs actual.",
    )
    upload_ownership_clicked = st.button("Save Ownership CSV to GCS", key="save_ownership_csv")

    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to run projection review.")
    else:
        try:
            if upload_ownership_clicked:
                if ownership_upload is None:
                    st.error("Choose an ownership CSV file first.")
                else:
                    csv_text = ownership_upload.getvalue().decode("utf-8-sig")
                    raw_df = pd.read_csv(io.StringIO(csv_text))
                    normalized_own = normalize_ownership_frame(raw_df)
                    if normalized_own.empty:
                        st.error("Could not find ownership rows. Include player ID/name and ownership columns.")
                    else:
                        client = build_storage_client(
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                            project=gcp_project or None,
                        )
                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                        blob_name = _write_ownership_csv(store, review_date, normalized_own.to_csv(index=False))
                        load_ownership_frame_for_date.clear()
                        st.success(f"Saved ownership file to `{blob_name}`")

            proj_df = load_projection_snapshot_frame(
                bucket_name=bucket_name,
                selected_date=review_date,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if proj_df.empty:
                st.warning("No projections snapshot found. Save one from `Slate + Vegas` first.")
            else:
                actual_df = load_actual_results_frame_for_date(
                    bucket_name=bucket_name,
                    selected_date=review_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                own_df = load_ownership_frame_for_date(
                    bucket_name=bucket_name,
                    selected_date=review_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )

                proj = proj_df.copy()
                if "ID" in proj.columns:
                    proj["ID"] = proj["ID"].astype(str).str.strip()
                if "Name" in proj.columns:
                    proj["Name"] = proj["Name"].astype(str).str.strip()
                if "TeamAbbrev" in proj.columns:
                    proj["TeamAbbrev"] = proj["TeamAbbrev"].astype(str).str.strip().str.upper()

                review = proj.copy()
                if not actual_df.empty and "ID" in review.columns:
                    review = review.merge(actual_df, on="ID", how="left", suffixes=("", "_actual"))
                elif not actual_df.empty and "Name" in review.columns:
                    review = review.merge(actual_df, on="Name", how="left", suffixes=("", "_actual"))

                if not own_df.empty:
                    review["actual_ownership"] = pd.NA
                    own_lookup = own_df.copy()
                    if "ID" in review.columns and "ID" in own_lookup.columns:
                        review["ID"] = review["ID"].map(_normalize_player_id)
                        own_lookup["ID"] = own_lookup["ID"].map(_normalize_player_id)
                        by_id = (
                            own_lookup.loc[own_lookup["ID"] != "", ["ID", "actual_ownership"]]
                            .dropna(subset=["actual_ownership"])
                            .drop_duplicates("ID")
                        )
                        if not by_id.empty:
                            review = review.merge(by_id, on="ID", how="left", suffixes=("", "_own_id"))
                            review["actual_ownership"] = pd.to_numeric(
                                review.get("actual_ownership_own_id"), errors="coerce"
                            )
                    if "Name" in review.columns and "Name" in own_lookup.columns:
                        review["name_key"] = review["Name"].map(_norm_name_key)
                        review["name_key_loose"] = review["Name"].map(_norm_name_key_loose)
                        own_lookup["name_key"] = own_lookup["Name"].map(_norm_name_key)
                        own_lookup["name_key_loose"] = own_lookup["Name"].map(_norm_name_key_loose)
                        by_name = (
                            own_lookup.loc[own_lookup["name_key"] != "", ["name_key", "actual_ownership"]]
                            .dropna(subset=["actual_ownership"])
                            .drop_duplicates("name_key")
                            .rename(columns={"actual_ownership": "actual_ownership_name"})
                        )
                        if not by_name.empty:
                            review = review.merge(by_name, on="name_key", how="left")
                        by_name_loose = (
                            own_lookup.loc[own_lookup["name_key_loose"] != "", ["name_key_loose", "actual_ownership"]]
                            .dropna(subset=["actual_ownership"])
                            .drop_duplicates("name_key_loose")
                            .rename(columns={"actual_ownership": "actual_ownership_name_loose"})
                        )
                        if not by_name_loose.empty:
                            review = review.merge(by_name_loose, on="name_key_loose", how="left")
                        review["actual_ownership"] = pd.to_numeric(
                            review.get("actual_ownership"), errors="coerce"
                        ).where(
                            pd.to_numeric(review.get("actual_ownership"), errors="coerce").notna(),
                            pd.to_numeric(review.get("actual_ownership_name"), errors="coerce"),
                        )
                        review["actual_ownership"] = pd.to_numeric(
                            review.get("actual_ownership"), errors="coerce"
                        ).where(
                            pd.to_numeric(review.get("actual_ownership"), errors="coerce").notna(),
                            pd.to_numeric(review.get("actual_ownership_name_loose"), errors="coerce"),
                        )
                    review = review.drop(
                        columns=[
                            "actual_ownership_own_id",
                            "actual_ownership_name",
                            "actual_ownership_name_loose",
                            "name_key",
                            "name_key_loose",
                        ],
                        errors="ignore",
                    )
                else:
                    review["actual_ownership"] = pd.NA

                for col in [
                    "blended_projection",
                    "projected_dk_points",
                    "our_dk_projection",
                    "vegas_dk_projection",
                    "actual_dk_points",
                    "projected_ownership",
                    "actual_ownership",
                    "actual_points",
                    "actual_rebounds",
                    "actual_assists",
                    "actual_threes",
                    "actual_minutes",
                ]:
                    if col in review.columns:
                        review[col] = pd.to_numeric(review[col], errors="coerce")

                if "blended_projection" not in review.columns and "projected_dk_points" in review.columns:
                    review["blended_projection"] = review["projected_dk_points"]

                review["blend_error"] = review["actual_dk_points"] - review["blended_projection"]
                review["our_error"] = review["actual_dk_points"] - review.get("our_dk_projection")
                review["vegas_error"] = review["actual_dk_points"] - review.get("vegas_dk_projection")
                review["ownership_error"] = review["actual_ownership"] - review.get("projected_ownership")

                matched_actual = int(review["actual_dk_points"].notna().sum()) if "actual_dk_points" in review.columns else 0
                mae_blend = float(review["blend_error"].abs().mean()) if matched_actual else 0.0
                mae_our = float(review["our_error"].abs().mean()) if matched_actual else 0.0
                mae_vegas = float(review["vegas_error"].abs().mean()) if matched_actual else 0.0
                own_rows = int(review["actual_ownership"].notna().sum()) if "actual_ownership" in review.columns else 0
                own_mae = float(review["ownership_error"].abs().mean()) if own_rows else 0.0

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Projection Rows", int(len(review)))
                m2.metric("Actual Matched", matched_actual)
                m3.metric("Blend MAE", f"{mae_blend:.2f}")
                m4.metric("Our MAE", f"{mae_our:.2f}")
                m5.metric("Vegas MAE", f"{mae_vegas:.2f}")
                if own_rows:
                    st.metric("Ownership MAE", f"{own_mae:.2f}")
                else:
                    st.caption("Ownership MAE unavailable: upload ownership CSV for this date.")

                cols = [
                    "ID",
                    "Name + ID",
                    "Name",
                    "TeamAbbrev",
                    "Position",
                    "Salary",
                    "blended_projection",
                    "our_dk_projection",
                    "vegas_dk_projection",
                    "vegas_markets_found",
                    "vegas_blend_weight",
                    "actual_dk_points",
                    "blend_error",
                    "our_error",
                    "vegas_error",
                    "projected_ownership",
                    "actual_ownership",
                    "ownership_error",
                    "actual_minutes",
                    "actual_points",
                    "actual_rebounds",
                    "actual_assists",
                    "actual_threes",
                ]
                show_cols = [c for c in cols if c in review.columns]
                view_df = review[show_cols].copy()
                sort_cols = [c for c in ["actual_dk_points", "blended_projection"] if c in show_cols]
                if sort_cols:
                    view_df = view_df.sort_values(by=sort_cols, ascending=False)
                review_role_filter = st.selectbox(
                    "Role Filter",
                    options=ROLE_FILTER_OPTIONS,
                    index=0,
                    key="projection_review_role_filter",
                    help="Filter the table by DraftKings role.",
                )
                view_df_filtered = _filter_frame_by_role(
                    view_df,
                    selected_role=review_role_filter,
                    position_col="Position",
                )
                st.caption(f"Showing `{len(view_df_filtered)}` of `{len(view_df)}` rows.")
                st.dataframe(view_df_filtered, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download Projection Review CSV",
                    data=view_df_filtered.to_csv(index=False),
                    file_name=f"projection_review_{review_date.isoformat()}.csv",
                    mime="text/csv",
                    key="download_projection_review_csv",
                )
        except Exception as exc:
            st.exception(exc)

with tab_tournament_review:
    st.subheader("Tournament Review")
    st.caption(
        "Upload contest standings to analyze field construction (stacks, salary left, ownership) "
        "and compare against our lineups and projection assumptions."
    )
    tr_date = st.date_input("Tournament Date", value=lineup_slate_date, key="tournament_review_date")
    tr_contest_id = st.text_input("Contest ID", value="contest", key="tournament_review_contest_id")
    tr_upload = st.file_uploader(
        "Upload Contest Standings CSV",
        type=["csv"],
        key="tournament_standings_upload",
    )
    t1, t2 = st.columns(2)
    tr_save_clicked = t1.button("Save Contest CSV to GCS", key="save_tournament_csv")
    tr_refresh_clicked = t2.button("Refresh Tournament Review", key="refresh_tournament_review")
    if tr_refresh_clicked:
        load_contest_standings_frame.clear()
        load_actual_results_frame_for_date.clear()
        st.session_state.pop("cbb_phantom_review_df", None)
        st.session_state.pop("cbb_phantom_summary_df", None)
        st.session_state.pop("cbb_phantom_review_meta", None)
        st.session_state.pop("cbb_ai_review_packet", None)
        st.session_state.pop("cbb_ai_review_prompt_user", None)
        st.session_state.pop("cbb_ai_review_prompt_system", None)

    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to run tournament review.")
    else:
        try:
            if tr_save_clicked:
                if tr_upload is None:
                    st.error("Choose a contest standings CSV before saving.")
                else:
                    csv_text = tr_upload.getvalue().decode("utf-8-sig")
                    client = build_storage_client(
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                        project=gcp_project or None,
                    )
                    store = CbbGcsStore(bucket_name=bucket_name, client=client)
                    blob_name = _write_contest_standings_csv(store, tr_date, tr_contest_id, csv_text)
                    load_contest_standings_frame.clear()
                    st.success(f"Saved contest standings to `{blob_name}`")

            if tr_upload is not None:
                standings_df = pd.read_csv(io.StringIO(tr_upload.getvalue().decode("utf-8-sig")))
            else:
                standings_df = load_contest_standings_frame(
                    bucket_name=bucket_name,
                    selected_date=tr_date,
                    contest_id=tr_contest_id,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )

            actual_probe_df = load_actual_results_frame_for_date(
                bucket_name=bucket_name,
                selected_date=tr_date,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )

            field_entries_df = pd.DataFrame()
            entries_df = pd.DataFrame()
            exposure_df = pd.DataFrame()
            proj_compare_df = pd.DataFrame()
            adjust_df = pd.DataFrame()
            st.session_state["cbb_tr_entries_df"] = entries_df.copy()
            st.session_state["cbb_tr_exposure_df"] = exposure_df.copy()
            st.session_state["cbb_tr_projection_compare_df"] = proj_compare_df.copy()
            st.session_state["cbb_tr_adjust_df"] = adjust_df.copy()
            if standings_df.empty:
                st.warning("No contest standings loaded. Upload a CSV or save/load one from GCS.")
            else:
                normalized_standings = normalize_contest_standings_frame(standings_df)
                slate_df = load_dk_slate_frame_for_date(
                    bucket_name=bucket_name,
                    selected_date=tr_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )

                projections_df = load_projection_snapshot_frame(
                    bucket_name=bucket_name,
                    selected_date=tr_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                projections_enriched_from_pool = False
                need_pool_fallback = projections_df.empty
                if not need_pool_fallback and not projections_df.empty:
                    mins_last7 = (
                        pd.to_numeric(projections_df.get("our_minutes_last7"), errors="coerce")
                        if "our_minutes_last7" in projections_df.columns
                        else pd.Series(dtype=float)
                    )
                    mins_avg = (
                        pd.to_numeric(projections_df.get("our_minutes_avg"), errors="coerce")
                        if "our_minutes_avg" in projections_df.columns
                        else pd.Series(dtype=float)
                    )
                    need_pool_fallback = not bool(mins_last7.notna().any() or mins_avg.notna().any())

                if need_pool_fallback and not slate_df.empty:
                    fallback_pool, _, _, _, _ = build_optimizer_pool_for_date(
                        bucket_name=bucket_name,
                        slate_date=tr_date,
                        bookmaker=(default_bookmakers_filter.strip() or None),
                        gcp_project=gcp_project or None,
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                    )
                    if projections_df.empty:
                        projections_df = fallback_pool
                        projections_enriched_from_pool = not projections_df.empty
                    elif not fallback_pool.empty:
                        projections_df = projections_df.copy()
                        fallback_pool = fallback_pool.copy()
                        projections_df["ID"] = projections_df.get("ID", "").astype(str).str.strip()
                        projections_df["Name"] = projections_df.get("Name", "").astype(str).str.strip()
                        projections_df["_name_norm"] = projections_df["Name"].map(
                            lambda x: re.sub(r"[^a-z0-9]", "", str(x or "").strip().lower())
                        )
                        fallback_pool["ID"] = fallback_pool.get("ID", "").astype(str).str.strip()
                        fallback_pool["Name"] = fallback_pool.get("Name", "").astype(str).str.strip()
                        fallback_pool["_name_norm"] = fallback_pool["Name"].map(
                            lambda x: re.sub(r"[^a-z0-9]", "", str(x or "").strip().lower())
                        )
                        enrich_cols = [
                            "our_minutes_avg",
                            "our_minutes_last7",
                            "our_usage_proxy",
                            "our_dk_projection",
                            "blended_projection",
                            "vegas_dk_projection",
                        ]
                        id_cols = ["ID"] + [c for c in enrich_cols if c in fallback_pool.columns]
                        name_cols = ["_name_norm"] + [c for c in enrich_cols if c in fallback_pool.columns]
                        fb_id = fallback_pool[id_cols].drop_duplicates(subset=["ID"])
                        fb_name = fallback_pool[name_cols].drop_duplicates(subset=["_name_norm"])
                        projections_df = projections_df.merge(fb_id, on="ID", how="left", suffixes=("", "_fb_id"))
                        projections_df = projections_df.merge(fb_name, on="_name_norm", how="left", suffixes=("", "_fb_name"))
                        for col in enrich_cols:
                            if col not in projections_df.columns:
                                projections_df[col] = pd.NA
                            base_series = pd.to_numeric(projections_df.get(col), errors="coerce")
                            id_series = pd.to_numeric(projections_df.get(f"{col}_fb_id"), errors="coerce")
                            name_series = pd.to_numeric(projections_df.get(f"{col}_fb_name"), errors="coerce")
                            projections_df[col] = base_series.where(base_series.notna(), id_series)
                            projections_df[col] = pd.to_numeric(projections_df[col], errors="coerce").where(
                                pd.to_numeric(projections_df[col], errors="coerce").notna(),
                                name_series,
                            )
                        drop_cols = ["_name_norm"]
                        for col in enrich_cols:
                            drop_cols.append(f"{col}_fb_id")
                            drop_cols.append(f"{col}_fb_name")
                        projections_df = projections_df.drop(columns=drop_cols, errors="ignore")
                        projections_enriched_from_pool = True

                entries_df, expanded_df = build_field_entries_and_players(normalized_standings, slate_df)
                if entries_df.empty:
                    st.warning("Could not parse lineup strings from this standings file.")
                else:
                    if projections_enriched_from_pool:
                        st.caption(
                            "Projection snapshot had missing minutes/projection fields. "
                            "Filled from current Slate + Vegas pool for this date."
                        )
                    entries_df = build_entry_actual_points_comparison(
                        entry_summary_df=entries_df,
                        expanded_players_df=expanded_df,
                        actual_results_df=actual_probe_df,
                    )
                    field_entries_df = entries_df.copy()
                    actual_own_df = extract_actual_ownership_from_standings(normalized_standings)
                    exposure_df = build_player_exposure_comparison(
                        expanded_players_df=expanded_df,
                        entry_count=int(len(entries_df)),
                        projection_df=projections_df,
                        actual_ownership_df=actual_own_df,
                        actual_results_df=actual_probe_df,
                    )
                    user_summary_df = build_user_strategy_summary(entries_df)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Field Entries", int(len(entries_df)))
                    m2.metric("Avg Salary Left", f"{float(entries_df['salary_left'].mean()):.0f}")
                    m3.metric("Avg Max Team Stack", f"{float(entries_df['max_team_stack'].mean()):.2f}")
                    m4.metric("Avg Max Game Stack", f"{float(entries_df['max_game_stack'].mean()):.2f}")
                    top10 = entries_df.nsmallest(10, "Rank")
                    m5.metric("Top-10 Avg Salary Left", f"{float(top10['salary_left'].mean()):.0f}")
                    if "computed_actual_points" in entries_df.columns:
                        st.caption(
                            "`Points` and `Rank` below are recomputed from current `cbb/players` actual results. "
                            "Uploaded contest values are kept in `points_from_file` and `rank_from_file` for reference."
                        )

                    our_generated = st.session_state.get("cbb_generated_lineups", [])
                    if our_generated:
                        our_df = summarize_generated_lineups(our_generated)
                        if not our_df.empty:
                            st.subheader("Our Generated Lineups vs Field")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Our Lineups", int(len(our_df)))
                            c2.metric("Our Avg Salary Left", f"{float(our_df['salary_left'].mean()):.0f}")
                            c3.metric("Our Avg Max Team Stack", f"{float(our_df['max_team_stack'].mean()):.2f}")
                            c4.metric("Our Avg Max Game Stack", f"{float(our_df['max_game_stack'].mean()):.2f}")

                    st.subheader("Field Lineup Construction")
                    st.caption("`Rank` and `Points` use computed final results.")
                    show_entry_cols = [
                        "Rank",
                        "rank_from_computed_points",
                        "rank_from_file",
                        "rank_from_points",
                        "EntryId",
                        "EntryName",
                        "Points",
                        "points_from_file",
                        "computed_actual_points",
                        "computed_minus_file_points",
                        "computed_players_matched",
                        "computed_coverage_pct",
                        "parsed_players",
                        "mapped_players",
                        "salary_used",
                        "salary_left",
                        "unique_teams",
                        "unique_games",
                        "max_team_stack",
                        "max_game_stack",
                    ]
                    entry_cols = [c for c in show_entry_cols if c in entries_df.columns]
                    st.dataframe(entries_df[entry_cols], hide_index=True, use_container_width=True)

                    st.subheader("Player Exposure vs Ownership")
                    if exposure_df.empty:
                        st.info("No player exposure table available.")
                    else:
                        exp_cols = [
                            "Name",
                            "TeamAbbrev",
                            "final_dk_points",
                            "high_points_low_own_flag",
                            "appearances",
                            "field_ownership_pct",
                            "projected_ownership",
                            "ownership_diff_vs_proj",
                            "actual_ownership_from_file",
                            "blended_projection",
                            "our_dk_projection",
                            "vegas_dk_projection",
                        ]
                        use_cols = [c for c in exp_cols if c in exposure_df.columns]
                        st.dataframe(exposure_df[use_cols], hide_index=True, use_container_width=True)

                    st.subheader("Projection vs Actual (Tournament Slate)")
                    proj_compare_df = build_projection_actual_comparison(
                        projection_df=projections_df,
                        actual_results_df=actual_probe_df,
                    )
                    if proj_compare_df.empty:
                        st.info("Projection comparison unavailable (need both projection snapshot and final actual stats).")
                    else:
                        matched_rows = int(pd.to_numeric(proj_compare_df["actual_dk_points"], errors="coerce").notna().sum())
                        blend_mae = float(pd.to_numeric(proj_compare_df["blend_error"], errors="coerce").abs().mean())
                        our_mae = float(pd.to_numeric(proj_compare_df["our_error"], errors="coerce").abs().mean())
                        vegas_mae = float(pd.to_numeric(proj_compare_df["vegas_error"], errors="coerce").abs().mean())
                        mins_mae_avg = float(pd.to_numeric(proj_compare_df["minutes_error_avg"], errors="coerce").abs().mean())
                        mins_mae_last7 = float(pd.to_numeric(proj_compare_df["minutes_error_last7"], errors="coerce").abs().mean())
                        pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
                        pm1.metric("Matched Players", matched_rows)
                        pm2.metric("Blend MAE", f"{blend_mae:.2f}")
                        pm3.metric("Our MAE", f"{our_mae:.2f}")
                        pm4.metric("Vegas MAE", f"{vegas_mae:.2f}")
                        pm5.metric("Minutes MAE (Season Avg)", f"{mins_mae_avg:.2f}")
                        pm6.metric("Minutes MAE (Last 7)", f"{mins_mae_last7:.2f}")

                        tr_role_filter = st.selectbox(
                            "Projection Comparison Role Filter",
                            options=ROLE_FILTER_OPTIONS,
                            index=0,
                            key="tournament_projection_role_filter",
                            help="Filter projection comparison rows by DraftKings role.",
                        )
                        proj_compare_view = _filter_frame_by_role(
                            proj_compare_df,
                            selected_role=tr_role_filter,
                            position_col="Position",
                        )
                        st.caption(
                            f"Showing `{len(proj_compare_view)}` of `{len(proj_compare_df)}` players. "
                            "Use multipliers below to tune future projection inputs."
                        )
                        compare_cols = [
                            "ID",
                            "Name + ID",
                            "Name",
                            "TeamAbbrev",
                            "Position",
                            "Salary",
                            "blended_projection",
                            "our_dk_projection",
                            "vegas_dk_projection",
                            "actual_dk_points",
                            "blend_error",
                            "our_error",
                            "vegas_error",
                            "our_minutes_avg",
                            "our_minutes_last7",
                            "actual_minutes",
                            "minutes_error_avg",
                            "minutes_error_last7",
                            "our_multiplier",
                            "minutes_multiplier_avg",
                            "minutes_multiplier_last7",
                        ]
                        compare_use_cols = [c for c in compare_cols if c in proj_compare_view.columns]
                        st.dataframe(proj_compare_view[compare_use_cols], hide_index=True, use_container_width=True)

                        adjust_df = build_projection_adjustment_factors(proj_compare_df)
                        st.caption(
                            "Example usage: `adjusted_our_dk = our_dk_projection * our_points_multiplier`, "
                            "`adjusted_minutes = our_minutes_last7 * minutes_multiplier_last7`."
                        )
                        st.dataframe(adjust_df, hide_index=True, use_container_width=True)
                        recommended = {str(r.get("segment")): r for r in adjust_df.to_dict(orient="records")}
                        rec_all = float((recommended.get("All") or {}).get("our_points_multiplier") or 1.0)
                        rec_g = float((recommended.get("Guard (G)") or {}).get("our_points_multiplier") or rec_all or 1.0)
                        rec_f = float((recommended.get("Forward (F)") or {}).get("our_points_multiplier") or rec_all or 1.0)
                        rec_mins = float((recommended.get("All") or {}).get("minutes_multiplier_last7") or 1.0)

                        st.caption("Adjustment Sandbox (preview only)")
                        a1, a2, a3, a4 = st.columns(4)
                        adj_mult_all = float(
                            a1.number_input(
                                "Our Pts Mult (All)",
                                min_value=0.25,
                                max_value=2.50,
                                value=max(0.25, min(2.50, rec_all)),
                                step=0.01,
                                key="tournament_adj_mult_all",
                            )
                        )
                        adj_mult_g = float(
                            a2.number_input(
                                "Our Pts Mult (G)",
                                min_value=0.25,
                                max_value=2.50,
                                value=max(0.25, min(2.50, rec_g)),
                                step=0.01,
                                key="tournament_adj_mult_g",
                            )
                        )
                        adj_mult_f = float(
                            a3.number_input(
                                "Our Pts Mult (F)",
                                min_value=0.25,
                                max_value=2.50,
                                value=max(0.25, min(2.50, rec_f)),
                                step=0.01,
                                key="tournament_adj_mult_f",
                            )
                        )
                        adj_mins_last7 = float(
                            a4.number_input(
                                "Minutes Mult (Last 7)",
                                min_value=0.25,
                                max_value=2.50,
                                value=max(0.25, min(2.50, rec_mins)),
                                step=0.01,
                                key="tournament_adj_mult_minutes_last7",
                            )
                        )

                        adjusted_df = proj_compare_df.copy()
                        adjusted_df["our_dk_projection_adjusted"] = pd.to_numeric(
                            adjusted_df.get("our_dk_projection"), errors="coerce"
                        )
                        pos_series = adjusted_df.get("Position", pd.Series(dtype=str)).astype(str).str.upper()
                        point_mult_series = pd.Series(adj_mult_all, index=adjusted_df.index, dtype="float64")
                        point_mult_series.loc[pos_series.str.startswith("G")] = adj_mult_g
                        point_mult_series.loc[pos_series.str.startswith("F")] = adj_mult_f
                        adjusted_df["our_dk_projection_adjusted"] = (
                            adjusted_df["our_dk_projection_adjusted"] * point_mult_series
                        )
                        adjusted_df["our_adjusted_error"] = (
                            pd.to_numeric(adjusted_df.get("actual_dk_points"), errors="coerce")
                            - pd.to_numeric(adjusted_df.get("our_dk_projection_adjusted"), errors="coerce")
                        )
                        adjusted_df["our_minutes_adjusted_last7"] = (
                            pd.to_numeric(adjusted_df.get("our_minutes_last7"), errors="coerce") * adj_mins_last7
                        )
                        adjusted_df["minutes_adjusted_error_last7"] = (
                            pd.to_numeric(adjusted_df.get("actual_minutes"), errors="coerce")
                            - pd.to_numeric(adjusted_df.get("our_minutes_adjusted_last7"), errors="coerce")
                        )

                        am1, am2 = st.columns(2)
                        am1.metric(
                            "Adjusted Our MAE",
                            f"{float(pd.to_numeric(adjusted_df['our_adjusted_error'], errors='coerce').abs().mean()):.2f}",
                        )
                        am2.metric(
                            "Adjusted Minutes MAE (Last 7)",
                            f"{float(pd.to_numeric(adjusted_df['minutes_adjusted_error_last7'], errors='coerce').abs().mean()):.2f}",
                        )
                        adjusted_cols = [
                            "ID",
                            "Name + ID",
                            "Name",
                            "TeamAbbrev",
                            "Position",
                            "our_dk_projection",
                            "our_dk_projection_adjusted",
                            "actual_dk_points",
                            "our_error",
                            "our_adjusted_error",
                            "our_minutes_last7",
                            "our_minutes_adjusted_last7",
                            "actual_minutes",
                            "minutes_error_last7",
                            "minutes_adjusted_error_last7",
                        ]
                        adjusted_use_cols = [c for c in adjusted_cols if c in adjusted_df.columns]
                        adjusted_view = _filter_frame_by_role(
                            adjusted_df[adjusted_use_cols],
                            selected_role=tr_role_filter,
                            position_col="Position",
                        )
                        st.dataframe(adjusted_view, hide_index=True, use_container_width=True)
                        st.download_button(
                            "Download Tournament Projection Comparison CSV",
                            data=proj_compare_view.to_csv(index=False),
                            file_name=f"tournament_projection_comparison_{tr_date.isoformat()}_{tr_contest_id}.csv",
                            mime="text/csv",
                            key="download_tournament_projection_comparison_csv",
                        )
                        st.download_button(
                            "Download Tournament Projection Adjustment Factors CSV",
                            data=adjust_df.to_csv(index=False),
                            file_name=f"tournament_projection_adjustments_{tr_date.isoformat()}_{tr_contest_id}.csv",
                            mime="text/csv",
                            key="download_tournament_projection_adjustment_factors_csv",
                        )
                        st.download_button(
                            "Download Tournament Adjusted Projection Preview CSV",
                            data=adjusted_view.to_csv(index=False),
                            file_name=f"tournament_adjusted_projection_preview_{tr_date.isoformat()}_{tr_contest_id}.csv",
                            mime="text/csv",
                            key="download_tournament_adjusted_projection_preview_csv",
                        )

                    st.subheader("User Strategy Summary")
                    if user_summary_df.empty:
                        st.info("No user-level summary available.")
                    else:
                        st.dataframe(user_summary_df, hide_index=True, use_container_width=True)

                    st.download_button(
                        "Download Tournament Entry Construction CSV",
                        data=entries_df.to_csv(index=False),
                        file_name=f"tournament_entries_{tr_date.isoformat()}_{tr_contest_id}.csv",
                        mime="text/csv",
                        key="download_tournament_entries",
                    )
                    st.download_button(
                        "Download Tournament Exposure CSV",
                        data=exposure_df.to_csv(index=False),
                        file_name=f"tournament_exposure_{tr_date.isoformat()}_{tr_contest_id}.csv",
                        mime="text/csv",
                        key="download_tournament_exposure",
                    )

                    st.session_state["cbb_tr_entries_df"] = entries_df.copy()
                    st.session_state["cbb_tr_exposure_df"] = exposure_df.copy()
                    st.session_state["cbb_tr_projection_compare_df"] = proj_compare_df.copy()
                    st.session_state["cbb_tr_adjust_df"] = adjust_df.copy()

            st.markdown("---")
            st.subheader("Phantom Entries (Saved Runs)")
            st.caption(
                "Score saved generated lineups against actual player results for this date. "
                "Optional field comparison uses the loaded contest standings."
            )
            ap1, ap2, ap3 = st.columns(3)
            ap1.metric("Actual Player Rows", int(len(actual_probe_df)))
            ap2.metric(
                "Actual Unique Players",
                int(actual_probe_df["ID"].astype(str).nunique()) if not actual_probe_df.empty and "ID" in actual_probe_df.columns else 0,
            )
            ap3.metric(
                "Actual Avg DK Pts",
                (
                    f"{float(pd.to_numeric(actual_probe_df['actual_dk_points'], errors='coerce').mean()):.2f}"
                    if not actual_probe_df.empty and "actual_dk_points" in actual_probe_df.columns
                    else "0.00"
                ),
            )
            if actual_probe_df.empty:
                st.info(
                    "No actual results currently loaded for this date. "
                    "Expected blob: "
                    f"`cbb/players/{tr_date.isoformat()}_players.csv`"
                )

            p1, p2 = st.columns(2)
            phantom_refresh = p1.button("Refresh Saved Runs for Phantom Review", key="refresh_phantom_runs")
            save_phantom_outputs = p2.checkbox(
                "Save phantom outputs to GCS",
                value=True,
                key="save_phantom_outputs_to_gcs",
            )
            if phantom_refresh:
                load_saved_lineup_run_manifests.clear()
                load_saved_lineup_version_payload.clear()
                load_actual_results_frame_for_date.clear()

            phantom_manifests = load_saved_lineup_run_manifests(
                bucket_name=bucket_name,
                selected_date=tr_date,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if not phantom_manifests:
                st.info("No saved lineup runs found for this date.")
            else:
                run_option_map: dict[str, dict[str, Any]] = {}
                run_options: list[str] = []
                for manifest in phantom_manifests:
                    run_id = str(manifest.get("run_id") or "")
                    generated_at = str(manifest.get("generated_at_utc") or "")
                    run_mode = str(manifest.get("run_mode") or "single")
                    label = f"{generated_at} | {run_id} | {run_mode}"
                    run_options.append(label)
                    run_option_map[label] = manifest

                selected_run_label = st.selectbox(
                    "Saved Run for Phantom Review",
                    options=run_options,
                    index=0,
                    key="tournament_phantom_run_picker",
                )
                selected_manifest = run_option_map[selected_run_label]
                selected_run_id = str(selected_manifest.get("run_id") or "")
                version_meta_list = selected_manifest.get("versions") or []
                version_meta_map = {str(v.get("version_key") or ""): v for v in version_meta_list}
                available_version_keys = [k for k in version_meta_map.keys() if k]
                if not available_version_keys:
                    st.info("Selected run has no saved versions.")
                else:
                    selected_versions = st.multiselect(
                        "Versions to Score",
                        options=available_version_keys,
                        default=available_version_keys,
                        format_func=lambda k: (
                            f"{version_meta_map[k].get('version_label', k)} "
                            f"[{k}]"
                        ),
                        key="tournament_phantom_versions",
                    )
                    compare_to_field = st.checkbox(
                        "Compare phantom scores to field standings",
                        value=not field_entries_df.empty,
                        disabled=field_entries_df.empty,
                        key="tournament_phantom_compare_to_field",
                    )
                    if field_entries_df.empty:
                        st.caption("Field comparison disabled until a contest standings CSV is loaded.")

                    run_phantom_clicked = st.button("Run Phantom Review", key="run_phantom_review")
                    if run_phantom_clicked:
                        if not selected_versions:
                            st.error("Choose at least one run version to score.")
                        else:
                            actual_df = actual_probe_df.copy()
                            if actual_df.empty:
                                st.error(
                                    "No actual player stats found for this date. "
                                    "Run game import/backfill so `cbb/players/<date>_players.csv` exists."
                                )
                            else:
                                phantom_parts: list[pd.DataFrame] = []
                                skipped_versions: list[str] = []
                                for version_key in selected_versions:
                                    payload = load_saved_lineup_version_payload(
                                        bucket_name=bucket_name,
                                        selected_date=tr_date,
                                        run_id=selected_run_id,
                                        version_key=version_key,
                                        gcp_project=gcp_project or None,
                                        service_account_json=cred_json,
                                        service_account_json_b64=cred_json_b64,
                                    )
                                    if not isinstance(payload, dict):
                                        skipped_versions.append(version_key)
                                        continue
                                    lineups = payload.get("lineups") or []
                                    version_label = str(
                                        payload.get("version_label")
                                        or version_meta_map.get(version_key, {}).get("version_label")
                                        or version_key
                                    )
                                    scored_df = score_generated_lineups_against_actuals(
                                        generated_lineups=lineups,
                                        actual_results_df=actual_df,
                                        version_key=version_key,
                                        version_label=version_label,
                                    )
                                    if scored_df.empty:
                                        skipped_versions.append(version_key)
                                        continue
                                    phantom_parts.append(scored_df)

                                if not phantom_parts:
                                    st.error("No phantom lineups were scored from the selected versions.")
                                else:
                                    phantom_df = pd.concat(phantom_parts, ignore_index=True)
                                    field_for_compare = field_entries_df if compare_to_field else pd.DataFrame()
                                    phantom_df = compare_phantom_entries_to_field(phantom_df, field_for_compare)
                                    phantom_summary_df = summarize_phantom_entries(phantom_df)

                                    st.session_state["cbb_phantom_review_df"] = phantom_df
                                    st.session_state["cbb_phantom_summary_df"] = phantom_summary_df
                                    st.session_state["cbb_phantom_review_meta"] = {
                                        "date": tr_date.isoformat(),
                                        "run_id": selected_run_id,
                                        "contest_id": tr_contest_id,
                                        "versions": list(selected_versions),
                                        "compared_to_field": bool(compare_to_field and not field_entries_df.empty),
                                    }
                                    if skipped_versions:
                                        st.warning(
                                            "Skipped versions with no saved payload/lineups: "
                                            + ", ".join(sorted(set(skipped_versions)))
                                        )

                                    if save_phantom_outputs:
                                        client = build_storage_client(
                                            service_account_json=cred_json,
                                            service_account_json_b64=cred_json_b64,
                                            project=gcp_project or None,
                                        )
                                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                                        version_blobs: list[dict[str, Any]] = []
                                        for version_key in selected_versions:
                                            version_slice = phantom_df.loc[phantom_df["version_key"] == version_key].copy()
                                            if version_slice.empty:
                                                continue
                                            blob_name = store.write_phantom_review_csv(
                                                tr_date,
                                                selected_run_id,
                                                version_key,
                                                version_slice.to_csv(index=False),
                                            )
                                            version_blobs.append(
                                                {
                                                    "version_key": version_key,
                                                    "row_count": int(len(version_slice)),
                                                    "blob_name": blob_name,
                                                }
                                            )
                                        summary_payload = {
                                            "game_date": tr_date.isoformat(),
                                            "run_id": selected_run_id,
                                            "contest_id": tr_contest_id,
                                            "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                                            "compared_to_field": bool(compare_to_field and not field_entries_df.empty),
                                            "field_entry_count": int(len(field_entries_df)),
                                            "lineup_count": int(len(phantom_df)),
                                            "version_count": int(phantom_df["version_key"].nunique()),
                                            "version_csv_blobs": version_blobs,
                                            "summary_rows": _json_safe(phantom_summary_df.to_dict(orient="records")),
                                        }
                                        summary_blob = store.write_phantom_review_summary_json(
                                            tr_date,
                                            selected_run_id,
                                            summary_payload,
                                        )
                                        st.success(
                                            f"Saved phantom review summary to `{summary_blob}` "
                                            f"with {len(version_blobs)} version CSV file(s)."
                                        )

            phantom_df_state = st.session_state.get("cbb_phantom_review_df")
            phantom_summary_state = st.session_state.get("cbb_phantom_summary_df")
            phantom_meta = st.session_state.get("cbb_phantom_review_meta") or {}
            if isinstance(phantom_df_state, pd.DataFrame) and not phantom_df_state.empty:
                if phantom_meta:
                    st.caption(
                        f"Phantom run: `{phantom_meta.get('run_id', '')}` | Date: `{phantom_meta.get('date', '')}` | "
                        f"Compared to field: `{phantom_meta.get('compared_to_field', False)}`"
                    )
                if isinstance(phantom_summary_state, pd.DataFrame) and not phantom_summary_state.empty:
                    st.subheader("Phantom Summary")
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Versions", int(phantom_summary_state["version_key"].nunique()))
                    s2.metric("Lineups Scored", int(len(phantom_df_state)))
                    s3.metric("Best Actual", f"{float(phantom_df_state['actual_points'].max()):.2f}")
                    coverage_mean = pd.to_numeric(phantom_df_state["coverage_pct"], errors="coerce").mean()
                    s4.metric("Avg Coverage %", f"{float(coverage_mean):.1f}")
                    st.dataframe(phantom_summary_state, hide_index=True, use_container_width=True)
                    st.download_button(
                        "Download Phantom Summary CSV",
                        data=phantom_summary_state.to_csv(index=False),
                        file_name=f"phantom_summary_{tr_date.isoformat()}_{phantom_meta.get('run_id', 'run')}.csv",
                        mime="text/csv",
                        key="download_phantom_summary_csv",
                    )

                st.subheader("Phantom Lineup Results")
                display_cols = [
                    "version_key",
                    "version_label",
                    "lineup_number",
                    "lineup_strategy",
                    "pair_id",
                    "pair_role",
                    "cluster_id",
                    "cluster_script",
                    "anchor_game_key",
                    "seed_lineup_id",
                    "mutation_type",
                    "stack_signature",
                    "salary_texture_bucket",
                    "salary_used",
                    "salary_left",
                    "projected_points",
                    "actual_points",
                    "actual_minus_projected",
                    "would_rank",
                    "would_beat_pct",
                    "coverage_pct",
                    "missing_players",
                    "missing_names",
                ]
                use_cols = [c for c in display_cols if c in phantom_df_state.columns]
                phantom_view = phantom_df_state[use_cols].copy()
                sort_cols = [c for c in ["actual_points", "would_beat_pct"] if c in phantom_view.columns]
                if sort_cols:
                    phantom_view = phantom_view.sort_values(sort_cols, ascending=False)
                st.dataframe(phantom_view, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download Phantom Lineups CSV",
                    data=phantom_df_state.to_csv(index=False),
                    file_name=f"phantom_lineups_{tr_date.isoformat()}_{phantom_meta.get('run_id', 'run')}.csv",
                    mime="text/csv",
                    key="download_phantom_lineups_csv",
                )
        except Exception as exc:
            st.exception(exc)

with tab_agentic_review:
    st.subheader("Agentic Review")
    st.caption(
        "Global AI diagnostics across slates for projection quality, ownership quality, "
        "and lineup-process changes. Optional single-slate drilldown is included below."
    )
    ar_date = st.date_input("Agentic Review Date", value=lineup_slate_date, key="agentic_review_date")
    ar_contest_default = str(st.session_state.get("tournament_review_contest_id", "contest"))
    ar_contest_id = st.text_input("Agentic Contest ID", value=ar_contest_default, key="agentic_review_contest_id")

    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to run agentic review.")
    else:
        try:
            openai_key = (os.getenv("OPENAI_API_KEY", "").strip() or (_secret("openai_api_key") or "").strip())
            st.caption(
                "OpenAI key: "
                + ("loaded from secrets/env (`OPENAI_API_KEY` or `openai_api_key`)" if openai_key else "not set")
            )

            st.subheader("Single-Slate Drilldown (Optional)")
            st.caption(
                "Uses outputs from Tournament Review (entries, exposure, projection comparison, phantom summaries) "
                "to build a deterministic packet and recommendation prompt."
            )

            phantom_df_state = st.session_state.get("cbb_phantom_review_df")
            phantom_summary_state = st.session_state.get("cbb_phantom_summary_df")

            ai_entries_df = st.session_state.get("cbb_tr_entries_df")
            ai_exposure_df = st.session_state.get("cbb_tr_exposure_df")
            ai_proj_compare_df = st.session_state.get("cbb_tr_projection_compare_df")
            ai_adjust_df = st.session_state.get("cbb_tr_adjust_df")
            if not isinstance(ai_entries_df, pd.DataFrame):
                ai_entries_df = pd.DataFrame()
            if not isinstance(ai_exposure_df, pd.DataFrame):
                ai_exposure_df = pd.DataFrame()
            if not isinstance(ai_proj_compare_df, pd.DataFrame):
                ai_proj_compare_df = pd.DataFrame()
            if not isinstance(ai_adjust_df, pd.DataFrame):
                ai_adjust_df = pd.DataFrame()
            ai_phantom_df = phantom_df_state if isinstance(phantom_df_state, pd.DataFrame) else pd.DataFrame()
            ai_phantom_summary_df = (
                phantom_summary_state if isinstance(phantom_summary_state, pd.DataFrame) else pd.DataFrame()
            )

            if ai_entries_df.empty or ai_exposure_df.empty or ai_proj_compare_df.empty:
                st.info(
                    "Run Tournament Review first (with standings + projections + actuals) to populate "
                    "single-slate packet inputs."
                )
            else:
                aic1, aic2 = st.columns([1, 1])
                focus_limit = int(
                    aic1.slider(
                        "Focus Items",
                        min_value=5,
                        max_value=40,
                        value=15,
                        step=1,
                        key="ai_review_focus_limit",
                    )
                )
                rebuild_packet = aic2.button("Build AI Review Packet", key="build_ai_review_packet")

                if rebuild_packet or ("cbb_ai_review_packet" not in st.session_state):
                    ai_packet = build_daily_ai_review_packet(
                        review_date=ar_date.isoformat(),
                        contest_id=ar_contest_id,
                        projection_comparison_df=ai_proj_compare_df,
                        entries_df=ai_entries_df,
                        exposure_df=ai_exposure_df,
                        phantom_summary_df=ai_phantom_summary_df,
                        phantom_lineups_df=ai_phantom_df,
                        adjustment_factors_df=ai_adjust_df,
                        focus_limit=focus_limit,
                    )
                    st.session_state["cbb_ai_review_packet"] = ai_packet
                    st.session_state["cbb_ai_review_prompt_system"] = AI_REVIEW_SYSTEM_PROMPT
                    st.session_state["cbb_ai_review_prompt_user"] = build_ai_review_user_prompt(ai_packet)

                ai_packet_state = st.session_state.get("cbb_ai_review_packet")
                ai_prompt_system = str(st.session_state.get("cbb_ai_review_prompt_system") or AI_REVIEW_SYSTEM_PROMPT)
                ai_prompt_user = str(st.session_state.get("cbb_ai_review_prompt_user") or "")
                if isinstance(ai_packet_state, dict) and ai_packet_state:
                    scorecards = ai_packet_state.get("scorecards") or {}
                    projection_quality = scorecards.get("projection_quality") or {}
                    ownership_quality = scorecards.get("ownership_quality") or {}
                    lineup_quality = scorecards.get("lineup_quality") or {}

                    am1, am2, am3, am4 = st.columns(4)
                    am1.metric(
                        "Projection Rows",
                        _safe_int_value((projection_quality or {}).get("matched_rows"), default=0),
                    )
                    am2.metric(
                        "Blend MAE",
                        f"{_safe_float_value((projection_quality or {}).get('blend_mae'), default=0.0):.2f}",
                    )
                    am3.metric(
                        "Ownership MAE",
                        f"{_safe_float_value((ownership_quality or {}).get('ownership_mae'), default=0.0):.2f}",
                    )
                    am4.metric(
                        "Lineups Scored",
                        _safe_int_value((lineup_quality or {}).get("lineups_scored"), default=0),
                    )

                    packet_json = json.dumps(ai_packet_state, indent=2, ensure_ascii=True)
                    st.download_button(
                        "Download AI Review Packet JSON",
                        data=packet_json,
                        file_name=f"ai_review_packet_{ar_date.isoformat()}_{ar_contest_id}.json",
                        mime="application/json",
                        key="download_ai_review_packet_json",
                    )
                    st.download_button(
                        "Download AI System Prompt TXT",
                        data=ai_prompt_system,
                        file_name=f"ai_system_prompt_{ar_date.isoformat()}_{ar_contest_id}.txt",
                        mime="text/plain",
                        key="download_ai_system_prompt_txt",
                    )
                    st.download_button(
                        "Download AI User Prompt TXT",
                        data=ai_prompt_user,
                        file_name=f"ai_user_prompt_{ar_date.isoformat()}_{ar_contest_id}.txt",
                        mime="text/plain",
                        key="download_ai_user_prompt_txt",
                    )
                    with st.expander("Packet Preview"):
                        st.json(ai_packet_state)
                    with st.expander("Prompt Preview"):
                        st.text_area(
                            "User Prompt",
                            value=ai_prompt_user,
                            height=260,
                            key="ai_review_prompt_preview_text",
                        )

                    st.caption("Optional: run GPT directly from this app using your OpenAI key.")
                    ai_model = st.text_input(
                        "OpenAI Model",
                        value=str(st.session_state.get("ai_review_model", "gpt-5-mini")),
                        key="ai_review_model",
                        help="Example: gpt-5-mini, gpt-5, gpt-4.1-mini",
                    ).strip()
                    ai_max_tokens = int(
                        st.number_input(
                            "Max Output Tokens",
                            min_value=200,
                            max_value=8000,
                            value=1800,
                            step=100,
                            key="ai_review_max_output_tokens",
                        )
                    )
                    run_openai_review_clicked = st.button("Run OpenAI Review", key="run_openai_review")
                    if run_openai_review_clicked:
                        if not openai_key:
                            st.error("Set `OPENAI_API_KEY` or Streamlit secret `openai_api_key` first.")
                        else:
                            with st.spinner("Generating AI recommendations..."):
                                try:
                                    ai_text = request_openai_review(
                                        api_key=openai_key,
                                        user_prompt=ai_prompt_user,
                                        system_prompt=ai_prompt_system,
                                        model=(ai_model or "gpt-5-mini"),
                                        max_output_tokens=ai_max_tokens,
                                    )
                                    st.session_state["cbb_ai_review_output"] = ai_text
                                except Exception as exc:
                                    st.exception(exc)

                    ai_review_output = str(st.session_state.get("cbb_ai_review_output") or "").strip()
                    if ai_review_output:
                        st.subheader("AI Recommendations")
                        st.text_area(
                            "Model Output",
                            value=ai_review_output,
                            height=360,
                            key="ai_review_output_preview",
                        )
                        st.download_button(
                            "Download AI Recommendations TXT",
                            data=ai_review_output,
                            file_name=f"ai_recommendations_{ar_date.isoformat()}_{ar_contest_id}.txt",
                            mime="text/plain",
                            key="download_ai_recommendations_txt",
                        )

            st.markdown("---")
            st.subheader("Global Agentic Review (Multi-Slate)")
            st.caption(
                "Build a running review across a lookback window to identify global projection, ownership, "
                "and lineup-process changes."
            )
            g1, g2, g3 = st.columns(3)
            lookback_days = int(
                g1.slider(
                    "Lookback Days",
                    min_value=7,
                    max_value=180,
                    value=30,
                    step=1,
                    key="global_ai_review_lookback_days",
                )
            )
            global_focus_limit = int(
                g2.slider(
                    "Global Focus Players",
                    min_value=5,
                    max_value=60,
                    value=25,
                    step=1,
                    key="global_ai_review_focus_limit",
                )
            )
            use_saved_run_dates = bool(
                g3.checkbox(
                    "Use Saved Run Dates Only",
                    value=True,
                    key="global_ai_use_saved_run_dates",
                    help="If enabled, only dates with saved lineup runs are scanned.",
                )
            )
            build_global_packet_clicked = st.button("Build Global AI Packet", key="build_global_ai_packet")

            if build_global_packet_clicked:
                try:
                    today = date.today()
                    cutoff = today - timedelta(days=max(1, lookback_days))
                    store = None
                    try:
                        client = build_storage_client(
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                            project=gcp_project or None,
                        )
                        store = CbbGcsStore(bucket_name=bucket_name, client=client)
                    except Exception:
                        store = None
                    if use_saved_run_dates:
                        candidate_dates = load_saved_lineup_run_dates(
                            bucket_name=bucket_name,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                    else:
                        candidate_dates = [d for d in iter_dates(cutoff, today)]
                    candidate_dates = [d for d in candidate_dates if isinstance(d, date) and d >= cutoff]
                    candidate_dates = sorted(set(candidate_dates))

                    daily_packets: list[dict[str, Any]] = []
                    scanned_dates = 0
                    used_dates = 0
                    for review_day in candidate_dates:
                        scanned_dates += 1
                        proj_snap_df = load_projection_snapshot_frame(
                            bucket_name=bucket_name,
                            selected_date=review_day,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        actual_day_df = load_actual_results_frame_for_date(
                            bucket_name=bucket_name,
                            selected_date=review_day,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        if proj_snap_df.empty or actual_day_df.empty:
                            continue

                        proj_cmp_day_df = build_projection_actual_comparison(
                            projection_df=proj_snap_df,
                            actual_results_df=actual_day_df,
                        )
                        if proj_cmp_day_df.empty:
                            continue

                        own_day_df = load_ownership_frame_for_date(
                            bucket_name=bucket_name,
                            selected_date=review_day,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        expo_day_df = pd.DataFrame()
                        if not own_day_df.empty:
                            proj_base = proj_snap_df.copy()
                            if "ID" in proj_base.columns:
                                proj_base["ID"] = proj_base["ID"].astype(str).str.strip()
                            if "Name" in proj_base.columns:
                                proj_base["Name"] = proj_base["Name"].astype(str).str.strip()
                            if "TeamAbbrev" in proj_base.columns:
                                proj_base["TeamAbbrev"] = proj_base["TeamAbbrev"].astype(str).str.strip().str.upper()
                            if "ID" in own_day_df.columns:
                                own_day_df["ID"] = own_day_df["ID"].astype(str).str.strip()
                            if "Name" in own_day_df.columns:
                                own_day_df["Name"] = own_day_df["Name"].astype(str).str.strip()

                            if "ID" in proj_base.columns and "ID" in own_day_df.columns:
                                expo_day_df = proj_base.merge(
                                    own_day_df[["ID", "actual_ownership"]],
                                    on="ID",
                                    how="left",
                                )
                            elif "Name" in proj_base.columns and "Name" in own_day_df.columns:
                                expo_day_df = proj_base.merge(
                                    own_day_df[["Name", "actual_ownership"]],
                                    on="Name",
                                    how="left",
                                )
                            else:
                                expo_day_df = proj_base.copy()
                                expo_day_df["actual_ownership"] = pd.NA

                            expo_day_df["projected_ownership"] = pd.to_numeric(
                                expo_day_df.get("projected_ownership"), errors="coerce"
                            )
                            expo_day_df["actual_ownership_from_file"] = pd.to_numeric(
                                expo_day_df.get("actual_ownership"), errors="coerce"
                            )
                            expo_day_df["ownership_diff_vs_proj"] = (
                                expo_day_df["actual_ownership_from_file"] - expo_day_df["projected_ownership"]
                            )
                            expo_day_df["field_ownership_pct"] = expo_day_df["actual_ownership_from_file"]

                        phantom_summary_day_df = pd.DataFrame()
                        if store is not None:
                            try:
                                run_ids = store.list_lineup_run_ids(review_day)
                                for run_id in run_ids:
                                    summary_payload = store.read_phantom_review_summary_json(review_day, run_id)
                                    if not isinstance(summary_payload, dict):
                                        continue
                                    summary_rows = summary_payload.get("summary_rows")
                                    if not isinstance(summary_rows, list) or not summary_rows:
                                        continue
                                    phantom_summary_day_df = pd.DataFrame(summary_rows)
                                    if not phantom_summary_day_df.empty:
                                        break
                            except Exception:
                                phantom_summary_day_df = pd.DataFrame()

                        day_packet = build_daily_ai_review_packet(
                            review_date=review_day.isoformat(),
                            contest_id=f"{ar_contest_id}-global",
                            projection_comparison_df=proj_cmp_day_df,
                            entries_df=pd.DataFrame(),
                            exposure_df=expo_day_df,
                            phantom_summary_df=phantom_summary_day_df,
                            phantom_lineups_df=pd.DataFrame(),
                            adjustment_factors_df=pd.DataFrame(),
                            focus_limit=global_focus_limit,
                        )
                        daily_packets.append(day_packet)
                        used_dates += 1

                    global_packet = build_global_ai_review_packet(
                        daily_packets=daily_packets,
                        focus_limit=global_focus_limit,
                    )
                    global_user_prompt = build_global_ai_review_user_prompt(global_packet)
                    st.session_state["cbb_global_ai_review_packet"] = global_packet
                    st.session_state["cbb_global_ai_review_user_prompt"] = global_user_prompt
                    st.session_state["cbb_global_ai_review_meta"] = {
                        "scanned_dates": scanned_dates,
                        "used_dates": used_dates,
                        "lookback_days": lookback_days,
                        "use_saved_run_dates": use_saved_run_dates,
                    }
                except Exception as exc:
                    st.exception(exc)

            global_packet_state = st.session_state.get("cbb_global_ai_review_packet")
            global_user_prompt = str(st.session_state.get("cbb_global_ai_review_user_prompt") or "").strip()
            global_meta = st.session_state.get("cbb_global_ai_review_meta") or {}
            if isinstance(global_packet_state, dict) and global_packet_state:
                w = global_packet_state.get("window_summary") or {}
                gs = global_packet_state.get("global_scorecards") or {}
                gp = gs.get("projection") or {}
                go = gs.get("ownership") or {}
                gl = gs.get("lineup") or {}
                gm1, gm2, gm3, gm4 = st.columns(4)
                gm1.metric("Slates Used", int(_safe_int_value(w.get("slate_count"), default=0)))
                gm2.metric(
                    "Weighted Blend MAE",
                    f"{_safe_float_value(gp.get('weighted_blend_mae'), default=0.0):.2f}",
                )
                gm3.metric(
                    "Weighted Ownership MAE",
                    f"{_safe_float_value(go.get('weighted_ownership_mae'), default=0.0):.2f}",
                )
                gm4.metric(
                    "Lineups Scored",
                    int(_safe_int_value(gl.get("total_lineups_scored"), default=0)),
                )
                st.caption(
                    "Global packet build summary: "
                    f"scanned_dates={int(_safe_int_value(global_meta.get('scanned_dates'), default=0))}, "
                    f"used_dates={int(_safe_int_value(global_meta.get('used_dates'), default=0))}, "
                    f"lookback_days={int(_safe_int_value(global_meta.get('lookback_days'), default=0))}"
                )

                global_packet_json = json.dumps(global_packet_state, indent=2, ensure_ascii=True)
                st.download_button(
                    "Download Global AI Packet JSON",
                    data=global_packet_json,
                    file_name=f"global_ai_review_packet_{ar_date.isoformat()}_{ar_contest_id}.json",
                    mime="application/json",
                    key="download_global_ai_review_packet_json",
                )
                st.download_button(
                    "Download Global AI User Prompt TXT",
                    data=global_user_prompt,
                    file_name=f"global_ai_user_prompt_{ar_date.isoformat()}_{ar_contest_id}.txt",
                    mime="text/plain",
                    key="download_global_ai_user_prompt_txt",
                )
                with st.expander("Global Packet Preview"):
                    st.json(global_packet_state)
                with st.expander("Global Prompt Preview"):
                    st.text_area(
                        "Global User Prompt",
                        value=global_user_prompt,
                        height=260,
                        key="global_ai_prompt_preview_text",
                    )

                run_global_openai = st.button("Run Global OpenAI Review", key="run_global_openai_review")
                if run_global_openai:
                    if not openai_key:
                        st.error("Set `OPENAI_API_KEY` or Streamlit secret `openai_api_key` first.")
                    else:
                        with st.spinner("Generating global AI recommendations..."):
                            try:
                                global_ai_text = request_openai_review(
                                    api_key=openai_key,
                                    user_prompt=global_user_prompt,
                                    system_prompt=AI_REVIEW_SYSTEM_PROMPT,
                                    model=str(st.session_state.get("ai_review_model", "gpt-5-mini")),
                                    max_output_tokens=int(st.session_state.get("ai_review_max_output_tokens", 1800)),
                                )
                                st.session_state["cbb_global_ai_review_output"] = global_ai_text
                            except Exception as exc:
                                st.exception(exc)
                global_ai_output = str(st.session_state.get("cbb_global_ai_review_output") or "").strip()
                if global_ai_output:
                    st.subheader("Global AI Recommendations")
                    st.text_area(
                        "Global Model Output",
                        value=global_ai_output,
                        height=360,
                        key="global_ai_review_output_preview",
                    )
                    st.download_button(
                        "Download Global AI Recommendations TXT",
                        data=global_ai_output,
                        file_name=f"global_ai_recommendations_{ar_date.isoformat()}_{ar_contest_id}.txt",
                        mime="text/plain",
                        key="download_global_ai_recommendations_txt",
                    )
        except Exception as exc:
            st.exception(exc)

with tab_game:
    st.subheader("Game Odds")
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to load game odds.")
    else:
        try:
            odds_df = load_odds_frame_for_date(
                bucket_name=bucket_name,
                selected_date=game_selected_date,
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

with tab_game:
    st.subheader("Game Slate Agent")
    st.caption(
        "AI scout for game-level DFS context. Combines today's odds, prior-day box score form, "
        "and season Vegas review calibration to rank stack and winner angles."
    )
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to run Game Slate Agent.")
    else:
        try:
            openai_key = (os.getenv("OPENAI_API_KEY", "").strip() or (_secret("openai_api_key") or "").strip())
            st.caption(
                "OpenAI key: "
                + ("loaded from secrets/env (`OPENAI_API_KEY` or `openai_api_key`)" if openai_key else "not set")
            )

            ga1, ga2, ga3, ga4 = st.columns([1, 1, 1, 1])
            game_agent_focus_limit = int(
                ga1.slider(
                    "Focus Games",
                    min_value=3,
                    max_value=20,
                    value=8,
                    step=1,
                    key="game_slate_ai_focus_limit",
                )
            )
            game_agent_model = ga2.text_input(
                "OpenAI Model",
                value=str(st.session_state.get("game_slate_ai_model", "gpt-5-mini")),
                key="game_slate_ai_model",
                help="Example: gpt-5-mini, gpt-5, gpt-4.1-mini",
            ).strip()
            game_agent_max_tokens = int(
                ga3.number_input(
                    "Max Output Tokens",
                    min_value=200,
                    max_value=8000,
                    value=int(st.session_state.get("game_slate_ai_max_output_tokens", 1400)),
                    step=100,
                    key="game_slate_ai_max_output_tokens",
                )
            )
            game_agent_timeout_seconds = int(
                ga4.number_input(
                    "Request Timeout (sec)",
                    min_value=30,
                    max_value=600,
                    value=int(st.session_state.get("game_slate_ai_timeout_seconds", 180)),
                    step=10,
                    key="game_slate_ai_timeout_seconds",
                )
            )

            gb1, gb2 = st.columns([1, 1])
            build_game_slate_packet = gb1.button("Build Game Slate Packet", key="build_game_slate_ai_packet")
            run_game_slate_agent = gb2.button("Run Game Slate Agent", key="run_game_slate_ai_review")

            if build_game_slate_packet:
                odds_agent_df = load_odds_frame_for_date(
                    bucket_name=bucket_name,
                    selected_date=game_selected_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                if odds_agent_df.empty:
                    st.warning("No cached odds found for selected date. Run `Run Odds Import` first.")
                else:
                    pool_bookmakers = _csv_values(game_bookmakers_filter)
                    pool_bookmaker = pool_bookmakers[0] if pool_bookmakers else None
                    pool_df_agent = pd.DataFrame()
                    raw_slate_df_agent = pd.DataFrame()
                    try:
                        pool_df_agent, _, raw_slate_df_agent, _, _ = build_optimizer_pool_for_date(
                            bucket_name=bucket_name,
                            slate_date=game_selected_date,
                            bookmaker=pool_bookmaker,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                    except Exception:
                        pool_df_agent = pd.DataFrame()
                        raw_slate_df_agent = pd.DataFrame()

                    prior_boxscore_date = game_selected_date - timedelta(days=1)
                    prior_boxscore_df = load_actual_results_frame_for_date(
                        bucket_name=bucket_name,
                        selected_date=prior_boxscore_date,
                        gcp_project=gcp_project or None,
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                    )
                    vegas_history_df = load_season_vegas_history_frame(
                        bucket_name=bucket_name,
                        selected_date=game_selected_date,
                        gcp_project=gcp_project or None,
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                    )
                    vegas_review_df = load_vegas_review_frame_for_date(
                        bucket_name=bucket_name,
                        selected_date=game_selected_date,
                        gcp_project=gcp_project or None,
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                    )
                    game_packet_kwargs: dict[str, Any] = {
                        "review_date": game_selected_date.isoformat(),
                        "odds_df": odds_agent_df,
                        "prior_boxscore_df": prior_boxscore_df,
                        "vegas_history_df": vegas_history_df,
                        "vegas_review_df": vegas_review_df,
                        "focus_limit": game_agent_focus_limit,
                    }
                    if "player_pool_df" in inspect.signature(build_game_slate_ai_review_packet).parameters:
                        game_packet_kwargs["player_pool_df"] = pool_df_agent
                    game_packet = build_game_slate_ai_review_packet(**game_packet_kwargs)
                    st.session_state["cbb_game_slate_ai_packet"] = game_packet
                    st.session_state["cbb_game_slate_ai_prompt_system"] = GAME_SLATE_AI_REVIEW_SYSTEM_PROMPT
                    st.session_state["cbb_game_slate_ai_prompt_user"] = build_game_slate_ai_review_user_prompt(game_packet)
                    st.session_state["cbb_game_slate_ai_meta"] = {
                        "review_date": game_selected_date.isoformat(),
                        "prior_boxscore_date": prior_boxscore_date.isoformat(),
                        "focus_limit": int(game_agent_focus_limit),
                        "player_pool_rows": int(len(pool_df_agent)),
                        "dk_slate_rows": int(len(raw_slate_df_agent)),
                    }
                    st.session_state.pop("cbb_game_slate_ai_output", None)
                    st.success("Game Slate AI packet built.")

            game_packet_state = st.session_state.get("cbb_game_slate_ai_packet")
            game_prompt_system = str(
                st.session_state.get("cbb_game_slate_ai_prompt_system") or GAME_SLATE_AI_REVIEW_SYSTEM_PROMPT
            )
            game_prompt_user = str(st.session_state.get("cbb_game_slate_ai_prompt_user") or "")
            game_meta = st.session_state.get("cbb_game_slate_ai_meta") or {}

            if isinstance(game_packet_state, dict) and game_packet_state:
                packet_review_date = str((game_packet_state.get("review_context") or {}).get("review_date") or "")
                if packet_review_date and packet_review_date != game_selected_date.isoformat():
                    st.warning(
                        f"Packet is for `{packet_review_date}` while Game Data is set to `{game_selected_date.isoformat()}`. "
                        "Rebuild the packet to align dates."
                    )

                market_summary = game_packet_state.get("market_summary") or {}
                vegas_summary = game_packet_state.get("vegas_calibration") or {}
                focus_tables = game_packet_state.get("focus_tables") or {}
                gpp_team_targets = focus_tables.get("gpp_team_stack_targets") or []
                gpp_game_targets = focus_tables.get("gpp_game_stack_targets") or []
                gpp_player_targets = focus_tables.get("gpp_player_core_targets") or []
                gm1, gm2, gm3, gm4, gm5 = st.columns(5)
                gm1.metric("Games", int(_safe_int_value(market_summary.get("games"), default=0)))
                gm2.metric("Avg Total", f"{_safe_float_value(market_summary.get('avg_total_line'), default=0.0):.1f}")
                gm3.metric("GPP Team Targets", int(len(gpp_team_targets)))
                gm4.metric(
                    "Vegas Winner Acc %",
                    f"{_safe_float_value(vegas_summary.get('winner_pick_accuracy_pct'), default=0.0):.1f}",
                )
                gm5.metric("Player Pool Rows", int(_safe_int_value(game_meta.get("player_pool_rows"), default=0)))

                st.caption(
                    "Packet context: "
                    f"review_date={packet_review_date or 'n/a'}, "
                    f"prior_boxscore_date={str(game_meta.get('prior_boxscore_date') or 'n/a')}, "
                    f"dk_slate_rows={int(_safe_int_value(game_meta.get('dk_slate_rows'), default=0))}"
                )
                if gpp_game_targets:
                    st.markdown("**GPP Game Stack Targets**")
                    st.dataframe(pd.DataFrame(gpp_game_targets), hide_index=True, use_container_width=True)
                if gpp_team_targets:
                    st.markdown("**GPP Team Stack Cores**")
                    st.dataframe(pd.DataFrame(gpp_team_targets), hide_index=True, use_container_width=True)
                if gpp_player_targets:
                    st.markdown("**GPP Player Core Targets**")
                    st.dataframe(pd.DataFrame(gpp_player_targets), hide_index=True, use_container_width=True)
                else:
                    st.info(
                        "No player-level stack cores found. "
                        "Upload DK slate and build the active pool in `Slate + Vegas` for player-specific stack targets."
                    )

                game_packet_json = json.dumps(game_packet_state, indent=2, ensure_ascii=True)
                st.download_button(
                    "Download Game Slate Packet JSON",
                    data=game_packet_json,
                    file_name=f"game_slate_ai_packet_{game_selected_date.isoformat()}.json",
                    mime="application/json",
                    key="download_game_slate_ai_packet_json",
                )
                st.download_button(
                    "Download Game Slate Prompt TXT",
                    data=game_prompt_user,
                    file_name=f"game_slate_ai_prompt_{game_selected_date.isoformat()}.txt",
                    mime="text/plain",
                    key="download_game_slate_ai_prompt_txt",
                )
                with st.expander("Game Slate Packet Preview"):
                    st.json(game_packet_state)
                with st.expander("Game Slate Prompt Preview"):
                    st.text_area(
                        "Game Slate User Prompt",
                        value=game_prompt_user,
                        height=260,
                        key="game_slate_ai_prompt_preview_text",
                    )

                if run_game_slate_agent:
                    if not openai_key:
                        st.error("Set `OPENAI_API_KEY` or Streamlit secret `openai_api_key` first.")
                    else:
                        with st.spinner("Generating game-slate AI recommendations..."):
                            try:
                                game_ai_text = request_openai_review(
                                    api_key=openai_key,
                                    user_prompt=game_prompt_user,
                                    system_prompt=game_prompt_system,
                                    model=(game_agent_model or "gpt-5-mini"),
                                    max_output_tokens=game_agent_max_tokens,
                                    timeout_seconds=game_agent_timeout_seconds,
                                )
                                st.session_state["cbb_game_slate_ai_output"] = game_ai_text
                            except Exception as exc:
                                st.exception(exc)
                game_ai_output = str(st.session_state.get("cbb_game_slate_ai_output") or "").strip()
                if game_ai_output:
                    st.subheader("Game Slate AI Recommendations")
                    st.text_area(
                        "Game Slate Model Output",
                        value=game_ai_output,
                        height=340,
                        key="game_slate_ai_output_preview",
                    )
                    st.download_button(
                        "Download Game Slate Recommendations TXT",
                        data=game_ai_output,
                        file_name=f"game_slate_ai_recommendations_{game_selected_date.isoformat()}.txt",
                        mime="text/plain",
                        key="download_game_slate_ai_recommendations_txt",
                    )
            elif run_game_slate_agent:
                st.error("Build Game Slate packet first.")
        except Exception as exc:
            st.exception(exc)

with tab_props:
    st.caption(
        f"Props Date: `{props_selected_date.isoformat()}` | Mode: "
        + ("Historical Snapshot" if props_fetch_mode == "Historical Snapshot" else "Pregame Live")
    )
    st.subheader("Player Props")
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to load player props.")
    else:
        try:
            props_df = load_props_frame_for_date(
                bucket_name=bucket_name,
                selected_date=props_selected_date,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if props_df.empty:
                st.warning("No cached props found for selected date. Run `Run Props Import` first.")
            else:
                display = props_df.rename(
                    columns={
                        "game_date": "Game Date",
                        "home_team": "Home Team",
                        "away_team": "Away Team",
                        "bookmaker": "Bookmaker",
                        "market": "Market",
                        "player_name": "Player",
                        "line": "Line",
                        "over_price": "Over Price",
                        "under_price": "Under Price",
                    }
                )
                table_cols = [
                    "Game Date",
                    "Home Team",
                    "Away Team",
                    "Bookmaker",
                    "Market",
                    "Player",
                    "Line",
                    "Over Price",
                    "Under Price",
                ]
                st.dataframe(display[table_cols], hide_index=True, use_container_width=True)
        except Exception as exc:
            st.exception(exc)

with tab_game:
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

with tab_game:
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

                    raw_payload = store.read_raw_json(game_selected_date)
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

                    players_blob = store.players_blob_name(game_selected_date)
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
