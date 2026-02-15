from __future__ import annotations

import io
import inspect
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
    remove_injured_players,
)
from college_basketball_dfs.cbb_tail_model import fit_total_tail_model, score_odds_games_for_tail
from college_basketball_dfs.cbb_tournament_review import (
    build_field_entries_and_players,
    build_player_exposure_comparison,
    build_user_strategy_summary,
    summarize_generated_lineups,
    normalize_contest_standings_frame,
    extract_actual_ownership_from_standings,
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


def _injuries_blob_name() -> str:
    return "cbb/injuries/injuries_master.csv"


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
def load_injuries_frame(
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
    csv_text = _read_injuries_csv(store)
    if not csv_text or not csv_text.strip():
        return normalize_injuries_frame(None)
    frame = pd.read_csv(io.StringIO(csv_text))
    return normalize_injuries_frame(frame)


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
    out["actual_ownership"] = out["actual_ownership"].map(_ownership_to_float)
    out = out.loc[(out["ID"] != "") | (out["Name"] != "")]
    return out[["ID", "Name", "TeamAbbrev", "actual_ownership"]].reset_index(drop=True)


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
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership"])
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

with st.sidebar:
    st.header("Pipeline Settings")
    selected_date = st.date_input("Slate Date", value=prior_day())
    optimizer_slate_date = st.date_input("DK/Optimizer Slate Date", value=selected_date)
    props_date_preset = st.selectbox(
        "Props Date Preset",
        options=["Custom", "Today", "Tomorrow"],
        index=2,
        help="Use Today/Tomorrow for pregame props pulls without manual date entry.",
    )
    if props_date_preset == "Today":
        props_selected_date = date.today()
        st.caption(f"Props Date: {props_selected_date.isoformat()}")
    elif props_date_preset == "Tomorrow":
        props_selected_date = date.today() + timedelta(days=1)
        st.caption(f"Props Date: {props_selected_date.isoformat()}")
    else:
        props_selected_date = st.date_input("Props Date", value=prior_day())
    default_season_start = season_start_for_date(date.today())
    backfill_start = st.date_input("Backfill Start", value=default_season_start)
    backfill_end = st.date_input("Backfill End", value=prior_day())
    bucket_name = st.text_input("GCS Bucket", value=default_bucket)
    base_url = st.text_input("NCAA API Base URL", value=default_base_url)
    gcp_project = st.text_input("GCP Project (optional)", value=default_project)
    bookmakers_filter = st.text_input(
        "Bookmakers Filter",
        value=default_bookmakers,
        help="Comma-separated bookmaker keys (example: fanduel). Leave blank for all.",
    )
    st.caption(
        "The Odds API key source: "
        + ("loaded from secrets/env" if odds_api_key else "missing (`the_odds_api_key`)")
    )
    force_refresh = st.checkbox("Force API refresh (ignore cached raw JSON)", value=False)
    backfill_sleep = st.number_input("Backfill Sleep Seconds", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
    stop_on_error = st.checkbox("Stop Backfill On Error", value=False)
    props_markets = st.text_input(
        "Props Markets",
        value="player_points,player_rebounds,player_assists",
        help="Comma-separated The Odds API player prop market keys.",
    )
    props_fetch_mode = st.selectbox(
        "Props Fetch Mode",
        options=["Pregame Live", "Historical Snapshot"],
        index=0,
        help="Use Pregame Live for today/tomorrow pulls prior to tip-off.",
    )
    props_import_mode = st.selectbox(
        "Props Import Mode",
        options=["Auto (Cache -> API)", "Cache Only", "Force API Refresh"],
        index=0,
        help="Choose whether props import can call API or only load from cached GCS data.",
    )
    props_event_sleep_seconds = st.number_input(
        "Props Event Sleep Seconds",
        min_value=0.0,
        max_value=5.0,
        value=float(os.getenv("CBB_ODDS_EVENT_SLEEP_SECONDS", "0.6")),
        step=0.1,
        help="Delay between event-level props requests to avoid Odds API frequency limits (429).",
    )
    st.caption("Run imports and backfills from the tabs below.")

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

with tab_game:
    st.subheader("Game Imports")
    c1, c2, c3 = st.columns(3)
    run_clicked = c1.button("Run Cache Pipeline", key="run_cache_pipeline")
    run_odds_clicked = c2.button("Run Odds Import", key="run_odds_import")
    preview_clicked = c3.button("Load Cached Preview", key="preview_cached_data")

with tab_props:
    st.subheader("Prop Imports")
    run_props_clicked = st.button("Run Props Import", key="run_props_import")

with tab_backfill:
    st.subheader("Backfill Jobs")
    c4, c5 = st.columns(2)
    run_backfill_clicked = c4.button("Run Season Backfill", key="run_season_backfill")
    run_odds_backfill_clicked = c5.button("Run Odds Season Backfill", key="run_odds_season_backfill")

with tab_dk:
    st.subheader("DraftKings Slate Upload")
    dk_slate_date = st.date_input("DraftKings Slate Date", value=selected_date, key="dk_slate_date")
    uploaded_dk_slate = st.file_uploader(
        "Upload DraftKings Slate CSV",
        type=["csv"],
        key="dk_slate_upload",
        help="Upload the DraftKings player/salary slate CSV for this date.",
    )
    d1, d2 = st.columns(2)
    upload_dk_slate_clicked = d1.button("Upload DK Slate to GCS", key="upload_dk_slate_to_gcs")
    load_dk_slate_clicked = d2.button("Refresh Cached Slate View", key="refresh_dk_slate_view")

with tab_game:
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
                        game_date=selected_date,
                        bucket_name=bucket_name,
                        odds_api_key=odds_api_key,
                        bookmakers=(bookmakers_filter.strip() or None),
                        historical_mode=(selected_date < date.today()),
                        historical_snapshot_time=f"{selected_date.isoformat()}T23:59:59Z"
                        if selected_date < date.today()
                        else None,
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
                                "bookmakers": (bookmakers_filter.strip() or None),
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
                                "bookmakers": (bookmakers_filter.strip() or None),
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
                            "bookmakers": (bookmakers_filter.strip() or None),
                            "historical_mode": (props_fetch_mode == "Historical Snapshot"),
                            "historical_snapshot_time": None,
                            "inter_event_sleep_seconds": float(props_event_sleep_seconds),
                            "force_refresh": (True if props_import_mode == "Force API Refresh" else force_refresh),
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
                        bookmakers=(bookmakers_filter.strip() or None),
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
    st.caption("Persistent injury list stored in GCS by player + team.")
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to manage injuries.")
    else:
        try:
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
            add_injury_clicked = st.button("Add Injury", key="add_injury_button")

            injuries_df = load_injuries_frame(
                bucket_name=bucket_name,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )

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
                    merged = pd.concat([injuries_df, new_row], ignore_index=True)
                    merged = normalize_injuries_frame(merged)
                    client = build_storage_client(
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                        project=gcp_project or None,
                    )
                    store = CbbGcsStore(bucket_name=bucket_name, client=client)
                    _write_injuries_csv(store, merged.to_csv(index=False))
                    load_injuries_frame.clear()
                    st.success("Injury added.")
                    injuries_df = load_injuries_frame(
                        bucket_name=bucket_name,
                        gcp_project=gcp_project or None,
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                    )

            st.subheader("Injury List Editor")
            edited_injuries = st.data_editor(
                injuries_df,
                num_rows="dynamic",
                use_container_width=True,
                key="injuries_editor_df",
            )
            save_col, reload_col = st.columns(2)
            save_injuries_clicked = save_col.button("Save Injury List", key="save_injury_list")
            reload_injuries_clicked = reload_col.button("Reload from GCS", key="reload_injury_list")

            if save_injuries_clicked:
                normalized = normalize_injuries_frame(edited_injuries)
                client = build_storage_client(
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                    project=gcp_project or None,
                )
                store = CbbGcsStore(bucket_name=bucket_name, client=client)
                blob_name = _write_injuries_csv(store, normalized.to_csv(index=False))
                load_injuries_frame.clear()
                st.success(f"Saved injury list to `{blob_name}`")

            if reload_injuries_clicked:
                load_injuries_frame.clear()
                st.info("Reloaded injury cache from GCS.")
        except Exception as exc:
            st.exception(exc)

with tab_slate_vegas:
    st.subheader("Slate + Vegas Player Pool")
    st.caption("Lineup Generator uses the `blended_projection` (Projected DK Points) from this table.")
    vegas_bookmaker = st.text_input(
        "Vegas Bookmaker Source",
        value=(bookmakers_filter.strip() or "fanduel"),
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
                slate_date=optimizer_slate_date,
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
                st.caption(
                    f"Season stats rows used: `{len(season_history_df):,}` | "
                    f"Average projected minutes: `{avg_mins:.1f}`"
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
                        "vegas_dk_projection",
                    ]
                    existing_cols = [c for c in show_cols if c in pool_df.columns]
                    display_pool = pool_df[existing_cols].sort_values("projected_dk_points", ascending=False)
                    numeric_cols = [
                        "Salary",
                        "projection_per_dollar",
                        "blended_projection",
                        "our_minutes_avg",
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
                        "vegas_dk_projection",
                    ]
                    for col in numeric_cols:
                        if col in display_pool.columns:
                            display_pool[col] = pd.to_numeric(display_pool[col], errors="coerce")
                    st.dataframe(display_pool, hide_index=True, use_container_width=True)
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
                            optimizer_slate_date,
                            display_pool.to_csv(index=False),
                        )
                        st.success(f"Saved projections to `{blob_name}` (same date overwrites).")
                    st.download_button(
                        "Download Active Pool CSV",
                        data=display_pool.to_csv(index=False),
                        file_name=f"cbb_active_pool_{optimizer_slate_date.isoformat()}.csv",
                        mime="text/csv",
                        key="download_active_pool_csv",
                    )
        except Exception as exc:
            st.exception(exc)

with tab_lineups:
    st.subheader("DK Lineup Generator")
    lineup_bookmaker = st.text_input(
        "Lineup Bookmaker Source",
        value=(bookmakers_filter.strip() or "fanduel"),
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
            "All Versions generates and saves all three lineup models: "
            "standard_v1, spike_v1_legacy, spike_v2_tail."
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
            ],
            index=0,
            help=(
                "Run one lineup model. Use All Versions to save all three in a single run."
            ),
        )
        if lineup_model_label == "Spike v2 (Tail A/B)":
            selected_model_key = "spike_v2_tail"
            lineup_strategy = "spike"
            include_tail_signals = True
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
        c5.write("All Versions: `standard_v1`, `spike_v1_legacy`, `spike_v2_tail`")
        selected_model_key = "standard_v1"
        lineup_strategy = "standard"
        include_tail_signals = False

    max_salary_left = int(
        c6.slider(
            "Max Salary Left Per Lineup",
            min_value=0,
            max_value=10000,
            value=2000,
            step=100,
            help="Lineups must use at least 50000 - this value in salary.",
        )
    )
    global_max_exposure_pct = float(
        c7.slider(
            "Global Max Player Exposure %",
            min_value=0,
            max_value=100,
            value=100,
            step=1,
            help="Caps every player's max lineup rate across the run (locks override this cap).",
        )
    )
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
    auto_save_runs_to_gcs = st.checkbox(
        "Auto-save generated lineup runs to GCS",
        value=True,
        help="Saves manifest + per-version lineup JSON/CSV + DK upload CSV.",
    )

    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to generate lineups.")
    else:
        try:
            pool_df, removed_injured_df, raw_slate_df, _, _ = build_optimizer_pool_for_date(
                bucket_name=bucket_name,
                slate_date=optimizer_slate_date,
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
                            max_salary_left=max_salary_left,
                            lineup_strategy=str(version_cfg["lineup_strategy"]),
                            include_tail_signals=bool(version_cfg.get("include_tail_signals", False)),
                            spike_max_pair_overlap=int(version_cfg["spike_max_pair_overlap"]),
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
                        "slate_date": optimizer_slate_date.isoformat(),
                        "run_mode": run_mode_key,
                        "settings": {
                            "selected_model_key": selected_model_key,
                            "lineup_count": lineup_count,
                            "contest_type": contest_type,
                            "lineup_seed": lineup_seed,
                            "max_salary_left": max_salary_left,
                            "global_max_exposure_pct": global_max_exposure_pct,
                            "spike_max_pair_overlap": spike_max_pair_overlap,
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
                        saved_meta = persist_lineup_run_bundle(store, optimizer_slate_date, run_bundle)
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
                            file_name=f"dk_lineups_{optimizer_slate_date.isoformat()}_{active_version_key}.csv",
                            mime="text/csv",
                            key="download_dk_upload_csv",
                        )
                elif generate_lineups_clicked:
                    st.error("No lineups were generated. Adjust locks/exclusions/exposure settings and retry.")

                st.markdown("---")
                st.subheader("Saved Lineup Runs (GCS)")
                rr1, rr2 = st.columns(2)
                refresh_saved_runs_clicked = rr1.button("Refresh Saved Runs", key="refresh_saved_runs")
                if refresh_saved_runs_clicked:
                    load_saved_lineup_run_manifests.clear()
                    load_saved_lineup_version_payload.clear()

                saved_manifests = load_saved_lineup_run_manifests(
                    bucket_name=bucket_name,
                    selected_date=optimizer_slate_date,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                if not saved_manifests:
                    st.info("No saved lineup runs found for this slate date yet.")
                else:
                    run_option_map: dict[str, dict[str, Any]] = {}
                    run_options: list[str] = []
                    for manifest in saved_manifests:
                        run_id = str(manifest.get("run_id") or "")
                        generated_at = str(manifest.get("generated_at_utc") or "")
                        run_mode = str(manifest.get("run_mode") or "single")
                        label = f"{generated_at} | {run_id} | {run_mode}"
                        run_options.append(label)
                        run_option_map[label] = manifest

                    selected_run_label = st.selectbox(
                        "Saved Run",
                        options=run_options,
                        index=0,
                        key="saved_run_picker",
                    )
                    selected_manifest = run_option_map[selected_run_label]
                    versions_meta = selected_manifest.get("versions") or []
                    if versions_meta:
                        version_meta_map = {str(v.get("version_key") or ""): v for v in versions_meta}
                        saved_version_keys = [k for k in version_meta_map.keys() if k]
                        if saved_version_keys:
                            selected_saved_version = st.selectbox(
                                "Saved Version",
                                options=saved_version_keys,
                                index=0,
                                format_func=lambda k: (
                                    f"{version_meta_map[k].get('version_label', k)} "
                                    f"[{k} | {'tail' if bool(version_meta_map[k].get('include_tail_signals', False)) else 'legacy'}]"
                                ),
                                key="saved_run_version_picker",
                            )
                            saved_payload = load_saved_lineup_version_payload(
                                bucket_name=bucket_name,
                                selected_date=optimizer_slate_date,
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
                                            f"dk_lineups_{optimizer_slate_date.isoformat()}_"
                                            f"{selected_manifest.get('run_id', 'run')}_{selected_saved_version}.csv"
                                        ),
                                        mime="text/csv",
                                        key=f"download_saved_dk_upload_csv_{selected_saved_version}",
                                    )
        except Exception as exc:
            st.exception(exc)

with tab_projection_review:
    st.subheader("Projection Review")
    review_date = st.date_input("Review Date", value=optimizer_slate_date, key="projection_review_date")
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
                    if "ID" in review.columns:
                        review = review.merge(own_df[["ID", "actual_ownership"]], on="ID", how="left")
                    elif "Name" in review.columns:
                        review = review.merge(own_df[["Name", "actual_ownership"]], on="Name", how="left")
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
                    "Salary",
                    "blended_projection",
                    "our_dk_projection",
                    "vegas_dk_projection",
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
                st.dataframe(view_df, hide_index=True, use_container_width=True)
                st.download_button(
                    "Download Projection Review CSV",
                    data=view_df.to_csv(index=False),
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
    tr_date = st.date_input("Tournament Date", value=optimizer_slate_date, key="tournament_review_date")
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
                if projections_df.empty and not slate_df.empty:
                    fallback_pool, _, _, _, _ = build_optimizer_pool_for_date(
                        bucket_name=bucket_name,
                        slate_date=tr_date,
                        bookmaker=(bookmakers_filter.strip() or None),
                        gcp_project=gcp_project or None,
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                    )
                    projections_df = fallback_pool

                entries_df, expanded_df = build_field_entries_and_players(normalized_standings, slate_df)
                if entries_df.empty:
                    st.warning("Could not parse lineup strings from this standings file.")
                else:
                    actual_own_df = extract_actual_ownership_from_standings(normalized_standings)
                    exposure_df = build_player_exposure_comparison(
                        expanded_players_df=expanded_df,
                        entry_count=int(len(entries_df)),
                        projection_df=projections_df,
                        actual_ownership_df=actual_own_df,
                    )
                    user_summary_df = build_user_strategy_summary(entries_df)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Field Entries", int(len(entries_df)))
                    m2.metric("Avg Salary Left", f"{float(entries_df['salary_left'].mean()):.0f}")
                    m3.metric("Avg Max Team Stack", f"{float(entries_df['max_team_stack'].mean()):.2f}")
                    m4.metric("Avg Max Game Stack", f"{float(entries_df['max_game_stack'].mean()):.2f}")
                    top10 = entries_df.nsmallest(10, "Rank")
                    m5.metric("Top-10 Avg Salary Left", f"{float(top10['salary_left'].mean()):.0f}")

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
                    show_entry_cols = [
                        "Rank",
                        "EntryId",
                        "EntryName",
                        "Points",
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
