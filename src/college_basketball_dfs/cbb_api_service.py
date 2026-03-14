from __future__ import annotations

import io
import json
import math
from datetime import date, datetime, timedelta
import os
from pathlib import Path
import re
from typing import Any

import pandas as pd

from .cbb_dk_registry import (
    build_dk_identity_registry,
    build_registry_history_from_local_directory,
    build_rotowire_dk_slate,
    derive_manual_overrides_from_dk_slate,
    extract_registry_rows_from_dk_slate,
    manual_overrides_to_history_frame,
)
from .cbb_backfill import iter_dates, season_start_for_date
from .cbb_dk_optimizer import build_player_pool, normalize_injuries_frame, remove_injured_players
from .cbb_gcs import CbbGcsStore, build_storage_client
from .cbb_odds import flatten_odds_payload, flatten_player_props_payload
from .cbb_rotowire import RotoWireClient, flatten_slates, normalize_players, select_slate
from .cbb_tail_model import fit_total_tail_model, score_odds_games_for_tail
from .cbb_tournament_review import (
    build_entry_actual_points_comparison,
    build_field_entries_and_players,
    build_ownership_projection_diagnostics,
    build_ownership_teacher_review,
    build_player_exposure_comparison,
    build_projection_actual_comparison,
    detect_contest_standings_upload,
    extract_actual_ownership_from_standings,
    normalize_contest_standings_frame,
)
from .cbb_vegas_review import (
    build_calibration_models_frame,
    build_spread_buckets_frame,
    build_total_buckets_frame,
    build_vegas_review_games_frame,
    summarize_vegas_accuracy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DRAFTKINGS_DIR = REPO_ROOT / "Draftkings"
DEFAULT_MANUAL_OVERRIDES_PATH = REPO_ROOT / "data" / "dk_manual_overrides.csv"
LOCAL_LINEUP_RUNS_ROOT = REPO_ROOT / "data" / "local_lineup_runs"

MANUAL_OVERRIDE_COLUMNS = [
    "player_name",
    "team_abbr",
    "opp_abbr",
    "salary",
    "position",
    "roster_position",
    "game_key",
    "dk_id",
    "name_plus_id",
    "slate_date",
    "slate_key",
    "reason",
    "source_name",
]


def _safe_path_segment(value: object, default: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", str(value or "").strip())
    return safe or default


def _normalize_slate_label(value: object, default: str = "Main") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text or default


def _slate_key_from_label(value: object, default: str = "main") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or default


def _resolve_selected_date(selected_date: date | str) -> date:
    target_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError("selected_date must be a valid date")
    return target_date.date()


def _api_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (date, datetime, pd.Timestamp)):
        try:
            return value.isoformat()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {str(k): _api_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_api_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _api_json_safe(value.item())
        except Exception:
            return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def filter_unresolved_resolution_rows(resolution_df: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(resolution_df, pd.DataFrame) or resolution_df.empty:
        return pd.DataFrame()
    status = resolution_df.get("dk_resolution_status")
    if not isinstance(status, pd.Series):
        return pd.DataFrame()
    status = status.fillna("").astype(str).str.strip().str.lower()
    return resolution_df.loc[status != "resolved"].copy()


def _resolve_api_bucket_name(bucket_name: str | None = None) -> str:
    resolved = (bucket_name or os.getenv("CBB_GCS_BUCKET", "")).strip()
    if not resolved:
        raise ValueError("CBB_GCS_BUCKET is required for Vegas and props review endpoints.")
    return resolved


def _build_api_store(
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> CbbGcsStore:
    resolved_bucket = _resolve_api_bucket_name(bucket_name)
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=(gcp_project or os.getenv("GCP_PROJECT") or None),
    )
    return CbbGcsStore(bucket_name=resolved_bucket, client=client)


def _dedupe_manual_overrides(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)

    out = frame.copy()
    for col in MANUAL_OVERRIDE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[MANUAL_OVERRIDE_COLUMNS].copy()
    for col in ["player_name", "team_abbr", "slate_key", "dk_id", "position", "roster_position"]:
        out[col] = out[col].astype(str).str.strip()
    out["team_abbr"] = out["team_abbr"].str.upper()
    out["slate_key"] = out["slate_key"].str.lower()
    out["position"] = out["position"].str.upper()
    out["roster_position"] = out["roster_position"].str.upper()
    out["salary"] = pd.to_numeric(out["salary"], errors="coerce")
    out["slate_date"] = pd.to_datetime(out["slate_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.loc[(out["player_name"] != "") & (out["team_abbr"] != "") & (out["dk_id"] != "")]
    if out.empty:
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    return out.drop_duplicates(subset=["player_name", "team_abbr", "slate_key"], keep="last").reset_index(drop=True)


def load_manual_overrides(path: str | Path | None = None) -> pd.DataFrame:
    target = Path(path) if path else DEFAULT_MANUAL_OVERRIDES_PATH
    if not target.exists():
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    try:
        return _dedupe_manual_overrides(pd.read_csv(target))
    except Exception:
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)


def write_manual_overrides(frame: pd.DataFrame, path: str | Path | None = None) -> Path:
    target = Path(path) if path else DEFAULT_MANUAL_OVERRIDES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    deduped = _dedupe_manual_overrides(frame)
    deduped.to_csv(target, index=False)
    return target


def merge_manual_overrides(existing_frame: pd.DataFrame, new_frame: pd.DataFrame) -> pd.DataFrame:
    if existing_frame is None or existing_frame.empty:
        merged = new_frame.copy() if new_frame is not None else pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    elif new_frame is None or new_frame.empty:
        merged = existing_frame.copy()
    else:
        merged = pd.concat([existing_frame, new_frame], ignore_index=True)
    return _dedupe_manual_overrides(merged)


def _maybe_build_api_store(
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> CbbGcsStore | None:
    resolved_bucket = str(bucket_name or os.getenv("CBB_GCS_BUCKET") or "").strip()
    if not resolved_bucket:
        return None
    return _build_api_store(
        bucket_name=resolved_bucket,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )


def _load_manual_overrides_from_csv_text(csv_text: str | None) -> pd.DataFrame:
    if not csv_text or not str(csv_text).strip():
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    try:
        frame = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    return _dedupe_manual_overrides(frame)


def _load_manual_overrides_from_gcs_store(store: CbbGcsStore | None) -> pd.DataFrame:
    if store is None:
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    reader = getattr(store, "read_dk_registry_manual_csv", None)
    if not callable(reader):
        return pd.DataFrame(columns=MANUAL_OVERRIDE_COLUMNS)
    return _load_manual_overrides_from_csv_text(reader())


def _write_manual_overrides_to_gcs_store(store: CbbGcsStore | None, frame: pd.DataFrame) -> str | None:
    if store is None:
        return None
    writer = getattr(store, "write_dk_registry_manual_csv", None)
    if not callable(writer):
        return None
    payload = io.StringIO()
    _dedupe_manual_overrides(frame).to_csv(payload, index=False)
    return str(writer(payload.getvalue()))


def _parse_gcs_dk_slate_blob_name(blob_name: str) -> tuple[str | None, str]:
    normalized = str(blob_name or "").strip().replace("\\", "/")
    match = re.search(r"(\d{4}-\d{2}-\d{2})", normalized)
    if not match:
        return None, "main"
    iso_date = match.group(1)
    parts = [part for part in normalized.split("/") if part]
    filename = parts[-1] if parts else ""
    if filename == f"{iso_date}_dk_slate.csv":
        return iso_date, "main"
    if filename.endswith("_dk_slate.csv") and len(parts) >= 2 and parts[-2] == iso_date:
        slate_key = filename[: -len("_dk_slate.csv")] or "main"
        return iso_date, str(slate_key).strip().lower() or "main"
    return iso_date, "main"


def _build_registry_history_from_gcs_store(store: CbbGcsStore | None) -> pd.DataFrame:
    if store is None:
        return pd.DataFrame()
    lister = getattr(store, "list_dk_slate_blob_names", None)
    if not callable(lister):
        return pd.DataFrame()
    reader = getattr(store, "read_players_csv_blob", None)

    frames: list[pd.DataFrame] = []
    for blob_name in list(lister() or []):
        slate_date, slate_key = _parse_gcs_dk_slate_blob_name(str(blob_name))
        if not slate_date:
            continue
        try:
            csv_text = (
                reader(blob_name)
                if callable(reader)
                else store.bucket.blob(blob_name).download_as_text(encoding="utf-8")
            )
            if not csv_text or not str(csv_text).strip():
                continue
            slate_df = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            continue
        history = extract_registry_rows_from_dk_slate(
            slate_df,
            slate_date=slate_date,
            slate_key=slate_key,
            source_name=f"gcs:{blob_name}",
        )
        if history is not None and not history.empty:
            frames.append(history)

    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()
    return combined.drop_duplicates(
        subset=["dk_id", "team_abbr", "salary", "slate_date", "source_name"],
        keep="last",
    ).reset_index(drop=True)


def _load_combined_manual_overrides(
    *,
    manual_overrides_path: str | Path | None = None,
    manual_frame: pd.DataFrame | None = None,
    store: CbbGcsStore | None = None,
) -> pd.DataFrame:
    merged = load_manual_overrides(manual_overrides_path)
    merged = merge_manual_overrides(merged, _load_manual_overrides_from_gcs_store(store))
    if manual_frame is not None and not manual_frame.empty:
        merged = merge_manual_overrides(merged, manual_frame)
    return merged


def _build_registry_from_sources(
    *,
    draftkings_dir: str | Path | None = None,
    manual_overrides_path: str | Path | None = None,
    manual_frame: pd.DataFrame | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    draftkings_base = Path(draftkings_dir) if draftkings_dir else DEFAULT_DRAFTKINGS_DIR
    history_frames: list[pd.DataFrame] = []
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )

    local_history = build_registry_history_from_local_directory(draftkings_base)
    if local_history is not None and not local_history.empty:
        history_frames.append(local_history)

    gcs_history = _build_registry_history_from_gcs_store(store)
    if gcs_history is not None and not gcs_history.empty:
        history_frames.append(gcs_history)

    manual_df = _load_combined_manual_overrides(
        manual_overrides_path=manual_overrides_path,
        manual_frame=manual_frame,
        store=store,
    )
    manual_history = manual_overrides_to_history_frame(manual_df)
    if manual_history is not None and not manual_history.empty:
        history_frames.append(manual_history)

    if not history_frames:
        return pd.DataFrame()
    history = pd.concat(history_frames, ignore_index=True)
    if history.empty:
        return pd.DataFrame()
    return build_dk_identity_registry(history)


def load_registry(
    draftkings_dir: str | Path | None = None,
    manual_overrides_path: str | Path | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    return _build_registry_from_sources(
        draftkings_dir=draftkings_dir,
        manual_overrides_path=manual_overrides_path,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )


def list_rotowire_slates_for_date(
    selected_date: date | str,
    contest_type: str | None = None,
    slate_name: str | None = None,
    site_id: int = 1,
    cookie_header: str | None = None,
) -> pd.DataFrame:
    target_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError("selected_date must be a valid date")

    client = RotoWireClient(cookie_header=cookie_header)
    try:
        catalog = client.fetch_slate_catalog(site_id=site_id)
    finally:
        client.close()

    slates_df = flatten_slates(catalog, site_id=site_id)
    if slates_df.empty:
        return slates_df
    slates_df = slates_df.loc[slates_df["slate_date"].astype(str) == target_date.date().isoformat()].copy()
    if contest_type:
        slates_df = slates_df.loc[slates_df["contest_type"].astype(str).str.lower() == contest_type.strip().lower()].copy()
    if slate_name:
        slates_df = slates_df.loc[slates_df["slate_name"].astype(str).str.lower() == slate_name.strip().lower()].copy()
    return slates_df.reset_index(drop=True)


def load_rotowire_players_for_slate(
    *,
    selected_date: date | str,
    contest_type: str = "Classic",
    slate_name: str = "All",
    slate_id: int | None = None,
    site_id: int = 1,
    cookie_header: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError("selected_date must be a valid date")

    client = RotoWireClient(cookie_header=cookie_header)
    try:
        catalog = client.fetch_slate_catalog(site_id=site_id)
        slates_df = flatten_slates(catalog, site_id=site_id)
        selected = select_slate(
            slates_df=slates_df,
            slate_id=slate_id,
            slate_date=target_date.date().isoformat() if slate_id is None else None,
            contest_type=contest_type if slate_id is None else None,
            slate_name=slate_name if slate_id is None else None,
            first_match=False,
        )
        raw_players = client.fetch_players(int(selected["slate_id"]))
    finally:
        client.close()

    normalized = normalize_players(raw_players, slate_row=selected.to_dict())
    meta = {
        "slate_id": int(selected["slate_id"]),
        "slate_date": str(selected.get("slate_date") or ""),
        "contest_type": str(selected.get("contest_type") or ""),
        "slate_name": str(selected.get("slate_name") or ""),
        "game_count": int(selected.get("game_count") or 0),
        "players": int(len(normalized)),
    }
    return normalized, meta


def format_registry_coverage_error(
    coverage: dict[str, Any] | None,
    resolution_df: pd.DataFrame | None,
    *,
    sample_limit: int = 25,
) -> str:
    unresolved_rows: list[dict[str, Any]] = []
    unresolved = filter_unresolved_resolution_rows(resolution_df)
    if not unresolved.empty:
        sample_cols = [
            col
            for col in ["player_name", "team_abbr", "dk_resolution_status", "dk_match_reason"]
            if col in unresolved.columns
        ]
        unresolved = unresolved[sample_cols].head(max(1, int(sample_limit)))
        unresolved_rows = unresolved.to_dict(orient="records")
    coverage_dict = dict(coverage or {})
    return (
        "DK registry resolution incomplete. Resolve mismatches first. "
        f"coverage={coverage_dict.get('coverage_pct')} "
        f"unresolved={coverage_dict.get('unresolved_players')} "
        f"conflicts={coverage_dict.get('conflict_players')} "
        f"sample={unresolved_rows}"
    )


def resolve_rotowire_slate(
    *,
    selected_date: date | str,
    slate_key: str = "main",
    contest_type: str = "Classic",
    slate_name: str = "All",
    slate_id: int | None = None,
    site_id: int = 1,
    cookie_header: str | None = None,
    draftkings_dir: str | Path | None = None,
    manual_overrides_path: str | Path | None = None,
    manual_overrides_frame: pd.DataFrame | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    rotowire_df, slate_meta = load_rotowire_players_for_slate(
        selected_date=selected_date,
        contest_type=contest_type,
        slate_name=slate_name,
        slate_id=slate_id,
        site_id=site_id,
        cookie_header=cookie_header,
    )
    registry_df = _build_registry_from_sources(
        draftkings_dir=draftkings_dir,
        manual_overrides_path=manual_overrides_path,
        manual_frame=manual_overrides_frame,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    resolved_slate_df, resolution_df, coverage = build_rotowire_dk_slate(
        rotowire_df=rotowire_df,
        registry_df=registry_df,
        slate_date=pd.to_datetime(selected_date, errors="coerce").date().isoformat(),
        slate_key=slate_key,
    )
    unresolved_df = filter_unresolved_resolution_rows(resolution_df)
    coverage_error = (
        ""
        if bool((coverage or {}).get("fully_resolved"))
        else format_registry_coverage_error(coverage, resolution_df)
    )
    return {
        "slate": dict(slate_meta or {}),
        "coverage": dict(coverage or {}),
        "coverage_error": coverage_error,
        "rotowire_df": rotowire_df,
        "resolved_slate_df": resolved_slate_df,
        "resolution_df": resolution_df,
        "unresolved_players": unresolved_df.to_dict(orient="records"),
    }


def load_cached_dk_slate_frame(
    *,
    selected_date: date | str,
    slate_key: str = "main",
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    target_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError("selected_date must be a valid date")
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return pd.DataFrame()
    try:
        csv_text = store.read_dk_slate_csv(
            target_date.date(),
            str(slate_key or "").strip().lower() or "main",
        )
    except TypeError:
        csv_text = store.read_dk_slate_csv(target_date.date())
    if not csv_text or not str(csv_text).strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return pd.DataFrame()


def resolve_active_slate_context(
    *,
    selected_date: date | str,
    slate_key: str = "main",
    contest_type: str = "Classic",
    slate_name: str = "All",
    slate_id: int | None = None,
    site_id: int = 1,
    cookie_header: str | None = None,
    draftkings_dir: str | Path | None = None,
    manual_overrides_path: str | Path | None = None,
    manual_overrides_frame: pd.DataFrame | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    target_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(target_date):
        raise ValueError("selected_date must be a valid date")

    resolved_bundle: dict[str, Any] = {
        "slate": {},
        "coverage": {},
        "coverage_error": "",
        "rotowire_df": pd.DataFrame(),
        "resolved_slate_df": pd.DataFrame(),
        "resolution_df": pd.DataFrame(),
        "unresolved_players": [],
    }
    rotowire_error = ""
    try:
        resolved_bundle = resolve_rotowire_slate(
            selected_date=target_date.date(),
            slate_key=slate_key,
            contest_type=contest_type,
            slate_name=slate_name,
            slate_id=slate_id,
            site_id=site_id,
            cookie_header=cookie_header,
            draftkings_dir=draftkings_dir,
            manual_overrides_path=manual_overrides_path,
            manual_overrides_frame=manual_overrides_frame,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
        )
    except Exception as exc:
        rotowire_error = str(exc).strip()

    rotowire_df = (
        resolved_bundle.get("rotowire_df").copy()
        if isinstance(resolved_bundle.get("rotowire_df"), pd.DataFrame)
        else pd.DataFrame()
    )
    resolved_slate_df = (
        resolved_bundle.get("resolved_slate_df").copy()
        if isinstance(resolved_bundle.get("resolved_slate_df"), pd.DataFrame)
        else pd.DataFrame()
    )
    resolution_df = (
        resolved_bundle.get("resolution_df").copy()
        if isinstance(resolved_bundle.get("resolution_df"), pd.DataFrame)
        else pd.DataFrame()
    )
    coverage = dict(resolved_bundle.get("coverage") or {})
    coverage_error = str(resolved_bundle.get("coverage_error") or "").strip()
    if rotowire_error and not coverage_error:
        coverage_error = f"RotoWire slate load failed. {rotowire_error}"

    legacy_dk_slate_df = load_cached_dk_slate_frame(
        selected_date=target_date.date(),
        slate_key=slate_key,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    legacy_available = not legacy_dk_slate_df.empty
    rotowire_ready = bool(coverage.get("fully_resolved")) and not resolved_slate_df.empty

    active_slate_df = pd.DataFrame()
    active_source = "unavailable"
    active_source_label = "No active slate"
    active_source_detail = ""
    if legacy_available:
        active_slate_df = legacy_dk_slate_df.copy()
        active_source = "uploaded_dk_slate"
        active_source_label = "Uploaded DraftKings slate"
        if coverage_error:
            active_source_detail = (
                "Using the uploaded DraftKings slate as the active optimizer source. "
                "Supplemental player mapping is partial, so external priors may be incomplete."
            )
        elif rotowire_error:
            active_source_detail = (
                "Using the uploaded DraftKings slate as the active optimizer source. "
                "Supplemental source data is unavailable, so only baseline priors are attached."
            )
        else:
            active_source_detail = "Using the uploaded DraftKings slate as the active optimizer source."
    elif rotowire_ready:
        active_source = "unavailable"
        active_source_label = "No active slate"
        active_source_detail = (
            "No uploaded DraftKings slate is cached for the selected date+slate. "
            "Upload a DraftKings slate CSV to enable optimizer, exports, and downstream slate review."
        )
    else:
        active_source_detail = (
            "No uploaded DraftKings slate is cached for the selected date+slate. "
            "Upload a DraftKings slate CSV to continue."
        )

    return {
        "slate": dict(resolved_bundle.get("slate") or {}),
        "coverage": coverage,
        "coverage_error": coverage_error,
        "rotowire_error": rotowire_error,
        "rotowire_df": rotowire_df,
        "resolved_slate_df": resolved_slate_df,
        "resolution_df": resolution_df,
        "unresolved_players": list(resolved_bundle.get("unresolved_players") or []),
        "legacy_dk_slate_df": legacy_dk_slate_df,
        "legacy_dk_available": bool(legacy_available),
        "legacy_dk_rows": int(len(legacy_dk_slate_df)),
        "active_slate_df": active_slate_df,
        "active_source": active_source,
        "active_source_label": active_source_label,
        "active_source_detail": active_source_detail,
        "active_ready": bool(not active_slate_df.empty),
        "rotowire_fully_resolved": bool(coverage.get("fully_resolved")),
    }


def ensure_full_registry_coverage(
    coverage: dict[str, Any] | None,
    resolution_df: pd.DataFrame | None,
    *,
    sample_limit: int = 25,
) -> None:
    if bool((coverage or {}).get("fully_resolved")):
        return
    raise RuntimeError(format_registry_coverage_error(coverage, resolution_df, sample_limit=sample_limit))


def build_registry_coverage(
    *,
    selected_date: date | str,
    slate_key: str = "main",
    contest_type: str = "Classic",
    slate_name: str = "All",
    slate_id: int | None = None,
    site_id: int = 1,
    cookie_header: str | None = None,
    draftkings_dir: str | Path | None = None,
    manual_overrides_path: str | Path | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    resolved = resolve_rotowire_slate(
        selected_date=selected_date,
        slate_key=slate_key,
        contest_type=contest_type,
        slate_name=slate_name,
        slate_id=slate_id,
        site_id=site_id,
        cookie_header=cookie_header,
        draftkings_dir=draftkings_dir,
        manual_overrides_path=manual_overrides_path,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    return {
        "slate": dict(resolved.get("slate") or {}),
        "coverage": dict(resolved.get("coverage") or {}),
        "unresolved_players": list(resolved.get("unresolved_players") or []),
    }


def _local_lineup_run_dir(
    slate_date: date,
    run_id: str,
    slate_key: str | None = None,
) -> Path:
    safe_date = slate_date.isoformat()
    safe_slate = _slate_key_from_label(slate_key, default="main")
    safe_run = _safe_path_segment(run_id, "run")
    return LOCAL_LINEUP_RUNS_ROOT / safe_date / safe_slate / safe_run


def _read_json_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _load_local_lineup_run_manifests(
    selected_date: date,
    selected_slate_key: str | None,
) -> list[dict[str, Any]]:
    slate_dir = LOCAL_LINEUP_RUNS_ROOT / selected_date.isoformat() / _slate_key_from_label(selected_slate_key, default="main")
    if not slate_dir.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for manifest_path in sorted(slate_dir.glob("*/manifest.json"), reverse=True):
        payload = _read_json_payload(manifest_path)
        if not isinstance(payload, dict):
            continue
        payload["_storage_source"] = "local"
        payload["_manifest_ref"] = str(manifest_path)
        manifests.append(payload)
    manifests.sort(key=lambda item: str(item.get("generated_at_utc") or ""), reverse=True)
    return manifests


def _load_saved_lineup_run_manifests(
    *,
    selected_date: date,
    selected_slate_key: str | None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> list[dict[str, Any]]:
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return []
    try:
        run_ids = store.list_lineup_run_ids(selected_date, selected_slate_key)
    except TypeError:
        run_ids = store.list_lineup_run_ids(selected_date)
    manifests: list[dict[str, Any]] = []
    for run_id in run_ids:
        try:
            payload = store.read_lineup_run_manifest_json(selected_date, run_id, selected_slate_key)
        except TypeError:
            payload = store.read_lineup_run_manifest_json(selected_date, run_id)
        if not isinstance(payload, dict):
            continue
        if selected_slate_key:
            manifest_slate_key = _slate_key_from_label(
                payload.get("slate_key") or payload.get("slate_label"),
                default="main",
            )
            if manifest_slate_key != _slate_key_from_label(selected_slate_key):
                continue
        payload["_storage_source"] = "gcs"
        try:
            payload["_manifest_ref"] = store.lineup_run_manifest_blob_name(
                selected_date,
                str(run_id),
                slate_key=selected_slate_key,
            )
        except TypeError:
            payload["_manifest_ref"] = store.lineup_run_manifest_blob_name(selected_date, str(run_id))
        manifests.append(payload)
    manifests.sort(key=lambda item: str(item.get("generated_at_utc") or ""), reverse=True)
    return manifests


def _load_all_saved_lineup_run_manifests(
    *,
    selected_date: date,
    selected_slate_key: str | None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()

    for manifest in _load_local_lineup_run_manifests(selected_date, selected_slate_key):
        entry = dict(manifest)
        run_key = (
            str(entry.get("run_id") or "").strip(),
            _slate_key_from_label(entry.get("slate_key") or entry.get("slate_label"), default="main"),
        )
        seen_keys.add(run_key)
        manifests.append(entry)

    for manifest in _load_saved_lineup_run_manifests(
        selected_date=selected_date,
        selected_slate_key=selected_slate_key,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    ):
        entry = dict(manifest)
        run_key = (
            str(entry.get("run_id") or "").strip(),
            _slate_key_from_label(entry.get("slate_key") or entry.get("slate_label"), default="main"),
        )
        if run_key in seen_keys:
            manifests = [
                existing
                for existing in manifests
                if (
                    str(existing.get("run_id") or "").strip(),
                    _slate_key_from_label(existing.get("slate_key") or existing.get("slate_label"), default="main"),
                )
                != run_key
            ]
        seen_keys.add(run_key)
        manifests.append(entry)

    manifests.sort(key=lambda item: str(item.get("generated_at_utc") or ""), reverse=True)
    return manifests


def _load_local_lineup_version_payload(
    selected_date: date,
    run_id: str,
    version_key: str,
    selected_slate_key: str | None,
) -> dict[str, Any] | None:
    version_dir = _local_lineup_run_dir(selected_date, run_id, selected_slate_key) / _safe_path_segment(version_key, "version")
    payload_path = version_dir / "lineups.json"
    payload = _read_json_payload(payload_path)
    if payload is None:
        return None
    payload["_storage_source"] = "local"
    payload["_payload_ref"] = str(payload_path)
    return payload


def _load_saved_lineup_version_payload(
    *,
    selected_date: date,
    run_id: str,
    version_key: str,
    selected_slate_key: str | None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any] | None:
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return None
    try:
        payload = store.read_lineup_version_json(selected_date, run_id, version_key, selected_slate_key)
    except TypeError:
        payload = store.read_lineup_version_json(selected_date, run_id, version_key)
    if not isinstance(payload, dict):
        return None
    payload["_storage_source"] = "gcs"
    try:
        payload["_payload_ref"] = store.lineup_version_json_blob_name(
            selected_date,
            run_id,
            version_key,
            slate_key=selected_slate_key,
        )
    except TypeError:
        payload["_payload_ref"] = store.lineup_version_json_blob_name(selected_date, run_id, version_key)
    return payload


def _load_lineup_version_payload(
    *,
    selected_date: date,
    run_id: str,
    version_key: str,
    selected_slate_key: str | None,
    preferred_source: str | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any] | None:
    if preferred_source == "local":
        local_payload = _load_local_lineup_version_payload(selected_date, run_id, version_key, selected_slate_key)
        if local_payload is not None:
            return local_payload
    if preferred_source == "gcs":
        gcs_payload = _load_saved_lineup_version_payload(
            selected_date=selected_date,
            run_id=run_id,
            version_key=version_key,
            selected_slate_key=selected_slate_key,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
        )
        if gcs_payload is not None:
            return gcs_payload

    local_payload = _load_local_lineup_version_payload(selected_date, run_id, version_key, selected_slate_key)
    if local_payload is not None:
        return local_payload
    return _load_saved_lineup_version_payload(
        selected_date=selected_date,
        run_id=run_id,
        version_key=version_key,
        selected_slate_key=selected_slate_key,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )


def _build_lineup_run_version_summary(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "version_key": str(entry.get("version_key") or ""),
        "version_label": str(entry.get("version_label") or ""),
        "lineup_strategy": str(entry.get("lineup_strategy") or ""),
        "model_profile": str(entry.get("model_profile") or ""),
        "include_tail_signals": bool(entry.get("include_tail_signals", False)),
        "lineup_count_generated": int(entry.get("lineup_count_generated") or 0),
        "warning_count": int(entry.get("warning_count") or 0),
        "json_ref": str(entry.get("json_blob") or entry.get("json_path") or ""),
        "csv_ref": str(entry.get("csv_blob") or entry.get("csv_path") or ""),
        "dk_upload_ref": str(entry.get("dk_upload_blob") or entry.get("dk_upload_path") or ""),
    }


def _build_lineup_run_summary(
    manifest: dict[str, Any],
    *,
    include_versions: bool = False,
) -> dict[str, Any]:
    versions = [
        _build_lineup_run_version_summary(item)
        for item in (manifest.get("versions") or [])
        if isinstance(item, dict)
    ]
    summary = {
        "run_id": str(manifest.get("run_id") or ""),
        "slate_date": str(manifest.get("slate_date") or ""),
        "slate_key": _slate_key_from_label(manifest.get("slate_key") or manifest.get("slate_label"), default="main"),
        "slate_label": _normalize_slate_label(manifest.get("slate_label") or manifest.get("slate_key"), default="Main"),
        "generated_at_utc": str(manifest.get("generated_at_utc") or ""),
        "run_mode": str(manifest.get("run_mode") or "single"),
        "version_count": int(len(versions)),
        "lineups_generated": int(sum(int(item.get("lineup_count_generated") or 0) for item in versions)),
        "warnings_count": int(sum(int(item.get("warning_count") or 0) for item in versions)),
        "storage_source": str(manifest.get("_storage_source") or ""),
        "manifest_ref": str(manifest.get("_manifest_ref") or ""),
    }
    if include_versions:
        summary["versions"] = versions
    return summary


def build_slate_status_payload(
    *,
    selected_date: date | str,
    slate_key: str = "main",
    contest_type: str = "Classic",
    slate_name: str = "All",
    slate_id: int | None = None,
    site_id: int = 1,
    cookie_header: str | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    unresolved_sample_limit: int = 25,
) -> dict[str, Any]:
    bundle = resolve_active_slate_context(
        selected_date=selected_date,
        slate_key=slate_key,
        contest_type=contest_type,
        slate_name=slate_name,
        slate_id=slate_id,
        site_id=site_id,
        cookie_header=cookie_header,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    resolution_df = bundle.get("resolution_df")
    unresolved_sample_df = filter_unresolved_resolution_rows(resolution_df).head(max(1, int(unresolved_sample_limit)))
    active_slate_df = bundle.get("active_slate_df")
    legacy_dk_slate_df = bundle.get("legacy_dk_slate_df")
    rotowire_df = bundle.get("rotowire_df")
    slate_meta = dict(bundle.get("slate") or {})
    coverage = dict(bundle.get("coverage") or {})
    selected_day = _resolve_selected_date(selected_date)

    return _api_json_safe({
        "selected_date": selected_day.isoformat(),
        "slate_key": _slate_key_from_label(slate_key),
        "slate_label": _normalize_slate_label(slate_meta.get("slate_name") or slate_key, default="Main"),
        "slate": slate_meta,
        "active_rows": int(len(active_slate_df)) if isinstance(active_slate_df, pd.DataFrame) else 0,
        "cached_dk_rows": int(len(legacy_dk_slate_df)) if isinstance(legacy_dk_slate_df, pd.DataFrame) else 0,
        "slate_games": int(slate_meta.get("game_count") or 0),
        "active_source": {
            "code": str(bundle.get("active_source") or ""),
            "label": str(bundle.get("active_source_label") or ""),
            "detail": str(bundle.get("active_source_detail") or ""),
            "ready": bool(bundle.get("active_ready")),
        },
        "cached_dk_slate": {
            "available": bool(bundle.get("legacy_dk_available")),
            "rows": int(bundle.get("legacy_dk_rows") or 0),
        },
        "rotowire": {
            "loaded": bool(isinstance(rotowire_df, pd.DataFrame) and not rotowire_df.empty),
            "rows": int(len(rotowire_df)) if isinstance(rotowire_df, pd.DataFrame) else 0,
            "fully_resolved": bool(bundle.get("rotowire_fully_resolved")),
            "error": str(bundle.get("rotowire_error") or ""),
        },
        "registry_coverage": {
            **coverage,
            "error": str(bundle.get("coverage_error") or ""),
            "unresolved_sample": unresolved_sample_df.to_dict(orient="records"),
        },
    })


def build_lineup_runs_payload(
    *,
    selected_date: date | str,
    slate_key: str = "main",
    include_versions: bool = False,
    limit: int | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    selected_day = _resolve_selected_date(selected_date)
    manifests = _load_all_saved_lineup_run_manifests(
        selected_date=selected_day,
        selected_slate_key=slate_key,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if limit is not None and int(limit) >= 0:
        manifests = manifests[: int(limit)]
    runs = [_build_lineup_run_summary(manifest, include_versions=include_versions) for manifest in manifests]
    return _api_json_safe({
        "selected_date": selected_day.isoformat(),
        "slate_key": _slate_key_from_label(slate_key),
        "rows": int(len(runs)),
        "runs": runs,
    })


def build_lineup_run_detail_payload(
    *,
    selected_date: date | str,
    run_id: str,
    slate_key: str = "main",
    include_lineups: bool = True,
    include_upload_csv: bool = False,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    selected_day = _resolve_selected_date(selected_date)
    selected_slate_key = _slate_key_from_label(slate_key)
    resolved_run_id = str(run_id or "").strip()
    if not resolved_run_id:
        raise ValueError("run_id is required")

    manifests = _load_all_saved_lineup_run_manifests(
        selected_date=selected_day,
        selected_slate_key=selected_slate_key,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    manifest = next(
        (item for item in manifests if str(item.get("run_id") or "").strip() == resolved_run_id),
        None,
    )
    if manifest is None:
        raise ValueError(f"lineup run not found for {selected_day.isoformat()} / {selected_slate_key}: {resolved_run_id}")

    versions_out: list[dict[str, Any]] = []
    preferred_source = str(manifest.get("_storage_source") or "").strip().lower() or None
    for version_entry in manifest.get("versions") or []:
        if not isinstance(version_entry, dict):
            continue
        version_summary = _build_lineup_run_version_summary(version_entry)
        version_key = str(version_summary.get("version_key") or "").strip()
        payload = None
        if version_key:
            payload = _load_lineup_version_payload(
                selected_date=selected_day,
                run_id=resolved_run_id,
                version_key=version_key,
                selected_slate_key=selected_slate_key,
                preferred_source=preferred_source,
                bucket_name=bucket_name,
                gcp_project=gcp_project,
                service_account_json=service_account_json,
                service_account_json_b64=service_account_json_b64,
            )
        detail = dict(version_summary)
        detail["loaded"] = bool(isinstance(payload, dict))
        if isinstance(payload, dict):
            detail["version_label"] = str(payload.get("version_label") or detail.get("version_label") or version_key)
            detail["lineup_strategy"] = str(payload.get("lineup_strategy") or detail.get("lineup_strategy") or "")
            detail["model_profile"] = str(payload.get("model_profile") or detail.get("model_profile") or "")
            detail["include_tail_signals"] = bool(payload.get("include_tail_signals", detail.get("include_tail_signals", False)))
            detail["payload_source"] = str(payload.get("_storage_source") or preferred_source or "")
            detail["payload_ref"] = str(payload.get("_payload_ref") or "")
            warnings = payload.get("warnings") or []
            lineups = payload.get("lineups") or []
            detail["warnings"] = [str(item) for item in warnings]
            detail["lineups_loaded"] = int(len(lineups))
            if include_lineups:
                detail["lineups"] = lineups if isinstance(lineups, list) else []
            if include_upload_csv:
                detail["dk_upload_csv"] = str(payload.get("dk_upload_csv") or "")
        versions_out.append(detail)

    run_detail = _build_lineup_run_summary(manifest, include_versions=False)
    run_detail["settings"] = manifest.get("settings") or {}
    return _api_json_safe({
        "selected_date": selected_day.isoformat(),
        "slate_key": selected_slate_key,
        "run": run_detail,
        "versions": versions_out,
    })


def _default_roster_position(value: Any) -> str:
    position = str(value or "").strip().upper()
    if not position:
        return ""
    if position.startswith("G"):
        return "G/UTIL"
    if position.startswith("F"):
        return "F/UTIL"
    if position.startswith("C"):
        return "C/UTIL"
    return position


def _extract_dk_id_from_name_plus_id(value: Any) -> str:
    match = re.search(r"\(([^()]+)\)\s*$", str(value or "").strip())
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def import_dk_slate_overrides(
    *,
    dk_slate_csv_bytes: bytes,
    selected_date: date | str,
    slate_key: str = "main",
    contest_type: str = "Classic",
    slate_name: str = "All",
    slate_id: int | None = None,
    site_id: int = 1,
    cookie_header: str | None = None,
    draftkings_dir: str | Path | None = None,
    manual_overrides_path: str | Path | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    dk_slate_df = pd.read_csv(io.BytesIO(dk_slate_csv_bytes))
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    resolved_before = resolve_rotowire_slate(
        selected_date=selected_date,
        slate_key=slate_key,
        contest_type=contest_type,
        slate_name=slate_name,
        slate_id=slate_id,
        site_id=site_id,
        cookie_header=cookie_header,
        draftkings_dir=draftkings_dir,
        manual_overrides_path=manual_overrides_path,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    rotowire_df = resolved_before.get("rotowire_df")
    slate_meta = dict(resolved_before.get("slate") or {})
    resolution_before = resolved_before.get("resolution_df")
    coverage_before = dict(resolved_before.get("coverage") or {})

    derived_overrides, still_unresolved, derive_meta = derive_manual_overrides_from_dk_slate(
        rotowire_df=rotowire_df,
        resolution_df=resolution_before,
        dk_slate_df=dk_slate_df,
        slate_date=selected_date,
        slate_key=slate_key,
        source_name=f"dk_import:{pd.to_datetime(selected_date, errors='coerce').date().isoformat()}:{slate_key}",
    )

    existing_manual = _load_combined_manual_overrides(
        manual_overrides_path=manual_overrides_path,
        store=store,
    )
    merged_manual = merge_manual_overrides(existing_manual, derived_overrides)
    written_path: str | None = None
    written_blob: str | None = None
    if persist:
        if manual_overrides_path is not None or store is None:
            written = write_manual_overrides(merged_manual, manual_overrides_path)
            written_path = str(written)
        written_blob = _write_manual_overrides_to_gcs_store(store, merged_manual)
        if written_path is None:
            written_path = written_blob

    resolved_after = resolve_rotowire_slate(
        selected_date=selected_date,
        slate_key=slate_key,
        contest_type=contest_type,
        slate_name=slate_name,
        slate_id=slate_id,
        site_id=site_id,
        cookie_header=cookie_header,
        draftkings_dir=draftkings_dir,
        manual_overrides_path=manual_overrides_path,
        manual_overrides_frame=merged_manual,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    resolution_after = resolved_after.get("resolution_df")
    coverage_after = dict(resolved_after.get("coverage") or {})
    unresolved_before = filter_unresolved_resolution_rows(resolution_before)
    unresolved_after = filter_unresolved_resolution_rows(resolution_after)

    return {
        "slate": slate_meta,
        "coverage_before": dict(coverage_before or {}),
        "coverage_after": dict(coverage_after or {}),
        "derived_override_count": int(len(derived_overrides)),
        "remaining_unresolved_before_count": int(len(unresolved_before)),
        "remaining_unresolved_after_count": int(len(unresolved_after)),
        "remaining_unresolved_after": unresolved_after[
            [
                col
                for col in ["player_name", "team_abbr", "salary", "dk_resolution_status", "dk_match_reason"]
                if col in unresolved_after.columns
            ]
        ].to_dict(orient="records"),
        "derived_overrides": derived_overrides.to_dict(orient="records"),
        "derivation_meta": dict(derive_meta or {}),
        "persisted_manual_overrides_path": written_path,
        "persisted_manual_overrides_blob": written_blob,
        "persisted": bool(persist),
    }


def save_manual_resolution_overrides(
    *,
    overrides_frame: pd.DataFrame | list[dict[str, Any]],
    selected_date: date | str,
    slate_key: str = "main",
    manual_overrides_path: str | Path | None = None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    source_name: str | None = None,
    reason_default: str = "manual_alias_review",
) -> dict[str, Any]:
    frame = overrides_frame.copy() if isinstance(overrides_frame, pd.DataFrame) else pd.DataFrame(overrides_frame or [])
    if frame.empty:
        return {
            "saved_override_count": 0,
            "persisted_manual_overrides_path": None,
            "saved_overrides": [],
        }

    out = frame.copy()
    empty_series = pd.Series([""] * len(out), index=out.index, dtype="object")

    def _text_col(name: str) -> pd.Series:
        return out.get(name, empty_series).where(pd.notna(out.get(name, empty_series)), "").astype(str).str.strip()

    out["player_name"] = _text_col("player_name")
    out["team_abbr"] = _text_col("team_abbr").str.upper()
    out["opp_abbr"] = _text_col("opp_abbr").str.upper()
    out["position"] = _text_col("position").str.upper()
    out["roster_position"] = _text_col("roster_position").str.upper()
    out["game_key"] = _text_col("game_key").str.upper()
    out["dk_id"] = _text_col("dk_id")
    out["name_plus_id"] = _text_col("name_plus_id")
    out["reason"] = _text_col("reason")
    out["source_name"] = _text_col("source_name")
    out["salary"] = pd.to_numeric(out.get("salary"), errors="coerce")

    missing_position = out["position"] == ""
    out.loc[missing_position, "position"] = out.loc[missing_position, "roster_position"].str.split("/", n=1).str[0]

    missing_roster = out["roster_position"] == ""
    out.loc[missing_roster, "roster_position"] = out.loc[missing_roster, "position"].map(_default_roster_position)

    missing_dk_id = out["dk_id"] == ""
    out.loc[missing_dk_id, "dk_id"] = out.loc[missing_dk_id, "name_plus_id"].map(_extract_dk_id_from_name_plus_id)

    missing_name_plus_id = out["name_plus_id"] == ""
    out.loc[missing_name_plus_id, "name_plus_id"] = out.loc[missing_name_plus_id].apply(
        lambda row: (
            f"{str(row.get('player_name') or '').strip()} ({str(row.get('dk_id') or '').strip()})"
            if str(row.get("player_name") or "").strip() and str(row.get("dk_id") or "").strip()
            else ""
        ),
        axis=1,
    )

    missing_game_key = out["game_key"] == ""
    out.loc[missing_game_key, "game_key"] = out.loc[missing_game_key].apply(
        lambda row: (
            f"{str(row.get('team_abbr') or '').strip().upper()}@{str(row.get('opp_abbr') or '').strip().upper()}"
            if str(row.get("team_abbr") or "").strip() and str(row.get("opp_abbr") or "").strip()
            else ""
        ),
        axis=1,
    )

    iso_date = pd.to_datetime(selected_date, errors="coerce").date().isoformat()
    resolved_source_name = (
        str(source_name).strip()
        if str(source_name or "").strip()
        else f"manual_alias_review:{iso_date}:{str(slate_key or '').strip().lower() or 'main'}"
    )
    out["slate_date"] = iso_date
    out["slate_key"] = str(slate_key or "").strip().lower() or "main"
    out.loc[out["reason"] == "", "reason"] = str(reason_default or "manual_alias_review").strip()
    out.loc[out["source_name"] == "", "source_name"] = resolved_source_name

    out = out.loc[
        (out["player_name"] != "")
        & (out["team_abbr"] != "")
        & (out["dk_id"] != "")
        & (out["name_plus_id"] != "")
    ].copy()
    if out.empty:
        return {
            "saved_override_count": 0,
            "persisted_manual_overrides_path": None,
            "saved_overrides": [],
        }

    overrides_only = out[MANUAL_OVERRIDE_COLUMNS].copy()
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    existing_manual = _load_combined_manual_overrides(
        manual_overrides_path=manual_overrides_path,
        store=store,
    )
    merged_manual = merge_manual_overrides(existing_manual, overrides_only)
    written_path: str | None = None
    if manual_overrides_path is not None or store is None:
        written = write_manual_overrides(merged_manual, manual_overrides_path)
        written_path = str(written)
    written_blob = _write_manual_overrides_to_gcs_store(store, merged_manual)
    return {
        "saved_override_count": int(len(overrides_only)),
        "persisted_manual_overrides_path": written_path or written_blob,
        "persisted_manual_overrides_blob": written_blob,
        "saved_overrides": overrides_only.to_dict(orient="records"),
    }


def _serialize_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    out = frame.copy()
    if limit is not None and limit > 0:
        out = out.head(int(limit)).copy()
    out = out.where(pd.notna(out), None)
    return out.to_dict(orient="records")


def _extract_iso_date_from_blob_name(blob_name: str) -> str | None:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", str(blob_name or ""))
    if not match:
        return None
    try:
        parsed = pd.to_datetime(match.group(1), errors="coerce")
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _collect_blob_dates(
    store: CbbGcsStore,
    prefix: str,
    suffixes: list[str] | None = None,
) -> set[str]:
    out: set[str] = set()
    suffix_list = [str(s).strip().lower() for s in (suffixes or []) if str(s).strip()]
    for blob in store.bucket.list_blobs(prefix=prefix):
        name = str(getattr(blob, "name", "") or "")
        if not name:
            continue
        lower_name = name.lower()
        if suffix_list and not any(lower_name.endswith(sfx) for sfx in suffix_list):
            continue
        iso_date = _extract_iso_date_from_blob_name(name)
        if iso_date:
            out.add(iso_date)
    return out


def _iter_iso_dates(start_date: date, end_date: date) -> list[str]:
    cursor = start_date
    out: list[str] = []
    while cursor <= end_date:
        out.append(cursor.isoformat())
        cursor = cursor + timedelta(days=1)
    return out


def _coverage_stats(expected_dates: list[str], found_dates: set[str]) -> dict[str, Any]:
    found = [d for d in expected_dates if d in found_dates]
    missing = [d for d in expected_dates if d not in found_dates]
    total = len(expected_dates)
    count_found = len(found)
    pct = 0.0 if total <= 0 else (100.0 * float(count_found) / float(total))
    return {
        "found_dates": count_found,
        "total_dates": total,
        "coverage_pct": round(pct, 2),
        "missing_dates": missing[:180],
        "sample_found_dates": found[-10:],
    }


def build_cache_coverage_payload(
    *,
    start_date: date | str,
    end_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    parsed_start = pd.to_datetime(start_date, errors="coerce")
    parsed_end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(parsed_start) or pd.isna(parsed_end):
        raise ValueError("start_date and end_date must be valid dates")
    start_day = parsed_start.date()
    end_day = parsed_end.date()
    if end_day < start_day:
        raise ValueError("end_date must be on or after start_date")

    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )

    expected_dates = _iter_iso_dates(start_day, end_day)
    raw_dates = _collect_blob_dates(store, prefix="cbb/raw/", suffixes=[".json"])
    odds_dates = _collect_blob_dates(store, prefix="cbb/odds/", suffixes=[".json"])
    odds_games_dates = _collect_blob_dates(store, prefix="cbb/odds_games/", suffixes=["_odds.csv"])
    props_dates = _collect_blob_dates(store, prefix="cbb/props/", suffixes=[".json"])
    props_lines_dates = _collect_blob_dates(store, prefix="cbb/props_lines/", suffixes=["_props.csv"])
    players_dates = _collect_blob_dates(store, prefix="cbb/players/", suffixes=["_players.csv"])
    dk_slate_dates = _collect_blob_dates(store, prefix="cbb/dk_slates/", suffixes=["_dk_slate.csv"])
    projections_dates = _collect_blob_dates(store, prefix="cbb/projections/", suffixes=["_projections.csv"])
    ownership_dates = _collect_blob_dates(store, prefix="cbb/ownership/", suffixes=["_ownership.csv"])
    injuries_feed_dates = _collect_blob_dates(store, prefix="cbb/injuries/feed/", suffixes=["_injuries_feed.csv"])

    coverage = {
        "raw_game_data": _coverage_stats(expected_dates, raw_dates),
        "odds_data": _coverage_stats(expected_dates, odds_dates),
        "odds_games_csv": _coverage_stats(expected_dates, odds_games_dates),
        "props_data": _coverage_stats(expected_dates, props_dates),
        "props_lines_csv": _coverage_stats(expected_dates, props_lines_dates),
        "players_results": _coverage_stats(expected_dates, players_dates),
        "dk_slates": _coverage_stats(expected_dates, dk_slate_dates),
        "projections_snapshots": _coverage_stats(expected_dates, projections_dates),
        "ownership_files": _coverage_stats(expected_dates, ownership_dates),
        "injuries_feed": _coverage_stats(expected_dates, injuries_feed_dates),
    }

    return {
        "bucket_name": store.bucket_name,
        "start_date": start_day.isoformat(),
        "end_date": end_day.isoformat(),
        "total_dates": len(expected_dates),
        "coverage": coverage,
    }


def _read_vegas_source_payloads(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> tuple[CbbGcsStore, dict[str, Any] | None, dict[str, Any] | None]:
    parsed_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(parsed_date):
        raise ValueError("selected_date must be a valid date")
    game_date = parsed_date.date()
    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    raw_payload = store.read_raw_json(game_date)
    odds_payload = store.read_odds_json(game_date)
    return store, raw_payload, odds_payload


def build_vegas_game_lines_payload(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    row_limit: int = 300,
) -> dict[str, Any]:
    parsed_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(parsed_date):
        raise ValueError("selected_date must be a valid date")
    game_date = parsed_date.date()
    store, raw_payload, odds_payload = _read_vegas_source_payloads(
        selected_date=game_date,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )

    games_df = pd.DataFrame()
    if isinstance(raw_payload, dict) and isinstance(odds_payload, dict):
        games_df = build_vegas_review_games_frame(raw_payloads=[raw_payload], odds_payloads=[odds_payload])
    summary = summarize_vegas_accuracy(games_df)
    display_cols = [
        "game_date",
        "away_team",
        "home_team",
        "total_points",
        "actual_total",
        "total_error",
        "spread_home",
        "actual_home_margin",
        "spread_error",
        "moneyline_home",
        "moneyline_away",
        "winner_pick_correct",
        "odds_match_type",
    ]
    rows_df = games_df[[c for c in display_cols if c in games_df.columns]].copy()
    if "total_points" in rows_df.columns:
        rows_df = rows_df.sort_values(["game_date", "total_points"], ascending=[False, False], na_position="last")
    else:
        rows_df = rows_df.sort_values(["game_date"], ascending=[False], na_position="last")

    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "available": bool(not games_df.empty),
        "source_status": {
            "raw_cached": bool(isinstance(raw_payload, dict)),
            "odds_cached": bool(isinstance(odds_payload, dict)),
        },
        "summary": summary,
        "rows": _serialize_records(rows_df, limit=row_limit),
    }


def build_vegas_market_context_payload(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    row_limit: int = 200,
) -> dict[str, Any]:
    parsed_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(parsed_date):
        raise ValueError("selected_date must be a valid date")
    game_date = parsed_date.date()
    store, raw_payload, odds_payload = _read_vegas_source_payloads(
        selected_date=game_date,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )

    games_df = pd.DataFrame()
    if isinstance(raw_payload, dict) and isinstance(odds_payload, dict):
        games_df = build_vegas_review_games_frame(raw_payloads=[raw_payload], odds_payloads=[odds_payload])

    summary = summarize_vegas_accuracy(games_df)
    calibration_df = build_calibration_models_frame(games_df)
    total_buckets_df = build_total_buckets_frame(games_df)
    spread_buckets_df = build_spread_buckets_frame(games_df)

    ranked_games = games_df.copy()
    if not ranked_games.empty:
        for col in ["total_points", "spread_home", "moneyline_home", "moneyline_away"]:
            if col in ranked_games.columns:
                ranked_games[col] = pd.to_numeric(ranked_games[col], errors="coerce")
        ranked_games["abs_spread"] = pd.to_numeric(ranked_games.get("spread_home"), errors="coerce").abs()
        ranked_games = ranked_games.sort_values(
            ["total_points", "abs_spread"],
            ascending=[False, True],
            na_position="last",
        )
    ranking_cols = [
        "game_date",
        "away_team",
        "home_team",
        "total_points",
        "spread_home",
        "moneyline_home",
        "moneyline_away",
        "bookmakers_count",
    ]
    ranked_games = ranked_games[[c for c in ranking_cols if c in ranked_games.columns]].copy()

    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "available": bool(not games_df.empty),
        "source_status": {
            "raw_cached": bool(isinstance(raw_payload, dict)),
            "odds_cached": bool(isinstance(odds_payload, dict)),
        },
        "summary": summary,
        "calibration_models": _serialize_records(calibration_df),
        "total_buckets": _serialize_records(total_buckets_df),
        "spread_buckets": _serialize_records(spread_buckets_df),
        "ranked_games": _serialize_records(ranked_games, limit=row_limit),
    }


def build_props_review_payload(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    row_limit: int = 400,
) -> dict[str, Any]:
    parsed_date = pd.to_datetime(selected_date, errors="coerce")
    if pd.isna(parsed_date):
        raise ValueError("selected_date must be a valid date")
    game_date = parsed_date.date()
    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    props_payload = store.read_props_json(game_date)
    if not isinstance(props_payload, dict):
        return {
            "selected_date": game_date.isoformat(),
            "bucket_name": store.bucket_name,
            "available": False,
            "source_status": {"props_cached": False},
            "summary": {
                "rows": 0,
                "markets": 0,
                "players": 0,
                "events": 0,
                "bookmakers": 0,
            },
            "market_coverage": [],
            "rows": [],
        }

    rows = flatten_player_props_payload(props_payload)
    props_df = pd.DataFrame(rows)
    if props_df.empty:
        return {
            "selected_date": game_date.isoformat(),
            "bucket_name": store.bucket_name,
            "available": False,
            "source_status": {"props_cached": True},
            "summary": {
                "rows": 0,
                "markets": 0,
                "players": 0,
                "events": 0,
                "bookmakers": 0,
            },
            "market_coverage": [],
            "rows": [],
        }

    props_df["game_date"] = pd.to_datetime(props_df.get("game_date"), errors="coerce").dt.strftime("%Y-%m-%d")
    props_df["market"] = props_df.get("market", "").astype(str)
    props_df["player_name"] = props_df.get("player_name", "").astype(str)
    props_df["bookmaker"] = props_df.get("bookmaker", "").astype(str)
    for col in ["line", "over_price", "under_price"]:
        if col in props_df.columns:
            props_df[col] = pd.to_numeric(props_df[col], errors="coerce")

    market_coverage_df = (
        props_df.groupby("market", dropna=False, observed=False)
        .agg(
            rows=("player_name", "count"),
            players=("player_name", "nunique"),
            events=("event_id", "nunique"),
            avg_line=("line", "mean"),
        )
        .reset_index()
        .sort_values(["rows", "players"], ascending=[False, False], na_position="last")
    )

    display_cols = [
        "game_date",
        "away_team",
        "home_team",
        "bookmaker",
        "market",
        "player_name",
        "line",
        "over_price",
        "under_price",
    ]
    display_df = props_df[[c for c in display_cols if c in props_df.columns]].copy()
    display_df = display_df.sort_values(["market", "player_name"], ascending=[True, True], na_position="last")

    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "available": True,
        "source_status": {"props_cached": True},
        "summary": {
            "rows": int(len(props_df)),
            "markets": int(props_df["market"].nunique()),
            "players": int(props_df["player_name"].nunique()),
            "events": int(props_df["event_id"].nunique()) if "event_id" in props_df.columns else 0,
            "bookmakers": int(props_df["bookmaker"].nunique()),
        },
        "market_coverage": _serialize_records(market_coverage_df),
        "rows": _serialize_records(display_df, limit=row_limit),
    }

NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


def _norm_name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _norm_name_key_loose(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in NAME_SUFFIX_TOKENS]
    return "".join(tokens)


def _normalize_player_id(value: Any) -> str:
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


def _to_iso_date(value: date | str) -> date:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("selected_date must be a valid date")
    return parsed.date()


def _read_csv_frame(csv_text: str | None) -> pd.DataFrame:
    if not csv_text or not str(csv_text).strip():
        return pd.DataFrame()
    try:
        return pd.read_csv(io.StringIO(str(csv_text)))
    except Exception:
        return pd.DataFrame()


def _read_injuries_feed_csv(store: CbbGcsStore, selected_date: date | None = None) -> str | None:
    reader = getattr(store, "read_injuries_feed_csv", None)
    if not callable(reader):
        return None
    try:
        return reader(game_date=selected_date)
    except TypeError:
        try:
            if selected_date is not None:
                return reader(selected_date)
        except TypeError:
            return reader()
    return reader()


def _read_injuries_manual_csv(store: CbbGcsStore) -> str | None:
    reader = getattr(store, "read_injuries_manual_csv", None)
    if not callable(reader):
        return None
    return reader()


def _read_injuries_csv(store: CbbGcsStore) -> str | None:
    reader = getattr(store, "read_injuries_csv", None)
    if not callable(reader):
        return None
    return reader()


def _list_players_blob_names(store: CbbGcsStore) -> list[str]:
    list_fn = getattr(store, "list_players_blob_names", None)
    if not callable(list_fn):
        return []
    return list(list_fn() or [])


def _read_players_csv_blob(store: CbbGcsStore, blob_name: str) -> str:
    read_fn = getattr(store, "read_players_csv_blob", None)
    if not callable(read_fn):
        return ""
    return str(read_fn(blob_name) or "")


def load_odds_frame_for_date(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    game_date = _to_iso_date(selected_date)
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return pd.DataFrame()
    payload = store.read_odds_json(game_date)
    if payload is None:
        return pd.DataFrame()
    rows = flatten_odds_payload(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame


def load_props_frame_for_date(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    game_date = _to_iso_date(selected_date)
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return pd.DataFrame()
    payload = store.read_props_json(game_date)
    if payload is None:
        return pd.DataFrame()
    rows = flatten_player_props_payload(payload)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["game_date"] = pd.to_datetime(frame["game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return frame


def load_injuries_feed_frame(
    *,
    selected_date: date | str | None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    game_date = _to_iso_date(selected_date) if selected_date is not None else None
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return normalize_injuries_frame(None)
    csv_text = _read_injuries_feed_csv(store, selected_date=game_date)
    if not csv_text or not csv_text.strip():
        return normalize_injuries_frame(None)
    return normalize_injuries_frame(_read_csv_frame(csv_text))


def load_injuries_manual_frame(
    *,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return normalize_injuries_frame(None)
    csv_text = _read_injuries_manual_csv(store)
    if not csv_text or not csv_text.strip():
        return normalize_injuries_frame(None)
    return normalize_injuries_frame(_read_csv_frame(csv_text))


def load_injuries_frame(
    *,
    selected_date: date | str | None,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    feed_df = load_injuries_feed_frame(
        selected_date=selected_date,
        bucket_name=bucket_name,
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
        store = _maybe_build_api_store(
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
        )
        if store is None:
            return normalize_injuries_frame(None)
        legacy_csv = _read_injuries_csv(store)
        if not legacy_csv or not legacy_csv.strip():
            return normalize_injuries_frame(None)
        return normalize_injuries_frame(_read_csv_frame(legacy_csv))

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
    combined = combined.drop_duplicates(subset=["_injury_key"], keep="last")
    combined = combined.drop(columns=["_injury_key", "_source"], errors="ignore")
    return normalize_injuries_frame(combined)


def load_season_player_history_frame(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    game_date = _to_iso_date(selected_date)
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return pd.DataFrame()
    season_start = season_start_for_date(game_date)
    selected_iso = game_date.isoformat()
    frames: list[pd.DataFrame] = []
    for blob_name in _list_players_blob_names(store):
        match = re.search(r"(\d{4}-\d{2}-\d{2})_players\.csv$", str(blob_name or ""))
        if not match:
            continue
        blob_date = match.group(1)
        if blob_date < season_start.isoformat() or blob_date > selected_iso:
            continue
        csv_text = _read_players_csv_blob(store, blob_name)
        if not csv_text.strip():
            continue
        df = _read_csv_frame(csv_text)
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
        if cols:
            frames.append(df[cols].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_season_vegas_history_frame(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    game_date = _to_iso_date(selected_date)
    store = _maybe_build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    if store is None:
        return pd.DataFrame()
    season_start = season_start_for_date(game_date)
    end_date = game_date - timedelta(days=1)
    if end_date < season_start:
        return pd.DataFrame()

    raw_payloads: list[dict[str, Any]] = []
    odds_payloads: list[dict[str, Any]] = []
    for one_day in iter_dates(season_start, end_date):
        raw_payload = store.read_raw_json(one_day)
        if isinstance(raw_payload, dict):
            raw_payloads.append(raw_payload)
        odds_payload = store.read_odds_json(one_day)
        if isinstance(odds_payload, dict):
            odds_payloads.append(odds_payload)
    if not raw_payloads or not odds_payloads:
        return pd.DataFrame()
    return build_vegas_review_games_frame(raw_payloads=raw_payloads, odds_payloads=odds_payloads)


def _decode_csv_payload(csv_bytes: bytes) -> tuple[str, pd.DataFrame]:
    if not csv_bytes:
        return "", pd.DataFrame()
    decode_attempts = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
    for enc in decode_attempts:
        try:
            text = csv_bytes.decode(enc)
            break
        except Exception:
            text = ""
    frame = _read_csv_frame(text)
    return text, frame


def _read_actual_results_frame(store: CbbGcsStore, selected_date: date) -> pd.DataFrame:
    blob_name = store.players_blob_name(selected_date)
    try:
        csv_text = store.read_players_csv_blob(blob_name)
    except Exception:
        return pd.DataFrame()
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()

    raw = _read_csv_frame(csv_text)
    if raw.empty:
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
    cols = [c for c in needed if c in raw.columns]
    if not cols:
        return pd.DataFrame()

    out = raw[cols].copy().rename(
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
    bonus = (dd_count >= 2).astype(float) * 1.5 + (dd_count >= 3).astype(float) * 3.0
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
    out["ID"] = out["ID"].map(_normalize_player_id)
    out["Name"] = out["Name"].astype(str).str.strip()
    out["team_name"] = out["team_name"].astype(str).str.strip()
    return out


def _read_projection_snapshot_frame(
    store: CbbGcsStore,
    selected_date: date,
    slate_key: str | None = None,
) -> pd.DataFrame:
    csv_text = store.read_projections_csv(selected_date, slate_key=slate_key)
    return _read_csv_frame(csv_text)


def _read_lineupstarter_frame(
    store: CbbGcsStore,
    selected_date: date,
    slate_key: str | None = None,
) -> pd.DataFrame:
    csv_text = store.read_lineupstarter_csv(selected_date, slate_key=slate_key)
    return _read_csv_frame(csv_text)


def _ownership_to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _lineupstarter_numeric(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "").replace("$", "").replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


_ROTOWIRE_STATUS_SUFFIXES = (
    "gtd",
    "dtd",
    "out",
    "questionable",
    "doubtful",
    "probable",
    "inactive",
    "suspended",
)

_ROTOWIRE_TEAM_ABBR_ALIASES = {
    "SANTAC": "STC",
    "SANFR": "SFO",
    "OREST": "ORST",
    "NDAKST": "NDSU",
    "NORDAK": "UND",
}


def _clean_rotowire_player_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    cleaned = text
    changed = True
    while changed:
        changed = False
        for suffix in _ROTOWIRE_STATUS_SUFFIXES:
            updated = re.sub(rf"(?:[\s\-/()]|^)*{re.escape(suffix)}$", "", cleaned, flags=re.IGNORECASE).strip()
            if updated != cleaned:
                cleaned = updated
                changed = True
    return cleaned.strip(" -/()")


def _normalize_rotowire_team_abbr(value: Any) -> str:
    raw = str(value or "").strip().upper().lstrip("@")
    if not raw:
        return ""
    return _ROTOWIRE_TEAM_ABBR_ALIASES.get(raw, raw)


def normalize_rotowire_upload_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "player_name",
        "team_abbr",
        "opp_abbr",
        "salary",
        "proj_fantasy_points",
        "proj_minutes",
        "proj_value_per_1k",
        "avg_fpts_last3",
        "avg_fpts_last5",
        "avg_fpts_last7",
        "avg_fpts_season",
        "usage_rate",
        "supplement_priority",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    out = df.copy()
    normalized_columns = {re.sub(r"[^a-z0-9]", "", str(c).strip().lower()): c for c in out.columns}
    rename_aliases = {
        "player": "player_name",
        "playername": "player_name",
        "name": "player_name",
        "athlete": "player_name",
        "team": "team_abbr",
        "teamabbr": "team_abbr",
        "teamabbrev": "team_abbr",
        "opponent": "opp_abbr",
        "opp": "opp_abbr",
        "oppabbr": "opp_abbr",
        "proj": "proj_fantasy_points",
        "projection": "proj_fantasy_points",
        "projectedpoints": "proj_fantasy_points",
        "projectedfantasypoints": "proj_fantasy_points",
        "projfantasypoints": "proj_fantasy_points",
        "projfpts": "proj_fantasy_points",
        "pts": "proj_fantasy_points",
        "projminutes": "proj_minutes",
        "projectedminutes": "proj_minutes",
        "minutes": "proj_minutes",
        "min": "proj_minutes",
        "projvalueper1k": "proj_value_per_1k",
        "valueper1k": "proj_value_per_1k",
        "value": "proj_value_per_1k",
        "val": "proj_value_per_1k",
        "avgfptslast3": "avg_fpts_last3",
        "avgfptslast5": "avg_fpts_last5",
        "avgfptslast7": "avg_fpts_last7",
        "avgfptsseason": "avg_fpts_season",
        "usagerate": "usage_rate",
        "salary": "salary",
        "dksalary": "salary",
        "supplementpriority": "supplement_priority",
    }
    resolved_rename: dict[str, str] = {}
    for alias, dest in rename_aliases.items():
        source = normalized_columns.get(alias)
        if source:
            resolved_rename[source] = dest
    if resolved_rename:
        out = out.rename(columns=resolved_rename)

    for col in columns:
        if col not in out.columns:
            out[col] = ""

    out["player_name"] = out["player_name"].map(_clean_rotowire_player_name)
    out["team_abbr"] = out["team_abbr"].map(_normalize_rotowire_team_abbr)
    out["opp_abbr"] = out["opp_abbr"].map(_normalize_rotowire_team_abbr)
    for col in [
        "salary",
        "proj_fantasy_points",
        "proj_minutes",
        "proj_value_per_1k",
        "avg_fpts_last3",
        "avg_fpts_last5",
        "avg_fpts_last7",
        "avg_fpts_season",
        "usage_rate",
        "supplement_priority",
    ]:
        out[col] = out[col].map(_lineupstarter_numeric)
    out["proj_minutes"] = out["proj_minutes"].where(
        pd.to_numeric(out["proj_minutes"], errors="coerce").fillna(0.0) > 0.0,
        pd.NA,
    )
    out["proj_fantasy_points"] = out["proj_fantasy_points"].where(
        out["proj_fantasy_points"].notna(),
        (out["proj_value_per_1k"] * out["salary"]) / 1000.0,
    )

    out = out.loc[
        (out["player_name"] != "")
        & (
            out["proj_fantasy_points"].notna()
            | out["proj_minutes"].notna()
        )
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)

    out["proj_value_per_1k"] = out["proj_value_per_1k"].where(
        out["proj_value_per_1k"].notna(),
        (out["proj_fantasy_points"] / out["salary"].replace(0.0, pd.NA)) * 1000.0,
    )
    out = out.sort_values(
        ["player_name", "team_abbr", "supplement_priority", "proj_fantasy_points", "proj_minutes"],
        ascending=[True, True, False, False, False],
        kind="stable",
    )
    with_team = out.loc[out["team_abbr"] != ""].drop_duplicates(subset=["player_name", "team_abbr"], keep="first")
    # Blank-team uploads are inherently ambiguous, so only collapse rows that are
    # fully identical after normalization instead of assuming name-only uniqueness.
    without_team = out.loc[out["team_abbr"] == ""].drop_duplicates(subset=columns, keep="first")
    out = pd.concat([with_team, without_team], ignore_index=True, sort=False)
    out = out.sort_values(
        ["player_name", "team_abbr", "supplement_priority", "proj_fantasy_points", "proj_minutes"],
        ascending=[True, True, False, False, False],
        kind="stable",
    )
    return out[columns].reset_index(drop=True)


def _lineupstarter_position_tokens(value: Any) -> set[str]:
    raw = str(value or "").strip().upper()
    tokens: set[str] = set()
    if "G" in raw:
        tokens.add("G")
    if "F" in raw:
        tokens.add("F")
    if "C" in raw:
        tokens.add("C")
    return tokens


def _prepare_lineupstarter_slate_reference(slate_df: pd.DataFrame | None) -> pd.DataFrame:
    if slate_df is None or slate_df.empty:
        return pd.DataFrame(
            columns=[
                "ID",
                "Name",
                "Name + ID",
                "TeamAbbrev",
                "Position",
                "Roster Position",
                "Salary",
                "Game Info",
                "name_key",
                "name_key_loose",
                "position_base",
            ]
        )

    ref = slate_df.copy()
    empty_series = pd.Series([""] * len(ref), index=ref.index, dtype="object")
    ref["ID"] = ref.get("ID", empty_series).map(_normalize_player_id)
    ref["Name"] = ref.get("Name", empty_series).astype(str).str.strip()
    ref["Name + ID"] = ref.get("Name + ID", empty_series).astype(str).str.strip()
    ref["Name + ID"] = ref["Name + ID"].where(
        ref["Name + ID"] != "",
        ref["Name"] + " (" + ref["ID"] + ")",
    )
    ref["TeamAbbrev"] = ref.get("TeamAbbrev", empty_series).astype(str).str.strip().str.upper()
    ref["Position"] = ref.get("Position", empty_series).astype(str).str.strip().str.upper()
    ref["Roster Position"] = ref.get("Roster Position", empty_series).astype(str).str.strip().str.upper()
    ref["Salary"] = pd.to_numeric(ref.get("Salary"), errors="coerce")
    ref["Game Info"] = ref.get("Game Info", empty_series).astype(str).str.strip()
    ref["name_key"] = ref["Name"].map(_norm_name_key)
    ref["name_key_loose"] = ref["Name"].map(_norm_name_key_loose)
    ref["position_base"] = ref["Position"].astype(str).str.strip().str.upper().str[:1]
    ref = ref.loc[(ref["ID"] != "") & (ref["Name"] != "")]
    return ref[
        [
            "ID",
            "Name",
            "Name + ID",
            "TeamAbbrev",
            "Position",
            "Roster Position",
            "Salary",
            "Game Info",
            "name_key",
            "name_key_loose",
            "position_base",
        ]
    ].reset_index(drop=True)


def _filter_lineupstarter_candidates(
    candidates: pd.DataFrame,
    source_team_abbr: str,
    source_position_tokens: set[str],
) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out

    if source_team_abbr:
        team_match = out["TeamAbbrev"].astype(str).str.upper() == source_team_abbr
        if bool(team_match.any()):
            out = out.loc[team_match].copy()

    if source_position_tokens:
        pos_match = out["position_base"].astype(str).str.upper().isin(source_position_tokens)
        if bool(pos_match.any()):
            out = out.loc[pos_match].copy()

    return out


def normalize_lineupstarter_upload_frame(
    df: pd.DataFrame | None,
    slate_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    empty_out = pd.DataFrame(
        columns=[
            "ID",
            "Name",
            "Name + ID",
            "TeamAbbrev",
            "Position",
            "Roster Position",
            "Salary",
            "Game Info",
            "lineupstarter_projected_points",
            "lineupstarter_projected_ownership",
            "lineupstarter_source_player_name",
            "lineupstarter_source_team_abbr",
            "lineupstarter_source_position",
            "lineupstarter_match_method",
            "lineupstarter_match_status",
        ]
    )
    empty_meta = {
        "rows_total": 0,
        "matched_rows": 0,
        "unresolved_rows": 0,
        "coverage_pct": 0.0,
        "fully_resolved": False,
    }
    if df is None or df.empty:
        return empty_out, empty_meta

    out = df.copy()
    normalized_columns = {re.sub(r"[^a-z0-9]", "", str(c).strip().lower()): c for c in out.columns}
    rename_aliases = {
        "player": "lineupstarter_source_player_name",
        "playername": "lineupstarter_source_player_name",
        "name": "lineupstarter_source_player_name",
        "athlete": "lineupstarter_source_player_name",
        "team": "lineupstarter_source_team_abbr",
        "teamabbr": "lineupstarter_source_team_abbr",
        "teamabbrev": "lineupstarter_source_team_abbr",
        "position": "lineupstarter_source_position",
        "pos": "lineupstarter_source_position",
        "salary": "Salary",
        "proj": "lineupstarter_projected_points",
        "projection": "lineupstarter_projected_points",
        "projectedpoints": "lineupstarter_projected_points",
        "projectedfantasypoints": "lineupstarter_projected_points",
        "fantasyprojection": "lineupstarter_projected_points",
        "projown": "lineupstarter_projected_ownership",
        "projownpct": "lineupstarter_projected_ownership",
        "projownership": "lineupstarter_projected_ownership",
        "projectedownership": "lineupstarter_projected_ownership",
        "projectedown": "lineupstarter_projected_ownership",
        "projectedownpct": "lineupstarter_projected_ownership",
        "dkid": "uploaded_dk_id",
        "dk_id": "uploaded_dk_id",
        "draftkingsid": "uploaded_dk_id",
        "id": "uploaded_dk_id",
        "dkplayername": "uploaded_dk_player_name",
        "dk_player_name": "uploaded_dk_player_name",
        "dknameplusid": "uploaded_dk_name_plus_id",
        "dk_name_plus_id": "uploaded_dk_name_plus_id",
        "dkteamabbr": "uploaded_dk_team_abbr",
        "dk_team_abbr": "uploaded_dk_team_abbr",
        "dksalary": "uploaded_dk_salary",
        "dk_salary": "uploaded_dk_salary",
        "dkgamekey": "uploaded_dk_game_key",
        "dk_game_key": "uploaded_dk_game_key",
    }
    resolved_rename: dict[str, str] = {}
    for alias, dest in rename_aliases.items():
        source = normalized_columns.get(alias)
        if source:
            resolved_rename[source] = dest
    if resolved_rename:
        out = out.rename(columns=resolved_rename)

    for col in [
        "lineupstarter_source_player_name",
        "lineupstarter_source_team_abbr",
        "lineupstarter_source_position",
        "Salary",
        "lineupstarter_projected_points",
        "lineupstarter_projected_ownership",
        "uploaded_dk_id",
        "uploaded_dk_player_name",
        "uploaded_dk_name_plus_id",
        "uploaded_dk_team_abbr",
        "uploaded_dk_salary",
        "uploaded_dk_game_key",
    ]:
        if col not in out.columns:
            out[col] = ""

    out["lineupstarter_source_player_name"] = out["lineupstarter_source_player_name"].astype(str).str.strip()
    out["lineupstarter_source_team_abbr"] = out["lineupstarter_source_team_abbr"].astype(str).str.strip().str.upper()
    out["lineupstarter_source_position"] = out["lineupstarter_source_position"].astype(str).str.strip().str.upper()
    out["Salary"] = out["Salary"].map(_lineupstarter_numeric)
    out["lineupstarter_projected_points"] = out["lineupstarter_projected_points"].map(_lineupstarter_numeric)
    out["lineupstarter_projected_ownership"] = out["lineupstarter_projected_ownership"].map(_lineupstarter_numeric)
    out["uploaded_dk_id"] = out["uploaded_dk_id"].map(_normalize_player_id)
    out["uploaded_dk_player_name"] = out["uploaded_dk_player_name"].astype(str).str.strip()
    out["uploaded_dk_name_plus_id"] = out["uploaded_dk_name_plus_id"].astype(str).str.strip()
    out["uploaded_dk_team_abbr"] = out["uploaded_dk_team_abbr"].astype(str).str.strip().str.upper()
    out["uploaded_dk_salary"] = out["uploaded_dk_salary"].map(_lineupstarter_numeric)
    out["uploaded_dk_game_key"] = out["uploaded_dk_game_key"].astype(str).str.strip().str.upper()
    out["name_key"] = out["lineupstarter_source_player_name"].map(_norm_name_key)
    out["name_key_loose"] = out["lineupstarter_source_player_name"].map(_norm_name_key_loose)
    out = out.loc[out["lineupstarter_source_player_name"] != ""].copy()
    if out.empty:
        return empty_out, empty_meta

    ref = _prepare_lineupstarter_slate_reference(slate_df)
    records: list[dict[str, Any]] = []
    matched_rows = 0

    for row in out.to_dict(orient="records"):
        source_team_abbr = str(row.get("lineupstarter_source_team_abbr") or "").strip().upper()
        source_position = str(row.get("lineupstarter_source_position") or "").strip().upper()
        source_position_tokens = _lineupstarter_position_tokens(source_position)
        salary = _lineupstarter_numeric(row.get("Salary"))
        uploaded_dk_id = _normalize_player_id(row.get("uploaded_dk_id"))
        match_method = ""
        match_status = "unresolved"
        candidate = pd.DataFrame()

        if uploaded_dk_id and not ref.empty:
            candidate = ref.loc[ref["ID"] == uploaded_dk_id].copy()
            if salary is not None:
                candidate = candidate.loc[pd.to_numeric(candidate["Salary"], errors="coerce") == float(salary)].copy()
            candidate = _filter_lineupstarter_candidates(candidate, source_team_abbr, source_position_tokens)
            if len(candidate) == 1:
                match_method = "uploaded_dk_id_salary"
                match_status = "matched"

        if match_status != "matched" and salary is not None and not ref.empty:
            exact_candidate = ref.loc[
                (ref["name_key"] == str(row.get("name_key") or ""))
                & (pd.to_numeric(ref["Salary"], errors="coerce") == float(salary))
            ].copy()
            exact_candidate = _filter_lineupstarter_candidates(exact_candidate, source_team_abbr, source_position_tokens)
            if len(exact_candidate) == 1:
                candidate = exact_candidate
                match_method = "exact_name_salary"
                match_status = "matched"

        if match_status != "matched" and salary is not None and not ref.empty:
            loose_candidate = ref.loc[
                (ref["name_key_loose"] == str(row.get("name_key_loose") or ""))
                & (pd.to_numeric(ref["Salary"], errors="coerce") == float(salary))
            ].copy()
            loose_candidate = _filter_lineupstarter_candidates(loose_candidate, source_team_abbr, source_position_tokens)
            if len(loose_candidate) == 1:
                candidate = loose_candidate
                match_method = "loose_name_salary"
                match_status = "matched"

        if match_status == "matched" and not candidate.empty:
            matched_rows += 1
            best = candidate.iloc[0]
            records.append(
                {
                    "ID": _normalize_player_id(best.get("ID")),
                    "Name": str(best.get("Name") or "").strip(),
                    "Name + ID": str(best.get("Name + ID") or "").strip(),
                    "TeamAbbrev": str(best.get("TeamAbbrev") or "").strip().upper(),
                    "Position": str(best.get("Position") or "").strip().upper(),
                    "Roster Position": str(best.get("Roster Position") or "").strip().upper(),
                    "Salary": _lineupstarter_numeric(best.get("Salary")),
                    "Game Info": str(best.get("Game Info") or "").strip(),
                    "lineupstarter_projected_points": _lineupstarter_numeric(row.get("lineupstarter_projected_points")),
                    "lineupstarter_projected_ownership": _lineupstarter_numeric(row.get("lineupstarter_projected_ownership")),
                    "lineupstarter_source_player_name": str(row.get("lineupstarter_source_player_name") or "").strip(),
                    "lineupstarter_source_team_abbr": source_team_abbr,
                    "lineupstarter_source_position": source_position,
                    "lineupstarter_match_method": match_method,
                    "lineupstarter_match_status": match_status,
                }
            )
        else:
            records.append(
                {
                    "ID": "",
                    "Name": "",
                    "Name + ID": "",
                    "TeamAbbrev": "",
                    "Position": "",
                    "Roster Position": "",
                    "Salary": salary,
                    "Game Info": "",
                    "lineupstarter_projected_points": _lineupstarter_numeric(row.get("lineupstarter_projected_points")),
                    "lineupstarter_projected_ownership": _lineupstarter_numeric(row.get("lineupstarter_projected_ownership")),
                    "lineupstarter_source_player_name": str(row.get("lineupstarter_source_player_name") or "").strip(),
                    "lineupstarter_source_team_abbr": source_team_abbr,
                    "lineupstarter_source_position": source_position,
                    "lineupstarter_match_method": "",
                    "lineupstarter_match_status": "unresolved",
                }
            )

    normalized = pd.DataFrame(records)
    rows_total = int(len(normalized))
    unresolved_rows = int(rows_total - matched_rows)
    coverage_pct = (100.0 * float(matched_rows) / float(rows_total)) if rows_total else 0.0
    coverage = {
        "rows_total": rows_total,
        "matched_rows": int(matched_rows),
        "unresolved_rows": int(unresolved_rows),
        "coverage_pct": round(coverage_pct, 2),
        "fully_resolved": bool(rows_total > 0 and matched_rows == rows_total),
    }
    return normalized, coverage


def normalize_ownership_upload_frame(df: pd.DataFrame | None) -> pd.DataFrame:
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
    out["actual_ownership"] = out["actual_ownership"].map(_ownership_to_float)
    out["name_key"] = out["Name"].map(_norm_name_key)
    out["name_key_loose"] = out["Name"].map(_norm_name_key_loose)
    out = out.loc[(out["ID"] != "") | (out["Name"] != "")]
    out = out.sort_values(["ID", "Name"], ascending=[True, True]).drop_duplicates(
        subset=["ID", "name_key", "TeamAbbrev"], keep="last"
    )
    return out[["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"]].reset_index(drop=True)


def _read_ownership_frame(
    store: CbbGcsStore,
    selected_date: date,
    slate_key: str | None = None,
) -> pd.DataFrame:
    csv_text = store.read_ownership_csv(selected_date, slate_key=slate_key)
    if not csv_text or not str(csv_text).strip():
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"])
    return normalize_ownership_upload_frame(_read_csv_frame(csv_text))


def import_lineupstarter_projection_csv(
    *,
    csv_bytes: bytes,
    resolved_slate_df: pd.DataFrame,
    selected_date: date | str,
    slate_key: str | None = "main",
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    _, frame = _decode_csv_payload(csv_bytes)
    normalized, coverage = normalize_lineupstarter_upload_frame(frame, slate_df=resolved_slate_df)
    if normalized.empty:
        raise ValueError("Could not find LineupStarter rows. Include player, salary, projection, and projected ownership columns.")
    if not bool(coverage.get("fully_resolved")):
        unresolved = normalized.loc[normalized["lineupstarter_match_status"] != "matched"].copy()
        sample_names = unresolved["lineupstarter_source_player_name"].astype(str).head(10).tolist()
        raise ValueError(
            "LineupStarter mapping incomplete: "
            f"{int(coverage.get('unresolved_rows') or 0)} unresolved rows. "
            f"Examples: {sample_names}"
        )

    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    blob_name = store.write_lineupstarter_csv(game_date, normalized.to_csv(index=False), slate_key=slate_key)
    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "rows_saved": int(len(normalized)),
        "players": int(normalized["Name"].nunique()),
        "blob_name": blob_name,
        "coverage": coverage,
    }


def load_lineupstarter_projection_frame(
    *,
    selected_date: date | str,
    slate_key: str | None = "main",
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> pd.DataFrame:
    game_date = _to_iso_date(selected_date)
    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    return _read_lineupstarter_frame(store, game_date, slate_key=slate_key)


def _read_contest_standings_frame(
    store: CbbGcsStore,
    selected_date: date,
    contest_id: str,
    slate_key: str | None = None,
) -> pd.DataFrame:
    csv_text = store.read_contest_standings_csv(selected_date, contest_id, slate_key=slate_key)
    return _read_csv_frame(csv_text)


def _read_dk_slate_frame(
    store: CbbGcsStore,
    selected_date: date,
    slate_key: str | None = None,
) -> pd.DataFrame:
    csv_text = store.read_dk_slate_csv(selected_date, slate_key=slate_key)
    return _read_csv_frame(csv_text)


def _attach_actual_ownership(review_df: pd.DataFrame, own_df: pd.DataFrame) -> pd.DataFrame:
    if review_df.empty:
        return review_df
    out = review_df.copy()
    out["actual_ownership"] = pd.NA
    if own_df is None or own_df.empty:
        return out

    own_lookup = own_df.copy()
    if "ID" in out.columns and "ID" in own_lookup.columns:
        out["ID"] = out["ID"].map(_normalize_player_id)
        own_lookup["ID"] = own_lookup["ID"].map(_normalize_player_id)
        by_id = own_lookup.loc[own_lookup["ID"] != "", ["ID", "actual_ownership"]].dropna(subset=["actual_ownership"])
        if not by_id.empty:
            by_id = by_id.drop_duplicates("ID").rename(columns={"actual_ownership": "actual_ownership_id"})
            out = out.merge(by_id, on="ID", how="left")
            out["actual_ownership"] = pd.to_numeric(out.get("actual_ownership_id"), errors="coerce")

    if "Name" in out.columns and "Name" in own_lookup.columns:
        out["name_key"] = out["Name"].map(_norm_name_key)
        out["name_key_loose"] = out["Name"].map(_norm_name_key_loose)
        own_lookup["name_key"] = own_lookup["Name"].map(_norm_name_key)
        own_lookup["name_key_loose"] = own_lookup["Name"].map(_norm_name_key_loose)

        by_name = (
            own_lookup.loc[own_lookup["name_key"] != "", ["name_key", "actual_ownership"]]
            .dropna(subset=["actual_ownership"])
            .drop_duplicates("name_key")
            .rename(columns={"actual_ownership": "actual_ownership_name"})
        )
        if not by_name.empty:
            out = out.merge(by_name, on="name_key", how="left")

        by_name_loose = (
            own_lookup.loc[own_lookup["name_key_loose"] != "", ["name_key_loose", "actual_ownership"]]
            .dropna(subset=["actual_ownership"])
            .drop_duplicates("name_key_loose")
            .rename(columns={"actual_ownership": "actual_ownership_name_loose"})
        )
        if not by_name_loose.empty:
            out = out.merge(by_name_loose, on="name_key_loose", how="left")

        out["actual_ownership"] = pd.to_numeric(out.get("actual_ownership"), errors="coerce").where(
            pd.to_numeric(out.get("actual_ownership"), errors="coerce").notna(),
            pd.to_numeric(out.get("actual_ownership_name"), errors="coerce"),
        )
        out["actual_ownership"] = pd.to_numeric(out.get("actual_ownership"), errors="coerce").where(
            pd.to_numeric(out.get("actual_ownership"), errors="coerce").notna(),
            pd.to_numeric(out.get("actual_ownership_name_loose"), errors="coerce"),
        )

    return out.drop(
        columns=["actual_ownership_id", "actual_ownership_name", "actual_ownership_name_loose", "name_key", "name_key_loose"],
        errors="ignore",
    )

def build_injuries_review_payload(
    *,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    row_limit: int = 300,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )

    feed_df = normalize_injuries_frame(_read_csv_frame(store.read_injuries_feed_csv(game_date)))
    manual_df = normalize_injuries_frame(_read_csv_frame(store.read_injuries_manual_csv()))

    legacy_fallback_used = False
    effective_df = pd.DataFrame()
    if feed_df.empty and manual_df.empty:
        legacy_df = normalize_injuries_frame(_read_csv_frame(store.read_injuries_csv()))
        if not legacy_df.empty:
            legacy_fallback_used = True
            effective_df = legacy_df.copy()
    else:
        frames: list[pd.DataFrame] = []
        if not feed_df.empty:
            one = feed_df.copy()
            one["_source"] = "feed"
            frames.append(one)
        if not manual_df.empty:
            one = manual_df.copy()
            one["_source"] = "manual"
            frames.append(one)
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            combined["_injury_key"] = combined.apply(
                lambda r: f"{str(r.get('player_name') or '').strip().lower()}|{str(r.get('team') or '').strip().lower()}",
                axis=1,
            )
            combined = combined.drop_duplicates(subset=["_injury_key"], keep="last")
            effective_df = combined.drop(columns=["_injury_key", "_source"], errors="ignore")

    if effective_df.empty:
        effective_df = normalize_injuries_frame(None)
    else:
        effective_df = normalize_injuries_frame(effective_df)

    status_counts_df = (
        effective_df.assign(status=effective_df.get("status", "").astype(str).str.strip().str.lower())
        .groupby("status", as_index=False)
        .agg(
            rows=("player_name", "count"),
            active=("active", lambda s: int(pd.Series(s).astype(bool).sum())),
        )
        .sort_values(["rows", "status"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
        if not effective_df.empty
        else pd.DataFrame(columns=["status", "rows", "active"])
    )
    remove_candidates = (
        effective_df.loc[
            effective_df.get("active", False).astype(bool)
            & effective_df.get("status", "").astype(str).str.strip().str.lower().isin({"out", "doubtful"})
        ].copy()
        if not effective_df.empty
        else pd.DataFrame()
    )

    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "available": bool(not effective_df.empty),
        "legacy_fallback_used": bool(legacy_fallback_used),
        "summary": {
            "effective_rows": int(len(effective_df)),
            "feed_rows": int(len(feed_df)),
            "manual_rows": int(len(manual_df)),
            "active_rows": int(pd.Series(effective_df.get("active", False)).astype(bool).sum()) if not effective_df.empty else 0,
            "remove_candidates": int(len(remove_candidates)),
        },
        "status_counts": _serialize_records(status_counts_df),
        "effective_rows": _serialize_records(effective_df, limit=row_limit),
        "feed_rows": _serialize_records(feed_df, limit=row_limit),
        "manual_rows": _serialize_records(manual_df, limit=row_limit),
    }


def import_injuries_manual_csv(
    *,
    csv_bytes: bytes,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    _, frame = _decode_csv_payload(csv_bytes)
    normalized = normalize_injuries_frame(frame)
    if normalized.empty:
        raise ValueError("Could not parse injuries rows. Include player, team, and status columns.")

    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    blob_name = store.write_injuries_manual_csv(normalized.to_csv(index=False))
    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "rows_saved": int(len(normalized)),
        "blob_name": blob_name,
        "status_counts": _serialize_records(
            normalized.groupby("status", as_index=False).agg(rows=("player_name", "count")).sort_values(
                ["rows", "status"], ascending=[False, True], kind="stable"
            )
        ),
    }


def import_injuries_feed_csv(
    *,
    csv_bytes: bytes,
    selected_date: date | str,
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    _, frame = _decode_csv_payload(csv_bytes)
    normalized = normalize_injuries_frame(frame)
    if normalized.empty:
        raise ValueError("Could not parse injuries rows. Include player, team, and status columns.")

    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    blob_name = store.write_injuries_feed_csv(normalized.to_csv(index=False), game_date=game_date)
    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "rows_saved": int(len(normalized)),
        "blob_name": blob_name,
        "status_counts": _serialize_records(
            normalized.groupby("status", as_index=False).agg(rows=("player_name", "count")).sort_values(
                ["rows", "status"], ascending=[False, True], kind="stable"
            )
        ),
    }


def import_projection_ownership_csv(
    *,
    csv_bytes: bytes,
    selected_date: date | str,
    slate_key: str | None = "main",
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    _, frame = _decode_csv_payload(csv_bytes)
    normalized = normalize_ownership_upload_frame(frame)
    if normalized.empty:
        raise ValueError("Could not find ownership rows. Include player ID/name and ownership columns.")

    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    blob_name = store.write_ownership_csv(game_date, normalized.to_csv(index=False), slate_key=slate_key)
    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "rows_saved": int(len(normalized)),
        "players": int(normalized["Name"].nunique()),
        "blob_name": blob_name,
    }


def build_projection_review_payload(
    *,
    selected_date: date | str,
    slate_key: str | None = "main",
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    row_limit: int = 500,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    proj_df = _read_projection_snapshot_frame(store, game_date, slate_key=slate_key)
    if proj_df.empty:
        return {
            "selected_date": game_date.isoformat(),
            "bucket_name": store.bucket_name,
            "available": False,
            "summary": {
                "projection_rows": 0,
                "actual_matched": 0,
                "blend_mae": None,
                "our_mae": None,
                "vegas_mae": None,
                "ownership_rows": 0,
                "ownership_mae": None,
                "ownership_bias": None,
                "ownership_rank_spearman": None,
            },
            "rows": [],
        }

    actual_df = _read_actual_results_frame(store, game_date)
    own_df = _read_ownership_frame(store, game_date, slate_key=slate_key)

    review = proj_df.copy()
    if "ID" in review.columns:
        review["ID"] = review["ID"].map(_normalize_player_id)
    if "Name" in review.columns:
        review["Name"] = review["Name"].astype(str).str.strip()
    if "TeamAbbrev" in review.columns:
        review["TeamAbbrev"] = review["TeamAbbrev"].astype(str).str.strip().str.upper()

    if not actual_df.empty and "ID" in review.columns:
        review = review.merge(actual_df, on="ID", how="left", suffixes=("", "_actual"))
    elif not actual_df.empty and "Name" in review.columns:
        review = review.merge(actual_df, on="Name", how="left", suffixes=("", "_actual"))

    review = _attach_actual_ownership(review, own_df)

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
        review["blended_projection"] = pd.to_numeric(review["projected_dk_points"], errors="coerce")

    review["blend_error"] = pd.to_numeric(review.get("actual_dk_points"), errors="coerce") - pd.to_numeric(
        review.get("blended_projection"), errors="coerce"
    )
    review["our_error"] = pd.to_numeric(review.get("actual_dk_points"), errors="coerce") - pd.to_numeric(
        review.get("our_dk_projection"), errors="coerce"
    )
    review["vegas_error"] = pd.to_numeric(review.get("actual_dk_points"), errors="coerce") - pd.to_numeric(
        review.get("vegas_dk_projection"), errors="coerce"
    )
    review["ownership_error"] = pd.to_numeric(review.get("actual_ownership"), errors="coerce") - pd.to_numeric(
        review.get("projected_ownership"), errors="coerce"
    )

    matched_actual = int(pd.to_numeric(review.get("actual_dk_points"), errors="coerce").notna().sum())
    own_rows = int(pd.to_numeric(review.get("actual_ownership"), errors="coerce").notna().sum())

    own_rank_corr: float | None = None
    matched_own = review.loc[
        pd.to_numeric(review.get("actual_ownership"), errors="coerce").notna()
        & pd.to_numeric(review.get("projected_ownership"), errors="coerce").notna()
    ].copy()
    if len(matched_own) >= 3:
        try:
            corr = matched_own[["projected_ownership", "actual_ownership"]].corr(method="spearman", numeric_only=True).iloc[0, 1]
            corr_num = pd.to_numeric(corr, errors="coerce")
            if pd.notna(corr_num):
                own_rank_corr = float(corr_num)
        except Exception:
            own_rank_corr = None

    show_cols = [
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
    view_df = review[[c for c in show_cols if c in review.columns]].copy()
    sort_cols = [c for c in ["actual_dk_points", "blended_projection"] if c in view_df.columns]
    if sort_cols:
        view_df = view_df.sort_values(sort_cols, ascending=False, na_position="last")

    return {
        "selected_date": game_date.isoformat(),
        "bucket_name": store.bucket_name,
        "available": True,
        "summary": {
            "projection_rows": int(len(review)),
            "actual_matched": matched_actual,
            "blend_mae": float(pd.to_numeric(review["blend_error"], errors="coerce").abs().mean()) if matched_actual else None,
            "our_mae": float(pd.to_numeric(review["our_error"], errors="coerce").abs().mean()) if matched_actual else None,
            "vegas_mae": float(pd.to_numeric(review["vegas_error"], errors="coerce").abs().mean()) if matched_actual else None,
            "ownership_rows": own_rows,
            "ownership_mae": float(pd.to_numeric(review["ownership_error"], errors="coerce").abs().mean()) if own_rows else None,
            "ownership_bias": float(pd.to_numeric(review["ownership_error"], errors="coerce").mean()) if own_rows else None,
            "ownership_rank_spearman": own_rank_corr,
        },
        "rows": _serialize_records(view_df, limit=row_limit),
    }

def import_contest_standings_csv(
    *,
    csv_bytes: bytes,
    selected_date: date | str,
    contest_id: str,
    slate_key: str | None = "main",
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    csv_text, frame = _decode_csv_payload(csv_bytes)
    if frame.empty:
        raise ValueError("Could not parse uploaded standings CSV.")
    profile = detect_contest_standings_upload(frame)
    if str(profile.get("kind") or "") != "contest_standings":
        raise ValueError(str(profile.get("message") or "Unrecognized tournament standings CSV format."))

    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    blob_name = store.write_contest_standings_csv(game_date, contest_id, csv_text, slate_key=slate_key)
    return {
        "selected_date": game_date.isoformat(),
        "contest_id": str(contest_id),
        "bucket_name": store.bucket_name,
        "blob_name": blob_name,
        "profile": profile,
    }


def build_tournament_review_payload(
    *,
    selected_date: date | str,
    contest_id: str,
    slate_key: str | None = "main",
    bucket_name: str | None = None,
    gcp_project: str | None = None,
    service_account_json: str | None = None,
    service_account_json_b64: str | None = None,
    entries_limit: int = 200,
    exposure_limit: int = 250,
) -> dict[str, Any]:
    game_date = _to_iso_date(selected_date)
    if not str(contest_id or "").strip():
        raise ValueError("contest_id is required")

    store = _build_api_store(
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    standings_df = _read_contest_standings_frame(store, game_date, contest_id, slate_key=slate_key)
    if standings_df.empty:
        return {
            "selected_date": game_date.isoformat(),
            "contest_id": str(contest_id),
            "bucket_name": store.bucket_name,
            "available": False,
            "message": "No contest standings loaded. Upload or save a standings CSV first.",
            "upload_profile": detect_contest_standings_upload(None),
            "summary": {},
            "entries_rows": [],
            "exposure_rows": [],
            "ownership_buckets": [],
            "ownership_top_misses": [],
            "projection_rows": [],
        }

    upload_profile = detect_contest_standings_upload(standings_df)
    normalized_standings = normalize_contest_standings_frame(standings_df)
    slate_df = _read_dk_slate_frame(store, game_date, slate_key=slate_key)
    proj_df = _read_projection_snapshot_frame(store, game_date, slate_key=slate_key)
    actual_df = _read_actual_results_frame(store, game_date)

    entries_df, expanded_df = build_field_entries_and_players(normalized_standings, slate_df)
    if not entries_df.empty:
        entries_df = build_entry_actual_points_comparison(
            entry_summary_df=entries_df,
            expanded_players_df=expanded_df,
            actual_results_df=actual_df,
        )

    actual_own_df = extract_actual_ownership_from_standings(normalized_standings)
    exposure_df = build_player_exposure_comparison(
        expanded_players_df=expanded_df,
        entry_count=int(len(entries_df)),
        projection_df=proj_df,
        actual_ownership_df=actual_own_df,
        actual_results_df=actual_df,
    )
    own_diag = build_ownership_projection_diagnostics(exposure_df)
    ownership_teacher_review = build_ownership_teacher_review(exposure_df)
    proj_compare_df = build_projection_actual_comparison(projection_df=proj_df, actual_results_df=actual_df)

    projection_summary = {
        "matched_players": int(pd.to_numeric(proj_compare_df.get("actual_dk_points"), errors="coerce").notna().sum())
        if not proj_compare_df.empty
        else 0,
        "blend_mae": float(pd.to_numeric(proj_compare_df.get("blend_error"), errors="coerce").abs().mean())
        if not proj_compare_df.empty
        else None,
        "our_mae": float(pd.to_numeric(proj_compare_df.get("our_error"), errors="coerce").abs().mean())
        if not proj_compare_df.empty
        else None,
        "vegas_mae": float(pd.to_numeric(proj_compare_df.get("vegas_error"), errors="coerce").abs().mean())
        if not proj_compare_df.empty
        else None,
        "minutes_mae_avg": float(pd.to_numeric(proj_compare_df.get("minutes_error_avg"), errors="coerce").abs().mean())
        if not proj_compare_df.empty
        else None,
        "minutes_mae_last7": float(pd.to_numeric(proj_compare_df.get("minutes_error_last7"), errors="coerce").abs().mean())
        if not proj_compare_df.empty
        else None,
    }

    top_entry_df = entries_df.nsmallest(10, "Rank") if (not entries_df.empty and "Rank" in entries_df.columns) else pd.DataFrame()
    summary = {
        "field_entries": int(len(entries_df)),
        "avg_salary_left": float(pd.to_numeric(entries_df.get("salary_left"), errors="coerce").mean())
        if not entries_df.empty and "salary_left" in entries_df.columns
        else None,
        "avg_max_team_stack": float(pd.to_numeric(entries_df.get("max_team_stack"), errors="coerce").mean())
        if not entries_df.empty and "max_team_stack" in entries_df.columns
        else None,
        "avg_max_game_stack": float(pd.to_numeric(entries_df.get("max_game_stack"), errors="coerce").mean())
        if not entries_df.empty and "max_game_stack" in entries_df.columns
        else None,
        "top10_avg_salary_left": float(pd.to_numeric(top_entry_df.get("salary_left"), errors="coerce").mean())
        if not top_entry_df.empty and "salary_left" in top_entry_df.columns
        else None,
        "top10_entries": int(len(top_entry_df)),
        "exposure_rows": int(len(exposure_df)),
        "ownership_samples": int((own_diag.get("summary") or {}).get("samples") or 0),
        "ownership_teacher_samples": int((ownership_teacher_review.get("summary") or {}).get("samples") or 0),
    }

    entry_cols = [
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
    entries_view_df = entries_df[[c for c in entry_cols if c in entries_df.columns]].copy() if not entries_df.empty else pd.DataFrame()

    exposure_cols = [
        "Name",
        "TeamAbbrev",
        "final_dk_points",
        "high_points_low_own_flag",
        "appearances",
        "field_ownership_pct",
        "projected_ownership",
        "actual_ownership_from_file",
        "ownership_diff_vs_proj",
        "blended_projection",
        "our_dk_projection",
        "vegas_dk_projection",
        "blend_error",
    ]
    exposure_view_df = (
        exposure_df[[c for c in exposure_cols if c in exposure_df.columns]].copy()
        if isinstance(exposure_df, pd.DataFrame) and not exposure_df.empty
        else pd.DataFrame()
    )

    own_buckets_df = own_diag.get("buckets_df") if isinstance(own_diag, dict) else pd.DataFrame()
    own_misses_df = own_diag.get("top_misses_df") if isinstance(own_diag, dict) else pd.DataFrame()
    own_teacher_rows_df = ownership_teacher_review.get("matched_df") if isinstance(ownership_teacher_review, dict) else pd.DataFrame()
    own_teacher_segments_df = (
        ownership_teacher_review.get("segments_df") if isinstance(ownership_teacher_review, dict) else pd.DataFrame()
    )
    own_teacher_help_df = (
        ownership_teacher_review.get("top_teacher_help_df") if isinstance(ownership_teacher_review, dict) else pd.DataFrame()
    )
    own_teacher_hurt_df = (
        ownership_teacher_review.get("top_teacher_hurt_df") if isinstance(ownership_teacher_review, dict) else pd.DataFrame()
    )

    return {
        "selected_date": game_date.isoformat(),
        "contest_id": str(contest_id),
        "bucket_name": store.bucket_name,
        "available": True,
        "upload_profile": upload_profile,
        "summary": summary,
        "ownership_summary": own_diag.get("summary") if isinstance(own_diag, dict) else {},
        "ownership_teacher_summary": (
            ownership_teacher_review.get("summary") if isinstance(ownership_teacher_review, dict) else {}
        ),
        "projection_summary": projection_summary,
        "entries_rows": _serialize_records(entries_view_df, limit=entries_limit),
        "exposure_rows": _serialize_records(exposure_view_df, limit=exposure_limit),
        "ownership_buckets": _serialize_records(own_buckets_df),
        "ownership_top_misses": _serialize_records(own_misses_df),
        "ownership_teacher_rows": _serialize_records(own_teacher_rows_df, limit=exposure_limit),
        "ownership_teacher_segments": _serialize_records(own_teacher_segments_df),
        "ownership_teacher_top_help": _serialize_records(own_teacher_help_df),
        "ownership_teacher_top_hurt": _serialize_records(own_teacher_hurt_df),
        "projection_rows": _serialize_records(proj_compare_df, limit=exposure_limit),
    }
