from __future__ import annotations

import io
from datetime import date, timedelta
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
    manual_overrides_to_history_frame,
)
from .cbb_dk_optimizer import normalize_injuries_frame
from .cbb_gcs import CbbGcsStore, build_storage_client
from .cbb_rotowire import RotoWireClient, flatten_slates, normalize_players, select_slate
from .cbb_odds import flatten_player_props_payload
from .cbb_tournament_review import (
    build_entry_actual_points_comparison,
    build_field_entries_and_players,
    build_ownership_projection_diagnostics,
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


def load_registry(
    draftkings_dir: str | Path | None = None,
    manual_overrides_path: str | Path | None = None,
) -> pd.DataFrame:
    draftkings_base = Path(draftkings_dir) if draftkings_dir else DEFAULT_DRAFTKINGS_DIR
    history_frames: list[pd.DataFrame] = []

    local_history = build_registry_history_from_local_directory(draftkings_base)
    if local_history is not None and not local_history.empty:
        history_frames.append(local_history)

    manual_df = load_manual_overrides(manual_overrides_path)
    manual_history = manual_overrides_to_history_frame(manual_df)
    if manual_history is not None and not manual_history.empty:
        history_frames.append(manual_history)

    if not history_frames:
        return pd.DataFrame()
    history = pd.concat(history_frames, ignore_index=True)
    if history.empty:
        return pd.DataFrame()
    return build_dk_identity_registry(history)


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
) -> dict[str, Any]:
    rotowire_df, slate_meta = load_rotowire_players_for_slate(
        selected_date=selected_date,
        contest_type=contest_type,
        slate_name=slate_name,
        slate_id=slate_id,
        site_id=site_id,
        cookie_header=cookie_header,
    )
    registry_df = load_registry(draftkings_dir=draftkings_dir, manual_overrides_path=manual_overrides_path)
    _, resolution_df, coverage = build_rotowire_dk_slate(
        rotowire_df=rotowire_df,
        registry_df=registry_df,
        slate_date=pd.to_datetime(selected_date, errors="coerce").date().isoformat(),
        slate_key=slate_key,
    )
    unresolved = resolution_df.loc[resolution_df["dk_resolution_status"] != "resolved"].copy()
    return {
        "slate": slate_meta,
        "coverage": dict(coverage or {}),
        "unresolved_players": unresolved.to_dict(orient="records"),
    }


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
    persist: bool = True,
) -> dict[str, Any]:
    dk_slate_df = pd.read_csv(io.BytesIO(dk_slate_csv_bytes))
    rotowire_df, slate_meta = load_rotowire_players_for_slate(
        selected_date=selected_date,
        contest_type=contest_type,
        slate_name=slate_name,
        slate_id=slate_id,
        site_id=site_id,
        cookie_header=cookie_header,
    )
    registry_before = load_registry(draftkings_dir=draftkings_dir, manual_overrides_path=manual_overrides_path)
    _, resolution_before, coverage_before = build_rotowire_dk_slate(
        rotowire_df=rotowire_df,
        registry_df=registry_before,
        slate_date=pd.to_datetime(selected_date, errors="coerce").date().isoformat(),
        slate_key=slate_key,
    )

    derived_overrides, still_unresolved, derive_meta = derive_manual_overrides_from_dk_slate(
        rotowire_df=rotowire_df,
        resolution_df=resolution_before,
        dk_slate_df=dk_slate_df,
        slate_date=selected_date,
        slate_key=slate_key,
        source_name=f"dk_import:{pd.to_datetime(selected_date, errors='coerce').date().isoformat()}:{slate_key}",
    )

    existing_manual = load_manual_overrides(manual_overrides_path)
    merged_manual = merge_manual_overrides(existing_manual, derived_overrides)
    written_path: str | None = None
    if persist:
        written = write_manual_overrides(merged_manual, manual_overrides_path)
        written_path = str(written)

    registry_after = load_registry(draftkings_dir=draftkings_dir, manual_overrides_path=manual_overrides_path)
    _, resolution_after, coverage_after = build_rotowire_dk_slate(
        rotowire_df=rotowire_df,
        registry_df=registry_after,
        slate_date=pd.to_datetime(selected_date, errors="coerce").date().isoformat(),
        slate_key=slate_key,
    )

    return {
        "slate": slate_meta,
        "coverage_before": dict(coverage_before or {}),
        "coverage_after": dict(coverage_after or {}),
        "derived_override_count": int(len(derived_overrides)),
        "remaining_unresolved_before_count": int((resolution_before["dk_resolution_status"] != "resolved").sum()),
        "remaining_unresolved_after_count": int((resolution_after["dk_resolution_status"] != "resolved").sum()),
        "remaining_unresolved_after": resolution_after.loc[
            resolution_after["dk_resolution_status"] != "resolved",
            ["player_name", "team_abbr", "salary", "dk_resolution_status", "dk_match_reason"],
        ].to_dict(orient="records"),
        "derived_overrides": derived_overrides.to_dict(orient="records"),
        "derivation_meta": dict(derive_meta or {}),
        "persisted_manual_overrides_path": written_path,
        "persisted": bool(persist),
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

    return {
        "selected_date": game_date.isoformat(),
        "contest_id": str(contest_id),
        "bucket_name": store.bucket_name,
        "available": True,
        "upload_profile": upload_profile,
        "summary": summary,
        "ownership_summary": own_diag.get("summary") if isinstance(own_diag, dict) else {},
        "projection_summary": projection_summary,
        "entries_rows": _serialize_records(entries_view_df, limit=entries_limit),
        "exposure_rows": _serialize_records(exposure_view_df, limit=exposure_limit),
        "ownership_buckets": _serialize_records(own_buckets_df),
        "ownership_top_misses": _serialize_records(own_misses_df),
        "projection_rows": _serialize_records(proj_compare_df, limit=exposure_limit),
    }
