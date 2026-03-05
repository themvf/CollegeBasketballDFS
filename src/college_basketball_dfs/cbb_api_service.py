from __future__ import annotations

import io
from datetime import date
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .cbb_dk_registry import (
    build_dk_identity_registry,
    build_registry_history_from_local_directory,
    build_rotowire_dk_slate,
    derive_manual_overrides_from_dk_slate,
    manual_overrides_to_history_frame,
)
from .cbb_gcs import CbbGcsStore, build_storage_client
from .cbb_rotowire import RotoWireClient, flatten_slates, normalize_players, select_slate
from .cbb_odds import flatten_player_props_payload
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
