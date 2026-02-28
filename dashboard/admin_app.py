from __future__ import annotations

import io
import inspect
import json
import math
import os
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
except ModuleNotFoundError:  # pragma: no cover - optional dependency in some deploy targets
    px = None

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
    build_game_focus_summary,
    build_player_pool,
    enrich_lineups_minutes_from_pool,
    generate_lineups,
    lineups_slots_frame,
    lineups_summary_frame,
    normalize_injuries_frame,
    projection_role_bucket_key,
    projection_salary_bucket_key,
    recommended_focus_stack_settings,
    remove_injured_players,
)
from college_basketball_dfs.cbb_tail_model import (
    fit_total_tail_model,
    map_slate_games_to_tail_features,
    score_odds_games_for_tail,
)
from college_basketball_dfs.cbb_tournament_review import (
    build_entry_actual_points_comparison,
    build_field_entries_and_players,
    build_ownership_projection_diagnostics,
    build_player_exposure_comparison,
    build_projection_bias_heatmap,
    build_projection_actual_comparison,
    build_projection_adjustment_factors,
    build_segment_impact_table,
    build_top10_winner_gap_analysis,
    compare_phantom_entries_to_field,
    build_user_strategy_summary,
    score_generated_lineups_against_actuals,
    summarize_phantom_entries,
    summarize_generated_lineups,
    normalize_contest_standings_frame,
    detect_contest_standings_upload,
    extract_actual_ownership_from_standings,
)
from college_basketball_dfs.cbb_ai_review import (
    AI_REVIEW_SYSTEM_PROMPT,
    GAME_SLATE_AI_REVIEW_SYSTEM_PROMPT,
    MARKET_CORRELATION_AI_REVIEW_SYSTEM_PROMPT,
    build_ai_review_user_prompt,
    build_daily_ai_review_packet,
    build_game_slate_ai_review_packet,
    build_game_slate_ai_review_user_prompt,
    build_global_ai_review_packet,
    build_global_ai_review_user_prompt,
    build_market_correlation_ai_review_packet,
    build_market_correlation_ai_review_user_prompt,
    request_openai_review,
)
from college_basketball_dfs.cbb_rotowire import (
    RotoWireClient,
    flatten_slates as flatten_rotowire_slates,
    normalize_players as normalize_rotowire_players,
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


def _resolve_rotowire_cookie() -> str | None:
    return os.getenv("ROTOWIRE_COOKIE") or _secret("rotowire_cookie")


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
PROJECTION_ROLE_BUCKET_ORDER = ("guard", "forward", "center", "other")
PROJECTION_ROLE_BUCKET_LABELS = {
    "guard": "Guard",
    "forward": "Forward",
    "center": "Center",
    "other": "Other",
}
SLATE_PRESET_OPTIONS = ["Main", "Afternoon", "Full Day", "Night", "Custom"]
LINEUP_MODEL_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "label": "Standard v1 (Balanced Core)",
        "version_key": "standard_v1",
        "lineup_strategy": "standard",
        "include_tail_signals": False,
        "model_profile": "legacy_baseline",
        "all_versions_weight": 0.08,
        "gpp_overrides": {
            "salary_left_target": 260,
            "low_own_bucket_exposure_pct": 34.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 13.0,
            "low_own_bucket_min_projection": 20.0,
            "low_own_bucket_min_tail_score": 56.0,
            "low_own_bucket_objective_bonus": 1.2,
            "preferred_game_bonus": 0.8,
            "preferred_game_stack_lineup_pct": 45.0,
            "preferred_game_stack_min_players": 2,
            "max_unsupported_false_chalk_per_lineup": 2,
            "ceiling_boost_lineup_pct": 32.0,
            "ceiling_boost_stack_bonus": 2.4,
            "ceiling_boost_salary_left_target": 150,
        },
    },
    {
        "label": "Spike v2 (High-Variance Tail)",
        "version_key": "spike_v2_tail",
        "lineup_strategy": "spike",
        "include_tail_signals": True,
        "model_profile": "tail_spike_pairs",
        "all_versions_weight": 0.27,
        "gpp_overrides": {
            "salary_left_target": 420,
            "low_own_bucket_exposure_pct": 68.0,
            "low_own_bucket_min_per_lineup": 2,
            "low_own_bucket_max_projected_ownership": 11.0,
            "low_own_bucket_min_projection": 16.0,
            "low_own_bucket_min_tail_score": 62.0,
            "low_own_bucket_objective_bonus": 2.0,
            "preferred_game_bonus": 1.3,
            "preferred_game_stack_lineup_pct": 60.0,
            "preferred_game_stack_min_players": 2,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 70.0,
            "ceiling_boost_stack_bonus": 3.5,
            "ceiling_boost_salary_left_target": 230,
        },
    },
    {
        "label": "Standout v1 (Leverage Surge)",
        "version_key": "standout_v1_capture",
        "lineup_strategy": "standard",
        "include_tail_signals": True,
        "model_profile": "standout_capture_v1",
        "all_versions_weight": 0.18,
        "gpp_overrides": {
            "salary_left_target": 340,
            "low_own_bucket_exposure_pct": 54.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 12.0,
            "low_own_bucket_min_projection": 18.0,
            "low_own_bucket_min_tail_score": 58.0,
            "low_own_bucket_objective_bonus": 1.7,
            "preferred_game_bonus": 1.1,
            "preferred_game_stack_lineup_pct": 55.0,
            "preferred_game_stack_min_players": 2,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 52.0,
            "ceiling_boost_stack_bonus": 2.9,
            "ceiling_boost_salary_left_target": 190,
        },
    },
    {
        "label": "Chalk-Value v1 (Leverage Pivots)",
        "version_key": "chalk_value_capture_v1",
        "lineup_strategy": "standard",
        "include_tail_signals": True,
        "model_profile": "chalk_value_capture_v1",
        "all_versions_weight": 0.12,
        "gpp_overrides": {
            "salary_left_target": 290,
            "low_own_bucket_exposure_pct": 32.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 13.0,
            "low_own_bucket_min_projection": 19.0,
            "low_own_bucket_min_tail_score": 55.0,
            "low_own_bucket_objective_bonus": 1.3,
            "preferred_game_bonus": 0.9,
            "preferred_game_stack_lineup_pct": 60.0,
            "preferred_game_stack_min_players": 3,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 38.0,
            "ceiling_boost_stack_bonus": 2.5,
            "ceiling_boost_salary_left_target": 160,
        },
    },
    {
        "label": "Salary-Efficiency v1 (Ceiling)",
        "version_key": "salary_efficiency_ceiling_v1",
        "lineup_strategy": "standard",
        "include_tail_signals": True,
        "model_profile": "salary_efficiency_ceiling_v1",
        "all_versions_weight": 0.35,
        "gpp_overrides": {
            "salary_left_target": 370,
            "low_own_bucket_exposure_pct": 48.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 12.0,
            "low_own_bucket_min_projection": 20.0,
            "low_own_bucket_min_tail_score": 60.0,
            "low_own_bucket_objective_bonus": 1.6,
            "preferred_game_bonus": 1.1,
            "preferred_game_stack_lineup_pct": 70.0,
            "preferred_game_stack_min_players": 3,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 66.0,
            "ceiling_boost_stack_bonus": 3.1,
            "ceiling_boost_salary_left_target": 210,
        },
    },
)
LINEUP_MODEL_BY_KEY = {str(cfg["version_key"]): cfg for cfg in LINEUP_MODEL_REGISTRY}
LINEUP_MODEL_BY_LABEL = {str(cfg["label"]): str(cfg["version_key"]) for cfg in LINEUP_MODEL_REGISTRY}
DEFAULT_ALL_VERSION_WEIGHTS = {
    str(cfg["version_key"]): float(cfg.get("all_versions_weight", 1.0)) for cfg in LINEUP_MODEL_REGISTRY
}
ALL_VERSIONS_KEYS_TEXT = ", ".join(str(cfg["version_key"]) for cfg in LINEUP_MODEL_REGISTRY)
ALL_VERSIONS_WEIGHT_TEXT = ", ".join(
    f"{str(cfg['version_key'])}={int(round(float(cfg.get('all_versions_weight', 1.0)) * 100.0))}%"
    for cfg in LINEUP_MODEL_REGISTRY
)


def _lineup_model_config(version_key: str) -> dict[str, Any]:
    cfg = LINEUP_MODEL_BY_KEY.get(
        str(version_key or "").strip(),
        LINEUP_MODEL_BY_KEY.get("standard_v1", {}),
    )
    return {
        "version_key": str(cfg.get("version_key") or "standard_v1"),
        "version_label": str(cfg.get("label") or "Standard v1"),
        "lineup_strategy": str(cfg.get("lineup_strategy") or "standard"),
        "include_tail_signals": bool(cfg.get("include_tail_signals", False)),
        "model_profile": str(cfg.get("model_profile") or "legacy_baseline"),
        "gpp_overrides": dict(cfg.get("gpp_overrides") or {}),
    }


def _contest_is_gpp(contest_type: str) -> bool:
    return str(contest_type or "").strip().lower() in {"small gpp", "large gpp"}


def _resolve_lineup_runtime_controls(
    *,
    contest_type: str,
    version_cfg: dict[str, Any],
    apply_gpp_variance_presets: bool,
    salary_left_target: int,
    low_own_bucket_exposure_pct: float,
    low_own_bucket_min_per_lineup: int,
    low_own_bucket_max_projected_ownership: float,
    low_own_bucket_min_projection: float,
    low_own_bucket_min_tail_score: float,
    low_own_bucket_objective_bonus: float,
    preferred_game_bonus: float,
    preferred_game_stack_lineup_pct: float,
    preferred_game_stack_min_players: int,
    max_unsupported_false_chalk_per_lineup: int,
    ceiling_boost_lineup_pct: float,
    ceiling_boost_stack_bonus: float,
    ceiling_boost_salary_left_target: int,
) -> dict[str, Any]:
    controls: dict[str, Any] = {
        "salary_left_target": int(salary_left_target),
        "low_own_bucket_exposure_pct": float(low_own_bucket_exposure_pct),
        "low_own_bucket_min_per_lineup": int(low_own_bucket_min_per_lineup),
        "low_own_bucket_max_projected_ownership": float(low_own_bucket_max_projected_ownership),
        "low_own_bucket_min_projection": float(low_own_bucket_min_projection),
        "low_own_bucket_min_tail_score": float(low_own_bucket_min_tail_score),
        "low_own_bucket_objective_bonus": float(low_own_bucket_objective_bonus),
        "preferred_game_bonus": float(preferred_game_bonus),
        "preferred_game_stack_lineup_pct": float(preferred_game_stack_lineup_pct),
        "preferred_game_stack_min_players": int(preferred_game_stack_min_players),
        "max_unsupported_false_chalk_per_lineup": int(max_unsupported_false_chalk_per_lineup),
        "ceiling_boost_lineup_pct": float(ceiling_boost_lineup_pct),
        "ceiling_boost_stack_bonus": float(ceiling_boost_stack_bonus),
        "ceiling_boost_salary_left_target": int(ceiling_boost_salary_left_target),
        "variance_preset_applied": False,
    }
    if not apply_gpp_variance_presets or not _contest_is_gpp(contest_type):
        return controls
    preset = version_cfg.get("gpp_overrides") if isinstance(version_cfg.get("gpp_overrides"), dict) else {}
    if not preset:
        return controls
    for key, val in preset.items():
        if key in controls and key != "variance_preset_applied":
            controls[key] = val
    controls["variance_preset_applied"] = True
    return controls


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


def _read_uploaded_csv_frame(uploaded_file: Any) -> dict[str, Any]:
    raw_bytes = uploaded_file.getvalue() if uploaded_file is not None else b""
    decode_warning = ""
    try:
        csv_text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        csv_text = raw_bytes.decode("utf-8-sig", errors="replace")
        decode_warning = "Upload contained non-UTF-8 bytes; undecodable characters were replaced."

    parse_mode = "default"
    parse_error = ""
    frame = pd.DataFrame()
    try:
        frame = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        parse_error = str(exc)
        parse_mode = "failed"
        try:
            frame = pd.read_csv(io.StringIO(csv_text), engine="python", on_bad_lines="skip")
            parse_mode = "relaxed"
        except Exception as relaxed_exc:
            parse_error = f"{parse_error}; relaxed parser error: {relaxed_exc}"

    return {
        "csv_text": csv_text,
        "frame": frame,
        "parse_mode": parse_mode,
        "parse_error": parse_error,
        "decode_warning": decode_warning,
    }


def _normalize_slate_label(value: object, default: str = "Main") -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text or default


def _slate_key_from_label(value: object, default: str = "main") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or default


def _default_rotowire_slate_name(shared_label: object) -> str:
    normalized = _slate_key_from_label(shared_label, default="")
    mapping = {
        "main": "All",
        "full_day": "All",
        "afternoon": "Afternoon",
        "night": "Night",
    }
    return mapping.get(normalized, "")


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


def _normalize_game_key_token(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text.split(" ")[0]


def _ranked_bonus(
    rank_index: int,
    *,
    start: float,
    step: float,
    floor: float,
) -> float:
    return float(max(floor, start - (step * float(max(0, rank_index)))))


def _allocate_weighted_counts(
    *,
    keys: list[str],
    total_count: int,
    weights: dict[str, float] | None,
    min_per_key: int = 1,
) -> dict[str, int]:
    clean_keys = [str(k).strip() for k in keys if str(k).strip()]
    if not clean_keys:
        return {}
    total = max(0, int(total_count))
    if total <= 0:
        return {k: 0 for k in clean_keys}

    key_count = len(clean_keys)
    floor = max(0, int(min_per_key))
    if total < (floor * key_count):
        floor = 0
    counts = {k: floor for k in clean_keys}
    remaining = total - (floor * key_count)
    if remaining <= 0:
        return counts

    raw_weights = {k: float((weights or {}).get(k, 1.0)) for k in clean_keys}
    for key, value in list(raw_weights.items()):
        if value <= 0.0:
            raw_weights[key] = 1.0
    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0:
        weight_sum = float(key_count)
        raw_weights = {k: 1.0 for k in clean_keys}

    expected = {k: (remaining * raw_weights[k] / weight_sum) for k in clean_keys}
    assigned = 0
    for key in clean_keys:
        whole = int(math.floor(expected[key]))
        counts[key] += whole
        assigned += whole
    leftover = remaining - assigned
    if leftover > 0:
        rank_keys = sorted(clean_keys, key=lambda k: (expected[k] - math.floor(expected[k])), reverse=True)
        for key in rank_keys[:leftover]:
            counts[key] += 1
    return counts


def _build_game_agent_lineup_objective_adjustments(
    *,
    packet: dict[str, Any] | None,
    pool_df: pd.DataFrame,
    contest_type: str,
    strength_multiplier: float = 1.0,
    focus_games: int = 3,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not isinstance(packet, dict) or not packet:
        return {}, {"reason": "missing_packet"}
    if pool_df.empty:
        return {}, {"reason": "empty_pool"}

    focus_tables = packet.get("focus_tables") if isinstance(packet.get("focus_tables"), dict) else {}
    game_targets = focus_tables.get("gpp_game_stack_targets") if isinstance(focus_tables, dict) else []
    team_targets = focus_tables.get("gpp_team_stack_targets") if isinstance(focus_tables, dict) else []
    player_targets = focus_tables.get("gpp_player_core_targets") if isinstance(focus_tables, dict) else []
    game_targets = game_targets if isinstance(game_targets, list) else []
    team_targets = team_targets if isinstance(team_targets, list) else []
    player_targets = player_targets if isinstance(player_targets, list) else []
    if not game_targets and not team_targets and not player_targets:
        return {}, {"reason": "no_targets"}

    contest_norm = str(contest_type or "").strip().lower()
    if contest_norm == "cash":
        contest_weight = 0.45
    elif contest_norm == "small gpp":
        contest_weight = 0.85
    else:
        contest_weight = 1.0
    strength = max(0.0, float(strength_multiplier)) * contest_weight
    if strength <= 0.0:
        return {}, {"reason": "zero_strength"}

    working = pool_df.copy()
    if "ID" not in working.columns:
        return {}, {"reason": "missing_id_col"}
    working["_id_norm"] = working["ID"].map(_normalize_player_id)
    if "TeamAbbrev" in working.columns:
        working["_team_norm"] = working["TeamAbbrev"].astype(str).str.strip().str.upper()
    else:
        working["_team_norm"] = ""
    if "game_key" in working.columns:
        working["_game_norm"] = working["game_key"].map(_normalize_game_key_token)
    else:
        working["_game_norm"] = ""
    if "Name" in working.columns:
        working["_name_norm"] = working["Name"].map(_norm_name_key_loose)
    else:
        working["_name_norm"] = ""
    working = working.loc[working["_id_norm"] != ""].copy()
    if working.empty:
        return {}, {"reason": "no_pool_ids"}

    adjustments: dict[str, float] = {pid: 0.0 for pid in working["_id_norm"].astype(str).tolist()}
    applied_games = 0
    applied_teams = 0
    applied_players = 0
    applied_game_keys: set[str] = set()
    player_cap = max(2.5, 8.0 * strength)
    game_limit = max(1, min(int(focus_games), 10))
    team_limit = max(2, min(int(focus_games * 2), 20))
    player_limit = max(3, min(int(focus_games * 3), 36))

    for idx, row in enumerate(game_targets[:game_limit]):
        if not isinstance(row, dict):
            continue
        game_key = _normalize_game_key_token(row.get("game_key"))
        if not game_key:
            continue
        bonus = _ranked_bonus(
            idx,
            start=2.25 * strength,
            step=0.35 * strength,
            floor=0.20 * strength,
        )
        game_ids = (
            working.loc[working["_game_norm"] == game_key, "_id_norm"]
            .astype(str)
            .dropna()
            .tolist()
        )
        if not game_ids:
            continue
        for pid in game_ids:
            adjustments[pid] = min(player_cap, adjustments.get(pid, 0.0) + bonus)
        applied_games += 1
        applied_game_keys.add(game_key)

    for idx, row in enumerate(team_targets[:team_limit]):
        if not isinstance(row, dict):
            continue
        team = str(row.get("team_abbrev") or "").strip().upper()
        if not team:
            continue
        bonus = _ranked_bonus(
            idx,
            start=1.90 * strength,
            step=0.25 * strength,
            floor=0.15 * strength,
        )
        team_ids = (
            working.loc[working["_team_norm"] == team, "_id_norm"]
            .astype(str)
            .dropna()
            .tolist()
        )
        if not team_ids:
            continue
        for pid in team_ids:
            adjustments[pid] = min(player_cap, adjustments.get(pid, 0.0) + bonus)
        applied_teams += 1

    for idx, row in enumerate(player_targets[:player_limit]):
        if not isinstance(row, dict):
            continue
        player_name_key = _norm_name_key_loose(row.get("player_name"))
        if not player_name_key:
            continue
        team = str(row.get("team_abbrev") or "").strip().upper()
        game_key = _normalize_game_key_token(row.get("game_key"))
        mask = working["_name_norm"] == player_name_key
        if team:
            mask = mask & (working["_team_norm"] == team)
        if game_key:
            mask = mask & (working["_game_norm"] == game_key)
        if not bool(mask.any()):
            # Fallback: name + team when game_key labels mismatch.
            mask = working["_name_norm"] == player_name_key
            if team:
                mask = mask & (working["_team_norm"] == team)
        player_ids = working.loc[mask, "_id_norm"].astype(str).dropna().tolist()
        if not player_ids:
            continue
        bonus = _ranked_bonus(
            idx,
            start=2.75 * strength,
            step=0.30 * strength,
            floor=0.25 * strength,
        )
        for pid in player_ids:
            adjustments[pid] = min(player_cap, adjustments.get(pid, 0.0) + bonus)
        applied_players += 1

    final_adjustments = {
        str(pid): round(float(val), 4)
        for pid, val in adjustments.items()
        if abs(float(val)) >= 1e-6
    }
    if not final_adjustments:
        return {}, {
            "reason": "no_pool_match",
            "applied_games": applied_games,
            "applied_teams": applied_teams,
            "applied_players": applied_players,
            "applied_game_keys": sorted(applied_game_keys),
        }

    adjusted_preview_df = working.copy()
    adjusted_preview_df["_adj"] = adjusted_preview_df["_id_norm"].map(final_adjustments).fillna(0.0)
    adjusted_preview_df = adjusted_preview_df.loc[adjusted_preview_df["_adj"] > 0].copy()
    adjusted_preview_df = adjusted_preview_df.sort_values("_adj", ascending=False)
    preview_rows = (
        adjusted_preview_df.loc[:, ["Name", "TeamAbbrev", "_id_norm", "_adj"]]
        .rename(columns={"_id_norm": "player_id", "_adj": "objective_bonus"})
        .head(8)
        .to_dict(orient="records")
        if not adjusted_preview_df.empty
        else []
    )
    return final_adjustments, {
        "applied_games": int(applied_games),
        "applied_teams": int(applied_teams),
        "applied_players": int(applied_players),
        "adjusted_players": int(len(final_adjustments)),
        "applied_game_keys": sorted(applied_game_keys),
        "preview_rows": preview_rows,
    }


def _extract_game_keys_from_text(value: Any) -> set[str]:
    text = str(value or "").strip().upper()
    if not text or text.lower() in {"nan", "none", "null"}:
        return set()
    return {
        str(match.group(0) or "").strip().upper()
        for match in re.finditer(r"[A-Z0-9.&']+@[A-Z0-9.&']+", text)
        if str(match.group(0) or "").strip()
    }


def _lineup_game_keys_from_payload(lineup: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for player in (lineup.get("players") or []):
        game_key = _normalize_game_key_token(player.get("game_key"))
        if "@" in game_key:
            keys.add(game_key)
    anchor_key = _normalize_game_key_token(lineup.get("anchor_game_key"))
    if "@" in anchor_key:
        keys.add(anchor_key)
    keys.update(_extract_game_keys_from_text(lineup.get("stack_signature")))
    return keys


def build_lineup_consistency_packet(
    *,
    run_bundle: dict[str, Any] | None,
    active_version_key: str,
    active_version: dict[str, Any] | None,
    phantom_df: pd.DataFrame | None,
    phantom_summary_df: pd.DataFrame | None,
    phantom_meta: dict[str, Any] | None,
) -> dict[str, Any]:
    run_data = run_bundle if isinstance(run_bundle, dict) else {}
    settings = run_data.get("settings") if isinstance(run_data.get("settings"), dict) else {}
    version_data = active_version if isinstance(active_version, dict) else {}
    version_warnings = [str(x).strip() for x in (version_data.get("warnings") or []) if str(x).strip()]
    low_own_candidates_unavailable = any(
        ("low-owned upside bucket was enabled" in msg.lower() and "no candidates met the filters" in msg.lower())
        for msg in version_warnings
    )
    generated_lineups = version_data.get("lineups") or []
    generated_lineups = generated_lineups if isinstance(generated_lineups, list) else []
    total_generated = int(len(generated_lineups))
    run_id = str(run_data.get("run_id") or "").strip()
    phantom_meta = phantom_meta if isinstance(phantom_meta, dict) else {}

    phantom_frame = phantom_df.copy() if isinstance(phantom_df, pd.DataFrame) else pd.DataFrame()
    phantom_summary = phantom_summary_df.copy() if isinstance(phantom_summary_df, pd.DataFrame) else pd.DataFrame()
    if not phantom_frame.empty and active_version_key and "version_key" in phantom_frame.columns:
        phantom_active = phantom_frame.loc[
            phantom_frame["version_key"].astype(str).str.strip() == str(active_version_key).strip()
        ].copy()
    else:
        phantom_active = phantom_frame.copy()

    summary_active_row: dict[str, Any] = {}
    if not phantom_summary.empty and active_version_key and "version_key" in phantom_summary.columns:
        filtered = phantom_summary.loc[
            phantom_summary["version_key"].astype(str).str.strip() == str(active_version_key).strip()
        ].copy()
        if not filtered.empty:
            summary_active_row = filtered.iloc[0].to_dict()
    if not summary_active_row and not phantom_summary.empty:
        summary_active_row = phantom_summary.iloc[0].to_dict()

    checks: list[dict[str, Any]] = []

    def add_check(
        *,
        area: str,
        status: str,
        target: str,
        actual: str,
        gap: str = "",
        note: str = "",
    ) -> None:
        checks.append(
            {
                "area": str(area),
                "status": str(status).strip().lower(),
                "target": str(target),
                "actual": str(actual),
                "gap": str(gap),
                "note": str(note),
            }
        )

    has_phantom_data = bool((not phantom_active.empty) or summary_active_row)
    phantom_run_id = str(phantom_meta.get("run_id") or "").strip()
    if has_phantom_data and run_id and phantom_run_id and run_id != phantom_run_id:
        add_check(
            area="Phantom Alignment",
            status="warn",
            target=f"phantom run_id={run_id}",
            actual=f"phantom run_id={phantom_run_id}",
            gap="run mismatch",
            note="Phantom review results are from a different run bundle.",
        )
    elif has_phantom_data:
        add_check(
            area="Review Mode",
            status="pass",
            target="pre-game checks + post-phantom diagnostics",
            actual=f"phantom lineups={int(len(phantom_active))}",
            note="Phantom metrics were included for the active version.",
        )
    else:
        add_check(
            area="Review Mode",
            status="pass",
            target="pre-game readiness checks",
            actual="phantom metrics unavailable (optional)",
            note="Core consistency checks run before lock without phantom results.",
        )

    lineup_player_ids: list[set[str]] = []
    for lineup in generated_lineups:
        ids: set[str] = set()
        for player in (lineup.get("players") or []):
            pid = _normalize_player_id(player.get("ID"))
            if pid:
                ids.add(pid)
        if not ids:
            for pid in (lineup.get("player_ids") or []):
                normalized = _normalize_player_id(pid)
                if normalized:
                    ids.add(normalized)
        lineup_player_ids.append(ids)

    locked_ids = {_normalize_player_id(x) for x in (settings.get("locked_ids") or []) if _normalize_player_id(x)}
    excluded_ids = {_normalize_player_id(x) for x in (settings.get("excluded_ids") or []) if _normalize_player_id(x)}
    if total_generated > 0 and locked_ids:
        missing_locks = int(sum(1 for ids in lineup_player_ids if not locked_ids.issubset(ids)))
        status = "pass" if missing_locks == 0 else ("warn" if missing_locks <= 1 else "fail")
        add_check(
            area="Lock Adherence",
            status=status,
            target=f"all lineups include locked IDs ({len(locked_ids)})",
            actual=f"{total_generated - missing_locks}/{total_generated} lineups",
            gap=f"{missing_locks} violations",
            note="Ensures lock settings were respected.",
        )
    if total_generated > 0 and excluded_ids:
        violated = int(sum(1 for ids in lineup_player_ids if bool(ids & excluded_ids)))
        status = "pass" if violated == 0 else ("warn" if violated <= 1 else "fail")
        add_check(
            area="Exclusion Adherence",
            status=status,
            target=f"no lineups include excluded IDs ({len(excluded_ids)})",
            actual=f"{violated} violating lineups",
            gap=f"{violated}",
            note="Ensures exclusion settings were respected.",
        )

    if total_generated > 0:
        global_cap_pct = max(0.0, min(100.0, _safe_float_value(settings.get("global_max_exposure_pct"), default=100.0)))
        global_cap_count = int(math.floor((global_cap_pct / 100.0) * float(total_generated)))
        exposure_counts: dict[str, int] = {}
        for ids in lineup_player_ids:
            for pid in ids:
                exposure_counts[pid] = exposure_counts.get(pid, 0) + 1
        if exposure_counts and global_cap_pct < 100.0:
            non_lock_counts = [count for pid, count in exposure_counts.items() if pid not in locked_ids]
            max_count = max(non_lock_counts) if non_lock_counts else 0
            worst_over = max(0, int(max_count - global_cap_count))
            status = "pass" if worst_over <= 0 else ("warn" if worst_over <= 1 else "fail")
            add_check(
                area="Global Exposure Cap",
                status=status,
                target=f"max exposure <= {global_cap_count}/{total_generated} ({global_cap_pct:.1f}%)",
                actual=f"worst player appears {max_count}/{total_generated}",
                gap=f"{worst_over}",
                note="Checks global player exposure cap enforcement.",
            )

        per_player_caps = settings.get("exposure_caps_pct") if isinstance(settings.get("exposure_caps_pct"), dict) else {}
        if per_player_caps:
            per_cap_violations = 0
            for pid_raw, pct_raw in per_player_caps.items():
                pid = _normalize_player_id(pid_raw)
                if not pid:
                    continue
                if pid in locked_ids:
                    continue
                cap_pct = max(0.0, min(100.0, _safe_float_value(pct_raw, default=100.0)))
                cap_count = int(math.floor((cap_pct / 100.0) * float(total_generated)))
                actual_count = int(exposure_counts.get(pid, 0))
                if actual_count > cap_count:
                    per_cap_violations += 1
            status = "pass" if per_cap_violations == 0 else ("warn" if per_cap_violations <= 2 else "fail")
            add_check(
                area="Per-Player Exposure Caps",
                status=status,
                target="all configured player caps satisfied",
                actual=f"{per_cap_violations} players over cap",
                gap=f"{per_cap_violations}",
                note="Checks player-specific exposure cap settings.",
            )

    if total_generated > 0 and settings.get("max_salary_left") is not None:
        max_salary_left_setting = _safe_int_value(settings.get("max_salary_left"), default=50000)
        salary_left_vals = pd.to_numeric(
            pd.Series([lineup.get("salary_left") for lineup in generated_lineups]),
            errors="coerce",
        ).fillna(0)
        over_limit = int((salary_left_vals > max_salary_left_setting).sum())
        status = "pass" if over_limit == 0 else ("warn" if over_limit <= 1 else "fail")
        add_check(
            area="Max Salary Left Constraint",
            status=status,
            target=f"salary_left <= {max_salary_left_setting}",
            actual=f"{over_limit}/{total_generated} lineups exceed",
            gap=f"{over_limit}",
            note="Checks hard salary-left constraint adherence.",
        )

    target_low_own_pct = _safe_float_value(settings.get("low_own_bucket_exposure_pct"), default=0.0)
    low_own_min = max(0, _safe_int_value(settings.get("low_own_bucket_min_per_lineup"), default=0))
    if total_generated > 0 and target_low_own_pct > 0 and low_own_min > 0:
        expected = max(1, int(round((target_low_own_pct / 100.0) * total_generated)))
        actual = int(
            sum(1 for lineup in generated_lineups if _safe_int_value(lineup.get("low_own_upside_count"), 0) >= low_own_min)
        )
        if low_own_candidates_unavailable:
            status = "warn" if actual < expected else "pass"
            note = (
                f"configured exposure={target_low_own_pct:.1f}%; "
                "optimizer reported no eligible low-own candidates for current filters."
            )
        else:
            status = "pass" if actual >= expected else ("warn" if actual >= max(1, expected - 1) else "fail")
            note = f"configured exposure={target_low_own_pct:.1f}%"
        add_check(
            area="Low-Own Bucket",
            status=status,
            target=f">= {expected}/{total_generated} lineups (min {low_own_min} low-own players)",
            actual=f"{actual}/{total_generated}",
            gap=f"{actual - expected:+d}",
            note=note,
        )
        if low_own_candidates_unavailable:
            add_check(
                area="Low-Own Candidate Availability",
                status="warn",
                target="eligible low-own candidates available when bucket enabled",
                actual="optimizer reported no eligible candidates",
                gap="n/a",
                note=(
                    "Relax low-own filters (max projected ownership, min projection/tail), "
                    "or reduce low-own bucket exposure."
                ),
            )

    target_ceiling_pct = _safe_float_value(settings.get("ceiling_boost_lineup_pct"), default=0.0)
    if total_generated > 0 and target_ceiling_pct > 0:
        expected = max(1, int(round((target_ceiling_pct / 100.0) * total_generated)))
        actual = int(sum(1 for lineup in generated_lineups if bool(lineup.get("ceiling_boost_active"))))
        status = "pass" if actual >= expected else ("warn" if actual >= max(1, expected - 1) else "fail")
        add_check(
            area="Ceiling Archetype Share",
            status=status,
            target=f">= {expected}/{total_generated} lineups",
            actual=f"{actual}/{total_generated}",
            gap=f"{actual - expected:+d}",
            note=f"configured ceiling share={target_ceiling_pct:.1f}%",
        )

    salary_left_target = settings.get("salary_left_target")
    if total_generated > 0 and salary_left_target is not None:
        salary_left_vals = pd.to_numeric(
            pd.Series([lineup.get("salary_left") for lineup in generated_lineups]),
            errors="coerce",
        ).dropna()
        if not salary_left_vals.empty:
            avg_left = float(salary_left_vals.mean())
            target_left = _safe_float_value(salary_left_target, default=0.0)
            delta = avg_left - target_left
            abs_delta = abs(delta)
            status = "pass" if abs_delta <= 120.0 else ("warn" if abs_delta <= 220.0 else "fail")
            add_check(
                area="Salary Utilization",
                status=status,
                target=f"avg salary_left ~ {target_left:.0f}",
                actual=f"{avg_left:.1f}",
                gap=f"{delta:+.1f}",
                note="Uses generated lineup salary_left means.",
            )

    preferred_games = {
        _normalize_game_key_token(x)
        for x in ((settings.get("game_agent_bias_meta") or {}).get("applied_game_keys") or [])
        if _normalize_game_key_token(x)
    }
    if bool(settings.get("apply_game_agent_stack_bias")):
        if total_generated > 0 and preferred_games:
            lineup_hits = 0
            preferred_player_hits = 0
            for lineup in generated_lineups:
                lineup_keys = _lineup_game_keys_from_payload(lineup)
                if lineup_keys & preferred_games:
                    lineup_hits += 1
                preferred_player_hits += int(
                    sum(
                        1
                        for player in (lineup.get("players") or [])
                        if _normalize_game_key_token(player.get("game_key")) in preferred_games
                    )
                )
            lineup_hit_pct = (100.0 * float(lineup_hits) / float(max(1, total_generated)))
            avg_pref_players = float(preferred_player_hits) / float(max(1, total_generated))
            status = "pass" if lineup_hit_pct >= 20.0 else ("warn" if lineup_hit_pct >= 10.0 else "fail")
            add_check(
                area="Game Agent Stack Bias",
                status=status,
                target=f">=20% lineups include preferred games ({len(preferred_games)} games)",
                actual=f"{lineup_hit_pct:.1f}% lineups | {avg_pref_players:.2f} preferred players/lineup",
                gap=f"{lineup_hit_pct - 20.0:+.1f}pp vs minimum",
                note="Checks whether generated portfolios reflect Game Agent game targets.",
            )
        elif total_generated > 0 and not preferred_games:
            add_check(
                area="Game Agent Stack Bias",
                status="warn",
                target="preferred game keys present when stack bias enabled",
                actual="no preferred games in settings",
                gap="missing target list",
                note="Bias enabled but no game-level targets were captured.",
            )

    if summary_active_row:
        beat = _safe_float_value(summary_active_row.get("avg_would_beat_pct"), default=0.0)
        drift = _safe_float_value(summary_active_row.get("avg_actual_minus_projected"), default=0.0)
        winner_gap = _safe_float_value(summary_active_row.get("winner_gap"), default=0.0)
        beat_status = "pass" if beat >= 30.0 else ("warn" if beat >= 20.0 else "fail")
        drift_status = "pass" if drift >= -35.0 else ("warn" if drift >= -55.0 else "fail")
        gap_status = "pass" if winner_gap <= 45.0 else ("warn" if winner_gap <= 70.0 else "fail")
        add_check(
            area="Phantom Beat Rate",
            status=beat_status,
            target="avg_would_beat_pct >= 30",
            actual=f"{beat:.2f}",
            gap=f"{beat - 30.0:+.2f}",
            note="Version-level phantom competitiveness.",
        )
        add_check(
            area="Projection Drift",
            status=drift_status,
            target="avg_actual_minus_projected >= -35",
            actual=f"{drift:.2f}",
            gap=f"{drift + 35.0:+.2f}",
            note="Negative drift indicates under-delivery vs projection.",
        )
        if _safe_float_value(summary_active_row.get("winner_points"), default=0.0) > 0.0:
            add_check(
                area="Winner Gap",
                status=gap_status,
                target="winner_gap <= 45",
                actual=f"{winner_gap:.2f}",
                gap=f"{winner_gap - 45.0:+.2f}",
                note="Top-end capture versus field winner.",
            )

    version_allocation_rows: list[dict[str, Any]] = []
    versions_dict = run_data.get("versions") if isinstance(run_data.get("versions"), dict) else {}
    if str(run_data.get("run_mode") or "").strip().lower() == "all" and bool(settings.get("promote_phantom_constructions")):
        promo_meta = settings.get("phantom_promotion_meta") if isinstance(settings.get("phantom_promotion_meta"), dict) else {}
        promo_used_dates = int(_safe_int_value(promo_meta.get("used_dates"), default=0))
        promo_weights = {
            str(k): float(v)
            for k, v in (promo_meta.get("weights") or {}).items()
            if str(k).strip()
        }
        if promo_used_dates > 0 and promo_weights:
            add_check(
                area="Phantom Promotion Inputs",
                status="pass",
                target="historical phantom data available for promotion",
                actual=f"used_dates={promo_used_dates}, weighted_versions={len(promo_weights)}",
                gap="0",
                note="Version allocation used historical phantom review summaries.",
            )
        else:
            add_check(
                area="Phantom Promotion Inputs",
                status="warn",
                target="historical phantom data available for promotion",
                actual=f"used_dates={promo_used_dates}, weighted_versions={len(promo_weights)}",
                gap="missing history",
                note="Promotion fell back to default/equal weighting due to limited historical phantom data.",
            )
        version_keys = [str(k) for k in versions_dict.keys()]
        requested_counts = {
            str(k): int(
                _safe_int_value((versions_dict.get(k) or {}).get("lineup_count_requested"), default=len((versions_dict.get(k) or {}).get("lineups") or []))
            )
            for k in version_keys
        }
        total_requested = int(sum(requested_counts.values()))
        weights = {
            str(k): float(v)
            for k, v in (promo_meta.get("weights") or {}).items()
            if str(k).strip()
        }
        expected_counts = _allocate_weighted_counts(
            keys=version_keys,
            total_count=total_requested,
            weights=weights,
            min_per_key=1,
        )
        max_gap = 0
        for key in version_keys:
            requested = int(requested_counts.get(key, 0))
            expected = int(expected_counts.get(key, 0))
            gap = requested - expected
            max_gap = max(max_gap, abs(gap))
            version_allocation_rows.append(
                {
                    "version_key": key,
                    "requested_lineups": requested,
                    "expected_lineups_from_weights": expected,
                    "allocation_gap": gap,
                }
            )
        if version_allocation_rows:
            promo_status = "pass" if max_gap <= 1 else ("warn" if max_gap <= 2 else "fail")
            add_check(
                area="Version Promotion Allocation",
                status=promo_status,
                target="requested lineup counts align with phantom promotion weights",
                actual=f"max abs gap={max_gap}",
                gap=f"{max_gap}",
                note="Validates promoted version mix against computed weights.",
            )

    preferred_game_phantom_rows: list[dict[str, Any]] = []
    if not phantom_active.empty and preferred_games:
        total_phantom_lineups = int(len(phantom_active))
        exposure_counts: dict[str, int] = {game_key: 0 for game_key in sorted(preferred_games)}
        for _, row in phantom_active.iterrows():
            keys: set[str] = set()
            anchor_key = _normalize_game_key_token(row.get("anchor_game_key"))
            if "@" in anchor_key:
                keys.add(anchor_key)
            keys.update(_extract_game_keys_from_text(row.get("stack_signature")))
            for game_key in preferred_games:
                if game_key in keys:
                    exposure_counts[game_key] += 1
        for game_key, count in exposure_counts.items():
            rate = (float(count) / float(max(1, total_phantom_lineups)))
            preferred_game_phantom_rows.append(
                {
                    "game_key": game_key,
                    "phantom_lineups_with_game": int(count),
                    "phantom_lineup_rate": round(rate, 4),
                    "under_exposed_flag": bool(rate < 0.20),
                }
            )

    upside_rows: list[dict[str, Any]] = []
    if not phantom_summary.empty:
        view_cols = [c for c in ["version_key", "version_label", "avg_would_beat_pct", "best_actual_points", "winner_gap"] if c in phantom_summary.columns]
        if view_cols:
            top_versions = (
                phantom_summary[view_cols]
                .copy()
                .sort_values(["avg_would_beat_pct", "best_actual_points"], ascending=[False, False])
                .head(5)
            )
            upside_rows.extend(top_versions.to_dict(orient="records"))
    if not upside_rows and generated_lineups:
        generated_frame = pd.DataFrame(generated_lineups)
        if not generated_frame.empty:
            if "projected_points" in generated_frame.columns:
                top_proj = (
                    generated_frame.copy()
                    .sort_values("projected_points", ascending=False)
                    .head(5)
                )
                top_proj_cols = [c for c in ["lineup_number", "projected_points", "salary_left", "lineup_strategy", "low_own_upside_count", "ceiling_boost_active"] if c in top_proj.columns]
                upside_rows.extend(top_proj[top_proj_cols].to_dict(orient="records"))

    stack_upside_rows: list[dict[str, Any]] = []
    if not phantom_active.empty and "stack_signature" in phantom_active.columns:
        stack_work = phantom_active.copy()
        stack_work["stack_signature"] = stack_work["stack_signature"].astype(str).str.strip()
        stack_work = stack_work.loc[(stack_work["stack_signature"] != "") & (stack_work["stack_signature"].str.lower() != "nan")]
        if not stack_work.empty and "would_beat_pct" in stack_work.columns:
            stack_summary = (
                stack_work.groupby("stack_signature", as_index=False)
                .agg(
                    lineups=("stack_signature", "count"),
                    avg_would_beat_pct=("would_beat_pct", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
                    best_actual_points=("actual_points", lambda s: float(pd.to_numeric(s, errors="coerce").max())),
                )
                .sort_values(["avg_would_beat_pct", "best_actual_points"], ascending=[False, False])
                .head(10)
            )
            stack_upside_rows = stack_summary.to_dict(orient="records")
    if not stack_upside_rows and generated_lineups:
        generated_frame = pd.DataFrame(generated_lineups)
        if not generated_frame.empty and "stack_signature" in generated_frame.columns:
            stack_work = generated_frame.copy()
            stack_work["stack_signature"] = stack_work["stack_signature"].astype(str).str.strip()
            stack_work = stack_work.loc[(stack_work["stack_signature"] != "") & (stack_work["stack_signature"].str.lower() != "nan")]
            if not stack_work.empty and "projected_points" in stack_work.columns:
                stack_summary = (
                    stack_work.groupby("stack_signature", as_index=False)
                    .agg(
                        lineups=("stack_signature", "count"),
                        avg_projected_points=("projected_points", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
                        best_projected_points=("projected_points", lambda s: float(pd.to_numeric(s, errors="coerce").max())),
                    )
                    .sort_values(["avg_projected_points", "best_projected_points"], ascending=[False, False])
                    .head(10)
                )
                stack_upside_rows = stack_summary.to_dict(orient="records")

    fail_count = int(sum(1 for row in checks if str(row.get("status")).lower() == "fail"))
    warn_count = int(sum(1 for row in checks if str(row.get("status")).lower() == "warn"))
    pass_count = int(sum(1 for row in checks if str(row.get("status")).lower() == "pass"))
    overall_status = "fail" if fail_count > 0 else ("warn" if warn_count > 0 else "pass")

    gap_rows = [row for row in checks if str(row.get("status")).lower() in {"warn", "fail"}]
    if preferred_game_phantom_rows:
        for row in preferred_game_phantom_rows:
            if bool(row.get("under_exposed_flag")):
                gap_rows.append(
                    {
                        "area": "Preferred Game Phantom Exposure",
                        "status": "warn",
                        "target": "phantom_lineup_rate >= 0.20",
                        "actual": f"{str(row.get('game_key'))} -> {100.0 * _safe_float_value(row.get('phantom_lineup_rate')):.1f}%",
                        "gap": f"{(100.0 * (_safe_float_value(row.get('phantom_lineup_rate')) - 0.20)):+.1f}pp",
                        "note": "Preferred game may be underrepresented in phantom constructions.",
                    }
                )

    return {
        "schema_version": "lineup_consistency_v1",
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "review_context": {
            "run_id": run_id,
            "slate_date": str(run_data.get("slate_date") or ""),
            "slate_key": str(run_data.get("slate_key") or ""),
            "active_version_key": str(active_version_key or ""),
            "active_version_label": str(version_data.get("version_label") or active_version_key or ""),
        },
        "data_availability": {
            "phantom_available": bool(has_phantom_data),
            "phantom_lineups": int(len(phantom_active)),
            "generated_lineups": int(total_generated),
            "active_version_warning_count": int(len(version_warnings)),
        },
        "status_summary": {
            "overall_status": overall_status,
            "pass_checks": pass_count,
            "warn_checks": warn_count,
            "fail_checks": fail_count,
        },
        "settings_snapshot": {
            "lineup_count": _safe_int_value(settings.get("lineup_count"), default=0),
            "contest_type": str(settings.get("contest_type") or ""),
            "salary_left_target": settings.get("salary_left_target"),
            "low_own_bucket_exposure_pct": settings.get("low_own_bucket_exposure_pct"),
            "low_own_bucket_min_per_lineup": settings.get("low_own_bucket_min_per_lineup"),
            "ceiling_boost_lineup_pct": settings.get("ceiling_boost_lineup_pct"),
            "apply_game_agent_stack_bias": bool(settings.get("apply_game_agent_stack_bias")),
            "preferred_game_keys": sorted(preferred_games),
            "promote_phantom_constructions": bool(settings.get("promote_phantom_constructions")),
        },
        "checks": checks,
        "gap_candidates": gap_rows,
        "upside_candidates": upside_rows,
        "stack_upside_candidates": stack_upside_rows,
        "preferred_game_phantom_exposure": preferred_game_phantom_rows,
        "version_allocation_check": version_allocation_rows,
        "active_version_warnings": version_warnings,
        "active_version_phantom_summary": _json_safe(summary_active_row),
    }


def _extract_slate_game_keys(
    *,
    pool_df: pd.DataFrame | None,
    raw_slate_df: pd.DataFrame | None,
) -> list[str]:
    keys: set[str] = set()
    if isinstance(pool_df, pd.DataFrame) and not pool_df.empty and "game_key" in pool_df.columns:
        for raw_key in pool_df["game_key"].tolist():
            key = _normalize_game_key_token(raw_key)
            if "@" in key:
                keys.add(key)
    if keys:
        return sorted(keys)

    if isinstance(raw_slate_df, pd.DataFrame) and not raw_slate_df.empty:
        for game_col in ["Game Info", "GameInfo", "game_info"]:
            if game_col not in raw_slate_df.columns:
                continue
            for raw_key in raw_slate_df[game_col].tolist():
                key = _normalize_game_key_token(raw_key)
                if "@" in key:
                    keys.add(key)
            if keys:
                break
    return sorted(keys)


def _filter_odds_to_slate_context(
    *,
    odds_df: pd.DataFrame,
    pool_df: pd.DataFrame | None,
    raw_slate_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if odds_df.empty:
        return odds_df.copy(), {
            "reason": "empty_odds",
            "odds_rows_before": 0,
            "odds_rows_after": 0,
            "requested_slate_games": 0,
            "mapped_slate_games": 0,
        }

    slate_game_keys = _extract_slate_game_keys(pool_df=pool_df, raw_slate_df=raw_slate_df)
    meta: dict[str, Any] = {
        "reason": "",
        "odds_rows_before": int(len(odds_df)),
        "odds_rows_after": int(len(odds_df)),
        "requested_slate_games": int(len(slate_game_keys)),
        "mapped_slate_games": 0,
    }
    if not slate_game_keys:
        meta["reason"] = "no_slate_game_keys"
        return odds_df.copy(), meta

    mapped = map_slate_games_to_tail_features(slate_game_keys=slate_game_keys, odds_tail_df=odds_df)
    mapped_rows = mapped.copy() if isinstance(mapped, pd.DataFrame) else pd.DataFrame()
    meta["mapped_slate_games"] = int(len(mapped_rows))
    if mapped_rows.empty or "game_tail_event_id" not in mapped_rows.columns or "event_id" not in odds_df.columns:
        meta["reason"] = "fallback_unfiltered"
        return odds_df.copy(), meta

    event_ids = {
        str(x).strip()
        for x in mapped_rows["game_tail_event_id"].tolist()
        if str(x or "").strip()
    }
    if not event_ids:
        meta["reason"] = "fallback_unfiltered"
        return odds_df.copy(), meta

    event_norm = odds_df["event_id"].astype(str).str.strip()
    filtered = odds_df.loc[event_norm.isin(event_ids)].copy()
    if filtered.empty:
        meta["reason"] = "fallback_unfiltered"
        return odds_df.copy(), meta

    meta["reason"] = "filtered_by_event_id"
    meta["odds_rows_after"] = int(len(filtered))
    return filtered, meta


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


def _dk_slate_blob_name(slate_date: date, slate_key: str | None = None) -> str:
    if slate_key is None:
        return f"cbb/dk_slates/{slate_date.isoformat()}_dk_slate.csv"
    safe = _slate_key_from_label(slate_key)
    return f"cbb/dk_slates/{slate_date.isoformat()}/{safe}_dk_slate.csv"


def _read_dk_slate_csv(store: CbbGcsStore, slate_date: date, slate_key: str | None = None) -> str | None:
    reader = getattr(store, "read_dk_slate_csv", None)
    if callable(reader):
        try:
            return reader(slate_date, slate_key)
        except TypeError:
            return reader(slate_date)
    if slate_key is None:
        candidate_names = [_dk_slate_blob_name(slate_date, "main"), _dk_slate_blob_name(slate_date)]
    else:
        primary_name = _dk_slate_blob_name(slate_date, slate_key=slate_key)
        candidate_names = [primary_name]
        if _slate_key_from_label(slate_key) == "main":
            candidate_names.append(_dk_slate_blob_name(slate_date))
    for blob_name in candidate_names:
        blob = store.bucket.blob(blob_name)
        if blob.exists():
            return blob.download_as_text(encoding="utf-8")
    return None


def _write_dk_slate_csv(
    store: CbbGcsStore,
    slate_date: date,
    csv_text: str,
    slate_key: str | None = None,
) -> str:
    writer = getattr(store, "write_dk_slate_csv", None)
    if callable(writer):
        try:
            return writer(slate_date, csv_text, slate_key)
        except TypeError:
            return writer(slate_date, csv_text)
    blob_name = _dk_slate_blob_name(slate_date, slate_key=slate_key)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _delete_dk_slate_csv(
    store: CbbGcsStore,
    slate_date: date,
    slate_key: str | None = None,
) -> tuple[bool, str]:
    blob_name = _dk_slate_blob_name(slate_date, slate_key=slate_key)
    deleter = getattr(store, "delete_dk_slate_csv", None)
    if callable(deleter):
        try:
            return bool(deleter(slate_date, slate_key)), blob_name
        except TypeError:
            return bool(deleter(slate_date)), blob_name
    if slate_key is None:
        candidate_names = [_dk_slate_blob_name(slate_date, "main"), _dk_slate_blob_name(slate_date)]
    else:
        candidate_names = [blob_name]
        if _slate_key_from_label(slate_key) == "main":
            candidate_names.append(_dk_slate_blob_name(slate_date))
    for name in candidate_names:
        blob = store.bucket.blob(name)
        if blob.exists():
            blob.delete()
            return True, name
    return False, blob_name


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


def _projections_blob_name(slate_date: date, slate_key: str | None = None) -> str:
    # Canonical date-level storage; slate_key intentionally ignored.
    return f"cbb/projections/{slate_date.isoformat()}_projections.csv"


def _write_projections_csv(
    store: CbbGcsStore,
    slate_date: date,
    csv_text: str,
    slate_key: str | None = None,
) -> str:
    writer = getattr(store, "write_projections_csv", None)
    if callable(writer):
        try:
            return writer(slate_date, csv_text, slate_key)
        except TypeError:
            return writer(slate_date, csv_text)
    blob_name = _projections_blob_name(slate_date, slate_key=slate_key)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _read_projections_csv(store: CbbGcsStore, slate_date: date, slate_key: str | None = None) -> str | None:
    reader = getattr(store, "read_projections_csv", None)
    if callable(reader):
        try:
            return reader(slate_date, slate_key)
        except TypeError:
            return reader(slate_date)
    blob = store.bucket.blob(_projections_blob_name(slate_date))
    if blob.exists():
        return blob.download_as_text(encoding="utf-8")
    return None


def _ownership_blob_name(slate_date: date, slate_key: str | None = None) -> str:
    # Canonical date-level storage; slate_key intentionally ignored.
    return f"cbb/ownership/{slate_date.isoformat()}_ownership.csv"


def _read_ownership_csv(store: CbbGcsStore, slate_date: date, slate_key: str | None = None) -> str | None:
    reader = getattr(store, "read_ownership_csv", None)
    if callable(reader):
        try:
            return reader(slate_date, slate_key)
        except TypeError:
            return reader(slate_date)
    blob = store.bucket.blob(_ownership_blob_name(slate_date))
    if blob.exists():
        return blob.download_as_text(encoding="utf-8")
    return None


def _write_ownership_csv(
    store: CbbGcsStore,
    slate_date: date,
    csv_text: str,
    slate_key: str | None = None,
) -> str:
    writer = getattr(store, "write_ownership_csv", None)
    if callable(writer):
        try:
            return writer(slate_date, csv_text, slate_key)
        except TypeError:
            return writer(slate_date, csv_text)
    blob_name = _ownership_blob_name(slate_date, slate_key=slate_key)
    blob = store.bucket.blob(blob_name)
    blob.upload_from_string(csv_text, content_type="text/csv")
    return blob_name


def _contest_standings_blob_name(
    slate_date: date,
    contest_id: str,
    slate_key: str | None = None,
) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(contest_id or "").strip())
    safe = safe or "contest"
    # Canonical date+contest storage; slate_key intentionally ignored.
    return f"cbb/contest_standings/{slate_date.isoformat()}_{safe}.csv"


def _read_contest_standings_csv(
    store: CbbGcsStore,
    slate_date: date,
    contest_id: str,
    slate_key: str | None = None,
) -> str | None:
    reader = getattr(store, "read_contest_standings_csv", None)
    if callable(reader):
        try:
            return reader(slate_date, contest_id, slate_key)
        except TypeError:
            return reader(slate_date, contest_id)
    blob = store.bucket.blob(_contest_standings_blob_name(slate_date, contest_id))
    if blob.exists():
        return blob.download_as_text(encoding="utf-8")
    return None


def _write_contest_standings_csv(
    store: CbbGcsStore,
    slate_date: date,
    contest_id: str,
    csv_text: str,
    slate_key: str | None = None,
) -> str:
    writer = getattr(store, "write_contest_standings_csv", None)
    if callable(writer):
        try:
            return writer(slate_date, contest_id, csv_text, slate_key)
        except TypeError:
            return writer(slate_date, contest_id, csv_text)
    blob_name = _contest_standings_blob_name(slate_date, contest_id, slate_key=slate_key)
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


def _annotate_lineups_with_version_metadata(
    lineups: list[dict[str, Any]] | None,
    *,
    version_key: str,
    version_label: str,
    lineup_strategy: str,
    model_profile: str,
    include_tail_signals: bool,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for lineup in (lineups or []):
        if not isinstance(lineup, dict):
            continue
        one = dict(lineup)
        one["version_key"] = str(one.get("version_key") or version_key)
        one["version_label"] = str(one.get("version_label") or version_label or version_key)
        one["lineup_model_key"] = str(one.get("lineup_model_key") or version_key)
        one["lineup_model_label"] = str(one.get("lineup_model_label") or version_label or version_key)
        one["lineup_strategy"] = str(one.get("lineup_strategy") or lineup_strategy or "")
        one["model_profile"] = str(one.get("model_profile") or model_profile or "")
        one["include_tail_signals"] = bool(one.get("include_tail_signals", include_tail_signals))
        ceiling_projection = _safe_float_value(one.get("ceiling_projection"), default=float("nan"))
        if math.isnan(ceiling_projection):
            ceiling_projection = round(_safe_float_value(one.get("projected_points"), 0.0) * 1.18, 2)
        one["ceiling_projection"] = round(float(ceiling_projection), 2)
        annotated.append(one)
    return annotated


def _build_lineup_versions_export_frame(
    generated_versions: dict[str, Any] | None,
    *,
    run_id: str = "",
) -> pd.DataFrame:
    versions = generated_versions if isinstance(generated_versions, dict) else {}
    export_lineups: list[dict[str, Any]] = []
    for version_key, version_data in versions.items():
        version_dict = version_data if isinstance(version_data, dict) else {}
        annotated = _annotate_lineups_with_version_metadata(
            version_dict.get("lineups") or [],
            version_key=str(version_key),
            version_label=str(version_dict.get("version_label") or version_key),
            lineup_strategy=str(version_dict.get("lineup_strategy") or ""),
            model_profile=str(version_dict.get("model_profile") or ""),
            include_tail_signals=bool(version_dict.get("include_tail_signals", False)),
        )
        export_lineups.extend(annotated)

    if not export_lineups:
        return pd.DataFrame()

    export_df = lineups_slots_frame(export_lineups)
    if export_df.empty:
        export_df = lineups_summary_frame(export_lineups)
    if export_df.empty:
        return export_df

    export_df = export_df.copy()
    if run_id:
        export_df.insert(0, "Run ID", str(run_id))
    return export_df


def persist_lineup_run_bundle(
    store: CbbGcsStore,
    slate_date: date,
    bundle: dict[str, Any],
    slate_key: str | None = None,
    slate_label: str | None = None,
) -> dict[str, Any]:
    run_id = str(bundle.get("run_id") or _new_lineup_run_id())
    generated_at_utc = str(bundle.get("generated_at_utc") or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    settings = _json_safe(bundle.get("settings") or {})
    resolved_slate_key = _slate_key_from_label(
        slate_key or bundle.get("slate_key") or bundle.get("slate_label"),
        default="main",
    )
    resolved_slate_label = _normalize_slate_label(
        slate_label or bundle.get("slate_label") or resolved_slate_key.replace("_", " ").title(),
        default="Main",
    )
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
            "slate_key": resolved_slate_key,
            "slate_label": resolved_slate_label,
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
        try:
            json_blob = store.write_lineup_version_json(
                slate_date,
                run_id,
                version_name,
                payload,
                slate_key=resolved_slate_key,
            )
            csv_blob = store.write_lineup_version_csv(
                slate_date,
                run_id,
                version_name,
                lineups_csv,
                slate_key=resolved_slate_key,
            )
            upload_blob = store.write_lineup_version_upload_csv(
                slate_date,
                run_id,
                version_name,
                upload_csv,
                slate_key=resolved_slate_key,
            )
        except TypeError:
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
        "slate_key": resolved_slate_key,
        "slate_label": resolved_slate_label,
        "generated_at_utc": generated_at_utc,
        "run_mode": str(bundle.get("run_mode") or "single"),
        "settings": settings,
        "versions": version_entries,
    }
    try:
        manifest_blob = store.write_lineup_run_manifest_json(
            slate_date,
            run_id,
            manifest,
            slate_key=resolved_slate_key,
        )
    except TypeError:
        manifest_blob = store.write_lineup_run_manifest_json(slate_date, run_id, manifest)
    return {
        "run_id": run_id,
        "slate_key": resolved_slate_key,
        "slate_label": resolved_slate_label,
        "manifest_blob": manifest_blob,
        "version_count": len(version_entries),
        "manifest": manifest,
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_saved_lineup_run_manifests(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
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
        if isinstance(payload, dict):
            if selected_slate_key:
                manifest_slate_key = _slate_key_from_label(
                    payload.get("slate_key") or payload.get("slate_label"),
                    default="main",
                )
                if manifest_slate_key != _slate_key_from_label(selected_slate_key):
                    continue
            manifests.append(payload)
    manifests.sort(key=lambda x: str(x.get("generated_at_utc") or ""), reverse=True)
    return manifests


@st.cache_data(ttl=300, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
def compute_projection_calibration_from_phantom(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
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
    try:
        run_dates = store.list_lineup_run_dates(selected_slate_key)
    except TypeError:
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
        try:
            run_ids = store.list_lineup_run_ids(run_date, selected_slate_key)
        except TypeError:
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
    selected_slate_key: str | None,
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
            slate_key=selected_slate_key,
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


def _default_role_bucket_calibration(
    lookback_days: int,
    min_samples_per_bucket: int,
) -> dict[str, Any]:
    bucket_rows = [
        {
            "role_bucket": key,
            "role_bucket_label": PROJECTION_ROLE_BUCKET_LABELS.get(key, key),
            "samples": 0,
            "avg_actual": 0.0,
            "avg_projection": 0.0,
            "avg_error": 0.0,
            "mae": 0.0,
            "raw_scale": 1.0,
            "scale": 1.0,
            "used_for_adjustment": False,
        }
        for key in PROJECTION_ROLE_BUCKET_ORDER
    ]
    return {
        "lookback_days": int(max(0, lookback_days)),
        "min_samples_per_bucket": int(max(1, min_samples_per_bucket)),
        "used_dates": 0,
        "player_rows": 0,
        "scales": {key: 1.0 for key in PROJECTION_ROLE_BUCKET_ORDER},
        "bucket_rows": bucket_rows,
    }


@st.cache_data(ttl=300, show_spinner=False)
def compute_projection_role_bucket_calibration(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
    lookback_days: int,
    min_samples_per_bucket: int,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> dict[str, Any]:
    lookback = int(max(0, lookback_days))
    min_samples = int(max(1, min_samples_per_bucket))
    default_result = _default_role_bucket_calibration(lookback, min_samples)

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
            slate_key=selected_slate_key,
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
        if "Position" in comp.columns:
            comp["Position"] = comp["Position"].astype(str)
        else:
            comp["Position"] = ""
        comp["blended_projection"] = pd.to_numeric(comp.get("blended_projection"), errors="coerce")
        comp["actual_dk_points"] = pd.to_numeric(comp.get("actual_dk_points"), errors="coerce")
        comp["blend_error"] = pd.to_numeric(comp.get("blend_error"), errors="coerce")
        comp = comp.loc[
            comp["blended_projection"].notna()
            & comp["actual_dk_points"].notna()
            & (comp["blended_projection"] > 0.0)
        ].copy()
        if comp.empty:
            cursor -= timedelta(days=1)
            continue

        comp["role_bucket"] = comp["Position"].map(projection_role_bucket_key)
        all_rows.append(comp[["role_bucket", "actual_dk_points", "blended_projection", "blend_error"]])
        used_dates += 1
        cursor -= timedelta(days=1)

    if not all_rows:
        return default_result

    full = pd.concat(all_rows, ignore_index=True)
    if full.empty:
        return default_result

    output_rows: list[dict[str, Any]] = []
    scales: dict[str, float] = {}
    for key in PROJECTION_ROLE_BUCKET_ORDER:
        seg = full.loc[full["role_bucket"] == key].copy()
        samples = int(len(seg))
        avg_actual = float(pd.to_numeric(seg.get("actual_dk_points"), errors="coerce").mean()) if samples else 0.0
        avg_projection = float(pd.to_numeric(seg.get("blended_projection"), errors="coerce").mean()) if samples else 0.0
        avg_error = float(pd.to_numeric(seg.get("blend_error"), errors="coerce").mean()) if samples else 0.0
        mae = float(pd.to_numeric(seg.get("blend_error"), errors="coerce").abs().mean()) if samples else 0.0
        raw_scale = 1.0
        if avg_projection > 0.0 and samples > 0:
            raw_scale = float(avg_actual / avg_projection)
        clipped_scale = float(min(1.20, max(0.80, raw_scale)))
        use_scale = clipped_scale if samples >= min_samples else 1.0
        scales[key] = float(use_scale)
        output_rows.append(
            {
                "role_bucket": key,
                "role_bucket_label": PROJECTION_ROLE_BUCKET_LABELS.get(key, key),
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
def compute_phantom_version_performance_weights(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
    lookback_days: int,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> dict[str, Any]:
    lookback = int(max(0, lookback_days))
    default = {
        "lookback_days": lookback,
        "used_dates": 0,
        "rows": 0,
        "weights": {},
        "version_stats": [],
    }
    if lookback <= 0:
        return default

    end_date = selected_date - timedelta(days=1)
    if end_date < date(2000, 1, 1):
        return default
    start_date = end_date - timedelta(days=max(1, lookback) - 1)

    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)
    try:
        run_dates = store.list_lineup_run_dates(selected_slate_key)
    except TypeError:
        run_dates = store.list_lineup_run_dates()
    candidate_dates = [d for d in run_dates if isinstance(d, date) and start_date <= d <= end_date]
    if not candidate_dates:
        return default

    all_summary_rows: list[pd.DataFrame] = []
    used_dates = 0
    for run_date in sorted(candidate_dates, reverse=True):
        try:
            run_ids = store.list_lineup_run_ids(run_date, selected_slate_key)
        except TypeError:
            run_ids = store.list_lineup_run_ids(run_date)
        day_rows: list[pd.DataFrame] = []
        for run_id in run_ids:
            payload = store.read_phantom_review_summary_json(run_date, run_id)
            if not isinstance(payload, dict):
                continue
            rows = payload.get("summary_rows")
            if not isinstance(rows, list) or not rows:
                continue
            frame = pd.DataFrame(rows)
            if frame.empty:
                continue
            frame["run_date"] = run_date.isoformat()
            day_rows.append(frame)
        if day_rows:
            all_summary_rows.append(pd.concat(day_rows, ignore_index=True))
            used_dates += 1

    if not all_summary_rows:
        return default

    full = pd.concat(all_summary_rows, ignore_index=True)
    if full.empty:
        return default
    if "version_key" not in full.columns:
        return default
    full["version_key"] = full["version_key"].astype(str).str.strip()
    full = full.loc[full["version_key"] != ""].copy()
    if full.empty:
        return default

    full["lineups"] = pd.to_numeric(full.get("lineups"), errors="coerce").fillna(0.0)
    full["avg_would_beat_pct"] = pd.to_numeric(full.get("avg_would_beat_pct"), errors="coerce").fillna(0.0)
    full["avg_actual_minus_projected"] = pd.to_numeric(full.get("avg_actual_minus_projected"), errors="coerce").fillna(0.0)
    full = full.loc[full["lineups"] > 0.0].copy()
    if full.empty:
        return default

    stats = (
        full.groupby("version_key", as_index=False)
        .agg(
            lineups=("lineups", "sum"),
            beat_mean=("avg_would_beat_pct", lambda s: float((pd.to_numeric(s, errors="coerce")).mean())),
            delta_mean=("avg_actual_minus_projected", lambda s: float((pd.to_numeric(s, errors="coerce")).mean())),
        )
        .sort_values("lineups", ascending=False)
        .reset_index(drop=True)
    )
    if stats.empty:
        return default

    beat_component = (pd.to_numeric(stats["beat_mean"], errors="coerce").fillna(0.0) / 100.0).clip(lower=0.0)
    delta_component = (
        1.0 + (pd.to_numeric(stats["delta_mean"], errors="coerce").fillna(0.0) / 80.0)
    ).clip(lower=0.20, upper=1.35)
    raw_score = ((0.75 * beat_component) + (0.25 * delta_component)).clip(lower=0.05)
    weighted_score = raw_score * pd.to_numeric(stats["lineups"], errors="coerce").fillna(0.0).clip(lower=1.0)

    total_weighted = float(weighted_score.sum())
    if total_weighted <= 0.0:
        return default
    version_weights = {
        str(row["version_key"]): float(weighted_score.iloc[idx] / total_weighted)
        for idx, row in stats.iterrows()
    }
    return {
        "lookback_days": lookback,
        "used_dates": int(used_dates),
        "rows": int(len(full)),
        "weights": version_weights,
        "version_stats": stats.to_dict(orient="records"),
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_saved_lineup_version_payload(
    bucket_name: str,
    selected_date: date,
    run_id: str,
    version_key: str,
    selected_slate_key: str | None,
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
    try:
        payload = store.read_lineup_version_json(selected_date, run_id, version_key, selected_slate_key)
    except TypeError:
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
    csv_text = _read_dk_slate_csv(store, selected_date, slate_key=slate_key)
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
    csv_text = _read_projections_csv(store, selected_date, slate_key=selected_slate_key)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


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
    csv_text = _read_ownership_csv(store, selected_date, slate_key=selected_slate_key)
    if not csv_text or not csv_text.strip():
        return pd.DataFrame(columns=["ID", "Name", "TeamAbbrev", "actual_ownership", "name_key", "name_key_loose"])
    df = pd.read_csv(io.StringIO(csv_text))
    return normalize_ownership_frame(df)


@st.cache_data(ttl=600, show_spinner=False)
def _read_contest_standings_ownership_history_for_date(
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
    prefix = f"{store.contest_standings_prefix}/{selected_date.isoformat()}_"
    try:
        blob_names = sorted(
            {
                str(blob.name or "")
                for blob in store.bucket.list_blobs(prefix=prefix)
                if str(blob.name or "").lower().endswith(".csv")
            }
        )
    except Exception:
        return pd.DataFrame(columns=["Name", "TeamAbbrev", "actual_ownership", "review_date"])

    frames: list[pd.DataFrame] = []
    for blob_name in blob_names:
        try:
            csv_text = store.bucket.blob(blob_name).download_as_text(encoding="utf-8")
        except Exception:
            continue
        if not csv_text or not csv_text.strip():
            continue
        try:
            standings_df = pd.read_csv(io.StringIO(csv_text))
        except Exception:
            continue
        if standings_df.empty:
            continue
        normalized = normalize_contest_standings_frame(standings_df)
        extracted = extract_actual_ownership_from_standings(normalized)
        if extracted.empty:
            continue
        one = extracted.rename(columns={"player_name": "Name"}).copy()
        one["Name"] = one["Name"].astype(str).str.strip()
        one["TeamAbbrev"] = ""
        one["actual_ownership"] = pd.to_numeric(one.get("actual_ownership"), errors="coerce")
        one["review_date"] = selected_date.isoformat()
        one = one.loc[(one["Name"] != "") & one["actual_ownership"].notna(), ["Name", "TeamAbbrev", "actual_ownership", "review_date"]]
        if not one.empty:
            frames.append(one)

    if not frames:
        return pd.DataFrame(columns=["Name", "TeamAbbrev", "actual_ownership", "review_date"])
    return pd.concat(frames, ignore_index=True)


@st.cache_data(ttl=900, show_spinner=False)
def load_historical_ownership_frame_for_date(
    bucket_name: str,
    selected_date: date,
    selected_slate_key: str | None,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> pd.DataFrame:
    season_start = season_start_for_date(selected_date)
    if selected_date <= season_start:
        return pd.DataFrame(columns=["Name", "TeamAbbrev", "actual_ownership", "review_date"])

    history_dates = iter_dates(season_start, selected_date - timedelta(days=1))
    frames: list[pd.DataFrame] = []
    for history_date in history_dates:
        own_df = load_ownership_frame_for_date(
            bucket_name=bucket_name,
            selected_date=history_date,
            selected_slate_key=selected_slate_key,
            gcp_project=gcp_project,
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
        )
        if own_df.empty and selected_slate_key:
            own_df = load_ownership_frame_for_date(
                bucket_name=bucket_name,
                selected_date=history_date,
                selected_slate_key=None,
                gcp_project=gcp_project,
                service_account_json=service_account_json,
                service_account_json_b64=service_account_json_b64,
            )
        if own_df.empty:
            own_df = _read_contest_standings_ownership_history_for_date(
                bucket_name=bucket_name,
                selected_date=history_date,
                gcp_project=gcp_project,
                service_account_json=service_account_json,
                service_account_json_b64=service_account_json_b64,
            )
        if own_df.empty:
            continue
        one = own_df.copy()
        if "TeamAbbrev" not in one.columns:
            one["TeamAbbrev"] = ""
        one["Name"] = one["Name"].astype(str).str.strip()
        one["TeamAbbrev"] = one["TeamAbbrev"].astype(str).str.strip().str.upper()
        one["actual_ownership"] = pd.to_numeric(one.get("actual_ownership"), errors="coerce")
        review_date_series = pd.to_datetime(
            one.get("review_date", pd.Series([pd.NA] * len(one), index=one.index)),
            errors="coerce",
        )
        one["review_date"] = review_date_series.where(
            review_date_series.notna(),
            pd.Timestamp(history_date),
        )
        one = one.loc[(one["Name"] != "") & one["actual_ownership"].notna(), ["Name", "TeamAbbrev", "actual_ownership", "review_date"]]
        if not one.empty:
            frames.append(one)

    if not frames:
        return pd.DataFrame(columns=["Name", "TeamAbbrev", "actual_ownership", "review_date"])
    return pd.concat(frames, ignore_index=True)


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


@st.cache_data(ttl=300, show_spinner=False)
def load_rotowire_slates_frame(
    site_id: int,
    cookie_header: str | None,
) -> pd.DataFrame:
    client = RotoWireClient(cookie_header=(cookie_header or None))
    try:
        catalog = client.fetch_slate_catalog(site_id=site_id)
    finally:
        client.close()
    return flatten_rotowire_slates(catalog, site_id=site_id)


@st.cache_data(ttl=300, show_spinner=False)
def load_rotowire_players_frame(
    site_id: int,
    slate_id: int,
    slate_date: str | None,
    contest_type: str | None,
    slate_name: str | None,
    cookie_header: str | None,
) -> pd.DataFrame:
    client = RotoWireClient(cookie_header=(cookie_header or None))
    try:
        raw_players = client.fetch_players(slate_id=int(slate_id))
    finally:
        client.close()
    slate_row = {
        "site_id": site_id,
        "slate_id": int(slate_id),
        "slate_date": slate_date,
        "contest_type": contest_type,
        "slate_name": slate_name,
    }
    return normalize_rotowire_players(raw_players, slate_row=slate_row)


@st.cache_data(ttl=600, show_spinner=False)
def load_contest_standings_frame(
    bucket_name: str,
    selected_date: date,
    contest_id: str,
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
    csv_text = _read_contest_standings_csv(
        store,
        selected_date,
        contest_id,
        slate_key=selected_slate_key,
    )
    if not csv_text or not csv_text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(csv_text))


def build_optimizer_pool_for_date(
    bucket_name: str,
    slate_date: date,
    slate_key: str | None,
    bookmaker: str | None,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
    recent_form_games: int = 7,
    recent_points_weight: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    slate_df = load_dk_slate_frame_for_date(
        bucket_name=bucket_name,
        selected_date=slate_date,
        slate_key=slate_key,
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
    ownership_history_df = load_historical_ownership_frame_for_date(
        bucket_name=bucket_name,
        selected_date=slate_date,
        selected_slate_key=slate_key,
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
        ownership_history_df=ownership_history_df,
        bookmaker_filter=(bookmaker or None),
        odds_games_df=odds_scored_df,
        recent_form_games=int(max(1, recent_form_games)),
        recent_points_weight=float(max(0.0, min(1.0, recent_points_weight))),
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

active_slate_default = str(st.session_state.get("shared_slate_preset", "Main"))
if active_slate_default not in SLATE_PRESET_OPTIONS:
    active_slate_default = "Main"
shared_slate_preset = st.selectbox(
    "Active Slate Label",
    options=SLATE_PRESET_OPTIONS,
    index=SLATE_PRESET_OPTIONS.index(active_slate_default),
    key="shared_slate_preset",
    help=(
        "Shared across DK Slate, Slate + Vegas, Lineup Generator, "
        "Projection Review, Tournament Review, and Game Slate Agent."
    ),
)
shared_slate_custom = str(st.session_state.get("shared_slate_custom_label", "Main"))
if shared_slate_preset == "Custom":
    shared_slate_custom = st.text_input(
        "Custom Active Slate Label",
        value=shared_slate_custom or "Main",
        key="shared_slate_custom_label",
        help="Example: Early, Turbo, Showdown.",
    )
shared_slate_label = _normalize_slate_label(
    shared_slate_custom if shared_slate_preset == "Custom" else shared_slate_preset
)
shared_slate_key = _slate_key_from_label(shared_slate_label)
st.caption(f"Active slate context: `{shared_slate_label}` (key: `{shared_slate_key}`)")

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

tab_game, tab_props, tab_backfill, tab_dk, tab_injuries, tab_slate_vegas, tab_rotowire, tab_lineups, tab_projection_review, tab_tournament_review = st.tabs(
    [
        "Game Data",
        "Prop Data",
        "Backfill",
        "DK Slate",
        "Injuries",
        "Slate + Vegas",
        "RotoWire Scraper",
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
    dk_slate_label = shared_slate_label
    dk_slate_key = shared_slate_key
    st.caption(
        "Each date + slate label stores a separate DK slate file. "
        f"Using active slate `{dk_slate_label}` (key: `{dk_slate_key}`)."
    )
    uploaded_dk_slate = st.file_uploader(
        "Upload DraftKings Slate CSV",
        type=["csv"],
        key="dk_slate_upload",
        help="Upload the DraftKings player/salary slate CSV for this date+slate.",
    )
    delete_dk_slate_confirm = st.checkbox(
        "Confirm delete for selected date + slate",
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
            st.error("Check `Confirm delete for selected date + slate` before deleting.")
        else:
            with st.spinner("Deleting DraftKings slate from GCS..."):
                try:
                    client = build_storage_client(
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                        project=gcp_project or None,
                    )
                    store = CbbGcsStore(bucket_name=bucket_name, client=client)
                    deleted, blob_name = _delete_dk_slate_csv(store, dk_slate_date, slate_key=dk_slate_key)
                    load_dk_slate_frame_for_date.clear()
                    if deleted:
                        st.session_state.pop("cbb_dk_upload_summary", None)
                        st.success(f"Deleted `{blob_name}`")
                    else:
                        st.warning(
                            "No cached slate found for selected date+slate. "
                            f"Expected `{blob_name}`"
                        )
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
                        blob_name = _write_dk_slate_csv(
                            store,
                            dk_slate_date,
                            csv_text,
                            slate_key=dk_slate_key,
                        )
                        load_dk_slate_frame_for_date.clear()
                        st.session_state["cbb_dk_upload_summary"] = {
                            "slate_date": dk_slate_date.isoformat(),
                            "slate_label": dk_slate_label,
                            "slate_key": dk_slate_key,
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
                slate_key=dk_slate_key,
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
            )
            if dk_df.empty:
                st.warning(
                    "No cached DraftKings slate found for selected date+slate. "
                    "Upload a CSV first."
                )
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
    sv1, sv2 = st.columns(2)
    slate_recent_form_games = int(
        sv1.slider(
            "Pool Recent Form Games",
            min_value=3,
            max_value=12,
            value=5,
            step=1,
            key="slate_vegas_recent_form_games",
            help="Rolling game window used for recent minutes/points signals in the player pool.",
        )
    )
    slate_recent_points_weight_pct = float(
        sv2.slider(
            "Pool Recent Points Weight %",
            min_value=0,
            max_value=100,
            value=35,
            step=1,
            key="slate_vegas_recent_points_weight_pct",
            help="Blends season points with recent-form points in pool projections.",
        )
    )
    slate_recent_points_weight = slate_recent_points_weight_pct / 100.0
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
                slate_key=shared_slate_key,
                bookmaker=(vegas_bookmaker.strip() or None),
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
                recent_form_games=slate_recent_form_games,
                recent_points_weight=slate_recent_points_weight,
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
                mins_recent = pd.to_numeric(pool_df.get("our_minutes_recent", pd.Series(dtype=float)), errors="coerce")
                avg_mins_recent = float(mins_recent.mean()) if len(mins_recent) and mins_recent.notna().any() else 0.0
                st.caption(
                    f"Season stats rows used: `{len(season_history_df):,}` | "
                    f"Average projected minutes: `{avg_mins:.1f}` | "
                    f"Average recent minutes ({slate_recent_form_games}g): `{avg_mins_recent:.1f}`"
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
                        "our_minutes_recent",
                        "our_points_recent",
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
                        "our_minutes_recent",
                        "our_points_recent",
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
                            slate_key=shared_slate_key,
                        )
                        st.success(
                            f"Saved projections to `{blob_name}` "
                            "(same date overwrites; shared across slates)."
                        )
                    st.download_button(
                        "Download Active Pool CSV",
                        data=display_pool.to_csv(index=False),
                        file_name=f"cbb_active_pool_{slate_vegas_date.isoformat()}.csv",
                        mime="text/csv",
                        key="download_active_pool_csv",
                    )
        except Exception as exc:
            st.exception(exc)

with tab_rotowire:
    st.subheader("RotoWire Slate Export")
    st.caption(
        "Fetch RotoWire optimizer projections and expected minutes for a selected slate, "
        "then download CSV or JSON."
    )
    rotowire_site_id = 1
    default_rotowire_cookie = (_resolve_rotowire_cookie() or "").strip()
    rw1, rw2, rw3 = st.columns(3)
    rotowire_selected_date = rw1.date_input(
        "RotoWire Slate Date",
        value=game_selected_date,
        key="rotowire_slate_date",
    )
    rotowire_contest_type = rw2.selectbox(
        "Contest Type Filter",
        options=["All", "Classic", "Showdown"],
        index=1,
        key="rotowire_contest_type",
    )
    rotowire_name_filter = rw3.text_input(
        "Slate Name Filter",
        value=_default_rotowire_slate_name(shared_slate_label),
        key="rotowire_name_filter",
        help="Optional substring match on slate name, for example: Night, Afternoon, or All.",
    )
    rotowire_cookie_header = st.text_input(
        "RotoWire Cookie Header (optional)",
        value=default_rotowire_cookie,
        type="password",
        key="rotowire_cookie_header",
        help="Paste the full Cookie header only if the endpoint requires member authentication.",
    )
    rr1, rr2 = st.columns(2)
    refresh_rotowire_clicked = rr1.button("Refresh RotoWire Data", key="refresh_rotowire_data")
    if refresh_rotowire_clicked:
        load_rotowire_slates_frame.clear()
        load_rotowire_players_frame.clear()
    rr2.caption(
        "Cookie source: loaded from secrets/env."
        if default_rotowire_cookie
        else "Cookie source: not set."
    )

    try:
        rotowire_cookie_value = rotowire_cookie_header.strip() or None
        rotowire_slates_df = load_rotowire_slates_frame(
            site_id=rotowire_site_id,
            cookie_header=rotowire_cookie_value,
        )
        if rotowire_slates_df.empty:
            st.warning("No RotoWire slates returned for DraftKings.")
        else:
            filtered_rotowire_slates = rotowire_slates_df.loc[
                rotowire_slates_df["slate_date"].astype(str) == rotowire_selected_date.isoformat()
            ].copy()
            if rotowire_contest_type != "All":
                filtered_rotowire_slates = filtered_rotowire_slates.loc[
                    filtered_rotowire_slates["contest_type"].astype(str).str.lower()
                    == rotowire_contest_type.strip().lower()
                ]
            if rotowire_name_filter.strip():
                filtered_rotowire_slates = filtered_rotowire_slates.loc[
                    filtered_rotowire_slates["slate_name"].astype(str).str.contains(
                        rotowire_name_filter.strip(),
                        case=False,
                        na=False,
                    )
                ]

            st.caption(
                f"Matched `{len(filtered_rotowire_slates)}` of `{len(rotowire_slates_df)}` "
                "available DraftKings slates."
            )
            slate_show_cols = [
                "slate_id",
                "contest_type",
                "slate_name",
                "start_datetime",
                "end_datetime",
                "game_count",
            ]
            st.dataframe(
                filtered_rotowire_slates[[c for c in slate_show_cols if c in filtered_rotowire_slates.columns]],
                hide_index=True,
                use_container_width=True,
            )

            if filtered_rotowire_slates.empty:
                st.info("Adjust the filters above to select a slate.")
            else:
                rotowire_slate_labels = {
                    int(row["slate_id"]): (
                        f"{row['contest_type']} | {row['slate_name']} | "
                        f"{row['start_datetime']} | slate {int(row['slate_id'])}"
                    )
                    for _, row in filtered_rotowire_slates.iterrows()
                }
                selected_rotowire_slate_id = st.selectbox(
                    "Selected RotoWire Slate",
                    options=list(rotowire_slate_labels.keys()),
                    format_func=lambda slate_id: rotowire_slate_labels.get(int(slate_id), str(slate_id)),
                    key="rotowire_selected_slate_id",
                )
                selected_rotowire_slate = filtered_rotowire_slates.loc[
                    filtered_rotowire_slates["slate_id"] == int(selected_rotowire_slate_id)
                ].iloc[0]
                rotowire_players_df = load_rotowire_players_frame(
                    site_id=rotowire_site_id,
                    slate_id=int(selected_rotowire_slate_id),
                    slate_date=str(selected_rotowire_slate.get("slate_date") or ""),
                    contest_type=str(selected_rotowire_slate.get("contest_type") or ""),
                    slate_name=str(selected_rotowire_slate.get("slate_name") or ""),
                    cookie_header=rotowire_cookie_value,
                )
                rp1, rp2, rp3 = st.columns(3)
                rp1.metric("RotoWire Players", int(len(rotowire_players_df)))
                proj_mean = pd.to_numeric(
                    rotowire_players_df.get("proj_fantasy_points", pd.Series(dtype=float)),
                    errors="coerce",
                )
                mins_mean = pd.to_numeric(
                    rotowire_players_df.get("proj_minutes", pd.Series(dtype=float)),
                    errors="coerce",
                )
                rp2.metric(
                    "Avg Projected FPTS",
                    f"{float(proj_mean.mean()):.2f}" if len(proj_mean) and proj_mean.notna().any() else "0.00",
                )
                rp3.metric(
                    "Avg Projected Minutes",
                    f"{float(mins_mean.mean()):.1f}" if len(mins_mean) and mins_mean.notna().any() else "0.0",
                )

                rotowire_show_cols = [
                    "player_name",
                    "team_abbr",
                    "opp_abbr",
                    "site_positions",
                    "salary",
                    "proj_fantasy_points",
                    "proj_minutes",
                    "proj_value_per_1k",
                    "avg_fpts_last5",
                    "avg_fpts_season",
                    "usage_rate",
                    "implied_points",
                    "spread",
                    "over_under",
                    "injury_status",
                ]
                rotowire_display = rotowire_players_df[
                    [c for c in rotowire_show_cols if c in rotowire_players_df.columns]
                ].copy()
                numeric_rotowire_cols = [
                    "salary",
                    "proj_fantasy_points",
                    "proj_minutes",
                    "proj_value_per_1k",
                    "avg_fpts_last5",
                    "avg_fpts_season",
                    "usage_rate",
                    "implied_points",
                    "spread",
                    "over_under",
                ]
                for col in numeric_rotowire_cols:
                    if col in rotowire_display.columns:
                        rotowire_display[col] = pd.to_numeric(rotowire_display[col], errors="coerce")
                st.dataframe(rotowire_display, hide_index=True, use_container_width=True)

                export_stub = _slate_key_from_label(
                    f"{selected_rotowire_slate.get('slate_date')}_{selected_rotowire_slate.get('contest_type')}_{selected_rotowire_slate.get('slate_name')}",
                    default="rotowire",
                )
                export_csv = rotowire_players_df.to_csv(index=False)
                export_json = json.dumps(rotowire_players_df.to_dict(orient="records"), indent=2)
                rd1, rd2 = st.columns(2)
                rd1.download_button(
                    "Download RotoWire CSV",
                    data=export_csv,
                    file_name=f"rotowire_{export_stub}.csv",
                    mime="text/csv",
                    key="download_rotowire_csv",
                )
                rd2.download_button(
                    "Download RotoWire JSON",
                    data=export_json,
                    file_name=f"rotowire_{export_stub}.json",
                    mime="application/json",
                    key="download_rotowire_json",
                )
    except Exception as exc:
        st.exception(exc)

with tab_lineups:
    st.subheader("DK Lineup Generator")
    if "auto_save_runs_to_gcs" not in st.session_state:
        st.session_state["auto_save_runs_to_gcs"] = True
    auto_save_runs_to_gcs = bool(st.session_state.get("auto_save_runs_to_gcs", True))
    lineup_slate_date = st.date_input("Lineup Slate Date", value=game_selected_date, key="lineup_slate_date")
    lineup_slate_label = shared_slate_label
    lineup_slate_key = shared_slate_key
    st.caption(f"Using slate `{lineup_slate_label}` (key: `{lineup_slate_key}`) for lineup reads/saves.")
    lineup_bookmaker = st.text_input(
        "Lineup Bookmaker Source",
        value=(default_bookmakers_filter.strip() or "fanduel"),
        key="lineup_bookmaker_source",
    )
    rf1, rf2 = st.columns(2)
    recent_form_games = int(
        rf1.slider(
            "Recent Form Games",
            min_value=3,
            max_value=12,
            value=int(st.session_state.get("slate_vegas_recent_form_games", 5)),
            step=1,
            help=(
                "Rolling game window used for recency signals. "
                "Smaller windows react faster to role/minutes changes."
            ),
        )
    )
    recent_points_weight_pct = float(
        rf2.slider(
            "Recent Points Weight %",
            min_value=0,
            max_value=100,
            value=int(st.session_state.get("slate_vegas_recent_points_weight_pct", 30)),
            step=1,
            help=(
                "Blends season points with recent-form points before DK scoring. "
                "0 = season-only, 100 = recent-only."
            ),
        )
    )
    recent_points_weight = recent_points_weight_pct / 100.0
    c1, c2, c3, c4 = st.columns(4)
    lineup_count = int(c1.slider("Lineups", min_value=1, max_value=150, value=20, step=1))
    contest_type = c2.selectbox("Contest Type", options=["Cash", "Small GPP", "Large GPP"], index=2)
    lineup_seed = int(c3.number_input("Random Seed", min_value=1, max_value=999999, value=7, step=1))
    run_mode_label = c4.selectbox(
        "Run Mode",
        options=["Single Version", "All Versions"],
        index=0,
        help=(
            "All Versions generates and saves all lineup models: "
            f"{ALL_VERSIONS_KEYS_TEXT}."
        ),
    )
    run_mode_key = "all" if run_mode_label == "All Versions" else "single"

    c5, c6, c7 = st.columns(3)
    if run_mode_key == "single":
        default_model_index = 0
        if _contest_is_gpp(contest_type):
            default_model_index = next(
                (
                    idx
                    for idx, cfg in enumerate(LINEUP_MODEL_REGISTRY)
                    if str(cfg.get("version_key") or "") == "salary_efficiency_ceiling_v1"
                ),
                0,
            )
        lineup_model_label = c5.selectbox(
            "Lineup Model",
            options=[str(cfg["label"]) for cfg in LINEUP_MODEL_REGISTRY],
            index=default_model_index,
            help=(
                "Run one lineup model. Use All Versions to save all models in a single run."
            ),
        )
        selected_model_key = LINEUP_MODEL_BY_LABEL.get(lineup_model_label, "standard_v1")
        selected_model_cfg = LINEUP_MODEL_BY_KEY.get(selected_model_key, LINEUP_MODEL_BY_KEY["standard_v1"])
        lineup_strategy = str(selected_model_cfg.get("lineup_strategy") or "standard")
        include_tail_signals = bool(selected_model_cfg.get("include_tail_signals", False))
    else:
        c5.caption("Lineup Models")
        c5.write(
            f"All Versions: `{ALL_VERSIONS_KEYS_TEXT.replace(', ', '`, `')}`"
        )
        selected_model_key = "standard_v1"
        lineup_strategy = "standard"
        include_tail_signals = False

    if run_mode_key == "all":
        st.caption(
            "Default All Versions allocation targets (before optional phantom promotion): "
            f"{ALL_VERSIONS_WEIGHT_TEXT}."
        )
    apply_gpp_variance_presets = bool(
        st.checkbox(
            "Apply GPP Variance Presets by Model",
            value=True,
            help=(
                "For Small/Large GPP, each lineup model gets its own variance-oriented controls "
                "(low-own share, ceiling share, salary-left target, and stack bias bonus)."
            ),
        )
    )
    if apply_gpp_variance_presets and _contest_is_gpp(contest_type):
        st.caption(
            "GPP variance presets are active: each model runs with distinct variance controls."
        )

    game_packet_for_lineups = st.session_state.get("cbb_game_slate_ai_packet")
    game_packet_ready = isinstance(game_packet_for_lineups, dict) and bool(game_packet_for_lineups)
    ga1, ga2, ga3 = st.columns(3)
    apply_game_agent_stack_bias = bool(
        ga1.checkbox(
            "Apply Game Agent Stack Bias",
            value=True,
            disabled=not game_packet_ready,
            help=(
                "Uses Game Slate packet stack targets to boost objective scores for matching games, "
                "teams, and players before lineup construction."
            ),
        )
    )
    game_agent_bias_strength_pct = float(
        ga2.slider(
            "Agent Bias Strength %",
            min_value=0,
            max_value=200,
            value=100,
            step=5,
            disabled=not apply_game_agent_stack_bias,
            help="100% uses default boost weights; lower softens, higher amplifies.",
        )
    )
    game_agent_focus_games = int(
        ga3.slider(
            "Agent Focus Games",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            disabled=not apply_game_agent_stack_bias,
            help="How many top stack games to prioritize from the packet.",
        )
    )
    if game_packet_ready:
        packet_review_date = str((game_packet_for_lineups.get("review_context") or {}).get("review_date") or "")
        if packet_review_date:
            st.caption(f"Game Agent packet loaded for `{packet_review_date}`.")
    else:
        st.caption("Game Agent stack bias is disabled until you build a packet in `Slate + Vegas`.")

    max_salary_left = int(
        c6.slider(
            "Max Salary Left Per Lineup",
            min_value=0,
            max_value=10000,
            value=400,
            step=50,
            help="Lineups must use at least 50000 - this value in salary.",
        )
    )
    global_max_exposure_pct = float(
        c7.slider(
            "Global Max Player Exposure %",
            min_value=0,
            max_value=100,
            value=50,
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
            value=200,
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
    rb1, rb2 = st.columns(2)
    auto_role_bucket_calibration = bool(
        rb1.checkbox(
            "Role-Bucket Residual Calibration",
            value=True,
            help="Apply per-role projection scales (Guard/Forward/Center/Other) from recent residuals.",
        )
    )
    role_calibration_min_samples = int(
        rb2.slider(
            "Min Samples Per Role Bucket",
            min_value=5,
            max_value=120,
            value=25,
            step=1,
            help="Role buckets below this count stay at scale 1.0.",
        )
    )
    role_calibration_lookback_days = int(max(7, calibration_lookback_days))
    if auto_role_bucket_calibration:
        role_calibration_lookback_days = int(
            st.slider(
                "Role-Bucket Calibration Lookback Days",
                min_value=7,
                max_value=90,
                value=int(max(7, calibration_lookback_days)),
                step=1,
                help="Uses prior slates with both projections and final results to calibrate role-level drift.",
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
            value=15,
            step=1,
            help="Base shrink applied proportionally to projection uncertainty score.",
        )
    )
    dnp_risk_threshold_pct = float(
        u3.slider(
            "DNP Risk Threshold %",
            min_value=10,
            max_value=60,
            value=35,
            step=1,
            help="Extra shrink applies above this DNP-risk level.",
        )
    )
    high_risk_extra_shrink_pct = float(
        u4.slider(
            "High-Risk Extra Shrink %",
            min_value=0,
            max_value=30,
            value=8,
            step=1,
            help="Additional shrink for players above the DNP-risk threshold.",
        )
    )
    og1, og2, og3, og4 = st.columns(4)
    apply_ownership_guardrails = bool(
        og1.checkbox(
            "Ownership Surprise Guardrails",
            value=True,
            help="Raises ownership floor for projected-low players with strong surge/chalk signals.",
        )
    )
    ownership_guardrail_proj_threshold = float(
        og2.slider(
            "Guardrail Projected Own <= %",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            disabled=not apply_ownership_guardrails,
        )
    )
    ownership_guardrail_surge_threshold = float(
        og3.slider(
            "Guardrail Surge Score >= ",
            min_value=40,
            max_value=95,
            value=78,
            step=1,
            disabled=not apply_ownership_guardrails,
        )
    )
    ownership_guardrail_floor_cap = float(
        og4.slider(
            "Guardrail Ownership Cap %",
            min_value=12,
            max_value=35,
            value=22,
            step=1,
            disabled=not apply_ownership_guardrails,
        )
    )
    lo1, lo2, lo3, lo4 = st.columns(4)
    low_own_bucket_exposure_pct = float(
        lo1.slider(
            "Low-Own Bucket Exposure %",
            min_value=0,
            max_value=80,
            value=24,
            step=1,
            help="Portion of lineups that must include at least one low-owned upside candidate.",
        )
    )
    low_own_bucket_min_per_lineup = int(
        lo2.slider(
            "Low-Own Min Players",
            min_value=0,
            max_value=3,
            value=1,
            step=1,
            disabled=low_own_bucket_exposure_pct <= 0.0,
        )
    )
    low_own_bucket_max_projected_ownership = float(
        lo3.slider(
            "Low-Own Max Projected Own %",
            min_value=3,
            max_value=20,
            value=14,
            step=1,
            disabled=low_own_bucket_exposure_pct <= 0.0,
        )
    )
    low_own_bucket_min_projection = float(
        lo4.slider(
            "Low-Own Min Projection",
            min_value=10,
            max_value=40,
            value=20,
            step=1,
            disabled=low_own_bucket_exposure_pct <= 0.0,
        )
    )
    low_own_bucket_min_tail_score = 55.0
    low_own_bucket_objective_bonus = 1.3
    preferred_game_bonus = 0.6
    max_unsupported_false_chalk_per_lineup = int(
        st.slider(
            "Max Non-Focus False Chalk",
            min_value=0,
            max_value=4,
            value=2,
            step=1,
            help=(
                "Hard cap on unsupported high-owned plays outside the focus games. "
                "Lower is stricter."
            ),
        )
    )
    cb1, cb2, cb3, cb4 = st.columns(4)
    ceiling_boost_lineup_pct = float(
        cb1.slider(
            "Ceiling Archetype Lineups %",
            min_value=0,
            max_value=80,
            value=25,
            step=1,
            help="Allocates a subset of lineups to more aggressive top-end construction scoring.",
        )
    )
    ceiling_boost_stack_bonus = float(
        cb2.slider(
            "Ceiling Stack Bonus",
            min_value=0.0,
            max_value=6.0,
            value=2.2,
            step=0.1,
            disabled=ceiling_boost_lineup_pct <= 0.0,
        )
    )
    ceiling_boost_salary_left_target = int(
        cb3.slider(
            "Ceiling Salary Left Target",
            min_value=0,
            max_value=500,
            value=120,
            step=10,
            disabled=ceiling_boost_lineup_pct <= 0.0,
        )
    )
    promote_phantom_constructions = bool(
        cb4.checkbox(
            "Promote Top Phantom Constructions",
            value=True,
            help="In All Versions mode, reallocate version lineup counts using recent phantom beat-rate.",
        )
    )
    phantom_promotion_lookback_days = int(
        st.slider(
            "Phantom Promotion Lookback Days",
            min_value=3,
            max_value=60,
            value=14,
            step=1,
            disabled=(not promote_phantom_constructions) or (run_mode_key != "all"),
            help="Used only for All Versions mode when promotion is enabled.",
        )
    )
    effective_max_salary_left = min(max_salary_left, 50) if strict_salary_utilization else max_salary_left
    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to generate lineups.")
    else:
        try:
            pool_df, removed_injured_df, raw_slate_df, _, _ = build_optimizer_pool_for_date(
                bucket_name=bucket_name,
                slate_date=lineup_slate_date,
                slate_key=lineup_slate_key,
                bookmaker=(lineup_bookmaker.strip() or None),
                gcp_project=gcp_project or None,
                service_account_json=cred_json,
                service_account_json_b64=cred_json_b64,
                recent_form_games=recent_form_games,
                recent_points_weight=recent_points_weight,
            )

            if raw_slate_df.empty:
                st.warning(
                    "No DraftKings slate found for selected optimizer date+slate. "
                    "Upload in `DK Slate` tab first."
                )
            elif pool_df.empty:
                st.warning("No players available after injury filtering. Check `Injuries` tab or slate date.")
            else:
                pool_sorted = pool_df.sort_values("projected_dk_points", ascending=False).copy()
                focus_settings = recommended_focus_stack_settings(pool_sorted, contest_type, focus_game_count=2)
                focus_summary_df = focus_settings.get("focus_summary")
                if not isinstance(focus_summary_df, pd.DataFrame):
                    focus_summary_df = build_game_focus_summary(pool_sorted)
                focus_summary_df = focus_summary_df.copy() if isinstance(focus_summary_df, pd.DataFrame) else pd.DataFrame()
                st.markdown("**Slate Thesis Check**")
                if focus_summary_df.empty:
                    st.info("No game-focus summary available for this slate.")
                    apply_focus_game_stack_guardrails = False
                    focus_game_count = 0
                    focus_game_stack_lineup_pct = 0.0
                    focus_game_stack_min_players = 0
                else:
                    thesis_view = focus_summary_df.loc[
                        :,
                        [
                            "game_key",
                            "game_stack_focus_score",
                            "projected_points_sum",
                            "projected_ownership_mean",
                            "historical_ownership_mean",
                            "game_tail_score_mean",
                            "game_total_line",
                            "recommended_stack_size",
                        ],
                    ].copy()
                    thesis_view = thesis_view.rename(
                        columns={
                            "game_key": "Game",
                            "game_stack_focus_score": "Focus Score",
                            "projected_points_sum": "Proj Sum",
                            "projected_ownership_mean": "Proj Own Avg",
                            "historical_ownership_mean": "Hist Own Avg",
                            "game_tail_score_mean": "Tail Avg",
                            "game_total_line": "Total",
                            "recommended_stack_size": "Rec Stack",
                        }
                    )
                    st.dataframe(thesis_view.head(6), hide_index=True, use_container_width=True)
                    focus_default_pct = int(round(float(focus_settings.get("stack_lineup_pct") or 0.0)))
                    focus_default_min_players = int(focus_settings.get("min_players") or 0)
                    focus_default_games = max(1, min(3, int(len(focus_summary_df))))
                    tg1, tg2, tg3, tg4 = st.columns(4)
                    apply_focus_game_stack_guardrails = bool(
                        tg1.checkbox(
                            "Apply Focus-Game Stack Guardrails",
                            value=bool(_contest_is_gpp(contest_type)),
                            help="Requires a share of lineups to include an actual stack from the top focus games.",
                        )
                    )
                    focus_game_count = int(
                        tg2.slider(
                            "Focus Games",
                            min_value=1,
                            max_value=max(1, min(6, int(len(focus_summary_df)))),
                            value=focus_default_games,
                            step=1,
                            disabled=not apply_focus_game_stack_guardrails,
                        )
                    )
                    focus_game_stack_lineup_pct = float(
                        tg3.slider(
                            "Focus-Game Stack Lineups %",
                            min_value=0,
                            max_value=100,
                            value=focus_default_pct,
                            step=5,
                            disabled=not apply_focus_game_stack_guardrails,
                        )
                    )
                    focus_game_stack_min_players = int(
                        tg4.slider(
                            "Focus-Game Min Players",
                            min_value=2,
                            max_value=4,
                            value=max(2, min(4, focus_default_min_players or 2)),
                            step=1,
                            disabled=not apply_focus_game_stack_guardrails,
                        )
                    )
                    selected_focus_games_preview = (
                        focus_summary_df.head(max(1, focus_game_count))["game_key"].astype(str).tolist()
                        if apply_focus_game_stack_guardrails
                        else []
                    )
                    if selected_focus_games_preview:
                        st.caption(
                            "Auto focus games for this slate: "
                            + ", ".join(f"`{g}`" for g in selected_focus_games_preview)
                        )
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
                    # Prevent stale prior-date run data from being shown/exported if generation fails mid-run.
                    st.session_state.pop("cbb_generated_run_bundle", None)
                    st.session_state.pop("cbb_active_version_key", None)
                    st.session_state.pop("cbb_generated_lineups", None)
                    st.session_state.pop("cbb_generated_lineups_warnings", None)
                    st.session_state.pop("cbb_generated_upload_csv", None)
                    st.session_state.pop("cbb_lineup_consistency_agent_output", None)
                    locked_ids = [label_to_id[x] for x in locked_labels]
                    excluded_ids = [label_to_id[x] for x in excluded_labels]
                    progress_text = st.empty()
                    progress_bar = st.progress(0, text="Starting lineup generation...")
                    projection_scale = 1.0
                    calibration_meta: dict[str, Any] = {}
                    projection_salary_bucket_scales: dict[str, float] = {}
                    salary_bucket_calibration_meta: dict[str, Any] = {}
                    projection_role_bucket_scales: dict[str, float] = {}
                    role_bucket_calibration_meta: dict[str, Any] = {}
                    uncertainty_weight = uncertainty_shrink_pct / 100.0
                    dnp_risk_threshold = dnp_risk_threshold_pct / 100.0
                    high_risk_extra_shrink = high_risk_extra_shrink_pct / 100.0
                    if auto_projection_calibration:
                        calibration_meta = compute_projection_calibration_from_phantom(
                            bucket_name=bucket_name,
                            selected_date=lineup_slate_date,
                            selected_slate_key=lineup_slate_key,
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
                            selected_slate_key=lineup_slate_key,
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
                    if auto_role_bucket_calibration:
                        role_bucket_calibration_meta = compute_projection_role_bucket_calibration(
                            bucket_name=bucket_name,
                            selected_date=lineup_slate_date,
                            selected_slate_key=lineup_slate_key,
                            lookback_days=role_calibration_lookback_days,
                            min_samples_per_bucket=role_calibration_min_samples,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        projection_role_bucket_scales = {
                            str(k): float(v)
                            for k, v in (role_bucket_calibration_meta.get("scales") or {}).items()
                            if str(k).strip()
                        }
                        role_rows = role_bucket_calibration_meta.get("bucket_rows") or []
                        role_applied = [
                            r
                            for r in role_rows
                            if bool(r.get("used_for_adjustment")) and abs(float(r.get("scale", 1.0)) - 1.0) >= 1e-6
                        ]
                        if role_applied:
                            role_summary = ", ".join(
                                f"{PROJECTION_ROLE_BUCKET_LABELS.get(str(r.get('role_bucket')), str(r.get('role_bucket')))}: {float(r.get('scale')):.3f}"
                                for r in role_applied
                            )
                            st.caption(
                                "Role-bucket calibration applied: "
                                f"{role_summary} "
                                f"(dates={int(role_bucket_calibration_meta.get('used_dates') or 0)}, "
                                f"rows={int(role_bucket_calibration_meta.get('player_rows') or 0)})."
                            )
                        else:
                            st.caption(
                                "Role-bucket calibration found no strong adjustments "
                                f"(dates={int(role_bucket_calibration_meta.get('used_dates') or 0)}, "
                                f"rows={int(role_bucket_calibration_meta.get('player_rows') or 0)})."
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
                    objective_score_adjustments: dict[str, float] = {}
                    game_agent_bias_meta: dict[str, Any] = {}
                    if apply_game_agent_stack_bias:
                        packet_review_date = str(
                            ((game_packet_for_lineups or {}).get("review_context") or {}).get("review_date") or ""
                        )
                        if packet_review_date and packet_review_date != lineup_slate_date.isoformat():
                            st.warning(
                                "Game Agent packet date does not match lineup slate date: "
                                f"packet=`{packet_review_date}`, lineup=`{lineup_slate_date.isoformat()}`."
                            )
                        objective_score_adjustments, game_agent_bias_meta = _build_game_agent_lineup_objective_adjustments(
                            packet=game_packet_for_lineups if isinstance(game_packet_for_lineups, dict) else {},
                            pool_df=pool_sorted,
                            contest_type=contest_type,
                            strength_multiplier=(game_agent_bias_strength_pct / 100.0),
                            focus_games=game_agent_focus_games,
                        )
                        if objective_score_adjustments:
                            st.caption(
                                "Game Agent stack bias applied: "
                                f"adjusted_players={int(game_agent_bias_meta.get('adjusted_players') or 0)}, "
                                f"games={int(game_agent_bias_meta.get('applied_games') or 0)}, "
                                f"teams={int(game_agent_bias_meta.get('applied_teams') or 0)}, "
                                f"players={int(game_agent_bias_meta.get('applied_players') or 0)}."
                            )
                            preview_rows = game_agent_bias_meta.get("preview_rows") or []
                            if preview_rows:
                                st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)
                        else:
                            st.caption(
                                "Game Agent stack bias enabled but no objective adjustments were applied "
                                f"(reason={str(game_agent_bias_meta.get('reason') or 'n/a')})."
                            )

                    selected_preferred_game_keys: list[str] = []
                    if apply_focus_game_stack_guardrails and not focus_summary_df.empty:
                        selected_preferred_game_keys.extend(
                            focus_summary_df.head(max(1, int(focus_game_count)))["game_key"].astype(str).tolist()
                        )
                    for key in list(game_agent_bias_meta.get("applied_game_keys") or []):
                        norm_key = str(key or "").strip().upper()
                        if norm_key and norm_key not in selected_preferred_game_keys:
                            selected_preferred_game_keys.append(norm_key)
                    if selected_preferred_game_keys:
                        st.caption(
                            "Preferred games for lineup construction: "
                            + ", ".join(f"`{str(k).strip().upper()}`" for k in selected_preferred_game_keys)
                        )

                    if run_mode_key == "all":
                        version_plan = [
                            _lineup_model_config(str(cfg["version_key"]))
                            for cfg in LINEUP_MODEL_REGISTRY
                        ]
                        total_requested = int(lineup_count * len(version_plan))
                        default_weights = {
                            str(k): float(v)
                            for k, v in DEFAULT_ALL_VERSION_WEIGHTS.items()
                            if str(k).strip()
                        }
                        default_count_map = _allocate_weighted_counts(
                            keys=[str(cfg.get("version_key") or "") for cfg in version_plan],
                            total_count=total_requested,
                            weights=default_weights,
                            min_per_key=1,
                        )
                        for cfg in version_plan:
                            vkey = str(cfg.get("version_key") or "")
                            cfg["lineup_count"] = int(max(1, default_count_map.get(vkey, lineup_count)))
                        preview = ", ".join(
                            f"{cfg['version_key']}={int(cfg['lineup_count'])}" for cfg in version_plan
                        )
                        st.caption(f"Default allocation applied: {preview}")
                    else:
                        version_plan = [_lineup_model_config(selected_model_key)]
                        for cfg in version_plan:
                            cfg["lineup_count"] = int(lineup_count)

                    phantom_promotion_meta: dict[str, Any] = {}
                    if run_mode_key == "all" and promote_phantom_constructions and version_plan:
                        phantom_promotion_meta = compute_phantom_version_performance_weights(
                            bucket_name=bucket_name,
                            selected_date=lineup_slate_date,
                            selected_slate_key=lineup_slate_key,
                            lookback_days=phantom_promotion_lookback_days,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                        historical_weights = {
                            str(k): float(v)
                            for k, v in (phantom_promotion_meta.get("weights") or {}).items()
                            if str(k).strip()
                        }
                        total_requested = int(lineup_count * len(version_plan))
                        version_keys = [str(cfg.get("version_key") or "") for cfg in version_plan]
                        count_map = _allocate_weighted_counts(
                            keys=version_keys,
                            total_count=total_requested,
                            weights=historical_weights,
                            min_per_key=1,
                        )
                        for cfg in version_plan:
                            vkey = str(cfg.get("version_key") or "")
                            cfg["lineup_count"] = int(max(1, count_map.get(vkey, lineup_count)))

                        preview = ", ".join(
                            f"{cfg['version_key']}={int(cfg['lineup_count'])}" for cfg in version_plan
                        )
                        st.caption(
                            "Phantom promotion allocation applied: "
                            f"{preview} "
                            f"(lookback_days={int(phantom_promotion_meta.get('lookback_days') or 0)}, "
                            f"used_dates={int(phantom_promotion_meta.get('used_dates') or 0)})."
                        )

                    total_units = max(1, int(sum(int(cfg.get("lineup_count") or 0) for cfg in version_plan)))
                    generated_versions: dict[str, Any] = {}

                    for version_idx, version_cfg in enumerate(version_plan):
                        version_lineup_count = int(max(0, version_cfg.get("lineup_count") or 0))
                        if version_lineup_count <= 0:
                            continue
                        version_offset = int(sum(int(c.get("lineup_count") or 0) for c in version_plan[:version_idx]))
                        runtime_controls = _resolve_lineup_runtime_controls(
                            contest_type=contest_type,
                            version_cfg=version_cfg,
                            apply_gpp_variance_presets=apply_gpp_variance_presets,
                            salary_left_target=salary_left_target,
                            low_own_bucket_exposure_pct=low_own_bucket_exposure_pct,
                            low_own_bucket_min_per_lineup=low_own_bucket_min_per_lineup,
                            low_own_bucket_max_projected_ownership=low_own_bucket_max_projected_ownership,
                            low_own_bucket_min_projection=low_own_bucket_min_projection,
                            low_own_bucket_min_tail_score=low_own_bucket_min_tail_score,
                            low_own_bucket_objective_bonus=low_own_bucket_objective_bonus,
                            preferred_game_bonus=preferred_game_bonus,
                            preferred_game_stack_lineup_pct=focus_game_stack_lineup_pct,
                            preferred_game_stack_min_players=focus_game_stack_min_players,
                            max_unsupported_false_chalk_per_lineup=max_unsupported_false_chalk_per_lineup,
                            ceiling_boost_lineup_pct=ceiling_boost_lineup_pct,
                            ceiling_boost_stack_bonus=ceiling_boost_stack_bonus,
                            ceiling_boost_salary_left_target=ceiling_boost_salary_left_target,
                        )
                        if bool(runtime_controls.get("variance_preset_applied")):
                            st.caption(
                                f"[{version_cfg['version_key']}] variance preset: "
                                f"low_own={float(runtime_controls['low_own_bucket_exposure_pct']):.0f}%, "
                                f"focus_stack={float(runtime_controls['preferred_game_stack_lineup_pct']):.0f}%/{int(runtime_controls['preferred_game_stack_min_players'])}, "
                                f"false_chalk<={int(runtime_controls['max_unsupported_false_chalk_per_lineup'])}, "
                                f"ceiling={float(runtime_controls['ceiling_boost_lineup_pct']):.0f}%, "
                                f"salary_left_target={int(runtime_controls['salary_left_target'])}"
                            )

                        def _lineup_progress(done: int, total: int, status: str) -> None:
                            done_local = max(0, min(version_lineup_count, int(done)))
                            units_done = version_offset + done_local
                            pct = int((units_done / total_units) * 100)
                            pct = max(0, min(100, pct))
                            progress_bar.progress(pct, text=f"[{version_cfg['version_label']}] {status}")
                            progress_text.caption(f"{units_done}/{total_units}")

                        lineups, warnings = generate_lineups(
                            pool_df=pool_sorted,
                            num_lineups=version_lineup_count,
                            contest_type=contest_type,
                            locked_ids=locked_ids,
                            excluded_ids=excluded_ids,
                            exposure_caps_pct=exposure_caps,
                            global_max_exposure_pct=global_max_exposure_pct,
                            max_salary_left=effective_max_salary_left,
                            lineup_strategy=str(version_cfg["lineup_strategy"]),
                            include_tail_signals=bool(version_cfg.get("include_tail_signals", False)),
                            cluster_target_count=int(version_cfg.get("cluster_target_count", 15)),
                            cluster_variants_per_cluster=int(version_cfg.get("cluster_variants_per_cluster", 10)),
                            projection_scale=projection_scale,
                            projection_salary_bucket_scales=projection_salary_bucket_scales,
                            projection_role_bucket_scales=projection_role_bucket_scales,
                            apply_ownership_guardrails=apply_ownership_guardrails,
                            ownership_guardrail_projected_threshold=ownership_guardrail_proj_threshold,
                            ownership_guardrail_surge_threshold=ownership_guardrail_surge_threshold,
                            ownership_guardrail_projection_rank_threshold=0.60,
                            ownership_guardrail_floor_base=ownership_guardrail_proj_threshold,
                            ownership_guardrail_floor_cap=ownership_guardrail_floor_cap,
                            apply_uncertainty_shrink=apply_uncertainty_shrink,
                            uncertainty_weight=uncertainty_weight,
                            high_risk_extra_shrink=high_risk_extra_shrink,
                            dnp_risk_threshold=dnp_risk_threshold,
                            low_own_bucket_exposure_pct=float(runtime_controls["low_own_bucket_exposure_pct"]),
                            low_own_bucket_min_per_lineup=int(runtime_controls["low_own_bucket_min_per_lineup"]),
                            low_own_bucket_max_projected_ownership=float(runtime_controls["low_own_bucket_max_projected_ownership"]),
                            low_own_bucket_min_projection=float(runtime_controls["low_own_bucket_min_projection"]),
                            low_own_bucket_min_tail_score=float(runtime_controls["low_own_bucket_min_tail_score"]),
                            low_own_bucket_objective_bonus=float(runtime_controls["low_own_bucket_objective_bonus"]),
                            preferred_game_keys=selected_preferred_game_keys,
                            preferred_game_bonus=float(runtime_controls["preferred_game_bonus"]),
                            preferred_game_stack_lineup_pct=(
                                float(runtime_controls["preferred_game_stack_lineup_pct"])
                                if apply_focus_game_stack_guardrails
                                else 0.0
                            ),
                            preferred_game_stack_min_players=(
                                int(runtime_controls["preferred_game_stack_min_players"])
                                if apply_focus_game_stack_guardrails
                                else 0
                            ),
                            auto_preferred_game_count=0,
                            max_unsupported_false_chalk_per_lineup=int(
                                runtime_controls["max_unsupported_false_chalk_per_lineup"]
                            ),
                            ceiling_boost_lineup_pct=float(runtime_controls["ceiling_boost_lineup_pct"]),
                            ceiling_boost_stack_bonus=float(runtime_controls["ceiling_boost_stack_bonus"]),
                            ceiling_boost_salary_left_target=int(runtime_controls["ceiling_boost_salary_left_target"]),
                            objective_score_adjustments=objective_score_adjustments,
                            salary_left_target=int(runtime_controls["salary_left_target"]),
                            random_seed=lineup_seed + version_idx,
                            model_profile=str(version_cfg.get("model_profile") or "legacy_baseline"),
                            progress_callback=_lineup_progress,
                        )
                        annotated_lineups = _annotate_lineups_with_version_metadata(
                            lineups,
                            version_key=str(version_cfg["version_key"]),
                            version_label=str(version_cfg["version_label"]),
                            lineup_strategy=str(version_cfg["lineup_strategy"]),
                            model_profile=str(version_cfg.get("model_profile") or ""),
                            include_tail_signals=bool(version_cfg.get("include_tail_signals", False)),
                        )
                        generated_versions[str(version_cfg["version_key"])] = {
                            "version_key": str(version_cfg["version_key"]),
                            "version_label": str(version_cfg["version_label"]),
                            "lineup_strategy": str(version_cfg["lineup_strategy"]),
                            "include_tail_signals": bool(version_cfg.get("include_tail_signals", False)),
                            "model_profile": str(version_cfg.get("model_profile") or ""),
                            "variance_preset_applied": bool(runtime_controls.get("variance_preset_applied")),
                            "runtime_controls": runtime_controls,
                            "lineup_count_requested": int(version_lineup_count),
                            "lineups": annotated_lineups,
                            "warnings": warnings,
                            "upload_csv": build_dk_upload_csv(annotated_lineups) if annotated_lineups else "",
                        }

                    total_generated = sum(len(v.get("lineups") or []) for v in generated_versions.values())
                    progress_bar.progress(100, text=f"Finished: {total_generated} total lineups generated.")

                    run_bundle = {
                        "run_id": _new_lineup_run_id(),
                        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "slate_date": lineup_slate_date.isoformat(),
                        "slate_label": lineup_slate_label,
                        "slate_key": lineup_slate_key,
                        "run_mode": run_mode_key,
                        "settings": {
                            "slate_label": lineup_slate_label,
                            "slate_key": lineup_slate_key,
                            "selected_model_key": selected_model_key,
                            "lineup_count": lineup_count,
                            "contest_type": contest_type,
                            "lineup_seed": lineup_seed,
                            "all_versions_default_weights": DEFAULT_ALL_VERSION_WEIGHTS,
                            "apply_gpp_variance_presets": apply_gpp_variance_presets,
                            "max_salary_left": effective_max_salary_left,
                            "requested_max_salary_left": max_salary_left,
                            "strict_salary_utilization": strict_salary_utilization,
                            "salary_left_target": salary_left_target,
                            "global_max_exposure_pct": global_max_exposure_pct,
                            "cluster_target_count": 15,
                            "cluster_variants_per_cluster": 10,
                            "projection_scale": projection_scale,
                            "projection_salary_bucket_scales": projection_salary_bucket_scales,
                            "projection_role_bucket_scales": projection_role_bucket_scales,
                            "apply_ownership_guardrails": apply_ownership_guardrails,
                            "ownership_guardrail_projected_threshold": ownership_guardrail_proj_threshold,
                            "ownership_guardrail_surge_threshold": ownership_guardrail_surge_threshold,
                            "ownership_guardrail_floor_cap": ownership_guardrail_floor_cap,
                            "apply_uncertainty_shrink": apply_uncertainty_shrink,
                            "uncertainty_weight": uncertainty_weight,
                            "high_risk_extra_shrink": high_risk_extra_shrink,
                            "dnp_risk_threshold": dnp_risk_threshold,
                            "low_own_bucket_exposure_pct": low_own_bucket_exposure_pct,
                            "low_own_bucket_min_per_lineup": low_own_bucket_min_per_lineup,
                            "low_own_bucket_max_projected_ownership": low_own_bucket_max_projected_ownership,
                            "low_own_bucket_min_projection": low_own_bucket_min_projection,
                            "low_own_bucket_min_tail_score": low_own_bucket_min_tail_score,
                            "low_own_bucket_objective_bonus": low_own_bucket_objective_bonus,
                            "preferred_game_bonus": preferred_game_bonus,
                            "apply_focus_game_stack_guardrails": apply_focus_game_stack_guardrails,
                            "focus_game_count": focus_game_count,
                            "focus_game_stack_lineup_pct": focus_game_stack_lineup_pct,
                            "focus_game_stack_min_players": focus_game_stack_min_players,
                            "selected_preferred_game_keys": selected_preferred_game_keys,
                            "max_unsupported_false_chalk_per_lineup": max_unsupported_false_chalk_per_lineup,
                            "ceiling_boost_lineup_pct": ceiling_boost_lineup_pct,
                            "ceiling_boost_stack_bonus": ceiling_boost_stack_bonus,
                            "ceiling_boost_salary_left_target": ceiling_boost_salary_left_target,
                            "auto_projection_calibration": auto_projection_calibration,
                            "calibration_lookback_days": calibration_lookback_days,
                            "calibration_meta": calibration_meta,
                            "auto_salary_bucket_calibration": auto_salary_bucket_calibration,
                            "bucket_calibration_lookback_days": bucket_calibration_lookback_days,
                            "bucket_calibration_min_samples": bucket_calibration_min_samples,
                            "salary_bucket_calibration_meta": salary_bucket_calibration_meta,
                            "auto_role_bucket_calibration": auto_role_bucket_calibration,
                            "role_calibration_lookback_days": role_calibration_lookback_days,
                            "role_calibration_min_samples": role_calibration_min_samples,
                            "role_bucket_calibration_meta": role_bucket_calibration_meta,
                            "apply_game_agent_stack_bias": apply_game_agent_stack_bias,
                            "game_agent_bias_strength_pct": game_agent_bias_strength_pct,
                            "game_agent_focus_games": game_agent_focus_games,
                            "game_agent_bias_meta": game_agent_bias_meta,
                            "recent_form_games": recent_form_games,
                            "recent_points_weight": recent_points_weight,
                            "recent_points_weight_pct": recent_points_weight_pct,
                            "promote_phantom_constructions": promote_phantom_constructions,
                            "phantom_promotion_lookback_days": phantom_promotion_lookback_days,
                            "phantom_promotion_meta": phantom_promotion_meta,
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
                        saved_meta = persist_lineup_run_bundle(
                            store,
                            lineup_slate_date,
                            run_bundle,
                            slate_key=lineup_slate_key,
                            slate_label=lineup_slate_label,
                        )
                        st.session_state["cbb_generated_run_manifest"] = saved_meta.get("manifest")
                        load_saved_lineup_run_manifests.clear()
                        load_saved_lineup_version_payload.clear()
                        st.success(
                            f"Saved run `{saved_meta.get('run_id')}` with {saved_meta.get('version_count')} version(s) "
                            f"to `{saved_meta.get('manifest_blob')}`."
                        )

                run_bundle = st.session_state.get("cbb_generated_run_bundle") or {}
                run_bundle_slate_date = str(run_bundle.get("slate_date") or "").strip()
                run_bundle_slate_key = _slate_key_from_label(
                    run_bundle.get("slate_key") or run_bundle.get("slate_label"),
                    default="main",
                )
                selected_lineup_slate_date = lineup_slate_date.isoformat()
                selected_lineup_slate_key = _slate_key_from_label(lineup_slate_key)
                generated_versions = run_bundle.get("versions") or {}
                if generated_versions and run_bundle_slate_date and (
                    run_bundle_slate_date != selected_lineup_slate_date
                    or run_bundle_slate_key != selected_lineup_slate_key
                ):
                    st.info(
                        "Lineup Generator only shows runs for the selected slate date+label. "
                        f"Current in-session run is `{run_bundle_slate_date}` (`{run_bundle_slate_key}`); "
                        "use `Tournament Review` for historical runs."
                    )
                    generated_versions = {}
                    st.session_state.pop("cbb_generated_lineups", None)
                    st.session_state.pop("cbb_generated_lineups_warnings", None)
                    st.session_state.pop("cbb_generated_upload_csv", None)
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
                    generated = enrich_lineups_minutes_from_pool(generated, pool_sorted)
                    generated = _annotate_lineups_with_version_metadata(
                        generated,
                        version_key=str(active_version.get("version_key") or active_version_key),
                        version_label=str(active_version.get("version_label") or active_version_key),
                        lineup_strategy=str(active_version.get("lineup_strategy") or ""),
                        model_profile=str(active_version.get("model_profile") or ""),
                        include_tail_signals=bool(active_version.get("include_tail_signals", False)),
                    )
                    active_version["lineups"] = generated
                    active_version["upload_csv"] = str(active_version.get("upload_csv") or build_dk_upload_csv(generated))
                    generated_versions[active_version_key] = active_version
                    run_bundle["versions"] = generated_versions
                    st.session_state["cbb_generated_run_bundle"] = run_bundle
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
                        f"Slate: `{run_bundle.get('slate_label', run_bundle_slate_key)}` | Version: `{active_version_key}`"
                    )

                    if generated:
                        summary_df = lineups_summary_frame(generated)
                        st.dataframe(summary_df, hide_index=True, use_container_width=True)

                        slots_df = lineups_slots_frame(generated)
                        if not slots_df.empty:
                            st.subheader("Generated Lineups (Slot View)")
                            st.dataframe(slots_df, hide_index=True, use_container_width=True)

                        export_date = run_bundle_slate_date or selected_lineup_slate_date
                        export_slate = run_bundle_slate_key or selected_lineup_slate_key
                        all_generated_export_df = _build_lineup_versions_export_frame(
                            generated_versions,
                            run_id=str(run_bundle.get("run_id") or ""),
                        )
                        dl1, dl2 = st.columns(2)
                        with dl1:
                            if not all_generated_export_df.empty:
                                st.download_button(
                                    "Download All Generated Lineups CSV",
                                    data=all_generated_export_df.to_csv(index=False),
                                    file_name=(
                                        f"all_generated_lineups_{export_date}_{export_slate}_"
                                        f"{run_bundle.get('run_id', 'run')}.csv"
                                    ),
                                    mime="text/csv",
                                    key="download_all_generated_lineups_csv",
                                )
                        with dl2:
                            st.download_button(
                                "Download DK Upload CSV",
                                data=upload_csv,
                                file_name=f"dk_lineups_{export_date}_{export_slate}_{active_version_key}.csv",
                                mime="text/csv",
                                key="download_dk_upload_csv",
                            )
                elif generate_lineups_clicked:
                    st.error("No lineups were generated. Adjust locks/exclusions/exposure settings and retry.")

                st.markdown("---")
                st.subheader("Lineup Consistency Agent")
                st.caption(
                    "Pre-game readiness checks for settings and prior-agent alignment. Phantom diagnostics are optional."
                )
                phantom_df_state = st.session_state.get("cbb_phantom_review_df")
                phantom_summary_state = st.session_state.get("cbb_phantom_summary_df")
                phantom_meta = st.session_state.get("cbb_phantom_review_meta") or {}
                consistency_phantom_df = (
                    phantom_df_state.copy() if isinstance(phantom_df_state, pd.DataFrame) else pd.DataFrame()
                )
                consistency_phantom_summary_df = (
                    phantom_summary_state.copy()
                    if isinstance(phantom_summary_state, pd.DataFrame)
                    else pd.DataFrame()
                )

                if not generated_versions:
                    st.info("Generate or load a lineup run first.")
                else:
                    if consistency_phantom_df.empty and consistency_phantom_summary_df.empty:
                        st.info(
                            "Running in pre-game mode: phantom diagnostics are unavailable, "
                            "so this audit focuses on lineup settings/policy adherence."
                        )
                    consistency_version_key = str(st.session_state.get("cbb_active_version_key") or "").strip()
                    if consistency_version_key not in generated_versions:
                        consistency_version_key = next(iter(generated_versions.keys()), "")
                    consistency_version = generated_versions.get(consistency_version_key) or {}

                    consistency_packet = build_lineup_consistency_packet(
                        run_bundle=run_bundle,
                        active_version_key=consistency_version_key,
                        active_version=consistency_version,
                        phantom_df=consistency_phantom_df,
                        phantom_summary_df=consistency_phantom_summary_df,
                        phantom_meta=phantom_meta,
                    )
                    status_summary = consistency_packet.get("status_summary") or {}
                    cs1, cs2, cs3, cs4 = st.columns(4)
                    cs1.metric("Overall", str(status_summary.get("overall_status", "n/a")).upper())
                    cs2.metric("Pass", _safe_int_value(status_summary.get("pass_checks"), default=0))
                    cs3.metric("Warn", _safe_int_value(status_summary.get("warn_checks"), default=0))
                    cs4.metric("Fail", _safe_int_value(status_summary.get("fail_checks"), default=0))

                    checks_df = pd.DataFrame(consistency_packet.get("checks") or [])
                    if checks_df.empty:
                        st.info("No consistency checks were produced.")
                    else:
                        st.dataframe(checks_df, hide_index=True, use_container_width=True)

                    gap_df = pd.DataFrame(consistency_packet.get("gap_candidates") or [])
                    upside_df = pd.DataFrame(consistency_packet.get("upside_candidates") or [])
                    stack_upside_df = pd.DataFrame(consistency_packet.get("stack_upside_candidates") or [])
                    preferred_game_df = pd.DataFrame(consistency_packet.get("preferred_game_phantom_exposure") or [])
                    version_alloc_df = pd.DataFrame(consistency_packet.get("version_allocation_check") or [])

                    gg1, gg2 = st.columns(2)
                    with gg1:
                        st.caption("Gap Candidates")
                        if gap_df.empty:
                            st.info("No warning/fail gap candidates detected.")
                        else:
                            st.dataframe(gap_df.head(20), hide_index=True, use_container_width=True)
                    with gg2:
                        st.caption("Upside Candidates (Version-Level)")
                        if upside_df.empty:
                            st.info("No version-level upside candidates available.")
                        else:
                            st.dataframe(upside_df.head(20), hide_index=True, use_container_width=True)

                    if not stack_upside_df.empty:
                        st.caption("Upside Stack Signatures")
                        st.dataframe(stack_upside_df.head(20), hide_index=True, use_container_width=True)
                    if not preferred_game_df.empty:
                        st.caption("Preferred-Game Phantom Exposure")
                        st.dataframe(preferred_game_df.head(20), hide_index=True, use_container_width=True)
                    if not version_alloc_df.empty:
                        st.caption("Version Allocation Consistency")
                        st.dataframe(version_alloc_df.head(20), hide_index=True, use_container_width=True)

                    consistency_packet_json = json.dumps(_json_safe(consistency_packet), indent=2, ensure_ascii=True)
                    consistency_prompt_system = (
                        "You are a DFS lineup process auditor. Use only evidence in the JSON packet. "
                        "Prioritize consistency checks, actionable gaps, and upside opportunities."
                    )
                    consistency_prompt_user = (
                        "Review this lineup consistency packet and produce an actionable audit.\n\n"
                        "Required sections:\n"
                        "1) Pre-Game Consistency Verdict (pass/warn/fail with 3-6 bullets)\n"
                        "2) Settings/Policy Violations (with exact metric evidence)\n"
                        "3) Gaps and Missed Upside (players/stacks/construction)\n"
                        "4) Recommended Next-Run Parameter Changes (max 8)\n"
                        "5) Optional Post-Phantom Follow-Ups (if phantom metrics exist)\n"
                        "6) Guardrails To Avoid Overfitting\n\n"
                        "Constraints:\n"
                        "- Use only evidence in the JSON packet.\n"
                        "- Cite exact metric names and values.\n"
                        "- If data is insufficient, state exactly what is missing.\n\n"
                        "JSON packet:\n"
                        f"{consistency_packet_json}\n"
                    )
                    dc1, dc2 = st.columns(2)
                    dc1.download_button(
                        "Download Consistency Packet JSON",
                        data=consistency_packet_json,
                        file_name=(
                            f"lineup_consistency_packet_{lineup_slate_date.isoformat()}_"
                            f"{consistency_version_key or 'version'}.json"
                        ),
                        mime="application/json",
                        key="download_lineup_consistency_packet_json",
                    )
                    dc2.download_button(
                        "Download Consistency Prompt",
                        data=consistency_prompt_user,
                        file_name=(
                            f"lineup_consistency_prompt_{lineup_slate_date.isoformat()}_"
                            f"{consistency_version_key or 'version'}.txt"
                        ),
                        mime="text/plain",
                        key="download_lineup_consistency_prompt_txt",
                    )

                    openai_key = (os.getenv("OPENAI_API_KEY", "").strip() or (_secret("openai_api_key") or "").strip())
                    lc1, lc2 = st.columns(2)
                    consistency_model = lc1.text_input(
                        "Consistency Model",
                        value=str(st.session_state.get("ai_review_model", "gpt-5-mini")),
                        key="lineup_consistency_model",
                    ).strip()
                    consistency_tokens = int(
                        lc2.number_input(
                            "Consistency Max Tokens",
                            min_value=400,
                            max_value=8000,
                            value=1400,
                            step=100,
                            key="lineup_consistency_max_tokens",
                        )
                    )
                    run_consistency_agent = st.button(
                        "Run Lineup Consistency Agent",
                        key="run_lineup_consistency_agent",
                    )
                    if run_consistency_agent:
                        if not openai_key:
                            st.error(
                                "Set `OPENAI_API_KEY` (or `openai_api_key` in Streamlit secrets) "
                                "to run the consistency agent."
                            )
                        else:
                            try:
                                with st.spinner("Running lineup consistency agent..."):
                                    consistency_output = request_openai_review(
                                        api_key=openai_key,
                                        user_prompt=consistency_prompt_user,
                                        system_prompt=consistency_prompt_system,
                                        model=consistency_model or "gpt-5-mini",
                                        max_output_tokens=consistency_tokens,
                                    )
                                    st.session_state["cbb_lineup_consistency_agent_output"] = consistency_output
                            except Exception as exc:
                                st.exception(exc)

                    consistency_output = str(
                        st.session_state.get("cbb_lineup_consistency_agent_output") or ""
                    ).strip()
                    if consistency_output:
                        st.text_area(
                            "Lineup Consistency Agent Output",
                            value=consistency_output,
                            height=320,
                            key="lineup_consistency_agent_output_text",
                        )
                        st.download_button(
                            "Download Consistency Output",
                            data=consistency_output,
                            file_name=(
                                f"lineup_consistency_output_{lineup_slate_date.isoformat()}_"
                                f"{consistency_version_key or 'version'}.txt"
                            ),
                            mime="text/plain",
                            key="download_lineup_consistency_output_txt",
                        )

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
                        bundle_slate = str(current_run_bundle.get("slate_date") or "").strip()
                        bundle_slate_key = _slate_key_from_label(
                            current_run_bundle.get("slate_key") or current_run_bundle.get("slate_label"),
                            default="main",
                        )
                        current_picker_slate_key = _slate_key_from_label(lineup_slate_key)
                        if bundle_slate and (
                            bundle_slate != lineup_slate_date.isoformat()
                            or bundle_slate_key != current_picker_slate_key
                        ):
                            st.error(
                                "Current in-session run is for a different slate date/label "
                                f"(`{bundle_slate}` / `{bundle_slate_key}`). "
                                "Switch `Lineup Slate Date/Lineup Slate` to match or generate a new run."
                            )
                        else:
                            save_date = lineup_slate_date
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
                            save_slate_key = bundle_slate_key or current_picker_slate_key
                            save_slate_label = _normalize_slate_label(
                                current_run_bundle.get("slate_label") or lineup_slate_label
                            )
                            saved_meta = persist_lineup_run_bundle(
                                store,
                                save_date,
                                current_run_bundle,
                                slate_key=save_slate_key,
                                slate_label=save_slate_label,
                            )
                            st.session_state["cbb_generated_run_manifest"] = saved_meta.get("manifest")
                            load_saved_lineup_run_dates.clear()
                            load_saved_lineup_run_manifests.clear()
                            load_saved_lineup_version_payload.clear()
                            st.success(
                                f"Saved current run `{saved_meta.get('run_id')}` "
                                f"to `{saved_meta.get('manifest_blob')}`."
                            )

                st.caption(
                    f"Showing saved runs for `{lineup_slate_date.isoformat()}` + "
                    f"`{lineup_slate_label}` only. "
                    "Use `Tournament Review` to inspect historical dates."
                )
                sr1, sr2 = st.columns([1, 2])
                load_latest_clicked = sr1.button("Load Latest Run", key="load_latest_saved_run")
                run_id_filter = sr2.text_input("Filter Run ID (optional)", key="saved_run_id_filter")

                merged_manifests: list[dict[str, Any]] = []
                manifests_for_date = load_saved_lineup_run_manifests(
                    bucket_name=bucket_name,
                    selected_date=lineup_slate_date,
                    selected_slate_key=lineup_slate_key,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )
                for manifest in manifests_for_date:
                    entry = dict(manifest)
                    entry["_saved_date"] = lineup_slate_date
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
                        manifest_slate_label = _normalize_slate_label(
                            manifest.get("slate_label") or manifest.get("slate_key")
                        )
                        label = f"{generated_at} | {run_id} | {run_mode} | {manifest_slate_label}"
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
                    selected_manifest_date = lineup_slate_date
                    selected_manifest_slate_key = _slate_key_from_label(
                        selected_manifest.get("slate_key") or selected_manifest.get("slate_label"),
                        default=_slate_key_from_label(lineup_slate_key),
                    )

                    saved_versions_bundle: dict[str, Any] = {}

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
                            saved_payload_map: dict[str, dict[str, Any]] = {}
                            for version_key in saved_version_keys:
                                payload = load_saved_lineup_version_payload(
                                    bucket_name=bucket_name,
                                    selected_date=selected_manifest_date,
                                    run_id=str(selected_manifest.get("run_id") or ""),
                                    version_key=version_key,
                                    selected_slate_key=selected_manifest_slate_key,
                                    gcp_project=gcp_project or None,
                                    service_account_json=cred_json,
                                    service_account_json_b64=cred_json_b64,
                                )
                                if not isinstance(payload, dict):
                                    continue
                                version_meta = version_meta_map.get(version_key, {})
                                annotated_lineups = _annotate_lineups_with_version_metadata(
                                    payload.get("lineups") or [],
                                    version_key=version_key,
                                    version_label=str(
                                        payload.get("version_label")
                                        or version_meta.get("version_label")
                                        or version_key
                                    ),
                                    lineup_strategy=str(
                                        payload.get("lineup_strategy")
                                        or version_meta.get("lineup_strategy")
                                        or ""
                                    ),
                                    model_profile=str(
                                        payload.get("model_profile")
                                        or version_meta.get("model_profile")
                                        or ""
                                    ),
                                    include_tail_signals=bool(
                                        payload.get("include_tail_signals")
                                        if "include_tail_signals" in payload
                                        else version_meta.get("include_tail_signals", False)
                                    ),
                                )
                                saved_payload_map[version_key] = {
                                    **payload,
                                    "lineups": annotated_lineups,
                                }
                                saved_versions_bundle[version_key] = {
                                    "version_key": version_key,
                                    "version_label": str(
                                        payload.get("version_label")
                                        or version_meta.get("version_label")
                                        or version_key
                                    ),
                                    "lineup_strategy": str(
                                        payload.get("lineup_strategy")
                                        or version_meta.get("lineup_strategy")
                                        or ""
                                    ),
                                    "model_profile": str(
                                        payload.get("model_profile")
                                        or version_meta.get("model_profile")
                                        or ""
                                    ),
                                    "include_tail_signals": bool(
                                        payload.get("include_tail_signals")
                                        if "include_tail_signals" in payload
                                        else version_meta.get("include_tail_signals", False)
                                    ),
                                    "lineups": annotated_lineups,
                                }

                            saved_payload = saved_payload_map.get(selected_saved_version)
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
                                    all_saved_export_df = _build_lineup_versions_export_frame(
                                        saved_versions_bundle,
                                        run_id=str(selected_manifest.get("run_id") or ""),
                                    )
                                    sdl1, sdl2 = st.columns(2)
                                    with sdl1:
                                        if not all_saved_export_df.empty:
                                            st.download_button(
                                                "Download Saved Generated Lineups CSV",
                                                data=all_saved_export_df.to_csv(index=False),
                                                file_name=(
                                                    f"saved_generated_lineups_{selected_manifest_date.isoformat()}_"
                                                    f"{selected_manifest_slate_key}_"
                                                    f"{selected_manifest.get('run_id', 'run')}.csv"
                                                ),
                                                mime="text/csv",
                                                key=f"download_saved_generated_lineups_csv_{selected_saved_version}",
                                            )
                                    with sdl2:
                                        st.download_button(
                                            "Download Saved DK Upload CSV",
                                            data=saved_upload_csv,
                                            file_name=(
                                                f"dk_lineups_{selected_manifest_date.isoformat()}_"
                                                f"{selected_manifest_slate_key}_"
                                                f"{selected_manifest.get('run_id', 'run')}_{selected_saved_version}.csv"
                                            ),
                                            mime="text/csv",
                                            key=f"download_saved_dk_upload_csv_{selected_saved_version}",
                                        )
        except Exception as exc:
            st.exception(exc)

with tab_projection_review:
    st.subheader("Projection Review")
    st.caption(
        "Projection snapshots and ownership uploads are date-scoped shared files. "
        "Slate context is used for DK-slate filtering and lineup-run lookups."
    )
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
                        blob_name = _write_ownership_csv(
                            store,
                            review_date,
                            normalized_own.to_csv(index=False),
                            slate_key=shared_slate_key,
                        )
                        load_ownership_frame_for_date.clear()
                        st.success(f"Saved ownership file to `{blob_name}`")

            proj_df = load_projection_snapshot_frame(
                bucket_name=bucket_name,
                selected_date=review_date,
                selected_slate_key=shared_slate_key,
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
                    selected_slate_key=shared_slate_key,
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
        "and compare against our lineups and projection assumptions. "
        "Standings files are date+contest scoped (shared across slates)."
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
        st.session_state.pop("cbb_phantom_source_lineups", None)
        st.session_state.pop("cbb_phantom_source_meta", None)
        st.session_state.pop("cbb_ai_review_packet", None)
        st.session_state.pop("cbb_ai_review_prompt_user", None)
        st.session_state.pop("cbb_ai_review_prompt_system", None)
        st.session_state.pop("cbb_tournament_postmortem_output", None)
        st.session_state.pop("cbb_tr_expanded_df", None)

    if not bucket_name:
        st.info("Set a GCS bucket in sidebar to run tournament review.")
    else:
        try:
            upload_csv_text = ""
            upload_frame = pd.DataFrame()
            upload_parse_mode = "none"
            upload_parse_error = ""
            upload_decode_warning = ""
            upload_profile: dict[str, Any] | None = None
            upload_is_usable = False
            if tr_upload is not None:
                upload_payload = _read_uploaded_csv_frame(tr_upload)
                upload_csv_text = str(upload_payload.get("csv_text") or "")
                upload_frame = (
                    upload_payload.get("frame").copy()
                    if isinstance(upload_payload.get("frame"), pd.DataFrame)
                    else pd.DataFrame()
                )
                upload_parse_mode = str(upload_payload.get("parse_mode") or "failed")
                upload_parse_error = str(upload_payload.get("parse_error") or "").strip()
                upload_decode_warning = str(upload_payload.get("decode_warning") or "").strip()
                if upload_parse_mode == "failed":
                    st.error(
                        "Could not parse uploaded CSV. Upload DraftKings contest standings "
                        "(`contest-standings-<contest_id>.csv`)."
                    )
                    if upload_parse_error:
                        st.caption(f"Parser error: {upload_parse_error}")
                else:
                    upload_profile = detect_contest_standings_upload(upload_frame)
                    row_count = int(upload_profile.get("row_count", len(upload_frame)))
                    col_count = int(upload_profile.get("column_count", len(upload_frame.columns)))
                    st.caption(f"Uploaded `{tr_upload.name}` | Rows: {row_count:,} | Columns: {col_count:,}")
                    if upload_decode_warning:
                        st.warning(upload_decode_warning)
                    if upload_parse_mode == "relaxed":
                        st.warning("Loaded with relaxed CSV parsing; malformed rows were skipped.")
                    if str(upload_profile.get("kind") or "") != "contest_standings":
                        st.error(str(upload_profile.get("message") or "Unrecognized tournament standings CSV format."))
                        preview_cols = [str(c) for c in (upload_profile.get("columns_preview") or [])]
                        if preview_cols:
                            st.caption(f"Detected columns: {', '.join(preview_cols)}")
                    else:
                        upload_is_usable = bool(upload_profile.get("is_usable"))
                        if int(upload_profile.get("lineup_nonempty_rows") or 0) <= 0:
                            st.warning(
                                str(
                                    upload_profile.get("message")
                                    or "Contest standings loaded, but lineup rows are currently empty."
                                )
                            )

            if tr_save_clicked:
                if tr_upload is None:
                    st.error("Choose a contest standings CSV before saving.")
                elif upload_parse_mode == "failed":
                    st.error("Fix CSV parsing errors before saving to GCS.")
                elif not upload_is_usable:
                    st.error("Uploaded file is not a contest standings export. Not saved.")
                else:
                    client = build_storage_client(
                        service_account_json=cred_json,
                        service_account_json_b64=cred_json_b64,
                        project=gcp_project or None,
                    )
                    store = CbbGcsStore(bucket_name=bucket_name, client=client)
                    blob_name = _write_contest_standings_csv(
                        store,
                        tr_date,
                        tr_contest_id,
                        upload_csv_text,
                        slate_key=shared_slate_key,
                    )
                    load_contest_standings_frame.clear()
                    st.success(f"Saved contest standings to `{blob_name}`")

            if tr_upload is not None:
                standings_df = upload_frame.copy() if upload_is_usable else pd.DataFrame()
            else:
                standings_df = load_contest_standings_frame(
                    bucket_name=bucket_name,
                    selected_date=tr_date,
                    contest_id=tr_contest_id,
                    selected_slate_key=shared_slate_key,
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
            expanded_df = pd.DataFrame()
            exposure_df = pd.DataFrame()
            proj_compare_df = pd.DataFrame()
            adjust_df = pd.DataFrame()
            st.session_state["cbb_tr_entries_df"] = entries_df.copy()
            st.session_state["cbb_tr_expanded_df"] = expanded_df.copy()
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
                    slate_key=shared_slate_key,
                    gcp_project=gcp_project or None,
                    service_account_json=cred_json,
                    service_account_json_b64=cred_json_b64,
                )

                projections_df = load_projection_snapshot_frame(
                    bucket_name=bucket_name,
                    selected_date=tr_date,
                    selected_slate_key=shared_slate_key,
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
                    own_proj = (
                        pd.to_numeric(projections_df.get("projected_ownership"), errors="coerce")
                        if "projected_ownership" in projections_df.columns
                        else pd.Series(dtype=float)
                    )
                    missing_minutes = not bool(mins_last7.notna().any() or mins_avg.notna().any())
                    missing_ownership = not bool(own_proj.notna().any())
                    need_pool_fallback = bool(missing_minutes or missing_ownership)

                if need_pool_fallback and not slate_df.empty:
                    fallback_pool, _, _, _, _ = build_optimizer_pool_for_date(
                        bucket_name=bucket_name,
                        slate_date=tr_date,
                        slate_key=shared_slate_key,
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
                            "projected_ownership",
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

                    our_generated_lineups = st.session_state.get("cbb_generated_lineups", [])
                    our_df = pd.DataFrame()
                    if our_generated_lineups:
                        our_df = summarize_generated_lineups(our_generated_lineups)
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

                        st.markdown("**Ownership Projection Diagnostics**")
                        actual_source_options: list[tuple[str, str]] = [("Field Ownership (From Standings)", "field_ownership_pct")]
                        if "actual_ownership_from_file" in exposure_df.columns and bool(
                            pd.to_numeric(exposure_df["actual_ownership_from_file"], errors="coerce").notna().any()
                        ):
                            actual_source_options.append(("Actual Ownership Upload", "actual_ownership_from_file"))
                        actual_source_label = st.selectbox(
                            "Actual Ownership Source",
                            options=[label for label, _ in actual_source_options],
                            index=0,
                            key="tournament_ownership_diag_actual_source",
                            help="Choose the actual ownership series to compare against projected ownership.",
                        )
                        actual_source_col = next(
                            (col for label, col in actual_source_options if label == actual_source_label),
                            "field_ownership_pct",
                        )
                        top_misses_n = int(
                            st.slider(
                                "Top Ownership Misses to Show",
                                min_value=10,
                                max_value=100,
                                value=25,
                                step=5,
                                key="tournament_ownership_diag_top_misses_n",
                            )
                        )
                        own_diag = build_ownership_projection_diagnostics(
                            exposure_df,
                            projected_col="projected_ownership",
                            actual_col=actual_source_col,
                            top_misses_n=top_misses_n,
                        )
                        own_summary = own_diag.get("summary") if isinstance(own_diag.get("summary"), dict) else {}
                        own_samples = int(own_summary.get("samples") or 0)
                        if own_samples <= 0:
                            st.info("Ownership diagnostics unavailable: need players with both projected and actual ownership.")
                        else:
                            od1, od2, od3, od4, od5, od6 = st.columns(6)
                            od1.metric("Matched Players", own_samples)
                            od2.metric("Ownership MAE", f"{_safe_float_value(own_summary.get('mae')):.2f}")
                            od3.metric("Ownership RMSE", f"{_safe_float_value(own_summary.get('rmse')):.2f}")
                            od4.metric("Bias (Actual - Proj)", f"{_safe_float_value(own_summary.get('bias')):+.2f}")
                            od5.metric("Within +/-3 pts", f"{_safe_float_value(own_summary.get('within_3_pct')):.1f}%")
                            od6.metric("Corr (Pearson)", f"{_safe_float_value(own_summary.get('corr')):.3f}")
                            oe1, oe2 = st.columns(2)
                            oe1.metric("Overprojected Share", f"{_safe_float_value(own_summary.get('overprojected_pct')):.1f}%")
                            oe2.metric("Underprojected Share", f"{_safe_float_value(own_summary.get('underprojected_pct')):.1f}%")
                            st.caption(
                                "Positive bias means actual ownership was higher than projected on average (we underprojected)."
                            )

                            own_matched_df = (
                                own_diag.get("matched_df").copy()
                                if isinstance(own_diag.get("matched_df"), pd.DataFrame)
                                else pd.DataFrame()
                            )
                            if not own_matched_df.empty:
                                scatter_cols = [c for c in ["Name", "TeamAbbrev", "projected_ownership", "actual_ownership"] if c in own_matched_df.columns]
                                scatter_df = own_matched_df[scatter_cols].copy()
                                scatter_df["projected_ownership"] = pd.to_numeric(
                                    scatter_df.get("projected_ownership"), errors="coerce"
                                )
                                scatter_df["actual_ownership"] = pd.to_numeric(
                                    scatter_df.get("actual_ownership"), errors="coerce"
                                )
                                scatter_df = scatter_df.dropna(subset=["projected_ownership", "actual_ownership"]).copy()
                                if not scatter_df.empty:
                                    if px is None:
                                        st.info(
                                            "Plotly is not installed in this environment; showing ownership scatter values as a table."
                                        )
                                        st.dataframe(scatter_df, use_container_width=True)
                                    else:
                                        own_fig = px.scatter(
                                            scatter_df,
                                            x="projected_ownership",
                                            y="actual_ownership",
                                            hover_name="Name" if "Name" in scatter_df.columns else None,
                                            color="TeamAbbrev" if "TeamAbbrev" in scatter_df.columns else None,
                                            labels={
                                                "projected_ownership": "Projected Ownership %",
                                                "actual_ownership": "Actual Ownership %",
                                            },
                                            title="Ownership Calibration: Projected vs Actual",
                                        )
                                        max_axis = float(
                                            max(
                                                pd.to_numeric(scatter_df["projected_ownership"], errors="coerce").max() or 0.0,
                                                pd.to_numeric(scatter_df["actual_ownership"], errors="coerce").max() or 0.0,
                                            )
                                        )
                                        max_axis = max(10.0, max_axis + 1.0)
                                        own_fig.add_shape(
                                            type="line",
                                            x0=0.0,
                                            y0=0.0,
                                            x1=max_axis,
                                            y1=max_axis,
                                            line=dict(color="#8c8c8c", dash="dash"),
                                        )
                                        own_fig.update_layout(height=420, margin=dict(l=8, r=8, t=44, b=8))
                                        st.plotly_chart(
                                            own_fig,
                                            use_container_width=True,
                                            key=f"tournament_ownership_calibration_scatter_{actual_source_col}",
                                        )

                            own_buckets_df = (
                                own_diag.get("buckets_df").copy()
                                if isinstance(own_diag.get("buckets_df"), pd.DataFrame)
                                else pd.DataFrame()
                            )
                            if not own_buckets_df.empty:
                                st.caption("Ownership calibration by projected-ownership bucket")
                                st.dataframe(own_buckets_df, hide_index=True, use_container_width=True)
                                st.download_button(
                                    "Download Ownership Calibration Buckets CSV",
                                    data=own_buckets_df.to_csv(index=False),
                                    file_name=f"tournament_ownership_calibration_buckets_{tr_date.isoformat()}_{tr_contest_id}.csv",
                                    mime="text/csv",
                                    key="download_tournament_ownership_calibration_buckets_csv",
                                )

                            own_misses_df = (
                                own_diag.get("top_misses_df").copy()
                                if isinstance(own_diag.get("top_misses_df"), pd.DataFrame)
                                else pd.DataFrame()
                            )
                            if not own_misses_df.empty:
                                st.caption("Largest ownership misses (absolute error)")
                                st.dataframe(own_misses_df, hide_index=True, use_container_width=True)
                                st.download_button(
                                    "Download Ownership Misses CSV",
                                    data=own_misses_df.to_csv(index=False),
                                    file_name=f"tournament_ownership_misses_{tr_date.isoformat()}_{tr_contest_id}.csv",
                                    mime="text/csv",
                                    key="download_tournament_ownership_misses_csv",
                                )

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

                        st.subheader("Lineup Generator Diagnostics")
                        st.caption(
                            "Use these visuals to isolate projection bias, confidence calibration gaps, and leverage misses."
                        )
                        d1, d2 = st.columns([2, 1])
                        diag_projection_source = d1.selectbox(
                            "Projection Source for Bias Heatmap",
                            options=["Blended Projection", "Our Projection", "Vegas Projection"],
                            index=0,
                            key="tournament_diag_projection_source",
                        )
                        diag_min_samples = int(
                            d2.number_input(
                                "Min Samples per Cell",
                                min_value=1,
                                max_value=200,
                                value=5,
                                step=1,
                                key="tournament_diag_min_samples",
                                help="Cells with fewer samples are masked in the heatmap.",
                            )
                        )
                        error_col = {
                            "Blended Projection": "blend_error",
                            "Our Projection": "our_error",
                            "Vegas Projection": "vegas_error",
                        }.get(diag_projection_source, "blend_error")
                        heatmap_packet = build_projection_bias_heatmap(
                            proj_compare_view,
                            error_col=error_col,
                            min_samples_per_cell=diag_min_samples,
                        )
                        heatmap_df = (
                            heatmap_packet.get("avg_error_matrix_df").copy()
                            if isinstance(heatmap_packet.get("avg_error_matrix_df"), pd.DataFrame)
                            else pd.DataFrame()
                        )
                        samples_df = (
                            heatmap_packet.get("samples_matrix_df").copy()
                            if isinstance(heatmap_packet.get("samples_matrix_df"), pd.DataFrame)
                            else pd.DataFrame()
                        )
                        heatmap_cells_df = (
                            heatmap_packet.get("cells_df").copy()
                            if isinstance(heatmap_packet.get("cells_df"), pd.DataFrame)
                            else pd.DataFrame()
                        )
                        if heatmap_cells_df.empty:
                            st.info("Not enough projection-vs-actual rows to build diagnostics heatmap.")
                        else:
                            st.markdown("**1) Projection Bias Heatmap (Salary x Position)**")
                            st.caption("Average Error (Actual FPTS - Projected FPTS)")
                            chart_df = heatmap_df.apply(pd.to_numeric, errors="coerce")
                            text_values = chart_df.applymap(lambda v: "" if pd.isna(v) else f"{float(v):.2f}")
                            chart_vals = pd.to_numeric(chart_df.stack(), errors="coerce").dropna()
                            max_abs = float(chart_vals.abs().max()) if not chart_vals.empty else 0.0
                            color_bound = max(1.0, max_abs)
                            if px is None:
                                st.info(
                                    "Plotly is not installed in this environment; showing heatmap values as a table."
                                )
                                st.dataframe(chart_df, use_container_width=True)
                            else:
                                heatmap_fig = px.imshow(
                                    chart_df,
                                    labels={"x": "Salary Bucket", "y": "Primary Position", "color": "Avg Error"},
                                    color_continuous_scale="RdBu",
                                    zmin=-color_bound,
                                    zmax=color_bound,
                                    aspect="auto",
                                )
                                heatmap_fig.update_traces(text=text_values.values, texttemplate="%{text}")
                                heatmap_fig.update_layout(
                                    height=440,
                                    margin=dict(l=8, r=8, t=24, b=8),
                                    coloraxis_colorbar_title_text="Avg Error",
                                )
                                heatmap_fig.update_xaxes(side="bottom")
                                st.plotly_chart(
                                    heatmap_fig,
                                    use_container_width=True,
                                    key=f"tournament_projection_bias_heatmap_{error_col}",
                                )

                            if not samples_df.empty:
                                st.caption("Sample counts by cell")
                                st.dataframe(samples_df, use_container_width=True)
                            diag_cols = [
                                "primary_position",
                                "salary_bucket",
                                "samples",
                                "avg_error",
                                "mae",
                                "median_error",
                                "std_error",
                            ]
                            diag_show_cols = [c for c in diag_cols if c in heatmap_cells_df.columns]
                            if diag_show_cols:
                                diag_table = heatmap_cells_df[diag_show_cols].sort_values(
                                    ["salary_bucket", "primary_position"], ascending=[True, True], kind="stable"
                                )
                                st.download_button(
                                    "Download Projection Bias Cells CSV",
                                    data=diag_table.to_csv(index=False),
                                    file_name=f"tournament_projection_bias_cells_{tr_date.isoformat()}_{tr_contest_id}.csv",
                                    mime="text/csv",
                                    key="download_tournament_projection_bias_cells_csv",
                                )
                            impact_df = build_segment_impact_table(
                                proj_compare_view,
                                error_col=error_col,
                                min_samples_per_cell=diag_min_samples,
                            )
                            if not impact_df.empty:
                                st.caption("Segment impact table (negative delta = improvement)")
                                impact_show = impact_df.rename(
                                    columns={
                                        "segment": "Segment",
                                        "samples": "Samples",
                                        "mae_pre": "MAE Pre",
                                        "mae_post": "MAE Post",
                                        "mae_delta": "MAE Delta",
                                        "bias_pre": "Bias Pre",
                                        "bias_post": "Bias Post",
                                        "abs_bias_delta": "Abs Bias Delta",
                                    }
                                )
                                st.dataframe(impact_show, hide_index=True, use_container_width=True)
                                st.download_button(
                                    "Download Segment Impact CSV",
                                    data=impact_show.to_csv(index=False),
                                    file_name=f"tournament_segment_impact_{tr_date.isoformat()}_{tr_contest_id}.csv",
                                    mime="text/csv",
                                    key="download_tournament_segment_impact_csv",
                                )

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

                    st.subheader("Top-10 Winner Gap Analysis")
                    phantom_source_lineups = st.session_state.get("cbb_phantom_source_lineups", [])
                    phantom_source_meta = st.session_state.get("cbb_phantom_source_meta") or {}
                    if str(phantom_source_meta.get("date") or "") != tr_date.isoformat():
                        phantom_source_lineups = []
                    source_options: list[tuple[str, str]] = []
                    if isinstance(our_generated_lineups, list) and bool(our_generated_lineups):
                        source_options.append(("generated_session", "Current Session Lineups"))
                    if isinstance(phantom_source_lineups, list) and bool(phantom_source_lineups):
                        phantom_run_id = str(phantom_source_meta.get("run_id") or "").strip()
                        phantom_source_label = (
                            f"Phantom Run Lineups ({phantom_run_id})" if phantom_run_id else "Phantom Run Lineups"
                        )
                        source_options.append(("phantom_run", phantom_source_label))

                    selected_source_key = ""
                    selected_source_label = ""
                    selected_gap_lineups: list[dict[str, Any]] = []
                    if len(source_options) > 1:
                        option_labels = [label for _, label in source_options]
                        default_label = option_labels[0]
                        selected_label = st.selectbox(
                            "Comparison Lineup Source",
                            options=option_labels,
                            index=0,
                            key="tournament_top10_gap_lineup_source",
                            help="Choose whether to compare winners against current generated lineups or loaded phantom-run lineups.",
                        )
                        selected_source_key = next((k for k, lbl in source_options if lbl == selected_label), "")
                        selected_source_label = selected_label
                    elif len(source_options) == 1:
                        selected_source_key, selected_source_label = source_options[0]

                    if selected_source_key == "generated_session":
                        selected_gap_lineups = list(our_generated_lineups)
                    elif selected_source_key == "phantom_run":
                        selected_gap_lineups = list(phantom_source_lineups)

                    top10_gap = build_top10_winner_gap_analysis(
                        entries_df=entries_df,
                        expanded_players_df=expanded_df,
                        projection_comparison_df=proj_compare_df,
                        generated_lineups=selected_gap_lineups,
                        top_n_winners=10,
                        top_points_focus=10,
                    )
                    top10_gap_summary = top10_gap.get("summary") or {}
                    top10_focus_df = top10_gap.get("focus_players_df")
                    top10_missing_df = top10_gap.get("missing_focus_players_df")
                    top10_hits_df = top10_gap.get("lineup_top3_hit_distribution_df")

                    tg1, tg2, tg3, tg4, tg5 = st.columns(5)
                    tg1.metric("Top-10 Winners", int(top10_gap_summary.get("top10_entries_count") or 0))
                    tg2.metric("Top-10 Unique Players", int(top10_gap_summary.get("top10_unique_players") or 0))
                    if bool(top10_gap_summary.get("our_lineups_available")):
                        if selected_source_label:
                            st.caption(f"Lineup source: `{selected_source_label}`")
                        top_scorer_flag = pd.to_numeric(top10_gap_summary.get("top_scorer_in_our_lineups"), errors="coerce")
                        top3_together_flag = pd.to_numeric(top10_gap_summary.get("top3_all_in_single_lineup"), errors="coerce")
                        tg3.metric(
                            "Top Scorer In Our Lineups",
                            (
                                "n/a"
                                if pd.isna(top_scorer_flag)
                                else ("Yes" if bool(int(top_scorer_flag)) else "No")
                            ),
                        )
                        top3_target_count = int(top10_gap_summary.get("top3_target_count") or 0)
                        top3_covered_count = int(top10_gap_summary.get("top3_covered_count") or 0)
                        tg4.metric(
                            "Top-3 Covered",
                            ("n/a" if top3_target_count <= 0 else f"{top3_covered_count}/{top3_target_count}"),
                        )
                        tg5.metric(
                            "Top-3 Together",
                            (
                                "n/a"
                                if pd.isna(top3_together_flag)
                                else ("Yes" if bool(int(top3_together_flag)) else "No")
                            ),
                        )
                    else:
                        tg3.metric("Top Scorer In Our Lineups", "n/a")
                        tg4.metric("Top-3 Covered", "n/a")
                        tg5.metric("Top-3 Together", "n/a")
                        st.info(
                            "No lineup source is loaded for comparison. Generate lineups in this session "
                            "or run Phantom Review (Saved Runs) to compare against those lineups."
                        )

                    top_scorer_name = str(top10_gap_summary.get("top_scorer_name") or "").strip()
                    top_scorer_points = pd.to_numeric(top10_gap_summary.get("top_scorer_actual_points"), errors="coerce")
                    if top_scorer_name and pd.notna(top_scorer_points):
                        st.caption(f"Top scorer in winning lineups: `{top_scorer_name}` ({float(top_scorer_points):.2f} DK points).")

                    if isinstance(top10_focus_df, pd.DataFrame) and not top10_focus_df.empty:
                        focus_cols = [
                            "Name",
                            "TeamAbbrev",
                            "actual_dk_points",
                            "blended_projection",
                            "blend_error",
                            "top10_entries_with_player",
                            "top10_entry_rate_pct",
                            "our_lineups_with_player",
                            "our_lineup_rate_pct",
                            "missed_by_our_lineups",
                        ]
                        focus_use_cols = [c for c in focus_cols if c in top10_focus_df.columns]
                        st.dataframe(top10_focus_df[focus_use_cols], hide_index=True, use_container_width=True)
                    else:
                        st.info(
                            "Top-10 winner player comparison needs parsed top-10 entries plus projection-vs-actual rows."
                        )

                    if (
                        bool(top10_gap_summary.get("our_lineups_available"))
                        and isinstance(top10_missing_df, pd.DataFrame)
                        and not top10_missing_df.empty
                    ):
                        st.caption(
                            "High-impact players from top winners that did not appear in any lineups from the selected source."
                        )
                        miss_cols = [
                            "Name",
                            "TeamAbbrev",
                            "actual_dk_points",
                            "blended_projection",
                            "blend_error",
                            "top10_entries_with_player",
                            "top10_entry_rate_pct",
                        ]
                        miss_use_cols = [c for c in miss_cols if c in top10_missing_df.columns]
                        st.dataframe(top10_missing_df[miss_use_cols], hide_index=True, use_container_width=True)

                    if bool(top10_gap_summary.get("our_lineups_available")) and isinstance(top10_hits_df, pd.DataFrame):
                        if not top10_hits_df.empty:
                            st.caption("Distribution of how many top-3 scorers appeared together in each lineup from the selected source.")
                            st.dataframe(top10_hits_df, hide_index=True, use_container_width=True)

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
                    st.session_state["cbb_tr_expanded_df"] = expanded_df.copy()
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
                st.session_state.pop("cbb_phantom_source_lineups", None)
                st.session_state.pop("cbb_phantom_source_meta", None)

            phantom_manifests = load_saved_lineup_run_manifests(
                bucket_name=bucket_name,
                selected_date=tr_date,
                selected_slate_key=shared_slate_key,
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
                    run_slate = _normalize_slate_label(manifest.get("slate_label") or manifest.get("slate_key"))
                    label = f"{generated_at} | {run_id} | {run_mode} | {run_slate}"
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
                selected_manifest_slate_key = _slate_key_from_label(
                    selected_manifest.get("slate_key") or selected_manifest.get("slate_label"),
                    default="main",
                )
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
                        st.session_state.pop("cbb_phantom_source_lineups", None)
                        st.session_state.pop("cbb_phantom_source_meta", None)
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
                                phantom_source_lineups: list[dict[str, Any]] = []
                                skipped_versions: list[str] = []
                                for version_key in selected_versions:
                                    payload = load_saved_lineup_version_payload(
                                        bucket_name=bucket_name,
                                        selected_date=tr_date,
                                        run_id=selected_run_id,
                                        version_key=version_key,
                                        selected_slate_key=selected_manifest_slate_key,
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
                                    for idx, lineup in enumerate(lineups):
                                        if not isinstance(lineup, dict):
                                            continue
                                        lineup_copy = dict(lineup)
                                        lineup_uid = (
                                            f"{version_key}:"
                                            + (
                                                str(lineup.get("lineup_number") or "").strip()
                                                or f"{idx + 1}"
                                            )
                                        )
                                        lineup_copy["lineup_uid"] = lineup_uid
                                        lineup_copy.setdefault("version_key", version_key)
                                        lineup_copy.setdefault("version_label", version_label)
                                        phantom_source_lineups.append(lineup_copy)
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
                                    st.session_state["cbb_phantom_source_lineups"] = phantom_source_lineups
                                    st.session_state["cbb_phantom_source_meta"] = {
                                        "date": tr_date.isoformat(),
                                        "run_id": selected_run_id,
                                        "versions": list(selected_versions),
                                        "lineup_count": int(len(phantom_source_lineups)),
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

            st.markdown("---")
            st.subheader("Tournament Postmortem Agent")
            st.caption(
                "Combines field standings and phantom-run results to highlight what worked, what failed, "
                "and what lineup construction changes to test next."
            )

            pm_entries_df = field_entries_df.copy() if isinstance(field_entries_df, pd.DataFrame) else pd.DataFrame()
            pm_expanded_df = expanded_df.copy() if isinstance(expanded_df, pd.DataFrame) else pd.DataFrame()
            pm_exposure_df = exposure_df.copy() if isinstance(exposure_df, pd.DataFrame) else pd.DataFrame()
            pm_proj_compare_df = proj_compare_df.copy() if isinstance(proj_compare_df, pd.DataFrame) else pd.DataFrame()
            pm_adjust_df = adjust_df.copy() if isinstance(adjust_df, pd.DataFrame) else pd.DataFrame()
            if pm_entries_df.empty:
                state_entries = st.session_state.get("cbb_tr_entries_df")
                pm_entries_df = state_entries.copy() if isinstance(state_entries, pd.DataFrame) else pd.DataFrame()
            if pm_expanded_df.empty:
                state_expanded = st.session_state.get("cbb_tr_expanded_df")
                pm_expanded_df = state_expanded.copy() if isinstance(state_expanded, pd.DataFrame) else pd.DataFrame()
            if pm_exposure_df.empty:
                state_exposure = st.session_state.get("cbb_tr_exposure_df")
                pm_exposure_df = state_exposure.copy() if isinstance(state_exposure, pd.DataFrame) else pd.DataFrame()
            if pm_proj_compare_df.empty:
                state_proj_compare = st.session_state.get("cbb_tr_projection_compare_df")
                pm_proj_compare_df = (
                    state_proj_compare.copy() if isinstance(state_proj_compare, pd.DataFrame) else pd.DataFrame()
                )
            if pm_adjust_df.empty:
                state_adjust = st.session_state.get("cbb_tr_adjust_df")
                pm_adjust_df = state_adjust.copy() if isinstance(state_adjust, pd.DataFrame) else pd.DataFrame()
            pm_phantom_df = phantom_df_state.copy() if isinstance(phantom_df_state, pd.DataFrame) else pd.DataFrame()
            pm_phantom_summary_df = (
                phantom_summary_state.copy() if isinstance(phantom_summary_state, pd.DataFrame) else pd.DataFrame()
            )

            if pm_entries_df.empty or pm_proj_compare_df.empty:
                st.info(
                    "Run Tournament Review first (standings + projection comparison), then run Phantom Review "
                    "to include generated-lineup construction analysis. Exposure/ownership metrics are optional but recommended."
                )
            else:
                pf1, pf2 = st.columns(2)
                post_focus_limit = int(
                    pf1.slider(
                        "Postmortem Focus Items",
                        min_value=5,
                        max_value=40,
                        value=12,
                        step=1,
                        key="tournament_postmortem_focus_limit",
                    )
                )
                missed_stack_underexposure_ratio = float(
                    pf2.slider(
                        "Missed Stack Under-Exposure %",
                        min_value=0,
                        max_value=100,
                        value=60,
                        step=5,
                        key="tournament_postmortem_stack_underexposure_ratio_pct",
                        help=(
                            "Flags stacks when phantom lineup rate is below this percentage of field stack rate. "
                            "Example: 60% means phantom rate < 0.60 * field rate is considered missed."
                        ),
                    )
                ) / 100.0
                post_packet = build_daily_ai_review_packet(
                    review_date=tr_date.isoformat(),
                    contest_id=tr_contest_id,
                    projection_comparison_df=pm_proj_compare_df,
                    entries_df=pm_entries_df,
                    exposure_df=pm_exposure_df,
                    phantom_summary_df=pm_phantom_summary_df,
                    phantom_lineups_df=pm_phantom_df,
                    adjustment_factors_df=pm_adjust_df,
                    focus_limit=post_focus_limit,
                )
                scorecards = post_packet.get("scorecards") or {}
                projection_quality = scorecards.get("projection_quality") or {}
                ownership_quality = scorecards.get("ownership_quality") or {}
                lineup_quality = scorecards.get("lineup_quality") or {}
                field_quality = scorecards.get("field_quality") or {}

                top10_entries_df = pm_entries_df.copy()
                if "Rank" in top10_entries_df.columns:
                    top10_entries_df = top10_entries_df.nsmallest(10, "Rank")
                else:
                    top10_entries_df = top10_entries_df.head(10)
                top10_user_df = build_user_strategy_summary(top10_entries_df)
                if not top10_user_df.empty and "best_rank" in top10_user_df.columns:
                    top10_user_df = top10_user_df.sort_values(["best_rank", "most_points"], ascending=[True, False])

                phantom_construction_df = pd.DataFrame()
                if not pm_phantom_df.empty:
                    phantom_working = pm_phantom_df.copy()
                    group_cols = [
                        c
                        for c in ["version_key", "version_label", "lineup_strategy", "stack_signature", "salary_texture_bucket"]
                        if c in phantom_working.columns
                    ]
                    if not group_cols:
                        group_cols = [c for c in ["version_key", "version_label"] if c in phantom_working.columns]
                    if group_cols:
                        for col in group_cols:
                            phantom_working[col] = (
                                phantom_working[col].astype(str).str.strip().replace({"": "n/a", "nan": "n/a"})
                            )
                        count_col = "lineup_number" if "lineup_number" in phantom_working.columns else group_cols[0]
                        agg_kwargs: dict[str, Any] = {"lineups": (count_col, "count")}
                        if "actual_points" in phantom_working.columns:
                            agg_kwargs["avg_actual_points"] = ("actual_points", lambda s: float(pd.to_numeric(s, errors="coerce").mean()))
                            agg_kwargs["best_actual_points"] = ("actual_points", lambda s: float(pd.to_numeric(s, errors="coerce").max()))
                        if "actual_minus_projected" in phantom_working.columns:
                            agg_kwargs["avg_actual_minus_projected"] = (
                                "actual_minus_projected",
                                lambda s: float(pd.to_numeric(s, errors="coerce").mean()),
                            )
                        if "would_beat_pct" in phantom_working.columns:
                            agg_kwargs["avg_would_beat_pct"] = (
                                "would_beat_pct",
                                lambda s: float(pd.to_numeric(s, errors="coerce").mean()),
                            )
                        if "salary_left" in phantom_working.columns:
                            agg_kwargs["avg_salary_left"] = ("salary_left", lambda s: float(pd.to_numeric(s, errors="coerce").mean()))
                        phantom_construction_df = phantom_working.groupby(group_cols, as_index=False).agg(**agg_kwargs)
                        sort_col = "avg_would_beat_pct" if "avg_would_beat_pct" in phantom_construction_df.columns else "lineups"
                        phantom_construction_df = phantom_construction_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

                missed_player_df = pd.DataFrame()
                low_own_standout_df = pd.DataFrame()
                own_surprise_standout_df = pd.DataFrame()
                if not pm_proj_compare_df.empty:
                    missed_player_df = pm_proj_compare_df.copy()
                    missed_player_df["Name"] = missed_player_df.get("Name", "").astype(str).str.strip()
                    missed_player_df["TeamAbbrev"] = missed_player_df.get("TeamAbbrev", "").astype(str).str.strip().str.upper()
                    missed_player_df["name_key"] = missed_player_df["Name"].map(
                        lambda x: re.sub(r"[^a-z0-9]", "", str(x or "").strip().lower())
                    )
                    missed_player_df["blend_error"] = pd.to_numeric(missed_player_df.get("blend_error"), errors="coerce")
                    missed_player_df["actual_dk_points"] = pd.to_numeric(missed_player_df.get("actual_dk_points"), errors="coerce")
                    if not pm_exposure_df.empty:
                        exposure_join = pm_exposure_df.copy()
                        exposure_join["Name"] = exposure_join.get("Name", "").astype(str).str.strip()
                        exposure_join["TeamAbbrev"] = exposure_join.get("TeamAbbrev", "").astype(str).str.strip().str.upper()
                        exposure_join["name_key"] = exposure_join["Name"].map(
                            lambda x: re.sub(r"[^a-z0-9]", "", str(x or "").strip().lower())
                        )
                        keep_cols = [
                            c
                            for c in [
                                "Name",
                                "TeamAbbrev",
                                "name_key",
                                "field_ownership_pct",
                                "projected_ownership",
                                "actual_ownership_from_file",
                                "ownership_diff_vs_proj",
                                "final_dk_points",
                            ]
                            if c in exposure_join.columns
                        ]
                        exposure_join = exposure_join[keep_cols]
                        strict_cols = [
                            c
                            for c in [
                                "Name",
                                "TeamAbbrev",
                                "field_ownership_pct",
                                "projected_ownership",
                                "actual_ownership_from_file",
                                "ownership_diff_vs_proj",
                                "final_dk_points",
                            ]
                            if c in exposure_join.columns
                        ]
                        exposure_strict = exposure_join.drop_duplicates(["Name", "TeamAbbrev"])
                        missed_player_df = missed_player_df.merge(
                            exposure_strict[strict_cols],
                            on=["Name", "TeamAbbrev"],
                            how="left",
                        )
                        exposure_by_name = exposure_join.sort_values("field_ownership_pct", ascending=False).drop_duplicates(
                            ["name_key"]
                        )
                        missed_player_df = missed_player_df.merge(
                            exposure_by_name[
                                [
                                    c
                                    for c in [
                                        "name_key",
                                        "field_ownership_pct",
                                        "projected_ownership",
                                        "actual_ownership_from_file",
                                        "ownership_diff_vs_proj",
                                        "final_dk_points",
                                    ]
                                    if c in exposure_by_name.columns
                                ]
                            ],
                            on="name_key",
                            how="left",
                            suffixes=("", "_by_name"),
                        )
                        for col in [
                            "field_ownership_pct",
                            "projected_ownership",
                            "actual_ownership_from_file",
                            "ownership_diff_vs_proj",
                            "final_dk_points",
                        ]:
                            by_name_col = f"{col}_by_name"
                            if by_name_col in missed_player_df.columns:
                                missed_player_df[col] = pd.to_numeric(missed_player_df.get(col), errors="coerce").where(
                                    pd.to_numeric(missed_player_df.get(col), errors="coerce").notna(),
                                    pd.to_numeric(missed_player_df.get(by_name_col), errors="coerce"),
                                )
                        missed_player_df = missed_player_df.drop(
                            columns=[f"{c}_by_name" for c in ["field_ownership_pct", "projected_ownership", "actual_ownership_from_file", "ownership_diff_vs_proj", "final_dk_points"]],
                            errors="ignore",
                        )
                    for col in ["field_ownership_pct", "projected_ownership", "actual_ownership_from_file"]:
                        if col not in missed_player_df.columns:
                            missed_player_df[col] = pd.NA
                    missed_player_df["field_ownership_pct"] = pd.to_numeric(
                        missed_player_df["field_ownership_pct"], errors="coerce"
                    )
                    missed_player_df["projected_ownership"] = pd.to_numeric(
                        missed_player_df["projected_ownership"], errors="coerce"
                    )
                    missed_player_df["actual_ownership_from_file"] = pd.to_numeric(
                        missed_player_df["actual_ownership_from_file"], errors="coerce"
                    )
                    missed_player_df["true_low_own_smash_flag"] = (
                        missed_player_df["actual_dk_points"].fillna(0.0) >= 35.0
                    ) & (
                        missed_player_df["field_ownership_pct"].fillna(999.0) <= 10.0
                    )
                    missed_player_df["projected_low_own_smash_flag"] = (
                        missed_player_df["actual_dk_points"].fillna(0.0) >= 35.0
                    ) & (
                        missed_player_df["projected_ownership"].fillna(999.0) <= 10.0
                    )
                    missed_player_df["ownership_surprise_smash_flag"] = (
                        missed_player_df["projected_low_own_smash_flag"].fillna(False)
                        & (missed_player_df["field_ownership_pct"].fillna(0.0) > 10.0)
                    )
                    missed_player_df = missed_player_df.loc[missed_player_df["blend_error"].fillna(-999.0) > 0]
                    missed_player_df = missed_player_df.sort_values("blend_error", ascending=False).head(post_focus_limit)
                    low_own_standout_df = (
                        missed_player_df.loc[missed_player_df["true_low_own_smash_flag"].fillna(False)]
                        .sort_values("actual_dk_points", ascending=False)
                        .head(post_focus_limit)
                    )
                    own_surprise_standout_df = (
                        missed_player_df.loc[missed_player_df["ownership_surprise_smash_flag"].fillna(False)]
                        .sort_values("actual_dk_points", ascending=False)
                        .head(post_focus_limit)
                    )
                    missed_player_df = missed_player_df.drop(columns=["name_key"], errors="ignore")
                    low_own_standout_df = low_own_standout_df.drop(columns=["name_key"], errors="ignore")
                    own_surprise_standout_df = own_surprise_standout_df.drop(columns=["name_key"], errors="ignore")

                team_stack_summary_df = pd.DataFrame()
                game_stack_summary_df = pd.DataFrame()
                missed_game_stack_df = pd.DataFrame()
                phantom_game_exposure_df = pd.DataFrame()
                if not pm_phantom_df.empty:
                    phantom_keys_rows: list[dict[str, Any]] = []
                    phantom_stack_work = pm_phantom_df.copy()
                    if "lineup_number" in phantom_stack_work.columns:
                        phantom_stack_work["_lineup_uid"] = phantom_stack_work["lineup_number"].astype(str).str.strip()
                    else:
                        phantom_stack_work["_lineup_uid"] = phantom_stack_work.index.map(lambda x: f"lineup_{x}")
                    for _, prow in phantom_stack_work.iterrows():
                        lineup_uid = str(prow.get("_lineup_uid") or "").strip()
                        if not lineup_uid:
                            continue
                        keys: set[str] = set()
                        anchor_key = str(prow.get("anchor_game_key") or "").strip().upper()
                        if anchor_key and anchor_key.lower() not in {"nan", "none", "null"}:
                            keys.add(anchor_key)
                        stack_signature_text = str(prow.get("stack_signature") or "").strip().upper()
                        if stack_signature_text and stack_signature_text.lower() not in {"nan", "none", "null"}:
                            for match in re.finditer(r"[A-Z0-9.&']+@[A-Z0-9.&']+", stack_signature_text):
                                game_key = str(match.group(0) or "").strip().upper()
                                if game_key:
                                    keys.add(game_key)
                        for game_key in keys:
                            phantom_keys_rows.append({"lineup_uid": lineup_uid, "game_key": game_key})
                    if phantom_keys_rows:
                        phantom_game_exposure_df = pd.DataFrame(phantom_keys_rows).drop_duplicates(
                            subset=["lineup_uid", "game_key"]
                        )
                        phantom_game_exposure_df = (
                            phantom_game_exposure_df.groupby("game_key", as_index=False)
                            .agg(phantom_lineups_with_game=("lineup_uid", "nunique"))
                            .sort_values("phantom_lineups_with_game", ascending=False)
                            .reset_index(drop=True)
                        )

                if not pm_expanded_df.empty and "EntryId" in pm_expanded_df.columns:
                    stack_work = pm_expanded_df.copy()
                    stack_work["EntryId"] = stack_work["EntryId"].astype(str).str.strip()
                    if "TeamAbbrev" in stack_work.columns:
                        stack_work["TeamAbbrev"] = stack_work["TeamAbbrev"].astype(str).str.strip().str.upper()
                    else:
                        stack_work["TeamAbbrev"] = ""
                    if "game_key" in stack_work.columns:
                        stack_work["game_key"] = stack_work["game_key"].astype(str).str.strip().str.upper()
                    else:
                        stack_work["game_key"] = ""
                    top10_entry_ids: set[str] = set()
                    if "EntryId" in top10_entries_df.columns:
                        top10_entry_ids = set(top10_entries_df["EntryId"].astype(str).str.strip().tolist())

                    team_entries = (
                        stack_work.loc[stack_work["TeamAbbrev"] != ""]
                        .groupby(["EntryId", "TeamAbbrev"], as_index=False)
                        .size()
                        .rename(columns={"size": "stack_size"})
                    )
                    team_entries = team_entries.loc[team_entries["stack_size"] >= 2].copy()
                    if not team_entries.empty:
                        team_entries["in_top10"] = team_entries["EntryId"].isin(top10_entry_ids)
                        team_stack_summary_df = (
                            team_entries.groupby("TeamAbbrev", as_index=False)
                            .agg(
                                field_entries_with_stack=("EntryId", "nunique"),
                                top10_entries_with_stack=("in_top10", "sum"),
                                avg_stack_size=("stack_size", "mean"),
                            )
                            .sort_values(["top10_entries_with_stack", "field_entries_with_stack", "avg_stack_size"], ascending=[False, False, False])
                            .reset_index(drop=True)
                        )

                    game_entries = (
                        stack_work.loc[stack_work["game_key"] != ""]
                        .groupby(["EntryId", "game_key"], as_index=False)
                        .size()
                        .rename(columns={"size": "stack_size"})
                    )
                    game_entries = game_entries.loc[game_entries["stack_size"] >= 3].copy()
                    if not game_entries.empty:
                        field_lineup_total = int(stack_work["EntryId"].astype(str).str.strip().nunique())
                        field_lineup_total = max(1, field_lineup_total)
                        phantom_lineup_total = 0
                        if not pm_phantom_df.empty:
                            if "lineup_number" in pm_phantom_df.columns:
                                phantom_lineup_total = int(pm_phantom_df["lineup_number"].astype(str).str.strip().nunique())
                            else:
                                phantom_lineup_total = int(len(pm_phantom_df))
                        phantom_lineup_total = max(0, phantom_lineup_total)

                        game_entries["in_top10"] = game_entries["EntryId"].isin(top10_entry_ids)
                        game_stack_summary_df = (
                            game_entries.groupby("game_key", as_index=False)
                            .agg(
                                field_entries_with_stack=("EntryId", "nunique"),
                                top10_entries_with_stack=("in_top10", "sum"),
                                avg_stack_size=("stack_size", "mean"),
                            )
                            .sort_values(["top10_entries_with_stack", "field_entries_with_stack", "avg_stack_size"], ascending=[False, False, False])
                            .reset_index(drop=True)
                        )
                        game_stack_summary_df["field_stack_rate"] = (
                            pd.to_numeric(game_stack_summary_df["field_entries_with_stack"], errors="coerce").fillna(0.0)
                            / float(field_lineup_total)
                        )
                        if not phantom_game_exposure_df.empty:
                            game_stack_summary_df = game_stack_summary_df.merge(
                                phantom_game_exposure_df,
                                on="game_key",
                                how="left",
                            )
                            game_stack_summary_df["phantom_lineups_with_game"] = (
                                pd.to_numeric(game_stack_summary_df["phantom_lineups_with_game"], errors="coerce")
                                .fillna(0)
                                .astype(int)
                            )
                        else:
                            game_stack_summary_df["phantom_lineups_with_game"] = 0

                        if phantom_lineup_total > 0:
                            game_stack_summary_df["phantom_stack_rate"] = (
                                pd.to_numeric(game_stack_summary_df["phantom_lineups_with_game"], errors="coerce").fillna(0.0)
                                / float(phantom_lineup_total)
                            )
                        else:
                            game_stack_summary_df["phantom_stack_rate"] = 0.0
                        game_stack_summary_df["phantom_to_field_stack_rate"] = (
                            pd.to_numeric(game_stack_summary_df["phantom_stack_rate"], errors="coerce").fillna(0.0)
                            / pd.to_numeric(game_stack_summary_df["field_stack_rate"], errors="coerce").replace(0, pd.NA)
                        )
                        game_stack_summary_df["phantom_to_field_stack_rate"] = pd.to_numeric(
                            game_stack_summary_df["phantom_to_field_stack_rate"], errors="coerce"
                        ).fillna(0.0)
                        game_stack_summary_df["stack_underexposure_gap"] = (
                            pd.to_numeric(game_stack_summary_df["field_stack_rate"], errors="coerce").fillna(0.0)
                            - pd.to_numeric(game_stack_summary_df["phantom_stack_rate"], errors="coerce").fillna(0.0)
                        )
                        if phantom_lineup_total > 0:
                            missed_game_stack_df = game_stack_summary_df.loc[
                                (pd.to_numeric(game_stack_summary_df["top10_entries_with_stack"], errors="coerce").fillna(0) > 0)
                                & (
                                    (
                                        pd.to_numeric(game_stack_summary_df["phantom_lineups_with_game"], errors="coerce")
                                        .fillna(0)
                                        <= 0
                                    )
                                    | (
                                        pd.to_numeric(game_stack_summary_df["phantom_stack_rate"], errors="coerce").fillna(0.0)
                                        < (
                                            pd.to_numeric(game_stack_summary_df["field_stack_rate"], errors="coerce").fillna(0.0)
                                            * float(missed_stack_underexposure_ratio)
                                        )
                                    )
                                )
                            ].copy()
                            missed_game_stack_df = missed_game_stack_df.sort_values(
                                ["top10_entries_with_stack", "stack_underexposure_gap", "field_entries_with_stack"],
                                ascending=[False, False, False],
                            ).head(post_focus_limit)

                rw1, rw2, rw3, rw4, rw5 = st.columns(5)
                rw1.metric("Field Entries", _safe_int_value(field_quality.get("field_entries"), default=0))
                rw2.metric("Projection Blend MAE", f"{_safe_float_value(projection_quality.get('blend_mae'), default=0.0):.2f}")
                rw3.metric("Ownership MAE", f"{_safe_float_value(ownership_quality.get('ownership_mae'), default=0.0):.2f}")
                rw4.metric("Phantom Lineups", _safe_int_value(lineup_quality.get("lineups_scored"), default=0))
                rw5.metric("Phantom Avg Beat %", f"{_safe_float_value(lineup_quality.get('avg_would_beat_pct'), default=0.0):.1f}")

                blend_mae = _safe_float_value(projection_quality.get("blend_mae"), default=0.0)
                proj_rank_corr = _safe_float_value(projection_quality.get("blended_rank_spearman"), default=0.0)
                ownership_mae = _safe_float_value(ownership_quality.get("ownership_mae"), default=0.0)
                ownership_rank_corr = _safe_float_value(ownership_quality.get("ownership_rank_spearman"), default=0.0)
                lineup_scored = _safe_int_value(lineup_quality.get("lineups_scored"), default=0)
                lineup_avg_delta = _safe_float_value(lineup_quality.get("avg_actual_minus_projected"), default=0.0)
                lineup_avg_beat = _safe_float_value(lineup_quality.get("avg_would_beat_pct"), default=0.0)
                field_avg_salary_left = _safe_float_value(field_quality.get("avg_salary_left"), default=0.0)
                field_top10_salary_left = _safe_float_value(field_quality.get("top10_avg_salary_left"), default=0.0)

                winner_points = 0.0
                if "Points" in pm_entries_df.columns:
                    winner_points = _safe_float_value(pd.to_numeric(pm_entries_df["Points"], errors="coerce").max(), default=0.0)
                best_phantom_points = 0.0
                if "actual_points" in pm_phantom_df.columns:
                    best_phantom_points = _safe_float_value(
                        pd.to_numeric(pm_phantom_df["actual_points"], errors="coerce").max(),
                        default=0.0,
                    )
                elif "best_actual_points" in pm_phantom_summary_df.columns:
                    best_phantom_points = _safe_float_value(
                        pd.to_numeric(pm_phantom_summary_df["best_actual_points"], errors="coerce").max(),
                        default=0.0,
                    )
                phantom_gap_to_winner = (winner_points - best_phantom_points) if (winner_points > 0.0 and best_phantom_points > 0.0) else None

                top10_concentration = 0.0
                if not top10_user_df.empty and "entries" in top10_user_df.columns:
                    top10_total_entries = float(pd.to_numeric(top10_user_df["entries"], errors="coerce").fillna(0.0).sum())
                    top10_max_entries = float(pd.to_numeric(top10_user_df["entries"], errors="coerce").fillna(0.0).max())
                    if top10_total_entries > 0:
                        top10_concentration = top10_max_entries / top10_total_entries
                mapping_coverage = 0.0
                if "mapped_players" in pm_entries_df.columns and "parsed_players" in pm_entries_df.columns:
                    parsed_players = pd.to_numeric(pm_entries_df["parsed_players"], errors="coerce").replace(0, pd.NA)
                    mapped_players = pd.to_numeric(pm_entries_df["mapped_players"], errors="coerce")
                    mapping_coverage = float((mapped_players / parsed_players).fillna(0.0).mean())
                projected_own_rows = int(
                    pd.to_numeric(pm_exposure_df.get("projected_ownership"), errors="coerce").notna().sum()
                ) if not pm_exposure_df.empty else 0
                actual_own_rows = int(
                    pd.to_numeric(pm_exposure_df.get("actual_ownership_from_file"), errors="coerce").notna().sum()
                ) if not pm_exposure_df.empty else 0

                right_notes: list[str] = []
                wrong_notes: list[str] = []
                if proj_rank_corr >= 0.20:
                    right_notes.append(f"Projection ordering held up with positive rank correlation (`{proj_rank_corr:.2f}`).")
                else:
                    wrong_notes.append(f"Projection ordering was weak (`{proj_rank_corr:.2f}` rank correlation).")
                if blend_mae <= 7.0:
                    right_notes.append(f"Projection absolute error was acceptable (`blend_mae={blend_mae:.2f}`).")
                else:
                    wrong_notes.append(f"Projection error was high (`blend_mae={blend_mae:.2f}`).")
                if ownership_rank_corr >= 0.15:
                    right_notes.append(f"Ownership directionality was useful (`rank_corr={ownership_rank_corr:.2f}`).")
                else:
                    wrong_notes.append(f"Ownership rank correlation was weak (`{ownership_rank_corr:.2f}`).")
                if ownership_mae <= 10.0:
                    right_notes.append(f"Ownership absolute error stayed reasonable (`ownership_mae={ownership_mae:.2f}`).")
                else:
                    wrong_notes.append(f"Ownership misses were large (`ownership_mae={ownership_mae:.2f}`).")
                if projected_own_rows <= 0:
                    wrong_notes.append("Projected ownership values are missing from exposure rows; verify projection snapshot for this slate.")
                else:
                    right_notes.append(f"Projected ownership coverage is populated on `{projected_own_rows}` exposure rows.")
                if actual_own_rows <= 0:
                    wrong_notes.append(
                        "Actual ownership rows from standings are missing; verify contest standings format and slate mapping."
                    )
                if lineup_scored > 0:
                    if lineup_avg_delta >= 0.0:
                        right_notes.append(
                            "Phantom constructions beat their own projections on average "
                            f"(`avg_actual_minus_projected={lineup_avg_delta:.2f}`)."
                        )
                    else:
                        wrong_notes.append(
                            "Phantom constructions underperformed projections "
                            f"(`avg_actual_minus_projected={lineup_avg_delta:.2f}`)."
                        )
                    if lineup_avg_beat >= 50.0:
                        right_notes.append(f"Phantom builds were competitive vs field (`avg_would_beat_pct={lineup_avg_beat:.1f}`).")
                    else:
                        wrong_notes.append(f"Phantom builds lagged field (`avg_would_beat_pct={lineup_avg_beat:.1f}`).")
                else:
                    wrong_notes.append("No phantom runs are scored yet for this date; construction feedback is incomplete.")
                if field_top10_salary_left <= field_avg_salary_left:
                    right_notes.append(
                        "Top-10 lineups generally spent more salary than field average "
                        f"(`top10={field_top10_salary_left:.0f}`, `field={field_avg_salary_left:.0f}`)."
                    )
                else:
                    wrong_notes.append(
                        "Top-10 lineups left more salary than the field average "
                        f"(`top10={field_top10_salary_left:.0f}`, `field={field_avg_salary_left:.0f}`)."
                    )
                if phantom_gap_to_winner is not None:
                    if phantom_gap_to_winner <= 0.0:
                        right_notes.append("At least one phantom lineup matched or beat the field winner on actual points.")
                    else:
                        wrong_notes.append(
                            "Best phantom lineup trailed the field winner by "
                            f"`{phantom_gap_to_winner:.2f}` points."
                        )
                if mapping_coverage < 0.4:
                    wrong_notes.append(
                        "Slate mapping coverage is low "
                        f"(`{100.0 * mapping_coverage:.1f}%` mapped players); check that Tournament Review is on the correct slate key."
                    )
                elif mapping_coverage > 0:
                    right_notes.append(f"Slate mapping coverage is strong (`{100.0 * mapping_coverage:.1f}%`).")
                if not low_own_standout_df.empty:
                    right_notes.append(
                        f"Detected `{len(low_own_standout_df)}` true low-owned standouts who outperformed projections."
                    )
                if not own_surprise_standout_df.empty:
                    wrong_notes.append(
                        f"Detected `{len(own_surprise_standout_df)}` ownership-surprise standouts "
                        "(projected low-owned but actually came in above 10%)."
                    )
                if not missed_game_stack_df.empty:
                    wrong_notes.append(
                        f"Detected `{len(missed_game_stack_df)}` top field game stacks that were missing or materially under-exposed in phantom lineups."
                    )

                st.subheader("What Went Right")
                if right_notes:
                    st.markdown("\n".join([f"- {note}" for note in right_notes]))
                else:
                    st.caption("No strong positive signals yet from this sample.")

                st.subheader("What Went Wrong")
                if wrong_notes:
                    st.markdown("\n".join([f"- {note}" for note in wrong_notes]))
                else:
                    st.caption("No major failure signals detected from current metrics.")

                st.subheader("Top-10 User Insights")
                if top10_user_df.empty:
                    st.info("No top-10 user summary available from the loaded field entries.")
                else:
                    u1, u2, u3 = st.columns(3)
                    u1.metric("Top-10 Handles", int(len(top10_user_df)))
                    u2.metric("Largest Handle Share", f"{100.0 * top10_concentration:.1f}%")
                    u3.metric("Best Handle Score", f"{_safe_float_value(top10_user_df['most_points'].max(), default=0.0):.2f}")
                    st.dataframe(top10_user_df, hide_index=True, use_container_width=True)

                st.subheader("Phantom Construction Insights")
                if pm_phantom_df.empty:
                    st.info("Run Phantom Review to include generated-lineup construction insights.")
                elif phantom_construction_df.empty:
                    st.info("Phantom rows loaded, but no construction-level grouping columns were found.")
                else:
                    st.dataframe(phantom_construction_df.head(30), hide_index=True, use_container_width=True)

                st.subheader("Missed Player Standouts")
                if missed_player_df.empty:
                    st.info("No positive projection-miss player rows were found.")
                else:
                    missed_show_cols = [
                        "Name",
                        "TeamAbbrev",
                        "Position",
                        "Salary",
                        "blended_projection",
                        "actual_dk_points",
                        "blend_error",
                        "field_ownership_pct",
                        "projected_ownership",
                        "actual_ownership_from_file",
                        "true_low_own_smash_flag",
                        "projected_low_own_smash_flag",
                        "ownership_surprise_smash_flag",
                    ]
                    missed_use_cols = [c for c in missed_show_cols if c in missed_player_df.columns]
                    st.dataframe(missed_player_df[missed_use_cols], hide_index=True, use_container_width=True)

                st.subheader("True Low-Ownership Standouts")
                if low_own_standout_df.empty:
                    st.info("No true low-owned standout players were detected from current data.")
                else:
                    low_show_cols = [
                        "Name",
                        "TeamAbbrev",
                        "Position",
                        "actual_dk_points",
                        "blended_projection",
                        "blend_error",
                        "field_ownership_pct",
                        "projected_ownership",
                        "actual_ownership_from_file",
                        "true_low_own_smash_flag",
                        "projected_low_own_smash_flag",
                    ]
                    low_use_cols = [c for c in low_show_cols if c in low_own_standout_df.columns]
                    st.dataframe(low_own_standout_df[low_use_cols], hide_index=True, use_container_width=True)

                st.subheader("Ownership-Surprise Standouts")
                if own_surprise_standout_df.empty:
                    st.info(
                        "No ownership-surprise standouts were detected "
                        "(players projected <=10% that actually came in >10% and smashed)."
                    )
                else:
                    surprise_show_cols = [
                        "Name",
                        "TeamAbbrev",
                        "Position",
                        "actual_dk_points",
                        "blended_projection",
                        "blend_error",
                        "field_ownership_pct",
                        "projected_ownership",
                        "actual_ownership_from_file",
                        "ownership_surprise_smash_flag",
                    ]
                    surprise_use_cols = [c for c in surprise_show_cols if c in own_surprise_standout_df.columns]
                    st.dataframe(own_surprise_standout_df[surprise_use_cols], hide_index=True, use_container_width=True)

                st.subheader("Field Stack Signals")
                ss1, ss2 = st.columns(2)
                with ss1:
                    st.caption("Team stacks (2+ players in same lineup)")
                    if team_stack_summary_df.empty:
                        st.info("No team-stack summary available.")
                    else:
                        st.dataframe(team_stack_summary_df.head(20), hide_index=True, use_container_width=True)
                with ss2:
                    st.caption("Game stacks (3+ players from same game in same lineup)")
                    if game_stack_summary_df.empty:
                        st.info("No game-stack summary available.")
                    else:
                        st.dataframe(game_stack_summary_df.head(20), hide_index=True, use_container_width=True)
                if not missed_game_stack_df.empty:
                    st.caption(
                        "Likely missed field game stacks "
                        "(top-10 field presence with missing/materially under-exposed phantom lineup rate)"
                    )
                    st.dataframe(missed_game_stack_df, hide_index=True, use_container_width=True)

                improvement_rows: list[dict[str, Any]] = []
                improvement_rows.append(
                    {
                        "priority": 1,
                        "area": "Projection Calibration",
                        "why": f"blend_mae={blend_mae:.2f}, rank_corr={proj_rank_corr:.2f}",
                        "next_slate_change": "Apply role/salary-specific projection multipliers and verify by segment.",
                        "success_metric": "Reduce blend MAE by >= 1.0 while maintaining or improving rank correlation.",
                    }
                )
                improvement_rows.append(
                    {
                        "priority": 2,
                        "area": "Ownership Calibration",
                        "why": f"ownership_mae={ownership_mae:.2f}, rank_corr={ownership_rank_corr:.2f}",
                        "next_slate_change": "Adjust ownership model by projection tier and game environment flags.",
                        "success_metric": "Lower ownership MAE and increase ownership rank correlation on next slate.",
                    }
                )
                if lineup_scored > 0:
                    improvement_rows.append(
                        {
                            "priority": 3,
                            "area": "Phantom Construction",
                            "why": f"avg_actual_minus_projected={lineup_avg_delta:.2f}, avg_would_beat_pct={lineup_avg_beat:.1f}",
                            "next_slate_change": "Promote constructions with best phantom beat-rate and actual-minus-projection.",
                            "success_metric": "Increase avg_would_beat_pct and reduce negative actual-minus-projected drift.",
                        }
                    )
                if phantom_gap_to_winner is not None:
                    improvement_rows.append(
                        {
                            "priority": 4,
                            "area": "Ceiling Capture",
                            "why": f"winner_gap={phantom_gap_to_winner:.2f}",
                            "next_slate_change": "Increase exposure to high-upside game/team stack archetypes seen in top finishers.",
                            "success_metric": "Close winner gap and improve top-end phantom lineup outcomes.",
                        }
                    )
                improvement_rows.append(
                    {
                        "priority": 5,
                        "area": "Top-10 User Archetypes",
                        "why": f"largest_top10_handle_share={100.0 * top10_concentration:.1f}%",
                        "next_slate_change": "Mirror top-handle construction traits (salary left, team/game stack density).",
                        "success_metric": "Increase overlap with top-10 construction profile while keeping lineup diversity.",
                    }
                )
                if not low_own_standout_df.empty:
                    improvement_rows.append(
                        {
                            "priority": 6,
                            "area": "Low-Own Ceiling Capture",
                            "why": f"low_own_standouts={int(len(low_own_standout_df))}",
                            "next_slate_change": "Create explicit low-owned upside bucket and ensure target exposure in GPP runs.",
                            "success_metric": "Increase low-owned players that exceed projections by >= 8 DK points.",
                        }
                    )
                if not missed_game_stack_df.empty:
                    improvement_rows.append(
                        {
                            "priority": 7,
                            "area": "Missed Field Stacks",
                            "why": f"missed_field_game_stacks={int(len(missed_game_stack_df))}",
                            "next_slate_change": "Bias stack generation toward top field game-stack signals with phantom under-exposure.",
                            "success_metric": "Reduce missed top game-stack count and improve phantom top-end outcomes.",
                        }
                    )
                if not own_surprise_standout_df.empty:
                    improvement_rows.append(
                        {
                            "priority": 8,
                            "area": "Ownership Surprise Guardrails",
                            "why": f"ownership_surprise_standouts={int(len(own_surprise_standout_df))}",
                            "next_slate_change": (
                                "Add guardrails for players projected <=10% that are trending toward field ownership >10% "
                                "to avoid underweighting emerging chalk ceilings."
                            ),
                            "success_metric": "Lower ownership surprise miss count and improve ownership MAE on those players.",
                        }
                    )
                improvement_df = pd.DataFrame(improvement_rows).sort_values("priority")
                st.subheader("What To Improve Next Slate")
                st.dataframe(improvement_df, hide_index=True, use_container_width=True)

                postmortem_payload = {
                    "schema_version": "tournament_postmortem_v1",
                    "review_context": {
                        "review_date": tr_date.isoformat(),
                        "contest_id": str(tr_contest_id or "").strip(),
                    },
                    "data_quality": {
                        "mapped_player_coverage_pct": round(100.0 * float(mapping_coverage), 2),
                        "projected_ownership_rows": int(projected_own_rows),
                        "actual_ownership_rows": int(actual_own_rows),
                        "missed_stack_underexposure_ratio": float(missed_stack_underexposure_ratio),
                    },
                    "scorecards": scorecards,
                    "focus_tables": post_packet.get("focus_tables") or {},
                    "top10_user_summary": top10_user_df.to_dict(orient="records") if not top10_user_df.empty else [],
                    "missed_player_standouts": missed_player_df.to_dict(orient="records") if not missed_player_df.empty else [],
                    "low_own_standouts": low_own_standout_df.to_dict(orient="records") if not low_own_standout_df.empty else [],
                    "ownership_surprise_standouts": (
                        own_surprise_standout_df.to_dict(orient="records")
                        if not own_surprise_standout_df.empty
                        else []
                    ),
                    "field_team_stack_summary": (
                        team_stack_summary_df.head(40).to_dict(orient="records")
                        if not team_stack_summary_df.empty
                        else []
                    ),
                    "field_game_stack_summary": (
                        game_stack_summary_df.head(40).to_dict(orient="records")
                        if not game_stack_summary_df.empty
                        else []
                    ),
                    "phantom_game_exposure_summary": (
                        phantom_game_exposure_df.head(40).to_dict(orient="records")
                        if not phantom_game_exposure_df.empty
                        else []
                    ),
                    "likely_missed_game_stacks": (
                        missed_game_stack_df.to_dict(orient="records") if not missed_game_stack_df.empty else []
                    ),
                    "phantom_construction_summary": (
                        phantom_construction_df.head(40).to_dict(orient="records")
                        if not phantom_construction_df.empty
                        else []
                    ),
                    "phantom_summary": (
                        pm_phantom_summary_df.head(40).to_dict(orient="records")
                        if not pm_phantom_summary_df.empty
                        else []
                    ),
                    "improvement_plan": improvement_df.to_dict(orient="records"),
                }
                postmortem_prompt_system = (
                    "You are a DFS tournament postmortem analyst. Use only evidence from the JSON packet. "
                    "Be concrete, concise, and prioritize actions by expected impact."
                )
                postmortem_prompt_user = (
                    "Review this tournament postmortem packet and produce a practical debrief.\n\n"
                    "Required sections:\n"
                    "1) What Went Right (3-6 bullets)\n"
                    "2) What Went Wrong (3-6 bullets)\n"
                    "3) Missed Players, True Low-Ownership Standouts, and Ownership Surprises\n"
                    "4) Field vs Phantom Stack Findings (use lineup-level phantom exposure and include likely missed stacks)\n"
                    "5) Prioritized Tweaks for Next Slate (max 5, each with rationale + success metric)\n\n"
                    "Constraints:\n"
                    "- Use only evidence in the JSON packet.\n"
                    "- Cite exact metric names/values.\n"
                    "- Reference `data_quality` when signals are missing or unreliable.\n"
                    "- If evidence is insufficient, state what is missing.\n\n"
                    "JSON packet:\n"
                    f"{json.dumps(postmortem_payload, indent=2, ensure_ascii=True)}\n"
                )

                postmortem_packet_json = json.dumps(postmortem_payload, indent=2, ensure_ascii=True)
                d1, d2 = st.columns(2)
                d1.download_button(
                    "Download Tournament Postmortem Packet JSON",
                    data=postmortem_packet_json,
                    file_name=f"tournament_postmortem_packet_{tr_date.isoformat()}_{tr_contest_id}.json",
                    mime="application/json",
                    key="download_tournament_postmortem_packet_json",
                )
                d2.download_button(
                    "Download Tournament Postmortem Prompt",
                    data=postmortem_prompt_user,
                    file_name=f"tournament_postmortem_prompt_{tr_date.isoformat()}_{tr_contest_id}.txt",
                    mime="text/plain",
                    key="download_tournament_postmortem_prompt_txt",
                )

                openai_key = (os.getenv("OPENAI_API_KEY", "").strip() or (_secret("openai_api_key") or "").strip())
                ac1, ac2 = st.columns(2)
                postmortem_model = ac1.text_input(
                    "Postmortem Model",
                    value=str(st.session_state.get("ai_review_model", "gpt-5-mini")),
                    key="tournament_postmortem_model",
                ).strip()
                postmortem_tokens = int(
                    ac2.number_input(
                        "Postmortem Max Tokens",
                        min_value=400,
                        max_value=8000,
                        value=1600,
                        step=100,
                        key="tournament_postmortem_max_tokens",
                    )
                )
                run_postmortem_agent = st.button(
                    "Run Tournament Postmortem Agent",
                    key="run_tournament_postmortem_agent",
                )
                if run_postmortem_agent:
                    if not openai_key:
                        st.error("Set `OPENAI_API_KEY` (or `openai_api_key` in Streamlit secrets) to run the postmortem agent.")
                    else:
                        try:
                            with st.spinner("Running tournament postmortem agent..."):
                                postmortem_text = request_openai_review(
                                    api_key=openai_key,
                                    user_prompt=postmortem_prompt_user,
                                    system_prompt=postmortem_prompt_system,
                                    model=postmortem_model or "gpt-5-mini",
                                    max_output_tokens=postmortem_tokens,
                                )
                                st.session_state["cbb_tournament_postmortem_output"] = postmortem_text
                        except Exception as exc:
                            st.exception(exc)

                postmortem_output = str(st.session_state.get("cbb_tournament_postmortem_output") or "").strip()
                if postmortem_output:
                    st.text_area(
                        "Tournament Postmortem Output",
                        value=postmortem_output,
                        height=420,
                        key="tournament_postmortem_output_text",
                    )
                    st.download_button(
                        "Download Tournament Postmortem Output",
                        data=postmortem_output,
                        file_name=f"tournament_postmortem_output_{tr_date.isoformat()}_{tr_contest_id}.txt",
                        mime="text/plain",
                        key="download_tournament_postmortem_output_txt",
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
                            selected_slate_key=shared_slate_key,
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
                            selected_slate_key=shared_slate_key,
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
                            selected_slate_key=shared_slate_key,
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
                                try:
                                    run_ids = store.list_lineup_run_ids(review_day, shared_slate_key)
                                except TypeError:
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
                                global_max_tokens = int(st.session_state.get("ai_review_max_output_tokens", 1800))
                                global_ai_text = request_openai_review(
                                    api_key=openai_key,
                                    user_prompt=global_user_prompt,
                                    system_prompt=AI_REVIEW_SYSTEM_PROMPT,
                                    model=str(st.session_state.get("ai_review_model", "gpt-5-mini")),
                                    max_output_tokens=global_max_tokens,
                                )
                                st.session_state["cbb_global_ai_review_output"] = global_ai_text
                            except Exception as exc:
                                exc_text = str(exc or "")
                                if "max_output_tokens" in exc_text.lower():
                                    retry_tokens = min(8000, max(global_max_tokens + 800, int(global_max_tokens * 1.8)))
                                    try:
                                        st.caption(
                                            "Global review hit max-output truncation; retrying with "
                                            f"`max_output_tokens={retry_tokens}`."
                                        )
                                        global_ai_text = request_openai_review(
                                            api_key=openai_key,
                                            user_prompt=global_user_prompt,
                                            system_prompt=AI_REVIEW_SYSTEM_PROMPT,
                                            model=str(st.session_state.get("ai_review_model", "gpt-5-mini")),
                                            max_output_tokens=retry_tokens,
                                        )
                                        st.session_state["cbb_global_ai_review_output"] = global_ai_text
                                    except Exception as retry_exc:
                                        st.exception(retry_exc)
                                else:
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

            st.markdown("---")
            st.subheader("Market Correlation Agent (Multi-Date)")
            st.caption(
                "Correlates market signals (totals/spreads/vegas features) against ownership and actual points "
                "across multiple dates, then surfaces projection-calibration suggestions by bucket."
            )
            mc1, mc2, mc3 = st.columns(3)
            market_end_date = mc1.date_input(
                "Market Review End Date",
                value=ar_date,
                key="market_corr_end_date",
            )
            default_start = market_end_date - timedelta(days=29)
            market_start_date = mc2.date_input(
                "Market Review Start Date",
                value=default_start,
                key="market_corr_start_date",
            )
            market_focus_limit = int(
                mc3.slider(
                    "Market Focus Rows",
                    min_value=5,
                    max_value=80,
                    value=25,
                    step=1,
                    key="market_corr_focus_limit",
                )
            )
            mc4, mc5, mc6 = st.columns(3)
            market_min_bucket_samples = int(
                mc4.slider(
                    "Min Bucket Samples",
                    min_value=5,
                    max_value=120,
                    value=20,
                    step=1,
                    key="market_corr_min_bucket_samples",
                )
            )
            market_use_saved_run_dates = bool(
                mc5.checkbox(
                    "Use Saved Run Dates Only",
                    value=True,
                    key="market_corr_use_saved_dates",
                    help="If enabled, only dates with saved lineup runs are scanned.",
                )
            )
            build_market_packet_clicked = mc6.button(
                "Build Market Packet",
                key="build_market_corr_packet",
            )

            if build_market_packet_clicked:
                if market_start_date > market_end_date:
                    st.error("`Market Review Start Date` must be on or before `Market Review End Date`.")
                else:
                    try:
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

                        if market_use_saved_run_dates:
                            candidate_dates = load_saved_lineup_run_dates(
                                bucket_name=bucket_name,
                                selected_slate_key=shared_slate_key,
                                gcp_project=gcp_project or None,
                                service_account_json=cred_json,
                                service_account_json_b64=cred_json_b64,
                            )
                        else:
                            candidate_dates = [d for d in iter_dates(market_start_date, market_end_date)]
                        candidate_dates = [
                            d
                            for d in candidate_dates
                            if isinstance(d, date) and market_start_date <= d <= market_end_date
                        ]
                        candidate_dates = sorted(set(candidate_dates))

                        scanned_dates = 0
                        used_dates = 0
                        market_frames: list[pd.DataFrame] = []

                        for review_day in candidate_dates:
                            scanned_dates += 1
                            proj_snap_day = load_projection_snapshot_frame(
                                bucket_name=bucket_name,
                                selected_date=review_day,
                                selected_slate_key=shared_slate_key,
                                gcp_project=gcp_project or None,
                                service_account_json=cred_json,
                                service_account_json_b64=cred_json_b64,
                            )
                            actual_day = load_actual_results_frame_for_date(
                                bucket_name=bucket_name,
                                selected_date=review_day,
                                gcp_project=gcp_project or None,
                                service_account_json=cred_json,
                                service_account_json_b64=cred_json_b64,
                            )
                            if proj_snap_day.empty or actual_day.empty:
                                continue

                            proj_cmp_day = build_projection_actual_comparison(
                                projection_df=proj_snap_day,
                                actual_results_df=actual_day,
                            )
                            if proj_cmp_day.empty:
                                continue

                            market_day = proj_snap_day.copy()
                            cmp_day = proj_cmp_day.copy()

                            if "ID" in market_day.columns:
                                market_day["ID"] = market_day["ID"].astype(str).str.strip()
                            if "Name" in market_day.columns:
                                market_day["Name"] = market_day["Name"].astype(str).str.strip()
                            if "TeamAbbrev" in market_day.columns:
                                market_day["TeamAbbrev"] = market_day["TeamAbbrev"].astype(str).str.strip().str.upper()

                            if "ID" in cmp_day.columns:
                                cmp_day["ID"] = cmp_day["ID"].astype(str).str.strip()
                            if "Name" in cmp_day.columns:
                                cmp_day["Name"] = cmp_day["Name"].astype(str).str.strip()

                            cmp_cols = [c for c in ["actual_dk_points", "blend_error", "our_error", "vegas_error", "actual_minutes"] if c in cmp_day.columns]
                            if cmp_cols:
                                if "ID" in market_day.columns and "ID" in cmp_day.columns:
                                    cmp_by_id = (
                                        cmp_day.loc[cmp_day["ID"] != "", ["ID"] + cmp_cols]
                                        .drop_duplicates("ID")
                                    )
                                    market_day = market_day.merge(cmp_by_id, on="ID", how="left", suffixes=("", "_cmp_id"))
                                    for col in cmp_cols:
                                        cmp_col = f"{col}_cmp_id"
                                        if cmp_col in market_day.columns:
                                            if col not in market_day.columns:
                                                market_day[col] = pd.NA
                                            base_vals = pd.to_numeric(market_day[col], errors="coerce")
                                            fill_vals = pd.to_numeric(market_day[cmp_col], errors="coerce")
                                            market_day[col] = base_vals.where(base_vals.notna(), fill_vals)
                                            market_day = market_day.drop(columns=[cmp_col], errors="ignore")
                                elif "Name" in market_day.columns and "Name" in cmp_day.columns:
                                    cmp_by_name = (
                                        cmp_day.loc[cmp_day["Name"] != "", ["Name"] + cmp_cols]
                                        .drop_duplicates("Name")
                                    )
                                    market_day = market_day.merge(cmp_by_name, on="Name", how="left", suffixes=("", "_cmp_name"))
                                    for col in cmp_cols:
                                        cmp_col = f"{col}_cmp_name"
                                        if cmp_col in market_day.columns:
                                            if col not in market_day.columns:
                                                market_day[col] = pd.NA
                                            base_vals = pd.to_numeric(market_day[col], errors="coerce")
                                            fill_vals = pd.to_numeric(market_day[cmp_col], errors="coerce")
                                            market_day[col] = base_vals.where(base_vals.notna(), fill_vals)
                                            market_day = market_day.drop(columns=[cmp_col], errors="ignore")

                            own_day = load_ownership_frame_for_date(
                                bucket_name=bucket_name,
                                selected_date=review_day,
                                selected_slate_key=shared_slate_key,
                                gcp_project=gcp_project or None,
                                service_account_json=cred_json,
                                service_account_json_b64=cred_json_b64,
                            )
                            market_day["actual_ownership_from_file"] = pd.NA
                            if not own_day.empty:
                                own = own_day.copy()
                                if "ID" in own.columns:
                                    own["ID"] = own["ID"].astype(str).str.strip()
                                if "Name" in own.columns:
                                    own["Name"] = own["Name"].astype(str).str.strip()
                                if "actual_ownership" in own.columns:
                                    own["actual_ownership"] = pd.to_numeric(own["actual_ownership"], errors="coerce")
                                else:
                                    own["actual_ownership"] = pd.NA

                                if "ID" in market_day.columns and "ID" in own.columns:
                                    own_by_id = (
                                        own.loc[(own["ID"] != "") & own["actual_ownership"].notna(), ["ID", "actual_ownership"]]
                                        .drop_duplicates("ID")
                                        .rename(columns={"actual_ownership": "actual_ownership_id"})
                                    )
                                    if not own_by_id.empty:
                                        market_day = market_day.merge(own_by_id, on="ID", how="left")
                                        market_day["actual_ownership_from_file"] = pd.to_numeric(
                                            market_day.get("actual_ownership_id"),
                                            errors="coerce",
                                        )

                                if "Name" in market_day.columns and "Name" in own.columns:
                                    market_day["name_key"] = market_day["Name"].map(_norm_name_key)
                                    market_day["name_key_loose"] = market_day["Name"].map(_norm_name_key_loose)
                                    own["name_key"] = own["Name"].map(_norm_name_key)
                                    own["name_key_loose"] = own["Name"].map(_norm_name_key_loose)

                                    own_by_name = (
                                        own.loc[(own["name_key"] != "") & own["actual_ownership"].notna(), ["name_key", "actual_ownership"]]
                                        .drop_duplicates("name_key")
                                        .rename(columns={"actual_ownership": "actual_ownership_name"})
                                    )
                                    own_by_name_loose = (
                                        own.loc[(own["name_key_loose"] != "") & own["actual_ownership"].notna(), ["name_key_loose", "actual_ownership"]]
                                        .drop_duplicates("name_key_loose")
                                        .rename(columns={"actual_ownership": "actual_ownership_name_loose"})
                                    )
                                    if not own_by_name.empty:
                                        market_day = market_day.merge(own_by_name, on="name_key", how="left")
                                    if not own_by_name_loose.empty:
                                        market_day = market_day.merge(own_by_name_loose, on="name_key_loose", how="left")

                                    own_base = pd.to_numeric(market_day.get("actual_ownership_from_file"), errors="coerce")
                                    own_fill_name = pd.to_numeric(market_day.get("actual_ownership_name"), errors="coerce")
                                    own_fill_loose = pd.to_numeric(market_day.get("actual_ownership_name_loose"), errors="coerce")
                                    own_base = own_base.where(own_base.notna(), own_fill_name)
                                    own_base = own_base.where(own_base.notna(), own_fill_loose)
                                    market_day["actual_ownership_from_file"] = own_base
                                    market_day = market_day.drop(
                                        columns=[
                                            "name_key",
                                            "name_key_loose",
                                            "actual_ownership_id",
                                            "actual_ownership_name",
                                            "actual_ownership_name_loose",
                                        ],
                                        errors="ignore",
                                    )

                            if "blend_error" not in market_day.columns:
                                market_day["blend_error"] = pd.NA
                            if (
                                "actual_dk_points" in market_day.columns
                                and "blended_projection" in market_day.columns
                            ):
                                blend_vals = pd.to_numeric(market_day.get("blend_error"), errors="coerce")
                                computed_blend = pd.to_numeric(market_day["actual_dk_points"], errors="coerce") - pd.to_numeric(
                                    market_day["blended_projection"], errors="coerce"
                                )
                                market_day["blend_error"] = blend_vals.where(blend_vals.notna(), computed_blend)

                            market_day["projected_ownership"] = pd.to_numeric(
                                market_day.get("projected_ownership"),
                                errors="coerce",
                            )
                            market_day["actual_ownership_from_file"] = pd.to_numeric(
                                market_day.get("actual_ownership_from_file"),
                                errors="coerce",
                            )
                            market_day["ownership_error"] = (
                                market_day["actual_ownership_from_file"] - market_day["projected_ownership"]
                            )
                            market_day["review_date"] = review_day.isoformat()

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
                                "our_points_proj",
                                "blend_points_proj",
                                "vegas_points_line",
                                "vegas_rebounds_line",
                                "vegas_assists_line",
                                "vegas_threes_line",
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
                            market_frames.append(market_day[[c for c in keep_cols if c in market_day.columns]].copy())
                            used_dates += 1

                        combined_market_df = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame()
                        market_packet = build_market_correlation_ai_review_packet(
                            review_rows_df=combined_market_df,
                            focus_limit=market_focus_limit,
                            min_bucket_samples=market_min_bucket_samples,
                        )
                        market_prompt = build_market_correlation_ai_review_user_prompt(market_packet)
                        st.session_state["cbb_market_correlation_packet"] = market_packet
                        st.session_state["cbb_market_correlation_prompt_user"] = market_prompt
                        st.session_state["cbb_market_correlation_meta"] = {
                            "scanned_dates": int(scanned_dates),
                            "used_dates": int(used_dates),
                            "start_date": market_start_date.isoformat(),
                            "end_date": market_end_date.isoformat(),
                            "rows": int(len(combined_market_df)),
                            "used_saved_dates_only": bool(market_use_saved_run_dates),
                        }
                        st.session_state.pop("cbb_market_correlation_output", None)
                    except Exception as exc:
                        st.exception(exc)

            market_packet_state = st.session_state.get("cbb_market_correlation_packet")
            market_prompt_user = str(st.session_state.get("cbb_market_correlation_prompt_user") or "").strip()
            market_meta = st.session_state.get("cbb_market_correlation_meta") or {}
            if isinstance(market_packet_state, dict) and market_packet_state:
                ws = market_packet_state.get("window_summary") or {}
                gq = market_packet_state.get("global_quality") or {}
                mc_m1, mc_m2, mc_m3, mc_m4, mc_m5 = st.columns(5)
                mc_m1.metric("Dates Used", int(_safe_int_value(ws.get("dates_used"), default=0)))
                mc_m2.metric("Rows", int(_safe_int_value(ws.get("rows"), default=0)))
                mc_m3.metric(
                    "Blend MAE",
                    f"{_safe_float_value(gq.get('blend_mae'), default=0.0):.2f}",
                )
                mc_m4.metric(
                    "Ownership MAE",
                    f"{_safe_float_value(gq.get('ownership_mae'), default=0.0):.2f}",
                )
                mc_m5.metric(
                    "Total->Points Corr",
                    f"{_safe_float_value(gq.get('total_line_vs_actual_points_spearman'), default=0.0):.3f}",
                )
                st.caption(
                    "Market packet build summary: "
                    f"scanned_dates={int(_safe_int_value(market_meta.get('scanned_dates'), default=0))}, "
                    f"used_dates={int(_safe_int_value(market_meta.get('used_dates'), default=0))}, "
                    f"range={str(market_meta.get('start_date') or '')}..{str(market_meta.get('end_date') or '')}, "
                    f"rows={int(_safe_int_value(market_meta.get('rows'), default=0))}"
                )

                corr_df = pd.DataFrame(market_packet_state.get("correlation_table") or [])
                if not corr_df.empty:
                    st.caption("Correlation Table")
                    st.dataframe(corr_df, hide_index=True, use_container_width=True)

                bucket_tables = market_packet_state.get("bucket_calibration") or {}
                total_bucket_df = pd.DataFrame(bucket_tables.get("total_line_buckets") or [])
                spread_bucket_df = pd.DataFrame(bucket_tables.get("abs_spread_buckets") or [])
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.caption("Total-Line Buckets")
                    if total_bucket_df.empty:
                        st.info("No total-line bucket summary available.")
                    else:
                        st.dataframe(total_bucket_df, hide_index=True, use_container_width=True)
                with bc2:
                    st.caption("Abs-Spread Buckets")
                    if spread_bucket_df.empty:
                        st.info("No spread bucket summary available.")
                    else:
                        st.dataframe(spread_bucket_df, hide_index=True, use_container_width=True)

                rec_df = pd.DataFrame(market_packet_state.get("calibration_recommendations") or [])
                if not rec_df.empty:
                    st.caption("Calibration Recommendations")
                    st.dataframe(rec_df, hide_index=True, use_container_width=True)

                trend_df = pd.DataFrame(market_packet_state.get("trend_by_date") or [])
                if not trend_df.empty:
                    st.caption("Trend By Date")
                    st.dataframe(trend_df, hide_index=True, use_container_width=True)

                market_packet_json = json.dumps(market_packet_state, indent=2, ensure_ascii=True)
                mdc1, mdc2 = st.columns(2)
                mdc1.download_button(
                    "Download Market Packet JSON",
                    data=market_packet_json,
                    file_name=f"market_corr_packet_{market_end_date.isoformat()}_{ar_contest_id}.json",
                    mime="application/json",
                    key="download_market_corr_packet_json",
                )
                mdc2.download_button(
                    "Download Market Prompt TXT",
                    data=market_prompt_user,
                    file_name=f"market_corr_prompt_{market_end_date.isoformat()}_{ar_contest_id}.txt",
                    mime="text/plain",
                    key="download_market_corr_prompt_txt",
                )
                with st.expander("Market Packet Preview"):
                    st.json(market_packet_state)
                with st.expander("Market Prompt Preview"):
                    st.text_area(
                        "Market User Prompt",
                        value=market_prompt_user,
                        height=260,
                        key="market_corr_prompt_preview_text",
                    )

                ma1, ma2 = st.columns(2)
                market_model = ma1.text_input(
                    "Market OpenAI Model",
                    value=str(st.session_state.get("ai_review_model", "gpt-5-mini")),
                    key="market_corr_model",
                ).strip()
                market_max_tokens = int(
                    ma2.number_input(
                        "Market Max Output Tokens",
                        min_value=200,
                        max_value=8000,
                        value=1800,
                        step=100,
                        key="market_corr_max_output_tokens",
                    )
                )
                run_market_openai = st.button("Run Market Correlation OpenAI Review", key="run_market_corr_openai")
                if run_market_openai:
                    if not openai_key:
                        st.error("Set `OPENAI_API_KEY` or Streamlit secret `openai_api_key` first.")
                    else:
                        with st.spinner("Generating market-correlation AI recommendations..."):
                            try:
                                market_ai_text = request_openai_review(
                                    api_key=openai_key,
                                    user_prompt=market_prompt_user,
                                    system_prompt=MARKET_CORRELATION_AI_REVIEW_SYSTEM_PROMPT,
                                    model=(market_model or "gpt-5-mini"),
                                    max_output_tokens=market_max_tokens,
                                )
                                st.session_state["cbb_market_correlation_output"] = market_ai_text
                            except Exception as exc:
                                st.exception(exc)
                market_output = str(st.session_state.get("cbb_market_correlation_output") or "").strip()
                if market_output:
                    st.subheader("Market Correlation AI Recommendations")
                    st.text_area(
                        "Market Model Output",
                        value=market_output,
                        height=340,
                        key="market_corr_output_preview",
                    )
                    st.download_button(
                        "Download Market AI Recommendations TXT",
                        data=market_output,
                        file_name=f"market_corr_recommendations_{market_end_date.isoformat()}_{ar_contest_id}.txt",
                        mime="text/plain",
                        key="download_market_corr_recommendations_txt",
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

with tab_slate_vegas:
    st.subheader("Slate-Scoped Game Agent")
    st.caption(
        "AI scout for game-level DFS context. Anchors analysis to the DK slate/active pool first, "
        "then uses odds, prior-day box score form, and season Vegas calibration for stack angles."
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
                            slate_key=shared_slate_key,
                            bookmaker=pool_bookmaker,
                            gcp_project=gcp_project or None,
                            service_account_json=cred_json,
                            service_account_json_b64=cred_json_b64,
                        )
                    except Exception:
                        pool_df_agent = pd.DataFrame()
                        raw_slate_df_agent = pd.DataFrame()

                    scoped_odds_df, slate_scope_meta = _filter_odds_to_slate_context(
                        odds_df=odds_agent_df,
                        pool_df=pool_df_agent,
                        raw_slate_df=raw_slate_df_agent,
                    )
                    scope_reason = str(slate_scope_meta.get("reason") or "")
                    if scope_reason == "filtered_by_event_id":
                        st.caption(
                            "Slate scoping applied: "
                            f"odds_rows {int(slate_scope_meta.get('odds_rows_before') or 0)} -> "
                            f"{int(slate_scope_meta.get('odds_rows_after') or 0)}, "
                            f"slate_games={int(slate_scope_meta.get('requested_slate_games') or 0)}, "
                            f"mapped={int(slate_scope_meta.get('mapped_slate_games') or 0)}."
                        )
                    elif scope_reason in {"fallback_unfiltered", "no_slate_game_keys"}:
                        st.caption(
                            "Slate scoping fallback: using full game-odds set "
                            f"(reason={scope_reason}, "
                            f"slate_games={int(slate_scope_meta.get('requested_slate_games') or 0)}, "
                            f"mapped={int(slate_scope_meta.get('mapped_slate_games') or 0)})."
                        )

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
                        "odds_df": scoped_odds_df,
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
                        "odds_rows_before_scope": int(slate_scope_meta.get("odds_rows_before") or len(odds_agent_df)),
                        "odds_rows_after_scope": int(slate_scope_meta.get("odds_rows_after") or len(scoped_odds_df)),
                        "slate_scope_reason": str(slate_scope_meta.get("reason") or ""),
                        "requested_slate_games": int(slate_scope_meta.get("requested_slate_games") or 0),
                        "mapped_slate_games": int(slate_scope_meta.get("mapped_slate_games") or 0),
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
                        f"Packet is for `{packet_review_date}` while selected slate date is `{game_selected_date.isoformat()}`. "
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
                    f"dk_slate_rows={int(_safe_int_value(game_meta.get('dk_slate_rows'), default=0))}, "
                    f"odds_scope={str(game_meta.get('slate_scope_reason') or 'n/a')}, "
                    f"odds_rows={int(_safe_int_value(game_meta.get('odds_rows_after_scope'), default=0))}/"
                    f"{int(_safe_int_value(game_meta.get('odds_rows_before_scope'), default=0))}, "
                    f"mapped_slate_games={int(_safe_int_value(game_meta.get('mapped_slate_games'), default=0))}/"
                    f"{int(_safe_int_value(game_meta.get('requested_slate_games'), default=0))}"
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
