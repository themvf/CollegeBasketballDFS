from __future__ import annotations

import json
import os
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import pandas as pd

from .cbb_api_service import (
    ensure_full_registry_coverage,
    load_injuries_frame,
    load_lineupstarter_projection_frame,
    load_odds_frame_for_date,
    load_props_frame_for_date,
    load_season_player_history_frame,
    load_season_vegas_history_frame,
    resolve_rotowire_slate,
)
from .cbb_dk_optimizer import (
    build_dk_upload_csv,
    build_player_pool,
    generate_lineups,
    lineups_slots_frame,
    lineups_summary_frame,
    remove_injured_players,
)
from .cbb_tail_model import fit_total_tail_model, score_odds_games_for_tail
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JOBS_ROOT = REPO_ROOT / "data" / "api_jobs"


LINEUP_MODEL_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "label": "Standard v1 (Balanced Core)",
        "version_key": "standard_v1",
        "lineup_strategy": "standard",
        "include_tail_signals": False,
        "model_profile": "legacy_baseline",
        "gpp_overrides": {
            "salary_left_target": 125,
            "low_own_bucket_exposure_pct": 22.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 13.0,
            "low_own_bucket_min_projection": 18.0,
            "low_own_bucket_min_tail_score": 56.0,
            "low_own_bucket_objective_bonus": 0.8,
            "preferred_game_bonus": 0.9,
            "preferred_game_stack_lineup_pct": 60.0,
            "preferred_game_stack_min_players": 3,
            "max_unsupported_false_chalk_per_lineup": 0,
            "ceiling_boost_lineup_pct": 32.0,
            "ceiling_boost_stack_bonus": 2.4,
            "ceiling_boost_salary_left_target": 100,
        },
    },
    {
        "label": "Spike v2 (High-Variance Tail)",
        "version_key": "spike_v2_tail",
        "lineup_strategy": "spike",
        "include_tail_signals": True,
        "model_profile": "tail_spike_pairs",
        "gpp_overrides": {
            "salary_left_target": 220,
            "low_own_bucket_exposure_pct": 48.0,
            "low_own_bucket_min_per_lineup": 2,
            "low_own_bucket_max_projected_ownership": 11.0,
            "low_own_bucket_min_projection": 16.0,
            "low_own_bucket_min_tail_score": 62.0,
            "low_own_bucket_objective_bonus": 1.6,
            "preferred_game_bonus": 1.3,
            "preferred_game_stack_lineup_pct": 70.0,
            "preferred_game_stack_min_players": 3,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 70.0,
            "ceiling_boost_stack_bonus": 3.5,
            "ceiling_boost_salary_left_target": 150,
        },
    },
    {
        "label": "Standout v1 (Leverage Surge)",
        "version_key": "standout_v1_capture",
        "lineup_strategy": "standard",
        "include_tail_signals": True,
        "model_profile": "standout_capture_v1",
        "gpp_overrides": {
            "salary_left_target": 150,
            "low_own_bucket_exposure_pct": 36.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 12.0,
            "low_own_bucket_min_projection": 18.0,
            "low_own_bucket_min_tail_score": 58.0,
            "low_own_bucket_objective_bonus": 1.3,
            "preferred_game_bonus": 1.1,
            "preferred_game_stack_lineup_pct": 65.0,
            "preferred_game_stack_min_players": 3,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 52.0,
            "ceiling_boost_stack_bonus": 2.9,
            "ceiling_boost_salary_left_target": 130,
        },
    },
    {
        "label": "Chalk-Value v1 (Leverage Pivots)",
        "version_key": "chalk_value_capture_v1",
        "lineup_strategy": "standard",
        "include_tail_signals": True,
        "model_profile": "chalk_value_capture_v1",
        "gpp_overrides": {
            "salary_left_target": 125,
            "low_own_bucket_exposure_pct": 20.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 13.0,
            "low_own_bucket_min_projection": 18.0,
            "low_own_bucket_min_tail_score": 55.0,
            "low_own_bucket_objective_bonus": 0.9,
            "preferred_game_bonus": 1.0,
            "preferred_game_stack_lineup_pct": 70.0,
            "preferred_game_stack_min_players": 3,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 38.0,
            "ceiling_boost_stack_bonus": 2.5,
            "ceiling_boost_salary_left_target": 110,
        },
    },
    {
        "label": "Salary-Efficiency v1 (Ceiling)",
        "version_key": "salary_efficiency_ceiling_v1",
        "lineup_strategy": "standard",
        "include_tail_signals": True,
        "model_profile": "salary_efficiency_ceiling_v1",
        "gpp_overrides": {
            "salary_left_target": 100,
            "low_own_bucket_exposure_pct": 32.0,
            "low_own_bucket_min_per_lineup": 1,
            "low_own_bucket_max_projected_ownership": 12.0,
            "low_own_bucket_min_projection": 20.0,
            "low_own_bucket_min_tail_score": 60.0,
            "low_own_bucket_objective_bonus": 1.2,
            "preferred_game_bonus": 1.2,
            "preferred_game_stack_lineup_pct": 80.0,
            "preferred_game_stack_min_players": 3,
            "max_unsupported_false_chalk_per_lineup": 1,
            "ceiling_boost_lineup_pct": 66.0,
            "ceiling_boost_stack_bonus": 3.1,
            "ceiling_boost_salary_left_target": 125,
        },
    },
)
LINEUP_MODEL_BY_KEY = {str(cfg["version_key"]): cfg for cfg in LINEUP_MODEL_REGISTRY}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _lineup_model_config(version_key: str) -> dict[str, Any]:
    cfg = LINEUP_MODEL_BY_KEY.get(str(version_key or "").strip(), LINEUP_MODEL_BY_KEY.get("salary_efficiency_ceiling_v1", {}))
    return {
        "version_key": str(cfg.get("version_key") or "salary_efficiency_ceiling_v1"),
        "version_label": str(cfg.get("label") or "Salary-Efficiency v1 (Ceiling)"),
        "lineup_strategy": str(cfg.get("lineup_strategy") or "standard"),
        "include_tail_signals": bool(cfg.get("include_tail_signals", True)),
        "model_profile": str(cfg.get("model_profile") or "salary_efficiency_ceiling_v1"),
        "gpp_overrides": dict(cfg.get("gpp_overrides") or {}),
    }


def _contest_is_gpp(contest_type: str) -> bool:
    return "gpp" in str(contest_type or "").strip().lower()


def _runtime_controls(contest_type: str, model_cfg: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    controls = {
        "salary_left_target": int(request.get("salary_left_target", 180)),
        "low_own_bucket_exposure_pct": float(request.get("low_own_bucket_exposure_pct", 22.0)),
        "low_own_bucket_min_per_lineup": int(request.get("low_own_bucket_min_per_lineup", 1)),
        "low_own_bucket_max_projected_ownership": float(request.get("low_own_bucket_max_projected_ownership", 12.0)),
        "low_own_bucket_min_projection": float(request.get("low_own_bucket_min_projection", 18.0)),
        "low_own_bucket_min_tail_score": float(request.get("low_own_bucket_min_tail_score", 55.0)),
        "low_own_bucket_objective_bonus": float(request.get("low_own_bucket_objective_bonus", 0.9)),
        "preferred_game_bonus": float(request.get("preferred_game_bonus", 1.0)),
        "preferred_game_stack_lineup_pct": float(request.get("preferred_game_stack_lineup_pct", 60.0)),
        "preferred_game_stack_min_players": int(request.get("preferred_game_stack_min_players", 3)),
        "max_unsupported_false_chalk_per_lineup": int(request.get("max_unsupported_false_chalk_per_lineup", 1)),
        "ceiling_boost_lineup_pct": float(request.get("ceiling_boost_lineup_pct", 50.0)),
        "ceiling_boost_stack_bonus": float(request.get("ceiling_boost_stack_bonus", 2.5)),
        "ceiling_boost_salary_left_target": int(request.get("ceiling_boost_salary_left_target", 120)),
    }
    if _contest_is_gpp(contest_type):
        controls.update({k: v for k, v in (model_cfg.get("gpp_overrides") or {}).items()})
    for key in list(controls.keys()):
        if key in request and request.get(key) is not None:
            controls[key] = request.get(key)
    return controls


def _resolve_bookmaker_filter(request: dict[str, Any]) -> str | None:
    for key in ["bookmaker", "bookmaker_filter"]:
        raw = str(request.get(key) or "").strip()
        if raw:
            first = next((part.strip().lower() for part in raw.split(",") if part.strip()), "")
            if first:
                return first
    env_raw = str(os.getenv("CBB_ODDS_BOOKMAKERS") or "").strip()
    if env_raw:
        first = next((part.strip().lower() for part in env_raw.split(",") if part.strip()), "")
        if first:
            return first
    return "fanduel"


def _annotate_lineups_with_version_metadata(
    lineups: list[dict[str, Any]],
    version_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for lineup in lineups:
        one = dict(lineup)
        one["lineup_model_key"] = str(version_cfg.get("version_key") or "")
        one["lineup_model_label"] = str(version_cfg.get("version_label") or "")
        one["lineup_strategy"] = str(version_cfg.get("lineup_strategy") or "")
        one["model_profile"] = str(version_cfg.get("model_profile") or "")
        one["include_tail_signals"] = bool(version_cfg.get("include_tail_signals"))
        out.append(one)
    return out


@dataclass
class LineupArtifact:
    name: str
    path: Path
    content_type: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "filename": self.path.name,
            "content_type": self.content_type,
            "size_bytes": int(self.size_bytes),
        }


class LineupJobManager:
    def __init__(self, jobs_root: str | Path | None = None, max_workers: int = 2) -> None:
        self.jobs_root = Path(jobs_root) if jobs_root else DEFAULT_JOBS_ROOT
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._lock = Lock()

    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_root / str(job_id)

    def _state_path(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "state.json"

    def _artifacts_dir(self, job_id: str) -> Path:
        return self._job_dir(job_id) / "artifacts"

    def _write_state(self, job_id: str, state: dict[str, Any]) -> None:
        job_dir = self._job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        self._state_path(job_id).write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")

    def _read_state(self, job_id: str) -> dict[str, Any] | None:
        state_path = self._state_path(job_id)
        if not state_path.exists():
            return None
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _update_state(self, job_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            current = self._read_state(job_id) or {}
            current.update(updates)
            current["updated_at_utc"] = utc_now_iso()
            self._write_state(job_id, current)
            return current

    def _write_artifact(self, job_id: str, name: str, content: bytes, content_type: str) -> LineupArtifact:
        safe_name = str(name).strip().replace("\\", "_").replace("/", "_")
        if not safe_name:
            raise ValueError("artifact name cannot be empty")
        artifact_dir = self._artifacts_dir(job_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / safe_name
        path.write_bytes(content)
        return LineupArtifact(name=safe_name, path=path, content_type=str(content_type), size_bytes=path.stat().st_size)

    def submit(self, request: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        initial_state = {
            "job_id": job_id,
            "job_type": "lineup_generation_v1",
            "status": "queued",
            "progress_pct": 0,
            "message": "Queued",
            "created_at_utc": utc_now_iso(),
            "updated_at_utc": utc_now_iso(),
            "request": request,
            "artifacts": [],
            "result": {},
        }
        self._write_state(job_id, initial_state)
        self._executor.submit(self._run_job, job_id, request)
        return job_id

    def _run_job(self, job_id: str, request: dict[str, Any]) -> None:
        artifacts: list[LineupArtifact] = []

        def progress(pct: int, message: str) -> None:
            clamped = max(0, min(100, int(pct)))
            self._update_state(job_id, progress_pct=clamped, message=str(message), status="running")

        def write_artifact(name: str, content: bytes, content_type: str) -> None:
            artifact = self._write_artifact(job_id, name=name, content=content, content_type=content_type)
            artifacts.append(artifact)
            self._update_state(
                job_id,
                artifacts=[a.as_dict() for a in artifacts],
            )

        try:
            self._update_state(job_id, status="running", started_at_utc=utc_now_iso(), progress_pct=1, message="Starting")
            result = run_lineup_job_request(request=request, progress=progress, write_artifact=write_artifact)
            self._update_state(
                job_id,
                status="succeeded",
                completed_at_utc=utc_now_iso(),
                progress_pct=100,
                message="Complete",
                artifacts=[a.as_dict() for a in artifacts],
                result=result,
            )
        except Exception as exc:
            self._update_state(
                job_id,
                status="failed",
                completed_at_utc=utc_now_iso(),
                progress_pct=max(1, int((self._read_state(job_id) or {}).get("progress_pct") or 1)),
                message=f"Failed: {exc}",
                error=str(exc),
                traceback=traceback.format_exc(limit=25),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        return self._read_state(job_id)

    def list_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        state = self._read_state(job_id) or {}
        artifacts = state.get("artifacts")
        if isinstance(artifacts, list):
            return [dict(x) for x in artifacts if isinstance(x, dict)]
        return []

    def resolve_artifact_path(self, job_id: str, artifact_name: str) -> Path | None:
        safe_name = str(artifact_name or "").strip().replace("\\", "_").replace("/", "_")
        if not safe_name:
            return None
        path = self._artifacts_dir(job_id) / safe_name
        if not path.exists() or not path.is_file():
            return None
        return path


def run_lineup_job_request(
    *,
    request: dict[str, Any],
    progress: Callable[[int, str], None],
    write_artifact: Callable[[str, bytes, str], None],
) -> dict[str, Any]:
    selected_date = str(request.get("selected_date") or "").strip()
    if not selected_date:
        raise ValueError("selected_date is required")
    contest_type = str(request.get("contest_type") or "Large GPP")
    slate_key = str(request.get("slate_key") or "main").strip().lower()
    rotowire_contest_type = str(request.get("rotowire_contest_type") or "Classic")
    rotowire_slate_name = str(request.get("rotowire_slate_name") or "All")
    model_key = str(request.get("model_key") or "salary_efficiency_ceiling_v1")
    num_lineups = max(1, int(request.get("num_lineups") or 150))
    random_seed = int(request.get("random_seed") or 7)
    max_salary_left = int(request.get("max_salary_left") or 400)
    global_max_exposure_pct = float(request.get("global_max_exposure_pct") or 50.0)
    auto_preferred_game_count = int(request.get("auto_preferred_game_count") or 2)
    apply_focus_game_stack_guardrails = bool(request.get("apply_focus_game_stack_guardrails", True))
    apply_ownership_guardrails = bool(request.get("apply_ownership_guardrails", True))
    apply_uncertainty_shrink = bool(request.get("apply_uncertainty_shrink", True))
    uncertainty_weight = float(request.get("uncertainty_weight") or 0.15)
    high_risk_extra_shrink = float(request.get("high_risk_extra_shrink") or 0.08)
    dnp_risk_threshold = float(request.get("dnp_risk_threshold") or 0.35)
    rotowire_cookie = request.get("rotowire_cookie")
    bookmaker_filter = _resolve_bookmaker_filter(request)
    bucket_name = (str(request.get("bucket_name") or "").strip() or None)
    gcp_project = (str(request.get("gcp_project") or "").strip() or None)
    service_account_json = (str(request.get("service_account_json") or "").strip() or None)
    service_account_json_b64 = (str(request.get("service_account_json_b64") or "").strip() or None)

    progress(5, "Loading RotoWire slate")
    resolved_bundle = resolve_rotowire_slate(
        selected_date=selected_date,
        slate_key=slate_key,
        contest_type=rotowire_contest_type,
        slate_name=rotowire_slate_name,
        slate_id=request.get("rotowire_slate_id"),
        site_id=int(request.get("site_id") or 1),
        cookie_header=str(rotowire_cookie) if rotowire_cookie else None,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    rotowire_df = resolved_bundle.get("rotowire_df")
    slate_meta = dict(resolved_bundle.get("slate") or {})
    if rotowire_df.empty:
        raise RuntimeError("No RotoWire players returned for selected slate.")

    progress(14, "Resolving DK IDs from registry")
    resolved_slate = resolved_bundle.get("resolved_slate_df")
    resolution_df = resolved_bundle.get("resolution_df")
    coverage = dict(resolved_bundle.get("coverage") or {})
    ensure_full_registry_coverage(coverage, resolution_df)

    progress(18, "Loading saved LineupStarter priors")
    try:
        lineupstarter_df = load_lineupstarter_projection_frame(
            selected_date=selected_date,
            slate_key=slate_key,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            service_account_json=service_account_json,
            service_account_json_b64=service_account_json_b64,
        )
    except ValueError:
        lineupstarter_df = pd.DataFrame()

    progress(20, "Loading injury, props, and season context")
    injuries_df = load_injuries_frame(
        selected_date=selected_date,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    filtered_slate, removed_injured_df = remove_injured_players(resolved_slate, injuries_df)
    season_history_df = load_season_player_history_frame(
        selected_date=selected_date,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    odds_df = load_odds_frame_for_date(
        selected_date=selected_date,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    vegas_history_df = load_season_vegas_history_frame(
        selected_date=selected_date,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )
    tail_model = fit_total_tail_model(vegas_history_df)
    odds_scored_df = score_odds_games_for_tail(odds_df, tail_model) if not odds_df.empty else pd.DataFrame()
    props_df = load_props_frame_for_date(
        selected_date=selected_date,
        bucket_name=bucket_name,
        gcp_project=gcp_project,
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
    )

    progress(22, "Building player pool")
    pool_df = build_player_pool(
        slate_df=filtered_slate,
        props_df=props_df,
        season_stats_df=season_history_df,
        ownership_history_df=None,
        rotowire_df=rotowire_df,
        bookmaker_filter=bookmaker_filter,
        odds_games_df=odds_scored_df,
        recent_form_games=int(request.get("recent_form_games") or 7),
        recent_points_weight=float(request.get("recent_points_weight") or 0.0),
        lineupstarter_df=lineupstarter_df,
    )
    if pool_df.empty:
        raise RuntimeError("Player pool is empty after build_player_pool.")
    pool_sorted = pool_df.sort_values(["projected_dk_points", "Salary"], ascending=[False, False], kind="stable")

    model_cfg = _lineup_model_config(model_key)
    controls = _runtime_controls(contest_type=contest_type, model_cfg=model_cfg, request=request)
    selected_preferred_game_keys = [
        str(x).strip().upper() for x in (request.get("preferred_game_keys") or []) if str(x).strip()
    ]

    progress(30, f"Generating lineups with {model_cfg['version_label']}")

    def _lineup_progress(done: int, total: int, status: str) -> None:
        denom = max(1, int(total))
        ratio = max(0.0, min(1.0, float(done) / float(denom)))
        pct = 30 + int(round(60.0 * ratio))
        progress(pct, status)

    lineups, warnings = generate_lineups(
        pool_df=pool_sorted,
        num_lineups=num_lineups,
        contest_type=contest_type,
        locked_ids=list(request.get("locked_ids") or []),
        excluded_ids=list(request.get("excluded_ids") or []),
        exposure_caps_pct=dict(request.get("exposure_caps_pct") or {}),
        global_max_exposure_pct=global_max_exposure_pct,
        max_salary_left=max_salary_left,
        lineup_strategy=str(model_cfg["lineup_strategy"]),
        include_tail_signals=bool(model_cfg.get("include_tail_signals", False)),
        projection_scale=float(request.get("projection_scale") or 1.0),
        apply_ownership_guardrails=apply_ownership_guardrails,
        ownership_guardrail_projected_threshold=float(request.get("ownership_guardrail_projected_threshold") or 8.0),
        ownership_guardrail_surge_threshold=float(request.get("ownership_guardrail_surge_threshold") or 78.0),
        ownership_guardrail_projection_rank_threshold=0.60,
        ownership_guardrail_floor_base=float(request.get("ownership_guardrail_floor_base") or 8.0),
        ownership_guardrail_floor_cap=float(request.get("ownership_guardrail_floor_cap") or 22.0),
        apply_uncertainty_shrink=apply_uncertainty_shrink,
        uncertainty_weight=uncertainty_weight,
        high_risk_extra_shrink=high_risk_extra_shrink,
        dnp_risk_threshold=dnp_risk_threshold,
        low_own_bucket_exposure_pct=float(controls["low_own_bucket_exposure_pct"]),
        low_own_bucket_min_per_lineup=int(controls["low_own_bucket_min_per_lineup"]),
        low_own_bucket_max_projected_ownership=float(controls["low_own_bucket_max_projected_ownership"]),
        low_own_bucket_min_projection=float(controls["low_own_bucket_min_projection"]),
        low_own_bucket_min_tail_score=float(controls["low_own_bucket_min_tail_score"]),
        low_own_bucket_objective_bonus=float(controls["low_own_bucket_objective_bonus"]),
        preferred_game_keys=selected_preferred_game_keys,
        preferred_game_bonus=float(controls["preferred_game_bonus"]),
        preferred_game_stack_lineup_pct=float(controls["preferred_game_stack_lineup_pct"]) if apply_focus_game_stack_guardrails else 0.0,
        preferred_game_stack_min_players=int(controls["preferred_game_stack_min_players"]) if apply_focus_game_stack_guardrails else 0,
        auto_preferred_game_count=max(0, auto_preferred_game_count),
        max_unsupported_false_chalk_per_lineup=int(controls["max_unsupported_false_chalk_per_lineup"]),
        ceiling_boost_lineup_pct=float(controls["ceiling_boost_lineup_pct"]),
        ceiling_boost_stack_bonus=float(controls["ceiling_boost_stack_bonus"]),
        ceiling_boost_salary_left_target=int(controls["ceiling_boost_salary_left_target"]),
        salary_left_target=int(controls["salary_left_target"]),
        random_seed=random_seed,
        model_profile=str(model_cfg.get("model_profile") or "legacy_baseline"),
        progress_callback=_lineup_progress,
    )
    annotated_lineups = _annotate_lineups_with_version_metadata(lineups, model_cfg)

    progress(92, "Writing artifacts")
    summary_df = lineups_summary_frame(annotated_lineups)
    slots_df = lineups_slots_frame(annotated_lineups)
    upload_csv = build_dk_upload_csv(annotated_lineups)

    write_artifact("summary.csv", summary_df.to_csv(index=False).encode("utf-8"), "text/csv")
    write_artifact("slots.csv", slots_df.to_csv(index=False).encode("utf-8"), "text/csv")
    write_artifact("dk_upload.csv", upload_csv.encode("utf-8"), "text/csv")
    write_artifact("warnings.json", json.dumps({"warnings": warnings}, indent=2).encode("utf-8"), "application/json")
    write_artifact(
        "run_meta.json",
        json.dumps(
            {
                "generated_at_utc": utc_now_iso(),
                "request": request,
                "slate": slate_meta,
                "coverage": coverage,
                "lineupstarter_loaded": bool(not lineupstarter_df.empty),
                "lineupstarter_players": int(lineupstarter_df["ID"].nunique()) if not lineupstarter_df.empty else 0,
                "bookmaker_filter": bookmaker_filter,
                "injury_rows": int(len(injuries_df)),
                "removed_injured_rows": int(len(removed_injured_df)),
                "season_history_rows": int(len(season_history_df)),
                "props_rows": int(len(props_df)),
                "odds_rows": int(len(odds_df)),
                "odds_tail_rows": int(len(odds_scored_df)),
                "pool_rows": int(len(pool_df)),
                "model": model_cfg,
                "lineups_generated": len(annotated_lineups),
                "warnings_count": len(warnings),
            },
            indent=2,
            default=str,
        ).encode("utf-8"),
        "application/json",
    )
    progress(100, "Done")

    return {
        "lineups_generated": int(len(annotated_lineups)),
        "warnings": [str(w) for w in warnings],
        "slate": slate_meta,
        "coverage": coverage,
        "lineupstarter_loaded": bool(not lineupstarter_df.empty),
        "bookmaker_filter": bookmaker_filter,
        "injury_rows": int(len(injuries_df)),
        "removed_injured_rows": int(len(removed_injured_df)),
        "season_history_rows": int(len(season_history_df)),
        "props_rows": int(len(props_df)),
        "odds_rows": int(len(odds_df)),
        "odds_tail_rows": int(len(odds_scored_df)),
        "model": model_cfg,
        "runtime_controls": controls,
    }
