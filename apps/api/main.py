from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from college_basketball_dfs.cbb_api_service import (
    build_lineup_run_detail_payload,
    build_lineup_runs_payload,
    build_cache_coverage_payload,
    build_injuries_review_payload,
    build_projection_review_payload,
    build_props_review_payload,
    build_slate_status_payload,
    build_tournament_review_payload,
    build_registry_coverage,
    build_vegas_game_lines_payload,
    build_vegas_market_context_payload,
    import_contest_standings_csv,
    import_injuries_feed_csv,
    import_dk_slate_overrides,
    import_injuries_manual_csv,
    import_projection_ownership_csv,
    list_rotowire_slates_for_date,
)
from college_basketball_dfs.cbb_lineup_jobs import LineupJobManager


app = FastAPI(
    title="CollegeBasketballDFS API",
    version="0.1.0",
    description="Backend API for the Vercel migration from Streamlit.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_MANAGER = LineupJobManager()


class LineupGenerateRequest(BaseModel):
    selected_date: date
    contest_type: str = Field(default="Large GPP")
    slate_key: str = Field(default="main")
    rotowire_contest_type: str = Field(default="Classic")
    rotowire_slate_name: str = Field(default="All")
    rotowire_slate_id: int | None = None
    model_key: str = Field(default="salary_efficiency_ceiling_v1")
    num_lineups: int = Field(default=150, ge=1, le=1000)
    random_seed: int = Field(default=7, ge=0, le=999999)
    max_salary_left: int = Field(default=400, ge=0, le=10000)
    global_max_exposure_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    auto_preferred_game_count: int = Field(default=2, ge=0, le=8)
    apply_focus_game_stack_guardrails: bool = True
    apply_ownership_guardrails: bool = True
    apply_uncertainty_shrink: bool = True
    uncertainty_weight: float = Field(default=0.15, ge=0.0, le=0.5)
    high_risk_extra_shrink: float = Field(default=0.08, ge=0.0, le=0.5)
    dnp_risk_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    projection_scale: float = Field(default=1.0, gt=0.0, le=3.0)
    salary_left_target: int | None = Field(default=None, ge=0, le=1000)
    low_own_bucket_exposure_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    low_own_bucket_min_per_lineup: int | None = Field(default=None, ge=0, le=4)
    low_own_bucket_max_projected_ownership: float | None = Field(default=None, ge=0.0, le=100.0)
    low_own_bucket_min_projection: float | None = Field(default=None, ge=0.0, le=100.0)
    low_own_bucket_min_tail_score: float | None = Field(default=None, ge=0.0, le=100.0)
    low_own_bucket_objective_bonus: float | None = Field(default=None, ge=0.0, le=10.0)
    preferred_game_bonus: float | None = Field(default=None, ge=0.0, le=10.0)
    preferred_game_stack_lineup_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    preferred_game_stack_min_players: int | None = Field(default=None, ge=0, le=8)
    max_unsupported_false_chalk_per_lineup: int | None = Field(default=None, ge=0, le=8)
    ceiling_boost_lineup_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    ceiling_boost_stack_bonus: float | None = Field(default=None, ge=0.0, le=10.0)
    ceiling_boost_salary_left_target: int | None = Field(default=None, ge=0, le=1000)
    preferred_game_keys: list[str] = Field(default_factory=list)
    locked_ids: list[str] = Field(default_factory=list)
    excluded_ids: list[str] = Field(default_factory=list)
    exposure_caps_pct: dict[str, float] = Field(default_factory=dict)
    site_id: int = Field(default=1, ge=1, le=10)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.get("/v1/rotowire/slates")
def rotowire_slates(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    contest_type: str | None = Query(default=None),
    slate_name: str | None = Query(default=None),
    site_id: int = Query(default=1),
    x_rotowire_cookie: str | None = Header(default=None),
) -> dict[str, Any]:
    try:
        slates = list_rotowire_slates_for_date(
            selected_date=selected_date,
            contest_type=contest_type,
            slate_name=slate_name,
            site_id=site_id,
            cookie_header=x_rotowire_cookie,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "selected_date": selected_date.isoformat(),
        "contest_type": contest_type,
        "slate_name": slate_name,
        "site_id": site_id,
        "rows": int(len(slates)),
        "slates": slates.to_dict(orient="records"),
    }


@app.get("/v1/registry/coverage")
def registry_coverage(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    slate_key: str = Query(default="main"),
    contest_type: str = Query(default="Classic"),
    slate_name: str = Query(default="All"),
    slate_id: int | None = Query(default=None),
    site_id: int = Query(default=1),
    x_rotowire_cookie: str | None = Header(default=None),
) -> dict[str, Any]:
    try:
        result = build_registry_coverage(
            selected_date=selected_date,
            slate_key=slate_key,
            contest_type=contest_type,
            slate_name=slate_name,
            slate_id=slate_id,
            site_id=site_id,
            cookie_header=x_rotowire_cookie,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.get("/v1/slates/{selected_date}/{slate_key}/status")
def slate_status(
    selected_date: date,
    slate_key: str,
    contest_type: str = Query(default="Classic"),
    slate_name: str = Query(default="All"),
    slate_id: int | None = Query(default=None),
    site_id: int = Query(default=1),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
    unresolved_sample_limit: int = Query(default=25, ge=1, le=100),
    x_rotowire_cookie: str | None = Header(default=None),
) -> dict[str, Any]:
    try:
        return build_slate_status_payload(
            selected_date=selected_date,
            slate_key=slate_key,
            contest_type=contest_type,
            slate_name=slate_name,
            slate_id=slate_id,
            site_id=site_id,
            cookie_header=x_rotowire_cookie,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            unresolved_sample_limit=unresolved_sample_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/vegas/game-lines")
def vegas_game_lines(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    row_limit: int = Query(default=300, ge=1, le=2000),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_vegas_game_lines_payload(
            selected_date=selected_date,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            row_limit=row_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/vegas/market-context")
def vegas_market_context(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    row_limit: int = Query(default=200, ge=1, le=2000),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_vegas_market_context_payload(
            selected_date=selected_date,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            row_limit=row_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/vegas/prop-data")
def vegas_prop_data(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    row_limit: int = Query(default=400, ge=1, le=3000),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_props_review_payload(
            selected_date=selected_date,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            row_limit=row_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/ops/cache-coverage")
def ops_cache_coverage(
    start_date: date = Query(..., description="Start date in YYYY-MM-DD"),
    end_date: date = Query(..., description="End date in YYYY-MM-DD"),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_cache_coverage_payload(
            start_date=start_date,
            end_date=end_date,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/injuries/review")
def injuries_review(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    row_limit: int = Query(default=300, ge=1, le=2000),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_injuries_review_payload(
            selected_date=selected_date,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            row_limit=row_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/injuries/manual/upload")
async def injuries_manual_upload(
    file: UploadFile = File(..., description="Manual injuries CSV"),
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be a CSV")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="file payload was empty")
    try:
        return import_injuries_manual_csv(
            csv_bytes=payload,
            selected_date=selected_date,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/injuries/feed/upload")
async def injuries_feed_upload(
    file: UploadFile = File(..., description="Date-scoped injuries feed CSV"),
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be a CSV")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="file payload was empty")
    try:
        return import_injuries_feed_csv(
            csv_bytes=payload,
            selected_date=selected_date,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/reviews/projection")
def projection_review(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    slate_key: str = Query(default="main"),
    row_limit: int = Query(default=500, ge=1, le=3000),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_projection_review_payload(
            selected_date=selected_date,
            slate_key=slate_key,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            row_limit=row_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/reviews/projection/ownership/upload")
async def projection_ownership_upload(
    file: UploadFile = File(..., description="Ownership CSV"),
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    slate_key: str = Query(default="main"),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be a CSV")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="file payload was empty")
    try:
        return import_projection_ownership_csv(
            csv_bytes=payload,
            selected_date=selected_date,
            slate_key=slate_key,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/reviews/tournament")
def tournament_review(
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    contest_id: str = Query(..., description="Contest identifier used for standings blob naming"),
    slate_key: str = Query(default="main"),
    entries_limit: int = Query(default=200, ge=1, le=2000),
    exposure_limit: int = Query(default=250, ge=1, le=3000),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_tournament_review_payload(
            selected_date=selected_date,
            contest_id=contest_id,
            slate_key=slate_key,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
            entries_limit=entries_limit,
            exposure_limit=exposure_limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/reviews/tournament/standings/upload")
async def tournament_standings_upload(
    file: UploadFile = File(..., description="Contest standings CSV"),
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    contest_id: str = Query(..., description="Contest identifier used for standings blob naming"),
    slate_key: str = Query(default="main"),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be a CSV")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="file payload was empty")
    try:
        return import_contest_standings_csv(
            csv_bytes=payload,
            selected_date=selected_date,
            contest_id=contest_id,
            slate_key=slate_key,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/lineup-runs")
def lineup_runs(
    selected_date: date = Query(..., alias="date", description="Slate date in YYYY-MM-DD"),
    slate_key: str = Query(default="main"),
    include_versions: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=500),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_lineup_runs_payload(
            selected_date=selected_date,
            slate_key=slate_key,
            include_versions=include_versions,
            limit=limit,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/v1/lineup-runs/{run_id}")
def lineup_run_detail(
    run_id: str,
    selected_date: date = Query(..., alias="date", description="Slate date in YYYY-MM-DD"),
    slate_key: str = Query(default="main"),
    include_lineups: bool = Query(default=True),
    include_upload_csv: bool = Query(default=False),
    bucket_name: str | None = Query(default=None, description="Override GCS bucket; defaults to CBB_GCS_BUCKET"),
    gcp_project: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        return build_lineup_run_detail_payload(
            selected_date=selected_date,
            run_id=run_id,
            slate_key=slate_key,
            include_lineups=include_lineups,
            include_upload_csv=include_upload_csv,
            bucket_name=bucket_name,
            gcp_project=gcp_project,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/lineups/generate")
def lineups_generate(
    payload: LineupGenerateRequest,
    x_rotowire_cookie: str | None = Header(default=None),
) -> dict[str, Any]:
    request_data = payload.model_dump(mode="json")
    request_data["selected_date"] = payload.selected_date.isoformat()
    if x_rotowire_cookie:
        request_data["rotowire_cookie"] = x_rotowire_cookie
    job_id = JOB_MANAGER.submit(request_data)
    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/v1/lineups/jobs/{job_id}",
        "artifacts_url": f"/v1/lineups/jobs/{job_id}/artifacts",
    }


@app.get("/v1/lineups/jobs/{job_id}")
def lineups_job_status(job_id: str) -> dict[str, Any]:
    job = JOB_MANAGER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/v1/lineups/jobs/{job_id}/artifacts")
def lineups_job_artifacts(job_id: str) -> dict[str, Any]:
    job = JOB_MANAGER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    artifacts = JOB_MANAGER.list_artifacts(job_id)
    for item in artifacts:
        item["download_url"] = f"/v1/lineups/jobs/{job_id}/artifacts/{item['name']}"
    return {
        "job_id": job_id,
        "status": str(job.get("status") or ""),
        "artifacts": artifacts,
    }


@app.get("/v1/lineups/jobs/{job_id}/artifacts/{artifact_name}")
def lineups_job_artifact_download(job_id: str, artifact_name: str) -> FileResponse:
    path = JOB_MANAGER.resolve_artifact_path(job_id, artifact_name)
    if path is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    media = "application/octet-stream"
    suffix = path.suffix.lower()
    if suffix == ".csv":
        media = "text/csv"
    elif suffix == ".json":
        media = "application/json"
    return FileResponse(path=str(path), media_type=media, filename=path.name)


@app.post("/v1/registry/import-dk-slate")
async def import_dk_slate(
    file: UploadFile = File(..., description="DraftKings salary CSV"),
    selected_date: date = Query(..., description="Slate date in YYYY-MM-DD"),
    slate_key: str = Query(default="main"),
    contest_type: str = Query(default="Classic"),
    slate_name: str = Query(default="All"),
    slate_id: int | None = Query(default=None),
    site_id: int = Query(default=1),
    persist: bool = Query(default=True, description="Write overrides to data/dk_manual_overrides.csv"),
    x_rotowire_cookie: str | None = Header(default=None),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="file must be a CSV")
    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="file payload was empty")
    try:
        result = import_dk_slate_overrides(
            dk_slate_csv_bytes=payload,
            selected_date=selected_date,
            slate_key=slate_key,
            contest_type=contest_type,
            slate_name=slate_name,
            slate_id=slate_id,
            site_id=site_id,
            cookie_header=x_rotowire_cookie,
            persist=persist,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result
