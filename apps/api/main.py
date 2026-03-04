from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from college_basketball_dfs.cbb_api_service import (
    build_registry_coverage,
    import_dk_slate_overrides,
    list_rotowire_slates_for_date,
)


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
