from __future__ import annotations

import io
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from college_basketball_dfs.cbb_backfill import iter_dates, season_start_for_date
from college_basketball_dfs.cbb_gcs import CbbGcsStore, build_storage_client
from college_basketball_dfs.cbb_ncaa import prior_day
from college_basketball_dfs.cbb_tournament_review import (
    extract_actual_ownership_from_standings,
    normalize_contest_standings_frame,
)


PROJECTED_OWNERSHIP_ALIASES = (
    "projected_ownership",
    "projected ownership",
    "projectedownership",
    "proj_ownership",
    "projownership",
    "ownership_projection",
    "ownershipprojection",
    "ownership_proj",
    "ownershipproj",
    "own_pct",
    "own%",
    "ownership",
    "projected_ownership_v1",
)
ACTUAL_OWNERSHIP_ALIASES = (
    "actual_ownership",
    "actual ownership",
    "actualownership",
    "field_ownership",
    "field ownership",
    "field_ownership_pct",
    "field ownership pct",
    "pct_drafted",
    "pct drafted",
    "drafted",
)
OWNERSHIP_SOURCE_ALIASES = (
    "source_kind",
    "source",
    "ownership_source",
    "dataset_source",
)
PROJECTED_POINTS_ALIASES = (
    "projected_dk_points",
    "projected dk points",
    "projected_points",
    "projected points",
    "projection",
    "projection_points",
    "projection points",
    "proj_points",
    "proj points",
    "dk_projection",
    "dk projection",
    "blended_projection",
    "blended projection",
    "our_dk_projection",
    "our dk projection",
    "our_projection",
    "our projection",
    "vegas_dk_projection",
    "vegas dk projection",
)
NULLISH_TEXT_VALUES = {"", "nan", "none", "null"}
NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


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


def _read_contest_standings_ownership_for_date(store: CbbGcsStore, slate_date: date) -> pd.DataFrame:
    prefix = f"{store.contest_standings_prefix}/{slate_date.isoformat()}_"
    try:
        blob_names = sorted(
            {
                str(blob.name or "")
                for blob in store.bucket.list_blobs(prefix=prefix)
                if str(blob.name or "").lower().endswith(".csv")
            }
        )
    except Exception:
        return pd.DataFrame()
    if not blob_names:
        return pd.DataFrame()

    own_frames: list[pd.DataFrame] = []
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
        one["actual_ownership"] = pd.to_numeric(one.get("actual_ownership"), errors="coerce")
        one = one.loc[(one["Name"] != "") & one["actual_ownership"].notna(), ["Name", "actual_ownership"]]
        if one.empty:
            continue
        own_frames.append(one)

    if not own_frames:
        return pd.DataFrame()

    out = pd.concat(own_frames, ignore_index=True)
    out = (
        out.groupby("Name", as_index=False)["actual_ownership"]
        .mean()
        .sort_values("Name", kind="stable")
        .reset_index(drop=True)
    )
    return out


@st.cache_data(ttl=900, show_spinner=False)
def load_projection_player_totals_dataset(
    bucket_name: str,
    start_date: date,
    end_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)

    frames: list[pd.DataFrame] = []
    dates = iter_dates(start_date, end_date)
    projection_days = 0
    ownership_days = 0
    dk_slate_days = 0
    projection_rows = 0
    ownership_rows = 0
    ownership_rows_from_standings = 0
    ownership_days_from_standings = 0
    dk_slate_rows = 0

    for d in dates:
        day_frames: list[pd.DataFrame] = []

        projection_csv = store.read_projections_csv(d)
        if projection_csv and projection_csv.strip():
            try:
                projection_df = pd.read_csv(io.StringIO(projection_csv))
            except Exception:
                projection_df = pd.DataFrame()
            if not projection_df.empty:
                projection_days += 1
                projection_rows += int(len(projection_df))
                projection_df = projection_df.copy()
                projection_df["review_date"] = d.isoformat()
                projection_df["source_kind"] = "projection_snapshot"
                day_frames.append(projection_df)

        try:
            ownership_csv = store.read_ownership_csv(d)
        except Exception:
            ownership_csv = ""
        ownership_df = pd.DataFrame()
        ownership_source = ""
        if ownership_csv and ownership_csv.strip():
            try:
                ownership_df = pd.read_csv(io.StringIO(ownership_csv))
            except Exception:
                ownership_df = pd.DataFrame()
            if not ownership_df.empty:
                ownership_source = "ownership_csv"

        if ownership_df.empty:
            ownership_df = _read_contest_standings_ownership_for_date(store, d)
            if not ownership_df.empty:
                ownership_source = "contest_standings"

        if not ownership_df.empty:
            ownership_days += 1
            ownership_rows += int(len(ownership_df))
            if ownership_source == "contest_standings":
                ownership_days_from_standings += 1
                ownership_rows_from_standings += int(len(ownership_df))
            ownership_df = ownership_df.copy()
            ownership_df["review_date"] = d.isoformat()
            ownership_df["source_kind"] = ownership_source or "ownership_snapshot"
            day_frames.append(ownership_df)

        try:
            dk_slate_csv = store.read_dk_slate_csv(d)
        except Exception:
            dk_slate_csv = ""
        if dk_slate_csv and dk_slate_csv.strip():
            try:
                dk_slate_df = pd.read_csv(io.StringIO(dk_slate_csv))
            except Exception:
                dk_slate_df = pd.DataFrame()
            if not dk_slate_df.empty:
                dk_slate_days += 1
                dk_slate_rows += int(len(dk_slate_df))
                dk_slate_df = dk_slate_df.copy()
                dk_slate_df["review_date"] = d.isoformat()
                dk_slate_df["source_kind"] = "dk_slate"
                day_frames.append(dk_slate_df)

        frames.extend(day_frames)

    if not frames:
        return pd.DataFrame(), {
            "dates_scanned": int(len(dates)),
            "projection_days_found": 0,
            "ownership_days_found": 0,
            "ownership_days_from_standings": 0,
            "dk_slate_days_found": 0,
            "projection_rows": 0,
            "ownership_rows": 0,
            "ownership_rows_from_standings": 0,
            "dk_slate_rows": 0,
            "combined_rows": 0,
        }

    out = pd.concat(frames, ignore_index=True)
    return out, {
        "dates_scanned": int(len(dates)),
        "projection_days_found": int(projection_days),
        "ownership_days_found": int(ownership_days),
        "ownership_days_from_standings": int(ownership_days_from_standings),
        "dk_slate_days_found": int(dk_slate_days),
        "projection_rows": int(projection_rows),
        "ownership_rows": int(ownership_rows),
        "ownership_rows_from_standings": int(ownership_rows_from_standings),
        "dk_slate_rows": int(dk_slate_rows),
        "combined_rows": int(len(out)),
    }


def _build_actual_results_from_players_frame(players_df: pd.DataFrame) -> pd.DataFrame:
    if players_df is None or players_df.empty:
        return pd.DataFrame()

    def _pick_col(candidates: tuple[str, ...]) -> str | None:
        for col in candidates:
            if col in players_df.columns:
                return col
        return None

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
    cols = [c for c in needed if c in players_df.columns]
    if not cols:
        return pd.DataFrame()

    out = players_df[cols].copy().rename(
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
    team_fallback_col = _pick_col(("team_abbrev", "TeamAbbrev", "team", "Team"))
    if ("team_name" not in out.columns or out["team_name"].isna().all()) and team_fallback_col:
        out["team_name"] = players_df[team_fallback_col]
    position_col = _pick_col(("position", "Position", "roster_position", "Roster Position", "pos", "Pos"))
    if position_col:
        out["position"] = players_df[position_col]
    else:
        out["position"] = pd.NA
    salary_col = _pick_col(("salary", "Salary", "dk_salary", "dk salary"))
    if salary_col:
        out["salary"] = pd.to_numeric(players_df[salary_col], errors="coerce")
    else:
        out["salary"] = pd.NA

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


@st.cache_data(ttl=900, show_spinner=False)
def load_actual_player_history_dataset(
    bucket_name: str,
    start_date: date,
    end_date: date,
    gcp_project: str | None,
    service_account_json: str | None,
    service_account_json_b64: str | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = build_storage_client(
        service_account_json=service_account_json,
        service_account_json_b64=service_account_json_b64,
        project=gcp_project,
    )
    store = CbbGcsStore(bucket_name=bucket_name, client=client)

    frames: list[pd.DataFrame] = []
    dates = iter_dates(start_date, end_date)
    actual_days = 0

    for d in dates:
        try:
            players_blob = store.players_blob_name(d)
            players_csv = store.read_players_csv_blob(players_blob)
        except Exception:
            players_csv = ""
        if not players_csv or not players_csv.strip():
            continue
        try:
            players_df = pd.read_csv(io.StringIO(players_csv))
        except Exception:
            continue
        if players_df.empty:
            continue

        actual_df = _build_actual_results_from_players_frame(players_df)
        if actual_df.empty:
            continue
        actual_days += 1
        actual_df = actual_df.copy()
        actual_df["review_date"] = d.isoformat()
        frames.append(actual_df)

    if not frames:
        return pd.DataFrame(), {
            "dates_scanned": int(len(dates)),
            "actual_days_found": int(actual_days),
            "actual_rows": 0,
        }

    out = pd.concat(frames, ignore_index=True)
    return out, {
        "dates_scanned": int(len(dates)),
        "actual_days_found": int(actual_days),
        "actual_rows": int(len(out)),
    }


def _norm_col_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _resolve_column_alias(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    if df is None or df.empty:
        return None
    normalized = {_norm_col_key(c): str(c) for c in df.columns}
    for alias in aliases:
        alias_key = _norm_col_key(alias)
        if alias_key in normalized:
            return normalized[alias_key]
    return None


def _norm_text_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _norm_text_key_loose(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in NAME_SUFFIX_TOKENS]
    return "".join(tokens)


def _first_nonempty(series: pd.Series) -> str:
    for value in series:
        text = str(value or "").strip()
        if text and text.lower() not in NULLISH_TEXT_VALUES:
            return text
    return ""


def _name_key_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    loose = text.map(_norm_text_key_loose)
    strict = text.map(_norm_text_key)
    out = loose.where(loose != "", strict)
    return out.fillna("")


def _coerce_ownership_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace("%", "", regex=False)
    text = text.mask(text.str.lower().isin(NULLISH_TEXT_VALUES), pd.NA)
    out = pd.to_numeric(text, errors="coerce")
    valid = out.dropna()
    if not valid.empty and float(valid.quantile(0.95)) <= 1.0:
        out = out * 100.0
    return out


def _build_player_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "player_key",
        "player_name",
        "team",
        "position",
        "review_date",
        "row_order",
        "minutes",
        "actual_dk_points",
        "projected_dk_points",
        "projected_ownership",
        "actual_ownership",
        "source_kind",
        "salary",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    work = df.copy()
    id_col = _resolve_column_alias(work, ("ID", "id", "player_id", "playerid", "dkid", "draftkingsid", "nameid"))
    name_col = _resolve_column_alias(work, ("Name", "name", "player_name", "player", "athlete"))
    team_col = _resolve_column_alias(
        work,
        ("TeamAbbrev", "team_abbrev", "teamabbr", "teamabbreviation", "team_name", "team", "Team"),
    )
    position_col = _resolve_column_alias(work, ("Position", "position", "Roster Position", "roster_position", "Pos", "pos"))
    review_date_col = _resolve_column_alias(work, ("review_date", "review date", "slate_date", "game_date", "date"))
    salary_col = _resolve_column_alias(work, ("Salary", "salary", "dk_salary", "dk salary"))
    minutes_col = _resolve_column_alias(
        work,
        (
            "actual_minutes",
            "actual minutes",
            "minutes_played",
            "minutes",
            "our_minutes_recent",
            "our_minutes_last7",
            "our_minutes_avg",
        ),
    )
    points_col = _resolve_column_alias(work, ("actual_dk_points", "actual dk points", "final_dk_points", "dk_points"))
    projected_points_col = _resolve_column_alias(work, PROJECTED_POINTS_ALIASES)
    projected_ownership_col = _resolve_column_alias(work, PROJECTED_OWNERSHIP_ALIASES)
    actual_ownership_col = _resolve_column_alias(work, ACTUAL_OWNERSHIP_ALIASES)
    source_kind_col = _resolve_column_alias(work, OWNERSHIP_SOURCE_ALIASES)

    out = pd.DataFrame(index=work.index)
    out["player_id"] = work[id_col] if id_col else ""
    out["player_id"] = out["player_id"].astype(str).str.strip()
    # Supports DK exports where "Name + ID" / "nameid" stores "Name (123456)".
    parsed_id = out["player_id"].str.extract(r"\(([^()]+)\)\s*$")[0]
    out["player_id"] = parsed_id.where(parsed_id.notna() & (parsed_id.astype(str).str.strip() != ""), out["player_id"])
    out["player_name"] = work[name_col] if name_col else ""
    out["team"] = work[team_col] if team_col else ""
    out["position"] = work[position_col] if position_col else ""
    out["review_date"] = work[review_date_col] if review_date_col else pd.NA
    out["salary"] = pd.to_numeric(work[salary_col], errors="coerce") if salary_col else pd.NA
    out["minutes"] = pd.to_numeric(work[minutes_col], errors="coerce") if minutes_col else pd.NA
    out["actual_dk_points"] = pd.to_numeric(work[points_col], errors="coerce") if points_col else pd.NA
    out["projected_dk_points"] = (
        pd.to_numeric(work[projected_points_col], errors="coerce") if projected_points_col else pd.NA
    )
    out["projected_ownership"] = (
        _coerce_ownership_series(work[projected_ownership_col]) if projected_ownership_col else pd.NA
    )
    out["actual_ownership"] = _coerce_ownership_series(work[actual_ownership_col]) if actual_ownership_col else pd.NA
    out["source_kind"] = work[source_kind_col] if source_kind_col else ""
    out["row_order"] = pd.RangeIndex(start=0, stop=len(out), step=1)

    out["player_id"] = out["player_id"].where(~out["player_id"].str.lower().isin(NULLISH_TEXT_VALUES), "")
    out["player_id"] = out["player_id"].str.replace(r"\.0+$", "", regex=True)
    out["player_name"] = out["player_name"].astype(str).str.strip()
    out["team"] = out["team"].astype(str).str.strip().str.upper()
    out["position"] = out["position"].astype(str).str.strip().str.upper()
    out["player_name_key"] = out["player_name"].map(_norm_text_key_loose)
    out["player_name_key"] = out["player_name_key"].where(out["player_name_key"] != "", out["player_name"].map(_norm_text_key))
    out["player_id_key"] = out["player_id"].map(_norm_text_key)
    out["team"] = out["team"].replace({"NAN": "", "NONE": "", "NULL": ""})
    out["position"] = out["position"].replace({"NAN": "", "NONE": "", "NULL": ""})
    out["source_kind"] = out["source_kind"].astype(str).str.strip().str.lower()

    out["player_key"] = ""

    # Backfill missing team when name maps to a single known team in this frame.
    known_team = out.loc[(out["player_name_key"] != "") & (out["team"] != ""), ["player_name_key", "team"]].drop_duplicates()
    if not known_team.empty:
        team_counts = known_team.groupby("player_name_key", as_index=False)["team"].nunique()
        single_name_keys = set(team_counts.loc[team_counts["team"] == 1, "player_name_key"].astype(str).tolist())
        if single_name_keys:
            team_map = (
                known_team.loc[known_team["player_name_key"].isin(single_name_keys)]
                .drop_duplicates("player_name_key")
                .set_index("player_name_key")["team"]
                .to_dict()
            )
            missing_team = (out["team"] == "") & (out["player_name_key"] != "")
            out.loc[missing_team, "team"] = out.loc[missing_team, "player_name_key"].map(team_map).fillna("")

    has_name_key = out["player_name_key"] != ""
    has_id_key = out["player_id_key"] != ""
    # Prefer an ID+name composite to avoid accidental collisions when source "id" is non-player scoped.
    out.loc[has_id_key & has_name_key, "player_key"] = (
        "id:" + out.loc[has_id_key & has_name_key, "player_id_key"] + "|name:"
        + out.loc[has_id_key & has_name_key, "player_name_key"]
    )
    missing_key = out["player_key"] == ""
    out.loc[missing_key & has_id_key, "player_key"] = "id:" + out.loc[missing_key & has_id_key, "player_id_key"]
    has_team = out["team"] != ""
    missing_key = out["player_key"] == ""
    out.loc[missing_key & has_name_key & has_team, "player_key"] = (
        "name:" + out.loc[missing_key & has_name_key & has_team, "player_name_key"] + "|team:"
        + out.loc[missing_key & has_name_key & has_team, "team"]
    )
    missing_key = out["player_key"] == ""
    out.loc[missing_key & has_name_key, "player_key"] = "name:" + out.loc[missing_key & has_name_key, "player_name_key"]
    out = out.loc[out["player_key"] != ""].copy()
    out["review_date"] = pd.to_datetime(out["review_date"], errors="coerce")

    source_is_contest = out["source_kind"].astype(str).str.contains("contest", regex=False)
    if bool(source_is_contest.any()):
        contest_missing_actual = source_is_contest & out["actual_ownership"].isna() & out["projected_ownership"].notna()
        out.loc[contest_missing_actual, "actual_ownership"] = out.loc[contest_missing_actual, "projected_ownership"]
    source_not_contest = ~source_is_contest
    non_contest_missing_proj = source_not_contest & out["projected_ownership"].isna() & out["actual_ownership"].notna()
    out.loc[non_contest_missing_proj, "projected_ownership"] = out.loc[non_contest_missing_proj, "actual_ownership"]

    return out[cols].copy()


def _build_metric_daily_frame(history_df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    if history_df is None or history_df.empty or value_col not in history_df.columns:
        return pd.DataFrame(columns=["player_key", value_col, "_sort_date", "_sort_ord"])

    work = history_df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.loc[work["player_key"].astype(str).str.strip() != ""].copy()
    work = work.loc[work[value_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["player_key", value_col, "_sort_date", "_sort_ord"])

    work["review_date"] = pd.to_datetime(work.get("review_date"), errors="coerce")
    work["_review_day"] = work["review_date"].dt.strftime("%Y-%m-%d")
    has_day = work["_review_day"].astype(str).str.strip() != ""
    if bool(has_day.any()):
        # Collapse duplicate snapshot rows for the same player/day to one per-day value.
        work = (
            work.loc[has_day]
            .groupby(["player_key", "_review_day"], as_index=False)[value_col]
            .mean(numeric_only=True)
            .rename(columns={"_review_day": "review_day"})
        )
        work["_sort_date"] = pd.to_datetime(work["review_day"], errors="coerce")
        work["_sort_ord"] = 0.0
    else:
        work["_sort_date"] = pd.to_datetime(work.get("review_date"), errors="coerce")
        work["_sort_ord"] = pd.to_numeric(work.get("row_order"), errors="coerce").fillna(0.0)

    work = work.sort_values(["player_key", "_sort_date", "_sort_ord"], ascending=[True, False, False], kind="stable")
    return work[["player_key", value_col, "_sort_date", "_sort_ord"]].copy()


def _aggregate_player_metric(
    history_df: pd.DataFrame,
    *,
    value_col: str,
    season_col: str,
    last5_col: str,
    agg_mode: str,
) -> pd.DataFrame:
    daily = _build_metric_daily_frame(history_df, value_col=value_col)
    if daily.empty:
        return pd.DataFrame(columns=["player_key", season_col, last5_col])
    last5 = daily.groupby("player_key", sort=False, as_index=False).head(5).copy()

    mode = str(agg_mode or "").strip().lower()
    if mode == "mean":
        season = (
            daily.groupby("player_key", as_index=False)[value_col]
            .mean(numeric_only=True)
            .rename(columns={value_col: season_col})
        )
        recent = (
            last5.groupby("player_key", as_index=False)[value_col]
            .mean(numeric_only=True)
            .rename(columns={value_col: last5_col})
        )
    elif mode == "median":
        season = (
            daily.groupby("player_key", as_index=False)[value_col]
            .median(numeric_only=True)
            .rename(columns={value_col: season_col})
        )
        recent = (
            last5.groupby("player_key", as_index=False)[value_col]
            .median(numeric_only=True)
            .rename(columns={value_col: last5_col})
        )
    else:
        season = (
            daily.groupby("player_key", as_index=False)[value_col]
            .sum(numeric_only=True)
            .rename(columns={value_col: season_col})
        )
        recent = (
            last5.groupby("player_key", as_index=False)[value_col]
            .sum(numeric_only=True)
            .rename(columns={value_col: last5_col})
        )

    return season.merge(recent, on="player_key", how="outer")


def _aggregate_metric_sample_counts(
    history_df: pd.DataFrame,
    *,
    value_col: str,
    season_count_col: str,
    last5_count_col: str,
) -> pd.DataFrame:
    daily = _build_metric_daily_frame(history_df, value_col=value_col)
    if daily.empty:
        return pd.DataFrame(columns=["player_key", season_count_col, last5_count_col])
    last5 = daily.groupby("player_key", sort=False, as_index=False).head(5).copy()
    season_counts = (
        daily.groupby("player_key", as_index=False)[value_col]
        .count()
        .rename(columns={value_col: season_count_col})
    )
    recent_counts = (
        last5.groupby("player_key", as_index=False)[value_col]
        .count()
        .rename(columns={value_col: last5_count_col})
    )
    return season_counts.merge(recent_counts, on="player_key", how="outer")


def _variance_or_na(series: pd.Series, min_samples: int = 2) -> float | Any:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if int(len(vals)) < int(max(1, min_samples)):
        return pd.NA
    return float(vals.var(ddof=0))


def _aggregate_player_variance(
    history_df: pd.DataFrame,
    *,
    value_col: str,
    season_variance_col: str,
    last5_variance_col: str,
) -> pd.DataFrame:
    daily = _build_metric_daily_frame(history_df, value_col=value_col)
    if daily.empty:
        return pd.DataFrame(columns=["player_key", season_variance_col, last5_variance_col])

    last5 = daily.groupby("player_key", sort=False, as_index=False).head(5).copy()

    season = (
        daily.groupby("player_key", as_index=False)[value_col]
        .agg(_variance_or_na)
        .rename(columns={value_col: season_variance_col})
    )
    recent = (
        last5.groupby("player_key", as_index=False)[value_col]
        .agg(_variance_or_na)
        .rename(columns={value_col: last5_variance_col})
    )
    out = season.merge(recent, on="player_key", how="outer")
    out[season_variance_col] = pd.to_numeric(out[season_variance_col], errors="coerce")
    out[last5_variance_col] = pd.to_numeric(out[last5_variance_col], errors="coerce")
    return out


def build_ownership_accuracy_daily_frame(player_totals_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    history = _build_player_history_frame(player_totals_df)
    if history.empty:
        return pd.DataFrame(), {
            "slates": 0,
            "matched_rows": 0,
            "projected_rows": 0,
            "actual_rows": 0,
            "ambiguous_name_day_pairs_excluded": 0,
        }

    work = history.copy()
    work["review_date"] = pd.to_datetime(work.get("review_date"), errors="coerce")
    work["_review_day"] = work["review_date"].dt.strftime("%Y-%m-%d")
    work["_name_key"] = _name_key_series(work["player_name"] if "player_name" in work.columns else pd.Series("", index=work.index))
    work["team"] = work["team"].astype(str).str.strip().str.upper()
    work["position"] = work["position"].astype(str).str.strip().str.upper()
    work["player_name"] = work["player_name"].astype(str).str.strip()
    work = work.loc[(work["_review_day"].astype(str).str.strip() != "") & (work["_name_key"].astype(str).str.strip() != "")].copy()
    if work.empty:
        return pd.DataFrame(), {
            "slates": 0,
            "matched_rows": 0,
            "projected_rows": 0,
            "actual_rows": 0,
            "ambiguous_name_day_pairs_excluded": 0,
        }

    ambiguous_name_days = (
        work.loc[work["team"] != ""]
        .groupby(["_review_day", "_name_key"], as_index=False)["team"]
        .nunique()
        .rename(columns={"team": "_team_count"})
    )
    ambiguous_name_days = ambiguous_name_days.loc[ambiguous_name_days["_team_count"] > 1].copy()
    ambiguous_keys: set[str] = set()
    if not ambiguous_name_days.empty:
        ambiguous_keys = set(
            (
                ambiguous_name_days["_review_day"].astype(str)
                + "||"
                + ambiguous_name_days["_name_key"].astype(str)
            ).tolist()
        )
        work["_day_name_key"] = work["_review_day"].astype(str) + "||" + work["_name_key"].astype(str)
        work = work.loc[~work["_day_name_key"].isin(ambiguous_keys)].copy()
        work = work.drop(columns=["_day_name_key"], errors="ignore")

    projection_daily = (
        work.loc[work["projected_ownership"].notna()]
        .groupby(["_review_day", "_name_key"], as_index=False)
        .agg(
            player_name_proj=("player_name", _first_nonempty),
            team_proj=("team", _first_nonempty),
            position_proj=("position", _first_nonempty),
            projected_ownership=("projected_ownership", "mean"),
            projected_dk_points=("projected_dk_points", "mean"),
            salary=("salary", "mean"),
            projected_source_rows=("projected_ownership", "count"),
        )
    )
    actual_daily = (
        work.loc[work["actual_ownership"].notna()]
        .groupby(["_review_day", "_name_key"], as_index=False)
        .agg(
            player_name_actual=("player_name", _first_nonempty),
            team_actual=("team", _first_nonempty),
            position_actual=("position", _first_nonempty),
            actual_ownership=("actual_ownership", "mean"),
            actual_source_rows=("actual_ownership", "count"),
        )
    )

    if projection_daily.empty and actual_daily.empty:
        return pd.DataFrame(), {
            "slates": 0,
            "matched_rows": 0,
            "projected_rows": 0,
            "actual_rows": 0,
            "ambiguous_name_day_pairs_excluded": int(len(ambiguous_keys)),
        }

    out = projection_daily.merge(
        actual_daily,
        on=["_review_day", "_name_key"],
        how="outer",
    )
    out["review_date"] = pd.to_datetime(out["_review_day"], errors="coerce")
    out["Player Name"] = out["player_name_proj"].where(
        out["player_name_proj"].astype(str).str.strip() != "",
        out["player_name_actual"],
    )
    out["Team"] = out["team_proj"].where(out["team_proj"].astype(str).str.strip() != "", out["team_actual"])
    out["Position"] = out["position_proj"].where(
        out["position_proj"].astype(str).str.strip() != "",
        out["position_actual"],
    )
    out["Player Name"] = out["Player Name"].astype(str).str.strip()
    out["Team"] = out["Team"].astype(str).str.strip().str.upper()
    out["Position"] = out["Position"].astype(str).str.strip().str.upper()
    out["projected_ownership"] = pd.to_numeric(out.get("projected_ownership"), errors="coerce")
    out["actual_ownership"] = pd.to_numeric(out.get("actual_ownership"), errors="coerce")
    out["projected_dk_points"] = pd.to_numeric(out.get("projected_dk_points"), errors="coerce")
    out["salary"] = pd.to_numeric(out.get("salary"), errors="coerce")
    out["projected_source_rows"] = pd.to_numeric(out.get("projected_source_rows"), errors="coerce").round(0).astype("Int64")
    out["actual_source_rows"] = pd.to_numeric(out.get("actual_source_rows"), errors="coerce").round(0).astype("Int64")
    out["ownership_delta"] = out["actual_ownership"] - out["projected_ownership"]
    out["ownership_abs_delta"] = out["ownership_delta"].abs()
    out = out.loc[out["Player Name"] != ""].copy()
    out = out.sort_values(
        ["review_date", "ownership_abs_delta", "Player Name"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    meta = {
        "slates": int(out["review_date"].dropna().dt.strftime("%Y-%m-%d").nunique()),
        "matched_rows": int((out["projected_ownership"].notna() & out["actual_ownership"].notna()).sum()),
        "projected_rows": int(out["projected_ownership"].notna().sum()),
        "actual_rows": int(out["actual_ownership"].notna().sum()),
        "ambiguous_name_day_pairs_excluded": int(len(ambiguous_keys)),
    }
    keep_cols = [
        "review_date",
        "Player Name",
        "Team",
        "Position",
        "projected_dk_points",
        "salary",
        "projected_ownership",
        "actual_ownership",
        "ownership_delta",
        "ownership_abs_delta",
        "projected_source_rows",
        "actual_source_rows",
    ]
    return out[keep_cols].copy(), meta


def build_player_review_table(
    actual_player_history_df: pd.DataFrame,
    player_totals_df: pd.DataFrame,
    *,
    dfs_only: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, bool]]:
    actual_history = _build_player_history_frame(actual_player_history_df)
    projection_history = _build_player_history_frame(player_totals_df)
    projection_history_by_name = projection_history.copy()
    if not projection_history_by_name.empty:
        projection_history_by_name["player_key"] = _name_key_series(
            projection_history_by_name["player_name"]
            if "player_name" in projection_history_by_name.columns
            else pd.Series("", index=projection_history_by_name.index)
        )
        projection_history_by_name = projection_history_by_name.loc[
            projection_history_by_name["player_key"].astype(str).str.strip() != ""
        ].copy()
    projection_key_set = set(
        projection_history["player_key"].astype(str).str.strip().loc[
            projection_history["player_key"].astype(str).str.strip() != ""
        ].tolist()
    )
    projection_name_set = set(
        _name_key_series(
            projection_history["player_name"]
            if "player_name" in projection_history.columns
            else pd.Series(dtype=str)
        )
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .tolist()
    )

    identity_frames: list[pd.DataFrame] = []
    for frame in [actual_history, projection_history]:
        if frame.empty:
            continue
        identity_frames.append(frame[["player_key", "player_name", "team", "position", "review_date", "row_order"]].copy())

    if not identity_frames:
        meta = {"has_points": False, "has_ownership": False, "has_salary": False}
        return pd.DataFrame(), [], meta

    identity = pd.concat(identity_frames, ignore_index=True)
    if dfs_only and (projection_key_set or projection_name_set):
        identity["_name_key"] = _name_key_series(
            identity["player_name"] if "player_name" in identity.columns else pd.Series("", index=identity.index)
        )
        keep_mask = identity["player_key"].astype(str).isin(projection_key_set) | identity["_name_key"].astype(str).isin(
            projection_name_set
        )
        identity = identity.loc[keep_mask].copy()
        identity = identity.drop(columns=["_name_key"], errors="ignore")
    if identity.empty:
        meta = {"has_points": False, "has_ownership": False, "has_salary": False}
        return pd.DataFrame(), [], meta
    identity["review_date"] = pd.to_datetime(identity.get("review_date"), errors="coerce")
    identity["_sort_date"] = identity["review_date"]
    identity["_sort_ord"] = pd.to_numeric(identity.get("row_order"), errors="coerce").fillna(0.0)
    identity = identity.sort_values(
        ["player_key", "_sort_date", "_sort_ord"],
        ascending=[True, False, False],
        kind="stable",
    )
    identity_summary = (
        identity.groupby("player_key", as_index=False)
        .agg(
            player_name=("player_name", _first_nonempty),
            team=("team", _first_nonempty),
            position=("position", _first_nonempty),
        )
        .copy()
    )

    points_season_summary = _aggregate_player_metric(
        actual_history,
        value_col="actual_dk_points",
        season_col="Total Fantasy Points Season",
        last5_col="_unused_points_last5_sum",
        agg_mode="sum",
    )
    points_avg_summary = _aggregate_player_metric(
        actual_history,
        value_col="actual_dk_points",
        season_col="_unused_points_season_avg",
        last5_col="Average Fantasy Points Per Game Last 5",
        agg_mode="mean",
    )
    points_median_summary = _aggregate_player_metric(
        actual_history,
        value_col="actual_dk_points",
        season_col="_unused_points_season_median",
        last5_col="Median Fantasy Points Per Game Last 5",
        agg_mode="median",
    )
    projected_points_summary = _aggregate_player_metric(
        projection_history,
        value_col="projected_dk_points",
        season_col="Average Projected Fantasy Points Season",
        last5_col="Average Projected Fantasy Points Last 5",
        agg_mode="mean",
    )
    projected_points_summary_by_name = _aggregate_player_metric(
        projection_history_by_name,
        value_col="projected_dk_points",
        season_col="Average Projected Fantasy Points Season",
        last5_col="Average Projected Fantasy Points Last 5",
        agg_mode="mean",
    )
    projected_ownership_summary = _aggregate_player_metric(
        projection_history,
        value_col="projected_ownership",
        season_col="Average Projected Ownership Season",
        last5_col="Average Projected Ownership Last 5 Games",
        agg_mode="mean",
    )
    projected_ownership_summary_by_name = _aggregate_player_metric(
        projection_history_by_name,
        value_col="projected_ownership",
        season_col="Average Projected Ownership Season",
        last5_col="Average Projected Ownership Last 5 Games",
        agg_mode="mean",
    )
    projected_ownership_counts = _aggregate_metric_sample_counts(
        projection_history,
        value_col="projected_ownership",
        season_count_col="Projected Ownership Games Season",
        last5_count_col="Projected Ownership Games Last 5 Window",
    )
    projected_ownership_counts_by_name = _aggregate_metric_sample_counts(
        projection_history_by_name,
        value_col="projected_ownership",
        season_count_col="Projected Ownership Games Season",
        last5_count_col="Projected Ownership Games Last 5 Window",
    )
    actual_ownership_summary = _aggregate_player_metric(
        projection_history,
        value_col="actual_ownership",
        season_col="Average Actual Ownership Season",
        last5_col="Average Actual Ownership Last 5 Games",
        agg_mode="mean",
    )
    actual_ownership_summary_by_name = _aggregate_player_metric(
        projection_history_by_name,
        value_col="actual_ownership",
        season_col="Average Actual Ownership Season",
        last5_col="Average Actual Ownership Last 5 Games",
        agg_mode="mean",
    )
    actual_ownership_counts = _aggregate_metric_sample_counts(
        projection_history,
        value_col="actual_ownership",
        season_count_col="Actual Ownership Games Season",
        last5_count_col="Actual Ownership Games Last 5 Window",
    )
    actual_ownership_counts_by_name = _aggregate_metric_sample_counts(
        projection_history_by_name,
        value_col="actual_ownership",
        season_count_col="Actual Ownership Games Season",
        last5_count_col="Actual Ownership Games Last 5 Window",
    )
    fantasy_points_variance = _aggregate_player_variance(
        actual_history,
        value_col="actual_dk_points",
        season_variance_col="Fantasy Points Variance Season",
        last5_variance_col="Fantasy Points Variance Last 5 Games",
    )
    minutes_variance = _aggregate_player_variance(
        actual_history,
        value_col="minutes",
        season_variance_col="Minutes Variance Season",
        last5_variance_col="Minutes Variance Last 5 Games",
    )
    ownership_variance = _aggregate_player_variance(
        projection_history,
        value_col="projected_ownership",
        season_variance_col="Ownership Variance Season",
        last5_variance_col="Ownership Variance Last 5 Games",
    )
    ownership_variance_by_name = _aggregate_player_variance(
        projection_history_by_name,
        value_col="projected_ownership",
        season_variance_col="Ownership Variance Season",
        last5_variance_col="Ownership Variance Last 5 Games",
    )

    salary_source = actual_history if actual_history["salary"].notna().any() else projection_history
    salary_summary = _aggregate_player_metric(
        salary_source,
        value_col="salary",
        season_col="Average DK Salary This Season",
        last5_col="_unused_last5_salary",
        agg_mode="mean",
    )
    salary_variance = _aggregate_player_variance(
        salary_source,
        value_col="salary",
        season_variance_col="Salary Variance Season",
        last5_variance_col="Salary Variance Last 5 Games",
    )
    if "_unused_points_last5_sum" in points_season_summary.columns:
        points_season_summary = points_season_summary.drop(columns=["_unused_points_last5_sum"])
    if "_unused_points_season_avg" in points_avg_summary.columns:
        points_avg_summary = points_avg_summary.drop(columns=["_unused_points_season_avg"])
    if "_unused_last5_salary" in salary_summary.columns:
        salary_summary = salary_summary.drop(columns=["_unused_last5_salary"])
    if "_unused_points_season_median" in points_median_summary.columns:
        points_median_summary = points_median_summary.drop(columns=["_unused_points_season_median"])

    out = identity_summary.merge(points_season_summary, on="player_key", how="left")
    out = out.merge(points_avg_summary, on="player_key", how="left")
    out = out.merge(points_median_summary, on="player_key", how="left")
    out = out.merge(projected_points_summary, on="player_key", how="left")
    out = out.merge(projected_ownership_summary, on="player_key", how="left")
    out = out.merge(actual_ownership_summary, on="player_key", how="left")
    out = out.merge(projected_ownership_counts, on="player_key", how="left")
    out = out.merge(actual_ownership_counts, on="player_key", how="left")
    out = out.merge(salary_summary, on="player_key", how="left")
    out = out.merge(minutes_variance, on="player_key", how="left")
    out = out.merge(fantasy_points_variance, on="player_key", how="left")
    out = out.merge(ownership_variance, on="player_key", how="left")
    out = out.merge(salary_variance, on="player_key", how="left")
    out = out.rename(columns={"player_name": "Player Name", "position": "Position", "team": "Team"})
    out["_name_key"] = _name_key_series(
        out["Player Name"] if "Player Name" in out.columns else pd.Series("", index=out.index)
    )

    if not projected_points_summary_by_name.empty:
        proj_name = projected_points_summary_by_name.rename(
            columns={
                "player_key": "_name_key",
                "Average Projected Fantasy Points Season": "_name_avg_proj_pts_season",
                "Average Projected Fantasy Points Last 5": "_name_avg_proj_pts_last5",
            }
        )
        out = out.merge(proj_name, on="_name_key", how="left")
        if "Average Projected Fantasy Points Season" in out.columns:
            proj_season_base = pd.to_numeric(out["Average Projected Fantasy Points Season"], errors="coerce")
            proj_season_fill = pd.to_numeric(out.get("_name_avg_proj_pts_season"), errors="coerce")
            out["Average Projected Fantasy Points Season"] = proj_season_base.where(
                proj_season_base.notna(), proj_season_fill
            )
        if "Average Projected Fantasy Points Last 5" in out.columns:
            proj_last5_base = pd.to_numeric(out["Average Projected Fantasy Points Last 5"], errors="coerce")
            proj_last5_fill = pd.to_numeric(out.get("_name_avg_proj_pts_last5"), errors="coerce")
            out["Average Projected Fantasy Points Last 5"] = proj_last5_base.where(proj_last5_base.notna(), proj_last5_fill)

    if not projected_ownership_summary_by_name.empty:
        proj_own_name = projected_ownership_summary_by_name.rename(
            columns={
                "player_key": "_name_key",
                "Average Projected Ownership Season": "_name_avg_proj_own_season",
                "Average Projected Ownership Last 5 Games": "_name_avg_proj_own_last5",
            }
        )
        out = out.merge(proj_own_name, on="_name_key", how="left")
        if "Average Projected Ownership Season" in out.columns:
            proj_own_season_base = pd.to_numeric(out["Average Projected Ownership Season"], errors="coerce")
            proj_own_season_fill = pd.to_numeric(out.get("_name_avg_proj_own_season"), errors="coerce")
            out["Average Projected Ownership Season"] = proj_own_season_base.where(
                proj_own_season_base.notna(), proj_own_season_fill
            )
        if "Average Projected Ownership Last 5 Games" in out.columns:
            proj_own_last5_base = pd.to_numeric(out["Average Projected Ownership Last 5 Games"], errors="coerce")
            proj_own_last5_fill = pd.to_numeric(out.get("_name_avg_proj_own_last5"), errors="coerce")
            out["Average Projected Ownership Last 5 Games"] = proj_own_last5_base.where(
                proj_own_last5_base.notna(), proj_own_last5_fill
            )

    if not actual_ownership_summary_by_name.empty:
        actual_own_name = actual_ownership_summary_by_name.rename(
            columns={
                "player_key": "_name_key",
                "Average Actual Ownership Season": "_name_avg_actual_own_season",
                "Average Actual Ownership Last 5 Games": "_name_avg_actual_own_last5",
            }
        )
        out = out.merge(actual_own_name, on="_name_key", how="left")
        if "Average Actual Ownership Season" in out.columns:
            actual_own_season_base = pd.to_numeric(out["Average Actual Ownership Season"], errors="coerce")
            actual_own_season_fill = pd.to_numeric(out.get("_name_avg_actual_own_season"), errors="coerce")
            out["Average Actual Ownership Season"] = actual_own_season_base.where(
                actual_own_season_base.notna(), actual_own_season_fill
            )
        if "Average Actual Ownership Last 5 Games" in out.columns:
            actual_own_last5_base = pd.to_numeric(out["Average Actual Ownership Last 5 Games"], errors="coerce")
            actual_own_last5_fill = pd.to_numeric(out.get("_name_avg_actual_own_last5"), errors="coerce")
            out["Average Actual Ownership Last 5 Games"] = actual_own_last5_base.where(
                actual_own_last5_base.notna(), actual_own_last5_fill
            )

    if not ownership_variance_by_name.empty:
        own_var_name = ownership_variance_by_name.rename(
            columns={
                "player_key": "_name_key",
                "Ownership Variance Season": "_name_own_var_season",
                "Ownership Variance Last 5 Games": "_name_own_var_last5",
            }
        )
        out = out.merge(own_var_name, on="_name_key", how="left")
        if "Ownership Variance Season" in out.columns:
            var_season_base = pd.to_numeric(out["Ownership Variance Season"], errors="coerce")
            var_season_fill = pd.to_numeric(out.get("_name_own_var_season"), errors="coerce")
            out["Ownership Variance Season"] = var_season_base.where(var_season_base.notna(), var_season_fill)
        if "Ownership Variance Last 5 Games" in out.columns:
            var_last5_base = pd.to_numeric(out["Ownership Variance Last 5 Games"], errors="coerce")
            var_last5_fill = pd.to_numeric(out.get("_name_own_var_last5"), errors="coerce")
            out["Ownership Variance Last 5 Games"] = var_last5_base.where(var_last5_base.notna(), var_last5_fill)

    if not projected_ownership_counts_by_name.empty:
        proj_own_count_name = projected_ownership_counts_by_name.rename(
            columns={
                "player_key": "_name_key",
                "Projected Ownership Games Season": "_name_proj_own_games_season",
                "Projected Ownership Games Last 5 Window": "_name_proj_own_games_last5",
            }
        )
        out = out.merge(proj_own_count_name, on="_name_key", how="left")
        if "Projected Ownership Games Season" in out.columns:
            proj_season_cnt_base = pd.to_numeric(out["Projected Ownership Games Season"], errors="coerce")
            proj_season_cnt_fill = pd.to_numeric(out.get("_name_proj_own_games_season"), errors="coerce")
            out["Projected Ownership Games Season"] = proj_season_cnt_base.where(
                proj_season_cnt_base.notna(), proj_season_cnt_fill
            )
        if "Projected Ownership Games Last 5 Window" in out.columns:
            proj_last5_cnt_base = pd.to_numeric(out["Projected Ownership Games Last 5 Window"], errors="coerce")
            proj_last5_cnt_fill = pd.to_numeric(out.get("_name_proj_own_games_last5"), errors="coerce")
            out["Projected Ownership Games Last 5 Window"] = proj_last5_cnt_base.where(
                proj_last5_cnt_base.notna(), proj_last5_cnt_fill
            )

    if not actual_ownership_counts_by_name.empty:
        actual_own_count_name = actual_ownership_counts_by_name.rename(
            columns={
                "player_key": "_name_key",
                "Actual Ownership Games Season": "_name_actual_own_games_season",
                "Actual Ownership Games Last 5 Window": "_name_actual_own_games_last5",
            }
        )
        out = out.merge(actual_own_count_name, on="_name_key", how="left")
        if "Actual Ownership Games Season" in out.columns:
            actual_season_cnt_base = pd.to_numeric(out["Actual Ownership Games Season"], errors="coerce")
            actual_season_cnt_fill = pd.to_numeric(out.get("_name_actual_own_games_season"), errors="coerce")
            out["Actual Ownership Games Season"] = actual_season_cnt_base.where(
                actual_season_cnt_base.notna(), actual_season_cnt_fill
            )
        if "Actual Ownership Games Last 5 Window" in out.columns:
            actual_last5_cnt_base = pd.to_numeric(out["Actual Ownership Games Last 5 Window"], errors="coerce")
            actual_last5_cnt_fill = pd.to_numeric(out.get("_name_actual_own_games_last5"), errors="coerce")
            out["Actual Ownership Games Last 5 Window"] = actual_last5_cnt_base.where(
                actual_last5_cnt_base.notna(), actual_last5_cnt_fill
            )

    proj_own_season = pd.to_numeric(out.get("Average Projected Ownership Season"), errors="coerce")
    actual_own_season = pd.to_numeric(out.get("Average Actual Ownership Season"), errors="coerce")
    out["Average Ownership Season"] = proj_own_season.where(proj_own_season.notna(), actual_own_season)
    proj_own_last5 = pd.to_numeric(out.get("Average Projected Ownership Last 5 Games"), errors="coerce")
    actual_own_last5 = pd.to_numeric(out.get("Average Actual Ownership Last 5 Games"), errors="coerce")
    out["Average Ownership Last 5 Games"] = proj_own_last5.where(proj_own_last5.notna(), actual_own_last5)
    proj_own_games_season = pd.to_numeric(out.get("Projected Ownership Games Season"), errors="coerce")
    actual_own_games_season = pd.to_numeric(out.get("Actual Ownership Games Season"), errors="coerce")
    out["Ownership Games Season"] = proj_own_games_season.where(
        proj_own_games_season.notna(), actual_own_games_season
    )
    proj_own_games_last5 = pd.to_numeric(out.get("Projected Ownership Games Last 5 Window"), errors="coerce")
    actual_own_games_last5 = pd.to_numeric(out.get("Actual Ownership Games Last 5 Window"), errors="coerce")
    out["Ownership Games Last 5 Window"] = proj_own_games_last5.where(
        proj_own_games_last5.notna(), actual_own_games_last5
    )

    for col in [
        "Total Fantasy Points Season",
        "Average Fantasy Points Per Game Last 5",
        "Median Fantasy Points Per Game Last 5",
        "Average Projected Fantasy Points Season",
        "Average Projected Fantasy Points Last 5",
        "Average Projected Ownership Season",
        "Average Projected Ownership Last 5 Games",
        "Average Actual Ownership Season",
        "Average Actual Ownership Last 5 Games",
        "Average Ownership Season",
        "Average Ownership Last 5 Games",
        "Average DK Salary This Season",
        "Minutes Variance Season",
        "Minutes Variance Last 5 Games",
        "Fantasy Points Variance Season",
        "Fantasy Points Variance Last 5 Games",
        "Ownership Variance Season",
        "Ownership Variance Last 5 Games",
        "Projected Ownership Games Season",
        "Projected Ownership Games Last 5 Window",
        "Actual Ownership Games Season",
        "Actual Ownership Games Last 5 Window",
        "Ownership Games Season",
        "Ownership Games Last 5 Window",
        "Salary Variance Season",
        "Salary Variance Last 5 Games",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in [
        "Total Fantasy Points Season",
        "Average Fantasy Points Per Game Last 5",
        "Median Fantasy Points Per Game Last 5",
    ]:
        if col in out.columns:
            out[col] = out[col].fillna(0.0)

    for col in [
        "Total Fantasy Points Season",
        "Average Fantasy Points Per Game Last 5",
        "Median Fantasy Points Per Game Last 5",
        "Average Projected Fantasy Points Season",
        "Average Projected Fantasy Points Last 5",
        "Average Projected Ownership Season",
        "Average Projected Ownership Last 5 Games",
        "Average Actual Ownership Season",
        "Average Actual Ownership Last 5 Games",
        "Average Ownership Season",
        "Average Ownership Last 5 Games",
        "Average DK Salary This Season",
        "Minutes Variance Season",
        "Minutes Variance Last 5 Games",
        "Fantasy Points Variance Season",
        "Fantasy Points Variance Last 5 Games",
        "Ownership Variance Season",
        "Ownership Variance Last 5 Games",
        "Salary Variance Season",
        "Salary Variance Last 5 Games",
    ]:
        if col in out.columns:
            out[col] = out[col].round(2)

    out["Player Name"] = out["Player Name"].astype(str).str.strip()
    out["Team"] = out["Team"].astype(str).str.strip().str.upper()
    out["Position"] = out["Position"].astype(str).str.strip().str.upper()
    for cnt_col in [
        "Projected Ownership Games Season",
        "Projected Ownership Games Last 5 Window",
        "Actual Ownership Games Season",
        "Actual Ownership Games Last 5 Window",
        "Ownership Games Season",
        "Ownership Games Last 5 Window",
    ]:
        if cnt_col in out.columns:
            out[cnt_col] = pd.to_numeric(out[cnt_col], errors="coerce").round(0).astype("Int64")
    out = out.drop(
        columns=[
            "_name_key",
            "_name_avg_proj_pts_season",
            "_name_avg_proj_pts_last5",
            "_name_avg_proj_own_season",
            "_name_avg_proj_own_last5",
            "_name_avg_actual_own_season",
            "_name_avg_actual_own_last5",
            "_name_own_var_season",
            "_name_own_var_last5",
            "_name_proj_own_games_season",
            "_name_proj_own_games_last5",
            "_name_actual_own_games_season",
            "_name_actual_own_games_last5",
        ],
        errors="ignore",
    )
    out = out.loc[out["Player Name"] != ""].copy()
    out = out.sort_values(
        ["Team", "Total Fantasy Points Season", "Player Name"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    team_options = sorted([str(x).strip().upper() for x in out["Team"].dropna().unique().tolist() if str(x).strip()])
    meta = {
        "has_points": bool(points_season_summary["Total Fantasy Points Season"].notna().any())
        if not points_season_summary.empty
        else False,
        "has_projected_ownership": bool(
            pd.to_numeric(out.get("Average Projected Ownership Season"), errors="coerce").notna().any()
        )
        if not out.empty and "Average Projected Ownership Season" in out.columns
        else False,
        "has_actual_ownership": bool(
            pd.to_numeric(out.get("Average Actual Ownership Season"), errors="coerce").notna().any()
        )
        if not out.empty and "Average Actual Ownership Season" in out.columns
        else False,
        "has_ownership": bool(
            pd.to_numeric(out.get("Average Ownership Season"), errors="coerce").notna().any()
            or pd.to_numeric(out.get("Average Projected Ownership Season"), errors="coerce").notna().any()
        )
        if not out.empty
        else False,
        "has_salary": bool(salary_summary["Average DK Salary This Season"].notna().any()) if not salary_summary.empty else False,
        "has_projected_points": bool(
            pd.to_numeric(out.get("Average Projected Fantasy Points Season"), errors="coerce").notna().any()
        )
        if not out.empty and "Average Projected Fantasy Points Season" in out.columns
        else False,
    }
    return out, team_options, meta


st.set_page_config(page_title="Player Review", layout="wide")
st.title("Player Review")
st.caption("Review player fantasy totals, ownership trends, and average salary by team.")

default_bucket = os.getenv("CBB_GCS_BUCKET", "").strip() or (_secret("cbb_gcs_bucket") or "")
default_project = os.getenv("GCP_PROJECT", "").strip() or (_secret("gcp_project") or "")
default_start = season_start_for_date(date.today())
default_end = prior_day()

with st.sidebar:
    st.header("Player Review Settings")
    start_date = st.date_input("Start Date", value=default_start, key="player_review_start_date")
    end_date = st.date_input("End Date", value=default_end, key="player_review_end_date")
    bucket_name = st.text_input("GCS Bucket", value=default_bucket, key="player_review_bucket")
    gcp_project = st.text_input("GCP Project (optional)", value=default_project, key="player_review_project")
    player_universe_mode = st.selectbox(
        "Player Universe",
        options=["DFS-Relevant Only", "All Tracked Players"],
        index=0,
        key="player_review_universe_mode",
        help="DFS-Relevant Only keeps players that appear in projections/ownership/DK slate sources.",
    )
    st.caption("This page refreshes automatically when settings change.")
    if st.button("Clear Cached Results", key="clear_player_review_cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")

if start_date > end_date:
    st.error("Start Date must be on or before End Date.")
    st.stop()

if not bucket_name.strip():
    st.error("Set a GCS bucket to run Player Review.")
    st.stop()

cred_json = _resolve_credential_json()
cred_json_b64 = _resolve_credential_json_b64()

with st.spinner("Loading projection snapshots and actual results from GCS..."):
    player_totals_df, player_totals_meta = load_projection_player_totals_dataset(
        bucket_name=bucket_name.strip(),
        start_date=start_date,
        end_date=end_date,
        gcp_project=gcp_project.strip() or None,
        service_account_json=cred_json,
        service_account_json_b64=cred_json_b64,
    )
    actual_history_df, actual_history_meta = load_actual_player_history_dataset(
        bucket_name=bucket_name.strip(),
        start_date=start_date,
        end_date=end_date,
        gcp_project=gcp_project.strip() or None,
        service_account_json=cred_json,
        service_account_json_b64=cred_json_b64,
    )

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Dates Scanned", int(player_totals_meta.get("dates_scanned") or actual_history_meta.get("dates_scanned") or 0))
m2.metric("Projection Days", int(player_totals_meta.get("projection_days_found") or 0))
m3.metric("Ownership Days", int(player_totals_meta.get("ownership_days_found") or 0))
m4.metric("DK Slate Days", int(player_totals_meta.get("dk_slate_days_found") or 0))
m5.metric("Actual Player Days", int(actual_history_meta.get("actual_days_found") or 0))
m6.metric("Projection Rows", int(player_totals_meta.get("projection_rows") or 0))
m7.metric("Actual Player Rows", int(actual_history_meta.get("actual_rows") or 0))

if int(player_totals_meta.get("ownership_rows") or 0) or int(player_totals_meta.get("dk_slate_rows") or 0):
    st.caption(
        f"Supplemental source rows loaded - Ownership: {int(player_totals_meta.get('ownership_rows') or 0)}, "
        f"DK Slate: {int(player_totals_meta.get('dk_slate_rows') or 0)}."
    )
ownership_rows_from_standings = int(player_totals_meta.get("ownership_rows_from_standings") or 0)
if ownership_rows_from_standings > 0:
    st.caption(
        "Ownership fallback rows from Tournament Review contest standings: "
        f"{ownership_rows_from_standings}."
    )

dates_scanned = int(player_totals_meta.get("dates_scanned") or actual_history_meta.get("dates_scanned") or 0)
projection_days = int(player_totals_meta.get("projection_days_found") or 0)
if dates_scanned > 0 and projection_days < dates_scanned:
    st.warning(
        f"Projection snapshots were found for {projection_days} of {dates_scanned} scanned dates "
        f"({start_date.isoformat()} to {end_date.isoformat()}). "
        "Player Review now combines projections + ownership + DK slate files to fill missing columns."
    )
ownership_days = int(player_totals_meta.get("ownership_days_found") or 0)
ownership_days_from_standings = int(player_totals_meta.get("ownership_days_from_standings") or 0)
if dates_scanned > 0 and ownership_days < dates_scanned:
    if ownership_days_from_standings > 0:
        st.info(
            f"Ownership datasets were found for {ownership_days} of {dates_scanned} scanned dates "
            f"({start_date.isoformat()} to {end_date.isoformat()}); "
            f"{ownership_days_from_standings} day(s) came from Tournament Review contest standings."
        )
    else:
        st.info(
            f"Ownership files were found for {ownership_days} of {dates_scanned} scanned dates "
            f"({start_date.isoformat()} to {end_date.isoformat()})."
        )
dk_slate_days = int(player_totals_meta.get("dk_slate_days_found") or 0)
if dates_scanned > 0 and dk_slate_days < dates_scanned:
    st.info(
        f"DK slate files were found for {dk_slate_days} of {dates_scanned} scanned dates "
        f"({start_date.isoformat()} to {end_date.isoformat()})."
    )
actual_days_found = int(actual_history_meta.get("actual_days_found") or 0)
if dates_scanned > 0 and actual_days_found < dates_scanned:
    st.info(
        f"Actual player results were found for {actual_days_found} of {dates_scanned} scanned dates "
        f"({start_date.isoformat()} to {end_date.isoformat()})."
    )

player_review_df, player_review_teams, player_review_meta = build_player_review_table(
    actual_player_history_df=actual_history_df,
    player_totals_df=player_totals_df,
    dfs_only=str(player_universe_mode).strip().lower().startswith("dfs"),
)
ownership_daily_df, ownership_daily_meta = build_ownership_accuracy_daily_frame(player_totals_df=player_totals_df)

if player_review_df.empty or not player_review_teams:
    st.warning(
        "No player review rows were found for this range. "
        "This page needs projection snapshots and at least one valid player mapping."
    )
    st.stop()

universe_label = "DFS-Relevant Only" if str(player_universe_mode).strip().lower().startswith("dfs") else "All Tracked Players"
st.caption(f"Universe: {universe_label}")
coverage_cols = [
    c
    for c in [
        "Average Ownership Season",
        "Average DK Salary This Season",
        "Average Projected Fantasy Points Season",
    ]
    if c in player_review_df.columns
]
if coverage_cols:
    total_players = int(len(player_review_df))
    own_count = (
        int(pd.to_numeric(player_review_df.get("Average Ownership Season"), errors="coerce").notna().sum())
        if "Average Ownership Season" in coverage_cols
        else 0
    )
    sal_count = (
        int(pd.to_numeric(player_review_df.get("Average DK Salary This Season"), errors="coerce").notna().sum())
        if "Average DK Salary This Season" in coverage_cols
        else 0
    )
    proj_count = (
        int(pd.to_numeric(player_review_df.get("Average Projected Fantasy Points Season"), errors="coerce").notna().sum())
        if "Average Projected Fantasy Points Season" in coverage_cols
        else 0
    )
    coverage_metrics = [
        ("Players in Table", str(total_players)),
        ("Ownership Coverage", f"{(100.0 * own_count / float(max(1, total_players))):.1f}%"),
        ("Salary Coverage", f"{(100.0 * sal_count / float(max(1, total_players))):.1f}%"),
    ]
    if "Average Projected Fantasy Points Season" in coverage_cols:
        coverage_metrics.append(
            ("Projected Points Coverage", f"{(100.0 * proj_count / float(max(1, total_players))):.1f}%")
        )
    coverage_metric_cols = st.columns(len(coverage_metrics))
    for metric_col, (label, value) in zip(coverage_metric_cols, coverage_metrics):
        metric_col.metric(label, value)

ownership_sanity_season_col = (
    "Average Projected Ownership Season"
    if "Average Projected Ownership Season" in player_review_df.columns
    else "Average Ownership Season"
)
ownership_sanity_last5_col = (
    "Average Projected Ownership Last 5 Games"
    if "Average Projected Ownership Last 5 Games" in player_review_df.columns
    else "Average Ownership Last 5 Games"
)
if all(col in player_review_df.columns for col in [ownership_sanity_season_col, ownership_sanity_last5_col]):
    own_season = pd.to_numeric(player_review_df[ownership_sanity_season_col], errors="coerce")
    own_last5 = pd.to_numeric(player_review_df[ownership_sanity_last5_col], errors="coerce")
    own_valid = own_season.notna() & own_last5.notna()
    same_mask = own_valid & ((own_season - own_last5).abs() <= 1e-9)
    low_sample_count = 0
    if "Ownership Games Season" in player_review_df.columns:
        low_sample_count = int(
            (pd.to_numeric(player_review_df["Ownership Games Season"], errors="coerce").fillna(0) <= 5).sum()
        )
    if int(own_valid.sum()) > 0:
        st.caption(
            "Ownership sanity check: "
            f"{int(same_mask.sum())}/{int(own_valid.sum())} players have identical season vs last-5 ownership averages "
            f"({ownership_sanity_season_col} vs {ownership_sanity_last5_col}); "
            f"{low_sample_count} player rows have <=5 ownership game-days."
        )

selected_team = st.selectbox("Team", options=player_review_teams, index=0, key="player_review_team_select")
past5_points_mode = st.selectbox(
    "Past-5 Fantasy Points Mode",
    options=["Average", "Median"],
    index=0,
    key="player_review_past5_points_mode",
)
past5_points_col = (
    "Median Fantasy Points Per Game Last 5" if str(past5_points_mode).strip().lower() == "median" else "Average Fantasy Points Per Game Last 5"
)
team_df = player_review_df.loc[
    player_review_df["Team"].astype(str).str.strip().str.upper() == str(selected_team).strip().upper()
].copy()

show_cols = [
    "Team",
    "Player Name",
    "Position",
    "Total Fantasy Points Season",
    past5_points_col,
    "Average Projected Fantasy Points Season",
    "Average Projected Fantasy Points Last 5",
    "Average Ownership Season",
    "Average Ownership Last 5 Games",
    "Average DK Salary This Season",
]
show_cols = [c for c in show_cols if c in team_df.columns]

if team_df.empty:
    st.info("No player rows found for the selected team in this date range.")
else:
    team_df = team_df.sort_values(
        ["Total Fantasy Points Season", past5_points_col, "Player Name"],
        ascending=[False, False, True],
        kind="stable",
    )
    st.dataframe(team_df[show_cols], hide_index=True, use_container_width=True)
    st.download_button(
        "Download Team Player Review CSV",
        data=team_df[show_cols].to_csv(index=False),
        file_name=f"player_review_{selected_team}_{start_date.isoformat()}_{end_date.isoformat()}.csv",
        mime="text/csv",
        key="download_player_review_csv",
    )

st.subheader("Ownership Calibration: Projected Points vs Projected Ownership")
projected_points_options = [
    c
    for c in ["Average Projected Fantasy Points Season", "Average Projected Fantasy Points Last 5"]
    if c in player_review_df.columns
]
projected_ownership_season_col = (
    "Average Projected Ownership Season"
    if "Average Projected Ownership Season" in player_review_df.columns
    else "Average Ownership Season"
)
projected_ownership_last5_col = (
    "Average Projected Ownership Last 5 Games"
    if "Average Projected Ownership Last 5 Games" in player_review_df.columns
    else "Average Ownership Last 5 Games"
)
projected_ownership_games_season_col = (
    "Projected Ownership Games Season"
    if "Projected Ownership Games Season" in player_review_df.columns
    else "Ownership Games Season"
)
projected_ownership_games_last5_col = (
    "Projected Ownership Games Last 5 Window"
    if "Projected Ownership Games Last 5 Window" in player_review_df.columns
    else "Ownership Games Last 5 Window"
)
if projected_ownership_season_col not in player_review_df.columns or not projected_points_options:
    st.info("This diagnostic needs projected points and ownership columns from projection snapshots.")
else:
    d1, d2, d3, d4 = st.columns(4)
    own_diag_scope = d1.selectbox(
        "Scope",
        options=["All Teams", "Selected Team"],
        index=0,
        key="player_review_own_diag_scope",
    )
    own_diag_points_col = d2.selectbox(
        "Projected Points Window",
        options=projected_points_options,
        index=0,
        key="player_review_own_diag_points_window",
    )
    own_diag_min_points = float(
        d3.number_input(
            "Min Projected Points",
            min_value=0.0,
            value=15.0,
            step=0.5,
            key="player_review_own_diag_min_points",
        )
    )
    own_diag_max_own = float(
        d4.number_input(
            "Max Avg Ownership %",
            min_value=0.0,
            max_value=100.0,
            value=100.0,
            step=0.5,
            key="player_review_own_diag_max_own",
        )
    )

    own_diag_source = player_review_df.copy()
    if str(own_diag_scope).strip().lower() == "selected team":
        own_diag_source = own_diag_source.loc[
            own_diag_source["Team"].astype(str).str.strip().str.upper() == str(selected_team).strip().upper()
        ].copy()
    own_diag_source[own_diag_points_col] = pd.to_numeric(own_diag_source[own_diag_points_col], errors="coerce")
    own_diag_source[projected_ownership_season_col] = pd.to_numeric(
        own_diag_source[projected_ownership_season_col], errors="coerce"
    )
    if "Average DK Salary This Season" in own_diag_source.columns:
        own_diag_source["Average DK Salary This Season"] = pd.to_numeric(
            own_diag_source["Average DK Salary This Season"], errors="coerce"
        )
    own_diag_df = own_diag_source.loc[
        own_diag_source[own_diag_points_col].notna()
        & own_diag_source[projected_ownership_season_col].notna()
        & (own_diag_source[own_diag_points_col] >= float(own_diag_min_points))
        & (own_diag_source[projected_ownership_season_col] <= float(own_diag_max_own))
    ].copy()
    own_diag_df = own_diag_df.sort_values(
        [own_diag_points_col, projected_ownership_season_col, "Player Name"],
        ascending=[False, True, True],
        kind="stable",
    )
    st.caption(
        "Each point is a player-level average from projection snapshots. "
        "Use this to see whether ownership is scaling appropriately with your projected points."
    )
    st.metric("Players in Diagnostic", int(len(own_diag_df)))
    if own_diag_df.empty:
        st.info("No players matched the current projected-points and ownership thresholds.")
    else:
        corr = own_diag_df[[own_diag_points_col, projected_ownership_season_col]].corr(method="spearman").iloc[0, 1]
        if pd.notna(corr):
            st.caption(
                f"Spearman correlation ({own_diag_points_col} vs {projected_ownership_season_col}): {float(corr):.3f}"
            )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        salary_vals = (
            pd.to_numeric(own_diag_df.get("Average DK Salary This Season"), errors="coerce")
            if "Average DK Salary This Season" in own_diag_df.columns
            else pd.Series(dtype=float)
        )
        if not salary_vals.empty and salary_vals.notna().any():
            scatter = ax.scatter(
                own_diag_df[own_diag_points_col],
                own_diag_df[projected_ownership_season_col],
                c=salary_vals,
                cmap="viridis",
                alpha=0.68,
                s=36,
                edgecolors="none",
            )
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Avg DK Salary")
        else:
            ax.scatter(
                own_diag_df[own_diag_points_col],
                own_diag_df[projected_ownership_season_col],
                color="#1f77b4",
                alpha=0.68,
                s=36,
                edgecolors="none",
            )
        ax.set_xlabel(own_diag_points_col)
        ax.set_ylabel("Average Ownership Season (%)")
        ax.grid(alpha=0.2)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        tier_bins = [0.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, float("inf")]
        tier_labels = ["<10", "10-15", "15-20", "20-25", "25-30", "30-40", "40+"]
        own_diag_df = own_diag_df.copy()
        own_diag_df["Projected Points Tier"] = pd.cut(
            own_diag_df[own_diag_points_col],
            bins=tier_bins,
            labels=tier_labels,
            include_lowest=True,
            right=False,
        )
        tier_summary = (
            own_diag_df.groupby("Projected Points Tier", observed=False, as_index=False)
            .agg(
                Players=("Player Name", "count"),
                AvgProjectedPoints=(own_diag_points_col, "mean"),
                AvgOwnershipSeason=(projected_ownership_season_col, "mean"),
                MedianOwnershipSeason=(projected_ownership_season_col, "median"),
                Under10OwnPct=(
                    projected_ownership_season_col,
                    lambda s: float((pd.to_numeric(s, errors="coerce") < 10.0).mean() * 100.0),
                ),
            )
            .rename(
                columns={
                    "AvgProjectedPoints": "Avg Projected Points",
                    "AvgOwnershipSeason": "Avg Ownership Season (%)",
                    "MedianOwnershipSeason": "Median Ownership Season (%)",
                    "Under10OwnPct": "Players Under 10% Own",
                }
            )
        )
        tier_summary = tier_summary.loc[tier_summary["Players"] > 0].copy()
        for col in [
            "Avg Projected Points",
            "Avg Ownership Season (%)",
            "Median Ownership Season (%)",
            "Players Under 10% Own",
        ]:
            tier_summary[col] = pd.to_numeric(tier_summary[col], errors="coerce").round(2)
        st.caption("Ownership distribution by projected-points tier")
        st.dataframe(tier_summary, hide_index=True, use_container_width=True)

        own_diag_cols = [
            "Team",
            "Player Name",
            "Position",
            own_diag_points_col,
            projected_ownership_season_col,
            projected_ownership_last5_col,
            "Average DK Salary This Season",
            "Total Fantasy Points Season",
            past5_points_col,
            projected_ownership_games_season_col,
            projected_ownership_games_last5_col,
        ]
        own_diag_cols = [c for c in own_diag_cols if c in own_diag_df.columns]
        st.dataframe(own_diag_df[own_diag_cols], hide_index=True, use_container_width=True)
        st.download_button(
            "Download Projected Points vs Ownership CSV",
            data=own_diag_df[own_diag_cols].to_csv(index=False),
            file_name=f"player_review_projected_points_vs_ownership_{start_date.isoformat()}_{end_date.isoformat()}.csv",
            mime="text/csv",
            key="download_player_review_projected_points_vs_ownership_csv",
        )

st.subheader("Ownership Accuracy: Projected vs Actual")
actual_ownership_season_col = (
    "Average Actual Ownership Season"
    if "Average Actual Ownership Season" in player_review_df.columns
    else ""
)
actual_ownership_games_season_col = (
    "Actual Ownership Games Season"
    if "Actual Ownership Games Season" in player_review_df.columns
    else ""
)
if (
    not projected_points_options
    or projected_ownership_season_col not in player_review_df.columns
    or actual_ownership_season_col == ""
):
    st.info(
        "This section needs both projected ownership and actual ownership history "
        "(actual ownership is loaded from Tournament Review contest standings)."
    )
else:
    a1, a2, a3, a4, a5 = st.columns(5)
    own_acc_scope = a1.selectbox(
        "Scope",
        options=["All Teams", "Selected Team"],
        index=0,
        key="player_review_own_acc_scope",
    )
    own_acc_points_col = a2.selectbox(
        "Projected Points Window",
        options=projected_points_options,
        index=0,
        key="player_review_own_acc_points_window",
    )
    own_acc_min_points = float(
        a3.number_input(
            "Min Projected Points",
            min_value=0.0,
            value=15.0,
            step=0.5,
            key="player_review_own_acc_min_points",
        )
    )
    own_acc_min_proj_own = float(
        a4.number_input(
            "Min Projected Own %",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.5,
            key="player_review_own_acc_min_proj_own",
        )
    )
    own_acc_top_n = int(
        a5.number_input(
            "Top Miss Rows",
            min_value=5,
            max_value=200,
            value=40,
            step=5,
            key="player_review_own_acc_top_n",
        )
    )

    own_acc_df = player_review_df.copy()
    if str(own_acc_scope).strip().lower() == "selected team":
        own_acc_df = own_acc_df.loc[
            own_acc_df["Team"].astype(str).str.strip().str.upper() == str(selected_team).strip().upper()
        ].copy()
    for col in [
        own_acc_points_col,
        projected_ownership_season_col,
        actual_ownership_season_col,
        "Average DK Salary This Season",
        projected_ownership_games_season_col,
        actual_ownership_games_season_col,
    ]:
        if col and col in own_acc_df.columns:
            own_acc_df[col] = pd.to_numeric(own_acc_df[col], errors="coerce")

    own_acc_df = own_acc_df.loc[
        own_acc_df[own_acc_points_col].notna()
        & own_acc_df[projected_ownership_season_col].notna()
        & own_acc_df[actual_ownership_season_col].notna()
        & (own_acc_df[own_acc_points_col] >= float(own_acc_min_points))
        & (own_acc_df[projected_ownership_season_col] >= float(own_acc_min_proj_own))
    ].copy()
    own_acc_df["_own_delta"] = (
        own_acc_df[actual_ownership_season_col] - own_acc_df[projected_ownership_season_col]
    )
    own_acc_df["_own_abs_delta"] = own_acc_df["_own_delta"].abs()
    own_acc_df = own_acc_df.sort_values("_own_abs_delta", ascending=False, kind="stable").reset_index(drop=True)

    st.metric("Players with Projected+Actual Ownership", int(len(own_acc_df)))
    if own_acc_df.empty:
        st.info("No players matched the current filters.")
    else:
        mae = float(pd.to_numeric(own_acc_df["_own_abs_delta"], errors="coerce").mean())
        bias = float(pd.to_numeric(own_acc_df["_own_delta"], errors="coerce").mean())
        spearman = own_acc_df[[projected_ownership_season_col, actual_ownership_season_col]].corr(method="spearman").iloc[0, 1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Ownership MAE", f"{mae:.2f}")
        m2.metric("Ownership Bias (Actual - Projected)", f"{bias:.2f}")
        m3.metric("Ownership Rank Spearman", f"{float(spearman):.3f}" if pd.notna(spearman) else "N/A")

        fig_acc, ax_acc = plt.subplots(figsize=(7.5, 7.0))
        points_color = pd.to_numeric(own_acc_df.get(own_acc_points_col), errors="coerce")
        scatter_acc = ax_acc.scatter(
            own_acc_df[projected_ownership_season_col],
            own_acc_df[actual_ownership_season_col],
            c=points_color,
            cmap="viridis",
            alpha=0.72,
            s=40,
            edgecolors="none",
        )
        min_bound = float(
            min(
                pd.to_numeric(own_acc_df[projected_ownership_season_col], errors="coerce").min(),
                pd.to_numeric(own_acc_df[actual_ownership_season_col], errors="coerce").min(),
            )
        )
        max_bound = float(
            max(
                pd.to_numeric(own_acc_df[projected_ownership_season_col], errors="coerce").max(),
                pd.to_numeric(own_acc_df[actual_ownership_season_col], errors="coerce").max(),
            )
        )
        bound_min = max(0.0, min_bound)
        bound_max = max(1.0, max_bound)
        ax_acc.plot([bound_min, bound_max], [bound_min, bound_max], linestyle="--", color="#555555", linewidth=1.4)
        ax_acc.set_xlabel("Average Projected Ownership Season (%)")
        ax_acc.set_ylabel("Average Actual Ownership Season (%)")
        ax_acc.grid(alpha=0.2)
        cbar_acc = fig_acc.colorbar(scatter_acc, ax=ax_acc)
        cbar_acc.set_label(own_acc_points_col)
        st.pyplot(fig_acc, use_container_width=True)
        plt.close(fig_acc)

        decile_labels = pd.qcut(
            own_acc_df[projected_ownership_season_col],
            q=10,
            labels=False,
            duplicates="drop",
        )
        own_acc_df["_proj_own_decile"] = decile_labels
        decile_df = own_acc_df.loc[own_acc_df["_proj_own_decile"].notna()].copy()
        if decile_df.empty:
            st.info("Projected ownership has too few distinct values to compute deciles.")
        else:
            decile_df["_proj_own_decile"] = decile_df["_proj_own_decile"].astype(int) + 1
            decile_summary = (
                decile_df.groupby("_proj_own_decile", as_index=False)
                .agg(
                    Players=("Player Name", "count"),
                    AvgProjectedOwnership=(projected_ownership_season_col, "mean"),
                    AvgActualOwnership=(actual_ownership_season_col, "mean"),
                    AvgDelta=("_own_delta", "mean"),
                    MAE=("_own_abs_delta", "mean"),
                )
                .rename(
                    columns={
                        "_proj_own_decile": "Projected Own Decile",
                        "AvgProjectedOwnership": "Avg Projected Own (%)",
                        "AvgActualOwnership": "Avg Actual Own (%)",
                        "AvgDelta": "Avg Delta (Actual - Projected)",
                        "MAE": "MAE",
                    }
                )
            )
            if projected_ownership_games_season_col and projected_ownership_games_season_col in decile_df.columns:
                decile_summary["Avg Projected Samples"] = (
                    decile_df.groupby("_proj_own_decile")[projected_ownership_games_season_col].mean().values
                )
            if actual_ownership_games_season_col and actual_ownership_games_season_col in decile_df.columns:
                decile_summary["Avg Actual Samples"] = (
                    decile_df.groupby("_proj_own_decile")[actual_ownership_games_season_col].mean().values
                )
            for col in decile_summary.columns:
                if col != "Projected Own Decile":
                    decile_summary[col] = pd.to_numeric(decile_summary[col], errors="coerce").round(2)
            st.caption("Calibration table by projected ownership decile")
            st.dataframe(decile_summary, hide_index=True, use_container_width=True)

        miss_df = own_acc_df.head(int(own_acc_top_n)).copy()
        miss_cols = [
            "Team",
            "Player Name",
            "Position",
            own_acc_points_col,
            projected_ownership_season_col,
            actual_ownership_season_col,
            "_own_delta",
            "_own_abs_delta",
            projected_ownership_games_season_col,
            actual_ownership_games_season_col,
            "Average DK Salary This Season",
            "Total Fantasy Points Season",
        ]
        miss_cols = [c for c in miss_cols if c in miss_df.columns]
        miss_view = miss_df[miss_cols].rename(
            columns={
                projected_ownership_season_col: "Avg Projected Own (%)",
                actual_ownership_season_col: "Avg Actual Own (%)",
                "_own_delta": "Delta (Actual - Projected)",
                "_own_abs_delta": "Abs Delta",
                own_acc_points_col: own_acc_points_col,
                projected_ownership_games_season_col: "Projected Own Samples",
                actual_ownership_games_season_col: "Actual Own Samples",
            }
        )
        for col in [
            "Avg Projected Own (%)",
            "Avg Actual Own (%)",
            "Delta (Actual - Projected)",
            "Abs Delta",
            "Average DK Salary This Season",
            "Total Fantasy Points Season",
        ]:
            if col in miss_view.columns:
                miss_view[col] = pd.to_numeric(miss_view[col], errors="coerce").round(2)
        st.caption(f"Biggest ownership misses (top {int(own_acc_top_n)} by absolute error)")
        st.dataframe(miss_view, hide_index=True, use_container_width=True)
        st.download_button(
            "Download Ownership Misses CSV",
            data=miss_view.to_csv(index=False),
            file_name=f"player_review_ownership_misses_{start_date.isoformat()}_{end_date.isoformat()}.csv",
            mime="text/csv",
            key="download_player_review_ownership_misses_csv",
        )

st.subheader("Slate-Level Ownership Mass")
if ownership_daily_df.empty:
    st.info("No date-level projected/actual ownership history is available for this range.")
else:
    mass_df = (
        ownership_daily_df.groupby("review_date", as_index=False)
        .agg(
            ProjectedOwnershipMass=("projected_ownership", "sum"),
            ActualOwnershipMass=("actual_ownership", "sum"),
            ProjectedPlayers=("projected_ownership", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            ActualPlayers=("actual_ownership", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            MatchedPlayers=(
                "ownership_abs_delta",
                lambda s: int(
                    ownership_daily_df.loc[s.index, "projected_ownership"].notna()
                    .mul(ownership_daily_df.loc[s.index, "actual_ownership"].notna())
                    .sum()
                ),
            ),
        )
        .sort_values("review_date", ascending=True, kind="stable")
        .reset_index(drop=True)
    )
    mass_df["Mass Delta (Actual - Projected)"] = (
        pd.to_numeric(mass_df["ActualOwnershipMass"], errors="coerce")
        - pd.to_numeric(mass_df["ProjectedOwnershipMass"], errors="coerce")
    )
    mass_df["Abs Mass Delta"] = mass_df["Mass Delta (Actual - Projected)"].abs()
    for col in [
        "ProjectedOwnershipMass",
        "ActualOwnershipMass",
        "Mass Delta (Actual - Projected)",
        "Abs Mass Delta",
    ]:
        mass_df[col] = pd.to_numeric(mass_df[col], errors="coerce").round(2)
    mass_df["Slate Date"] = pd.to_datetime(mass_df["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    mm1, mm2, mm3 = st.columns(3)
    mm1.metric("Slates with Ownership History", int(ownership_daily_meta.get("slates") or 0))
    mm2.metric("Avg Mass Delta", f"{float(pd.to_numeric(mass_df['Mass Delta (Actual - Projected)'], errors='coerce').mean()):.2f}")
    mm3.metric("Avg Matched Players / Slate", f"{float(pd.to_numeric(mass_df['MatchedPlayers'], errors='coerce').mean()):.1f}")
    ambiguous_excluded = int(ownership_daily_meta.get("ambiguous_name_day_pairs_excluded") or 0)
    if ambiguous_excluded > 0:
        st.caption(
            f"Excluded {ambiguous_excluded} ambiguous name/day pairs where the same player name mapped to multiple teams."
        )

    fig_mass, ax_mass = plt.subplots(figsize=(11, 5.5))
    ax_mass.plot(mass_df["Slate Date"], mass_df["ProjectedOwnershipMass"], label="Projected ownership mass", linewidth=2.0)
    ax_mass.plot(mass_df["Slate Date"], mass_df["ActualOwnershipMass"], label="Actual ownership mass", linewidth=2.0)
    ax_mass.set_ylabel("Total Ownership %")
    ax_mass.set_xlabel("Slate Date")
    ax_mass.grid(alpha=0.2)
    ax_mass.legend()
    ax_mass.tick_params(axis="x", rotation=45)
    fig_mass.tight_layout()
    st.pyplot(fig_mass, use_container_width=True)
    plt.close(fig_mass)

    worst_mass_df = mass_df.sort_values("Abs Mass Delta", ascending=False, kind="stable").head(20).copy()
    mass_show_cols = [
        "Slate Date",
        "ProjectedOwnershipMass",
        "ActualOwnershipMass",
        "Mass Delta (Actual - Projected)",
        "Abs Mass Delta",
        "ProjectedPlayers",
        "ActualPlayers",
        "MatchedPlayers",
    ]
    worst_mass_df = worst_mass_df[mass_show_cols].rename(
        columns={
            "ProjectedOwnershipMass": "Projected Mass",
            "ActualOwnershipMass": "Actual Mass",
        }
    )
    st.caption("Worst slate-level ownership mass misses")
    st.dataframe(worst_mass_df, hide_index=True, use_container_width=True)
    st.download_button(
        "Download Ownership Mass by Slate CSV",
        data=mass_df[
            [
                "Slate Date",
                "ProjectedOwnershipMass",
                "ActualOwnershipMass",
                "Mass Delta (Actual - Projected)",
                "Abs Mass Delta",
                "ProjectedPlayers",
                "ActualPlayers",
                "MatchedPlayers",
            ]
        ].rename(
            columns={
                "ProjectedOwnershipMass": "Projected Mass",
                "ActualOwnershipMass": "Actual Mass",
            }
        ).to_csv(index=False),
        file_name=f"player_review_ownership_mass_{start_date.isoformat()}_{end_date.isoformat()}.csv",
        mime="text/csv",
        key="download_player_review_ownership_mass_csv",
    )

st.subheader("False Chalk / Missed Chalk")
if ownership_daily_df.empty:
    st.info("No date-level ownership rows are available for false-chalk diagnostics.")
else:
    chalk_dates = sorted(
        [
            str(x)
            for x in pd.to_datetime(ownership_daily_df["review_date"], errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
            if str(x).strip()
        ],
        reverse=True,
    )
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    chalk_scope = c1.selectbox(
        "Scope",
        options=["All Teams", "Selected Team"],
        index=0,
        key="player_review_false_chalk_scope",
    )
    chalk_date = c2.selectbox(
        "Slate Date",
        options=["All Dates"] + chalk_dates,
        index=0,
        key="player_review_false_chalk_date",
    )
    chalk_min_points = float(
        c3.number_input(
            "Min Projected Points",
            min_value=0.0,
            value=15.0,
            step=0.5,
            key="player_review_false_chalk_min_points",
        )
    )
    chalk_high_threshold = float(
        c4.number_input(
            "Chalk Threshold %",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=0.5,
            key="player_review_false_chalk_high_threshold",
        )
    )
    chalk_low_threshold = float(
        c5.number_input(
            "Low Threshold %",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key="player_review_false_chalk_low_threshold",
        )
    )
    chalk_row_limit = int(
        c6.number_input(
            "Rows",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            key="player_review_false_chalk_rows",
        )
    )

    chalk_df = ownership_daily_df.copy()
    if str(chalk_scope).strip().lower() == "selected team":
        chalk_df = chalk_df.loc[
            chalk_df["Team"].astype(str).str.strip().str.upper() == str(selected_team).strip().upper()
        ].copy()
    if str(chalk_date).strip().lower() != "all dates":
        chalk_df = chalk_df.loc[
            pd.to_datetime(chalk_df["review_date"], errors="coerce").dt.strftime("%Y-%m-%d") == str(chalk_date).strip()
        ].copy()
    for col in [
        "projected_dk_points",
        "projected_ownership",
        "actual_ownership",
        "salary",
        "projected_source_rows",
        "actual_source_rows",
    ]:
        if col in chalk_df.columns:
            chalk_df[col] = pd.to_numeric(chalk_df[col], errors="coerce")
    chalk_df = chalk_df.loc[
        chalk_df["projected_dk_points"].notna()
        & chalk_df["projected_ownership"].notna()
        & chalk_df["actual_ownership"].notna()
        & (chalk_df["projected_dk_points"] >= float(chalk_min_points))
    ].copy()
    chalk_df["Slate Date"] = pd.to_datetime(chalk_df["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    false_chalk_df = chalk_df.loc[
        (chalk_df["projected_ownership"] >= float(chalk_high_threshold))
        & (chalk_df["actual_ownership"] <= float(chalk_low_threshold))
    ].sort_values(["ownership_delta", "projected_ownership"], ascending=[True, False], kind="stable").head(chalk_row_limit)
    missed_chalk_df = chalk_df.loc[
        (chalk_df["projected_ownership"] <= float(chalk_low_threshold))
        & (chalk_df["actual_ownership"] >= float(chalk_high_threshold))
    ].sort_values(["ownership_delta", "actual_ownership"], ascending=[False, False], kind="stable").head(chalk_row_limit)

    fc_metric, mc_metric = st.columns(2)
    fc_metric.metric("False Chalk Rows", int(len(false_chalk_df)))
    mc_metric.metric("Missed Chalk Rows", int(len(missed_chalk_df)))

    chalk_cols = [
        "Slate Date",
        "Team",
        "Player Name",
        "Position",
        "projected_dk_points",
        "projected_ownership",
        "actual_ownership",
        "ownership_delta",
        "ownership_abs_delta",
        "salary",
        "projected_source_rows",
        "actual_source_rows",
    ]
    chalk_cols = [c for c in chalk_cols if c in chalk_df.columns]

    false_chalk_view = false_chalk_df[chalk_cols].rename(
        columns={
            "projected_dk_points": "Projected Points",
            "projected_ownership": "Projected Own (%)",
            "actual_ownership": "Actual Own (%)",
            "ownership_delta": "Delta (Actual - Projected)",
            "ownership_abs_delta": "Abs Delta",
            "salary": "Avg DK Salary",
            "projected_source_rows": "Projected Rows",
            "actual_source_rows": "Actual Rows",
        }
    )
    missed_chalk_view = missed_chalk_df[chalk_cols].rename(
        columns={
            "projected_dk_points": "Projected Points",
            "projected_ownership": "Projected Own (%)",
            "actual_ownership": "Actual Own (%)",
            "ownership_delta": "Delta (Actual - Projected)",
            "ownership_abs_delta": "Abs Delta",
            "salary": "Avg DK Salary",
            "projected_source_rows": "Projected Rows",
            "actual_source_rows": "Actual Rows",
        }
    )
    for frame in [false_chalk_view, missed_chalk_view]:
        for col in [
            "Projected Points",
            "Projected Own (%)",
            "Actual Own (%)",
            "Delta (Actual - Projected)",
            "Abs Delta",
            "Avg DK Salary",
        ]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce").round(2)

    fc_col, mc_col = st.columns(2)
    fc_col.caption(
        f"False chalk: projected >= {chalk_high_threshold:.1f}% and actual <= {chalk_low_threshold:.1f}%"
    )
    if false_chalk_view.empty:
        fc_col.info("No false-chalk rows matched the current filters.")
    else:
        fc_col.dataframe(false_chalk_view, hide_index=True, use_container_width=True)

    mc_col.caption(
        f"Missed chalk: projected <= {chalk_low_threshold:.1f}% and actual >= {chalk_high_threshold:.1f}%"
    )
    if missed_chalk_view.empty:
        mc_col.info("No missed-chalk rows matched the current filters.")
    else:
        mc_col.dataframe(missed_chalk_view, hide_index=True, use_container_width=True)

    false_chalk_download = false_chalk_view.copy()
    false_chalk_download["Classification"] = "False Chalk"
    missed_chalk_download = missed_chalk_view.copy()
    missed_chalk_download["Classification"] = "Missed Chalk"
    combined_chalk_download = pd.concat(
        [false_chalk_download, missed_chalk_download],
        ignore_index=True,
    )
    st.download_button(
        "Download False Chalk / Missed Chalk CSV",
        data=combined_chalk_download.to_csv(index=False),
        file_name=f"player_review_false_missed_chalk_{start_date.isoformat()}_{end_date.isoformat()}.csv",
        mime="text/csv",
        key="download_player_review_false_missed_chalk_csv",
    )

st.subheader("Low-Owned High-Scoring Targets")
if not all(col in player_review_df.columns for col in [past5_points_col, "Average Ownership Season"]):
    st.info("This filter needs both last-5 fantasy points and average ownership columns.")
else:
    default_min_past5 = 15.0

    t1, t2, t3 = st.columns(3)
    leverage_scope = t1.selectbox(
        "Scope",
        options=["All Teams", "Selected Team"],
        index=0,
        key="player_review_leverage_scope",
    )
    leverage_own_cap = float(
        t2.number_input(
            "Max Avg Ownership %",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            key="player_review_leverage_own_cap",
        )
    )
    leverage_min_past5 = float(
        t3.number_input(
            f"Min {past5_points_col}",
            min_value=0.0,
            value=default_min_past5,
            step=0.5,
            key="player_review_leverage_min_past5",
        )
    )

    leverage_source = player_review_df.copy()
    if str(leverage_scope).strip().lower() == "selected team":
        leverage_source = leverage_source.loc[
            leverage_source["Team"].astype(str).str.strip().str.upper() == str(selected_team).strip().upper()
        ].copy()

    leverage_source[past5_points_col] = pd.to_numeric(leverage_source[past5_points_col], errors="coerce")
    leverage_source["Average Ownership Season"] = pd.to_numeric(
        leverage_source["Average Ownership Season"], errors="coerce"
    )
    leverage_df = leverage_source.loc[
        leverage_source[past5_points_col].notna()
        & leverage_source["Average Ownership Season"].notna()
        & (leverage_source[past5_points_col] >= float(leverage_min_past5))
        & (leverage_source["Average Ownership Season"] < float(leverage_own_cap))
    ].copy()
    leverage_df = leverage_df.sort_values(
        [past5_points_col, "Average Ownership Season", "Total Fantasy Points Season", "Player Name"],
        ascending=[False, True, False, True],
        kind="stable",
    ).reset_index(drop=True)

    leverage_cols = [
        "Team",
        "Player Name",
        "Position",
        past5_points_col,
        "Average Ownership Season",
        "Average Ownership Last 5 Games",
        "Total Fantasy Points Season",
        "Average DK Salary This Season",
        "Ownership Games Season",
        "Ownership Games Last 5 Window",
    ]
    leverage_cols = [c for c in leverage_cols if c in leverage_df.columns]
    st.caption(
        "Players meeting: "
        f"`{past5_points_col} >= {leverage_min_past5:.1f}` and "
        f"`Average Ownership Season < {leverage_own_cap:.1f}%`."
    )
    st.metric("Matching Players", int(len(leverage_df)))
    if leverage_df.empty:
        st.info("No players matched the current thresholds.")
    else:
        st.dataframe(leverage_df[leverage_cols], hide_index=True, use_container_width=True)
        st.download_button(
            "Download Low-Owned High-Scoring Targets CSV",
            data=leverage_df[leverage_cols].to_csv(index=False),
            file_name=f"player_review_low_owned_targets_{start_date.isoformat()}_{end_date.isoformat()}.csv",
            mime="text/csv",
            key="download_player_review_low_owned_targets_csv",
        )

st.subheader("All Players")
all_players_df = player_review_df.copy()
all_players_df = all_players_df.sort_values(
    ["Total Fantasy Points Season", past5_points_col, "Team", "Player Name"],
    ascending=[False, False, True, True],
    kind="stable",
)
all_players_show_cols = show_cols + [
    "Ownership Games Season",
    "Ownership Games Last 5 Window",
    "Minutes Variance Season",
    "Minutes Variance Last 5 Games",
    "Fantasy Points Variance Season",
    "Fantasy Points Variance Last 5 Games",
    "Ownership Variance Season",
    "Ownership Variance Last 5 Games",
    "Salary Variance Season",
    "Salary Variance Last 5 Games",
]
all_players_show_cols = [c for c in all_players_show_cols if c in all_players_df.columns]
st.dataframe(all_players_df[all_players_show_cols], hide_index=True, use_container_width=True)
st.download_button(
    "Download All Players Review CSV",
    data=all_players_df[all_players_show_cols].to_csv(index=False),
    file_name=f"player_review_all_teams_{start_date.isoformat()}_{end_date.isoformat()}.csv",
    mime="text/csv",
    key="download_player_review_all_players_csv",
)

if not bool(player_review_meta.get("has_points")):
    st.caption("Note: Total fantasy points are unavailable because no matched actual DK points were found.")
if not bool(player_review_meta.get("has_ownership")):
    st.caption("Note: Ownership columns are unavailable because projection ownership values were not found.")
if not bool(player_review_meta.get("has_projected_ownership")):
    st.caption("Note: Projected ownership averages are unavailable for this range.")
if not bool(player_review_meta.get("has_actual_ownership")):
    st.caption(
        "Note: Actual ownership averages are unavailable; upload Tournament Review contest standings for this range."
    )
if not bool(player_review_meta.get("has_salary")):
    st.caption("Note: Average DK salary is unavailable because salary values were not found.")
if not bool(player_review_meta.get("has_projected_points")):
    st.caption("Note: Projected fantasy points are unavailable because projection point columns were not found.")
