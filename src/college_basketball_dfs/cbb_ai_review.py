from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


AI_REVIEW_SCHEMA_VERSION = "v1"
GAME_SLATE_AI_REVIEW_SCHEMA_VERSION = "v1_game_slate"
MARKET_CORRELATION_AI_REVIEW_SCHEMA_VERSION = "v1_market_correlation"

AI_REVIEW_SYSTEM_PROMPT = (
    "You are an expert college basketball DFS review analyst. "
    "Use only evidence in the provided JSON packet. "
    "Prioritize actionable recommendations that can improve projection calibration, "
    "ownership calibration, and lineup construction, with a GPP-first focus on finding differentiated edges. "
    "When market/odds evidence is available, explain how it should translate into ownership and stack strategy. "
    "If confidence is low, state uncertainty explicitly."
)

GAME_SLATE_AI_REVIEW_SYSTEM_PROMPT = (
    "You are an expert college basketball DFS game-slate analyst. "
    "Use only the evidence in the JSON packet to identify high-upside GPP game stacks, "
    "team cores, bring-back options, winner confidence, and leverage/upset angles. "
    "Rank recommendations by expected DFS impact and state confidence per call."
)

MARKET_CORRELATION_AI_REVIEW_SYSTEM_PROMPT = (
    "You are an expert DFS market-calibration analyst. "
    "Use only evidence in the provided JSON packet to explain how market signals "
    "(totals/spreads/vegas-derived features) correlate with ownership and points outcomes. "
    "Recommend concrete projection and ownership calibration changes by segment, with measurable success criteria. "
    "Focus on GPP edge: where to be different from field ownership and how that changes stack construction."
)

DEFAULT_OPENAI_REVIEW_MODEL = "gpt-5-mini"
OPENAI_REVIEW_MODEL_FALLBACKS = (
    "gpt-5-mini",
    "gpt-5",
    "gpt-4.1-mini",
    "gpt-4.1",
)
NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


class OpenAIReviewError(RuntimeError):
    """Raised when OpenAI review generation fails."""


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any, default: int = 0) -> int:
    as_float = _to_float(value)
    if as_float is None:
        return int(default)
    return int(as_float)


def _norm_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _norm_name_loose(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in NAME_SUFFIX_TOKENS]
    return "".join(tokens)


def _to_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return 0.0
    return float(cleaned.mean())


def _safe_rmse(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return 0.0
    return float((cleaned.pow(2).mean()) ** 0.5)


def _safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    if left.empty or right.empty:
        return None
    frame = pd.DataFrame({"l": pd.to_numeric(left, errors="coerce"), "r": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(frame) < 3:
        return None
    corr = frame["l"].rank(method="average").corr(frame["r"].rank(method="average"))
    if pd.isna(corr):
        return None
    return float(corr)


def _safe_pearson(left: pd.Series, right: pd.Series) -> float | None:
    if left.empty or right.empty:
        return None
    frame = pd.DataFrame({"l": pd.to_numeric(left, errors="coerce"), "r": pd.to_numeric(right, errors="coerce")}).dropna()
    if len(frame) < 3:
        return None
    corr = frame["l"].corr(frame["r"])
    if pd.isna(corr):
        return None
    return float(corr)


def _top_records(
    df: pd.DataFrame,
    keep_cols: list[str],
    sort_col: str,
    ascending: bool,
    limit: int,
) -> list[dict[str, Any]]:
    if df.empty or sort_col not in df.columns:
        return []
    out = df.copy()
    out[sort_col] = pd.to_numeric(out[sort_col], errors="coerce")
    out = out.loc[out[sort_col].notna()]
    if out.empty:
        return []
    out = out.sort_values(sort_col, ascending=ascending).head(limit)
    use_cols = [c for c in keep_cols if c in out.columns]
    return out[use_cols].to_dict(orient="records")


def _build_attention_candidates(
    projection_comparison_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    limit: int = 15,
) -> list[dict[str, Any]]:
    if projection_comparison_df.empty:
        return []

    proj = projection_comparison_df.copy()
    proj["Name"] = proj.get("Name", "").astype(str).str.strip()
    proj["TeamAbbrev"] = proj.get("TeamAbbrev", "").astype(str).str.strip().str.upper()
    proj["name_key"] = proj["Name"].map(_norm_name)
    proj["name_key_loose"] = proj["Name"].map(_norm_name_loose)
    proj["abs_projection_error"] = pd.to_numeric(proj.get("blend_error"), errors="coerce").abs()

    keep_proj_cols = [
        "Name",
        "TeamAbbrev",
        "name_key",
        "name_key_loose",
        "Position",
        "Salary",
        "blended_projection",
        "actual_dk_points",
        "blend_error",
        "abs_projection_error",
    ]
    proj = proj[[c for c in keep_proj_cols if c in proj.columns]].copy()

    if not exposure_df.empty:
        exp = exposure_df.copy()
        exp["Name"] = exp.get("Name", "").astype(str).str.strip()
        exp["TeamAbbrev"] = exp.get("TeamAbbrev", "").astype(str).str.strip().str.upper()
        exp["name_key"] = exp["Name"].map(_norm_name)
        exp["name_key_loose"] = exp["Name"].map(_norm_name_loose)
        exp["abs_ownership_error"] = pd.to_numeric(exp.get("ownership_diff_vs_proj"), errors="coerce").abs()
        exp["field_ownership_pct"] = pd.to_numeric(exp.get("field_ownership_pct"), errors="coerce")
        exp["projected_ownership"] = pd.to_numeric(exp.get("projected_ownership"), errors="coerce")
        exp["actual_ownership_from_file"] = pd.to_numeric(exp.get("actual_ownership_from_file"), errors="coerce")
        exp = exp[
            [
                c
                for c in [
                    "Name",
                    "TeamAbbrev",
                    "name_key",
                    "name_key_loose",
                    "field_ownership_pct",
                    "projected_ownership",
                    "actual_ownership_from_file",
                    "ownership_diff_vs_proj",
                    "abs_ownership_error",
                ]
                if c in exp.columns
            ]
        ]
        exp_strict = exp.drop_duplicates(["Name", "TeamAbbrev"])
        exp_by_name = exp.sort_values("field_ownership_pct", ascending=False).drop_duplicates(["name_key"])
        exp_by_name_loose = exp.sort_values("field_ownership_pct", ascending=False).drop_duplicates(["name_key_loose"])

        merged = proj.merge(
            exp_strict[
                [
                    "Name",
                    "TeamAbbrev",
                    "field_ownership_pct",
                    "projected_ownership",
                    "actual_ownership_from_file",
                    "ownership_diff_vs_proj",
                    "abs_ownership_error",
                ]
            ],
            on=["Name", "TeamAbbrev"],
            how="left",
        )
        merged = merged.merge(
            exp_by_name[
                [
                    "name_key",
                    "field_ownership_pct",
                    "projected_ownership",
                    "actual_ownership_from_file",
                    "ownership_diff_vs_proj",
                    "abs_ownership_error",
                ]
            ],
            on="name_key",
            how="left",
            suffixes=("", "_by_name"),
        )
        merged = merged.merge(
            exp_by_name_loose[
                [
                    "name_key_loose",
                    "field_ownership_pct",
                    "projected_ownership",
                    "actual_ownership_from_file",
                    "ownership_diff_vs_proj",
                    "abs_ownership_error",
                ]
            ],
            on="name_key_loose",
            how="left",
            suffixes=("", "_by_name_loose"),
        )
        for col in [
            "field_ownership_pct",
            "projected_ownership",
            "actual_ownership_from_file",
            "ownership_diff_vs_proj",
            "abs_ownership_error",
        ]:
            by_name_col = f"{col}_by_name"
            by_name_loose_col = f"{col}_by_name_loose"
            merged[col] = pd.to_numeric(merged[col], errors="coerce")
            if by_name_col in merged.columns:
                merged[col] = merged[col].where(merged[col].notna(), pd.to_numeric(merged[by_name_col], errors="coerce"))
            if by_name_loose_col in merged.columns:
                merged[col] = merged[col].where(
                    merged[col].notna(),
                    pd.to_numeric(merged[by_name_loose_col], errors="coerce"),
                )
        merged = merged.drop(
            columns=[
                "field_ownership_pct_by_name",
                "projected_ownership_by_name",
                "actual_ownership_from_file_by_name",
                "ownership_diff_vs_proj_by_name",
                "abs_ownership_error_by_name",
                "field_ownership_pct_by_name_loose",
                "projected_ownership_by_name_loose",
                "actual_ownership_from_file_by_name_loose",
                "ownership_diff_vs_proj_by_name_loose",
                "abs_ownership_error_by_name_loose",
            ],
            errors="ignore",
        )
    else:
        merged = proj.copy()
        merged["field_ownership_pct"] = pd.NA
        merged["projected_ownership"] = pd.NA
        merged["actual_ownership_from_file"] = pd.NA
        merged["ownership_diff_vs_proj"] = pd.NA
        merged["abs_ownership_error"] = pd.NA

    merged["abs_projection_error"] = pd.to_numeric(merged["abs_projection_error"], errors="coerce").fillna(0.0)
    merged["abs_ownership_error"] = pd.to_numeric(merged["abs_ownership_error"], errors="coerce").fillna(0.0)
    merged["field_ownership_pct"] = pd.to_numeric(merged["field_ownership_pct"], errors="coerce").fillna(0.0)

    merged["contest_impact_weight"] = (merged["field_ownership_pct"] / 5.0).clip(lower=1.0)
    merged["attention_index"] = (
        merged["abs_projection_error"] * (1.0 + (merged["abs_ownership_error"] / 10.0)) * merged["contest_impact_weight"]
    )
    merged = merged.sort_values("attention_index", ascending=False).head(limit).reset_index(drop=True)

    cols = [
        "Name",
        "TeamAbbrev",
        "Position",
        "Salary",
        "blended_projection",
        "actual_dk_points",
        "blend_error",
        "projected_ownership",
        "actual_ownership_from_file",
        "ownership_diff_vs_proj",
        "field_ownership_pct",
        "attention_index",
    ]
    return merged[[c for c in cols if c in merged.columns]].to_dict(orient="records")


def build_daily_ai_review_packet(
    *,
    review_date: str,
    contest_id: str,
    projection_comparison_df: pd.DataFrame,
    entries_df: pd.DataFrame,
    exposure_df: pd.DataFrame,
    phantom_summary_df: pd.DataFrame | None = None,
    phantom_lineups_df: pd.DataFrame | None = None,
    adjustment_factors_df: pd.DataFrame | None = None,
    focus_limit: int = 15,
) -> dict[str, Any]:
    projection_comparison_df = projection_comparison_df.copy() if isinstance(projection_comparison_df, pd.DataFrame) else pd.DataFrame()
    entries_df = entries_df.copy() if isinstance(entries_df, pd.DataFrame) else pd.DataFrame()
    exposure_df = exposure_df.copy() if isinstance(exposure_df, pd.DataFrame) else pd.DataFrame()
    phantom_summary_df = phantom_summary_df.copy() if isinstance(phantom_summary_df, pd.DataFrame) else pd.DataFrame()
    phantom_lineups_df = phantom_lineups_df.copy() if isinstance(phantom_lineups_df, pd.DataFrame) else pd.DataFrame()
    adjustment_factors_df = adjustment_factors_df.copy() if isinstance(adjustment_factors_df, pd.DataFrame) else pd.DataFrame()

    blend_error = _to_num_series(projection_comparison_df, "blend_error")
    our_error = _to_num_series(projection_comparison_df, "our_error")
    vegas_error = _to_num_series(projection_comparison_df, "vegas_error")
    blended_proj = _to_num_series(projection_comparison_df, "blended_projection")
    actual_points = _to_num_series(projection_comparison_df, "actual_dk_points")

    ownership_error = _to_num_series(exposure_df, "ownership_diff_vs_proj")
    projected_ownership = _to_num_series(exposure_df, "projected_ownership")
    actual_ownership = _to_num_series(exposure_df, "actual_ownership_from_file")

    lineup_actual_minus_proj = _to_num_series(phantom_lineups_df, "actual_minus_projected")
    lineup_actual = _to_num_series(phantom_lineups_df, "actual_points")
    lineup_would_beat = _to_num_series(phantom_lineups_df, "would_beat_pct")
    lineup_salary_left = _to_num_series(phantom_lineups_df, "salary_left")

    lineup_scored = int(len(phantom_lineups_df))
    lineup_avg_delta = _safe_mean(lineup_actual_minus_proj)
    lineup_avg_beat = _safe_mean(lineup_would_beat)
    lineup_best_actual = round(float(lineup_actual.max()), 4) if lineup_actual.notna().any() else 0.0
    lineup_avg_salary_left = _safe_mean(lineup_salary_left)

    if lineup_scored == 0 and not phantom_summary_df.empty:
        lineups_col = _to_num_series(phantom_summary_df, "lineups")
        delta_col = _to_num_series(phantom_summary_df, "avg_actual_minus_projected")
        beat_col = _to_num_series(phantom_summary_df, "avg_would_beat_pct")
        best_col = _to_num_series(phantom_summary_df, "best_actual_points")
        salary_col = _to_num_series(phantom_summary_df, "avg_salary_left")
        total_lineups = float(lineups_col.fillna(0.0).sum())
        if total_lineups > 0:
            lineup_scored = int(total_lineups)
            lineup_avg_delta = float((delta_col.fillna(0.0) * lineups_col.fillna(0.0)).sum() / total_lineups)
            lineup_avg_beat = float((beat_col.fillna(0.0) * lineups_col.fillna(0.0)).sum() / total_lineups)
            lineup_avg_salary_left = float((salary_col.fillna(0.0) * lineups_col.fillna(0.0)).sum() / total_lineups)
        if best_col.notna().any():
            lineup_best_actual = float(best_col.max())

    projection_quality = {
        "matched_rows": int(actual_points.notna().sum()),
        "blend_mae": round(_safe_mean(blend_error.abs()), 4),
        "blend_rmse": round(_safe_rmse(blend_error), 4),
        "our_mae": round(_safe_mean(our_error.abs()), 4),
        "vegas_mae": round(_safe_mean(vegas_error.abs()), 4),
        "blended_rank_spearman": _safe_spearman(blended_proj, actual_points),
    }
    ownership_quality = {
        "matched_rows": int(actual_ownership.notna().sum()),
        "ownership_mae": round(_safe_mean(ownership_error.abs()), 4),
        "ownership_rank_spearman": _safe_spearman(projected_ownership, actual_ownership),
    }
    lineup_quality = {
        "lineups_scored": int(lineup_scored),
        "avg_actual_minus_projected": round(float(lineup_avg_delta), 4),
        "best_actual_points": round(float(lineup_best_actual), 4),
        "avg_would_beat_pct": round(float(lineup_avg_beat), 4),
        "avg_salary_left": round(float(lineup_avg_salary_left), 4),
    }
    field_quality = {
        "field_entries": int(len(entries_df)),
        "avg_salary_left": round(_safe_mean(_to_num_series(entries_df, "salary_left")), 4),
        "top10_avg_salary_left": round(_safe_mean(_to_num_series(entries_df.nsmallest(10, "Rank") if "Rank" in entries_df.columns else entries_df, "salary_left")), 4),
        "avg_max_team_stack": round(_safe_mean(_to_num_series(entries_df, "max_team_stack")), 4),
        "avg_max_game_stack": round(_safe_mean(_to_num_series(entries_df, "max_game_stack")), 4),
    }

    top_projection_misses = _top_records(
        projection_comparison_df.assign(abs_blend_error=_to_num_series(projection_comparison_df, "blend_error").abs()),
        keep_cols=[
            "ID",
            "Name",
            "TeamAbbrev",
            "Position",
            "Salary",
            "blended_projection",
            "actual_dk_points",
            "blend_error",
            "abs_blend_error",
        ],
        sort_col="abs_blend_error",
        ascending=False,
        limit=focus_limit,
    )
    top_ownership_misses = _top_records(
        exposure_df.assign(abs_ownership_error=_to_num_series(exposure_df, "ownership_diff_vs_proj").abs()),
        keep_cols=[
            "Name",
            "TeamAbbrev",
            "field_ownership_pct",
            "projected_ownership",
            "actual_ownership_from_file",
            "ownership_diff_vs_proj",
            "abs_ownership_error",
            "final_dk_points",
        ],
        sort_col="abs_ownership_error",
        ascending=False,
        limit=focus_limit,
    )
    attention = _build_attention_candidates(
        projection_comparison_df=projection_comparison_df,
        exposure_df=exposure_df,
        limit=focus_limit,
    )
    projection_with_own = projection_comparison_df.copy()
    projection_with_own["Name"] = projection_with_own.get("Name", "").astype(str).str.strip()
    projection_with_own["TeamAbbrev"] = projection_with_own.get("TeamAbbrev", "").astype(str).str.strip().str.upper()
    projection_with_own["name_key"] = projection_with_own["Name"].map(_norm_name)
    projection_with_own["name_key_loose"] = projection_with_own["Name"].map(_norm_name_loose)
    projection_with_own["blend_error"] = pd.to_numeric(projection_with_own.get("blend_error"), errors="coerce")
    projection_with_own["actual_dk_points"] = pd.to_numeric(projection_with_own.get("actual_dk_points"), errors="coerce")

    if not exposure_df.empty:
        exposure_join = exposure_df.copy()
        exposure_join["Name"] = exposure_join.get("Name", "").astype(str).str.strip()
        exposure_join["TeamAbbrev"] = exposure_join.get("TeamAbbrev", "").astype(str).str.strip().str.upper()
        exposure_join["name_key"] = exposure_join["Name"].map(_norm_name)
        exposure_join["name_key_loose"] = exposure_join["Name"].map(_norm_name_loose)
        exposure_join["field_ownership_pct"] = pd.to_numeric(exposure_join.get("field_ownership_pct"), errors="coerce")
        exposure_join["projected_ownership"] = pd.to_numeric(exposure_join.get("projected_ownership"), errors="coerce")
        exposure_join["actual_ownership_from_file"] = pd.to_numeric(
            exposure_join.get("actual_ownership_from_file"), errors="coerce"
        )
        use_cols = [
            "Name",
            "TeamAbbrev",
            "name_key",
            "name_key_loose",
            "field_ownership_pct",
            "projected_ownership",
            "actual_ownership_from_file",
            "ownership_diff_vs_proj",
            "final_dk_points",
        ]
        exposure_join = exposure_join[[c for c in use_cols if c in exposure_join.columns]]
        exposure_strict = exposure_join.drop_duplicates(["Name", "TeamAbbrev"])
        exposure_by_name = exposure_join.sort_values("field_ownership_pct", ascending=False).drop_duplicates(["name_key"])
        exposure_by_name_loose = exposure_join.sort_values("field_ownership_pct", ascending=False).drop_duplicates(
            ["name_key_loose"]
        )
        strict_merge_cols = [
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
            if c in exposure_strict.columns
        ]
        name_merge_cols = [
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
        loose_merge_cols = [
            c
            for c in [
                "name_key_loose",
                "field_ownership_pct",
                "projected_ownership",
                "actual_ownership_from_file",
                "ownership_diff_vs_proj",
                "final_dk_points",
            ]
            if c in exposure_by_name_loose.columns
        ]
        projection_with_own = projection_with_own.merge(
            exposure_strict[strict_merge_cols],
            on=["Name", "TeamAbbrev"],
            how="left",
        )
        projection_with_own = projection_with_own.merge(
            exposure_by_name[name_merge_cols],
            on="name_key",
            how="left",
            suffixes=("", "_by_name"),
        )
        projection_with_own = projection_with_own.merge(
            exposure_by_name_loose[loose_merge_cols],
            on="name_key_loose",
            how="left",
            suffixes=("", "_by_name_loose"),
        )
        for col in [
            "field_ownership_pct",
            "projected_ownership",
            "actual_ownership_from_file",
            "ownership_diff_vs_proj",
            "final_dk_points",
        ]:
            by_name_col = f"{col}_by_name"
            by_name_loose_col = f"{col}_by_name_loose"
            if col not in projection_with_own.columns:
                projection_with_own[col] = pd.NA
            projection_with_own[col] = pd.to_numeric(projection_with_own[col], errors="coerce")
            if by_name_col in projection_with_own.columns:
                projection_with_own[col] = projection_with_own[col].where(
                    projection_with_own[col].notna(),
                    pd.to_numeric(projection_with_own[by_name_col], errors="coerce"),
                )
            if by_name_loose_col in projection_with_own.columns:
                projection_with_own[col] = projection_with_own[col].where(
                    projection_with_own[col].notna(),
                    pd.to_numeric(projection_with_own[by_name_loose_col], errors="coerce"),
                )
    for col in [
        "field_ownership_pct",
        "projected_ownership",
        "actual_ownership_from_file",
        "ownership_diff_vs_proj",
        "final_dk_points",
    ]:
        if col not in projection_with_own.columns:
            projection_with_own[col] = pd.NA
    projection_with_own["true_low_own_smash_flag"] = (
        pd.to_numeric(projection_with_own["actual_dk_points"], errors="coerce").fillna(0.0) >= 35.0
    ) & (
        pd.to_numeric(projection_with_own["field_ownership_pct"], errors="coerce").fillna(999.0) <= 10.0
    )
    projection_with_own["projected_low_own_smash_flag"] = (
        pd.to_numeric(projection_with_own["actual_dk_points"], errors="coerce").fillna(0.0) >= 35.0
    ) & (
        pd.to_numeric(projection_with_own["projected_ownership"], errors="coerce").fillna(999.0) <= 10.0
    )
    projection_with_own["ownership_surprise_smash_flag"] = (
        projection_with_own["projected_low_own_smash_flag"].fillna(False)
        & (
            pd.to_numeric(projection_with_own["field_ownership_pct"], errors="coerce").fillna(0.0) > 10.0
        )
    )
    underprojected_ceiling_top = _top_records(
        projection_with_own.loc[pd.to_numeric(projection_with_own["blend_error"], errors="coerce") > 0].copy(),
        keep_cols=[
            "ID",
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
        ],
        sort_col="blend_error",
        ascending=False,
        limit=focus_limit,
    )
    low_own_smash_top = _top_records(
        projection_with_own.loc[
            projection_with_own["true_low_own_smash_flag"].fillna(False)
        ].copy(),
        keep_cols=[
            "ID",
            "Name",
            "TeamAbbrev",
            "Position",
            "Salary",
            "actual_dk_points",
            "blended_projection",
            "blend_error",
            "field_ownership_pct",
            "projected_ownership",
            "actual_ownership_from_file",
            "true_low_own_smash_flag",
            "projected_low_own_smash_flag",
            "ownership_surprise_smash_flag",
        ],
        sort_col="actual_dk_points",
        ascending=False,
        limit=focus_limit,
    )
    ownership_surprise_smash_top = _top_records(
        projection_with_own.loc[
            projection_with_own["ownership_surprise_smash_flag"].fillna(False)
        ].copy(),
        keep_cols=[
            "ID",
            "Name",
            "TeamAbbrev",
            "Position",
            "Salary",
            "actual_dk_points",
            "blended_projection",
            "blend_error",
            "field_ownership_pct",
            "projected_ownership",
            "actual_ownership_from_file",
            "true_low_own_smash_flag",
            "projected_low_own_smash_flag",
            "ownership_surprise_smash_flag",
        ],
        sort_col="actual_dk_points",
        ascending=False,
        limit=focus_limit,
    )

    packet = {
        "schema_version": AI_REVIEW_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "review_context": {
            "review_date": str(review_date),
            "contest_id": str(contest_id or ""),
        },
        "scorecards": {
            "projection_quality": projection_quality,
            "ownership_quality": ownership_quality,
            "lineup_quality": lineup_quality,
            "field_quality": field_quality,
        },
        "focus_tables": {
            "attention_index_top": attention,
            "projection_miss_top": top_projection_misses,
            "ownership_miss_top": top_ownership_misses,
            "underprojected_ceiling_top": underprojected_ceiling_top,
            "low_own_smash_top": low_own_smash_top,
            "ownership_surprise_smash_top": ownership_surprise_smash_top,
        },
        "adjustment_factors": adjustment_factors_df.to_dict(orient="records") if not adjustment_factors_df.empty else [],
        "phantom_summary": phantom_summary_df.to_dict(orient="records") if not phantom_summary_df.empty else [],
        "notes_for_agent": [
            "Focus recommendations on concrete model changes, feature additions, and lineup constraints.",
            "Tie every recommendation to metrics from scorecards or rows in focus_tables.",
            "Prefer changes that are testable in the next slate.",
        ],
    }
    return packet


def build_ai_review_user_prompt(packet: dict[str, Any]) -> str:
    payload_json = json.dumps(packet, indent=2, ensure_ascii=True)
    return (
        "Review this DFS daily report packet and provide prioritized recommendations.\n\n"
        "Required output format:\n"
        "1) Executive Summary (3-5 bullets)\n"
        "2) Projection Model Recommendations (ranked by expected impact)\n"
        "3) Ownership Model Recommendations (ranked by expected impact)\n"
        "4) Odds/Market -> Ownership Translation (if packet has market evidence; otherwise state missing evidence)\n"
        "5) GPP Differentiation + Stack Leverage Plan (how to be different without sacrificing ceiling)\n"
        "6) High-Leverage Players/Archetypes to Investigate Next\n"
        "7) Next-Slate Experiment Plan (max 5 experiments; each includes hypothesis, implementation, and success metric)\n\n"
        "Constraints:\n"
        "- Use only evidence in the JSON packet.\n"
        "- Cite exact metric names/values when making a claim.\n"
        "- Label confidence as High/Medium/Low per recommendation.\n"
        "- Explicitly separate one-off noise vs recurring signal.\n"
        "- Do not invent settings/features not represented in the packet; if needed, list as dependency.\n"
        "- If data is insufficient, say exactly what is missing.\n\n"
        "JSON packet:\n"
        f"{payload_json}\n"
    )


def build_global_ai_review_packet(
    daily_packets: list[dict[str, Any]],
    focus_limit: int = 25,
) -> dict[str, Any]:
    packets = [p for p in (daily_packets or []) if isinstance(p, dict)]
    trends: list[dict[str, Any]] = []
    attention_rollup: dict[str, dict[str, Any]] = {}

    projection_weight = 0.0
    projection_blend_mae_sum = 0.0
    projection_blend_rmse_sum = 0.0
    projection_rank_corr_weight = 0.0
    projection_rank_corr_sum = 0.0

    ownership_weight = 0.0
    ownership_mae_sum = 0.0
    ownership_rank_corr_weight = 0.0
    ownership_rank_corr_sum = 0.0

    lineup_weight = 0.0
    lineup_delta_sum = 0.0
    lineup_beat_sum = 0.0

    for packet in packets:
        context = packet.get("review_context") or {}
        review_date = str(context.get("review_date") or "")
        scorecards = packet.get("scorecards") or {}
        projection = scorecards.get("projection_quality") or {}
        ownership = scorecards.get("ownership_quality") or {}
        lineup = scorecards.get("lineup_quality") or {}

        p_rows = float(_to_float(projection.get("matched_rows")) or 0.0)
        p_mae = float(_to_float(projection.get("blend_mae")) or 0.0)
        p_rmse = float(_to_float(projection.get("blend_rmse")) or 0.0)
        p_rank = _to_float(projection.get("blended_rank_spearman"))
        if p_rows > 0:
            projection_weight += p_rows
            projection_blend_mae_sum += p_rows * p_mae
            projection_blend_rmse_sum += p_rows * p_rmse
            if p_rank is not None:
                projection_rank_corr_weight += p_rows
                projection_rank_corr_sum += p_rows * float(p_rank)

        o_rows = float(_to_float(ownership.get("matched_rows")) or 0.0)
        o_mae = float(_to_float(ownership.get("ownership_mae")) or 0.0)
        o_rank = _to_float(ownership.get("ownership_rank_spearman"))
        if o_rows > 0:
            ownership_weight += o_rows
            ownership_mae_sum += o_rows * o_mae
            if o_rank is not None:
                ownership_rank_corr_weight += o_rows
                ownership_rank_corr_sum += o_rows * float(o_rank)

        l_rows = float(_to_float(lineup.get("lineups_scored")) or 0.0)
        l_delta = float(_to_float(lineup.get("avg_actual_minus_projected")) or 0.0)
        l_beat = float(_to_float(lineup.get("avg_would_beat_pct")) or 0.0)
        if l_rows > 0:
            lineup_weight += l_rows
            lineup_delta_sum += l_rows * l_delta
            lineup_beat_sum += l_rows * l_beat

        trends.append(
            {
                "review_date": review_date,
                "projection_rows": int(p_rows),
                "blend_mae": p_mae,
                "ownership_rows": int(o_rows),
                "ownership_mae": o_mae,
                "lineups_scored": int(l_rows),
                "lineup_avg_actual_minus_projected": l_delta,
            }
        )

        attention_rows = ((packet.get("focus_tables") or {}).get("attention_index_top")) or []
        for row in attention_rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("Name") or "").strip()
            team = str(row.get("TeamAbbrev") or "").strip().upper()
            if not name:
                continue
            key = f"{name}|{team}"
            existing = attention_rollup.get(
                key,
                {
                    "Name": name,
                    "TeamAbbrev": team,
                    "appearances": 0,
                    "attention_sum": 0.0,
                    "attention_max": 0.0,
                    "latest_date": "",
                    "last_blend_error": None,
                    "last_ownership_diff": None,
                },
            )
            attention_value = float(_to_float(row.get("attention_index")) or 0.0)
            existing["appearances"] = int(existing["appearances"]) + 1
            existing["attention_sum"] = float(existing["attention_sum"]) + attention_value
            existing["attention_max"] = max(float(existing["attention_max"]), attention_value)
            if review_date and review_date >= str(existing.get("latest_date") or ""):
                existing["latest_date"] = review_date
                existing["last_blend_error"] = _to_float(row.get("blend_error"))
                existing["last_ownership_diff"] = _to_float(row.get("ownership_diff_vs_proj"))
            attention_rollup[key] = existing

    trends = sorted(
        [t for t in trends if str(t.get("review_date") or "").strip()],
        key=lambda x: str(x.get("review_date") or ""),
    )
    recurring_focus = list(attention_rollup.values())
    for row in recurring_focus:
        appearances = max(1, int(row.get("appearances") or 0))
        row["attention_avg"] = float(row.get("attention_sum") or 0.0) / float(appearances)
        row.pop("attention_sum", None)
    recurring_focus = sorted(
        recurring_focus,
        key=lambda x: (int(x.get("appearances") or 0), float(x.get("attention_avg") or 0.0)),
        reverse=True,
    )[: max(1, int(focus_limit))]

    projection_summary = {
        "total_matched_rows": int(projection_weight),
        "weighted_blend_mae": (projection_blend_mae_sum / projection_weight) if projection_weight > 0 else 0.0,
        "weighted_blend_rmse": (projection_blend_rmse_sum / projection_weight) if projection_weight > 0 else 0.0,
        "weighted_rank_spearman": (
            projection_rank_corr_sum / projection_rank_corr_weight if projection_rank_corr_weight > 0 else None
        ),
    }
    ownership_summary = {
        "total_matched_rows": int(ownership_weight),
        "weighted_ownership_mae": (ownership_mae_sum / ownership_weight) if ownership_weight > 0 else 0.0,
        "weighted_rank_spearman": (
            ownership_rank_corr_sum / ownership_rank_corr_weight if ownership_rank_corr_weight > 0 else None
        ),
    }
    lineup_summary = {
        "total_lineups_scored": int(lineup_weight),
        "weighted_avg_actual_minus_projected": (lineup_delta_sum / lineup_weight) if lineup_weight > 0 else 0.0,
        "weighted_avg_would_beat_pct": (lineup_beat_sum / lineup_weight) if lineup_weight > 0 else 0.0,
    }

    return {
        "schema_version": "v1_global",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_summary": {
            "slate_count": int(len(packets)),
            "start_date": trends[0]["review_date"] if trends else "",
            "end_date": trends[-1]["review_date"] if trends else "",
        },
        "global_scorecards": {
            "projection": projection_summary,
            "ownership": ownership_summary,
            "lineup": lineup_summary,
        },
        "trend_by_date": trends,
        "recurring_focus_players": recurring_focus,
        "daily_packets": packets,
    }


def build_global_ai_review_user_prompt(global_packet: dict[str, Any]) -> str:
    payload_json = json.dumps(global_packet, indent=2, ensure_ascii=True)
    return (
        "You are reviewing multi-slate DFS performance to recommend global strategy changes.\n\n"
        "Required output format:\n"
        "1) Global Diagnostic Summary (what is systematically off)\n"
        "2) Odds -> Ownership Translation (which market signals explain ownership misses, and where evidence is weak)\n"
        "3) Projection Engine Global Changes (feature/process/model changes)\n"
        "4) Ownership Engine Global Changes\n"
        "5) GPP Stack Leverage Framework (how to be different vs field while preserving ceiling)\n"
        "6) Recurring Player Archetypes and Failure Modes\n"
        "7) 2-Week Implementation Plan (sequenced tasks + validation metrics)\n\n"
        "Constraints:\n"
        "- Use only evidence in the JSON packet.\n"
        "- Distinguish one-off noise vs recurring signal.\n"
        "- If outage/sparse dates exist, separate healthy-day conclusions from outage-day notes.\n"
        "- Use exact dates when citing one-off events.\n"
        "- For each recommendation, include expected impact and confidence.\n"
        "- Cite exact metric names/values.\n"
        "- Keep it concise: max 18 bullets total and roughly <= 1100 words.\n"
        "- Do not invent settings/features not represented in the packet; if needed, list as dependency.\n\n"
        "JSON packet:\n"
        f"{payload_json}\n"
    )


def _build_market_correlation_row(
    df: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    metric: str,
) -> dict[str, Any]:
    if x_col not in df.columns or y_col not in df.columns:
        return {
            "metric": metric,
            "x_col": x_col,
            "y_col": y_col,
            "samples": 0,
            "pearson": None,
            "spearman": None,
        }
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    return {
        "metric": metric,
        "x_col": x_col,
        "y_col": y_col,
        "samples": int(len(valid)),
        "pearson": _safe_pearson(valid["x"], valid["y"]),
        "spearman": _safe_spearman(valid["x"], valid["y"]),
    }


def _build_market_bucket_summary(
    df: pd.DataFrame,
    *,
    source_col: str,
    bucket_name: str,
) -> pd.DataFrame:
    if source_col not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out[source_col] = pd.to_numeric(out[source_col], errors="coerce")
    out = out.loc[out[source_col].notna()].copy()
    if out.empty:
        return pd.DataFrame()
    unique_values = int(out[source_col].nunique(dropna=True))
    if unique_values < 3:
        return pd.DataFrame()

    try:
        out["bucket"] = pd.qcut(out[source_col], q=min(4, unique_values), duplicates="drop")
    except Exception:
        return pd.DataFrame()
    out = out.loc[out["bucket"].notna()].copy()
    if out.empty:
        return pd.DataFrame()

    grouped = (
        out.groupby("bucket", as_index=False)
        .agg(
            samples=("bucket", "count"),
            avg_source=(source_col, "mean"),
            avg_blended_projection=("blended_projection", "mean"),
            avg_actual_dk_points=("actual_dk_points", "mean"),
            avg_blend_error=("blend_error", "mean"),
            blend_mae=("blend_error", lambda s: float(pd.to_numeric(s, errors="coerce").abs().mean())),
            avg_projected_ownership=("projected_ownership", "mean"),
            avg_actual_ownership=("actual_ownership_from_file", "mean"),
            ownership_mae=("ownership_error", lambda s: float(pd.to_numeric(s, errors="coerce").abs().mean())),
        )
        .sort_values("avg_source", ascending=True)
        .reset_index(drop=True)
    )
    grouped["bucket"] = grouped["bucket"].astype(str)
    grouped["bucket_name"] = str(bucket_name)
    avg_proj = pd.to_numeric(grouped["avg_blended_projection"], errors="coerce")
    avg_actual = pd.to_numeric(grouped["avg_actual_dk_points"], errors="coerce")
    grouped["suggested_projection_scale"] = (
        avg_actual / avg_proj.replace(0, pd.NA)
    )
    grouped["suggested_projection_scale"] = pd.to_numeric(grouped["suggested_projection_scale"], errors="coerce")
    return grouped


_SLATE_SIZE_BUCKET_ORDER = {
    "unknown": 0,
    "1 game": 1,
    "2-4 games": 2,
    "5-7 games": 3,
    "8-10 games": 4,
    "11+ games": 5,
}


def _slate_size_bucket_label(game_count: Any) -> str:
    n = _to_int(game_count, default=0)
    if n <= 0:
        return "unknown"
    if n == 1:
        return "1 game"
    if n <= 4:
        return "2-4 games"
    if n <= 7:
        return "5-7 games"
    if n <= 10:
        return "8-10 games"
    return "11+ games"


def _projection_decile_label(value: Any) -> str:
    decile = max(1, min(10, _to_int(value, default=1)))
    lo = (decile - 1) * 10
    hi = decile * 10
    return f"p{lo:02d}_{hi:02d}"


def _build_ownership_reverse_engineering(
    frame: pd.DataFrame,
    *,
    min_bucket_samples: int = 20,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "overall_metrics": {},
            "slate_size_summary": [],
            "curve_table": [],
        }

    out = frame.copy()
    if "review_date" not in out.columns:
        out["review_date"] = ""
    out["review_date"] = out["review_date"].astype(str).str.strip()

    for col in [
        "blended_projection",
        "projected_ownership",
        "actual_ownership_from_file",
        "actual_dk_points",
        "game_total_line",
        "abs_game_spread",
        "ownership_error",
        "slate_game_count",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    if "game_key" not in out.columns:
        out["game_key"] = ""
    out["game_key"] = out["game_key"].astype(str).str.strip().str.upper()

    # Infer slate game count from per-date unique game keys when available.
    game_counts_by_date = (
        out.loc[(out["review_date"] != "") & (out["game_key"] != "")]
        .groupby("review_date", as_index=False)["game_key"]
        .nunique()
        .rename(columns={"game_key": "slate_game_count_inferred"})
    )
    if not game_counts_by_date.empty:
        out = out.merge(game_counts_by_date, on="review_date", how="left")
        inferred = pd.to_numeric(out.get("slate_game_count_inferred"), errors="coerce")
        existing = pd.to_numeric(out.get("slate_game_count"), errors="coerce")
        out["slate_game_count"] = existing.where(existing.notna(), inferred)
        out = out.drop(columns=["slate_game_count_inferred"], errors="ignore")

    out["slate_size_bucket"] = out["slate_game_count"].map(_slate_size_bucket_label)
    out["slate_size_bucket"] = out["slate_size_bucket"].where(out["slate_size_bucket"].notna(), "unknown")

    out["projection_rank_pct"] = (
        out.groupby("review_date", dropna=False)["blended_projection"]
        .rank(method="average", pct=True)
    )
    global_rank = pd.to_numeric(out["blended_projection"], errors="coerce").rank(method="average", pct=True)
    out["projection_rank_pct"] = pd.to_numeric(out["projection_rank_pct"], errors="coerce").where(
        pd.to_numeric(out["projection_rank_pct"], errors="coerce").notna(),
        global_rank,
    )
    out["projection_rank_pct"] = pd.to_numeric(out["projection_rank_pct"], errors="coerce").clip(lower=0.0, upper=1.0)

    decile_bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    out["projection_decile"] = pd.cut(
        out["projection_rank_pct"],
        bins=decile_bins,
        labels=list(range(1, 11)),
        include_lowest=True,
    )
    out["projection_decile"] = pd.to_numeric(out["projection_decile"], errors="coerce")
    out["projection_bucket"] = out["projection_decile"].map(_projection_decile_label)

    valid = out.loc[
        out["actual_ownership_from_file"].notna()
        & out["projection_decile"].notna()
    ].copy()
    if valid.empty:
        return {
            "overall_metrics": {
                "rows_with_actual_ownership": 0,
                "current_ownership_mae": None,
                "baseline_ownership_mae": None,
                "mae_improvement_vs_current": None,
            },
            "slate_size_summary": [],
            "curve_table": [],
        }

    valid["projection_decile"] = pd.to_numeric(valid["projection_decile"], errors="coerce").astype(int)
    valid["projection_bucket"] = valid["projection_decile"].map(_projection_decile_label)
    valid["current_abs_error"] = (
        pd.to_numeric(valid["actual_ownership_from_file"], errors="coerce")
        - pd.to_numeric(valid["projected_ownership"], errors="coerce")
    ).abs()

    pair_stats = (
        valid.groupby(["slate_size_bucket", "projection_decile", "projection_bucket"], as_index=False)
        .agg(
            samples=("actual_ownership_from_file", "count"),
            avg_actual_ownership=("actual_ownership_from_file", "mean"),
            avg_projected_ownership=("projected_ownership", "mean"),
            avg_blended_projection=("blended_projection", "mean"),
            avg_actual_dk_points=("actual_dk_points", "mean"),
            avg_game_total_line=("game_total_line", "mean"),
            avg_abs_spread=("abs_game_spread", "mean"),
            current_ownership_mae=("current_abs_error", "mean"),
        )
        .reset_index(drop=True)
    )
    pair_stats["projected_minus_actual_ownership"] = (
        pd.to_numeric(pair_stats["avg_projected_ownership"], errors="coerce")
        - pd.to_numeric(pair_stats["avg_actual_ownership"], errors="coerce")
    )

    pair_stats["_slate_order"] = pair_stats["slate_size_bucket"].map(_SLATE_SIZE_BUCKET_ORDER).fillna(99)
    pair_stats = pair_stats.sort_values(["_slate_order", "projection_decile"], ascending=[True, True]).reset_index(drop=True)

    decile_stats = (
        valid.groupby("projection_decile", as_index=False)
        .agg(
            decile_samples=("actual_ownership_from_file", "count"),
            decile_actual_ownership=("actual_ownership_from_file", "mean"),
        )
        .reset_index(drop=True)
    )

    overall_actual_ownership = _safe_mean(pd.to_numeric(valid["actual_ownership_from_file"], errors="coerce"))
    bucket_min = max(3, int(min_bucket_samples))

    scoring = valid.merge(
        pair_stats[["slate_size_bucket", "projection_decile", "samples", "avg_actual_ownership"]],
        on=["slate_size_bucket", "projection_decile"],
        how="left",
    ).rename(
        columns={
            "samples": "pair_samples",
            "avg_actual_ownership": "pair_actual_ownership",
        }
    )
    scoring = scoring.merge(
        decile_stats,
        on="projection_decile",
        how="left",
    )
    scoring["expected_ownership_baseline"] = pd.to_numeric(
        scoring["pair_actual_ownership"],
        errors="coerce",
    )
    sparse_pair = pd.to_numeric(scoring["pair_samples"], errors="coerce").fillna(0) < float(bucket_min)
    scoring.loc[sparse_pair, "expected_ownership_baseline"] = pd.to_numeric(
        scoring.loc[sparse_pair, "decile_actual_ownership"],
        errors="coerce",
    )
    sparse_decile = sparse_pair & (
        pd.to_numeric(scoring["decile_samples"], errors="coerce").fillna(0) < float(bucket_min)
    )
    scoring.loc[sparse_decile, "expected_ownership_baseline"] = float(overall_actual_ownership)

    source_series = pd.Series(["pair_bucket"] * len(scoring), index=scoring.index, dtype="object")
    source_series.loc[sparse_pair] = "projection_decile"
    source_series.loc[sparse_decile] = "overall"
    scoring["baseline_source"] = source_series

    scoring["baseline_abs_error"] = (
        pd.to_numeric(scoring["actual_ownership_from_file"], errors="coerce")
        - pd.to_numeric(scoring["expected_ownership_baseline"], errors="coerce")
    ).abs()

    baseline_source_counts = scoring["baseline_source"].value_counts(dropna=False).to_dict()
    total_scored = max(1, len(scoring))
    overall_metrics = {
        "rows_with_actual_ownership": int(len(scoring)),
        "current_ownership_mae": round(_safe_mean(scoring["current_abs_error"]), 4),
        "baseline_ownership_mae": round(_safe_mean(scoring["baseline_abs_error"]), 4),
        "mae_improvement_vs_current": round(
            _safe_mean(scoring["current_abs_error"]) - _safe_mean(scoring["baseline_abs_error"]),
            4,
        ),
        "bucket_min_samples": int(bucket_min),
        "baseline_source_pair_bucket_pct": round(
            _safe_pct_value(float(baseline_source_counts.get("pair_bucket", 0.0)), float(total_scored)),
            2,
        ),
        "baseline_source_projection_decile_pct": round(
            _safe_pct_value(float(baseline_source_counts.get("projection_decile", 0.0)), float(total_scored)),
            2,
        ),
        "baseline_source_overall_pct": round(
            _safe_pct_value(float(baseline_source_counts.get("overall", 0.0)), float(total_scored)),
            2,
        ),
    }

    slate_summary = (
        scoring.groupby("slate_size_bucket", as_index=False)
        .agg(
            samples=("actual_ownership_from_file", "count"),
            avg_slate_game_count=("slate_game_count", "mean"),
            current_ownership_mae=("current_abs_error", "mean"),
            baseline_ownership_mae=("baseline_abs_error", "mean"),
            avg_projected_ownership=("projected_ownership", "mean"),
            avg_actual_ownership=("actual_ownership_from_file", "mean"),
            avg_baseline_expected_ownership=("expected_ownership_baseline", "mean"),
        )
        .reset_index(drop=True)
    )
    slate_summary["mae_improvement_vs_current"] = (
        pd.to_numeric(slate_summary["current_ownership_mae"], errors="coerce")
        - pd.to_numeric(slate_summary["baseline_ownership_mae"], errors="coerce")
    )
    slate_summary["_slate_order"] = slate_summary["slate_size_bucket"].map(_SLATE_SIZE_BUCKET_ORDER).fillna(99)
    slate_summary = slate_summary.sort_values("_slate_order", ascending=True).drop(columns=["_slate_order"], errors="ignore")

    curve_table = pair_stats.drop(columns=["_slate_order"], errors="ignore")

    return {
        "overall_metrics": overall_metrics,
        "slate_size_summary": slate_summary.to_dict(orient="records"),
        "curve_table": curve_table.to_dict(orient="records"),
    }


def build_market_correlation_ai_review_packet(
    *,
    review_rows_df: pd.DataFrame,
    focus_limit: int = 20,
    min_bucket_samples: int = 20,
) -> dict[str, Any]:
    frame = review_rows_df.copy() if isinstance(review_rows_df, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return {
            "schema_version": MARKET_CORRELATION_AI_REVIEW_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_summary": {
                "dates_used": 0,
                "start_date": "",
                "end_date": "",
                "rows": 0,
                "rows_with_actual_points": 0,
                "rows_with_actual_ownership": 0,
            },
            "global_quality": {},
            "correlation_table": [],
            "bucket_calibration": {
                "total_line_buckets": [],
                "abs_spread_buckets": [],
            },
            "ownership_reverse_engineering": {
                "overall_metrics": {},
                "slate_size_summary": [],
                "curve_table": [],
            },
            "trend_by_date": [],
            "calibration_recommendations": [],
            "notes_for_agent": [
                "No multi-date rows were available; ensure projection snapshots and actual results exist for selected dates.",
            ],
        }

    if "review_date" not in frame.columns:
        frame["review_date"] = ""
    frame["review_date"] = frame["review_date"].astype(str).str.strip()

    numeric_cols = [
        "blended_projection",
        "our_dk_projection",
        "vegas_dk_projection",
        "actual_dk_points",
        "projected_ownership",
        "actual_ownership_from_file",
        "blend_error",
        "ownership_error",
        "game_total_line",
        "game_spread_line",
        "game_tail_score",
        "vegas_blend_weight",
        "slate_game_count",
    ]
    for col in numeric_cols:
        if col not in frame.columns:
            frame[col] = pd.NA
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if frame["blend_error"].isna().all():
        frame["blend_error"] = frame["actual_dk_points"] - frame["blended_projection"]
    if frame["ownership_error"].isna().all():
        frame["ownership_error"] = frame["actual_ownership_from_file"] - frame["projected_ownership"]

    frame["abs_game_spread"] = pd.to_numeric(frame["game_spread_line"], errors="coerce").abs()
    frame["abs_blend_error"] = pd.to_numeric(frame["blend_error"], errors="coerce").abs()
    frame["abs_ownership_error"] = pd.to_numeric(frame["ownership_error"], errors="coerce").abs()
    ownership_reverse_engineering = _build_ownership_reverse_engineering(
        frame,
        min_bucket_samples=min_bucket_samples,
    )

    corr_specs = [
        ("blended_projection", "actual_dk_points", "blend_projection_vs_actual_points"),
        ("vegas_dk_projection", "actual_dk_points", "vegas_projection_vs_actual_points"),
        ("game_total_line", "actual_dk_points", "game_total_line_vs_actual_points"),
        ("abs_game_spread", "actual_dk_points", "abs_spread_vs_actual_points"),
        ("game_total_line", "projected_ownership", "game_total_line_vs_projected_ownership"),
        ("game_total_line", "actual_ownership_from_file", "game_total_line_vs_actual_ownership"),
        ("abs_game_spread", "projected_ownership", "abs_spread_vs_projected_ownership"),
        ("game_tail_score", "actual_dk_points", "game_tail_score_vs_actual_points"),
        ("vegas_blend_weight", "abs_blend_error", "vegas_blend_weight_vs_abs_blend_error"),
    ]
    correlation_rows = [
        _build_market_correlation_row(frame, x_col=x_col, y_col=y_col, metric=metric)
        for x_col, y_col, metric in corr_specs
    ]

    total_bucket_df = _build_market_bucket_summary(
        frame,
        source_col="game_total_line",
        bucket_name="game_total_line",
    )
    spread_bucket_df = _build_market_bucket_summary(
        frame,
        source_col="abs_game_spread",
        bucket_name="abs_game_spread",
    )

    recommendation_rows: list[dict[str, Any]] = []
    for bucket_df in [total_bucket_df, spread_bucket_df]:
        if bucket_df.empty:
            continue
        work = bucket_df.copy()
        work["samples"] = pd.to_numeric(work["samples"], errors="coerce").fillna(0).astype(int)
        work["avg_blend_error"] = pd.to_numeric(work["avg_blend_error"], errors="coerce")
        work["suggested_projection_scale"] = pd.to_numeric(work["suggested_projection_scale"], errors="coerce")
        work = work.loc[
            (work["samples"] >= int(max(1, min_bucket_samples)))
            & work["avg_blend_error"].notna()
            & (work["avg_blend_error"].abs() >= 1.0)
        ].copy()
        if work.empty:
            continue
        work["priority_score"] = work["avg_blend_error"].abs() * (work["samples"] ** 0.5)
        for _, row in work.iterrows():
            recommendation_rows.append(
                {
                    "bucket_name": str(row.get("bucket_name") or ""),
                    "bucket": str(row.get("bucket") or ""),
                    "samples": int(row.get("samples") or 0),
                    "avg_blend_error": float(row.get("avg_blend_error") or 0.0),
                    "blend_mae": float(row.get("blend_mae") or 0.0),
                    "suggested_projection_scale": (
                        float(row.get("suggested_projection_scale"))
                        if pd.notna(row.get("suggested_projection_scale"))
                        else None
                    ),
                    "priority_score": float(row.get("priority_score") or 0.0),
                }
            )
    recommendation_rows = sorted(
        recommendation_rows,
        key=lambda x: float(x.get("priority_score") or 0.0),
        reverse=True,
    )[: max(1, int(focus_limit))]

    trend_rows: list[dict[str, Any]] = []
    dated = frame.loc[frame["review_date"] != ""].copy()
    for review_date, day in dated.groupby("review_date", as_index=False):
        day = day.copy()
        trend_rows.append(
            {
                "review_date": str(review_date),
                "rows": int(len(day)),
                "rows_with_actual_points": int(pd.to_numeric(day["actual_dk_points"], errors="coerce").notna().sum()),
                "rows_with_actual_ownership": int(pd.to_numeric(day["actual_ownership_from_file"], errors="coerce").notna().sum()),
                "blend_mae": round(_safe_mean(pd.to_numeric(day["blend_error"], errors="coerce").abs()), 4),
                "ownership_mae": round(_safe_mean(pd.to_numeric(day["ownership_error"], errors="coerce").abs()), 4),
                "corr_total_vs_actual_points_spearman": _safe_spearman(
                    pd.to_numeric(day["game_total_line"], errors="coerce"),
                    pd.to_numeric(day["actual_dk_points"], errors="coerce"),
                ),
                "corr_total_vs_actual_ownership_spearman": _safe_spearman(
                    pd.to_numeric(day["game_total_line"], errors="coerce"),
                    pd.to_numeric(day["actual_ownership_from_file"], errors="coerce"),
                ),
            }
        )
    trend_rows = sorted(trend_rows, key=lambda x: str(x.get("review_date") or ""))

    nonempty_dates = sorted({d for d in frame["review_date"].tolist() if str(d).strip()})
    window_summary = {
        "dates_used": int(len(nonempty_dates)),
        "start_date": str(nonempty_dates[0]) if nonempty_dates else "",
        "end_date": str(nonempty_dates[-1]) if nonempty_dates else "",
        "rows": int(len(frame)),
        "rows_with_actual_points": int(frame["actual_dk_points"].notna().sum()),
        "rows_with_actual_ownership": int(frame["actual_ownership_from_file"].notna().sum()),
    }

    global_quality = {
        "blend_mae": round(_safe_mean(frame["abs_blend_error"]), 4),
        "ownership_mae": round(_safe_mean(frame["abs_ownership_error"]), 4),
        "blend_rank_spearman": _safe_spearman(frame["blended_projection"], frame["actual_dk_points"]),
        "total_line_vs_actual_points_spearman": _safe_spearman(frame["game_total_line"], frame["actual_dk_points"]),
        "total_line_vs_actual_ownership_spearman": _safe_spearman(
            frame["game_total_line"],
            frame["actual_ownership_from_file"],
        ),
    }

    return {
        "schema_version": MARKET_CORRELATION_AI_REVIEW_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window_summary": window_summary,
        "global_quality": global_quality,
        "correlation_table": correlation_rows,
        "bucket_calibration": {
            "total_line_buckets": (
                total_bucket_df.to_dict(orient="records") if not total_bucket_df.empty else []
            ),
            "abs_spread_buckets": (
                spread_bucket_df.to_dict(orient="records") if not spread_bucket_df.empty else []
            ),
        },
        "ownership_reverse_engineering": ownership_reverse_engineering,
        "trend_by_date": trend_rows,
        "calibration_recommendations": recommendation_rows,
        "notes_for_agent": [
            "Focus on robust signals (higher samples) and avoid overfitting to one-date outliers.",
            "Use suggested_projection_scale by bucket as starting points; validate with next-slate MAE deltas.",
            "Use ownership_reverse_engineering.curve_table for projection->ownership expectations by slate size.",
            "Separate ownership calibration from projection calibration when signals diverge.",
        ],
    }


def build_market_correlation_ai_review_user_prompt(packet: dict[str, Any]) -> str:
    payload_json = json.dumps(packet, indent=2, ensure_ascii=True)
    return (
        "Review this multi-date market-correlation packet and provide actionable DFS calibration guidance.\n\n"
        "Required output format:\n"
        "1) Market Signal Summary (3-6 bullets)\n"
        "2) Odds -> Points Correlations (what is real vs weak)\n"
        "3) Odds -> Ownership Correlations (what to adjust)\n"
        "4) Projection -> Ownership Curve by Slate Size (map projected points/rank to expected ownership ranges)\n"
        "5) Odds -> Stack Translation (which totals/spreads imply stackable games, and where to be under/over field)\n"
        "6) Projection Tightening Plan by Bucket (max 6 actions)\n"
        "7) Ownership Tightening Plan by Bucket (max 5 actions)\n"
        "8) GPP Differentiation Edge Plan (max 5 actions; include leverage rationale + risk)\n"
        "9) Next-Slate Validation Plan (metrics + pass/fail thresholds)\n\n"
        "Constraints:\n"
        "- Use only evidence in the JSON packet.\n"
        "- Cite exact metric names and values.\n"
        "- Prioritize recommendations with adequate sample size.\n"
        "- Use ownership_reverse_engineering.overall_metrics and curve_table when discussing expected ownership.\n"
        "- Highlight where market signals and ownership behavior diverge (edge opportunities).\n"
        "- If evidence is insufficient, state exactly what is missing.\n"
        "- Do not invent settings that are not represented in the packet.\n\n"
        "JSON packet:\n"
        f"{payload_json}\n"
    )


def _safe_pct_value(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float((numerator / denominator) * 100.0)


def _american_to_implied_probability(value: Any) -> float | None:
    price = _to_float(value)
    if price is None or price == 0:
        return None
    if price > 0:
        return float(100.0 / (price + 100.0))
    abs_price = abs(price)
    return float(abs_price / (abs_price + 100.0))


def _normalize_win_probabilities(
    home_prob: float | None,
    away_prob: float | None,
) -> tuple[float | None, float | None]:
    h = float(home_prob) if home_prob is not None else None
    a = float(away_prob) if away_prob is not None else None
    if h is None and a is None:
        return None, None
    if h is None and a is not None:
        if 0.0 <= a <= 1.0:
            return float(1.0 - a), a
        return None, None
    if a is None and h is not None:
        if 0.0 <= h <= 1.0:
            return h, float(1.0 - h)
        return None, None
    total = float((h or 0.0) + (a or 0.0))
    if total <= 0:
        return None, None
    return float(h / total), float(a / total)


def _build_prior_team_form_frame(prior_boxscore_df: pd.DataFrame) -> pd.DataFrame:
    if prior_boxscore_df.empty or "team_name" not in prior_boxscore_df.columns:
        return pd.DataFrame(
            columns=[
                "team_name",
                "players_logged",
                "team_dk_points",
                "team_minutes",
                "top3_dk_share_pct",
                "value_score",
            ]
        )

    base = prior_boxscore_df.copy()
    base["team_name"] = base.get("team_name", "").astype(str).str.strip()
    base["actual_dk_points"] = pd.to_numeric(base.get("actual_dk_points"), errors="coerce").fillna(0.0)
    base["actual_minutes"] = pd.to_numeric(base.get("actual_minutes"), errors="coerce").fillna(0.0)
    base = base.loc[base["team_name"] != ""].reset_index(drop=True)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "team_name",
                "players_logged",
                "team_dk_points",
                "team_minutes",
                "top3_dk_share_pct",
                "value_score",
            ]
        )

    rows: list[dict[str, Any]] = []
    for team_name, team_df in base.groupby("team_name", dropna=False):
        team_points = float(team_df["actual_dk_points"].sum())
        team_minutes = float(team_df["actual_minutes"].sum())
        players_logged = int(len(team_df))
        top3_points = float(team_df["actual_dk_points"].nlargest(3).sum())
        top3_share = 0.0 if team_points <= 0 else (top3_points / team_points)

        # Higher value_score means broader production (lower concentration at the top).
        value_score = float(team_points * (1.0 - min(1.0, max(0.0, top3_share))))
        rows.append(
            {
                "team_name": str(team_name),
                "players_logged": players_logged,
                "team_dk_points": round(team_points, 4),
                "team_minutes": round(team_minutes, 4),
                "top3_dk_share_pct": round(top3_share * 100.0, 4),
                "value_score": round(value_score, 4),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "team_name",
                "players_logged",
                "team_dk_points",
                "team_minutes",
                "top3_dk_share_pct",
                "value_score",
            ]
        )
    return pd.DataFrame(rows).sort_values("team_dk_points", ascending=False).reset_index(drop=True)


def _opponent_from_game_key(game_key: str, team_abbrev: str) -> str:
    key = str(game_key or "").strip().upper()
    team = str(team_abbrev or "").strip().upper()
    if not key or "@" not in key or not team:
        return ""
    away, home = key.split("@", 1)
    away = away.strip().upper()
    home = home.strip().upper()
    if team == away:
        return home
    if team == home:
        return away
    return ""


def _build_gpp_stack_targets_from_pool(
    player_pool_df: pd.DataFrame,
    focus_limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if player_pool_df.empty:
        return [], [], []

    pool = player_pool_df.copy()
    if "Name" not in pool.columns or "TeamAbbrev" not in pool.columns:
        return [], [], []

    pool["Name"] = pool["Name"].astype(str).str.strip()
    pool["TeamAbbrev"] = pool["TeamAbbrev"].astype(str).str.strip().str.upper()
    pool["Position"] = pool["Position"].astype(str).str.strip().str.upper() if "Position" in pool.columns else ""
    pool["game_key"] = pool["game_key"].astype(str).str.strip().str.upper() if "game_key" in pool.columns else ""
    pool["Salary"] = pd.to_numeric(pool["Salary"], errors="coerce") if "Salary" in pool.columns else pd.NA
    pool["projected_dk_points"] = pd.to_numeric(pool.get("projected_dk_points"), errors="coerce")
    pool["projected_ownership"] = pd.to_numeric(pool.get("projected_ownership"), errors="coerce")
    pool["leverage_score"] = pd.to_numeric(pool.get("leverage_score"), errors="coerce")
    pool["value_per_1k"] = pd.to_numeric(pool.get("value_per_1k"), errors="coerce")
    pool["game_tail_score"] = pd.to_numeric(pool.get("game_tail_score"), errors="coerce")
    pool["game_total_line"] = pd.to_numeric(pool.get("game_total_line"), errors="coerce")
    pool["game_spread_line"] = pd.to_numeric(pool.get("game_spread_line"), errors="coerce")
    pool = pool.loc[(pool["Name"] != "") & (pool["TeamAbbrev"] != "")]
    if pool.empty:
        return [], [], []

    pool["player_priority"] = (
        pool["projected_dk_points"].fillna(0.0)
        + (0.32 * pool["leverage_score"].fillna(0.0))
        + (2.0 * pool["value_per_1k"].fillna(0.0))
        - (0.10 * pool["projected_ownership"].fillna(0.0))
    )

    team_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    for (game_key, team_abbrev), grp in pool.groupby(["game_key", "TeamAbbrev"], dropna=False):
        working = grp.sort_values("player_priority", ascending=False).reset_index(drop=True)
        if working.empty:
            continue

        top_players = working.head(3).copy()
        top2_proj_sum = float(pd.to_numeric(top_players["projected_dk_points"], errors="coerce").head(2).fillna(0.0).sum())
        top3_proj_sum = float(pd.to_numeric(top_players["projected_dk_points"], errors="coerce").fillna(0.0).sum())
        avg_own = float(pd.to_numeric(top_players["projected_ownership"], errors="coerce").fillna(0.0).mean())
        avg_lev = float(pd.to_numeric(top_players["leverage_score"], errors="coerce").fillna(0.0).mean())
        avg_value = float(pd.to_numeric(top_players["value_per_1k"], errors="coerce").fillna(0.0).mean())
        tail_signal = float(pd.to_numeric(working["game_tail_score"], errors="coerce").fillna(0.0).mean())
        total_line = float(pd.to_numeric(working["game_total_line"], errors="coerce").dropna().median()) if pd.to_numeric(working["game_total_line"], errors="coerce").notna().any() else 0.0
        abs_spread = float(pd.to_numeric(working["game_spread_line"], errors="coerce").dropna().abs().median()) if pd.to_numeric(working["game_spread_line"], errors="coerce").notna().any() else 0.0

        team_stack_score = (
            (0.55 * top3_proj_sum)
            + (6.0 * avg_value)
            + (3.0 * avg_lev)
            - (0.20 * avg_own)
            + (0.06 * tail_signal)
        )
        team_rows.append(
            {
                "game_key": str(game_key or ""),
                "team_abbrev": str(team_abbrev or ""),
                "opponent_team": _opponent_from_game_key(str(game_key or ""), str(team_abbrev or "")),
                "team_stack_score": round(float(team_stack_score), 4),
                "team_top2_projection_sum": round(float(top2_proj_sum), 4),
                "team_top3_projection_sum": round(float(top3_proj_sum), 4),
                "team_avg_projected_ownership": round(float(avg_own), 4),
                "team_avg_leverage_score": round(float(avg_lev), 4),
                "team_avg_value_per_1k": round(float(avg_value), 4),
                "game_total_line": round(float(total_line), 4),
                "game_abs_spread": round(float(abs_spread), 4),
            }
        )

        for _, p in top_players.iterrows():
            player_rows.append(
                {
                    "game_key": str(game_key or ""),
                    "team_abbrev": str(team_abbrev or ""),
                    "player_name": str(p.get("Name") or "").strip(),
                    "position": str(p.get("Position") or "").strip(),
                    "salary": int(_to_int(p.get("Salary"), default=0)),
                    "projected_dk_points": round(float(_to_float(p.get("projected_dk_points")) or 0.0), 4),
                    "projected_ownership": round(float(_to_float(p.get("projected_ownership")) or 0.0), 4),
                    "leverage_score": round(float(_to_float(p.get("leverage_score")) or 0.0), 4),
                    "value_per_1k": round(float(_to_float(p.get("value_per_1k")) or 0.0), 4),
                    "player_priority": round(float(_to_float(p.get("player_priority")) or 0.0), 4),
                }
            )

    if not team_rows:
        return [], [], []

    team_targets_df = pd.DataFrame(team_rows).sort_values("team_stack_score", ascending=False).reset_index(drop=True)
    if not team_targets_df.empty:
        chalk_cutoff = float(team_targets_df["team_avg_projected_ownership"].quantile(0.67))
        leverage_cutoff = float(team_targets_df["team_avg_leverage_score"].quantile(0.67))

        def _label_stack_tier(row: pd.Series) -> str:
            own = float(_to_float(row.get("team_avg_projected_ownership")) or 0.0)
            lev = float(_to_float(row.get("team_avg_leverage_score")) or 0.0)
            if own >= chalk_cutoff:
                return "Tier 1 Chalk Core"
            if lev >= leverage_cutoff:
                return "Tier 2 Contrarian Core"
            return "Tier 3 Balanced Core"

        team_targets_df["stack_tier"] = team_targets_df.apply(_label_stack_tier, axis=1)

    player_targets_df = pd.DataFrame(player_rows).sort_values("player_priority", ascending=False).reset_index(drop=True)
    if not player_targets_df.empty:
        opponent_top: dict[tuple[str, str], str] = {}
        for _, row in player_targets_df.iterrows():
            game_key = str(row.get("game_key") or "")
            team_abbrev = str(row.get("team_abbrev") or "")
            if not game_key or not team_abbrev:
                continue
            opponent = _opponent_from_game_key(game_key, team_abbrev)
            if not opponent:
                continue
            key = (game_key, opponent)
            if key in opponent_top:
                continue
            opp_rows = player_targets_df.loc[
                (player_targets_df["game_key"] == game_key) & (player_targets_df["team_abbrev"] == opponent)
            ]
            if opp_rows.empty:
                continue
            opponent_top[key] = str(opp_rows.iloc[0].get("player_name") or "")

        bring_back: list[str] = []
        for _, row in team_targets_df.iterrows():
            key = (str(row.get("game_key") or ""), str(row.get("team_abbrev") or ""))
            bring_back.append(str(opponent_top.get(key, "")))
        team_targets_df["suggested_bring_back"] = bring_back

    # Build game-level stack targets by pairing top team + opposing bring-back team.
    game_rows: list[dict[str, Any]] = []
    for game_key, game_df in team_targets_df.groupby("game_key", dropna=False):
        if game_df.empty:
            continue
        ordered = game_df.sort_values("team_stack_score", ascending=False).reset_index(drop=True)
        primary = ordered.iloc[0]
        secondary = ordered.iloc[1] if len(ordered) > 1 else ordered.iloc[0]
        game_score = float(primary.get("team_stack_score") or 0.0) + (0.7 * float(secondary.get("team_stack_score") or 0.0))
        game_rows.append(
            {
                "game_key": str(game_key or ""),
                "game_stack_score": round(float(game_score), 4),
                "primary_team": str(primary.get("team_abbrev") or ""),
                "secondary_team": str(secondary.get("team_abbrev") or ""),
                "primary_stack_tier": str(primary.get("stack_tier") or ""),
                "secondary_stack_tier": str(secondary.get("stack_tier") or ""),
                "suggested_bring_back": str(primary.get("suggested_bring_back") or ""),
                "game_total_line": round(float(_to_float(primary.get("game_total_line")) or 0.0), 4),
                "game_abs_spread": round(float(_to_float(primary.get("game_abs_spread")) or 0.0), 4),
            }
        )

    game_targets_df = pd.DataFrame(game_rows).sort_values("game_stack_score", ascending=False).reset_index(drop=True)
    limit_n = int(max(3, focus_limit))
    game_targets = game_targets_df.head(limit_n).to_dict(orient="records") if not game_targets_df.empty else []
    team_targets = team_targets_df.head(limit_n * 2).to_dict(orient="records") if not team_targets_df.empty else []
    player_targets = player_targets_df.head(limit_n * 3).to_dict(orient="records") if not player_targets_df.empty else []
    return game_targets, team_targets, player_targets


def _summarize_vegas_history_for_game_slate(vegas_history_df: pd.DataFrame) -> dict[str, Any]:
    if vegas_history_df.empty:
        return {
            "games": 0,
            "total_line_games": 0,
            "spread_line_games": 0,
            "total_mae": 0.0,
            "spread_mae": 0.0,
            "winner_pick_accuracy_pct": 0.0,
            "high_total_over_rate_pct": 0.0,
            "close_spread_winner_accuracy_pct": 0.0,
        }

    base = vegas_history_df.copy()
    if "has_total_line" in base.columns:
        base["has_total_line"] = base["has_total_line"].astype(bool)
    else:
        base["has_total_line"] = False
    if "has_spread_line" in base.columns:
        base["has_spread_line"] = base["has_spread_line"].astype(bool)
    else:
        base["has_spread_line"] = False
    base["total_points"] = pd.to_numeric(
        base["total_points"] if "total_points" in base.columns else pd.Series(pd.NA, index=base.index),
        errors="coerce",
    )
    base["vegas_home_margin"] = pd.to_numeric(
        base["vegas_home_margin"] if "vegas_home_margin" in base.columns else pd.Series(pd.NA, index=base.index),
        errors="coerce",
    )
    base["total_abs_error"] = pd.to_numeric(
        base["total_abs_error"] if "total_abs_error" in base.columns else pd.Series(pd.NA, index=base.index),
        errors="coerce",
    )
    base["spread_abs_error"] = pd.to_numeric(
        base["spread_abs_error"] if "spread_abs_error" in base.columns else pd.Series(pd.NA, index=base.index),
        errors="coerce",
    )

    total_line_df = base.loc[base["has_total_line"]].copy()
    spread_line_df = base.loc[base["has_spread_line"]].copy()
    actual_winner_side = (
        base["actual_winner_side"].astype(str)
        if "actual_winner_side" in base.columns
        else pd.Series([""] * len(base), index=base.index)
    )
    predicted_winner_side = (
        base["predicted_winner_side"].astype(str)
        if "predicted_winner_side" in base.columns
        else pd.Series([""] * len(base), index=base.index)
    )
    winner_pick_mask = (
        base["has_spread_line"]
        & actual_winner_side.isin(["home", "away"])
        & predicted_winner_side.isin(["home", "away"])
    )

    high_total_over_rate = 0.0
    if not total_line_df.empty:
        cutoff = float(total_line_df["total_points"].dropna().quantile(0.75))
        high_total_df = total_line_df.loc[total_line_df["total_points"] >= cutoff].copy()
        if not high_total_df.empty:
            total_result = (
                high_total_df["total_result"].astype(str)
                if "total_result" in high_total_df.columns
                else pd.Series([""] * len(high_total_df), index=high_total_df.index)
            )
            high_total_over_rate = _safe_pct_value(float((total_result == "Over").sum()), float(len(high_total_df)))

    close_spread_winner_accuracy = 0.0
    if not spread_line_df.empty:
        close_df = spread_line_df.loc[spread_line_df["vegas_home_margin"].abs() <= 3.0].copy()
        if not close_df.empty:
            correct = pd.to_numeric(close_df.get("winner_pick_correct"), errors="coerce").fillna(0.0).sum()
            close_spread_winner_accuracy = _safe_pct_value(float(correct), float(len(close_df)))

    winner_pick_correct = pd.to_numeric(base.loc[winner_pick_mask, "winner_pick_correct"], errors="coerce").fillna(0.0)
    return {
        "games": int(len(base)),
        "total_line_games": int(len(total_line_df)),
        "spread_line_games": int(len(spread_line_df)),
        "total_mae": round(float(total_line_df["total_abs_error"].mean()) if not total_line_df.empty else 0.0, 4),
        "spread_mae": round(float(spread_line_df["spread_abs_error"].mean()) if not spread_line_df.empty else 0.0, 4),
        "winner_pick_accuracy_pct": round(
            _safe_pct_value(float(winner_pick_correct.sum()), float(len(winner_pick_correct))),
            4,
        ),
        "high_total_over_rate_pct": round(float(high_total_over_rate), 4),
        "close_spread_winner_accuracy_pct": round(float(close_spread_winner_accuracy), 4),
    }


def build_game_slate_ai_review_packet(
    *,
    review_date: str,
    odds_df: pd.DataFrame,
    player_pool_df: pd.DataFrame | None = None,
    prior_boxscore_df: pd.DataFrame | None = None,
    vegas_history_df: pd.DataFrame | None = None,
    vegas_review_df: pd.DataFrame | None = None,
    focus_limit: int = 8,
) -> dict[str, Any]:
    odds = odds_df.copy() if isinstance(odds_df, pd.DataFrame) else pd.DataFrame()
    player_pool_df = player_pool_df.copy() if isinstance(player_pool_df, pd.DataFrame) else pd.DataFrame()
    prior_boxscore_df = prior_boxscore_df.copy() if isinstance(prior_boxscore_df, pd.DataFrame) else pd.DataFrame()
    vegas_history_df = vegas_history_df.copy() if isinstance(vegas_history_df, pd.DataFrame) else pd.DataFrame()
    vegas_review_df = vegas_review_df.copy() if isinstance(vegas_review_df, pd.DataFrame) else pd.DataFrame()
    focus_n = int(max(3, focus_limit))
    gpp_game_stack_targets, gpp_team_stack_targets, gpp_player_core_targets = _build_gpp_stack_targets_from_pool(
        player_pool_df=player_pool_df,
        focus_limit=focus_n,
    )

    if odds.empty:
        return {
            "schema_version": GAME_SLATE_AI_REVIEW_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "review_context": {
                "review_date": str(review_date),
                "odds_rows": 0,
                "player_pool_rows": int(len(player_pool_df)),
            },
            "market_summary": {
                "games": 0,
                "avg_total_line": 0.0,
                "avg_abs_spread": 0.0,
                "close_spread_games": 0,
                "high_total_games": 0,
                "heavy_favorite_games": 0,
            },
            "vegas_calibration": _summarize_vegas_history_for_game_slate(vegas_history_df),
            "today_vegas_review": {
                "rows": 0,
                "matched_rows": 0,
                "winner_pick_accuracy_pct": 0.0,
                "total_mae": 0.0,
                "spread_mae": 0.0,
            },
            "focus_tables": {
                "stack_candidates_top": [],
                "winner_calls": [],
                "upset_candidates": [],
                "gpp_game_stack_targets": gpp_game_stack_targets,
                "gpp_team_stack_targets": gpp_team_stack_targets,
                "gpp_player_core_targets": gpp_player_core_targets,
                "prior_day_team_form_top": [],
                "prior_day_high_concentration_teams": [],
                "games_table": [],
            },
            "notes_for_agent": [
                "No odds rows were available for the selected date.",
                "Use cached odds import before running this review.",
            ],
        }

    odds["game_date"] = (
        odds["game_date"].astype(str).str.strip()
        if "game_date" in odds.columns
        else pd.Series([""] * len(odds), index=odds.index)
    )
    odds["home_team"] = (
        odds["home_team"].astype(str).str.strip()
        if "home_team" in odds.columns
        else pd.Series([""] * len(odds), index=odds.index)
    )
    odds["away_team"] = (
        odds["away_team"].astype(str).str.strip()
        if "away_team" in odds.columns
        else pd.Series([""] * len(odds), index=odds.index)
    )
    odds["bookmakers_count"] = pd.to_numeric(
        odds["bookmakers_count"] if "bookmakers_count" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["total_points"] = pd.to_numeric(
        odds["total_points"] if "total_points" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["spread_home"] = pd.to_numeric(
        odds["spread_home"] if "spread_home" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["spread_away"] = pd.to_numeric(
        odds["spread_away"] if "spread_away" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["moneyline_home"] = pd.to_numeric(
        odds["moneyline_home"] if "moneyline_home" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["moneyline_away"] = pd.to_numeric(
        odds["moneyline_away"] if "moneyline_away" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["p_plus_8"] = pd.to_numeric(
        odds["p_plus_8"] if "p_plus_8" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["p_plus_12"] = pd.to_numeric(
        odds["p_plus_12"] if "p_plus_12" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )
    odds["tail_sigma"] = pd.to_numeric(
        odds["tail_sigma"] if "tail_sigma" in odds.columns else pd.Series(pd.NA, index=odds.index),
        errors="coerce",
    )

    odds["abs_spread"] = pd.to_numeric(odds["spread_home"], errors="coerce").abs()
    odds["abs_spread"] = odds["abs_spread"].where(odds["abs_spread"].notna(), pd.to_numeric(odds["spread_away"], errors="coerce").abs())
    odds["abs_spread"] = odds["abs_spread"].fillna(0.0)

    odds["implied_home_win_prob"] = odds["moneyline_home"].map(_american_to_implied_probability)
    odds["implied_away_win_prob"] = odds["moneyline_away"].map(_american_to_implied_probability)

    normalized_probs = odds.apply(
        lambda row: _normalize_win_probabilities(
            row.get("implied_home_win_prob"),
            row.get("implied_away_win_prob"),
        ),
        axis=1,
    )
    odds["home_win_prob"] = normalized_probs.map(lambda x: x[0] if isinstance(x, tuple) else None)
    odds["away_win_prob"] = normalized_probs.map(lambda x: x[1] if isinstance(x, tuple) else None)

    spread_conf = (odds["abs_spread"] * 0.035).clip(lower=0.0, upper=0.45)
    spread_home_favored = odds["spread_home"] <= 0
    spread_home_prob = pd.Series(0.5, index=odds.index, dtype="float64")
    spread_home_prob = spread_home_prob.where(~spread_home_favored, 0.5 + spread_conf)
    spread_home_prob = spread_home_prob.where(spread_home_favored, 0.5 - spread_conf)

    odds["home_win_prob"] = pd.to_numeric(odds["home_win_prob"], errors="coerce").where(
        pd.to_numeric(odds["home_win_prob"], errors="coerce").notna(),
        spread_home_prob,
    )
    odds["away_win_prob"] = pd.to_numeric(odds["away_win_prob"], errors="coerce").where(
        pd.to_numeric(odds["away_win_prob"], errors="coerce").notna(),
        1.0 - odds["home_win_prob"],
    )
    odds["home_win_prob"] = odds["home_win_prob"].clip(lower=0.01, upper=0.99)
    odds["away_win_prob"] = odds["away_win_prob"].clip(lower=0.01, upper=0.99)

    odds["favorite_team"] = odds.apply(
        lambda row: str(row.get("home_team") or "")
        if float(pd.to_numeric(row.get("home_win_prob"), errors="coerce") or 0.0)
        >= float(pd.to_numeric(row.get("away_win_prob"), errors="coerce") or 0.0)
        else str(row.get("away_team") or ""),
        axis=1,
    )
    odds["projected_winner"] = odds["favorite_team"]
    odds["winner_confidence_pct"] = (
        pd.concat([odds["home_win_prob"], odds["away_win_prob"]], axis=1).max(axis=1).fillna(0.5) * 100.0
    )

    odds["underdog_team"] = odds.apply(
        lambda row: str(row.get("home_team") or "")
        if float(pd.to_numeric(row.get("home_win_prob"), errors="coerce") or 0.0)
        < float(pd.to_numeric(row.get("away_win_prob"), errors="coerce") or 0.0)
        else str(row.get("away_team") or ""),
        axis=1,
    )
    odds["underdog_win_prob"] = pd.concat([odds["home_win_prob"], odds["away_win_prob"]], axis=1).min(axis=1)

    total_series = pd.to_numeric(odds["total_points"], errors="coerce")
    total_center = float(total_series.dropna().median()) if total_series.notna().any() else 150.0
    total_scale = float(total_series.dropna().std()) if total_series.notna().sum() >= 2 else 0.0
    if total_scale <= 1e-6:
        odds["total_z"] = 0.0
    else:
        odds["total_z"] = ((total_series - total_center) / total_scale).clip(lower=-3.0, upper=3.0).fillna(0.0)

    odds["close_game_score"] = (1.0 - (odds["abs_spread"] / 14.0)).clip(lower=0.0, upper=1.0)

    volatility = pd.to_numeric(odds["p_plus_12"], errors="coerce")
    if volatility.notna().any():
        odds["volatility_signal"] = volatility.clip(lower=0.0, upper=1.0).fillna(float(volatility.median()))
    else:
        volatility = pd.to_numeric(odds["p_plus_8"], errors="coerce")
        if volatility.notna().any():
            odds["volatility_signal"] = volatility.clip(lower=0.0, upper=1.0).fillna(float(volatility.median()))
        else:
            sigma = pd.to_numeric(odds["tail_sigma"], errors="coerce")
            if sigma.notna().any():
                sigma_center = float(sigma.dropna().median())
                sigma_scale = float(sigma.dropna().std()) if sigma.notna().sum() >= 2 else 0.0
                if sigma_scale <= 1e-6:
                    odds["volatility_signal"] = 0.5
                else:
                    odds["volatility_signal"] = ((sigma - sigma_center) / sigma_scale).clip(-2.0, 2.0)
                    odds["volatility_signal"] = (0.5 + (0.2 * odds["volatility_signal"])).clip(0.0, 1.0).fillna(0.5)
            else:
                odds["volatility_signal"] = 0.5

    odds["stack_score"] = (
        55.0
        + (14.0 * odds["total_z"])
        + (22.0 * odds["close_game_score"])
        + (16.0 * odds["volatility_signal"])
    ).clip(lower=0.0, upper=100.0)
    odds["blowout_risk_score"] = ((odds["abs_spread"] / 15.0) * 100.0).clip(lower=0.0, upper=100.0)
    odds["upset_score"] = (
        100.0 * ((0.65 * odds["underdog_win_prob"].fillna(0.35)) + (0.35 * odds["close_game_score"]))
    ).clip(lower=0.0, upper=100.0)

    prior_form_df = _build_prior_team_form_frame(prior_boxscore_df)
    vegas_calibration = _summarize_vegas_history_for_game_slate(vegas_history_df)

    today_vegas_summary = {
        "rows": 0,
        "matched_rows": 0,
        "winner_pick_accuracy_pct": 0.0,
        "total_mae": 0.0,
        "spread_mae": 0.0,
    }
    if not vegas_review_df.empty:
        today = vegas_review_df.copy()
        if "has_odds_match" in today.columns:
            today["has_odds_match"] = today["has_odds_match"].astype(bool)
        else:
            today["has_odds_match"] = False
        today["total_abs_error"] = pd.to_numeric(
            today["total_abs_error"] if "total_abs_error" in today.columns else pd.Series(pd.NA, index=today.index),
            errors="coerce",
        )
        today["spread_abs_error"] = pd.to_numeric(
            today["spread_abs_error"] if "spread_abs_error" in today.columns else pd.Series(pd.NA, index=today.index),
            errors="coerce",
        )
        today_actual_winner = (
            today["actual_winner_side"].astype(str)
            if "actual_winner_side" in today.columns
            else pd.Series([""] * len(today), index=today.index)
        )
        today_predicted_winner = (
            today["predicted_winner_side"].astype(str)
            if "predicted_winner_side" in today.columns
            else pd.Series([""] * len(today), index=today.index)
        )
        winner_mask = (
            today_actual_winner.isin(["home", "away"])
            & today_predicted_winner.isin(["home", "away"])
        )
        winner_correct = pd.to_numeric(today.loc[winner_mask, "winner_pick_correct"], errors="coerce").fillna(0.0)
        today_vegas_summary = {
            "rows": int(len(today)),
            "matched_rows": int(today["has_odds_match"].sum()),
            "winner_pick_accuracy_pct": round(
                _safe_pct_value(float(winner_correct.sum()), float(len(winner_correct))),
                4,
            ),
            "total_mae": round(float(today["total_abs_error"].mean()) if today["total_abs_error"].notna().any() else 0.0, 4),
            "spread_mae": round(float(today["spread_abs_error"].mean()) if today["spread_abs_error"].notna().any() else 0.0, 4),
        }

    games_table_cols = [
        "game_date",
        "event_id",
        "home_team",
        "away_team",
        "bookmakers_count",
        "total_points",
        "abs_spread",
        "spread_home",
        "spread_away",
        "moneyline_home",
        "moneyline_away",
        "favorite_team",
        "projected_winner",
        "winner_confidence_pct",
        "underdog_team",
        "underdog_win_prob",
        "stack_score",
        "upset_score",
        "blowout_risk_score",
        "close_game_score",
        "volatility_signal",
    ]
    games_export = odds[[c for c in games_table_cols if c in odds.columns]].copy()
    numeric_round_cols = [
        "bookmakers_count",
        "total_points",
        "abs_spread",
        "spread_home",
        "spread_away",
        "moneyline_home",
        "moneyline_away",
        "winner_confidence_pct",
        "underdog_win_prob",
        "stack_score",
        "upset_score",
        "blowout_risk_score",
        "close_game_score",
        "volatility_signal",
    ]
    for col in numeric_round_cols:
        if col in games_export.columns:
            games_export[col] = pd.to_numeric(games_export[col], errors="coerce").round(4)

    stack_candidates = _top_records(
        games_export,
        keep_cols=[
            "game_date",
            "home_team",
            "away_team",
            "total_points",
            "abs_spread",
            "stack_score",
            "blowout_risk_score",
            "projected_winner",
            "winner_confidence_pct",
            "volatility_signal",
        ],
        sort_col="stack_score",
        ascending=False,
        limit=focus_n,
    )
    winner_calls = _top_records(
        games_export,
        keep_cols=[
            "game_date",
            "home_team",
            "away_team",
            "projected_winner",
            "winner_confidence_pct",
            "favorite_team",
            "moneyline_home",
            "moneyline_away",
            "spread_home",
            "spread_away",
        ],
        sort_col="winner_confidence_pct",
        ascending=False,
        limit=max(focus_n, int(len(games_export))),
    )

    upset_pool = games_export.copy()
    upset_pool = upset_pool.loc[
        (pd.to_numeric(upset_pool.get("underdog_win_prob"), errors="coerce") >= 0.18)
        & (pd.to_numeric(upset_pool.get("underdog_win_prob"), errors="coerce") <= 0.48)
    ]
    if upset_pool.empty:
        upset_pool = games_export.copy()
    upset_candidates = _top_records(
        upset_pool,
        keep_cols=[
            "game_date",
            "home_team",
            "away_team",
            "underdog_team",
            "underdog_win_prob",
            "upset_score",
            "projected_winner",
            "winner_confidence_pct",
            "spread_home",
            "spread_away",
        ],
        sort_col="upset_score",
        ascending=False,
        limit=focus_n,
    )

    prior_day_team_form_top = (
        prior_form_df.head(focus_n).to_dict(orient="records") if not prior_form_df.empty else []
    )
    prior_day_high_concentration_teams = []
    if not prior_form_df.empty and "top3_dk_share_pct" in prior_form_df.columns:
        high_concentration = prior_form_df.sort_values("top3_dk_share_pct", ascending=False).head(focus_n)
        prior_day_high_concentration_teams = high_concentration.to_dict(orient="records")

    total_values = pd.to_numeric(odds["total_points"], errors="coerce")
    abs_spread_values = pd.to_numeric(odds["abs_spread"], errors="coerce")
    high_total_cutoff = float(total_values.quantile(0.75)) if total_values.notna().any() else 0.0
    market_summary = {
        "games": int(len(odds)),
        "avg_total_line": round(float(total_values.mean()) if total_values.notna().any() else 0.0, 4),
        "avg_abs_spread": round(float(abs_spread_values.mean()) if abs_spread_values.notna().any() else 0.0, 4),
        "close_spread_games": int((abs_spread_values <= 4.0).sum()),
        "high_total_games": int((total_values >= high_total_cutoff).sum()) if high_total_cutoff > 0 else 0,
        "heavy_favorite_games": int((abs_spread_values >= 8.0).sum()),
    }

    return {
        "schema_version": GAME_SLATE_AI_REVIEW_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "review_context": {
            "review_date": str(review_date),
            "odds_rows": int(len(odds)),
            "player_pool_rows": int(len(player_pool_df)),
            "prior_boxscore_rows": int(len(prior_boxscore_df)),
            "vegas_history_rows": int(len(vegas_history_df)),
            "vegas_review_rows": int(len(vegas_review_df)),
        },
        "market_summary": market_summary,
        "vegas_calibration": vegas_calibration,
        "today_vegas_review": today_vegas_summary,
        "focus_tables": {
            "stack_candidates_top": stack_candidates,
            "winner_calls": winner_calls,
            "upset_candidates": upset_candidates,
            "gpp_game_stack_targets": gpp_game_stack_targets,
            "gpp_team_stack_targets": gpp_team_stack_targets,
            "gpp_player_core_targets": gpp_player_core_targets,
            "prior_day_team_form_top": prior_day_team_form_top,
            "prior_day_high_concentration_teams": prior_day_high_concentration_teams,
            "games_table": games_export.head(max(focus_n * 4, 20)).to_dict(orient="records"),
        },
        "notes_for_agent": [
            "Prioritize stacks where stack_score is high and blowout_risk_score is moderate or low.",
            "Use gpp_game_stack_targets and gpp_player_core_targets first when they are available.",
            "Use winner_confidence_pct and upset_score for confidence tiers, not binary locks.",
            "If data is sparse for a game, mark confidence Low and explain what is missing.",
        ],
    }


def build_game_slate_ai_review_user_prompt(packet: dict[str, Any]) -> str:
    payload_json = json.dumps(packet, indent=2, ensure_ascii=True)
    return (
        "Review this college basketball game-slate packet and provide actionable DFS game-level guidance.\n\n"
        "Required output format:\n"
        "1) Slate Summary (3-5 bullets)\n"
        "2) GPP Game Stack Tiers (Tier 1/2/3; ranked; include why + risk)\n"
        "3) Team Stack Cores + Bring-Backs (specific teams and 2-3 player cores)\n"
        "4) Winner Leans by Game (team + confidence + one-line rationale)\n"
        "5) Upset/Leverage Angles (if any)\n"
        "6) Fade or One-Off-Only Games\n"
        "7) Action Plan for This Slate (max 8 bullets)\n\n"
        "Constraints:\n"
        "- Use only evidence in the JSON packet.\n"
        "- If `focus_tables.gpp_game_stack_targets` exists, anchor your stack section to those rows first.\n"
        "- If `focus_tables.gpp_player_core_targets` exists, name players directly from that table.\n"
        "- Provide at least 3 stack recommendations when there are 3+ games available.\n"
        "- Cite exact metric names/values when making claims.\n"
        "- Mark confidence per recommendation as High/Medium/Low.\n"
        "- If data is missing, say exactly what is missing.\n\n"
        "JSON packet:\n"
        f"{payload_json}\n"
    )


def _collect_response_text_parts(value: Any) -> list[str]:
    parts: list[str] = []
    if isinstance(value, str):
        text = value.strip()
        if text:
            parts.append(text)
        return parts
    if isinstance(value, list):
        for item in value:
            parts.extend(_collect_response_text_parts(item))
        return parts
    if isinstance(value, dict):
        for key in ("output_text", "text", "value", "refusal", "summary"):
            if key in value:
                parts.extend(_collect_response_text_parts(value.get(key)))
        return parts
    return parts


def _extract_openai_output_text(body: dict[str, Any]) -> str:
    collected: list[str] = []

    collected.extend(_collect_response_text_parts(body.get("output_text")))
    if collected:
        return "\n\n".join(collected)

    output = body.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            collected.extend(_collect_response_text_parts(item.get("refusal")))
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        collected.extend(_collect_response_text_parts(part))
                        continue
                    part_type = str(part.get("type") or "").strip().lower()
                    if part_type in {"output_text", "text", "refusal", "summary_text", "reasoning"}:
                        collected.extend(_collect_response_text_parts(part))
                    else:
                        collected.extend(_collect_response_text_parts(part.get("text")))
                        collected.extend(_collect_response_text_parts(part.get("value")))
            else:
                collected.extend(_collect_response_text_parts(content))
    if collected:
        return "\n\n".join(collected)

    status = str(body.get("status") or "").strip().lower()
    incomplete_details = body.get("incomplete_details")
    if status == "incomplete" and isinstance(incomplete_details, dict):
        reason = str(incomplete_details.get("reason") or "unknown")
        raise OpenAIReviewError(
            "OpenAI API response was incomplete and had no output text "
            f"(reason: {reason}). Try increasing max output tokens."
        )

    body_preview = json.dumps(body, ensure_ascii=True)[:1000]
    raise OpenAIReviewError(
        "OpenAI API response did not include output text. "
        f"Response preview: {body_preview}"
    )


def request_openai_review(
    *,
    api_key: str,
    user_prompt: str,
    system_prompt: str = AI_REVIEW_SYSTEM_PROMPT,
    model: str = DEFAULT_OPENAI_REVIEW_MODEL,
    max_output_tokens: int = 1800,
    timeout_seconds: int = 180,
    max_request_retries: int = 2,
    retry_backoff_seconds: float = 1.5,
) -> str:
    key = str(api_key or "").strip()
    if not key:
        raise OpenAIReviewError("OpenAI API key is required.")

    requested_model = str(model or "").strip()
    model_candidates: list[str] = []
    if requested_model:
        model_candidates.append(requested_model)
    for fallback_model in OPENAI_REVIEW_MODEL_FALLBACKS:
        if fallback_model not in model_candidates:
            model_candidates.append(fallback_model)

    base_payload = {
        "max_output_tokens": int(max(200, max_output_tokens)),
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": str(system_prompt or "").strip()}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": str(user_prompt or "").strip()}],
            },
        ],
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    model_errors: list[str] = []
    for candidate_model in model_candidates:
        payload = dict(base_payload)
        payload["model"] = candidate_model
        model_not_found = False
        attempts = int(max(0, max_request_retries)) + 1
        for attempt in range(attempts):
            try:
                response = requests.post(
                    "https://api.openai.com/v1/responses",
                    headers=headers,
                    json=payload,
                    timeout=timeout_seconds,
                )
            except requests.RequestException as exc:
                if attempt < attempts - 1:
                    delay = float(retry_backoff_seconds) * float(2**attempt)
                    time.sleep(max(0.0, delay))
                    continue
                raise OpenAIReviewError(
                    "OpenAI request failed after retries: "
                    f"{exc}. timeout_seconds={int(timeout_seconds)}"
                ) from exc

            if response.status_code >= 400:
                detail = response.text[:1000]
                detail_lower = detail.lower()
                retryable_model_error = (
                    response.status_code == 400
                    and ("model_not_found" in detail_lower or "does not exist" in detail_lower)
                )
                if retryable_model_error:
                    model_errors.append(f"{candidate_model}: {detail}")
                    model_not_found = True
                    break

                retryable_http_error = response.status_code in {408, 429, 500, 502, 503, 504}
                if retryable_http_error and attempt < attempts - 1:
                    delay = float(retry_backoff_seconds) * float(2**attempt)
                    time.sleep(max(0.0, delay))
                    continue

                raise OpenAIReviewError(f"OpenAI API error ({response.status_code}): {detail}")

            try:
                body = response.json()
            except ValueError as exc:
                if attempt < attempts - 1:
                    delay = float(retry_backoff_seconds) * float(2**attempt)
                    time.sleep(max(0.0, delay))
                    continue
                raise OpenAIReviewError("OpenAI API returned invalid JSON.") from exc

            return _extract_openai_output_text(body)

        if model_not_found:
            continue

    attempted = ", ".join(model_candidates)
    details = " | ".join(model_errors[-2:]) if model_errors else "no model details returned"
    raise OpenAIReviewError(
        "OpenAI model not available for this API key. "
        f"Tried: {attempted}. Set a valid model in the app's `OpenAI Model` field. "
        f"Last errors: {details}"
    )
