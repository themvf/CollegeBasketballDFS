from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests


AI_REVIEW_SCHEMA_VERSION = "v1"

AI_REVIEW_SYSTEM_PROMPT = (
    "You are an expert college basketball DFS review analyst. "
    "Use only evidence in the provided JSON packet. "
    "Prioritize actionable recommendations that can improve projection calibration, "
    "ownership calibration, and lineup construction. "
    "If confidence is low, state uncertainty explicitly."
)


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
    proj["abs_projection_error"] = pd.to_numeric(proj.get("blend_error"), errors="coerce").abs()

    keep_proj_cols = [
        "Name",
        "TeamAbbrev",
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
                    "field_ownership_pct",
                    "projected_ownership",
                    "actual_ownership_from_file",
                    "ownership_diff_vs_proj",
                    "abs_ownership_error",
                ]
                if c in exp.columns
            ]
        ].drop_duplicates(["Name", "TeamAbbrev"])
        merged = proj.merge(exp, on=["Name", "TeamAbbrev"], how="left")
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
        "4) Lineup Construction Recommendations (ranked by expected impact)\n"
        "5) High-Leverage Players/Archetypes to Investigate Next\n"
        "6) Next-Slate Experiment Plan (max 5 experiments; each includes hypothesis, implementation, and success metric)\n\n"
        "Constraints:\n"
        "- Use only evidence in the JSON packet.\n"
        "- Cite exact metric names/values when making a claim.\n"
        "- Label confidence as High/Medium/Low per recommendation.\n"
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
        "2) Projection Engine Global Changes (feature/process/model changes)\n"
        "3) Ownership Engine Global Changes\n"
        "4) Lineup Construction Global Changes\n"
        "5) Recurring Player Archetypes and Failure Modes\n"
        "6) 2-Week Implementation Plan (sequenced tasks + validation metrics)\n\n"
        "Constraints:\n"
        "- Use only evidence in the JSON packet.\n"
        "- Distinguish one-off noise vs recurring signal.\n"
        "- For each recommendation, include expected impact and confidence.\n"
        "- Cite exact metric names/values.\n\n"
        "JSON packet:\n"
        f"{payload_json}\n"
    )


def request_openai_review(
    *,
    api_key: str,
    user_prompt: str,
    system_prompt: str = AI_REVIEW_SYSTEM_PROMPT,
    model: str = "gpt-5.1-mini",
    max_output_tokens: int = 1800,
    timeout_seconds: int = 90,
) -> str:
    key = str(api_key or "").strip()
    if not key:
        raise OpenAIReviewError("OpenAI API key is required.")

    payload = {
        "model": str(model or "gpt-5.1-mini"),
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
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=payload,
            timeout=timeout_seconds,
        )
    except requests.RequestException as exc:
        raise OpenAIReviewError(f"OpenAI request failed: {exc}") from exc

    if response.status_code >= 400:
        detail = response.text[:1000]
        raise OpenAIReviewError(f"OpenAI API error ({response.status_code}): {detail}")

    try:
        body = response.json()
    except ValueError as exc:
        raise OpenAIReviewError("OpenAI API returned invalid JSON.") from exc

    direct_text = body.get("output_text")
    if isinstance(direct_text, str) and direct_text.strip():
        return direct_text.strip()

    output = body.get("output")
    collected: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "").strip().lower()
                text = part.get("text")
                if part_type in {"output_text", "text"} and isinstance(text, str) and text.strip():
                    collected.append(text.strip())
    if collected:
        return "\n\n".join(collected)

    raise OpenAIReviewError("OpenAI API response did not include output text.")
