from __future__ import annotations

import re
from typing import Any

import pandas as pd

SALARY_CAP = 50000
NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}
HIGH_POINTS_THRESHOLD = 35.0
LOW_OWNERSHIP_THRESHOLD = 10.0
NULLISH_TEXT_VALUES = {"", "nan", "none", "null"}
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
LINEUP_SLOT_TOKENS = (
    "PG",
    "SG",
    "SF",
    "PF",
    "C",
    "G",
    "F",
    "UTIL",
    "FLEX",
    "CPT",
    "QB",
    "RB",
    "WR",
    "TE",
    "DST",
    "D",
    "P",
    "C1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "M",
    "W",
)
LINEUP_SLOT_SET = set(LINEUP_SLOT_TOKENS)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _norm_loose(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [tok for tok in text.split() if tok and tok not in NAME_SUFFIX_TOKENS]
    return "".join(tokens)


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


def _norm_col_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").strip().lower())


def _resolve_column_alias(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    if df is None or df.empty:
        return None
    normalized = {_norm_col_key(c): str(c) for c in df.columns}
    for alias in aliases:
        key = _norm_col_key(alias)
        if key in normalized:
            return normalized[key]
    return None


def parse_lineup_players(lineup_text: str) -> list[dict[str, str]]:
    text = str(lineup_text or "").strip()
    if not text:
        return []
    players: list[dict[str, str]] = []
    tokens = re.split(r"\s+", text)
    idx = 0
    while idx < len(tokens):
        slot = str(tokens[idx] or "").strip().upper().rstrip(",:;")
        if slot not in LINEUP_SLOT_SET:
            idx += 1
            continue
        j = idx + 1
        player_tokens: list[str] = []
        while j < len(tokens):
            next_slot = str(tokens[j] or "").strip().upper().rstrip(",:;")
            if next_slot in LINEUP_SLOT_SET:
                break
            player_tokens.append(tokens[j])
            j += 1
        player_name = " ".join(player_tokens).strip(" ,;|-")
        if player_name:
            players.append({"slot": slot, "player_name": player_name})
        idx = j
    if players:
        return players
    # Fallback for comma-separated names without slot labels.
    parts = [p.strip(" ,;|-") for p in re.split(r"\s*,\s*", text) if p.strip(" ,;|-")]
    if len(parts) >= 6:
        return [{"slot": "UTIL", "player_name": p} for p in parts]
    return players


def detect_contest_standings_upload(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {
            "kind": "empty",
            "is_usable": False,
            "message": "No upload loaded.",
            "row_count": 0,
            "column_count": 0,
            "lineup_nonempty_rows": 0,
            "columns_preview": [],
        }

    cols = [str(c).strip() for c in df.columns]
    col_keys = {_norm_col_key(c) for c in cols if str(c).strip()}
    lineup_col = next((c for c in df.columns if _norm_col_key(c) == "lineup"), None)
    lineup_nonempty_rows = 0
    if lineup_col is not None and lineup_col in df.columns:
        lineup_vals = df[lineup_col].astype(str).str.strip().str.lower()
        lineup_nonempty_rows = int((~lineup_vals.isin(NULLISH_TEXT_VALUES)).sum())

    has_rank_or_points = ("rank" in col_keys) or ("points" in col_keys)
    has_entry_cols = {"entryid", "entryname", "lineup"}.issubset(col_keys)
    looks_like_standings = has_entry_cols and has_rank_or_points
    looks_like_dk_entries_template = {"entryid", "contestname", "contestid", "entryfee"}.issubset(col_keys) and (
        "lineup" not in col_keys
    )
    looks_like_dk_salaries = {"nameid", "rosterposition", "salary", "gameinfo", "teamabbrev"}.issubset(col_keys)

    kind = "unknown"
    is_usable = False
    message = (
        "Unrecognized CSV format for Tournament Review. Upload DraftKings contest standings "
        "(`contest-standings-<contest_id>.csv`)."
    )
    if looks_like_standings:
        kind = "contest_standings"
        is_usable = True
        message = "Contest standings format detected."
        if lineup_nonempty_rows <= 0:
            message = (
                "Contest standings columns are present, but `Lineup` rows are empty. "
                "Contest may not have started or export is incomplete."
            )
    elif looks_like_dk_entries_template:
        kind = "dk_entries_template"
        message = (
            "This looks like a DraftKings entry-upload template (`DKEntries_*.csv`). "
            "Tournament Review needs contest standings export (`contest-standings-<contest_id>.csv`)."
        )
    elif looks_like_dk_salaries:
        kind = "dk_salaries"
        message = (
            "This looks like a DraftKings salaries slate file (`DKSalaries*.csv`). "
            "Tournament Review needs contest standings export (`contest-standings-<contest_id>.csv`)."
        )
    elif df.empty:
        kind = "empty"
        message = "Uploaded CSV loaded but has no rows."

    return {
        "kind": kind,
        "is_usable": bool(is_usable),
        "message": message,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "lineup_nonempty_rows": int(lineup_nonempty_rows),
        "columns_preview": cols[:12],
    }


def normalize_contest_standings_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "Rank",
                "rank_from_file",
                "rank_from_points",
                "EntryId",
                "EntryName",
                "Points",
                "Lineup",
                "Player",
                "%Drafted",
            ]
        )

    out = df.copy()
    unnamed = [c for c in out.columns if str(c).strip().lower().startswith("unnamed")]
    if unnamed:
        out = out.drop(columns=unnamed)

    rename_map = {
        "rank": "Rank",
        "entryid": "EntryId",
        "entryname": "EntryName",
        "points": "Points",
        "lineup": "Lineup",
        "player": "Player",
        "%drafted": "%Drafted",
        "drafted": "%Drafted",
        "timeRemaining": "TimeRemaining",
        "timeremaining": "TimeRemaining",
    }
    out = out.rename(columns={c: rename_map.get(str(c).strip().lower(), c) for c in out.columns})

    for col in ["Rank", "EntryId", "EntryName", "Points", "Lineup", "Player", "%Drafted"]:
        if col not in out.columns:
            out[col] = ""

    out["EntryId"] = out["EntryId"].astype(str).str.strip()
    out["EntryName"] = out["EntryName"].astype(str).str.strip()
    out["Lineup"] = out["Lineup"].astype(str).str.strip()
    out["Player"] = out["Player"].astype(str).str.strip()
    points_num = pd.to_numeric(out["Points"], errors="coerce")
    rank_file = pd.to_numeric(out["Rank"], errors="coerce")
    if points_num.notna().any():
        rank_points = points_num.rank(method="min", ascending=False)
        out["Rank"] = rank_points
        out["rank_from_points"] = rank_points
    else:
        out["Rank"] = rank_file
        out["rank_from_points"] = pd.NA
    out["rank_from_file"] = rank_file
    out["Points"] = points_num
    out["%Drafted"] = out["%Drafted"].map(_to_float)
    return out


def extract_actual_ownership_from_standings(standings_df: pd.DataFrame) -> pd.DataFrame:
    if standings_df.empty:
        return pd.DataFrame(columns=["player_name", "actual_ownership"])
    cols = [c for c in ["Player", "%Drafted"] if c in standings_df.columns]
    if len(cols) < 2:
        return pd.DataFrame(columns=["player_name", "actual_ownership"])
    own = standings_df[cols].copy()
    own = own.rename(columns={"Player": "player_name", "%Drafted": "actual_ownership"})
    own["player_name"] = own["player_name"].astype(str).str.strip()
    own["actual_ownership"] = pd.to_numeric(own["actual_ownership"], errors="coerce")
    own = own.loc[(own["player_name"] != "") & own["actual_ownership"].notna()]
    if own.empty:
        return pd.DataFrame(columns=["player_name", "actual_ownership"])
    return own.groupby("player_name", as_index=False)["actual_ownership"].max()


def build_field_entries_and_players(
    standings_df: pd.DataFrame,
    slate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = normalize_contest_standings_frame(standings_df)
    if s.empty:
        return pd.DataFrame(), pd.DataFrame()

    if slate_df is None or slate_df.empty:
        slate = pd.DataFrame(columns=["Name", "Salary", "TeamAbbrev", "Game Info"])
    else:
        slate = slate_df.copy()
    for col in ["Name", "Salary", "TeamAbbrev", "Game Info"]:
        if col not in slate.columns:
            slate[col] = ""

    slate["name_key"] = slate["Name"].map(_norm)
    slate["Salary"] = pd.to_numeric(slate["Salary"], errors="coerce")
    slate["TeamAbbrev"] = slate["TeamAbbrev"].astype(str).str.strip().str.upper()
    slate["game_key"] = slate["Game Info"].astype(str).str.strip().str.split(" ").str[0].str.upper()
    slate_map = slate.dropna(subset=["name_key"]).drop_duplicates("name_key")

    expanded_rows: list[dict[str, Any]] = []
    for _, row in s.iterrows():
        lineup_players = parse_lineup_players(str(row.get("Lineup") or ""))
        for p in lineup_players:
            expanded_rows.append(
                {
                    "EntryId": str(row.get("EntryId") or "").strip(),
                    "EntryName": str(row.get("EntryName") or "").strip(),
                    "Rank": pd.to_numeric(row.get("Rank"), errors="coerce"),
                    "Points": pd.to_numeric(row.get("Points"), errors="coerce"),
                    "slot": p["slot"],
                    "player_name": p["player_name"],
                    "name_key": _norm(p["player_name"]),
                }
            )
    expanded = pd.DataFrame(expanded_rows)
    if expanded.empty:
        return pd.DataFrame(), pd.DataFrame()

    expanded = expanded.merge(
        slate_map[["name_key", "Name", "Salary", "TeamAbbrev", "game_key"]],
        on="name_key",
        how="left",
    )
    expanded["resolved_name"] = expanded["Name"].where(expanded["Name"].notna(), expanded["player_name"])

    team_counts = (
        expanded.dropna(subset=["TeamAbbrev"])
        .groupby(["EntryId", "TeamAbbrev"], as_index=False)
        .size()
        .rename(columns={"size": "team_count"})
    )
    game_counts = (
        expanded.dropna(subset=["game_key"])
        .groupby(["EntryId", "game_key"], as_index=False)
        .size()
        .rename(columns={"size": "game_count"})
    )
    max_team = team_counts.groupby("EntryId", as_index=False)["team_count"].max().rename(columns={"team_count": "max_team_stack"})
    max_game = game_counts.groupby("EntryId", as_index=False)["game_count"].max().rename(columns={"game_count": "max_game_stack"})

    entry_summary = (
        expanded.groupby("EntryId", as_index=False)
        .agg(
            EntryName=("EntryName", "first"),
            Rank=("Rank", "min"),
            Points=("Points", "max"),
            parsed_players=("player_name", "count"),
            mapped_players=("Salary", lambda x: int(pd.Series(x).notna().sum())),
            salary_used=("Salary", "sum"),
            unique_teams=("TeamAbbrev", lambda x: int(pd.Series(x).dropna().nunique())),
            unique_games=("game_key", lambda x: int(pd.Series(x).dropna().nunique())),
        )
    )
    entry_summary["salary_used"] = pd.to_numeric(entry_summary["salary_used"], errors="coerce").fillna(0.0)
    entry_summary["salary_left"] = SALARY_CAP - entry_summary["salary_used"]
    entry_summary = entry_summary.merge(max_team, on="EntryId", how="left").merge(max_game, on="EntryId", how="left")
    entry_summary["max_team_stack"] = entry_summary["max_team_stack"].fillna(0).astype(int)
    entry_summary["max_game_stack"] = entry_summary["max_game_stack"].fillna(0).astype(int)
    entry_summary = entry_summary.sort_values(["Rank", "Points"], ascending=[True, False]).reset_index(drop=True)

    return entry_summary, expanded


def build_entry_actual_points_comparison(
    entry_summary_df: pd.DataFrame,
    expanded_players_df: pd.DataFrame,
    actual_results_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if entry_summary_df is None or entry_summary_df.empty:
        return pd.DataFrame()
    out = entry_summary_df.copy()
    for col, default in [
        ("computed_actual_points", 0.0),
        ("computed_players_matched", 0),
        ("computed_coverage_pct", 0.0),
        ("computed_minus_file_points", pd.NA),
        ("points_from_file", pd.NA),
        ("rank_from_computed_points", pd.NA),
    ]:
        if col not in out.columns:
            out[col] = default
    if "Points" in out.columns:
        out["points_from_file"] = pd.to_numeric(out["Points"], errors="coerce")

    if expanded_players_df is None or expanded_players_df.empty:
        # No computed actuals available; leave uploaded values as-is.
        return out

    actual = actual_results_df.copy() if isinstance(actual_results_df, pd.DataFrame) else pd.DataFrame()
    if actual.empty:
        # No computed actuals available; leave uploaded values as-is.
        return out
    if "Name" not in actual.columns:
        actual["Name"] = ""
    if "actual_dk_points" not in actual.columns:
        actual["actual_dk_points"] = pd.NA
    actual["Name"] = actual["Name"].astype(str).str.strip()
    actual["actual_dk_points"] = pd.to_numeric(actual["actual_dk_points"], errors="coerce")
    actual = actual.loc[actual["actual_dk_points"].notna()].copy()
    if actual.empty:
        return out

    by_name = (
        actual.assign(name_key=actual["Name"].map(_norm))
        .loc[lambda x: x["name_key"] != ""]
        .groupby("name_key", as_index=False)["actual_dk_points"]
        .mean(numeric_only=True)
        .set_index("name_key")["actual_dk_points"]
        .to_dict()
    )
    by_name_loose = (
        actual.assign(name_key_loose=actual["Name"].map(_norm_loose))
        .loc[lambda x: x["name_key_loose"] != ""]
        .groupby("name_key_loose", as_index=False)["actual_dk_points"]
        .mean(numeric_only=True)
        .set_index("name_key_loose")["actual_dk_points"]
        .to_dict()
    )

    expanded = expanded_players_df.copy()
    if "EntryId" not in expanded.columns:
        return out
    if "resolved_name" not in expanded.columns:
        expanded["resolved_name"] = expanded.get("player_name", "")
    expanded["resolved_name"] = expanded["resolved_name"].astype(str).str.strip()
    expanded["name_key"] = expanded["resolved_name"].map(_norm)
    expanded["name_key_loose"] = expanded["resolved_name"].map(_norm_loose)
    expanded["actual_points_match"] = expanded["name_key"].map(by_name)
    missing_mask = expanded["actual_points_match"].isna()
    expanded.loc[missing_mask, "actual_points_match"] = expanded.loc[missing_mask, "name_key_loose"].map(by_name_loose)

    entry_actual = (
        expanded.groupby("EntryId", as_index=False)
        .agg(
            computed_actual_points=("actual_points_match", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
            computed_players_matched=("actual_points_match", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
            parsed_players=("resolved_name", "count"),
        )
    )
    entry_actual["computed_coverage_pct"] = (
        100.0 * pd.to_numeric(entry_actual["computed_players_matched"], errors="coerce")
        / pd.to_numeric(entry_actual["parsed_players"], errors="coerce").replace(0, pd.NA)
    ).fillna(0.0)

    out = out.merge(
        entry_actual[["EntryId", "computed_actual_points", "computed_players_matched", "computed_coverage_pct"]],
        on="EntryId",
        how="left",
        suffixes=("", "_new"),
    )
    for col in ["computed_actual_points", "computed_players_matched", "computed_coverage_pct"]:
        new_col = f"{col}_new"
        if new_col in out.columns:
            out[col] = pd.to_numeric(out[new_col], errors="coerce").fillna(out[col])
            out = out.drop(columns=[new_col])

    if "points_from_file" in out.columns:
        out["computed_minus_file_points"] = pd.to_numeric(out["computed_actual_points"], errors="coerce") - pd.to_numeric(
            out["points_from_file"], errors="coerce"
        )

    # Canonical standings values for Tournament Review: use computed actuals/rank.
    out["Points"] = pd.to_numeric(out["computed_actual_points"], errors="coerce")
    out["Rank"] = pd.to_numeric(out["Points"], errors="coerce").rank(method="min", ascending=False)
    out["rank_from_computed_points"] = out["Rank"]
    return out


def build_player_exposure_comparison(
    expanded_players_df: pd.DataFrame,
    entry_count: int,
    projection_df: pd.DataFrame | None,
    actual_ownership_df: pd.DataFrame | None = None,
    actual_results_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if expanded_players_df is None or expanded_players_df.empty or entry_count <= 0:
        return pd.DataFrame()

    expanded = expanded_players_df.copy()
    if "resolved_name" not in expanded.columns:
        expanded["resolved_name"] = expanded.get("player_name", "")
    if "TeamAbbrev" not in expanded.columns:
        expanded["TeamAbbrev"] = ""
    expanded["resolved_name"] = expanded["resolved_name"].astype(str).str.strip()
    expanded["TeamAbbrev"] = expanded["TeamAbbrev"].astype(str).str.strip().str.upper()
    expanded["TeamAbbrev"] = expanded["TeamAbbrev"].replace({"NAN": "", "NONE": "", "NULL": ""})
    expanded = expanded.loc[expanded["resolved_name"] != ""].copy()
    if expanded.empty:
        return pd.DataFrame()

    expo = (
        expanded.groupby(["resolved_name", "TeamAbbrev"], as_index=False)
        .agg(appearances=("EntryId", "count"))
        .rename(columns={"resolved_name": "Name"})
    )
    expo["field_ownership_pct"] = (100.0 * expo["appearances"] / float(entry_count)).round(2)
    expo["name_key"] = expo["Name"].map(_norm)
    expo["name_key_loose"] = expo["Name"].map(_norm_loose)

    if projection_df is not None and not projection_df.empty:
        proj = projection_df.copy()
        if "Name" not in proj.columns:
            proj["Name"] = ""
        if "TeamAbbrev" not in proj.columns:
            proj["TeamAbbrev"] = ""
        projected_ownership_col = _resolve_column_alias(proj, PROJECTED_OWNERSHIP_ALIASES)
        if not projected_ownership_col:
            projected_ownership_col = "projected_ownership"
            proj[projected_ownership_col] = pd.NA
        elif projected_ownership_col != "projected_ownership":
            proj["projected_ownership"] = proj[projected_ownership_col]
        proj["name_key"] = proj["Name"].map(_norm)
        proj["name_key_loose"] = proj["Name"].map(_norm_loose)
        proj["TeamAbbrev"] = proj["TeamAbbrev"].astype(str).str.strip().str.upper()
        proj["TeamAbbrev"] = proj["TeamAbbrev"].replace({"NAN": "", "NONE": "", "NULL": ""})
        keep = [
            c
            for c in ["name_key", "TeamAbbrev", "projected_ownership", "blended_projection", "our_dk_projection", "vegas_dk_projection"]
            if c in proj.columns
        ]
        proj_keep = proj[keep].drop_duplicates(["name_key", "TeamAbbrev"])
        expo = expo.merge(proj_keep, on=["name_key", "TeamAbbrev"], how="left")
        fallback_keep = [c for c in ["name_key", "name_key_loose", "projected_ownership", "blended_projection", "our_dk_projection", "vegas_dk_projection"] if c in proj.columns]
        proj_name_keep = proj[fallback_keep].drop_duplicates(["name_key"])
        expo = expo.merge(
            proj_name_keep,
            on="name_key",
            how="left",
            suffixes=("", "_name_fallback"),
        )
        for col in ["projected_ownership", "blended_projection", "our_dk_projection", "vegas_dk_projection"]:
            fb_col = f"{col}_name_fallback"
            if col not in expo.columns:
                expo[col] = pd.NA
            if fb_col in expo.columns:
                expo[col] = pd.to_numeric(expo[col], errors="coerce").where(
                    pd.to_numeric(expo[col], errors="coerce").notna(),
                    pd.to_numeric(expo[fb_col], errors="coerce"),
                )
                expo = expo.drop(columns=[fb_col], errors="ignore")
        expo = expo.drop(columns=["name_key_loose_name_fallback"], errors="ignore")
        if "projected_ownership" not in expo.columns:
            expo["projected_ownership"] = pd.NA
        expo["ownership_diff_vs_proj"] = pd.to_numeric(expo["field_ownership_pct"], errors="coerce") - pd.to_numeric(
            expo["projected_ownership"], errors="coerce"
        )
    else:
        expo["projected_ownership"] = pd.NA
        expo["ownership_diff_vs_proj"] = pd.NA

    if actual_ownership_df is not None and not actual_ownership_df.empty:
        own = actual_ownership_df.copy()
        own["name_key"] = own["player_name"].map(_norm)
        own["name_key_loose"] = own["player_name"].map(_norm_loose)
        own["actual_ownership_from_file"] = pd.to_numeric(own.get("actual_ownership"), errors="coerce")
        by_name = (
            own.loc[(own["name_key"] != "") & own["actual_ownership_from_file"].notna()]
            .groupby("name_key", as_index=False)["actual_ownership_from_file"]
            .max()
            .set_index("name_key")["actual_ownership_from_file"]
            .to_dict()
        )
        by_name_loose = (
            own.loc[(own["name_key_loose"] != "") & own["actual_ownership_from_file"].notna()]
            .groupby("name_key_loose", as_index=False)["actual_ownership_from_file"]
            .max()
            .set_index("name_key_loose")["actual_ownership_from_file"]
            .to_dict()
        )
        expo["actual_ownership_from_file"] = expo["name_key"].map(by_name)
        missing_own = expo["actual_ownership_from_file"].isna()
        expo.loc[missing_own, "actual_ownership_from_file"] = expo.loc[missing_own, "name_key_loose"].map(by_name_loose)
    else:
        expo["actual_ownership_from_file"] = pd.NA

    if actual_results_df is not None and not actual_results_df.empty:
        actual = actual_results_df.copy()
        if "Name" not in actual.columns:
            actual["Name"] = ""
        if "actual_dk_points" not in actual.columns:
            actual["actual_dk_points"] = pd.NA
        actual["Name"] = actual["Name"].astype(str).str.strip()
        actual["actual_dk_points"] = pd.to_numeric(actual["actual_dk_points"], errors="coerce")
        actual = actual.loc[actual["actual_dk_points"].notna()]
        if actual.empty:
            expo["final_dk_points"] = pd.NA
        else:
            by_name = (
                actual.assign(name_key=actual["Name"].map(_norm))
                .loc[lambda x: x["name_key"] != ""]
                .groupby("name_key", as_index=False)["actual_dk_points"]
                .mean(numeric_only=True)
                .set_index("name_key")["actual_dk_points"]
                .to_dict()
            )
            by_name_loose = (
                actual.assign(name_key_loose=actual["Name"].map(_norm_loose))
                .loc[lambda x: x["name_key_loose"] != ""]
                .groupby("name_key_loose", as_index=False)["actual_dk_points"]
                .mean(numeric_only=True)
                .set_index("name_key_loose")["actual_dk_points"]
                .to_dict()
            )
            expo["final_dk_points"] = expo["name_key"].map(by_name)
            missing = expo["final_dk_points"].isna()
            expo.loc[missing, "final_dk_points"] = expo.loc[missing, "name_key_loose"].map(by_name_loose)
    else:
        expo["final_dk_points"] = pd.NA

    expo["high_points_low_own_flag"] = (
        pd.to_numeric(expo["final_dk_points"], errors="coerce").fillna(0.0) >= HIGH_POINTS_THRESHOLD
    ) & (
        pd.to_numeric(expo["field_ownership_pct"], errors="coerce").fillna(0.0) <= LOW_OWNERSHIP_THRESHOLD
    )

    expo = expo.sort_values("field_ownership_pct", ascending=False).reset_index(drop=True)
    return expo.drop(columns=["name_key", "name_key_loose"])


def build_projection_actual_comparison(
    projection_df: pd.DataFrame | None,
    actual_results_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if projection_df is None or projection_df.empty:
        return pd.DataFrame()
    if actual_results_df is None or actual_results_df.empty:
        return pd.DataFrame()

    proj = projection_df.copy()
    actual = actual_results_df.copy()

    for col in ["ID", "Name", "TeamAbbrev", "Position"]:
        if col not in proj.columns:
            proj[col] = ""
    for col in ["ID", "Name"]:
        if col not in actual.columns:
            actual[col] = ""
    if "actual_dk_points" not in actual.columns:
        actual["actual_dk_points"] = pd.NA
    if "actual_minutes" not in actual.columns:
        actual["actual_minutes"] = pd.NA

    proj["ID"] = proj["ID"].astype(str).str.strip()
    proj["Name"] = proj["Name"].astype(str).str.strip()
    proj["TeamAbbrev"] = proj["TeamAbbrev"].astype(str).str.strip().str.upper()
    proj["Position"] = proj["Position"].astype(str).str.strip().str.upper()
    proj["name_key"] = proj["Name"].map(_norm)
    proj["name_key_loose"] = proj["Name"].map(_norm_loose)

    actual["ID"] = actual["ID"].astype(str).str.strip()
    actual["Name"] = actual["Name"].astype(str).str.strip()
    actual["actual_dk_points"] = pd.to_numeric(actual["actual_dk_points"], errors="coerce")
    actual["actual_minutes"] = pd.to_numeric(actual["actual_minutes"], errors="coerce")
    actual["name_key"] = actual["Name"].map(_norm)
    actual["name_key_loose"] = actual["Name"].map(_norm_loose)

    by_id_points = (
        actual.loc[actual["ID"] != ""]
        .groupby("ID", as_index=False)["actual_dk_points"]
        .mean(numeric_only=True)
        .set_index("ID")["actual_dk_points"]
        .to_dict()
    )
    by_id_minutes = (
        actual.loc[actual["ID"] != ""]
        .groupby("ID", as_index=False)["actual_minutes"]
        .mean(numeric_only=True)
        .set_index("ID")["actual_minutes"]
        .to_dict()
    )
    by_name_points = (
        actual.loc[actual["name_key"] != ""]
        .groupby("name_key", as_index=False)["actual_dk_points"]
        .mean(numeric_only=True)
        .set_index("name_key")["actual_dk_points"]
        .to_dict()
    )
    by_name_minutes = (
        actual.loc[actual["name_key"] != ""]
        .groupby("name_key", as_index=False)["actual_minutes"]
        .mean(numeric_only=True)
        .set_index("name_key")["actual_minutes"]
        .to_dict()
    )
    by_name_loose_points = (
        actual.loc[actual["name_key_loose"] != ""]
        .groupby("name_key_loose", as_index=False)["actual_dk_points"]
        .mean(numeric_only=True)
        .set_index("name_key_loose")["actual_dk_points"]
        .to_dict()
    )
    by_name_loose_minutes = (
        actual.loc[actual["name_key_loose"] != ""]
        .groupby("name_key_loose", as_index=False)["actual_minutes"]
        .mean(numeric_only=True)
        .set_index("name_key_loose")["actual_minutes"]
        .to_dict()
    )

    proj["actual_dk_points"] = proj["ID"].map(by_id_points)
    proj["actual_minutes"] = proj["ID"].map(by_id_minutes)

    missing_points = proj["actual_dk_points"].isna()
    proj.loc[missing_points, "actual_dk_points"] = proj.loc[missing_points, "name_key"].map(by_name_points)
    missing_points = proj["actual_dk_points"].isna()
    proj.loc[missing_points, "actual_dk_points"] = proj.loc[missing_points, "name_key_loose"].map(by_name_loose_points)

    missing_minutes = proj["actual_minutes"].isna()
    proj.loc[missing_minutes, "actual_minutes"] = proj.loc[missing_minutes, "name_key"].map(by_name_minutes)
    missing_minutes = proj["actual_minutes"].isna()
    proj.loc[missing_minutes, "actual_minutes"] = proj.loc[missing_minutes, "name_key_loose"].map(by_name_loose_minutes)

    for col in [
        "Salary",
        "blended_projection",
        "our_dk_projection",
        "vegas_dk_projection",
        "our_minutes_avg",
        "our_minutes_last7",
        "actual_dk_points",
        "actual_minutes",
    ]:
        if col in proj.columns:
            proj[col] = pd.to_numeric(proj[col], errors="coerce")
        else:
            proj[col] = pd.NA

    proj["blend_error"] = proj["actual_dk_points"] - proj["blended_projection"]
    proj["our_error"] = proj["actual_dk_points"] - proj["our_dk_projection"]
    proj["vegas_error"] = proj["actual_dk_points"] - proj["vegas_dk_projection"]
    proj["minutes_error_avg"] = proj["actual_minutes"] - proj["our_minutes_avg"]
    proj["minutes_error_last7"] = proj["actual_minutes"] - proj["our_minutes_last7"]
    proj["our_multiplier"] = proj["actual_dk_points"] / proj["our_dk_projection"].replace(0, pd.NA)
    proj["minutes_multiplier_avg"] = proj["actual_minutes"] / proj["our_minutes_avg"].replace(0, pd.NA)
    proj["minutes_multiplier_last7"] = proj["actual_minutes"] / proj["our_minutes_last7"].replace(0, pd.NA)

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
    keep_cols = [c for c in show_cols if c in proj.columns]
    out = proj[keep_cols].copy()
    sort_cols = [c for c in ["actual_dk_points", "blended_projection"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)
    return out.reset_index(drop=True)


def build_projection_adjustment_factors(comparison_df: pd.DataFrame) -> pd.DataFrame:
    if comparison_df is None or comparison_df.empty:
        return pd.DataFrame(
            columns=[
                "segment",
                "samples",
                "our_points_multiplier",
                "blended_points_multiplier",
                "minutes_multiplier_avg",
                "minutes_multiplier_last7",
                "our_mae",
                "blend_mae",
                "minutes_mae_avg",
                "minutes_mae_last7",
            ]
        )

    def _segment_frame(df: pd.DataFrame, label: str, mask: pd.Series | None = None) -> dict[str, Any]:
        seg = df.loc[mask].copy() if mask is not None else df.copy()
        seg = seg.loc[pd.to_numeric(seg.get("actual_dk_points"), errors="coerce").notna()]
        if seg.empty:
            return {
                "segment": label,
                "samples": 0,
                "our_points_multiplier": 0.0,
                "blended_points_multiplier": 0.0,
                "minutes_multiplier_avg": 0.0,
                "minutes_multiplier_last7": 0.0,
                "our_mae": 0.0,
                "blend_mae": 0.0,
                "minutes_mae_avg": 0.0,
                "minutes_mae_last7": 0.0,
            }

        actual_pts = pd.to_numeric(seg["actual_dk_points"], errors="coerce")
        our_pts = pd.to_numeric(seg.get("our_dk_projection"), errors="coerce")
        blend_pts = pd.to_numeric(seg.get("blended_projection"), errors="coerce")
        actual_min = pd.to_numeric(seg.get("actual_minutes"), errors="coerce")
        our_min_avg = pd.to_numeric(seg.get("our_minutes_avg"), errors="coerce")
        our_min_last7 = pd.to_numeric(seg.get("our_minutes_last7"), errors="coerce")

        def _ratio(num: pd.Series, den: pd.Series) -> float:
            den_mean = float(den.mean()) if den.notna().any() else 0.0
            num_mean = float(num.mean()) if num.notna().any() else 0.0
            if den_mean == 0.0:
                return 0.0
            return num_mean / den_mean

        return {
            "segment": label,
            "samples": int(len(seg)),
            "our_points_multiplier": _ratio(actual_pts, our_pts),
            "blended_points_multiplier": _ratio(actual_pts, blend_pts),
            "minutes_multiplier_avg": _ratio(actual_min, our_min_avg),
            "minutes_multiplier_last7": _ratio(actual_min, our_min_last7),
            "our_mae": float((actual_pts - our_pts).abs().mean()) if our_pts.notna().any() else 0.0,
            "blend_mae": float((actual_pts - blend_pts).abs().mean()) if blend_pts.notna().any() else 0.0,
            "minutes_mae_avg": float((actual_min - our_min_avg).abs().mean()) if our_min_avg.notna().any() else 0.0,
            "minutes_mae_last7": (
                float((actual_min - our_min_last7).abs().mean()) if our_min_last7.notna().any() else 0.0
            ),
        }

    pos = comparison_df.get("Position", pd.Series(dtype=str)).astype(str).str.upper()
    rows = [
        _segment_frame(comparison_df, "All"),
        _segment_frame(comparison_df, "Guard (G)", pos.str.startswith("G")),
        _segment_frame(comparison_df, "Forward (F)", pos.str.startswith("F")),
    ]
    return pd.DataFrame(rows)


def build_user_strategy_summary(entry_summary_df: pd.DataFrame) -> pd.DataFrame:
    if entry_summary_df is None or entry_summary_df.empty:
        return pd.DataFrame()

    users = entry_summary_df.copy()
    users["handle"] = users["EntryName"].astype(str).str.replace(r"\s+\(\d+/\d+\)$", "", regex=True)
    points_col = "computed_actual_points" if "computed_actual_points" in users.columns else "Points"
    summary = (
        users.groupby("handle", as_index=False)
        .agg(
            entries=("EntryId", "count"),
            avg_points=(points_col, "mean"),
            most_points=(points_col, "max"),
            best_rank=("Rank", "min"),
            avg_salary_left=("salary_left", "mean"),
            avg_team_stack=("max_team_stack", "mean"),
            avg_game_stack=("max_game_stack", "mean"),
        )
        .sort_values(["entries", "avg_points"], ascending=[False, False])
        .reset_index(drop=True)
    )
    return summary


def summarize_generated_lineups(generated_lineups: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lineup in generated_lineups or []:
        players = lineup.get("players", [])
        teams = [str(p.get("TeamAbbrev") or "").strip().upper() for p in players if str(p.get("TeamAbbrev") or "").strip()]
        games = [str(p.get("game_key") or "").strip().upper() for p in players if str(p.get("game_key") or "").strip()]
        max_team_stack = 0
        if teams:
            counts = pd.Series(teams).value_counts()
            max_team_stack = int(counts.max())
        max_game_stack = 0
        if games:
            counts = pd.Series(games).value_counts()
            max_game_stack = int(counts.max())

        salary = float(lineup.get("salary") or 0.0)
        rows.append(
            {
                "lineup_number": lineup.get("lineup_number"),
                "salary_used": salary,
                "salary_left": SALARY_CAP - salary,
                "projected_points": lineup.get("projected_points"),
                "max_team_stack": max_team_stack,
                "max_game_stack": max_game_stack,
            }
        )
    return pd.DataFrame(rows)


def build_top10_winner_gap_analysis(
    entries_df: pd.DataFrame | None,
    expanded_players_df: pd.DataFrame | None,
    projection_comparison_df: pd.DataFrame | None,
    generated_lineups: list[dict[str, Any]] | None = None,
    top_n_winners: int = 10,
    top_points_focus: int = 10,
) -> dict[str, Any]:
    base_summary: dict[str, Any] = {
        "top10_entries_count": 0,
        "top10_unique_players": 0,
        "our_lineups_available": False,
        "our_lineups_count": 0,
        "top_scorer_name": "",
        "top_scorer_actual_points": pd.NA,
        "top_scorer_in_our_lineups": pd.NA,
        "top3_covered_count": pd.NA,
        "top3_target_count": 0,
        "top3_all_in_single_lineup": pd.NA,
        "top3_missing_names": [],
        "top5_covered_count": pd.NA,
        "top5_target_count": 0,
        "top5_missing_names": [],
    }
    empty_top_df = pd.DataFrame(
        columns=[
            "Name",
            "TeamAbbrev",
            "actual_dk_points",
            "blended_projection",
            "our_dk_projection",
            "vegas_dk_projection",
            "blend_error",
            "top10_entries_with_player",
            "top10_entry_rate_pct",
            "our_lineups_with_player",
            "our_lineup_rate_pct",
            "missed_by_our_lineups",
            "actual_rank",
        ]
    )
    empty_hits_df = pd.DataFrame(columns=["top3_hits", "lineups", "lineup_rate_pct"])
    out = {
        "summary": base_summary,
        "top10_entries_df": pd.DataFrame(),
        "top_players_df": empty_top_df.copy(),
        "focus_players_df": empty_top_df.copy(),
        "missing_focus_players_df": empty_top_df.copy(),
        "lineup_top3_hit_distribution_df": empty_hits_df.copy(),
    }

    if entries_df is None or entries_df.empty or expanded_players_df is None or expanded_players_df.empty:
        return out
    if "EntryId" not in entries_df.columns or "EntryId" not in expanded_players_df.columns:
        return out

    n_winners = max(1, int(top_n_winners or 10))
    n_focus = max(1, int(top_points_focus or 10))

    entries = entries_df.copy()
    entries["EntryId"] = entries["EntryId"].astype(str).str.strip()
    entries = entries.loc[entries["EntryId"] != ""].copy()
    if entries.empty:
        return out

    if "Rank" in entries.columns:
        entries["_rank_num"] = pd.to_numeric(entries["Rank"], errors="coerce")
        entries = entries.sort_values(["_rank_num"], ascending=[True], na_position="last")
    top10_entries = entries.drop_duplicates("EntryId").head(n_winners).copy()
    top10_entry_ids = set(top10_entries["EntryId"].astype(str).str.strip().tolist())
    if not top10_entry_ids:
        return out

    expanded = expanded_players_df.copy()
    expanded["EntryId"] = expanded["EntryId"].astype(str).str.strip()
    expanded = expanded.loc[expanded["EntryId"].isin(top10_entry_ids)].copy()
    if expanded.empty:
        out["top10_entries_df"] = top10_entries.reset_index(drop=True)
        out["summary"]["top10_entries_count"] = int(len(top10_entries))
        return out

    if "resolved_name" not in expanded.columns:
        expanded["resolved_name"] = expanded.get("player_name", "")
    expanded["resolved_name"] = expanded["resolved_name"].astype(str).str.strip()
    expanded["resolved_name"] = expanded["resolved_name"].replace({"nan": "", "none": "", "null": ""})
    if "TeamAbbrev" not in expanded.columns:
        expanded["TeamAbbrev"] = ""
    expanded["TeamAbbrev"] = expanded["TeamAbbrev"].astype(str).str.strip().str.upper()
    expanded["name_key"] = expanded["resolved_name"].map(_norm)
    expanded["name_key_loose"] = expanded["resolved_name"].map(_norm_loose)
    expanded = expanded.loc[(expanded["resolved_name"] != "") & (expanded["name_key"] != "")].copy()
    if expanded.empty:
        out["top10_entries_df"] = top10_entries.reset_index(drop=True)
        out["summary"]["top10_entries_count"] = int(len(top10_entries))
        return out

    player_entry_rows = expanded.drop_duplicates(["EntryId", "name_key"])
    player_counts = (
        player_entry_rows.groupby("name_key", as_index=False)
        .agg(
            top10_entries_with_player=("EntryId", "nunique"),
            name_key_loose=("name_key_loose", "first"),
            Name=("resolved_name", lambda s: s.mode().iloc[0] if not s.mode().empty else str(s.iloc[0])),
            TeamAbbrev=("TeamAbbrev", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
        )
        .reset_index(drop=True)
    )
    player_counts["top10_entry_rate_pct"] = (
        100.0
        * pd.to_numeric(player_counts["top10_entries_with_player"], errors="coerce").fillna(0.0)
        / float(max(1, len(top10_entries)))
    )

    if projection_comparison_df is not None and not projection_comparison_df.empty:
        proj = projection_comparison_df.copy()
        if "Name" not in proj.columns:
            proj["Name"] = ""
        proj["Name"] = proj["Name"].astype(str).str.strip()
        proj["name_key"] = proj["Name"].map(_norm)
        proj["name_key_loose"] = proj["Name"].map(_norm_loose)
        proj["_actual_sort"] = pd.to_numeric(proj.get("actual_dk_points"), errors="coerce").fillna(-1e9)
        proj = proj.sort_values("_actual_sort", ascending=False)
        proj_by_key = proj.loc[proj["name_key"] != ""].drop_duplicates("name_key")
        proj_by_loose = proj.loc[proj["name_key_loose"] != ""].drop_duplicates("name_key_loose")
        map_cols = [
            "actual_dk_points",
            "blended_projection",
            "our_dk_projection",
            "vegas_dk_projection",
            "blend_error",
            "our_error",
            "vegas_error",
            "Salary",
            "Position",
        ]
        for col in map_cols:
            if col not in player_counts.columns:
                player_counts[col] = pd.NA
            key_map = (
                proj_by_key.set_index("name_key")[col].to_dict() if (col in proj_by_key.columns and not proj_by_key.empty) else {}
            )
            loose_map = (
                proj_by_loose.set_index("name_key_loose")[col].to_dict()
                if (col in proj_by_loose.columns and not proj_by_loose.empty)
                else {}
            )
            player_counts[col] = player_counts["name_key"].map(key_map)
            missing = player_counts[col].isna()
            if missing.any():
                player_counts.loc[missing, col] = player_counts.loc[missing, "name_key_loose"].map(loose_map)
    else:
        for col in [
            "actual_dk_points",
            "blended_projection",
            "our_dk_projection",
            "vegas_dk_projection",
            "blend_error",
            "our_error",
            "vegas_error",
            "Salary",
            "Position",
        ]:
            player_counts[col] = pd.NA

    for col in [
        "actual_dk_points",
        "blended_projection",
        "our_dk_projection",
        "vegas_dk_projection",
        "blend_error",
        "our_error",
        "vegas_error",
        "Salary",
    ]:
        if col in player_counts.columns:
            player_counts[col] = pd.to_numeric(player_counts[col], errors="coerce")

    our_lineups_available = bool(generated_lineups)
    our_lineups_count = 0
    lineup_top3_hit_distribution_df = empty_hits_df.copy()
    if our_lineups_available:
        our_rows: list[dict[str, Any]] = []
        for idx, lineup in enumerate(generated_lineups or []):
            lineup_uid = str(lineup.get("lineup_number") or "").strip() or f"lineup_{idx + 1}"
            players = lineup.get("players") or []
            for p in players:
                player_name = str(p.get("Name") or "").strip()
                if not player_name:
                    continue
                our_rows.append(
                    {
                        "lineup_uid": lineup_uid,
                        "name_key": _norm(player_name),
                        "name_key_loose": _norm_loose(player_name),
                    }
                )
        our_players_df = pd.DataFrame(our_rows)
        if not our_players_df.empty:
            our_players_df = our_players_df.loc[our_players_df["name_key"] != ""].copy()
            our_players_df = our_players_df.drop_duplicates(["lineup_uid", "name_key"])
            our_lineups_count = int(our_players_df["lineup_uid"].nunique())
            if our_lineups_count > 0:
                our_key_counts = (
                    our_players_df.groupby("name_key", as_index=False)
                    .agg(our_lineups_with_player=("lineup_uid", "nunique"))
                    .set_index("name_key")["our_lineups_with_player"]
                    .to_dict()
                )
                our_loose_counts = (
                    our_players_df.groupby("name_key_loose", as_index=False)
                    .agg(our_lineups_with_player=("lineup_uid", "nunique"))
                    .set_index("name_key_loose")["our_lineups_with_player"]
                    .to_dict()
                )
                player_counts["our_lineups_with_player"] = pd.to_numeric(
                    player_counts["name_key"].map(our_key_counts), errors="coerce"
                )
                missing_our = player_counts["our_lineups_with_player"].isna()
                if missing_our.any():
                    player_counts.loc[missing_our, "our_lineups_with_player"] = pd.to_numeric(
                        player_counts.loc[missing_our, "name_key_loose"].map(our_loose_counts),
                        errors="coerce",
                    )
                player_counts["our_lineups_with_player"] = (
                    pd.to_numeric(player_counts["our_lineups_with_player"], errors="coerce").fillna(0).astype(int)
                )
                player_counts["our_lineup_rate_pct"] = (
                    100.0
                    * pd.to_numeric(player_counts["our_lineups_with_player"], errors="coerce").fillna(0.0)
                    / float(max(1, our_lineups_count))
                )
                player_counts["missed_by_our_lineups"] = pd.to_numeric(
                    player_counts["our_lineups_with_player"], errors="coerce"
                ).fillna(0) <= 0
            else:
                our_lineups_available = False
                player_counts["our_lineups_with_player"] = pd.NA
                player_counts["our_lineup_rate_pct"] = pd.NA
                player_counts["missed_by_our_lineups"] = pd.NA
        else:
            our_lineups_available = False
            player_counts["our_lineups_with_player"] = pd.NA
            player_counts["our_lineup_rate_pct"] = pd.NA
            player_counts["missed_by_our_lineups"] = pd.NA
    else:
        player_counts["our_lineups_with_player"] = pd.NA
        player_counts["our_lineup_rate_pct"] = pd.NA
        player_counts["missed_by_our_lineups"] = pd.NA

    player_counts["actual_rank"] = pd.to_numeric(player_counts["actual_dk_points"], errors="coerce").rank(
        method="min", ascending=False
    )
    player_counts = player_counts.sort_values(
        ["actual_dk_points", "top10_entries_with_player"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)

    focus_df = player_counts.loc[pd.to_numeric(player_counts["actual_dk_points"], errors="coerce").notna()].head(n_focus).copy()
    if focus_df.empty:
        focus_df = player_counts.head(n_focus).copy()

    missing_focus_df = empty_top_df.copy()
    top3_missing_names: list[str] = []
    top5_missing_names: list[str] = []
    top_scorer_in_our_lineups: bool | Any = pd.NA
    top3_covered_count: int | Any = pd.NA
    top5_covered_count: int | Any = pd.NA
    top3_all_in_single_lineup: bool | Any = pd.NA
    top_scorer_name = ""
    top_scorer_actual_points: Any = pd.NA

    top_by_points = player_counts.loc[pd.to_numeric(player_counts["actual_dk_points"], errors="coerce").notna()].copy()
    top3_df = top_by_points.head(3).copy()
    top5_df = top_by_points.head(5).copy()
    if not top_by_points.empty:
        top_scorer_name = str(top_by_points.iloc[0].get("Name") or "")
        top_scorer_actual_points = pd.to_numeric(top_by_points.iloc[0].get("actual_dk_points"), errors="coerce")

    if our_lineups_available:
        if "our_lineups_with_player" not in focus_df.columns:
            focus_df["our_lineups_with_player"] = 0
        missing_focus_df = focus_df.loc[
            pd.to_numeric(focus_df["our_lineups_with_player"], errors="coerce").fillna(0) <= 0
        ].copy()

        if not top_by_points.empty:
            top_scorer_in_our_lineups = bool(
                pd.to_numeric(top_by_points.iloc[0].get("our_lineups_with_player"), errors="coerce") > 0
            )
        if not top3_df.empty:
            top3_covered_count = int(
                (pd.to_numeric(top3_df["our_lineups_with_player"], errors="coerce").fillna(0) > 0).sum()
            )
            top3_missing_names = (
                top3_df.loc[pd.to_numeric(top3_df["our_lineups_with_player"], errors="coerce").fillna(0) <= 0, "Name"]
                .astype(str)
                .tolist()
            )
        if not top5_df.empty:
            top5_covered_count = int(
                (pd.to_numeric(top5_df["our_lineups_with_player"], errors="coerce").fillna(0) > 0).sum()
            )
            top5_missing_names = (
                top5_df.loc[pd.to_numeric(top5_df["our_lineups_with_player"], errors="coerce").fillna(0) <= 0, "Name"]
                .astype(str)
                .tolist()
            )

        if generated_lineups and not top3_df.empty:
            top_targets = top3_df[["name_key", "name_key_loose", "Name"]].copy()
            hit_rows: list[dict[str, Any]] = []
            for idx, lineup in enumerate(generated_lineups or []):
                lineup_uid = str(lineup.get("lineup_number") or "").strip() or f"lineup_{idx + 1}"
                players = lineup.get("players") or []
                lineup_key_set: set[str] = set()
                lineup_loose_set: set[str] = set()
                for p in players:
                    player_name = str(p.get("Name") or "").strip()
                    if not player_name:
                        continue
                    name_key = _norm(player_name)
                    name_key_loose = _norm_loose(player_name)
                    if name_key:
                        lineup_key_set.add(name_key)
                    if name_key_loose:
                        lineup_loose_set.add(name_key_loose)
                hit_count = 0
                for _, target in top_targets.iterrows():
                    target_key = str(target.get("name_key") or "").strip()
                    target_loose = str(target.get("name_key_loose") or "").strip()
                    if (target_key and target_key in lineup_key_set) or (target_loose and target_loose in lineup_loose_set):
                        hit_count += 1
                hit_rows.append({"lineup_uid": lineup_uid, "top3_hits": int(hit_count)})

            lineup_hits_df = pd.DataFrame(hit_rows)
            if not lineup_hits_df.empty:
                lineup_top3_hit_distribution_df = (
                    lineup_hits_df.groupby("top3_hits", as_index=False)
                    .agg(lineups=("lineup_uid", "count"))
                    .sort_values("top3_hits", ascending=False)
                    .reset_index(drop=True)
                )
                lineup_top3_hit_distribution_df["lineup_rate_pct"] = (
                    100.0
                    * pd.to_numeric(lineup_top3_hit_distribution_df["lineups"], errors="coerce").fillna(0.0)
                    / float(max(1, our_lineups_count))
                )
                required_hits = int(len(top_targets))
                top3_all_in_single_lineup = bool(
                    (pd.to_numeric(lineup_hits_df["top3_hits"], errors="coerce").fillna(0).astype(int) >= required_hits).any()
                )

    base_summary.update(
        {
            "top10_entries_count": int(len(top10_entries)),
            "top10_unique_players": int(player_counts["name_key"].nunique()),
            "our_lineups_available": bool(our_lineups_available),
            "our_lineups_count": int(our_lineups_count),
            "top_scorer_name": top_scorer_name,
            "top_scorer_actual_points": top_scorer_actual_points,
            "top_scorer_in_our_lineups": top_scorer_in_our_lineups,
            "top3_covered_count": top3_covered_count,
            "top3_target_count": int(len(top3_df)),
            "top3_all_in_single_lineup": top3_all_in_single_lineup,
            "top3_missing_names": top3_missing_names,
            "top5_covered_count": top5_covered_count,
            "top5_target_count": int(len(top5_df)),
            "top5_missing_names": top5_missing_names,
        }
    )

    drop_cols = ["name_key", "name_key_loose"]
    out["summary"] = base_summary
    out["top10_entries_df"] = top10_entries.reset_index(drop=True)
    out["top_players_df"] = player_counts.drop(columns=drop_cols, errors="ignore")
    out["focus_players_df"] = focus_df.drop(columns=drop_cols, errors="ignore")
    out["missing_focus_players_df"] = missing_focus_df.drop(columns=drop_cols, errors="ignore")
    out["lineup_top3_hit_distribution_df"] = lineup_top3_hit_distribution_df
    return out


def score_generated_lineups_against_actuals(
    generated_lineups: list[dict[str, Any]],
    actual_results_df: pd.DataFrame | None,
    version_key: str = "",
    version_label: str = "",
) -> pd.DataFrame:
    cols = [
        "version_key",
        "version_label",
        "lineup_number",
        "lineup_strategy",
        "pair_id",
        "pair_role",
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
        "projected_ownership_sum",
        "actual_points",
        "actual_minus_projected",
        "matched_players",
        "missing_players",
        "coverage_pct",
        "missing_names",
    ]
    if not generated_lineups:
        return pd.DataFrame(columns=cols)

    actual = actual_results_df.copy() if isinstance(actual_results_df, pd.DataFrame) else pd.DataFrame()
    if actual.empty:
        actual = pd.DataFrame(columns=["ID", "Name", "actual_dk_points"])
    if "ID" not in actual.columns:
        actual["ID"] = ""
    if "Name" not in actual.columns:
        actual["Name"] = ""
    if "actual_dk_points" not in actual.columns:
        actual["actual_dk_points"] = pd.NA
    actual["ID"] = actual["ID"].astype(str).str.strip()
    actual["Name"] = actual["Name"].astype(str).str.strip()
    actual["actual_dk_points"] = pd.to_numeric(actual["actual_dk_points"], errors="coerce")

    by_id = (
        actual.loc[actual["ID"] != "", ["ID", "actual_dk_points"]]
        .dropna(subset=["actual_dk_points"])
        .drop_duplicates("ID")
        .set_index("ID")["actual_dk_points"]
        .to_dict()
    )
    by_name = (
        actual.assign(name_key=actual["Name"].map(_norm))
        .loc[lambda x: x["name_key"] != ""]
        .groupby("name_key", as_index=False)["actual_dk_points"]
        .mean(numeric_only=True)
        .set_index("name_key")["actual_dk_points"]
        .to_dict()
    )
    by_name_loose = (
        actual.assign(name_key_loose=actual["Name"].map(_norm_loose))
        .loc[lambda x: x["name_key_loose"] != ""]
        .groupby("name_key_loose", as_index=False)["actual_dk_points"]
        .mean(numeric_only=True)
        .set_index("name_key_loose")["actual_dk_points"]
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    for lineup in generated_lineups:
        players = lineup.get("players") or []
        total_actual = 0.0
        matched = 0
        missing_names: list[str] = []
        for p in players:
            pid = str(p.get("ID") or "").strip()
            player_name = str(p.get("Name") or "").strip()
            actual_points = None
            if pid and pid in by_id:
                actual_points = by_id.get(pid)
            elif player_name:
                actual_points = by_name.get(_norm(player_name))
                if actual_points is None or pd.isna(actual_points):
                    actual_points = by_name_loose.get(_norm_loose(player_name))
            if actual_points is None or pd.isna(actual_points):
                missing_names.append(player_name or pid or "unknown")
                continue
            total_actual += float(actual_points)
            matched += 1

        salary_used = float(lineup.get("salary") or 0.0)
        projected = float(lineup.get("projected_points") or 0.0)
        total_slots = max(1, len(players))
        coverage_pct = 100.0 * float(matched) / float(total_slots)
        rows.append(
            {
                "version_key": version_key,
                "version_label": version_label,
                "lineup_number": lineup.get("lineup_number"),
                "lineup_strategy": lineup.get("lineup_strategy"),
                "pair_id": lineup.get("pair_id"),
                "pair_role": lineup.get("pair_role"),
                "cluster_id": lineup.get("cluster_id"),
                "cluster_script": lineup.get("cluster_script"),
                "anchor_game_key": lineup.get("anchor_game_key"),
                "seed_lineup_id": lineup.get("seed_lineup_id"),
                "mutation_type": lineup.get("mutation_type"),
                "stack_signature": lineup.get("stack_signature"),
                "salary_texture_bucket": lineup.get("salary_texture_bucket"),
                "salary_used": salary_used,
                "salary_left": SALARY_CAP - salary_used,
                "projected_points": projected,
                "projected_ownership_sum": float(lineup.get("projected_ownership_sum") or 0.0),
                "actual_points": total_actual,
                "actual_minus_projected": total_actual - projected,
                "matched_players": int(matched),
                "missing_players": int(max(0, total_slots - matched)),
                "coverage_pct": coverage_pct,
                "missing_names": " | ".join(missing_names),
            }
        )

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        return out
    out = out.sort_values(["actual_points", "lineup_number"], ascending=[False, True]).reset_index(drop=True)
    return out


def compare_phantom_entries_to_field(
    phantom_df: pd.DataFrame,
    field_entries_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if phantom_df is None or phantom_df.empty:
        return pd.DataFrame()
    out = phantom_df.copy()
    if field_entries_df is None or field_entries_df.empty or "Points" not in field_entries_df.columns:
        out["field_size"] = 0
        out["would_rank"] = pd.NA
        out["would_beat_pct"] = pd.NA
        out["field_best_points"] = pd.NA
        out["winner_gap"] = pd.NA
        out["pct_of_winner"] = pd.NA
        return out

    field_points = pd.to_numeric(field_entries_df["Points"], errors="coerce").dropna()
    field_n = int(len(field_points))
    if field_n <= 0:
        out["field_size"] = 0
        out["would_rank"] = pd.NA
        out["would_beat_pct"] = pd.NA
        out["field_best_points"] = pd.NA
        out["winner_gap"] = pd.NA
        out["pct_of_winner"] = pd.NA
        return out

    field_best_points = float(field_points.max())
    ranks: list[int] = []
    beats: list[float] = []
    winner_gaps: list[float] = []
    pct_of_winners: list[float] = []
    for _, row in out.iterrows():
        score = float(pd.to_numeric(row.get("actual_points"), errors="coerce") or 0.0)
        strictly_better = int((field_points > score).sum())
        strictly_worse = int((field_points < score).sum())
        ties = int((field_points == score).sum())
        rank_if_entered = 1 + strictly_better
        beat_pct = 100.0 * (strictly_worse + (0.5 * ties)) / float(field_n)
        winner_gap = field_best_points - score
        pct_of_winner = (100.0 * score / field_best_points) if field_best_points > 0 else 0.0
        ranks.append(rank_if_entered)
        beats.append(beat_pct)
        winner_gaps.append(winner_gap)
        pct_of_winners.append(pct_of_winner)

    out["field_size"] = field_n
    out["would_rank"] = ranks
    out["would_beat_pct"] = beats
    out["field_best_points"] = field_best_points
    out["winner_gap"] = winner_gaps
    out["pct_of_winner"] = pct_of_winners
    return out


def summarize_phantom_entries(phantom_df: pd.DataFrame) -> pd.DataFrame:
    if phantom_df is None or phantom_df.empty:
        return pd.DataFrame(
            columns=[
                "version_key",
                "version_label",
                "lineups",
                "avg_actual_points",
                "p90_actual_points",
                "best_actual_points",
                "avg_projected_points",
                "avg_actual_minus_projected",
                "avg_coverage_pct",
                "avg_salary_left",
                "avg_would_beat_pct",
                "best_would_rank",
                "winner_points",
                "winner_gap",
                "pct_of_winner",
            ]
        )

    group_cols = [c for c in ["version_key", "version_label"] if c in phantom_df.columns]
    if not group_cols:
        group_cols = ["version_key"]
        phantom_df = phantom_df.copy()
        phantom_df["version_key"] = "unknown"
    else:
        phantom_df = phantom_df.copy()

    for required_col in ["field_best_points", "would_beat_pct", "would_rank"]:
        if required_col not in phantom_df.columns:
            phantom_df[required_col] = pd.NA

    out = (
        phantom_df.groupby(group_cols, as_index=False)
        .agg(
            lineups=("lineup_number", "count"),
            avg_actual_points=("actual_points", "mean"),
            p90_actual_points=("actual_points", lambda s: float(pd.to_numeric(s, errors="coerce").quantile(0.90))),
            best_actual_points=("actual_points", "max"),
            avg_projected_points=("projected_points", "mean"),
            avg_actual_minus_projected=("actual_minus_projected", "mean"),
            avg_coverage_pct=("coverage_pct", "mean"),
            avg_salary_left=("salary_left", "mean"),
            avg_would_beat_pct=("would_beat_pct", "mean"),
            best_would_rank=("would_rank", "min"),
            winner_points=("field_best_points", "max"),
        )
    )
    out["winner_gap"] = pd.to_numeric(out["winner_points"], errors="coerce") - pd.to_numeric(
        out["best_actual_points"], errors="coerce"
    )
    out["pct_of_winner"] = (
        100.0
        * pd.to_numeric(out["best_actual_points"], errors="coerce")
        / pd.to_numeric(out["winner_points"], errors="coerce")
    )
    out["pct_of_winner"] = out["pct_of_winner"].where(pd.to_numeric(out["winner_points"], errors="coerce") > 0)
    out = out.sort_values(["best_actual_points", "avg_actual_points"], ascending=[False, False]).reset_index(drop=True)
    return out
