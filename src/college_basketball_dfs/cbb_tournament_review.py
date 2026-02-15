from __future__ import annotations

import re
from typing import Any

import pandas as pd

SALARY_CAP = 50000
NAME_SUFFIX_TOKENS = {"jr", "sr", "ii", "iii", "iv", "v"}


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


def parse_lineup_players(lineup_text: str) -> list[dict[str, str]]:
    if not str(lineup_text or "").strip():
        return []
    pattern = re.compile(r"\b(F|G|UTIL)\s+(.+?)(?=\s+(?:F|G|UTIL)\s+|$)")
    players: list[dict[str, str]] = []
    for match in pattern.finditer(str(lineup_text)):
        players.append(
            {
                "slot": match.group(1).strip().upper(),
                "player_name": match.group(2).strip(),
            }
        )
    return players


def normalize_contest_standings_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Rank", "EntryId", "EntryName", "Points", "Lineup", "Player", "%Drafted"])

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
    out["Points"] = pd.to_numeric(out["Points"], errors="coerce")
    out["Rank"] = pd.to_numeric(out["Rank"], errors="coerce")
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
    ]:
        if col not in out.columns:
            out[col] = default

    if expanded_players_df is None or expanded_players_df.empty:
        return out

    actual = actual_results_df.copy() if isinstance(actual_results_df, pd.DataFrame) else pd.DataFrame()
    if actual.empty:
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

    if "Points" in out.columns:
        out["computed_minus_file_points"] = pd.to_numeric(out["computed_actual_points"], errors="coerce") - pd.to_numeric(
            out["Points"], errors="coerce"
        )
    return out


def build_player_exposure_comparison(
    expanded_players_df: pd.DataFrame,
    entry_count: int,
    projection_df: pd.DataFrame | None,
    actual_ownership_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if expanded_players_df is None or expanded_players_df.empty or entry_count <= 0:
        return pd.DataFrame()

    expo = (
        expanded_players_df.groupby(["resolved_name", "TeamAbbrev"], as_index=False)
        .agg(appearances=("EntryId", "count"))
        .rename(columns={"resolved_name": "Name"})
    )
    expo["field_ownership_pct"] = (100.0 * expo["appearances"] / float(entry_count)).round(2)
    expo["name_key"] = expo["Name"].map(_norm)

    if projection_df is not None and not projection_df.empty:
        proj = projection_df.copy()
        if "Name" not in proj.columns:
            proj["Name"] = ""
        if "TeamAbbrev" not in proj.columns:
            proj["TeamAbbrev"] = ""
        proj["name_key"] = proj["Name"].map(_norm)
        proj["TeamAbbrev"] = proj["TeamAbbrev"].astype(str).str.strip().str.upper()
        keep = [
            c
            for c in ["name_key", "TeamAbbrev", "projected_ownership", "blended_projection", "our_dk_projection", "vegas_dk_projection"]
            if c in proj.columns
        ]
        proj_keep = proj[keep].drop_duplicates(["name_key", "TeamAbbrev"])
        expo = expo.merge(proj_keep, on=["name_key", "TeamAbbrev"], how="left")
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
        own = own.rename(columns={"actual_ownership": "actual_ownership_from_file"})
        expo = expo.merge(own[["name_key", "actual_ownership_from_file"]], on="name_key", how="left")
    else:
        expo["actual_ownership_from_file"] = pd.NA

    expo = expo.sort_values("field_ownership_pct", ascending=False).reset_index(drop=True)
    return expo.drop(columns=["name_key"])


def build_user_strategy_summary(entry_summary_df: pd.DataFrame) -> pd.DataFrame:
    if entry_summary_df is None or entry_summary_df.empty:
        return pd.DataFrame()

    users = entry_summary_df.copy()
    users["handle"] = users["EntryName"].astype(str).str.replace(r"\s+\(\d+/\d+\)$", "", regex=True)
    summary = (
        users.groupby("handle", as_index=False)
        .agg(
            entries=("EntryId", "count"),
            avg_points=("Points", "mean"),
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
        return out

    field_points = pd.to_numeric(field_entries_df["Points"], errors="coerce").dropna()
    field_n = int(len(field_points))
    if field_n <= 0:
        out["field_size"] = 0
        out["would_rank"] = pd.NA
        out["would_beat_pct"] = pd.NA
        return out

    ranks: list[int] = []
    beats: list[float] = []
    for _, row in out.iterrows():
        score = float(pd.to_numeric(row.get("actual_points"), errors="coerce") or 0.0)
        strictly_better = int((field_points > score).sum())
        strictly_worse = int((field_points < score).sum())
        ties = int((field_points == score).sum())
        rank_if_entered = 1 + strictly_better
        beat_pct = 100.0 * (strictly_worse + (0.5 * ties)) / float(field_n)
        ranks.append(rank_if_entered)
        beats.append(beat_pct)

    out["field_size"] = field_n
    out["would_rank"] = ranks
    out["would_beat_pct"] = beats
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
            ]
        )

    group_cols = [c for c in ["version_key", "version_label"] if c in phantom_df.columns]
    if not group_cols:
        group_cols = ["version_key"]
        phantom_df = phantom_df.copy()
        phantom_df["version_key"] = "unknown"

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
        )
    )
    out = out.sort_values(["best_actual_points", "avg_actual_points"], ascending=[False, False]).reset_index(drop=True)
    return out
