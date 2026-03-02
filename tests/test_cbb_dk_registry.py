from __future__ import annotations

import pandas as pd

from college_basketball_dfs.cbb_dk_registry import (
    build_dk_identity_registry,
    build_rotowire_dk_slate,
    extract_registry_rows_from_dk_slate,
)


def test_build_dk_identity_registry_tracks_latest_salary_and_seen_count() -> None:
    slate_day_one = pd.DataFrame(
        [
            {
                "Position": "G",
                "Name + ID": "John Doe (1001)",
                "Name": "John Doe",
                "ID": "1001",
                "Roster Position": "G/UTIL",
                "Salary": 7200,
                "Game Info": "TEAMA@TEAMB 02/20/2026 07:00PM ET",
                "TeamAbbrev": "TEAMA",
                "AvgPointsPerGame": 28.5,
            }
        ]
    )
    slate_day_two = pd.DataFrame(
        [
            {
                "Position": "G",
                "Name + ID": "John Doe (1001)",
                "Name": "John Doe",
                "ID": "1001",
                "Roster Position": "G/UTIL",
                "Salary": 7600,
                "Game Info": "TEAMA@TEAMC 02/21/2026 07:00PM ET",
                "TeamAbbrev": "TEAMA",
                "AvgPointsPerGame": 29.1,
            }
        ]
    )

    history = pd.concat(
        [
            extract_registry_rows_from_dk_slate(slate_day_one, slate_date="2026-02-20", slate_key="main", source_name="day1"),
            extract_registry_rows_from_dk_slate(slate_day_two, slate_date="2026-02-21", slate_key="main", source_name="day2"),
        ],
        ignore_index=True,
    )
    registry = build_dk_identity_registry(history)

    assert len(registry) == 1
    row = registry.iloc[0]
    assert row["dk_id"] == "1001"
    assert int(row["seen_count"]) == 2
    assert float(row["latest_salary"]) == 7600.0
    assert str(row["last_opp_abbr"]) == "TEAMC"


def test_build_rotowire_dk_slate_resolves_full_match() -> None:
    dk_slate = pd.DataFrame(
        [
            {
                "Position": "F",
                "Name + ID": "Jane Smith (2002)",
                "Name": "Jane Smith",
                "ID": "2002",
                "Roster Position": "F/UTIL",
                "Salary": 6800,
                "Game Info": "AWAY@HOME 02/22/2026 08:00PM ET",
                "TeamAbbrev": "AWAY",
                "AvgPointsPerGame": 24.2,
            }
        ]
    )
    registry = build_dk_identity_registry(
        extract_registry_rows_from_dk_slate(dk_slate, slate_date="2026-02-22", slate_key="main", source_name="sample")
    )
    rotowire_df = pd.DataFrame(
        [
            {
                "rw_id": 88,
                "player_name": "Jane Smith",
                "team_abbr": "AWAY",
                "opp_abbr": "HOME",
                "salary": 6800,
                "roto_position": "F",
                "site_positions": "F",
                "avg_fpts_season": 24.2,
                "is_home": False,
                "game_datetime": "2026-02-22T20:00:00",
            }
        ]
    )

    resolved_slate, resolution_df, meta = build_rotowire_dk_slate(
        rotowire_df=rotowire_df,
        registry_df=registry,
        slate_date="2026-02-22",
        slate_key="main",
    )

    assert bool(meta["fully_resolved"]) is True
    assert int(meta["resolved_players"]) == 1
    assert resolved_slate.iloc[0]["ID"] == "2002"
    assert resolved_slate.iloc[0]["Name + ID"] == "Jane Smith (2002)"
    assert resolution_df.iloc[0]["dk_resolution_status"] == "resolved"


def test_build_rotowire_dk_slate_flags_ambiguous_match_as_conflict() -> None:
    history = pd.DataFrame(
        [
            {
                "dk_id": "3001",
                "name_plus_id": "Alex Carter (3001)",
                "player_name": "Alex Carter",
                "player_name_norm": "alexcarter",
                "team_abbr": "TEAMX",
                "team_norm": "teamx",
                "position": "G",
                "position_base": "G",
                "roster_position": "G/UTIL",
                "salary": 5400,
                "opp_abbr": "TEAMY",
                "game_key": "TEAMX@TEAMY",
                "slate_date": pd.Timestamp("2026-02-20"),
                "slate_key": "main",
                "source_name": "a",
            },
            {
                "dk_id": "3002",
                "name_plus_id": "Alex Carter (3002)",
                "player_name": "Alex Carter",
                "player_name_norm": "alexcarter",
                "team_abbr": "TEAMX",
                "team_norm": "teamx",
                "position": "G",
                "position_base": "G",
                "roster_position": "G/UTIL",
                "salary": 5400,
                "opp_abbr": "TEAMY",
                "game_key": "TEAMX@TEAMY",
                "slate_date": pd.Timestamp("2026-02-21"),
                "slate_key": "main",
                "source_name": "b",
            },
        ]
    )
    registry = build_dk_identity_registry(history)
    rotowire_df = pd.DataFrame(
        [
            {
                "player_name": "Alex Carter",
                "team_abbr": "TEAMX",
                "opp_abbr": "TEAMY",
                "salary": 5400,
                "roto_position": "G",
                "site_positions": "G",
                "avg_fpts_season": 19.5,
                "is_home": False,
            }
        ]
    )

    resolved_slate, resolution_df, meta = build_rotowire_dk_slate(
        rotowire_df=rotowire_df,
        registry_df=registry,
        slate_date="2026-02-22",
        slate_key="main",
    )

    assert resolved_slate.iloc[0]["ID"] == ""
    assert resolution_df.iloc[0]["dk_resolution_status"] == "conflict"
    assert bool(meta["fully_resolved"]) is False
    assert int(meta["conflict_players"]) == 1
