from __future__ import annotations

from pathlib import Path

import pandas as pd

from college_basketball_dfs.cbb_api_service import (
    load_manual_overrides,
    load_registry,
    merge_manual_overrides,
    write_manual_overrides,
)


def test_merge_manual_overrides_dedupes_by_player_team_slate_key() -> None:
    existing = pd.DataFrame(
        [
            {
                "player_name": "Jane Smith",
                "team_abbr": "away",
                "salary": 6800,
                "position": "f",
                "roster_position": "f/util",
                "dk_id": "2002",
                "name_plus_id": "Jane Smith (2002)",
                "slate_date": "2026-03-04",
                "slate_key": "main",
                "reason": "initial",
                "source_name": "test",
            }
        ]
    )
    incoming = pd.DataFrame(
        [
            {
                "player_name": "Jane Smith",
                "team_abbr": "AWAY",
                "salary": 6900,
                "position": "F",
                "roster_position": "F/UTIL",
                "dk_id": "2002",
                "name_plus_id": "Jane Smith (2002)",
                "slate_date": "2026-03-04",
                "slate_key": "MAIN",
                "reason": "override",
                "source_name": "test2",
            }
        ]
    )

    merged = merge_manual_overrides(existing, incoming)
    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["team_abbr"] == "AWAY"
    assert row["slate_key"] == "main"
    assert float(row["salary"]) == 6900.0
    assert row["reason"] == "override"


def test_load_registry_merges_local_slate_history_and_manual_overrides(tmp_path: Path) -> None:
    dk_dir = tmp_path / "Draftkings"
    dk_dir.mkdir(parents=True, exist_ok=True)
    dk_csv = dk_dir / "DKSalaries_3_4_2026.csv"
    dk_csv.write_text(
        "\n".join(
            [
                "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame",
                "G,John Doe (1001),John Doe,1001,G/UTIL,7200,AWAY@HOME 03/04/2026 08:00PM ET,AWAY,30.2",
            ]
        ),
        encoding="utf-8",
    )

    manual_path = tmp_path / "dk_manual_overrides.csv"
    pd.DataFrame(
        [
            {
                "player_name": "Jane Smith",
                "team_abbr": "HOME",
                "opp_abbr": "AWAY",
                "salary": 6800,
                "position": "F",
                "roster_position": "F/UTIL",
                "game_key": "AWAY@HOME",
                "dk_id": "2002",
                "name_plus_id": "Jane Smith (2002)",
                "slate_date": "2026-03-04",
                "slate_key": "main",
                "reason": "manual",
                "source_name": "test",
            }
        ]
    ).to_csv(manual_path, index=False)

    registry = load_registry(draftkings_dir=dk_dir, manual_overrides_path=manual_path)
    assert not registry.empty
    assert set(registry["dk_id"].astype(str).tolist()) == {"1001", "2002"}


def test_write_and_load_manual_overrides_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "manual.csv"
    frame = pd.DataFrame(
        [
            {
                "player_name": "Chris Lane",
                "team_abbr": "abc",
                "salary": 6100,
                "position": "g",
                "roster_position": "g/util",
                "dk_id": "4111",
                "name_plus_id": "Chris Lane (4111)",
                "slate_date": "2026-03-05",
                "slate_key": "main",
                "reason": "manual",
                "source_name": "test",
            }
        ]
    )
    write_manual_overrides(frame, path=target)
    loaded = load_manual_overrides(target)
    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["player_name"] == "Chris Lane"
    assert row["team_abbr"] == "ABC"
    assert row["slate_key"] == "main"
