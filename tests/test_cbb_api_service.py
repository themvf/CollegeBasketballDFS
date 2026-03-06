from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from college_basketball_dfs.cbb_api_service import (
    import_dk_slate_overrides,
    import_lineupstarter_projection_csv,
    load_manual_overrides,
    load_registry,
    merge_manual_overrides,
    normalize_lineupstarter_upload_frame,
    save_manual_resolution_overrides,
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


def test_import_dk_slate_overrides_repairs_full_coverage(monkeypatch, tmp_path: Path) -> None:
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
                "game_datetime": "2026-03-05T20:00:00",
            }
        ]
    )

    def _fake_load_rotowire_players_for_slate(**_: object) -> tuple[pd.DataFrame, dict[str, object]]:
        return rotowire_df.copy(), {
            "slate_id": 3400,
            "slate_date": "2026-03-05",
            "contest_type": "Classic",
            "slate_name": "All",
            "game_count": 1,
            "players": 1,
        }

    monkeypatch.setattr(
        "college_basketball_dfs.cbb_api_service.load_rotowire_players_for_slate",
        _fake_load_rotowire_players_for_slate,
    )

    manual_path = tmp_path / "manual.csv"
    draftkings_dir = tmp_path / "Draftkings"
    draftkings_dir.mkdir(parents=True, exist_ok=True)
    (draftkings_dir / "DKSalaries_3_4_2026.csv").write_text(
        "\n".join(
            [
                "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame",
                "G,John Doe (1001),John Doe,1001,G/UTIL,7200,AWAY@HOME 03/04/2026 08:00PM ET,AWAY,30.2",
            ]
        ),
        encoding="utf-8",
    )
    dk_csv = (
        "Position,Name + ID,Name,ID,Roster Position,Salary,Game Info,TeamAbbrev,AvgPointsPerGame\n"
        "F,Jane Smith (2002),Jane Smith,2002,F/UTIL,6800,AWAY@HOME 03/05/2026 08:00PM ET,AWAY,24.2\n"
    ).encode("utf-8")

    result = import_dk_slate_overrides(
        dk_slate_csv_bytes=dk_csv,
        selected_date="2026-03-05",
        slate_key="main",
        contest_type="Classic",
        slate_name="All",
        draftkings_dir=draftkings_dir,
        manual_overrides_path=manual_path,
        persist=True,
    )

    assert bool(result["coverage_before"]["fully_resolved"]) is False
    assert bool(result["coverage_after"]["fully_resolved"]) is True
    assert int(result["derived_override_count"]) == 1
    loaded = load_manual_overrides(manual_path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["dk_id"] == "2002"


def test_save_manual_resolution_overrides_normalizes_editor_rows(tmp_path: Path) -> None:
    manual_path = tmp_path / "manual.csv"
    result = save_manual_resolution_overrides(
        overrides_frame=pd.DataFrame(
            [
                {
                    "player_name": "Jane Smith",
                    "team_abbr": "away",
                    "opp_abbr": "home",
                    "salary": "6800",
                    "position": "f",
                    "roster_position": "",
                    "dk_id": "",
                    "name_plus_id": "Jane Smith (2002)",
                    "reason": "",
                }
            ]
        ),
        selected_date="2026-03-05",
        slate_key="main",
        manual_overrides_path=manual_path,
    )

    assert int(result["saved_override_count"]) == 1
    loaded = load_manual_overrides(manual_path)
    assert len(loaded) == 1
    row = loaded.iloc[0]
    assert row["team_abbr"] == "AWAY"
    assert row["dk_id"] == "2002"
    assert row["roster_position"] == "F/UTIL"
    assert row["reason"] == "manual_alias_review"
    assert row["source_name"] == "manual_alias_review:2026-03-05:main"


def test_normalize_lineupstarter_upload_frame_matches_resolved_slate_by_name_and_salary() -> None:
    resolved_slate = pd.DataFrame(
        [
            {
                "Position": "G",
                "Name + ID": "John Doe (1001)",
                "Name": "John Doe",
                "ID": "1001",
                "Roster Position": "G/UTIL",
                "Salary": 6200,
                "Game Info": "AAA@BBB 03/05/2026 07:00PM ET",
                "TeamAbbrev": "AAA",
                "AvgPointsPerGame": 28.4,
            }
        ]
    )
    upload = pd.DataFrame(
        [
            {
                "Player": "John Doe",
                "Pos": "G",
                "Team": "AAA",
                "Salary": "$6,200",
                "Proj": "34.5",
                "projOwn%": "18.4%",
            }
        ]
    )

    normalized, coverage = normalize_lineupstarter_upload_frame(upload, slate_df=resolved_slate)
    assert bool(coverage["fully_resolved"]) is True
    assert int(coverage["matched_rows"]) == 1
    row = normalized.iloc[0]
    assert str(row["ID"]) == "1001"
    assert row["Name"] == "John Doe"
    assert row["lineupstarter_match_status"] == "matched"
    assert round(float(row["lineupstarter_projected_points"]), 2) == 34.5
    assert round(float(row["lineupstarter_projected_ownership"]), 2) == 18.4


def test_import_lineupstarter_projection_csv_writes_normalized_rows(monkeypatch) -> None:
    class _FakeStore:
        bucket_name = "test-bucket"

        def __init__(self) -> None:
            self.saved_csv = ""

        def write_lineupstarter_csv(self, game_date, csv_text: str, slate_key: str | None = None) -> str:
            self.saved_csv = csv_text
            assert str(game_date) == "2026-03-05"
            assert slate_key == "main"
            return "cbb/lineupstarter/2026-03-05_lineupstarter.csv"

    fake_store = _FakeStore()
    monkeypatch.setattr(
        "college_basketball_dfs.cbb_api_service._build_api_store",
        lambda **_: fake_store,
    )
    resolved_slate = pd.DataFrame(
        [
            {
                "Position": "G",
                "Name + ID": "John Doe (1001)",
                "Name": "John Doe",
                "ID": "1001",
                "Roster Position": "G/UTIL",
                "Salary": 6200,
                "Game Info": "AAA@BBB 03/05/2026 07:00PM ET",
                "TeamAbbrev": "AAA",
                "AvgPointsPerGame": 28.4,
            }
        ]
    )

    result = import_lineupstarter_projection_csv(
        csv_bytes=b"Player,Pos,Team,Salary,Proj,projOwn%\nJohn Doe,G,AAA,6200,34.5,18.4%\n",
        resolved_slate_df=resolved_slate,
        selected_date="2026-03-05",
        slate_key="main",
        bucket_name="test-bucket",
    )

    assert result["blob_name"] == "cbb/lineupstarter/2026-03-05_lineupstarter.csv"
    assert int(result["rows_saved"]) == 1
    saved = pd.read_csv(io.StringIO(fake_store.saved_csv))
    assert str(saved.loc[0, "ID"]) == "1001"
    assert round(float(saved.loc[0, "lineupstarter_projected_points"]), 2) == 34.5
    assert round(float(saved.loc[0, "lineupstarter_projected_ownership"]), 2) == 18.4
