import pandas as pd

from college_basketball_dfs.cbb_dk_optimizer import (
    build_dk_upload_csv,
    build_player_pool,
    generate_lineups,
    remove_injured_players,
)


def _sample_slate() -> pd.DataFrame:
    rows = []
    for i in range(1, 7):
        rows.append(
            {
                "Position": "G",
                "Name + ID": f"Guard {i} ({1000 + i})",
                "Name": f"Guard {i}",
                "ID": str(1000 + i),
                "Roster Position": "G/UTIL",
                "Salary": 6200 - (i * 120),
                "Game Info": "AAA@BBB 02/14/2026 04:00PM ET" if i <= 2 else "CCC@DDD 02/14/2026 05:00PM ET",
                "TeamAbbrev": "AAA" if i % 2 == 0 else "CCC",
                "AvgPointsPerGame": 25 + i,
            }
        )
    for i in range(1, 7):
        rows.append(
            {
                "Position": "F",
                "Name + ID": f"Forward {i} ({2000 + i})",
                "Name": f"Forward {i}",
                "ID": str(2000 + i),
                "Roster Position": "F/UTIL",
                "Salary": 6100 - (i * 140),
                "Game Info": "EEE@FFF 02/14/2026 06:00PM ET" if i <= 2 else "CCC@DDD 02/14/2026 05:00PM ET",
                "TeamAbbrev": "EEE" if i % 2 == 0 else "DDD",
                "AvgPointsPerGame": 24 + i,
            }
        )
    return pd.DataFrame(rows)


def _sample_props() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_name": "Guard 1", "bookmaker": "fanduel", "market": "player_points", "line": 18.5},
            {"player_name": "Guard 1", "bookmaker": "fanduel", "market": "player_assists", "line": 5.5},
            {"player_name": "Forward 1", "bookmaker": "fanduel", "market": "player_points", "line": 16.5},
            {"player_name": "Forward 1", "bookmaker": "fanduel", "market": "player_rebounds", "line": 9.5},
            {"player_name": "Forward 1", "bookmaker": "fanduel", "market": "player_threes_made", "line": 1.5},
        ]
    )


def test_remove_injured_players_filters_out_and_doubtful() -> None:
    slate = _sample_slate()
    injuries = pd.DataFrame(
        [
            {"player_name": "Guard 1", "team": "CCC", "status": "Out", "active": True},
            {"player_name": "Forward 2", "team": "EEE", "status": "Doubtful", "active": True},
            {"player_name": "Guard 2", "team": "AAA", "status": "Questionable", "active": True},
        ]
    )

    filtered, removed = remove_injured_players(slate, injuries)
    assert "Guard 1" in removed["Name"].tolist()
    assert "Forward 2" in removed["Name"].tolist()
    assert "Guard 2" not in removed["Name"].tolist()
    assert len(filtered) == len(slate) - 2


def test_build_player_pool_merges_props() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    g1 = pool.loc[pool["Name"] == "Guard 1"].iloc[0]
    f1 = pool.loc[pool["Name"] == "Forward 1"].iloc[0]
    assert g1["vegas_points_line"] == 18.5
    assert g1["vegas_assists_line"] == 5.5
    assert f1["vegas_threes_line"] == 1.5
    assert g1["projected_dk_points"] > 0
    assert f1["projected_ownership"] > 0


def test_generate_lineups_respects_locks_and_excludes() -> None:
    pool = build_player_pool(_sample_slate(), _sample_props(), bookmaker_filter="fanduel")
    locked = ["1001"]
    excluded = ["2006"]
    lineups, warnings = generate_lineups(
        pool_df=pool,
        num_lineups=5,
        contest_type="Small GPP",
        locked_ids=locked,
        excluded_ids=excluded,
        random_seed=13,
    )

    assert warnings == []
    assert len(lineups) == 5
    for lineup in lineups:
        ids = set(lineup["player_ids"])
        assert "1001" in ids
        assert "2006" not in ids
        assert lineup["salary"] <= 50000
        assert len(ids) == 8

    upload_csv = build_dk_upload_csv(lineups)
    assert upload_csv.startswith("G,G,G,F,F,F,UTIL,UTIL")
