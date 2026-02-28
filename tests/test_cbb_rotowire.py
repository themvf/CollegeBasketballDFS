import pytest

from college_basketball_dfs.cbb_rotowire import flatten_slates, normalize_players, parse_site_id, select_slate


def test_parse_site_id_defaults_to_draftkings() -> None:
    assert parse_site_id(site=None, site_id=None) == 1
    assert parse_site_id(site="draftkings", site_id=None) == 1
    assert parse_site_id(site="1", site_id=None) == 1


def test_flatten_slates_and_select_slate() -> None:
    catalog = {
        "slates": [
            {
                "slateID": 3369,
                "contestType": "Classic",
                "slateName": "All",
                "salaryCap": 50000,
                "startDate": "2026-02-28 12:00:00",
                "endDate": "2026-02-28 14:00:00",
                "defaultSlate": False,
                "startDateOnly": "2026-02-28",
                "timeOnly": "12:00 PM",
                "games": [5114090, 5114091],
            },
            {
                "slateID": 3371,
                "contestType": "Classic",
                "slateName": "Night",
                "salaryCap": 50000,
                "startDate": "2026-02-28 18:00:00",
                "endDate": "2026-02-28 22:30:00",
                "defaultSlate": False,
                "startDateOnly": "2026-02-28",
                "timeOnly": "06:00 PM",
                "games": [5114095],
            },
        ]
    }
    slates_df = flatten_slates(catalog, site_id=1)

    selected = select_slate(slates_df, slate_date="2026-02-28", contest_type="Classic", slate_name="Night")

    assert selected["slate_id"] == 3371
    assert selected["game_count"] == 1


def test_select_slate_rejects_ambiguous_match() -> None:
    catalog = {
        "slates": [
            {
                "slateID": 3369,
                "contestType": "Classic",
                "slateName": "All",
                "salaryCap": 50000,
                "startDate": "2026-02-28 12:00:00",
                "endDate": "2026-02-28 14:00:00",
                "defaultSlate": False,
                "startDateOnly": "2026-02-28",
                "timeOnly": "12:00 PM",
                "games": [5114090, 5114091],
            },
            {
                "slateID": 3371,
                "contestType": "Classic",
                "slateName": "Night",
                "salaryCap": 50000,
                "startDate": "2026-02-28 18:00:00",
                "endDate": "2026-02-28 22:30:00",
                "defaultSlate": False,
                "startDateOnly": "2026-02-28",
                "timeOnly": "06:00 PM",
                "games": [5114095],
            },
        ]
    }
    slates_df = flatten_slates(catalog, site_id=1)

    with pytest.raises(ValueError, match="Multiple slates matched"):
        select_slate(slates_df, slate_date="2026-02-28", contest_type="Classic")


def test_normalize_players_maps_projection_fields() -> None:
    raw_players = [
        {
            "slateID": 3369,
            "rwID": 23697,
            "firstName": "Tyler",
            "lastName": "Tanner",
            "rotoPos": "G",
            "pos": ["G", "UTIL"],
            "injuryStatus": "NO",
            "isHome": False,
            "team": {"abbr": "VANDY", "city": "Vanderbilt", "nickname": "Commodores"},
            "game": {"dateTime": "2026-02-28 14:00:00"},
            "salary": 9300,
            "pts": "42.09",
            "minutes": 36,
            "opponent": {"team": "KTY"},
            "odds": {
                "moneyline": "100",
                "overUnder": "157.5",
                "spread": "1.0",
                "impliedPts": 78.25,
                "impliedWinProb": 48.2,
            },
            "stats": {
                "season": {"games": 28, "minutes": "32.3"},
                "avgFpts": {
                    "last3": "33.2",
                    "last5": "33.0",
                    "last7": "38.0",
                    "last14": "36.8",
                    "season": "36.2",
                },
                "advanced": {"usage": "28.75"},
            },
            "link": "/cbasketball/player/tyler-tanner-23697",
            "teamLink": "/cbasketball/team/vanderbilt-1674",
        }
    ]
    slate_row = {
        "site_id": 1,
        "slate_id": 3369,
        "slate_date": "2026-02-28",
        "contest_type": "Classic",
        "slate_name": "All",
    }

    out = normalize_players(raw_players, slate_row=slate_row)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["player_name"] == "Tyler Tanner"
    assert row["team_abbr"] == "VANDY"
    assert row["opp_abbr"] == "KTY"
    assert row["proj_fantasy_points"] == pytest.approx(42.09)
    assert row["proj_minutes"] == pytest.approx(36.0)
    assert row["proj_value_per_1k"] == pytest.approx(42.09 / 9300 * 1000.0)
    assert row["avg_fpts_last5"] == pytest.approx(33.0)
    assert row["usage_rate"] == pytest.approx(28.75)
