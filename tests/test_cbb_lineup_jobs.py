from __future__ import annotations

import pandas as pd
import pytest

from college_basketball_dfs import cbb_lineup_jobs
from college_basketball_dfs.cbb_dk_optimizer import locked_projection_recency_settings
from college_basketball_dfs.cbb_lineup_jobs import (
    _annotate_lineups_with_version_metadata,
    _contest_is_gpp,
    _lineup_model_config,
    _runtime_controls,
    run_lineup_job_request,
)


def test_lineup_model_config_defaults_to_salary_efficiency() -> None:
    cfg = _lineup_model_config("does_not_exist")
    assert cfg["version_key"] == "salary_efficiency_ceiling_v1"
    assert cfg["lineup_strategy"] == "standard"
    assert cfg["include_tail_signals"] is True


def test_runtime_controls_uses_gpp_overrides_and_request_override() -> None:
    cfg = _lineup_model_config("salary_efficiency_ceiling_v1")
    controls = _runtime_controls(
        contest_type="Large GPP",
        model_cfg=cfg,
        request={"salary_left_target": 90, "low_own_bucket_exposure_pct": 35.0},
    )
    assert controls["salary_left_target"] == 90
    assert controls["low_own_bucket_exposure_pct"] == 35.0
    assert controls["preferred_game_stack_min_players"] == 3


def test_contest_is_gpp_parser() -> None:
    assert _contest_is_gpp("Large GPP") is True
    assert _contest_is_gpp("Small gpp") is True
    assert _contest_is_gpp("Cash") is False


def test_annotate_lineups_with_model_metadata() -> None:
    lineups = [
        {
            "lineup_number": 1,
            "players": [],
            "salary": 50000,
            "projected_points": 200.0,
            "projected_ownership_sum": 120.0,
        }
    ]
    cfg = _lineup_model_config("spike_v2_tail")
    annotated = _annotate_lineups_with_version_metadata(lineups, cfg)
    assert annotated[0]["lineup_model_key"] == "spike_v2_tail"
    assert annotated[0]["lineup_model_label"].startswith("Spike v2")
    assert annotated[0]["model_profile"] == "tail_spike_pairs"


def test_run_lineup_job_request_blocks_when_no_active_slate_source_exists(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_resolve_active_slate_context(**_: object) -> dict[str, object]:
        seen.update(_)
        return {
            "slate": {
                "slate_id": 3400,
                "slate_date": "2026-03-05",
                "contest_type": "Classic",
                "slate_name": "All",
            },
            "coverage": {
                "players_total": 1,
                "resolved_players": 0,
                "unresolved_players": 1,
                "conflict_players": 0,
                "coverage_pct": 0.0,
                "fully_resolved": False,
            },
            "coverage_error": "DK registry resolution incomplete.",
            "rotowire_df": pd.DataFrame([{"player_name": "Jane Smith"}]),
            "resolved_slate_df": pd.DataFrame(),
            "active_slate_df": pd.DataFrame(),
            "active_source": "unavailable",
            "active_source_detail": "No active slate source is available.",
            "active_ready": False,
            "resolution_df": pd.DataFrame(
                [
                    {
                        "player_name": "Jane Smith",
                        "team_abbr": "AWAY",
                        "dk_resolution_status": "unresolved",
                        "dk_match_reason": "no_registry_name_match",
                    }
                ]
            ),
        }

    monkeypatch.setattr(cbb_lineup_jobs, "resolve_active_slate_context", _fake_resolve_active_slate_context)

    with pytest.raises(RuntimeError, match="No active slate source is available"):
        run_lineup_job_request(
            request={
                "selected_date": "2026-03-05",
                "bucket_name": "test-bucket",
                "gcp_project": "test-project",
                "service_account_json": "{\"type\":\"service_account\"}",
                "service_account_json_b64": "ZXhhbXBsZQ==",
            },
            progress=lambda pct, message: None,
            write_artifact=lambda name, content, content_type: None,
        )
    assert seen["bucket_name"] == "test-bucket"
    assert seen["gcp_project"] == "test-project"
    assert seen["service_account_json"] == "{\"type\":\"service_account\"}"
    assert seen["service_account_json_b64"] == "ZXhhbXBsZQ=="


def test_run_lineup_job_request_loads_saved_lineupstarter_priors(monkeypatch) -> None:
    projection_recency = locked_projection_recency_settings()

    def _fake_resolve_active_slate_context(**_: object) -> dict[str, object]:
        return {
            "slate": {
                "slate_id": 3401,
                "slate_date": "2026-03-05",
                "contest_type": "Classic",
                "slate_name": "All",
            },
            "coverage": {
                "players_total": 1,
                "resolved_players": 0,
                "unresolved_players": 1,
                "conflict_players": 0,
                "coverage_pct": 0.0,
                "fully_resolved": False,
            },
            "rotowire_df": pd.DataFrame([{"player_name": "Jane Smith", "team_abbr": "AWAY"}]),
            "resolved_slate_df": pd.DataFrame(),
            "active_slate_df": pd.DataFrame(
                [
                    {
                        "Position": "G",
                        "Name + ID": "Jane Smith (2002)",
                        "Name": "Jane Smith",
                        "ID": "2002",
                        "Roster Position": "G/UTIL",
                        "Salary": 6800,
                        "Game Info": "AWAY@HOME 03/05/2026 08:00PM ET",
                        "TeamAbbrev": "AWAY",
                        "AvgPointsPerGame": 24.2,
                    }
                ]
            ),
            "resolution_df": pd.DataFrame(
                [
                    {
                        "player_name": "Jane Smith",
                        "team_abbr": "AWAY",
                        "dk_resolution_status": "resolved",
                        "dk_match_reason": "manual_override",
                    }
                ]
            ),
            "active_source": "legacy_dk_fallback",
            "active_source_detail": "Using cached legacy DraftKings slate because RotoWire DK resolution is incomplete.",
            "active_ready": True,
        }

    seen: dict[str, object] = {}

    def _fake_build_player_pool(**kwargs: object) -> pd.DataFrame:
        lineupstarter_df = kwargs.get("lineupstarter_df")
        assert isinstance(lineupstarter_df, pd.DataFrame)
        seen["lineupstarter_rows"] = int(len(lineupstarter_df))
        slate_df = kwargs.get("slate_df")
        assert isinstance(slate_df, pd.DataFrame)
        seen["active_slate_rows"] = int(len(slate_df))
        season_stats_df = kwargs.get("season_stats_df")
        props_df = kwargs.get("props_df")
        odds_games_df = kwargs.get("odds_games_df")
        assert isinstance(season_stats_df, pd.DataFrame)
        assert isinstance(props_df, pd.DataFrame)
        assert isinstance(odds_games_df, pd.DataFrame)
        rotowire_df = kwargs.get("rotowire_df")
        assert isinstance(rotowire_df, pd.DataFrame)
        seen["rotowire_rows"] = int(len(rotowire_df))
        seen["season_history_rows"] = int(len(season_stats_df))
        seen["props_rows"] = int(len(props_df))
        seen["odds_tail_rows"] = int(len(odds_games_df))
        seen["bookmaker_filter"] = str(kwargs.get("bookmaker_filter") or "")
        seen["recent_form_games"] = int(kwargs.get("recent_form_games") or 0)
        seen["recent_points_weight"] = float(kwargs.get("recent_points_weight") or 0.0)
        return pd.DataFrame(
            [
                {
                    "ID": "2002",
                    "Name": "Jane Smith",
                    "TeamAbbrev": "AWAY",
                    "Position": "G",
                    "Salary": 6800,
                    "projected_dk_points": 32.5,
                    "projected_ownership": 21.0,
                    "ceiling_projection": 38.0,
                }
            ]
        )

    def _fake_generate_lineups(**_: object):
        return (
            [
                {
                    "lineup_number": 1,
                    "players": [{"ID": "2002", "Name": "Jane Smith"}],
                    "player_ids": ["2002"],
                    "salary": 6800,
                    "projected_points": 32.5,
                    "projected_ownership_sum": 21.0,
                }
            ],
            [],
        )

    monkeypatch.setattr(cbb_lineup_jobs, "resolve_active_slate_context", _fake_resolve_active_slate_context)
    monkeypatch.setattr(
        cbb_lineup_jobs,
        "load_lineupstarter_projection_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "ID": "2002",
                    "lineupstarter_projected_points": 34.0,
                    "lineupstarter_projected_ownership": 26.0,
                }
            ]
        ),
    )
    monkeypatch.setattr(cbb_lineup_jobs, "load_injuries_frame", lambda **_: pd.DataFrame())
    monkeypatch.setattr(
        cbb_lineup_jobs,
        "load_season_player_history_frame",
        lambda **_: pd.DataFrame([{"player_name": "Jane Smith", "dk_fpts": 30.0}]),
    )
    monkeypatch.setattr(
        cbb_lineup_jobs,
        "load_props_frame_for_date",
        lambda **_: pd.DataFrame([{"player_name": "Jane Smith", "market": "player_points", "line": 17.5}]),
    )
    monkeypatch.setattr(
        cbb_lineup_jobs,
        "load_odds_frame_for_date",
        lambda **_: pd.DataFrame([{"event_id": "evt-1", "total_points": 145.0, "spread_home": -3.0}]),
    )
    monkeypatch.setattr(
        cbb_lineup_jobs,
        "load_season_vegas_history_frame",
        lambda **_: pd.DataFrame(
            [
                {
                    "total_error": 0.0,
                    "total_points": 145.0,
                    "vegas_home_margin": -3.0,
                    "has_total_line": True,
                }
            ]
        ),
    )
    monkeypatch.setattr(cbb_lineup_jobs, "fit_total_tail_model", lambda _: {"samples": 1})
    monkeypatch.setattr(
        cbb_lineup_jobs,
        "score_odds_games_for_tail",
        lambda odds_df, _: odds_df.assign(
            game_key="AWAY@HOME",
            tail_residual_mu=0.0,
            tail_sigma=8.0,
            p_plus_8=0.25,
            p_plus_12=0.12,
            volatility_score=8.0,
        ),
    )
    monkeypatch.setattr(cbb_lineup_jobs, "build_player_pool", _fake_build_player_pool)
    monkeypatch.setattr(cbb_lineup_jobs, "generate_lineups", _fake_generate_lineups)
    monkeypatch.setattr(cbb_lineup_jobs, "lineups_summary_frame", lambda _: pd.DataFrame([{"lineup_number": 1}]))
    monkeypatch.setattr(cbb_lineup_jobs, "lineups_slots_frame", lambda _: pd.DataFrame([{"G": "Jane Smith"}]))
    monkeypatch.setattr(cbb_lineup_jobs, "build_dk_upload_csv", lambda _: "G,G,G,F,F,F,UTIL,UTIL\n")

    result = run_lineup_job_request(
        request={
            "selected_date": "2026-03-05",
            "recent_form_games": 12,
            "recent_points_weight": 0.9,
        },
        progress=lambda pct, message: None,
        write_artifact=lambda name, content, content_type: None,
    )

    assert seen["lineupstarter_rows"] == 1
    assert seen["active_slate_rows"] == 1
    assert seen["rotowire_rows"] == 1
    assert seen["season_history_rows"] == 1
    assert seen["props_rows"] == 1
    assert seen["odds_tail_rows"] == 1
    assert seen["bookmaker_filter"] == "fanduel"
    assert seen["recent_form_games"] == int(projection_recency["recent_form_games"])
    assert seen["recent_points_weight"] == float(projection_recency["recent_points_weight"])
    assert bool(result["lineupstarter_loaded"]) is True
    assert result["active_source"] == "legacy_dk_fallback"
    assert int(result["season_history_rows"]) == 1
    assert int(result["props_rows"]) == 1
    assert int(result["odds_tail_rows"]) == 1
    assert result["projection_recency"] == projection_recency
