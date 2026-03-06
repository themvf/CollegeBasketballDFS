from __future__ import annotations

import pandas as pd
import pytest

from college_basketball_dfs import cbb_lineup_jobs
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


def test_run_lineup_job_request_blocks_on_incomplete_registry_coverage(monkeypatch) -> None:
    def _fake_resolve_rotowire_slate(**_: object) -> dict[str, object]:
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
            "rotowire_df": pd.DataFrame([{"player_name": "Jane Smith"}]),
            "resolved_slate_df": pd.DataFrame(),
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

    monkeypatch.setattr(cbb_lineup_jobs, "resolve_rotowire_slate", _fake_resolve_rotowire_slate)

    with pytest.raises(RuntimeError, match="DK registry resolution incomplete"):
        run_lineup_job_request(
            request={"selected_date": "2026-03-05"},
            progress=lambda pct, message: None,
            write_artifact=lambda name, content, content_type: None,
        )
