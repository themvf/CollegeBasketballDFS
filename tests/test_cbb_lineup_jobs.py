from __future__ import annotations

from college_basketball_dfs.cbb_lineup_jobs import (
    _annotate_lineups_with_version_metadata,
    _contest_is_gpp,
    _lineup_model_config,
    _runtime_controls,
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
