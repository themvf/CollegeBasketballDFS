import pandas as pd
import requests

from college_basketball_dfs.cbb_ai_review import (
    AI_REVIEW_SCHEMA_VERSION,
    GAME_SLATE_AI_REVIEW_SCHEMA_VERSION,
    build_ai_review_user_prompt,
    build_daily_ai_review_packet,
    build_game_slate_ai_review_packet,
    build_game_slate_ai_review_user_prompt,
    build_global_ai_review_packet,
    build_global_ai_review_user_prompt,
    request_openai_review,
)


def _sample_projection_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "Alpha Guard",
                "TeamAbbrev": "AAA",
                "Position": "G",
                "Salary": 8200,
                "blended_projection": 34.0,
                "actual_dk_points": 45.0,
                "blend_error": 11.0,
                "our_error": 9.0,
                "vegas_error": 13.0,
            },
            {
                "ID": "2",
                "Name": "Beta Forward",
                "TeamAbbrev": "BBB",
                "Position": "F",
                "Salary": 7600,
                "blended_projection": 30.0,
                "actual_dk_points": 24.0,
                "blend_error": -6.0,
                "our_error": -7.0,
                "vegas_error": -5.0,
            },
        ]
    )


def _sample_entries() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Rank": 1, "salary_left": 200, "max_team_stack": 3, "max_game_stack": 3},
            {"Rank": 2, "salary_left": 600, "max_team_stack": 2, "max_game_stack": 2},
        ]
    )


def _sample_exposure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": "Alpha Guard",
                "TeamAbbrev": "AAA",
                "field_ownership_pct": 35.0,
                "projected_ownership": 28.0,
                "actual_ownership_from_file": 40.0,
                "ownership_diff_vs_proj": 12.0,
                "final_dk_points": 45.0,
            },
            {
                "Name": "Beta Forward",
                "TeamAbbrev": "BBB",
                "field_ownership_pct": 22.0,
                "projected_ownership": 24.0,
                "actual_ownership_from_file": 18.0,
                "ownership_diff_vs_proj": -6.0,
                "final_dk_points": 24.0,
            },
        ]
    )


def _sample_phantom() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"actual_minus_projected": 8.0, "actual_points": 328.2, "would_beat_pct": 94.0, "salary_left": 300},
            {"actual_minus_projected": -4.0, "actual_points": 309.5, "would_beat_pct": 61.0, "salary_left": 700},
        ]
    )


def _sample_odds_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_date": "2026-02-20",
                "event_id": "ev1",
                "home_team": "Alpha U",
                "away_team": "Beta U",
                "bookmakers_count": 8,
                "moneyline_home": -180,
                "moneyline_away": 150,
                "spread_home": -4.5,
                "spread_away": 4.5,
                "total_points": 149.5,
                "p_plus_8": 0.31,
                "p_plus_12": 0.19,
            },
            {
                "game_date": "2026-02-20",
                "event_id": "ev2",
                "home_team": "Gamma U",
                "away_team": "Delta U",
                "bookmakers_count": 7,
                "moneyline_home": -110,
                "moneyline_away": -110,
                "spread_home": -1.0,
                "spread_away": 1.0,
                "total_points": 158.0,
                "p_plus_8": 0.42,
                "p_plus_12": 0.28,
            },
        ]
    )


def _sample_prior_boxscore() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_name": "Alpha U", "actual_minutes": 34, "actual_dk_points": 33.0},
            {"team_name": "Alpha U", "actual_minutes": 31, "actual_dk_points": 27.5},
            {"team_name": "Alpha U", "actual_minutes": 28, "actual_dk_points": 19.0},
            {"team_name": "Gamma U", "actual_minutes": 35, "actual_dk_points": 30.0},
            {"team_name": "Gamma U", "actual_minutes": 30, "actual_dk_points": 22.0},
            {"team_name": "Gamma U", "actual_minutes": 26, "actual_dk_points": 16.0},
        ]
    )


def _sample_vegas_history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "has_total_line": True,
                "has_spread_line": True,
                "total_points": 152.5,
                "vegas_home_margin": -3.5,
                "total_abs_error": 8.0,
                "spread_abs_error": 4.5,
                "actual_winner_side": "home",
                "predicted_winner_side": "home",
                "winner_pick_correct": True,
                "total_result": "Over",
            },
            {
                "has_total_line": True,
                "has_spread_line": True,
                "total_points": 145.0,
                "vegas_home_margin": 2.0,
                "total_abs_error": 6.0,
                "spread_abs_error": 2.0,
                "actual_winner_side": "away",
                "predicted_winner_side": "home",
                "winner_pick_correct": False,
                "total_result": "Under",
            },
        ]
    )


def _sample_player_pool_for_gpp() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Name": "Alpha Lead Guard",
                "TeamAbbrev": "ALP",
                "Position": "G",
                "Salary": 8600,
                "projected_dk_points": 38.5,
                "projected_ownership": 18.0,
                "leverage_score": 31.2,
                "value_per_1k": 4.48,
                "game_key": "ALP@BET",
                "game_tail_score": 74.0,
                "game_total_line": 151.5,
                "game_spread_line": -3.5,
            },
            {
                "Name": "Alpha Wing",
                "TeamAbbrev": "ALP",
                "Position": "F",
                "Salary": 7200,
                "projected_dk_points": 31.0,
                "projected_ownership": 14.0,
                "leverage_score": 27.4,
                "value_per_1k": 4.31,
                "game_key": "ALP@BET",
                "game_tail_score": 74.0,
                "game_total_line": 151.5,
                "game_spread_line": -3.5,
            },
            {
                "Name": "Beta Scorer",
                "TeamAbbrev": "BET",
                "Position": "G",
                "Salary": 7800,
                "projected_dk_points": 34.0,
                "projected_ownership": 12.0,
                "leverage_score": 30.1,
                "value_per_1k": 4.36,
                "game_key": "ALP@BET",
                "game_tail_score": 74.0,
                "game_total_line": 151.5,
                "game_spread_line": 3.5,
            },
            {
                "Name": "Gamma Center",
                "TeamAbbrev": "GAM",
                "Position": "F",
                "Salary": 8400,
                "projected_dk_points": 36.0,
                "projected_ownership": 20.0,
                "leverage_score": 24.0,
                "value_per_1k": 4.29,
                "game_key": "GAM@DEL",
                "game_tail_score": 62.0,
                "game_total_line": 146.0,
                "game_spread_line": -5.0,
            },
            {
                "Name": "Delta Guard",
                "TeamAbbrev": "DEL",
                "Position": "G",
                "Salary": 6900,
                "projected_dk_points": 30.0,
                "projected_ownership": 9.0,
                "leverage_score": 33.0,
                "value_per_1k": 4.35,
                "game_key": "GAM@DEL",
                "game_tail_score": 62.0,
                "game_total_line": 146.0,
                "game_spread_line": 5.0,
            },
        ]
    )


def test_build_daily_ai_review_packet_has_expected_schema() -> None:
    packet = build_daily_ai_review_packet(
        review_date="2026-02-18",
        contest_id="abc123",
        projection_comparison_df=_sample_projection_comparison(),
        entries_df=_sample_entries(),
        exposure_df=_sample_exposure(),
        phantom_summary_df=pd.DataFrame(),
        phantom_lineups_df=_sample_phantom(),
        adjustment_factors_df=pd.DataFrame([{"segment": "All", "our_points_multiplier": 1.05}]),
        focus_limit=10,
    )

    assert packet["schema_version"] == AI_REVIEW_SCHEMA_VERSION
    assert packet["review_context"]["review_date"] == "2026-02-18"
    assert packet["review_context"]["contest_id"] == "abc123"
    assert "projection_quality" in packet["scorecards"]
    assert "ownership_quality" in packet["scorecards"]
    assert "lineup_quality" in packet["scorecards"]
    assert "attention_index_top" in packet["focus_tables"]
    assert len(packet["focus_tables"]["attention_index_top"]) > 0


def test_build_daily_ai_review_packet_attention_falls_back_to_name_match() -> None:
    projection_df = pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "Alpha Guard Jr.",
                "TeamAbbrev": "AAA",
                "Position": "G",
                "Salary": 8200,
                "blended_projection": 34.0,
                "actual_dk_points": 45.0,
                "blend_error": 11.0,
                "our_error": 9.0,
                "vegas_error": 13.0,
            }
        ]
    )
    exposure_df = pd.DataFrame(
        [
            {
                "Name": "Alpha Guard",
                "TeamAbbrev": "ZZZ",
                "field_ownership_pct": 35.0,
                "projected_ownership": 28.0,
                "actual_ownership_from_file": 40.0,
                "ownership_diff_vs_proj": 12.0,
                "final_dk_points": 45.0,
            }
        ]
    )
    packet = build_daily_ai_review_packet(
        review_date="2026-02-18",
        contest_id="abc123",
        projection_comparison_df=projection_df,
        entries_df=_sample_entries(),
        exposure_df=exposure_df,
        phantom_summary_df=pd.DataFrame(),
        phantom_lineups_df=_sample_phantom(),
        adjustment_factors_df=pd.DataFrame(),
        focus_limit=10,
    )
    attention = packet["focus_tables"]["attention_index_top"]
    assert len(attention) == 1
    row = attention[0]
    assert round(float(row["projected_ownership"]), 2) == 28.0
    assert round(float(row["actual_ownership_from_file"]), 2) == 40.0


def test_build_daily_ai_review_packet_splits_true_low_own_vs_ownership_surprise() -> None:
    projection_df = pd.DataFrame(
        [
            {
                "ID": "1",
                "Name": "True Low Own",
                "TeamAbbrev": "AAA",
                "Position": "G",
                "Salary": 7000,
                "blended_projection": 24.0,
                "actual_dk_points": 40.0,
                "blend_error": 16.0,
                "our_error": 14.0,
                "vegas_error": 15.0,
            },
            {
                "ID": "2",
                "Name": "Ownership Surprise",
                "TeamAbbrev": "BBB",
                "Position": "F",
                "Salary": 6800,
                "blended_projection": 22.0,
                "actual_dk_points": 39.0,
                "blend_error": 17.0,
                "our_error": 15.0,
                "vegas_error": 16.0,
            },
        ]
    )
    exposure_df = pd.DataFrame(
        [
            {
                "Name": "True Low Own",
                "TeamAbbrev": "AAA",
                "field_ownership_pct": 7.0,
                "projected_ownership": 6.0,
                "actual_ownership_from_file": 8.0,
                "ownership_diff_vs_proj": 2.0,
            },
            {
                "Name": "Ownership Surprise",
                "TeamAbbrev": "BBB",
                "field_ownership_pct": 16.0,
                "projected_ownership": 6.5,
                "actual_ownership_from_file": 17.0,
                "ownership_diff_vs_proj": 10.5,
            },
        ]
    )
    packet = build_daily_ai_review_packet(
        review_date="2026-02-18",
        contest_id="abc123",
        projection_comparison_df=projection_df,
        entries_df=_sample_entries(),
        exposure_df=exposure_df,
        phantom_summary_df=pd.DataFrame(),
        phantom_lineups_df=_sample_phantom(),
        adjustment_factors_df=pd.DataFrame(),
        focus_limit=10,
    )
    low_own = packet["focus_tables"]["low_own_smash_top"]
    surprises = packet["focus_tables"]["ownership_surprise_smash_top"]
    assert len(low_own) == 1
    assert low_own[0]["Name"] == "True Low Own"
    assert bool(low_own[0]["true_low_own_smash_flag"]) is True
    assert len(surprises) == 1
    assert surprises[0]["Name"] == "Ownership Surprise"
    assert bool(surprises[0]["ownership_surprise_smash_flag"]) is True


def test_build_ai_review_user_prompt_contains_json_packet() -> None:
    packet = build_daily_ai_review_packet(
        review_date="2026-02-18",
        contest_id="abc123",
        projection_comparison_df=_sample_projection_comparison(),
        entries_df=_sample_entries(),
        exposure_df=_sample_exposure(),
        phantom_summary_df=pd.DataFrame(),
        phantom_lineups_df=_sample_phantom(),
        adjustment_factors_df=pd.DataFrame(),
        focus_limit=5,
    )
    prompt = build_ai_review_user_prompt(packet)
    assert "Required output format" in prompt
    assert '"schema_version": "v1"' in prompt
    assert '"review_date": "2026-02-18"' in prompt


def test_request_openai_review_uses_output_text(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"output_text": "Recommended change set"}

    def _fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("college_basketball_dfs.cbb_ai_review.requests.post", _fake_post)
    out = request_openai_review(api_key="test-key", user_prompt="prompt")
    assert out == "Recommended change set"


def test_request_openai_review_retries_model_not_found(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, status_code: int, text: str, body: dict[str, object]):
            self.status_code = status_code
            self.text = text
            self._body = body

        def json(self):
            return self._body

    calls: list[str] = []

    def _fake_post(*args, **kwargs):
        payload = kwargs.get("json") or {}
        model = str(payload.get("model") or "")
        calls.append(model)
        if model == "gpt-5.1-mini":
            return FakeResponse(
                400,
                '{"error":{"message":"The requested model \'gpt-5.1-mini\' does not exist.","code":"model_not_found"}}',
                {},
            )
        return FakeResponse(200, "{}", {"output_text": "Recovered with fallback model"})

    monkeypatch.setattr("college_basketball_dfs.cbb_ai_review.requests.post", _fake_post)
    out = request_openai_review(api_key="test-key", user_prompt="prompt", model="gpt-5.1-mini")
    assert out == "Recovered with fallback model"
    assert calls[0] == "gpt-5.1-mini"
    assert len(calls) >= 2


def test_request_openai_review_parses_nested_output_text_value(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": {"value": "Nested output text"},
                            }
                        ],
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("college_basketball_dfs.cbb_ai_review.requests.post", _fake_post)
    out = request_openai_review(api_key="test-key", user_prompt="prompt")
    assert out == "Nested output text"


def test_request_openai_review_parses_refusal_text(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "refusal",
                                "refusal": "Cannot provide that request.",
                            }
                        ],
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr("college_basketball_dfs.cbb_ai_review.requests.post", _fake_post)
    out = request_openai_review(api_key="test-key", user_prompt="prompt")
    assert out == "Cannot provide that request."


def test_request_openai_review_retries_request_timeout(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {"output_text": "Recovered after timeout retry"}

    calls = {"count": 0}

    def _fake_post(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise requests.exceptions.ReadTimeout("timed out")
        return FakeResponse()

    monkeypatch.setattr("college_basketball_dfs.cbb_ai_review.requests.post", _fake_post)
    monkeypatch.setattr("college_basketball_dfs.cbb_ai_review.time.sleep", lambda *_args, **_kwargs: None)
    out = request_openai_review(
        api_key="test-key",
        user_prompt="prompt",
        timeout_seconds=1,
        max_request_retries=1,
    )
    assert out == "Recovered after timeout retry"
    assert calls["count"] == 2


def test_build_global_ai_review_packet_aggregates_daily_packets() -> None:
    daily_a = build_daily_ai_review_packet(
        review_date="2026-02-18",
        contest_id="a",
        projection_comparison_df=_sample_projection_comparison(),
        entries_df=_sample_entries(),
        exposure_df=_sample_exposure(),
        phantom_summary_df=pd.DataFrame(),
        phantom_lineups_df=_sample_phantom(),
        adjustment_factors_df=pd.DataFrame(),
        focus_limit=5,
    )
    daily_b = build_daily_ai_review_packet(
        review_date="2026-02-19",
        contest_id="b",
        projection_comparison_df=_sample_projection_comparison(),
        entries_df=_sample_entries(),
        exposure_df=_sample_exposure(),
        phantom_summary_df=pd.DataFrame(),
        phantom_lineups_df=_sample_phantom(),
        adjustment_factors_df=pd.DataFrame(),
        focus_limit=5,
    )

    global_packet = build_global_ai_review_packet([daily_a, daily_b], focus_limit=10)
    assert global_packet["schema_version"] == "v1_global"
    assert global_packet["window_summary"]["slate_count"] == 2
    assert len(global_packet["trend_by_date"]) == 2
    assert len(global_packet["recurring_focus_players"]) > 0

    prompt = build_global_ai_review_user_prompt(global_packet)
    assert "Global Diagnostic Summary" in prompt
    assert '"schema_version": "v1_global"' in prompt


def test_build_game_slate_ai_review_packet_and_prompt() -> None:
    packet = build_game_slate_ai_review_packet(
        review_date="2026-02-20",
        odds_df=_sample_odds_games(),
        player_pool_df=_sample_player_pool_for_gpp(),
        prior_boxscore_df=_sample_prior_boxscore(),
        vegas_history_df=_sample_vegas_history(),
        vegas_review_df=pd.DataFrame(),
        focus_limit=5,
    )

    assert packet["schema_version"] == GAME_SLATE_AI_REVIEW_SCHEMA_VERSION
    assert packet["review_context"]["review_date"] == "2026-02-20"
    assert int(packet["market_summary"]["games"]) == 2
    focus_tables = packet.get("focus_tables") or {}
    assert len(focus_tables.get("stack_candidates_top") or []) > 0
    assert len(focus_tables.get("winner_calls") or []) > 0
    assert len(focus_tables.get("gpp_game_stack_targets") or []) > 0
    assert len(focus_tables.get("gpp_team_stack_targets") or []) > 0
    assert len(focus_tables.get("gpp_player_core_targets") or []) > 0
    assert len(focus_tables.get("games_table") or []) > 0

    prompt = build_game_slate_ai_review_user_prompt(packet)
    assert "GPP Game Stack Tiers" in prompt
    assert '"schema_version": "v1_game_slate"' in prompt
