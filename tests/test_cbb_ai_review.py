import pandas as pd

from college_basketball_dfs.cbb_ai_review import (
    AI_REVIEW_SCHEMA_VERSION,
    build_ai_review_user_prompt,
    build_daily_ai_review_packet,
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
