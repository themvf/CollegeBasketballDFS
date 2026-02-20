# AI Review Agent (Daily DFS Diagnostics)

## Purpose
Generate a daily, evidence-backed recommendation packet from existing Tournament Review outputs:
- projection quality
- ownership quality
- lineup quality
- highest-impact focus candidates

## Where It Lives
- Packet builder: `src/college_basketball_dfs/cbb_ai_review.py`
- UI export controls: `dashboard/admin_app.py` under `Tournament Review -> AI Review Packet (GPT)`

## Packet Schema (v1)
Top-level keys:
- `schema_version`
- `generated_at_utc`
- `review_context`
- `scorecards`
- `focus_tables`
- `adjustment_factors`
- `phantom_summary`
- `notes_for_agent`

`scorecards` contains:
- `projection_quality`: matched rows, MAE/RMSE, rank correlation
- `ownership_quality`: matched rows, MAE, rank correlation
- `lineup_quality`: scored lineup quality metrics
- `field_quality`: field construction context metrics

`focus_tables` contains:
- `attention_index_top`: weighted list of players to investigate first
- `projection_miss_top`: largest projection misses
- `ownership_miss_top`: largest ownership misses

## Prompt Template
The app generates:
- `AI_REVIEW_SYSTEM_PROMPT` (analyst behavior and guardrails)
- user prompt with strict output format and embedded JSON packet
- global user prompt for rolling multi-slate review (`build_global_ai_review_user_prompt`)

The required output sections in prompt are:
1. Executive Summary
2. Projection Model Recommendations
3. Ownership Model Recommendations
4. Lineup Construction Recommendations
5. High-Leverage Players/Archetypes to Investigate
6. Next-Slate Experiment Plan

## Streamlit Cloud Setup
Optional key for downstream model calls:
- `openai_api_key` (or env `OPENAI_API_KEY`)

Current implementation supports both:
- exporting packet/prompt artifacts
- running an in-app OpenAI review call from Tournament Review when key is present

## Rolling Global Review
`Tournament Review` also includes `Rolling Global Review (Multi-Slate)`:
- scans a lookback window (or saved run dates)
- builds one daily packet per date when projection snapshot + actual results are available
- aggregates into a global packet with:
  - weighted scorecards
  - date trend table
  - recurring high-attention players
- generates a global recommendation prompt and optional in-app OpenAI output
