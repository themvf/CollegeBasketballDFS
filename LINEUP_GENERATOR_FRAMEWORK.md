# Lineup Generator Framework

This document describes the current implementation of lineup generation and phantom review in this codebase.

Primary code references:
- `dashboard/admin_app.py`
- `src/college_basketball_dfs/cbb_dk_optimizer.py`
- `src/college_basketball_dfs/cbb_tournament_review.py`
- `src/college_basketball_dfs/cbb_gcs.py`

## 1. End-to-End Flow

1. Build player pool for a selected date + active slate label.
2. Apply injuries filter (`out`, `doubtful` by default).
3. Build projections and ownership features.
4. Configure lineup-generation controls in `Lineup Generator`.
5. Generate lineups in one model (`Single Version`) or multiple models (`All Versions`).
6. Save run bundle (manifest + per-version JSON/CSV/upload CSV).
7. In `Tournament Review`, run Phantom Review to score generated lineups vs actual results and optionally vs field standings.
8. Feed phantom summaries back into future runs:
- projection calibration (auto scale)
- version promotion weights (All Versions mode)
- consistency/postmortem diagnostics

## 2. Data Inputs and Dependencies

Lineup generation uses:
- DK slate CSV (`DK Slate` tab)
- injuries CSV (`Injuries` tab)
- odds and props feeds (`Slate + Vegas` data path)
- season player history

`build_optimizer_pool_for_date(...)` loads:
- DK slate
- injuries
- season history
- odds + tail model features
- props

Then calls `build_player_pool(...)` to produce:
- blended projection fields
- value/leverage features
- tail/game context features
- projected ownership

## 3. Roster and Feasibility Rules

Hard constraints in optimizer:
- salary cap: `50000`
- roster size: `8`
- minimum positions: `3` guards, `3` forwards
- lineup must include players from at least `2` games
- salary used must satisfy `max_salary_left`
- locks/excludes must be feasible and non-conflicting
- exposure caps enforced globally and optionally per player

## 4. Lineup Generator Settings (UI -> Engine)

### 4.1 Core Run Controls

| UI Control | Default | Saved Setting Key | Engine Mapping / Behavior |
|---|---:|---|---|
| `Lineup Slate Date` | current game date context | `slate_date` in run bundle | selects all read/write paths |
| `Active Slate Label` | shared app context | `slate_label`, `slate_key` | scopes lineup runs and later phantom reads |
| `Lineup Bookmaker Source` | `fanduel` fallback | `bookmaker` | passed to pool builder |
| `Recent Form Games` | `5` | `recent_form_games` | rolling game window for `our_minutes_recent` / `our_points_recent` |
| `Recent Points Weight %` | `35` | `recent_points_weight_pct` (`recent_points_weight`) | blends season points with recent points before DK scoring |
| `Lineups` | `20` | `lineup_count` | `num_lineups` |
| `Contest Type` | `Small GPP` | `contest_type` | objective model selection |
| `Random Seed` | `7` | `lineup_seed` | `random_seed` (`+version_idx` per version) |
| `Run Mode` | `Single Version` | `run_mode` | `single` or `all` |
| `Lineup Model` (single mode) | `Standard v1` | `selected_model_key` | maps to strategy/tail profile |

### 4.2 Model Profiles

| Version Key | Label | Strategy | Tail Signals | Profile |
|---|---|---|---:|---|
| `standard_v1` | Standard v1 | `standard` | false | `legacy_baseline` |
| `spike_v1_legacy` | Spike v1 (Legacy A/B) | `spike` | false | `legacy_spike_pairs` |
| `spike_v2_tail` | Spike v2 (Tail A/B) | `spike` | true | `tail_spike_pairs` |
| `cluster_v1_experimental` | Cluster v1 (Experimental) | `cluster` | false | `cluster_seed_mutation_v1` |

`All Versions` mode includes all four profiles.

### 4.3 Stack Bias and Salary Controls

| UI Control | Default | Saved Key | Engine Mapping |
|---|---:|---|---|
| `Apply Game Agent Stack Bias` | `true` | `apply_game_agent_stack_bias` | builds per-player objective bonuses from Game Agent packet |
| `Agent Bias Strength %` | `100` | `game_agent_bias_strength_pct` | multiplier to bias strength |
| `Agent Focus Games` | `3` | `game_agent_focus_games` | top-N game/team/player targets to apply |
| `Max Salary Left Per Lineup` | `500` | `max_salary_left` | `max_salary_left` hard constraint |
| `Global Max Player Exposure %` | `60` | `global_max_exposure_pct` | `global_max_exposure_pct` |
| `Strict Salary Utilization` | `false` | `strict_salary_utilization` | if enabled, max salary left tightened to `50` |
| `Salary Left Target` | `250` | `salary_left_target` | lineup scoring penalty target |

### 4.4 Projection Calibration and Risk Controls

| UI Control | Default | Saved Key | Engine Mapping |
|---|---:|---|---|
| `Auto Projection Calibration` | `true` | `auto_projection_calibration` | computes phantom-based projection scale |
| `Calibration Lookback Days` | `14` | `calibration_lookback_days` | lookback for phantom scale |
| `Salary-Bucket Residual Calibration` | `true` | `auto_salary_bucket_calibration` | computes per-salary scales |
| `Min Samples Per Salary Bucket` | `20` | `bucket_calibration_min_samples` | minimum bucket sample threshold |
| `Salary-Bucket Calibration Lookback Days` | `14` | `bucket_calibration_lookback_days` | lookback for salary bucket scales |
| `Role-Bucket Residual Calibration` | `true` | `auto_role_bucket_calibration` | computes role scales |
| `Min Samples Per Role Bucket` | `25` | `role_calibration_min_samples` | minimum role bucket samples |
| `Role-Bucket Calibration Lookback Days` | `14` | `role_calibration_lookback_days` | lookback for role scales |
| `Minutes/DNP Uncertainty Shrink` | `true` | `apply_uncertainty_shrink` | enables uncertainty shrink |
| `Uncertainty Shrink %` | `18` | `uncertainty_weight` | `uncertainty_weight=0.18` |
| `DNP Risk Threshold %` | `30` | `dnp_risk_threshold` | `dnp_risk_threshold=0.30` |
| `High-Risk Extra Shrink %` | `10` | `high_risk_extra_shrink` | additional shrink over threshold |

### 4.5 Ownership and Leverage Controls

| UI Control | Default | Saved Key | Engine Mapping |
|---|---:|---|---|
| `Ownership Surprise Guardrails` | `true` | `apply_ownership_guardrails` | enables guardrails |
| `Guardrail Projected Own <= %` | `10` | `ownership_guardrail_projected_threshold` | low-projected-own trigger |
| `Guardrail Surge Score >=` | `72` | `ownership_guardrail_surge_threshold` | surge trigger |
| `Guardrail Ownership Cap %` | `24` | `ownership_guardrail_floor_cap` | max implied floor |
| `Low-Own Bucket Exposure %` | `30` | `low_own_bucket_exposure_pct` | percent of lineups requiring low-own inclusion |
| `Low-Own Min Players` | `1` | `low_own_bucket_min_per_lineup` | minimum low-own count in required lineups |
| `Low-Own Max Projected Own %` | `10` | `low_own_bucket_max_projected_ownership` | low-own candidate cap |
| `Low-Own Min Projection` | `24` | `low_own_bucket_min_projection` | low-own candidate floor |

### 4.6 Ceiling and Phantom-Promotion Controls

| UI Control | Default | Saved Key | Engine Mapping |
|---|---:|---|---|
| `Ceiling Archetype Lineups %` | `25` | `ceiling_boost_lineup_pct` | subset of lineups uses ceiling boost behavior |
| `Ceiling Stack Bonus` | `2.2` | `ceiling_boost_stack_bonus` | extra stack-concentration bonus |
| `Ceiling Salary Left Target` | `120` | `ceiling_boost_salary_left_target` | alternate salary target for ceiling subset |
| `Promote Top Phantom Constructions` | `true` | `promote_phantom_constructions` | All Versions lineup count reallocation |
| `Phantom Promotion Lookback Days` | `14` | `phantom_promotion_lookback_days` | lookback for promotion weights |
| `Spike Max Shared Players (A vs B)` | `4` | `spike_max_pair_overlap` | pair overlap cap in spike mode |

### 4.7 Player-Level Filters

| UI Control | Saved Key | Engine Mapping |
|---|---|---|
| `Lock Players (in every lineup)` | `locked_ids` | `locked_ids` |
| `Exclude Players` | `excluded_ids` | `excluded_ids` |
| `Exposure Caps (max % by player)` | `exposure_caps_pct` | `exposure_caps_pct` |

### 4.8 Fixed/Implicit Engine Parameters (not direct UI controls)

These are currently hardcoded in the generation call path:
- `ownership_guardrail_projection_rank_threshold = 0.60`
- `ownership_guardrail_floor_base = ownership_guardrail_projected_threshold` (same value as slider)
- `low_own_bucket_min_tail_score = 55.0`
- `low_own_bucket_objective_bonus = 1.3`
- `preferred_game_keys = applied_game_keys from Game Agent bias meta`
- `preferred_game_bonus = 0.6`
- `cluster_target_count = 15`
- `cluster_variants_per_cluster = 10`
- `salary_left_penalty_divisor = 75.0` (engine default)
- `max_attempts_per_lineup = 1200` (engine default)

## 5. How Lineups Are Constructed

### 5.1 Pre-Scoring Pipeline

1. Recency inputs from pool builder (before scoring)
- computes `our_minutes_recent` and `our_points_recent` using `recent_form_games`
- computes `our_points_proj` using season/recent blend via `recent_points_weight`

2. `apply_projection_calibration(...)`
- applies global scale
- applies salary-bucket scales
- applies role-bucket scales
- recomputes `projection_per_dollar`, `value_per_1k`, `leverage_score`

3. `apply_ownership_surprise_guardrails(...)` (optional)
- flags low-projected ownership with strong surge + high projection rank
- raises projected ownership floor up to configured cap

4. `apply_projection_uncertainty_adjustment(...)` (optional)
- shrinks projections using uncertainty + DNP risk signals

5. `apply_contest_objective(...)`
- `Cash`: objective ~= projection
- `Small GPP`: projection + leverage component (+ optional tail)
- `Large GPP`: stronger leverage + ceiling (+ optional tail)

6. Optional objective bonus layers:
- Game Agent objective adjustments
- preferred-game bonus
- low-own candidate bonus

### 5.2 Candidate and Constraint Evaluation

Per lineup attempt, candidates are filtered by:
- not already selected
- not over exposure cap
- salary feasibility (cap and min salary used)
- position feasibility (`3G/3F` minimum remains achievable)
- spike overlap rules (if in pair mode)
- low-own requirement feasibility (if this lineup index is marked required)

Sampling:
- weighted random draw by objective score with jitter (`0.85-1.15`)
- multiple attempts; best-scoring valid lineup retained

Scoring penalties/bonuses include:
- salary-left distance from target
- overlap penalty vs previous lineups
- spike pair overlap and shared-game penalties
- spike pair bonus for different primary game
- optional preferred-game and ceiling-stack bonuses

### 5.3 Strategy Modes

### Standard
- independent lineups, overlap managed by penalty

### Spike (A/B pairs)
- lineups generated in pairs
- lineup B anchored against A with shared-player cap
- discourages identical game story in pair

### Cluster v1 Experimental
- builds cluster specs from game-level metrics:
  - high total
  - tight spread
  - tail leverage
  - contrarian
  - balanced
- each cluster has seed + mutation variants:
  - `same_role_swap`
  - `bring_back_rotation`
  - `stack_size_shift`
  - `chalk_pivot`
  - `value_risk_swap`
  - `salary_texture_shift`
- enforces anchor game presence for relevant cluster scripts

### 5.4 Generated Lineup Payload Fields

Each lineup stores, at minimum:
- identity: `lineup_number`
- players and ids
- `salary`, `salary_left`
- `projected_points`
- `projected_ownership_sum`
- `lineup_strategy`
- plus strategy-specific metadata (`pair_id`, `cluster_id`, `mutation_type`, etc)
- diagnostics: `stack_signature`, `salary_texture_bucket`, `low_own_upside_count`, `ceiling_boost_active`

## 6. Run Bundles and Persistence

Each run is persisted with:
- one manifest per run
- one JSON payload per version (`lineups.json`)
- one CSV per version (`lineups.csv`)
- one DK-upload CSV per version (`dk_upload.csv`)

GCS paths:
- run manifests: `cbb/lineup_runs/<date>/<slate_key>/<run_id>/manifest.json`
- version JSON: `.../<version_key>/lineups.json`
- version CSV: `.../<version_key>/lineups.csv`
- upload CSV: `.../<version_key>/dk_upload.csv`

## 7. Phantom Lineup Builder (Tournament Review)

Phantom review is built on saved lineup runs.

Workflow:
1. Choose date and saved run.
2. Select one or more versions to score.
3. Load actual player results (`cbb/players/<date>_players.csv`).
4. Score generated lineups via `score_generated_lineups_against_actuals(...)`.
5. Optionally compare to field standings via `compare_phantom_entries_to_field(...)`.
6. Summarize by version via `summarize_phantom_entries(...)`.

Scoring details:
- actual points match order:
  - by player ID
  - fallback by normalized name
  - fallback by loose normalized name
- outputs include:
  - `actual_points`
  - `actual_minus_projected`
  - `coverage_pct`
  - missing-player diagnostics
- if field comparison enabled:
  - `would_rank`
  - `would_beat_pct`
  - `winner_gap`
  - `pct_of_winner`

Phantom persistence:
- version CSVs: `cbb/phantom_reviews/<date>/<run_id>/<version>.csv`
- summary JSON: `cbb/phantom_reviews/<date>/<run_id>/summary.json`

## 8. Phantom Feedback Loop into Lineup Generation

### 8.1 Auto Projection Calibration

`compute_projection_calibration_from_phantom(...)`:
- scans prior phantom summaries in lookback window
- computes weighted `actual / projected`
- clips scale to `[0.75, 1.05]`
- used as `projection_scale`

### 8.2 Salary/Role Residual Calibration

From historical projection-vs-actual:
- salary bucket scales (with min sample threshold)
- role bucket scales (with min sample threshold)

### 8.3 Version Promotion (All Versions)

`compute_phantom_version_performance_weights(...)`:
- uses prior phantom summary rows
- computes weighted score by:
  - `avg_would_beat_pct` (primary)
  - `avg_actual_minus_projected` (secondary)
  - weighted by lineup volume
- converts to normalized weights
- allocates total lineups across versions with `_allocate_weighted_counts(...)`

Important behavior:
- In `All Versions` mode, total requested lineups = `lineup_count * number_of_versions`.
- Promotion redistributes that total across versions (minimum one per version).

## 9. Consistency and Review Layer

`Lineup Consistency Agent` runs against generated settings and lineups:
- pre-game checks always
- phantom diagnostics when phantom results exist
- validates:
  - locks/excludes
  - exposure caps
  - salary constraints
  - low-own/ceiling targets
  - game-bias alignment
  - phantom promotion allocation consistency

## 10. Practical Notes

- If low-own bucket is enabled but no candidates qualify, generation continues with warning.
- Locks override global/per-player exposure caps.
- Strict salary utilization can significantly reduce feasible lineup space.
- Phantom promotion only applies in `All Versions` mode.
- Phantom review requires actual results for the same date; field comparison also needs contest standings.
