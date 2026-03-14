export type CoverageResponse = {
  slate?: {
    slate_id?: number;
    slate_date?: string;
    contest_type?: string;
    slate_name?: string;
    game_count?: number;
    players?: number;
  };
  coverage?: {
    players_total?: number;
    resolved_players?: number;
    unresolved_players?: number;
    conflict_players?: number;
    coverage_pct?: number;
    fully_resolved?: boolean;
  };
  unresolved_players?: Array<{
    player_name?: string;
    team_abbr?: string;
    salary?: number;
    dk_resolution_status?: string;
    dk_match_reason?: string;
  }>;
};

export type SlateStatusResponse = {
  selected_date: string;
  slate_key: string;
  slate_label?: string;
  slate?: {
    slate_id?: number;
    slate_date?: string;
    contest_type?: string;
    slate_name?: string;
    game_count?: number;
    players?: number;
  };
  active_rows?: number;
  cached_dk_rows?: number;
  slate_games?: number;
  active_source?: {
    code?: string;
    label?: string;
    detail?: string;
    ready?: boolean;
  };
  cached_dk_slate?: {
    available?: boolean;
    rows?: number;
  };
  rotowire?: {
    loaded?: boolean;
    rows?: number;
    fully_resolved?: boolean;
    error?: string;
  };
  registry_coverage?: {
    players_total?: number;
    resolved_players?: number;
    unresolved_players?: number;
    conflict_players?: number;
    coverage_pct?: number;
    fully_resolved?: boolean;
    error?: string;
    unresolved_sample?: Array<{
      player_name?: string;
      team_abbr?: string;
      dk_resolution_status?: string;
      dk_match_reason?: string;
    }>;
  };
};

export type LineupRunVersionSummary = {
  version_key?: string;
  version_label?: string;
  lineup_strategy?: string;
  model_profile?: string;
  include_tail_signals?: boolean;
  lineup_count_generated?: number;
  warning_count?: number;
  json_ref?: string;
  csv_ref?: string;
  dk_upload_ref?: string;
};

export type LineupRunSummary = {
  run_id?: string;
  slate_date?: string;
  slate_key?: string;
  slate_label?: string;
  generated_at_utc?: string;
  run_mode?: string;
  version_count?: number;
  lineups_generated?: number;
  warnings_count?: number;
  storage_source?: string;
  manifest_ref?: string;
  versions?: LineupRunVersionSummary[];
  settings?: Record<string, unknown>;
};

export type LineupRunVersionDetail = LineupRunVersionSummary & {
  loaded?: boolean;
  payload_source?: string;
  payload_ref?: string;
  warnings?: string[];
  lineups_loaded?: number;
  lineups?: Array<Record<string, unknown>>;
  dk_upload_csv?: string;
};

export type LineupRunsResponse = {
  selected_date: string;
  slate_key: string;
  rows: number;
  runs: LineupRunSummary[];
};

export type LineupRunDetailResponse = {
  selected_date: string;
  slate_key: string;
  run?: LineupRunSummary;
  versions?: LineupRunVersionDetail[];
};

export type VegasGameLinesResponse = {
  selected_date: string;
  bucket_name?: string;
  available: boolean;
  source_status?: {
    raw_cached?: boolean;
    odds_cached?: boolean;
  };
  summary?: {
    total_games?: number;
    odds_matched_games?: number;
    total_line_games?: number;
    spread_line_games?: number;
    total_mae?: number;
    spread_mae?: number;
    winner_pick_accuracy_pct?: number;
  };
  rows?: Array<{
    game_date?: string;
    away_team?: string;
    home_team?: string;
    total_points?: number;
    actual_total?: number;
    total_error?: number;
    spread_home?: number;
    actual_home_margin?: number;
    spread_error?: number;
    moneyline_home?: number;
    moneyline_away?: number;
    winner_pick_correct?: boolean;
    odds_match_type?: string;
  }>;
};

export type VegasMarketContextResponse = {
  selected_date: string;
  bucket_name?: string;
  available: boolean;
  source_status?: {
    raw_cached?: boolean;
    odds_cached?: boolean;
  };
  summary?: {
    total_games?: number;
    total_mae?: number;
    spread_mae?: number;
  };
  calibration_models?: Array<{
    model?: string;
    samples?: number;
    slope?: number;
    intercept?: number;
    r2?: number;
    baseline_mae?: number;
    calibrated_mae?: number;
    mae_delta?: number;
  }>;
  total_buckets?: Array<{
    vegas_total_bucket?: string;
    games?: number;
    avg_vegas_total?: number;
    avg_actual_total?: number;
    mae_total?: number;
    over_rate_pct?: number;
    under_rate_pct?: number;
  }>;
  spread_buckets?: Array<{
    vegas_spread_bucket?: string;
    games?: number;
    avg_abs_vegas_margin?: number;
    avg_abs_actual_margin?: number;
    mae_spread?: number;
    winner_pick_accuracy_pct?: number;
  }>;
  ranked_games?: Array<{
    game_date?: string;
    away_team?: string;
    home_team?: string;
    total_points?: number;
    spread_home?: number;
    moneyline_home?: number;
    moneyline_away?: number;
    bookmakers_count?: number;
  }>;
};

export type VegasPropDataResponse = {
  selected_date: string;
  bucket_name?: string;
  available: boolean;
  source_status?: {
    props_cached?: boolean;
  };
  summary?: {
    rows?: number;
    markets?: number;
    players?: number;
    events?: number;
    bookmakers?: number;
  };
  market_coverage?: Array<{
    market?: string;
    rows?: number;
    players?: number;
    events?: number;
    avg_line?: number;
  }>;
  rows?: Array<{
    game_date?: string;
    away_team?: string;
    home_team?: string;
    bookmaker?: string;
    market?: string;
    player_name?: string;
    line?: number;
    over_price?: number;
    under_price?: number;
  }>;
};

export type CacheCoverageStats = {
  found_dates?: number;
  total_dates?: number;
  coverage_pct?: number;
  missing_dates?: string[];
  sample_found_dates?: string[];
};

export type CacheCoverageResponse = {
  bucket_name?: string;
  start_date: string;
  end_date: string;
  total_dates: number;
  coverage?: {
    raw_game_data?: CacheCoverageStats;
    odds_data?: CacheCoverageStats;
    odds_games_csv?: CacheCoverageStats;
    props_data?: CacheCoverageStats;
    props_lines_csv?: CacheCoverageStats;
    players_results?: CacheCoverageStats;
    dk_slates?: CacheCoverageStats;
    projections_snapshots?: CacheCoverageStats;
    ownership_files?: CacheCoverageStats;
    injuries_feed?: CacheCoverageStats;
  };
};

export type InjuriesReviewResponse = {
  selected_date: string;
  bucket_name?: string;
  available: boolean;
  legacy_fallback_used?: boolean;
  summary?: {
    effective_rows?: number;
    feed_rows?: number;
    manual_rows?: number;
    active_rows?: number;
    remove_candidates?: number;
  };
  status_counts?: Array<{
    status?: string;
    rows?: number;
    active?: number;
  }>;
  effective_rows?: Array<Record<string, unknown>>;
  feed_rows?: Array<Record<string, unknown>>;
  manual_rows?: Array<Record<string, unknown>>;
};

export type ProjectionReviewResponse = {
  selected_date: string;
  bucket_name?: string;
  available: boolean;
  summary?: {
    projection_rows?: number;
    actual_matched?: number;
    blend_mae?: number | null;
    our_mae?: number | null;
    vegas_mae?: number | null;
    ownership_rows?: number;
    ownership_mae?: number | null;
    ownership_bias?: number | null;
    ownership_rank_spearman?: number | null;
  };
  rows?: Array<Record<string, unknown>>;
};

export type TournamentReviewResponse = {
  selected_date: string;
  contest_id: string;
  bucket_name?: string;
  available: boolean;
  message?: string;
  upload_profile?: Record<string, unknown>;
  summary?: Record<string, unknown>;
  ownership_summary?: Record<string, unknown>;
  projection_summary?: Record<string, unknown>;
  entries_rows?: Array<Record<string, unknown>>;
  exposure_rows?: Array<Record<string, unknown>>;
  ownership_buckets?: Array<Record<string, unknown>>;
  ownership_top_misses?: Array<Record<string, unknown>>;
  projection_rows?: Array<Record<string, unknown>>;
};

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
}

export async function fetchRegistryCoverage(selectedDate: string): Promise<CoverageResponse | null> {
  const url = `${getApiBaseUrl()}/v1/registry/coverage?selected_date=${encodeURIComponent(selectedDate)}&contest_type=Classic&slate_name=All&slate_key=main`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as CoverageResponse;
  } catch {
    return null;
  }
}

export async function fetchSlateStatus(
  selectedDate: string,
  slateKey = "main",
): Promise<SlateStatusResponse | null> {
  const url =
    `${getApiBaseUrl()}/v1/slates/${encodeURIComponent(selectedDate)}/${encodeURIComponent(slateKey)}/status` +
    "?contest_type=Classic&slate_name=All";
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as SlateStatusResponse;
  } catch {
    return null;
  }
}

export async function fetchLineupRuns(
  selectedDate: string,
  slateKey = "main",
  includeVersions = false,
): Promise<LineupRunsResponse | null> {
  const url =
    `${getApiBaseUrl()}/v1/lineup-runs?date=${encodeURIComponent(selectedDate)}` +
    `&slate_key=${encodeURIComponent(slateKey)}&include_versions=${includeVersions ? "true" : "false"}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as LineupRunsResponse;
  } catch {
    return null;
  }
}

export async function fetchLineupRunDetail(
  selectedDate: string,
  runId: string,
  slateKey = "main",
): Promise<LineupRunDetailResponse | null> {
  const url =
    `${getApiBaseUrl()}/v1/lineup-runs/${encodeURIComponent(runId)}?date=${encodeURIComponent(selectedDate)}` +
    `&slate_key=${encodeURIComponent(slateKey)}&include_lineups=true`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as LineupRunDetailResponse;
  } catch {
    return null;
  }
}

export async function fetchHealth(): Promise<boolean> {
  const url = `${getApiBaseUrl()}/health`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return false;
    }
    const payload = (await response.json()) as { ok?: boolean };
    return Boolean(payload.ok);
  } catch {
    return false;
  }
}

export async function fetchVegasGameLines(selectedDate: string): Promise<VegasGameLinesResponse | null> {
  const url = `${getApiBaseUrl()}/v1/vegas/game-lines?selected_date=${encodeURIComponent(selectedDate)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as VegasGameLinesResponse;
  } catch {
    return null;
  }
}

export async function fetchVegasMarketContext(selectedDate: string): Promise<VegasMarketContextResponse | null> {
  const url = `${getApiBaseUrl()}/v1/vegas/market-context?selected_date=${encodeURIComponent(selectedDate)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as VegasMarketContextResponse;
  } catch {
    return null;
  }
}

export async function fetchVegasPropData(selectedDate: string): Promise<VegasPropDataResponse | null> {
  const url = `${getApiBaseUrl()}/v1/vegas/prop-data?selected_date=${encodeURIComponent(selectedDate)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as VegasPropDataResponse;
  } catch {
    return null;
  }
}

export async function fetchCacheCoverage(startDate: string, endDate: string): Promise<CacheCoverageResponse | null> {
  const url = `${getApiBaseUrl()}/v1/ops/cache-coverage?start_date=${encodeURIComponent(startDate)}&end_date=${encodeURIComponent(endDate)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as CacheCoverageResponse;
  } catch {
    return null;
  }
}

export async function fetchInjuriesReview(selectedDate: string): Promise<InjuriesReviewResponse | null> {
  const url = `${getApiBaseUrl()}/v1/injuries/review?selected_date=${encodeURIComponent(selectedDate)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as InjuriesReviewResponse;
  } catch {
    return null;
  }
}

export async function fetchProjectionReview(
  selectedDate: string,
  slateKey = "main",
): Promise<ProjectionReviewResponse | null> {
  const url =
    `${getApiBaseUrl()}/v1/reviews/projection?selected_date=${encodeURIComponent(selectedDate)}` +
    `&slate_key=${encodeURIComponent(slateKey)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as ProjectionReviewResponse;
  } catch {
    return null;
  }
}

export async function fetchTournamentReview(
  selectedDate: string,
  contestId: string,
  slateKey = "main",
): Promise<TournamentReviewResponse | null> {
  const url =
    `${getApiBaseUrl()}/v1/reviews/tournament?selected_date=${encodeURIComponent(selectedDate)}` +
    `&contest_id=${encodeURIComponent(contestId)}&slate_key=${encodeURIComponent(slateKey)}`;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return null;
    }
    return (await response.json()) as TournamentReviewResponse;
  } catch {
    return null;
  }
}
