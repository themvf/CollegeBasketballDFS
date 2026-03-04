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

function baseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
}

export async function fetchRegistryCoverage(selectedDate: string): Promise<CoverageResponse | null> {
  const url = `${baseUrl()}/v1/registry/coverage?selected_date=${encodeURIComponent(selectedDate)}&contest_type=Classic&slate_name=All&slate_key=main`;
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

export async function fetchHealth(): Promise<boolean> {
  const url = `${baseUrl()}/health`;
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
