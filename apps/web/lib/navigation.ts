export type PrimaryNavKey = "generate-lineup" | "player-team-review" | "agentic-review" | "vegas-review";

export type PrimaryNavItem = {
  key: PrimaryNavKey;
  label: string;
  href: string;
  description: string;
};

export type SubNavItem = {
  key: string;
  label: string;
  href: string;
  description: string;
};

export const PRIMARY_NAV_ITEMS: PrimaryNavItem[] = [
  {
    key: "generate-lineup",
    label: "Generate Lineup",
    href: "/generate-lineup/game-data",
    description: "End-to-end slate prep and lineup generation workflow.",
  },
  {
    key: "player-team-review",
    label: "Player and Team Review",
    href: "/player-team-review/player-review",
    description: "Player and team diagnostics to improve projections and ownership fit.",
  },
  {
    key: "agentic-review",
    label: "Agentic Review",
    href: "/agentic-review/single-slate",
    description: "AI-assisted review packets, prompts, and postmortem recommendations.",
  },
  {
    key: "vegas-review",
    label: "Vegas Review",
    href: "/vegas-review/game-lines",
    description: "Game-level market context from totals, spreads, and moneyline data.",
  },
];

const GENERATE_LINEUP_SUBNAV: SubNavItem[] = [
  {
    key: "game-data",
    label: "Game Data",
    href: "/generate-lineup/game-data",
    description: "Game ingestion and DK-ID coverage checks.",
  },
  {
    key: "prop-data",
    label: "Prop Data",
    href: "/generate-lineup/prop-data",
    description: "Prop import controls and coverage review.",
  },
  {
    key: "backfill",
    label: "Backfill",
    href: "/generate-lineup/backfill",
    description: "Season and odds backfill operations.",
  },
  {
    key: "dk-slate",
    label: "DK Slate",
    href: "/generate-lineup/dk-slate",
    description: "DraftKings slate ingestion and mapping.",
  },
  {
    key: "injuries",
    label: "Injuries",
    href: "/generate-lineup/injuries",
    description: "Injury feed and manual overrides.",
  },
  {
    key: "slate-vegas",
    label: "Slate + Vegas",
    href: "/generate-lineup/slate-vegas",
    description: "Slate pool and game environment diagnostics.",
  },
  {
    key: "rotowire-scraper",
    label: "RotoWire Scraper",
    href: "/generate-lineup/rotowire-scraper",
    description: "RotoWire pull, normalization, and validation.",
  },
  {
    key: "lineup-generator",
    label: "Lineup Generator",
    href: "/generate-lineup/lineup-generator",
    description: "Async generation and artifact downloads.",
  },
  {
    key: "saved-runs",
    label: "Saved Runs",
    href: "/generate-lineup/saved-runs",
    description: "Browse persisted lineup runs and inspect version outputs.",
  },
  {
    key: "projection-review",
    label: "Projection Review",
    href: "/generate-lineup/projection-review",
    description: "Projection bias and ownership calibration review.",
  },
  {
    key: "tournament-review",
    label: "Tournament Review",
    href: "/generate-lineup/tournament-review",
    description: "Contest postmortem and portfolio breakdown.",
  },
];

const PLAYER_TEAM_REVIEW_SUBNAV: SubNavItem[] = [
  {
    key: "player-review",
    label: "Player Review",
    href: "/player-team-review/player-review",
    description: "Player-level projection and ownership performance.",
  },
  {
    key: "team-review",
    label: "Team Review",
    href: "/player-team-review/team-review",
    description: "Team-level pace, total, and stack environment review.",
  },
  {
    key: "projection-review",
    label: "Projection Review",
    href: "/player-team-review/projection-review",
    description: "Projection quality, calibration, and residual analysis.",
  },
  {
    key: "tournament-review",
    label: "Tournament Review",
    href: "/player-team-review/tournament-review",
    description: "Lineup outcomes and missed leverage patterns.",
  },
];

const AGENTIC_REVIEW_SUBNAV: SubNavItem[] = [
  {
    key: "single-slate",
    label: "Single-Slate",
    href: "/agentic-review/single-slate",
    description: "Deterministic packet and recommendations for one contest.",
  },
  {
    key: "global-review",
    label: "Global Review",
    href: "/agentic-review/global-review",
    description: "Cross-slate AI diagnostics for recurring model gaps.",
  },
  {
    key: "postmortem-agent",
    label: "Postmortem Agent",
    href: "/agentic-review/postmortem-agent",
    description: "Contest postmortem prompt and model output workflow.",
  },
  {
    key: "game-slate-agent",
    label: "Game Slate Agent",
    href: "/agentic-review/game-slate-agent",
    description: "Pre-lock game environment and stack recommendation agent.",
  },
];

const VEGAS_REVIEW_SUBNAV: SubNavItem[] = [
  {
    key: "game-lines",
    label: "Game Lines",
    href: "/vegas-review/game-lines",
    description: "Totals, spreads, and moneyline overview.",
  },
  {
    key: "market-context",
    label: "Market Context",
    href: "/vegas-review/market-context",
    description: "Slate-level game environments ranked by market signal.",
  },
  {
    key: "prop-data",
    label: "Prop Data",
    href: "/vegas-review/prop-data",
    description: "Player prop availability and import health.",
  },
];

const SUBNAV_BY_PRIMARY: Record<PrimaryNavKey, SubNavItem[]> = {
  "generate-lineup": GENERATE_LINEUP_SUBNAV,
  "player-team-review": PLAYER_TEAM_REVIEW_SUBNAV,
  "agentic-review": AGENTIC_REVIEW_SUBNAV,
  "vegas-review": VEGAS_REVIEW_SUBNAV,
};

const LEGACY_ROUTE_MAP: Array<{ legacyPath: string; canonicalPath: string }> = [
  { legacyPath: "/", canonicalPath: "/generate-lineup/game-data" },
  { legacyPath: "/game-pipeline", canonicalPath: "/generate-lineup/game-data" },
  { legacyPath: "/props-import", canonicalPath: "/generate-lineup/prop-data" },
  { legacyPath: "/season-backfill", canonicalPath: "/generate-lineup/backfill" },
  { legacyPath: "/dk-slate", canonicalPath: "/generate-lineup/dk-slate" },
  { legacyPath: "/injuries", canonicalPath: "/generate-lineup/injuries" },
  { legacyPath: "/slate-vegas", canonicalPath: "/generate-lineup/slate-vegas" },
  { legacyPath: "/rotowire", canonicalPath: "/generate-lineup/rotowire-scraper" },
  { legacyPath: "/lineups", canonicalPath: "/generate-lineup/lineup-generator" },
  { legacyPath: "/projection-review", canonicalPath: "/player-team-review/projection-review" },
  { legacyPath: "/tournament-review", canonicalPath: "/player-team-review/tournament-review" },
];

function normalizePath(pathname: string): string {
  const base = String(pathname || "/").split("?")[0].split("#")[0];
  if (!base) {
    return "/";
  }
  if (base.length > 1 && base.endsWith("/")) {
    return base.slice(0, -1);
  }
  return base;
}

export function canonicalizePath(pathname: string): string {
  const normalized = normalizePath(pathname);
  const mapped = LEGACY_ROUTE_MAP.find(
    ({ legacyPath }) => normalized === legacyPath || normalized.startsWith(`${legacyPath}/`),
  );
  return mapped?.canonicalPath ?? normalized;
}

export function resolvePrimaryNavKey(pathname: string): PrimaryNavKey {
  const canonical = canonicalizePath(pathname);
  if (canonical.startsWith("/player-team-review")) {
    return "player-team-review";
  }
  if (canonical.startsWith("/agentic-review")) {
    return "agentic-review";
  }
  if (canonical.startsWith("/vegas-review")) {
    return "vegas-review";
  }
  return "generate-lineup";
}

export function getPrimaryNavItem(pathname: string): PrimaryNavItem {
  const activeKey = resolvePrimaryNavKey(pathname);
  return PRIMARY_NAV_ITEMS.find((item) => item.key === activeKey) ?? PRIMARY_NAV_ITEMS[0];
}

export function getSubNavItems(primaryKey: PrimaryNavKey): SubNavItem[] {
  return SUBNAV_BY_PRIMARY[primaryKey];
}

export function getActiveSubNavItem(pathname: string): SubNavItem | undefined {
  const canonical = canonicalizePath(pathname);
  const primary = resolvePrimaryNavKey(canonical);
  return SUBNAV_BY_PRIMARY[primary].find(
    (item) => canonical === item.href || canonical.startsWith(`${item.href}/`),
  );
}
