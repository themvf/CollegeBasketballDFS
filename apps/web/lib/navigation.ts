import type { Route } from "next";

export type NavItem = {
  href: Route;
  label: string;
  description: string;
  group: "overview" | "operations" | "analysis";
};

export const NAV_ITEMS: NavItem[] = [
  {
    href: "/",
    label: "Overview",
    description: "Migration health and registry coverage snapshot.",
    group: "overview",
  },
  {
    href: "/game-pipeline",
    label: "Game Pipeline",
    description: "Run game ingestion and cache refresh.",
    group: "operations",
  },
  {
    href: "/props-import",
    label: "Props Import",
    description: "Import and monitor player props data.",
    group: "operations",
  },
  {
    href: "/season-backfill",
    label: "Season Backfill",
    description: "Backfill historical dates and repair gaps.",
    group: "operations",
  },
  {
    href: "/dk-slate",
    label: "DK Slate",
    description: "DraftKings slate ingestion and identity mapping.",
    group: "operations",
  },
  {
    href: "/injuries",
    label: "Injuries",
    description: "Injury feed review and manual adjustments.",
    group: "operations",
  },
  {
    href: "/slate-vegas",
    label: "Slate + Vegas",
    description: "Slate prep and game-level market context.",
    group: "operations",
  },
  {
    href: "/rotowire",
    label: "RotoWire Scraper",
    description: "RotoWire slate feed, coverage, and overrides.",
    group: "operations",
  },
  {
    href: "/lineups",
    label: "Lineup Generator",
    description: "Async lineup jobs and artifact downloads.",
    group: "operations",
  },
  {
    href: "/projection-review",
    label: "Projection Review",
    description: "Projection diagnostics and calibration checks.",
    group: "analysis",
  },
  {
    href: "/tournament-review",
    label: "Tournament Review",
    description: "Postmortem and portfolio breakdown.",
    group: "analysis",
  },
];

export function getNavItem(pathname: string): NavItem | undefined {
  if (!pathname || pathname === "/") {
    return NAV_ITEMS.find((item) => item.href === "/");
  }
  return NAV_ITEMS.find((item) => pathname === item.href || pathname.startsWith(`${item.href}/`));
}
