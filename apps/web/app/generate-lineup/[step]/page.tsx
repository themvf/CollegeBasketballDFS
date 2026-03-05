import { notFound } from "next/navigation";
import BackfillView from "../../../components/backfill-view";
import DkSlateImportView from "../../../components/dk-slate-import-view";
import GameDataView from "../../../components/game-data-view";
import LineupGeneratorView from "../../../components/lineup-generator-view";
import PropDataView from "../../../components/prop-data-view";
import RotowireScraperView from "../../../components/rotowire-scraper-view";
import SectionPlaceholder from "../../../components/section-placeholder";
import SlateVegasView from "../../../components/slate-vegas-view";

type GenerateLineupStepPageProps = {
  params: Promise<{ step: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

type StepContent = {
  title: string;
  description: string;
  milestones: string[];
};

const STEP_CONTENT: Record<string, StepContent> = {
  "prop-data": {
    title: "Prop Data",
    description: "Import and validate player-prop coverage used for projection and ownership context.",
    milestones: [
      "Trigger pregame props import by slate date and marketplace filter.",
      "Show market-level completeness and missing-line diagnostics.",
      "Persist import health checks for same-day lineup QA.",
    ],
  },
  backfill: {
    title: "Backfill",
    description: "Run historical backfill jobs when game, odds, or player rows are missing.",
    milestones: [
      "Submit season backfill jobs with per-date progress.",
      "Expose odds backfill retries and surfaced API failures.",
      "Track cache hits and failed dates for operational follow-up.",
    ],
  },
  "dk-slate": {
    title: "DK Slate",
    description: "Load DraftKings salary data and resolve unmatched players into the registry.",
    milestones: [
      "Upload or auto-map DK slate players to stable IDs.",
      "Highlight unresolved and conflict mappings for same-day repair.",
      "Persist manual resolutions for future slates.",
    ],
  },
  injuries: {
    title: "Injuries",
    description: "Review feed + manual injury overrides before building the optimizer pool.",
    milestones: [
      "Merge source feed and manual status overrides.",
      "Flag players removed from the slate due to injury status.",
      "Keep a date-scoped injury snapshot for postmortem replay.",
    ],
  },
  "slate-vegas": {
    title: "Slate + Vegas",
    description: "Build the pre-lock pool with spreads, totals, moneyline, and stack context.",
    milestones: [
      "Refresh slate pool and blended projection columns.",
      "Expose game-environment diagnostics for stack prioritization.",
      "Save a pre-lock snapshot for projection and ownership review.",
    ],
  },
  "rotowire-scraper": {
    title: "RotoWire Scraper",
    description: "Pull RotoWire projections/minutes and inspect normalization + DK mapping quality.",
    milestones: [
      "Run authenticated scraper pull for the selected slate.",
      "Compare scraped rows against DK registry coverage.",
      "Surface unresolved players and conflict candidates.",
    ],
  },
  "projection-review": {
    title: "Projection Review",
    description: "Evaluate projected vs actual fantasy points and ownership calibration.",
    milestones: [
      "Compute projection MAE/RMSE and segment-level bias.",
      "Track ownership accuracy by projected points tier.",
      "Save review artifacts to support model iteration.",
    ],
  },
  "tournament-review": {
    title: "Tournament Review",
    description: "Run contest postmortem across stacks, leverage, and final lineup outcomes.",
    milestones: [
      "Compare your portfolio against top field constructions.",
      "Surface missed stack exposure and chalk-miss diagnostics.",
      "Export postmortem packet for agentic review.",
    ],
  },
};

function getTodayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

export default async function GenerateLineupStepPage({ params, searchParams }: GenerateLineupStepPageProps) {
  const resolvedParams = await params;
  const query = await searchParams;
  const selectedDate = typeof query.date === "string" ? query.date : getTodayIsoDate();
  const step = String(resolvedParams.step || "").trim().toLowerCase();

  if (step === "game-data") {
    return <GameDataView selectedDate={selectedDate} />;
  }

  if (step === "prop-data") {
    return <PropDataView selectedDate={selectedDate} />;
  }

  if (step === "backfill") {
    return <BackfillView endDate={selectedDate} />;
  }

  if (step === "dk-slate") {
    return <DkSlateImportView selectedDate={selectedDate} />;
  }

  if (step === "slate-vegas") {
    return <SlateVegasView selectedDate={selectedDate} />;
  }

  if (step === "rotowire-scraper") {
    return <RotowireScraperView selectedDate={selectedDate} />;
  }

  if (step === "lineup-generator") {
    return <LineupGeneratorView />;
  }

  const content = STEP_CONTENT[step];
  if (!content) {
    notFound();
  }

  return (
    <SectionPlaceholder
      title={content.title}
      description={content.description}
      milestones={content.milestones}
    />
  );
}
