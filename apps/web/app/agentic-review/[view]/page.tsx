import { notFound } from "next/navigation";
import SectionPlaceholder from "../../../components/section-placeholder";

type AgenticReviewPageProps = {
  params: Promise<{ view: string }>;
};

type AgenticContent = {
  title: string;
  description: string;
  milestones: string[];
};

const AGENTIC_CONTENT: Record<string, AgenticContent> = {
  "single-slate": {
    title: "Agentic Review - Single Slate",
    description: "Generate an AI packet and recommendation set for the currently selected contest date.",
    milestones: [
      "Build packet from projection, ownership, and tournament outputs.",
      "Generate focused recommendations for lineup process changes.",
      "Export packet + prompts for auditability.",
    ],
  },
  "global-review": {
    title: "Agentic Review - Global",
    description: "Scan a multi-slate lookback window to identify persistent model and process drift.",
    milestones: [
      "Aggregate projection, ownership, and lineup outcomes over lookback dates.",
      "Identify recurring misses in chalk handling and game-stack exposure.",
      "Prioritize changes by expected tournament impact.",
    ],
  },
  "postmortem-agent": {
    title: "Postmortem Agent",
    description: "Run contest postmortem prompts against saved packet artifacts.",
    milestones: [
      "Assemble deterministic postmortem packet for one contest.",
      "Generate concise recommendations with stack and ownership focus.",
      "Persist output for experiment tracking.",
    ],
  },
  "game-slate-agent": {
    title: "Game Slate Agent",
    description: "Pre-lock agent guidance for game-environment ranking and stack pressure.",
    milestones: [
      "Score each game for stack viability and environment strength.",
      "Flag games where projected ownership is likely undercalled.",
      "Emit actionable stack exposure targets for lineup generation.",
    ],
  },
};

export default async function AgenticReviewPage({ params }: AgenticReviewPageProps) {
  const resolvedParams = await params;
  const view = String(resolvedParams.view || "").trim().toLowerCase();
  const content = AGENTIC_CONTENT[view];
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
