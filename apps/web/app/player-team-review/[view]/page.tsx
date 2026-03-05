import { notFound } from "next/navigation";
import SectionPlaceholder from "../../../components/section-placeholder";

type PlayerTeamReviewPageProps = {
  params: Promise<{ view: string }>;
};

type ReviewContent = {
  title: string;
  description: string;
  milestones: string[];
};

const REVIEW_CONTENT: Record<string, ReviewContent> = {
  "player-review": {
    title: "Player Review",
    description: "Player-level diagnostics for projected points, minutes trend, and ownership miss patterns.",
    milestones: [
      "Filter by minimum projected points (default 15) and ownership caps.",
      "Highlight high-projection low-ownership tournament candidates.",
      "Track projected vs actual ownership by player cohort.",
    ],
  },
  "team-review": {
    title: "Team Review",
    description: "Team-level review of environment, rotation stability, and stack viability.",
    milestones: [
      "Rank teams by implied totals, pace proxies, and rotation confidence.",
      "Summarize stack hit rates and underexposed game environments.",
      "Surface team-level minutes volatility warnings.",
    ],
  },
  "projection-review": {
    title: "Projection Review",
    description: "Model calibration by position, salary tier, and team context.",
    milestones: [
      "Track MAE/RMSE and directional bias by segment.",
      "Show projection bias heatmap by salary x position.",
      "Compare pre/post adjustments and residual drift.",
    ],
  },
  "tournament-review": {
    title: "Tournament Review",
    description: "Contest-level postmortem with field comparison and leverage misses.",
    milestones: [
      "Audit top lineup stacks vs your generated portfolio.",
      "Quantify missed chalk, false chalk, and ownership directional errors.",
      "Export packet for agentic recommendation workflow.",
    ],
  },
};

export default async function PlayerTeamReviewPage({ params }: PlayerTeamReviewPageProps) {
  const resolvedParams = await params;
  const view = String(resolvedParams.view || "").trim().toLowerCase();
  const content = REVIEW_CONTENT[view];
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
