import { notFound } from "next/navigation";
import SectionPlaceholder from "../../../components/section-placeholder";

type VegasReviewPageProps = {
  params: Promise<{ view: string }>;
};

type VegasContent = {
  title: string;
  description: string;
  milestones: string[];
};

const VEGAS_CONTENT: Record<string, VegasContent> = {
  "game-lines": {
    title: "Vegas Review - Game Lines",
    description: "Review totals, spreads, and moneyline across the active slate.",
    milestones: [
      "Rank games by total and implied environment strength.",
      "Flag close-spread, high-total games for stack pressure.",
      "Publish a concise game-priority table for lineup generation.",
    ],
  },
  "market-context": {
    title: "Vegas Review - Market Context",
    description: "Track how market context aligns with ownership and projection assumptions.",
    milestones: [
      "Compare implied team totals against projection baselines.",
      "Highlight games where market signal conflicts with ownership model.",
      "Feed adjustments into stack and leverage controls.",
    ],
  },
  "prop-data": {
    title: "Vegas Review - Prop Data",
    description: "Prop availability diagnostics and fallback workflow when books are thin.",
    milestones: [
      "Monitor prop market coverage and stale-line conditions.",
      "Apply fallback weighting when player-prop data is unavailable.",
      "Expose data quality status to downstream lineup runs.",
    ],
  },
};

export default async function VegasReviewPage({ params }: VegasReviewPageProps) {
  const resolvedParams = await params;
  const view = String(resolvedParams.view || "").trim().toLowerCase();
  const content = VEGAS_CONTENT[view];
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
