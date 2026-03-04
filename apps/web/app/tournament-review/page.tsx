import SectionPlaceholder from "../../components/section-placeholder";

export default function TournamentReviewPage() {
  return (
    <SectionPlaceholder
      title="Tournament Review"
      description="Postmortem packet analysis for ownership misses, stack gaps, and portfolio outcomes."
      milestones={[
        "Port top-10 finisher comparison and phantom portfolio diagnostics.",
        "Add stack underexposure and false/missed chalk summaries.",
        "Generate repeatable review exports for model tuning.",
      ]}
    />
  );
}

