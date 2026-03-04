import SectionPlaceholder from "../../components/section-placeholder";

export default function SeasonBackfillPage() {
  return (
    <SectionPlaceholder
      title="Season Backfill"
      description="Bulk historical load pipeline for missing game and player data."
      milestones={[
        "Submit backfill jobs asynchronously with progress events.",
        "Track per-date error summaries with retry from UI.",
        "Persist run history and throughput metrics for operations.",
      ]}
    />
  );
}

