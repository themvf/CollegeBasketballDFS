import SectionPlaceholder from "../../components/section-placeholder";

export default function GamePipelinePage() {
  return (
    <SectionPlaceholder
      title="Game Pipeline"
      description="NCAA ingestion, raw cache management, and game-level pipeline controls."
      milestones={[
        "Expose fetch/backfill controls through API job endpoints.",
        "Surface per-date success/failure state with retry actions.",
        "Add raw JSON preview and transformed player table parity.",
      ]}
    />
  );
}

