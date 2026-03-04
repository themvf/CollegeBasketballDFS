import SectionPlaceholder from "../../components/section-placeholder";

export default function RotoWirePage() {
  return (
    <SectionPlaceholder
      title="RotoWire Scraper"
      description="RotoWire slate ingestion, normalized projections, and registry coverage."
      milestones={[
        "Add full slate picker and player grid from `/v1/rotowire/slates`.",
        "Expose DK coverage diagnostics and mismatch export.",
        "Embed manual override tools for unresolved players.",
      ]}
    />
  );
}

