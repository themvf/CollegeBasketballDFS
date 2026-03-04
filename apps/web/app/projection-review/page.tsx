import SectionPlaceholder from "../../components/section-placeholder";

export default function ProjectionReviewPage() {
  return (
    <SectionPlaceholder
      title="Projection Review"
      description="Projection calibration, ownership fit, and segment-level diagnostics."
      milestones={[
        "Port projection bias heatmaps and segment impact tables.",
        "Add projected-vs-actual ownership scatter and calibration deciles.",
        "Persist review packets for model iteration traceability.",
      ]}
    />
  );
}

