import SectionPlaceholder from "../../components/section-placeholder";

export default function PropsImportPage() {
  return (
    <SectionPlaceholder
      title="Props Import"
      description="Pregame props ingestion, validation, and coverage checks."
      milestones={[
        "Add API trigger for props import by date/mode.",
        "Add bookmaker filters and line distribution tables.",
        "Add missing-market diagnostics and exportable error report.",
      ]}
    />
  );
}

