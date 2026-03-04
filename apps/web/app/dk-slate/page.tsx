import SectionPlaceholder from "../../components/section-placeholder";

export default function DkSlatePage() {
  return (
    <SectionPlaceholder
      title="DK Slate"
      description="DraftKings slate ingestion, identity mapping, and mismatch remediation."
      milestones={[
        "Add unresolved/conflict bulk resolver from uploaded slate.",
        "Show registry confidence and historical match provenance.",
        "Enable one-click import to manual override table.",
      ]}
    />
  );
}

