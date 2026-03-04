import SectionPlaceholder from "../../components/section-placeholder";

export default function InjuriesPage() {
  return (
    <SectionPlaceholder
      title="Injuries"
      description="Injury feed controls, manual edits, and slate-level injury impact."
      milestones={[
        "Expose injury feed refresh and manual merge actions.",
        "Add conflict review table for player status overrides.",
        "Wire injury-impact indicators into lineup pool preview.",
      ]}
    />
  );
}

