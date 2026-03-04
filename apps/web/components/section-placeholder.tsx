type SectionPlaceholderProps = {
  title: string;
  description: string;
  milestones: string[];
};

export default function SectionPlaceholder({ title, description, milestones }: SectionPlaceholderProps) {
  return (
    <main className="page">
      <section className="hero">
        <h1>{title}</h1>
        <p>{description}</p>
      </section>
      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Migration Status</h2>
        <p className="meta">This module is scaffolded for navigation parity and ready for endpoint wiring.</p>
        <ol className="list">
          {milestones.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ol>
      </section>
    </main>
  );
}
