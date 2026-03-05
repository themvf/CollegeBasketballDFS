import { fetchHealth, fetchRegistryCoverage } from "../lib/api";

type GameDataViewProps = {
  selectedDate: string;
};

function statusClass(value: number, mode: "coverage" | "count"): string {
  if (mode === "coverage") {
    if (value >= 98) return "ok";
    if (value >= 90) return "warn";
    return "bad";
  }
  if (value <= 0) return "ok";
  if (value <= 5) return "warn";
  return "bad";
}

export default async function GameDataView({ selectedDate }: GameDataViewProps) {
  const [healthOk, coverage] = await Promise.all([fetchHealth(), fetchRegistryCoverage(selectedDate)]);
  const coveragePct = Number(coverage?.coverage?.coverage_pct ?? 0);
  const unresolvedCount = Number(coverage?.coverage?.unresolved_players ?? 0);
  const conflictCount = Number(coverage?.coverage?.conflict_players ?? 0);
  const resolvedCount = Number(coverage?.coverage?.resolved_players ?? 0);
  const totalCount = Number(coverage?.coverage?.players_total ?? 0);
  const unresolvedNames = (coverage?.unresolved_players ?? [])
    .slice(0, 8)
    .map((item) => item.player_name)
    .filter((name): name is string => Boolean(name));

  return (
    <main className="page">
      <section className="hero">
        <h1>Game Data</h1>
        <p>Start the lineup workflow by validating feed health and DK identity coverage for the selected slate date.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">API: {healthOk ? "Connected" : "Unavailable"}</span>
          <span className="badge">Source: RotoWire + DK Registry</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Registry Coverage</h2>
          <p className="meta">This should be near 100% before generating lineups.</p>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Coverage %</p>
              <p className={`value ${statusClass(coveragePct, "coverage")}`}>{coveragePct.toFixed(2)}%</p>
            </div>
            <div className="metric">
              <p className="label">Resolved</p>
              <p className="value">{resolvedCount}</p>
            </div>
            <div className="metric">
              <p className="label">Unresolved</p>
              <p className={`value ${statusClass(unresolvedCount, "count")}`}>{unresolvedCount}</p>
            </div>
            <div className="metric">
              <p className="label">Conflicts</p>
              <p className={`value ${statusClass(conflictCount, "count")}`}>{conflictCount}</p>
            </div>
            <div className="metric">
              <p className="label">Total Players</p>
              <p className="value">{totalCount}</p>
            </div>
            <div className="metric">
              <p className="label">Slate</p>
              <p className="value">{coverage?.slate?.slate_name ?? "N/A"}</p>
            </div>
          </div>
        </article>

        <article className="panel">
          <h2>Workflow Path</h2>
          <p className="meta">Generate Lineup flow mirrors your current admin path:</p>
          <ol className="list">
            <li>Game Data</li>
            <li>Prop Data</li>
            <li>Backfill</li>
            <li>DK Slate</li>
            <li>Injuries</li>
            <li>Slate + Vegas</li>
            <li>RotoWire Scraper</li>
            <li>Lineup Generator</li>
            <li>Projection Review</li>
            <li>Tournament Review</li>
          </ol>
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Current Unresolved Players</h2>
        {unresolvedNames.length === 0 ? (
          <p className="meta">No unresolved players for this date.</p>
        ) : (
          <ul className="list">
            {unresolvedNames.map((name) => (
              <li key={name}>{name}</li>
            ))}
          </ul>
        )}
      </section>
    </main>
  );
}
