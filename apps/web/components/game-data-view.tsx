import { fetchHealth, fetchSlateStatus } from "../lib/api";

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

function readinessClass(ready: boolean): string {
  return ready ? "ok" : "bad";
}

export default async function GameDataView({ selectedDate }: GameDataViewProps) {
  const [healthOk, slateStatus] = await Promise.all([fetchHealth(), fetchSlateStatus(selectedDate)]);
  const coverage = slateStatus?.registry_coverage ?? {};
  const coveragePct = Number(coverage.coverage_pct ?? 0);
  const unresolvedCount = Number(coverage.unresolved_players ?? 0);
  const conflictCount = Number(coverage.conflict_players ?? 0);
  const resolvedCount = Number(coverage.resolved_players ?? 0);
  const totalCount = Number(coverage.players_total ?? 0);
  const activeReady = Boolean(slateStatus?.active_source?.ready);
  const mappingOpenIssues = unresolvedCount + conflictCount;
  const notes = [
    slateStatus ? "" : "Slate status endpoint is unavailable.",
    !activeReady ? slateStatus?.active_source?.detail ?? "" : "",
    activeReady && coveragePct > 0 && coveragePct < 98
      ? `Supplemental mapping is partial (${coveragePct.toFixed(2)}% coverage, ${unresolvedCount} unresolved, ${conflictCount} conflicts). This does not block lineup generation.`
      : "",
  ].filter((value, index, arr): value is string => Boolean(value) && arr.indexOf(value) === index);
  const unresolvedRows = slateStatus?.registry_coverage?.unresolved_sample ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>Game Data</h1>
        <p>Validate the uploaded DraftKings slate and core readiness state before moving deeper into the lineup workflow.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">API: {healthOk ? "Connected" : "Unavailable"}</span>
          <span className="badge">Active Source: {slateStatus?.active_source?.label ?? "No active slate"}</span>
          <span className="badge">Ready: {activeReady ? "Yes" : "No"}</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Slate Status</h2>
          <p className="meta">This is the backend readiness state the web app will use for the lineup workflow.</p>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Active Slate</p>
              <p className={`value ${readinessClass(activeReady)}`}>{activeReady ? "Ready" : "Blocked"}</p>
            </div>
            <div className="metric">
              <p className="label">Active Rows</p>
              <p className="value">{Number(slateStatus?.active_rows ?? 0)}</p>
            </div>
            <div className="metric">
              <p className="label">Cached DK Rows</p>
              <p className="value">{Number(slateStatus?.cached_dk_rows ?? 0)}</p>
            </div>
            <div className="metric">
              <p className="label">Slate Games</p>
              <p className="value">{Number(slateStatus?.slate_games ?? 0)}</p>
            </div>
            <div className="metric">
              <p className="label">Coverage %</p>
              <p className={`value ${statusClass(coveragePct, "coverage")}`}>{coveragePct.toFixed(2)}%</p>
            </div>
            <div className="metric">
              <p className="label">Slate Label</p>
              <p className="value">{slateStatus?.slate_label ?? "Main"}</p>
            </div>
          </div>
        </article>

        <article className="panel">
          <h2>Supplemental Mapping</h2>
          <p className="meta">Optional enrichment quality. These numbers should improve over time, but they do not block a DK-slate-backed run.</p>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Mapped</p>
              <p className="value">{resolvedCount}</p>
            </div>
            <div className="metric">
              <p className="label">Open Issues</p>
              <p className={`value ${statusClass(mappingOpenIssues, "count")}`}>{mappingOpenIssues}</p>
            </div>
            <div className="metric">
              <p className="label">Coverage %</p>
              <p className={`value ${statusClass(coveragePct, "coverage")}`}>{coveragePct.toFixed(2)}%</p>
            </div>
            <div className="metric">
              <p className="label">Supplemental Players</p>
              <p className="value">{totalCount}</p>
            </div>
            <div className="metric">
              <p className="label">Unresolved</p>
              <p className={`value ${statusClass(unresolvedCount, "count")}`}>{unresolvedCount}</p>
            </div>
            <div className="metric">
              <p className="label">Conflicts</p>
              <p className={`value ${statusClass(conflictCount, "count")}`}>{conflictCount}</p>
            </div>
          </div>
        </article>
      </section>

      <section className="content-grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Current Notes</h2>
          {notes.length === 0 ? (
            <p className="meta">The DK slate is loaded and no operational issues are blocking the workflow.</p>
          ) : (
            <ul className="list">
              {notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </article>

        <article className="panel">
          <h2>Workflow Path</h2>
          <p className="meta">The web flow stays focused on the required lineup steps and keeps supplemental data out of the main path.</p>
          <ol className="list">
            <li>Game Data</li>
            <li>Prop Data</li>
            <li>Backfill</li>
            <li>DK Slate</li>
            <li>Injuries</li>
            <li>Slate + Vegas</li>
            <li>Lineup Generator</li>
            <li>Saved Runs</li>
            <li>Projection Review</li>
            <li>Tournament Review</li>
          </ol>
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Supplemental Mapping Exceptions</h2>
        {!slateStatus ? (
          <p className="meta">Slate status data is unavailable.</p>
        ) : unresolvedRows.length === 0 ? (
          <p className="meta">No supplemental mapping exceptions for this date.</p>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Status</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {unresolvedRows.map((row, idx) => (
                  <tr key={`${row.player_name ?? "player"}-${row.team_abbr ?? "team"}-${idx}`}>
                    <td>{row.player_name ?? "-"}</td>
                    <td>{row.team_abbr ?? "-"}</td>
                    <td>{row.dk_resolution_status ?? "-"}</td>
                    <td>{row.dk_match_reason ?? "-"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
