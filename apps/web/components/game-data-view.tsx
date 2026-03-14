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
  const blockers = [
    slateStatus ? "" : "Slate status endpoint is unavailable.",
    slateStatus?.active_source?.detail,
    coverage.error,
    slateStatus?.rotowire?.error,
  ].filter((value, index, arr): value is string => Boolean(value) && arr.indexOf(value) === index);
  const unresolvedRows = slateStatus?.registry_coverage?.unresolved_sample ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>Game Data</h1>
        <p>Validate the real active slate source, registry coverage, and readiness blockers before the lineup workflow moves on.</p>
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
          <h2>Registry Coverage</h2>
          <p className="meta">Resolve these issues before trusting any downstream player-pool or lineup output.</p>
          <div className="metric-grid">
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
              <p className="label">Players</p>
              <p className="value">{totalCount}</p>
            </div>
            <div className="metric">
              <p className="label">RotoWire Rows</p>
              <p className="value">{Number(slateStatus?.rotowire?.rows ?? 0)}</p>
            </div>
            <div className="metric">
              <p className="label">Source Cached</p>
              <p className={`value ${Boolean(slateStatus?.cached_dk_slate?.available) ? "ok" : "bad"}`}>
                {Boolean(slateStatus?.cached_dk_slate?.available) ? "Yes" : "No"}
              </p>
            </div>
          </div>
        </article>
      </section>

      <section className="content-grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Current Blockers</h2>
          {blockers.length === 0 ? (
            <p className="meta">No current blockers. The slate looks ready for the rest of the workflow.</p>
          ) : (
            <ul className="list">
              {blockers.map((blocker) => (
                <li key={blocker}>{blocker}</li>
              ))}
            </ul>
          )}
        </article>

        <article className="panel">
          <h2>Workflow Path</h2>
          <p className="meta">Generate Lineup flow now tracks the state explicitly instead of relying on hidden page state.</p>
          <ol className="list">
            <li>Game Data</li>
            <li>Prop Data</li>
            <li>Backfill</li>
            <li>DK Slate</li>
            <li>Injuries</li>
            <li>Slate + Vegas</li>
            <li>RotoWire Scraper</li>
            <li>Lineup Generator</li>
            <li>Saved Runs</li>
            <li>Projection Review</li>
            <li>Tournament Review</li>
          </ol>
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Current Unresolved Sample</h2>
        {!slateStatus ? (
          <p className="meta">Slate status data is unavailable.</p>
        ) : unresolvedRows.length === 0 ? (
          <p className="meta">No unresolved players for this date.</p>
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
