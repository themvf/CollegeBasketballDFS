import { fetchLineupRunDetail, fetchLineupRuns, type LineupRunDetailResponse } from "../lib/api";

type SavedLineupRunsViewProps = {
  selectedDate: string;
  slateKey?: string;
  selectedRunId?: string;
};

function formatDateTime(value: string | undefined): string {
  if (!value) {
    return "-";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatNumber(value: unknown, digits = 0): string {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return digits > 0 ? (0).toFixed(digits) : "0";
  }
  return digits > 0 ? numeric.toFixed(digits) : String(Math.round(numeric));
}

function formatLineupPlayers(lineup: Record<string, unknown>): string {
  const players = lineup.players;
  if (Array.isArray(players) && players.length > 0) {
    return players
      .map((player) => {
        if (!player || typeof player !== "object") {
          return "";
        }
        const row = player as Record<string, unknown>;
        return String(row["Name + ID"] || row.Name || row.player_name || row.name || row.ID || "").trim();
      })
      .filter(Boolean)
      .join(" | ");
  }
  return String(lineup.Players || lineup.players_label || "").trim() || "-";
}

function buildRunHref(selectedDate: string, slateKey: string, runId: string): string {
  const params = new URLSearchParams({
    date: selectedDate,
    slate_key: slateKey,
    run_id: runId,
  });
  return `/generate-lineup/saved-runs?${params.toString()}`;
}

function renderVersionTable(detail: LineupRunDetailResponse | null) {
  const versions = detail?.versions ?? [];
  if (versions.length === 0) {
    return <p className="meta">No version payloads were available for this run.</p>;
  }

  return (
    <div className="version-stack">
      {versions.map((version) => {
        const lineups = Array.isArray(version.lineups) ? version.lineups.slice(0, 8) : [];
        return (
          <article className="panel version-panel" key={version.version_key || version.version_label || "version"}>
            <div className="run-card-head">
              <div>
                <p className="eyebrow">{version.lineup_strategy || "strategy"}</p>
                <h3>{version.version_label || version.version_key || "Version"}</h3>
              </div>
              <div className="chip-row">
                <span className="chip">Model: {version.model_profile || "-"}</span>
                <span className="chip">Lineups: {formatNumber(version.lineup_count_generated)}</span>
                <span className="chip">Warnings: {formatNumber(version.warning_count)}</span>
              </div>
            </div>

            {Array.isArray(version.warnings) && version.warnings.length > 0 ? (
              <div className="callout warn" style={{ marginTop: 12 }}>
                <strong>Warnings</strong>
                <ul className="list compact-list">
                  {version.warnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            {lineups.length === 0 ? (
              <p className="meta" style={{ marginTop: 12 }}>
                No lineups loaded for this version.
              </p>
            ) : (
              <div className="table-shell" style={{ marginTop: 12 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Proj</th>
                      <th>Salary</th>
                      <th>Own Sum</th>
                      <th>Players</th>
                    </tr>
                  </thead>
                  <tbody>
                    {lineups.map((lineup, idx) => {
                      const row = lineup as Record<string, unknown>;
                      return (
                        <tr key={`${version.version_key || "v"}-${row.lineup_number ?? idx}`}>
                          <td>{formatNumber(row.lineup_number ?? row.Lineup ?? idx + 1)}</td>
                          <td>{formatNumber(row.projected_points ?? row["Projected Points"], 2)}</td>
                          <td>{formatNumber(row.salary ?? row.Salary)}</td>
                          <td>{formatNumber(row.projected_ownership_sum ?? row["Projected Ownership Sum"], 2)}</td>
                          <td>{formatLineupPlayers(row)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </article>
        );
      })}
    </div>
  );
}

export default async function SavedLineupRunsView({
  selectedDate,
  slateKey = "main",
  selectedRunId,
}: SavedLineupRunsViewProps) {
  const runsPayload = await fetchLineupRuns(selectedDate, slateKey, true);
  const runs = runsPayload?.runs ?? [];
  const activeRunId = selectedRunId || runs[0]?.run_id || "";
  const detail = activeRunId ? await fetchLineupRunDetail(selectedDate, activeRunId, slateKey) : null;
  const activeRun = detail?.run;
  const activeSettings = (activeRun?.settings ?? {}) as Record<string, unknown>;

  return (
    <main className="page">
      <section className="hero">
        <h1>Saved Runs</h1>
        <p>Browse persisted lineup runs by slate date, inspect model versions, and review the top generated lineups without going back to Streamlit.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Slate: {slateKey}</span>
          <span className="badge">Runs: {formatNumber(runs.length)}</span>
          <span className="badge">Selected: {activeRunId || "None"}</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Run Library</h2>
          <p className="meta">The list merges local backup manifests and GCS-backed saved runs for the selected date.</p>
          {runs.length === 0 ? (
            <div className="empty-state">
              <p>No saved lineup runs were found for this date.</p>
              <span>Generate a lineup run first, then this page becomes the default review surface.</span>
            </div>
          ) : (
            <div className="run-list">
              {runs.map((run) => {
                const runId = String(run.run_id || "");
                const active = runId === activeRunId;
                const href = buildRunHref(selectedDate, slateKey, runId);
                return (
                  <a className={`run-card ${active ? "active" : ""}`} href={href} key={runId}>
                    <div className="run-card-head">
                      <div>
                        <p className="eyebrow">{run.run_mode || "single"} run</p>
                        <h3>{runId}</h3>
                      </div>
                      <span className={`chip ${active ? "chip-active" : ""}`}>{run.storage_source || "unknown"}</span>
                    </div>
                    <div className="summary-inline">
                      <span>{formatDateTime(run.generated_at_utc)}</span>
                      <span>{formatNumber(run.lineups_generated)} lineups</span>
                      <span>{formatNumber(run.version_count)} versions</span>
                    </div>
                    {Array.isArray(run.versions) && run.versions.length > 0 ? (
                      <div className="chip-row" style={{ marginTop: 12 }}>
                        {run.versions.slice(0, 4).map((version) => (
                          <span className="chip" key={`${runId}-${version.version_key}`}>
                            {version.version_label || version.version_key}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </a>
                );
              })}
            </div>
          )}
        </article>

        <article className="panel">
          <h2>Run Summary</h2>
          {!activeRun ? (
            <div className="empty-state">
              <p>No run selected.</p>
              <span>Pick a saved run on the left to load version details and lineup output.</span>
            </div>
          ) : (
            <>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Generated</p>
                  <p className="value">{formatDateTime(activeRun.generated_at_utc)}</p>
                </div>
                <div className="metric">
                  <p className="label">Mode</p>
                  <p className="value">{activeRun.run_mode || "-"}</p>
                </div>
                <div className="metric">
                  <p className="label">Versions</p>
                  <p className="value">{formatNumber(activeRun.version_count)}</p>
                </div>
                <div className="metric">
                  <p className="label">Lineups</p>
                  <p className="value">{formatNumber(activeRun.lineups_generated)}</p>
                </div>
                <div className="metric">
                  <p className="label">Warnings</p>
                  <p className="value">{formatNumber(activeRun.warnings_count)}</p>
                </div>
                <div className="metric">
                  <p className="label">Source</p>
                  <p className="value">{activeRun.storage_source || "-"}</p>
                </div>
              </div>

              <div className="chip-row" style={{ marginTop: 14 }}>
                <span className="chip">Contest: {String(activeSettings.contest_type || "-")}</span>
                <span className="chip">Model Seed: {formatNumber(activeSettings.random_seed)}</span>
                <span className="chip">Requested Lineups: {formatNumber(activeSettings.num_lineups)}</span>
                <span className="chip">Max Salary Left: {formatNumber(activeSettings.max_salary_left)}</span>
              </div>

              <p className="meta" style={{ marginTop: 12 }}>
                Manifest: {activeRun.manifest_ref || "-"}
              </p>
            </>
          )}
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Version Detail</h2>
        <p className="meta">Showing the first eight lineups per version to keep the review surface readable.</p>
        {renderVersionTable(detail)}
      </section>
    </main>
  );
}
