import { fetchCacheCoverage, type CacheCoverageStats } from "../lib/api";

type BackfillViewProps = {
  endDate: string;
};

function addDays(isoDate: string, days: number): string {
  const d = new Date(`${isoDate}T00:00:00Z`);
  if (Number.isNaN(d.getTime())) {
    return isoDate;
  }
  d.setUTCDate(d.getUTCDate() + days);
  return d.toISOString().slice(0, 10);
}

function intValue(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0";
  }
  return String(Math.round(value));
}

function pctValue(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0.00%";
  }
  return `${value.toFixed(2)}%`;
}

function CoverageRow({ label, stats }: { label: string; stats: CacheCoverageStats | undefined }) {
  return (
    <tr>
      <td>{label}</td>
      <td>{intValue(stats?.found_dates)}</td>
      <td>{intValue(stats?.total_dates)}</td>
      <td>{pctValue(stats?.coverage_pct)}</td>
      <td>{intValue(stats?.missing_dates?.length ?? 0)}</td>
    </tr>
  );
}

export default async function BackfillView({ endDate }: BackfillViewProps) {
  const startDate = addDays(endDate, -117);
  const payload = await fetchCacheCoverage(startDate, endDate);
  const coverage = payload?.coverage ?? {};

  return (
    <main className="page">
      <section className="hero">
        <h1>Backfill</h1>
        <p>Cache coverage diagnostics across recent slate dates.</p>
        <div className="badge-row">
          <span className="badge">Start: {startDate}</span>
          <span className="badge">End: {endDate}</span>
          <span className="badge">Dates: {intValue(payload?.total_dates)}</span>
        </div>
      </section>

      {!payload ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>API Unavailable</h2>
          <p className="meta">
            Could not load cache coverage endpoint. Check API deployment and `CBB_GCS_BUCKET`.
          </p>
        </section>
      ) : (
        <>
          <section className="panel" style={{ marginTop: 16 }}>
            <h2>Coverage Overview</h2>
            <p className="meta">Bucket: {payload.bucket_name ?? "Not set"}</p>
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Dataset</th>
                    <th>Found Dates</th>
                    <th>Total Dates</th>
                    <th>Coverage</th>
                    <th>Missing Dates</th>
                  </tr>
                </thead>
                <tbody>
                  <CoverageRow label="Raw Game Data" stats={coverage.raw_game_data} />
                  <CoverageRow label="Odds Data" stats={coverage.odds_data} />
                  <CoverageRow label="Props Data" stats={coverage.props_data} />
                  <CoverageRow label="Players Results" stats={coverage.players_results} />
                  <CoverageRow label="DK Slates" stats={coverage.dk_slates} />
                  <CoverageRow label="Projection Snapshots" stats={coverage.projections_snapshots} />
                  <CoverageRow label="Ownership Files" stats={coverage.ownership_files} />
                  <CoverageRow label="Injuries Feed" stats={coverage.injuries_feed} />
                </tbody>
              </table>
            </div>
          </section>

          <section className="content-grid" style={{ marginTop: 16 }}>
            <article className="panel">
              <h2>Missing Raw Dates</h2>
              {(coverage.raw_game_data?.missing_dates ?? []).length === 0 ? (
                <p className="meta">No missing raw dates in this window.</p>
              ) : (
                <ul className="list">
                  {(coverage.raw_game_data?.missing_dates ?? []).slice(0, 25).map((d) => (
                    <li key={d}>{d}</li>
                  ))}
                </ul>
              )}
            </article>
            <article className="panel">
              <h2>Missing DK Slate Dates</h2>
              {(coverage.dk_slates?.missing_dates ?? []).length === 0 ? (
                <p className="meta">No missing DK slate dates in this window.</p>
              ) : (
                <ul className="list">
                  {(coverage.dk_slates?.missing_dates ?? []).slice(0, 25).map((d) => (
                    <li key={d}>{d}</li>
                  ))}
                </ul>
              )}
            </article>
          </section>
        </>
      )}
    </main>
  );
}
