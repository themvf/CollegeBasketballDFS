import { fetchHealth, fetchRegistryCoverage } from "../lib/api";

type HomeProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

function getTodayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

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

export default async function Home({ searchParams }: HomeProps) {
  const params = await searchParams;
  const selectedDate = typeof params.date === "string" ? params.date : getTodayIsoDate();

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
        <h1>CollegeBasketballDFS Migration Console</h1>
        <p>
          This is the Vercel migration starter. It surfaces the high-friction part of the current workflow first: DK
          identity coverage for a selected slate date.
        </p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">API: {healthOk ? "Connected" : "Unavailable"}</span>
          <span className="badge">Stack: Next.js + FastAPI</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Registry Coverage</h2>
          <p className="meta">Source: RotoWire Classic All slate resolved against local DK registry + manual overrides.</p>
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
          <h2>Next Cutover Steps</h2>
          <p className="meta">This repo now has an API shell (`apps/api`) and web shell (`apps/web`).</p>
          <ol className="list">
            <li>Replace Streamlit tab reads with REST endpoints by domain slice.</li>
            <li>Move lineup generation into asynchronous job execution with persisted run state.</li>
            <li>Add auth and per-user audit logs before production rollout.</li>
            <li>Retire Streamlit once parity checks pass for three consecutive slates.</li>
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

      <p className="footer">
        To change date use <code>?date=YYYY-MM-DD</code>. Example: <code>?date=2026-03-04</code>
      </p>
    </main>
  );
}
