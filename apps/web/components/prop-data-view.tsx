import { fetchVegasPropData } from "../lib/api";

type PropDataViewProps = {
  selectedDate: string;
};

function numberValue(value: number | undefined | null, digits = 2): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function intValue(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0";
  }
  return String(Math.round(value));
}

export default async function PropDataView({ selectedDate }: PropDataViewProps) {
  const payload = await fetchVegasPropData(selectedDate);
  const summary = payload?.summary ?? {};
  const marketCoverage = payload?.market_coverage ?? [];
  const rows = payload?.rows ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>Prop Data</h1>
        <p>Player-prop coverage and market health for the selected slate date.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Rows: {intValue(summary.rows)}</span>
          <span className="badge">Markets: {intValue(summary.markets)}</span>
        </div>
      </section>

      {!payload ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>API Unavailable</h2>
          <p className="meta">
            Could not load prop data from API. Check backend deployment and `NEXT_PUBLIC_API_BASE_URL`.
          </p>
        </section>
      ) : (
        <>
          <section className="content-grid">
            <article className="panel">
              <h2>Summary</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Rows</p>
                  <p className="value">{intValue(summary.rows)}</p>
                </div>
                <div className="metric">
                  <p className="label">Players</p>
                  <p className="value">{intValue(summary.players)}</p>
                </div>
                <div className="metric">
                  <p className="label">Events</p>
                  <p className="value">{intValue(summary.events)}</p>
                </div>
                <div className="metric">
                  <p className="label">Bookmakers</p>
                  <p className="value">{intValue(summary.bookmakers)}</p>
                </div>
              </div>
            </article>
          </section>

          <section className="content-grid" style={{ marginTop: 16 }}>
            <article className="panel">
              <h2>Market Coverage</h2>
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Market</th>
                      <th>Rows</th>
                      <th>Players</th>
                      <th>Events</th>
                      <th>Avg Line</th>
                    </tr>
                  </thead>
                  <tbody>
                    {marketCoverage.map((row, idx) => (
                      <tr key={`${row.market}-${idx}`}>
                        <td>{row.market ?? "-"}</td>
                        <td>{intValue(row.rows)}</td>
                        <td>{intValue(row.players)}</td>
                        <td>{intValue(row.events)}</td>
                        <td>{numberValue(row.avg_line, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>

            <article className="panel">
              <h2>Sample Lines</h2>
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Player</th>
                      <th>Game</th>
                      <th>Market</th>
                      <th>Book</th>
                      <th>Line</th>
                      <th>Over</th>
                      <th>Under</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, idx) => (
                      <tr key={`${row.player_name}-${row.market}-${idx}`}>
                        <td>{row.player_name ?? "-"}</td>
                        <td>{`${row.away_team ?? "-"} @ ${row.home_team ?? "-"}`}</td>
                        <td>{row.market ?? "-"}</td>
                        <td>{row.bookmaker ?? "-"}</td>
                        <td>{numberValue(row.line, 2)}</td>
                        <td>{numberValue(row.over_price, 0)}</td>
                        <td>{numberValue(row.under_price, 0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </section>
        </>
      )}
    </main>
  );
}
