import { fetchVegasGameLines, fetchVegasMarketContext } from "../lib/api";

type SlateVegasViewProps = {
  selectedDate: string;
};

function intValue(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0";
  }
  return String(Math.round(value));
}

function numValue(value: number | undefined | null, digits = 2): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

export default async function SlateVegasView({ selectedDate }: SlateVegasViewProps) {
  const [linesPayload, marketPayload] = await Promise.all([
    fetchVegasGameLines(selectedDate),
    fetchVegasMarketContext(selectedDate),
  ]);
  const lines = linesPayload?.summary ?? {};
  const market = marketPayload?.summary ?? {};
  const rankedGames = marketPayload?.ranked_games ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>Slate + Vegas</h1>
        <p>Pre-lock game environment diagnostics from totals, spreads, and moneylines.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Games: {intValue(lines.total_games)}</span>
          <span className="badge">Matched: {intValue(lines.odds_matched_games)}</span>
        </div>
      </section>

      {!linesPayload || !marketPayload ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>API Unavailable</h2>
          <p className="meta">Could not load Slate + Vegas data from the backend endpoints.</p>
        </section>
      ) : (
        <>
          <section className="content-grid">
            <article className="panel">
              <h2>Accuracy Snapshot</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Total MAE</p>
                  <p className="value">{numValue(lines.total_mae)}</p>
                </div>
                <div className="metric">
                  <p className="label">Spread MAE</p>
                  <p className="value">{numValue(lines.spread_mae)}</p>
                </div>
                <div className="metric">
                  <p className="label">Winner Pick Accuracy %</p>
                  <p className="value">{numValue(lines.winner_pick_accuracy_pct, 1)}</p>
                </div>
                <div className="metric">
                  <p className="label">Market Spread MAE</p>
                  <p className="value">{numValue(market.spread_mae)}</p>
                </div>
              </div>
            </article>
            <article className="panel">
              <h2>Navigation</h2>
              <p className="meta">Use detailed Vegas diagnostics from the dedicated Vegas Review section.</p>
              <ul className="list">
                <li>Vegas Review / Game Lines for line-by-line outcomes</li>
                <li>Vegas Review / Market Context for calibration models and buckets</li>
                <li>Vegas Review / Prop Data for player-prop coverage health</li>
              </ul>
            </article>
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <h2>Top Ranked Games</h2>
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Game</th>
                    <th>Total</th>
                    <th>Spread (Home)</th>
                    <th>ML Home</th>
                    <th>ML Away</th>
                    <th>Books</th>
                  </tr>
                </thead>
                <tbody>
                  {rankedGames.slice(0, 20).map((row, idx) => (
                    <tr key={`${row.game_date}-${row.away_team}-${row.home_team}-${idx}`}>
                      <td>{`${row.away_team ?? "-"} @ ${row.home_team ?? "-"}`}</td>
                      <td>{numValue(row.total_points, 1)}</td>
                      <td>{numValue(row.spread_home, 1)}</td>
                      <td>{numValue(row.moneyline_home, 0)}</td>
                      <td>{numValue(row.moneyline_away, 0)}</td>
                      <td>{intValue(row.bookmakers_count)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </main>
  );
}
