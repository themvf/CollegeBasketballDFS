import { notFound } from "next/navigation";
import {
  fetchVegasGameLines,
  fetchVegasMarketContext,
  fetchVegasPropData,
  type VegasGameLinesResponse,
  type VegasMarketContextResponse,
  type VegasPropDataResponse,
} from "../../../lib/api";

type VegasReviewPageProps = {
  params: Promise<{ view: string }>;
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

function getTodayIsoDate(): string {
  return new Date().toISOString().slice(0, 10);
}

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

function booleanText(value: boolean | undefined): string {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return "-";
}

function renderGameLinesView(selectedDate: string, payload: VegasGameLinesResponse | null) {
  const summary = payload?.summary ?? {};
  const rows = payload?.rows ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>Vegas Review - Game Lines</h1>
        <p>Totals, spreads, and moneyline alignment for the selected slate date.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Bucket: {payload?.bucket_name ?? "Missing"}</span>
          <span className="badge">Rows: {intValue(rows.length)}</span>
        </div>
      </section>

      {!payload ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>API Unavailable</h2>
          <p className="meta">
            Could not load Vegas game lines from the API. Check `NEXT_PUBLIC_API_BASE_URL`, `CBB_GCS_BUCKET`, and
            backend logs.
          </p>
        </section>
      ) : (
        <>
          <section className="content-grid">
            <article className="panel">
              <h2>Line Accuracy Summary</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Total Games</p>
                  <p className="value">{intValue(summary.total_games)}</p>
                </div>
                <div className="metric">
                  <p className="label">Matched Games</p>
                  <p className="value">{intValue(summary.odds_matched_games)}</p>
                </div>
                <div className="metric">
                  <p className="label">Total MAE</p>
                  <p className="value">{numberValue(summary.total_mae)}</p>
                </div>
                <div className="metric">
                  <p className="label">Spread MAE</p>
                  <p className="value">{numberValue(summary.spread_mae)}</p>
                </div>
              </div>
            </article>

            <article className="panel">
              <h2>Source Health</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Raw Cached</p>
                  <p className="value">{booleanText(payload.source_status?.raw_cached)}</p>
                </div>
                <div className="metric">
                  <p className="label">Odds Cached</p>
                  <p className="value">{booleanText(payload.source_status?.odds_cached)}</p>
                </div>
                <div className="metric">
                  <p className="label">Winner Pick Accuracy %</p>
                  <p className="value">{numberValue(summary.winner_pick_accuracy_pct, 1)}</p>
                </div>
                <div className="metric">
                  <p className="label">Spread-Line Games</p>
                  <p className="value">{intValue(summary.spread_line_games)}</p>
                </div>
              </div>
            </article>
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <h2>Game Lines Table</h2>
            {rows.length === 0 ? (
              <p className="meta">No Vegas rows found for this date.</p>
            ) : (
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Game</th>
                      <th>Total</th>
                      <th>Actual Total</th>
                      <th>Total Error</th>
                      <th>Spread (Home)</th>
                      <th>Actual Margin</th>
                      <th>Spread Error</th>
                      <th>ML Home</th>
                      <th>ML Away</th>
                      <th>Winner Correct</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, idx) => (
                      <tr key={`${row.game_date}-${row.away_team}-${row.home_team}-${idx}`}>
                        <td>{`${row.away_team ?? "-"} @ ${row.home_team ?? "-"}`}</td>
                        <td>{numberValue(row.total_points, 1)}</td>
                        <td>{numberValue(row.actual_total, 1)}</td>
                        <td>{numberValue(row.total_error, 1)}</td>
                        <td>{numberValue(row.spread_home, 1)}</td>
                        <td>{numberValue(row.actual_home_margin, 1)}</td>
                        <td>{numberValue(row.spread_error, 1)}</td>
                        <td>{numberValue(row.moneyline_home, 0)}</td>
                        <td>{numberValue(row.moneyline_away, 0)}</td>
                        <td>{booleanText(row.winner_pick_correct)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </main>
  );
}

function renderMarketContextView(selectedDate: string, payload: VegasMarketContextResponse | null) {
  const summary = payload?.summary ?? {};
  const calibration = payload?.calibration_models ?? [];
  const totalBuckets = payload?.total_buckets ?? [];
  const spreadBuckets = payload?.spread_buckets ?? [];
  const rankedGames = payload?.ranked_games ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>Vegas Review - Market Context</h1>
        <p>Game environment ranking and calibration diagnostics from totals/spreads.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Games: {intValue(summary.total_games)}</span>
          <span className="badge">Ranked Rows: {intValue(rankedGames.length)}</span>
        </div>
      </section>

      {!payload ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>API Unavailable</h2>
          <p className="meta">Could not load market context data from the API.</p>
        </section>
      ) : (
        <>
          <section className="content-grid">
            <article className="panel">
              <h2>Calibration Summary</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Total MAE</p>
                  <p className="value">{numberValue(summary.total_mae)}</p>
                </div>
                <div className="metric">
                  <p className="label">Spread MAE</p>
                  <p className="value">{numberValue(summary.spread_mae)}</p>
                </div>
                <div className="metric">
                  <p className="label">Raw Cached</p>
                  <p className="value">{booleanText(payload.source_status?.raw_cached)}</p>
                </div>
                <div className="metric">
                  <p className="label">Odds Cached</p>
                  <p className="value">{booleanText(payload.source_status?.odds_cached)}</p>
                </div>
              </div>
            </article>
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <h2>Calibration Models</h2>
            {calibration.length === 0 ? (
              <p className="meta">No calibration rows available.</p>
            ) : (
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th>Samples</th>
                      <th>Slope</th>
                      <th>Intercept</th>
                      <th>R2</th>
                      <th>Baseline MAE</th>
                      <th>Calibrated MAE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {calibration.map((row, idx) => (
                      <tr key={`${row.model}-${idx}`}>
                        <td>{row.model ?? "-"}</td>
                        <td>{intValue(row.samples)}</td>
                        <td>{numberValue(row.slope, 3)}</td>
                        <td>{numberValue(row.intercept, 3)}</td>
                        <td>{numberValue(row.r2, 3)}</td>
                        <td>{numberValue(row.baseline_mae, 2)}</td>
                        <td>{numberValue(row.calibrated_mae, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="content-grid" style={{ marginTop: 16 }}>
            <article className="panel">
              <h2>Total Buckets</h2>
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Bucket</th>
                      <th>Games</th>
                      <th>Avg Vegas</th>
                      <th>Avg Actual</th>
                      <th>MAE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {totalBuckets.map((row, idx) => (
                      <tr key={`${row.vegas_total_bucket}-${idx}`}>
                        <td>{row.vegas_total_bucket ?? "-"}</td>
                        <td>{intValue(row.games)}</td>
                        <td>{numberValue(row.avg_vegas_total, 1)}</td>
                        <td>{numberValue(row.avg_actual_total, 1)}</td>
                        <td>{numberValue(row.mae_total, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>

            <article className="panel">
              <h2>Spread Buckets</h2>
              <div className="table-shell">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Bucket</th>
                      <th>Games</th>
                      <th>Avg Vegas Abs</th>
                      <th>Avg Actual Abs</th>
                      <th>MAE</th>
                    </tr>
                  </thead>
                  <tbody>
                    {spreadBuckets.map((row, idx) => (
                      <tr key={`${row.vegas_spread_bucket}-${idx}`}>
                        <td>{row.vegas_spread_bucket ?? "-"}</td>
                        <td>{intValue(row.games)}</td>
                        <td>{numberValue(row.avg_abs_vegas_margin, 2)}</td>
                        <td>{numberValue(row.avg_abs_actual_margin, 2)}</td>
                        <td>{numberValue(row.mae_spread, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <h2>Ranked Games</h2>
            {rankedGames.length === 0 ? (
              <p className="meta">No ranked game rows for this date.</p>
            ) : (
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
                    {rankedGames.map((row, idx) => (
                      <tr key={`${row.game_date}-${row.away_team}-${row.home_team}-${idx}`}>
                        <td>{`${row.away_team ?? "-"} @ ${row.home_team ?? "-"}`}</td>
                        <td>{numberValue(row.total_points, 1)}</td>
                        <td>{numberValue(row.spread_home, 1)}</td>
                        <td>{numberValue(row.moneyline_home, 0)}</td>
                        <td>{numberValue(row.moneyline_away, 0)}</td>
                        <td>{intValue(row.bookmakers_count)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      )}
    </main>
  );
}

function renderPropDataView(selectedDate: string, payload: VegasPropDataResponse | null) {
  const summary = payload?.summary ?? {};
  const marketCoverage = payload?.market_coverage ?? [];
  const rows = payload?.rows ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>Vegas Review - Prop Data</h1>
        <p>Player-prop coverage snapshot and market completeness check for the active date.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Rows: {intValue(summary.rows)}</span>
          <span className="badge">Markets: {intValue(summary.markets)}</span>
        </div>
      </section>

      {!payload ? (
        <section className="panel" style={{ marginTop: 16 }}>
          <h2>API Unavailable</h2>
          <p className="meta">Could not load prop-data endpoint results.</p>
        </section>
      ) : (
        <>
          <section className="content-grid">
            <article className="panel">
              <h2>Props Summary</h2>
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

            <article className="panel">
              <h2>Source Health</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Props Cached</p>
                  <p className="value">{booleanText(payload.source_status?.props_cached)}</p>
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
              <h2>Sample Props</h2>
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

export default async function VegasReviewPage({ params, searchParams }: VegasReviewPageProps) {
  const resolvedParams = await params;
  const resolvedSearch = await searchParams;
  const view = String(resolvedParams.view || "").trim().toLowerCase();
  const selectedDate = typeof resolvedSearch.date === "string" ? resolvedSearch.date : getTodayIsoDate();

  if (view === "game-lines") {
    const payload = await fetchVegasGameLines(selectedDate);
    return renderGameLinesView(selectedDate, payload);
  }
  if (view === "market-context") {
    const payload = await fetchVegasMarketContext(selectedDate);
    return renderMarketContextView(selectedDate, payload);
  }
  if (view === "prop-data") {
    const payload = await fetchVegasPropData(selectedDate);
    return renderPropDataView(selectedDate, payload);
  }

  notFound();
}
