"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchProjectionReview, type ProjectionReviewResponse } from "../lib/api";

type PlayerReviewViewProps = {
  selectedDate: string;
  slateKey?: string;
};

type Row = Record<string, unknown>;

function rows(input: unknown): Row[] {
  if (!Array.isArray(input)) return [];
  return input as Row[];
}

function toNumber(value: unknown): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : NaN;
}

function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function projectedPoints(row: Row): number {
  const blend = toNumber(row.blended_projection);
  if (Number.isFinite(blend)) return blend;
  return toNumber(row.our_dk_projection);
}

export default function PlayerReviewView({ selectedDate, slateKey = "main" }: PlayerReviewViewProps) {
  const [payload, setPayload] = useState<ProjectionReviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  const [minProjPoints, setMinProjPoints] = useState<number>(15);
  const [maxProjOwnership, setMaxProjOwnership] = useState<number>(10);
  const [topRows, setTopRows] = useState<number>(120);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const next = await fetchProjectionReview(selectedDate, slateKey);
      if (!next) {
        setError("Could not load player review rows.");
      }
      setPayload(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPayload(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, [selectedDate, slateKey]);

  const allRows = useMemo(() => rows(payload?.rows), [payload]);

  const filtered = useMemo(() => {
    return allRows
      .filter((row) => {
        const pts = projectedPoints(row);
        const own = toNumber(row.projected_ownership);
        const ptsOk = Number.isFinite(pts) && pts >= minProjPoints;
        const ownOk = Number.isFinite(own) && own <= maxProjOwnership;
        return ptsOk && ownOk;
      })
      .sort((a, b) => projectedPoints(b) - projectedPoints(a))
      .slice(0, Math.max(1, topRows));
  }, [allRows, minProjPoints, maxProjOwnership, topRows]);

  return (
    <main className="page">
      <section className="hero">
        <h1>Player Review</h1>
        <p>Surface high projected fantasy point plays with low projected ownership for tournament leverage.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Default Min Projection: 15</span>
          <span className="badge">Target Ownership: &lt;= 10%</span>
        </div>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Filters</h2>
        <div className="form-grid">
          <label className="field">
            <span>Min Projected Points</span>
            <input
              type="number"
              min={0}
              max={100}
              step={0.5}
              value={minProjPoints}
              onChange={(event) => setMinProjPoints(Number(event.target.value || 0))}
            />
          </label>
          <label className="field">
            <span>Max Projected Own %</span>
            <input
              type="number"
              min={0}
              max={100}
              step={0.5}
              value={maxProjOwnership}
              onChange={(event) => setMaxProjOwnership(Number(event.target.value || 0))}
            />
          </label>
          <label className="field">
            <span>Top Rows</span>
            <input
              type="number"
              min={10}
              max={500}
              step={10}
              value={topRows}
              onChange={(event) => setTopRows(Number(event.target.value || 10))}
            />
          </label>
        </div>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Low-Owned Projection Candidates</h2>
        <p className="meta">Filtered rows: {filtered.length} | Source rows: {allRows.length}</p>
        {loading ? (
          <p className="meta">Loading...</p>
        ) : error ? (
          <p className="error-text">{error}</p>
        ) : filtered.length === 0 ? (
          <p className="meta">No players meet current thresholds.</p>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Pos</th>
                  <th>Salary</th>
                  <th>Projected Points</th>
                  <th>Projected Own%</th>
                  <th>Actual Points</th>
                  <th>Actual Own%</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((row, idx) => {
                  const pts = projectedPoints(row);
                  const own = toNumber(row.projected_ownership);
                  const actualPts = toNumber(row.actual_dk_points);
                  const actualOwn = toNumber(row.actual_ownership);
                  return (
                    <tr key={`${String(row.Name ?? row["Name + ID"] ?? "player")}-${idx}`}>
                      <td>{String(row.Name ?? row["Name + ID"] ?? "")}</td>
                      <td>{String(row.TeamAbbrev ?? "")}</td>
                      <td>{String(row.Position ?? "")}</td>
                      <td>{formatNumber(toNumber(row.Salary), 0)}</td>
                      <td>{formatNumber(pts)}</td>
                      <td>{formatNumber(own)}</td>
                      <td>{formatNumber(actualPts)}</td>
                      <td>{formatNumber(actualOwn)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </main>
  );
}
