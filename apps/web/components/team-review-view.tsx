"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchProjectionReview, type ProjectionReviewResponse } from "../lib/api";

type TeamReviewViewProps = {
  selectedDate: string;
  slateKey?: string;
};

type Row = Record<string, unknown>;

type TeamRow = {
  team: string;
  players: number;
  avgProj: number;
  avgOwn: number;
  avgActual: number;
  highProjCount: number;
};

function asRows(input: unknown): Row[] {
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

export default function TeamReviewView({ selectedDate, slateKey = "main" }: TeamReviewViewProps) {
  const [payload, setPayload] = useState<ProjectionReviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");

  const [minTeamPlayers, setMinTeamPlayers] = useState<number>(3);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const next = await fetchProjectionReview(selectedDate, slateKey);
      if (!next) {
        setError("Could not load team review rows.");
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

  const teamRows = useMemo(() => {
    const grouped = new Map<string, { players: number; projSum: number; ownSum: number; actualSum: number; highProjCount: number }>();
    for (const row of asRows(payload?.rows)) {
      const team = String(row.TeamAbbrev ?? "").trim().toUpperCase();
      if (!team) continue;
      const proj = projectedPoints(row);
      const own = toNumber(row.projected_ownership);
      const actual = toNumber(row.actual_dk_points);

      const current = grouped.get(team) ?? { players: 0, projSum: 0, ownSum: 0, actualSum: 0, highProjCount: 0 };
      current.players += 1;
      current.projSum += Number.isFinite(proj) ? proj : 0;
      current.ownSum += Number.isFinite(own) ? own : 0;
      current.actualSum += Number.isFinite(actual) ? actual : 0;
      current.highProjCount += Number.isFinite(proj) && proj >= 25 ? 1 : 0;
      grouped.set(team, current);
    }

    const out: TeamRow[] = [];
    grouped.forEach((value, team) => {
      if (value.players < minTeamPlayers) return;
      out.push({
        team,
        players: value.players,
        avgProj: value.players > 0 ? value.projSum / value.players : NaN,
        avgOwn: value.players > 0 ? value.ownSum / value.players : NaN,
        avgActual: value.players > 0 ? value.actualSum / value.players : NaN,
        highProjCount: value.highProjCount,
      });
    });
    return out.sort((a, b) => b.avgProj - a.avgProj);
  }, [payload, minTeamPlayers]);

  return (
    <main className="page">
      <section className="hero">
        <h1>Team Review</h1>
        <p>Summarize team-level projection and ownership environments from the current projection snapshot.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Teams: {teamRows.length}</span>
        </div>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Team Filters</h2>
        <div className="form-grid">
          <label className="field">
            <span>Min Players Per Team</span>
            <input
              type="number"
              min={1}
              max={12}
              step={1}
              value={minTeamPlayers}
              onChange={(event) => setMinTeamPlayers(Number(event.target.value || 1))}
            />
          </label>
        </div>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Team Environment Table</h2>
        {loading ? (
          <p className="meta">Loading...</p>
        ) : error ? (
          <p className="error-text">{error}</p>
        ) : teamRows.length === 0 ? (
          <p className="meta">No teams meet the filter.</p>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Team</th>
                  <th>Players</th>
                  <th>Avg Proj</th>
                  <th>Avg Proj Own%</th>
                  <th>Avg Actual</th>
                  <th>Proj 25+ Count</th>
                </tr>
              </thead>
              <tbody>
                {teamRows.map((row) => (
                  <tr key={row.team}>
                    <td>{row.team}</td>
                    <td>{row.players}</td>
                    <td>{formatNumber(row.avgProj)}</td>
                    <td>{formatNumber(row.avgOwn)}</td>
                    <td>{formatNumber(row.avgActual)}</td>
                    <td>{row.highProjCount}</td>
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
