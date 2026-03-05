"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchTournamentReview, getApiBaseUrl, type TournamentReviewResponse } from "../lib/api";

type TournamentReviewViewProps = {
  selectedDate: string;
  slateKey?: string;
  defaultContestId?: string;
};

type Row = Record<string, unknown>;

function rows(input: unknown): Row[] {
  if (!Array.isArray(input)) return [];
  return input as Row[];
}

function n(value: unknown, digits = 2): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function i(value: unknown): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "0";
  return String(Math.round(num));
}

export default function TournamentReviewView({
  selectedDate,
  slateKey = "main",
  defaultContestId = "contest",
}: TournamentReviewViewProps) {
  const apiBase = useMemo(() => getApiBaseUrl(), []);
  const [contestId, setContestId] = useState<string>(defaultContestId);
  const [payload, setPayload] = useState<TournamentReviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");

  const entryRows = rows(payload?.entries_rows);
  const exposureRows = rows(payload?.exposure_rows);
  const missRows = rows(payload?.ownership_top_misses);
  const bucketRows = rows(payload?.ownership_buckets);

  const load = async (contest: string) => {
    const trimmed = contest.trim() || defaultContestId;
    setLoading(true);
    setError("");
    try {
      const next = await fetchTournamentReview(selectedDate, trimmed, slateKey);
      if (!next) {
        setError("Could not load tournament review payload.");
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
    load(contestId);
  }, [selectedDate, slateKey]);

  const uploadStandings = async () => {
    if (!uploadFile) {
      setMessage("Choose a CSV file first.");
      return;
    }
    const trimmed = contestId.trim() || defaultContestId;
    setUploading(true);
    setMessage("");
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      const response = await fetch(
        `${apiBase}/v1/reviews/tournament/standings/upload?selected_date=${encodeURIComponent(selectedDate)}&contest_id=${encodeURIComponent(trimmed)}&slate_key=${encodeURIComponent(slateKey)}`,
        {
          method: "POST",
          body: formData,
        },
      );
      const text = await response.text();
      if (!response.ok) {
        throw new Error(text || `Upload failed (${response.status})`);
      }
      setMessage("Standings CSV uploaded.");
      setUploadFile(null);
      await load(trimmed);
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="page">
      <section className="hero">
        <h1>Tournament Review</h1>
        <p>Analyze field construction, ownership misses, and projection drift against contest standings.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Contest: {contestId || defaultContestId}</span>
          <span className="badge">Slate: {slateKey}</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Contest Controls</h2>
          <label className="field">
            <span>Contest ID</span>
            <input
              type="text"
              value={contestId}
              onChange={(event) => setContestId(event.target.value)}
              placeholder="contest"
            />
          </label>
          <label className="field" style={{ marginTop: 8 }}>
            <span>Contest Standings CSV</span>
            <input
              type="file"
              accept=".csv"
              onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
            <button className="action-btn" onClick={() => load(contestId)} disabled={loading}>
              Refresh Review
            </button>
            <button className="ghost-btn" onClick={uploadStandings} disabled={uploading}>
              {uploading ? "Uploading..." : "Upload Standings"}
            </button>
          </div>
          {message ? <p className="meta" style={{ marginTop: 10 }}>{message}</p> : null}
          {error ? <p className="error-text">{error}</p> : null}
          {payload?.message ? <p className="meta" style={{ marginTop: 10 }}>{payload.message}</p> : null}
        </article>

        <article className="panel">
          <h2>Field Summary</h2>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Field Entries</p>
              <p className="value">{i(payload?.summary?.field_entries)}</p>
            </div>
            <div className="metric">
              <p className="label">Avg Salary Left</p>
              <p className="value">{n(payload?.summary?.avg_salary_left, 0)}</p>
            </div>
            <div className="metric">
              <p className="label">Avg Team Stack</p>
              <p className="value">{n(payload?.summary?.avg_max_team_stack)}</p>
            </div>
            <div className="metric">
              <p className="label">Avg Game Stack</p>
              <p className="value">{n(payload?.summary?.avg_max_game_stack)}</p>
            </div>
            <div className="metric">
              <p className="label">Top10 Avg Salary Left</p>
              <p className="value">{n(payload?.summary?.top10_avg_salary_left, 0)}</p>
            </div>
            <div className="metric">
              <p className="label">Ownership Samples</p>
              <p className="value">{i(payload?.summary?.ownership_samples)}</p>
            </div>
          </div>
        </article>
      </section>

      <section className="content-grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Ownership Diagnostics</h2>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">MAE</p>
              <p className="value">{n(payload?.ownership_summary?.mae)}</p>
            </div>
            <div className="metric">
              <p className="label">Bias</p>
              <p className="value">{n(payload?.ownership_summary?.bias)}</p>
            </div>
            <div className="metric">
              <p className="label">Correlation</p>
              <p className="value">{n(payload?.ownership_summary?.corr, 3)}</p>
            </div>
            <div className="metric">
              <p className="label">Within 5%</p>
              <p className="value">{n(payload?.ownership_summary?.within_5_pct, 1)}%</p>
            </div>
          </div>
        </article>

        <article className="panel">
          <h2>Projection Diagnostics</h2>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Matched Players</p>
              <p className="value">{i(payload?.projection_summary?.matched_players)}</p>
            </div>
            <div className="metric">
              <p className="label">Blend MAE</p>
              <p className="value">{n(payload?.projection_summary?.blend_mae)}</p>
            </div>
            <div className="metric">
              <p className="label">Our MAE</p>
              <p className="value">{n(payload?.projection_summary?.our_mae)}</p>
            </div>
            <div className="metric">
              <p className="label">Minutes MAE (L7)</p>
              <p className="value">{n(payload?.projection_summary?.minutes_mae_last7)}</p>
            </div>
          </div>
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Field Entries</h2>
        {loading ? (
          <p className="meta">Loading...</p>
        ) : entryRows.length === 0 ? (
          <p className="meta">No entry rows available.</p>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Entry</th>
                  <th>Points</th>
                  <th>Salary Left</th>
                  <th>Max Team Stack</th>
                  <th>Max Game Stack</th>
                </tr>
              </thead>
              <tbody>
                {entryRows.map((row, idx) => (
                  <tr key={`${String(row.EntryId ?? "entry")}-${idx}`}>
                    <td>{i(row.Rank)}</td>
                    <td>{String(row.EntryName ?? row.EntryId ?? "")}</td>
                    <td>{n(row.Points)}</td>
                    <td>{n(row.salary_left, 0)}</td>
                    <td>{i(row.max_team_stack)}</td>
                    <td>{i(row.max_game_stack)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="content-grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Exposure Snapshot</h2>
          {exposureRows.length === 0 ? (
            <p className="meta">No exposure rows available.</p>
          ) : (
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Team</th>
                    <th>Field Own%</th>
                    <th>Proj Own%</th>
                    <th>Own Diff</th>
                    <th>Actual FPTS</th>
                  </tr>
                </thead>
                <tbody>
                  {exposureRows.slice(0, 200).map((row, idx) => (
                    <tr key={`${String(row.Name ?? "player")}-${idx}`}>
                      <td>{String(row.Name ?? "")}</td>
                      <td>{String(row.TeamAbbrev ?? "")}</td>
                      <td>{n(row.field_ownership_pct)}</td>
                      <td>{n(row.projected_ownership)}</td>
                      <td>{n(row.ownership_diff_vs_proj)}</td>
                      <td>{n(row.final_dk_points)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>

        <article className="panel">
          <h2>Ownership Buckets / Misses</h2>
          {bucketRows.length === 0 ? (
            <p className="meta">No ownership bucket rows available.</p>
          ) : (
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Bucket</th>
                    <th>Samples</th>
                    <th>Proj Avg</th>
                    <th>Actual Avg</th>
                    <th>MAE</th>
                  </tr>
                </thead>
                <tbody>
                  {bucketRows.map((row, idx) => (
                    <tr key={`${String(row.ownership_bucket ?? "bucket")}-${idx}`}>
                      <td>{String(row.ownership_bucket ?? "")}</td>
                      <td>{i(row.samples)}</td>
                      <td>{n(row.avg_projected_ownership)}</td>
                      <td>{n(row.avg_actual_ownership)}</td>
                      <td>{n(row.mae)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {missRows.length > 0 ? (
            <div className="table-shell" style={{ marginTop: 10 }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Team</th>
                    <th>Proj Own</th>
                    <th>Actual Own</th>
                    <th>Abs Err</th>
                  </tr>
                </thead>
                <tbody>
                  {missRows.slice(0, 40).map((row, idx) => (
                    <tr key={`${String(row.Name ?? "player")}-${idx}`}>
                      <td>{String(row.Name ?? "")}</td>
                      <td>{String(row.TeamAbbrev ?? "")}</td>
                      <td>{n(row.projected_ownership)}</td>
                      <td>{n(row.actual_ownership)}</td>
                      <td>{n(row.abs_ownership_error)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </article>
      </section>
    </main>
  );
}
