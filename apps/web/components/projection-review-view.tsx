"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchProjectionReview, getApiBaseUrl, type ProjectionReviewResponse } from "../lib/api";

type ProjectionReviewViewProps = {
  selectedDate: string;
  slateKey?: string;
};

type ReviewRow = Record<string, unknown>;

function asRows(input: unknown): ReviewRow[] {
  if (!Array.isArray(input)) return [];
  return input as ReviewRow[];
}

function formatNumber(value: unknown, digits = 2): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "-";
  return num.toFixed(digits);
}

function formatInt(value: unknown): string {
  const num = Number(value);
  if (!Number.isFinite(num)) return "0";
  return String(Math.round(num));
}

export default function ProjectionReviewView({ selectedDate, slateKey = "main" }: ProjectionReviewViewProps) {
  const apiBase = useMemo(() => getApiBaseUrl(), []);
  const [payload, setPayload] = useState<ProjectionReviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");

  const rows = asRows(payload?.rows);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const next = await fetchProjectionReview(selectedDate, slateKey);
      if (!next) {
        setError("Could not load projection review payload.");
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

  const uploadOwnership = async () => {
    if (!uploadFile) {
      setMessage("Choose a CSV file first.");
      return;
    }
    setUploading(true);
    setMessage("");
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      const response = await fetch(
        `${apiBase}/v1/reviews/projection/ownership/upload?selected_date=${encodeURIComponent(selectedDate)}&slate_key=${encodeURIComponent(slateKey)}`,
        {
          method: "POST",
          body: formData,
        },
      );
      const text = await response.text();
      if (!response.ok) {
        throw new Error(text || `Upload failed (${response.status})`);
      }
      setMessage("Ownership CSV uploaded.");
      setUploadFile(null);
      await load();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="page">
      <section className="hero">
        <h1>Projection Review</h1>
        <p>Compare projected vs actual points and ownership calibration for the selected slate date.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Slate: {slateKey}</span>
          <span className="badge">Rows: {formatInt(payload?.summary?.projection_rows)}</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Model Accuracy</h2>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Projection Rows</p>
              <p className="value">{formatInt(payload?.summary?.projection_rows)}</p>
            </div>
            <div className="metric">
              <p className="label">Actual Matched</p>
              <p className="value">{formatInt(payload?.summary?.actual_matched)}</p>
            </div>
            <div className="metric">
              <p className="label">Blend MAE</p>
              <p className="value">{formatNumber(payload?.summary?.blend_mae)}</p>
            </div>
            <div className="metric">
              <p className="label">Our MAE</p>
              <p className="value">{formatNumber(payload?.summary?.our_mae)}</p>
            </div>
            <div className="metric">
              <p className="label">Vegas MAE</p>
              <p className="value">{formatNumber(payload?.summary?.vegas_mae)}</p>
            </div>
          </div>
        </article>

        <article className="panel">
          <h2>Ownership Accuracy</h2>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Ownership Rows</p>
              <p className="value">{formatInt(payload?.summary?.ownership_rows)}</p>
            </div>
            <div className="metric">
              <p className="label">Ownership MAE</p>
              <p className="value">{formatNumber(payload?.summary?.ownership_mae)}</p>
            </div>
            <div className="metric">
              <p className="label">Ownership Bias</p>
              <p className="value">{formatNumber(payload?.summary?.ownership_bias)}</p>
            </div>
            <div className="metric">
              <p className="label">Rank Spearman</p>
              <p className="value">{formatNumber(payload?.summary?.ownership_rank_spearman, 3)}</p>
            </div>
          </div>

          <label className="field" style={{ marginTop: 12 }}>
            <span>Upload Actual Ownership CSV</span>
            <input
              type="file"
              accept=".csv"
              onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
            <button className="action-btn" disabled={uploading} onClick={uploadOwnership}>
              {uploading ? "Uploading..." : "Upload Ownership CSV"}
            </button>
            <button className="ghost-btn" disabled={loading} onClick={load}>
              Refresh
            </button>
          </div>
          {message ? <p className="meta" style={{ marginTop: 10 }}>{message}</p> : null}
          {error ? <p className="error-text">{error}</p> : null}
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Projection Rows</h2>
        {loading ? (
          <p className="meta">Loading...</p>
        ) : rows.length === 0 ? (
          <p className="meta">No review rows available.</p>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Pos</th>
                  <th>Salary</th>
                  <th>Blend Proj</th>
                  <th>Actual</th>
                  <th>Blend Err</th>
                  <th>Proj Own</th>
                  <th>Actual Own</th>
                  <th>Own Err</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, idx) => (
                  <tr key={`${String(row.Name ?? row["Name + ID"] ?? "player")}-${idx}`}>
                    <td>{String(row.Name ?? row["Name + ID"] ?? "")}</td>
                    <td>{String(row.TeamAbbrev ?? "")}</td>
                    <td>{String(row.Position ?? "")}</td>
                    <td>{formatInt(row.Salary)}</td>
                    <td>{formatNumber(row.blended_projection)}</td>
                    <td>{formatNumber(row.actual_dk_points)}</td>
                    <td>{formatNumber(row.blend_error)}</td>
                    <td>{formatNumber(row.projected_ownership)}</td>
                    <td>{formatNumber(row.actual_ownership)}</td>
                    <td>{formatNumber(row.ownership_error)}</td>
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
