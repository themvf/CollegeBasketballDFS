"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchInjuriesReview, getApiBaseUrl, type InjuriesReviewResponse } from "../lib/api";

type InjuriesViewProps = {
  selectedDate: string;
};

type InjuryRow = {
  player_name?: string;
  team?: string;
  status?: string;
  active?: boolean;
  notes?: string;
  updated_at?: string;
};

function asRows(input: unknown): InjuryRow[] {
  if (!Array.isArray(input)) return [];
  return input as InjuryRow[];
}

function numberValue(value: unknown): string {
  const num = Number(value ?? 0);
  return Number.isFinite(num) ? String(Math.round(num)) : "0";
}

export default function InjuriesView({ selectedDate }: InjuriesViewProps) {
  const apiBase = useMemo(() => getApiBaseUrl(), []);
  const [payload, setPayload] = useState<InjuriesReviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);
  const [message, setMessage] = useState<string>("");

  const effectiveRows = asRows(payload?.effective_rows);
  const feedRows = asRows(payload?.feed_rows);
  const manualRows = asRows(payload?.manual_rows);

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const next = await fetchInjuriesReview(selectedDate);
      if (!next) {
        setError("Could not load injuries review payload.");
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
  }, [selectedDate]);

  const uploadManualCsv = async () => {
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
        `${apiBase}/v1/injuries/manual/upload?selected_date=${encodeURIComponent(selectedDate)}`,
        {
          method: "POST",
          body: formData,
        },
      );
      const text = await response.text();
      if (!response.ok) {
        throw new Error(text || `Upload failed (${response.status})`);
      }
      setMessage("Manual injuries CSV uploaded.");
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
        <h1>Injuries</h1>
        <p>Review feed and manual injury status before building the optimizer pool.</p>
        <div className="badge-row">
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Bucket: {payload?.bucket_name ?? "N/A"}</span>
          <span className="badge">Legacy Fallback: {payload?.legacy_fallback_used ? "Yes" : "No"}</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Injury Summary</h2>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Effective Rows</p>
              <p className="value">{numberValue(payload?.summary?.effective_rows)}</p>
            </div>
            <div className="metric">
              <p className="label">Feed Rows</p>
              <p className="value">{numberValue(payload?.summary?.feed_rows)}</p>
            </div>
            <div className="metric">
              <p className="label">Manual Rows</p>
              <p className="value">{numberValue(payload?.summary?.manual_rows)}</p>
            </div>
            <div className="metric">
              <p className="label">Active Rows</p>
              <p className="value">{numberValue(payload?.summary?.active_rows)}</p>
            </div>
            <div className="metric">
              <p className="label">Remove Candidates</p>
              <p className="value warn">{numberValue(payload?.summary?.remove_candidates)}</p>
            </div>
          </div>
        </article>

        <article className="panel">
          <h2>Manual Overrides</h2>
          <p className="meta">Upload `injuries_manual.csv` format to override feed statuses.</p>
          <label className="field">
            <span>Manual Injuries CSV</span>
            <input
              type="file"
              accept=".csv"
              onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
            <button className="action-btn" disabled={uploading} onClick={uploadManualCsv}>
              {uploading ? "Uploading..." : "Upload Manual CSV"}
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
        <h2>Status Counts</h2>
        {loading ? (
          <p className="meta">Loading...</p>
        ) : !payload?.status_counts?.length ? (
          <p className="meta">No status rows available.</p>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Status</th>
                  <th>Rows</th>
                  <th>Active</th>
                </tr>
              </thead>
              <tbody>
                {payload.status_counts.map((row, idx) => (
                  <tr key={`${row.status ?? "status"}-${idx}`}>
                    <td>{String(row.status ?? "")}</td>
                    <td>{numberValue(row.rows)}</td>
                    <td>{numberValue(row.active)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="content-grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Effective Injuries</h2>
          {!effectiveRows.length ? (
            <p className="meta">No effective injury rows.</p>
          ) : (
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Team</th>
                    <th>Status</th>
                    <th>Active</th>
                    <th>Notes</th>
                    <th>Updated</th>
                  </tr>
                </thead>
                <tbody>
                  {effectiveRows.map((row, idx) => (
                    <tr key={`${row.player_name ?? "player"}-${row.team ?? "team"}-${idx}`}>
                      <td>{row.player_name ?? ""}</td>
                      <td>{row.team ?? ""}</td>
                      <td>{row.status ?? ""}</td>
                      <td>{row.active ? "Yes" : "No"}</td>
                      <td>{row.notes ?? ""}</td>
                      <td>{row.updated_at ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>

        <article className="panel">
          <h2>Source Split</h2>
          <p className="meta">Feed rows: {feedRows.length} | Manual rows: {manualRows.length}</p>
          <div className="table-shell" style={{ marginTop: 8 }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Status</th>
                  <th>Active</th>
                </tr>
              </thead>
              <tbody>
                {[...feedRows.map((row) => ({ ...row, _source: "feed" })), ...manualRows.map((row) => ({ ...row, _source: "manual" }))]
                  .slice(0, 200)
                  .map((row, idx) => (
                    <tr key={`${row._source}-${row.player_name ?? "player"}-${idx}`}>
                      <td>{String((row as { _source: string })._source || "")}</td>
                      <td>{row.player_name ?? ""}</td>
                      <td>{row.team ?? ""}</td>
                      <td>{row.status ?? ""}</td>
                      <td>{row.active ? "Yes" : "No"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </article>
      </section>
    </main>
  );
}
