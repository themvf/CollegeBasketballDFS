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

type UploadStatusCount = {
  status?: string;
  rows?: number;
};

type UploadResponse = {
  rows_saved?: number;
  blob_name?: string;
  status_counts?: UploadStatusCount[];
  detail?: string;
};

function asRows(input: unknown): InjuryRow[] {
  if (!Array.isArray(input)) return [];
  return input as InjuryRow[];
}

function numberValue(value: unknown): string {
  const num = Number(value ?? 0);
  return Number.isFinite(num) ? String(Math.round(num)) : "0";
}

function parseUploadResponse(text: string): UploadResponse | null {
  if (!text.trim()) return null;
  try {
    return JSON.parse(text) as UploadResponse;
  } catch {
    return null;
  }
}

function formatStatusCounts(statusCounts: UploadStatusCount[] | undefined): string {
  if (!Array.isArray(statusCounts) || statusCounts.length === 0) return "";
  return statusCounts
    .map((row) => `${numberValue(row.rows)} ${String(row.status ?? "unknown").trim() || "unknown"}`)
    .join(", ");
}

export default function InjuriesView({ selectedDate }: InjuriesViewProps) {
  const apiBase = useMemo(() => getApiBaseUrl(), []);
  const [payload, setPayload] = useState<InjuriesReviewResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [feedUploadFile, setFeedUploadFile] = useState<File | null>(null);
  const [manualUploadFile, setManualUploadFile] = useState<File | null>(null);
  const [feedUploading, setFeedUploading] = useState<boolean>(false);
  const [manualUploading, setManualUploading] = useState<boolean>(false);
  const [feedMessage, setFeedMessage] = useState<string>("");
  const [manualMessage, setManualMessage] = useState<string>("");

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

  const uploadCsv = async (
    mode: "feed" | "manual",
    uploadFile: File | null,
    setUploadingState: (value: boolean) => void,
    setMessageState: (value: string) => void,
    clearFile: () => void,
  ) => {
    if (!uploadFile) {
      setMessageState("Choose a CSV file first.");
      return;
    }
    setUploadingState(true);
    setMessageState("");
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      const response = await fetch(
        `${apiBase}/v1/injuries/${mode}/upload?selected_date=${encodeURIComponent(selectedDate)}`,
        {
          method: "POST",
          body: formData,
        },
      );
      const text = await response.text();
      const payload = parseUploadResponse(text);
      if (!response.ok) {
        throw new Error(String(payload?.detail || text || `Upload failed (${response.status})`));
      }
      const rowsSaved = numberValue(payload?.rows_saved);
      const statusSummary = formatStatusCounts(payload?.status_counts);
      const uploadLabel = mode === "feed" ? "Feed" : "Manual";
      const detailText = statusSummary ? ` (${statusSummary})` : "";
      setMessageState(`${uploadLabel} upload saved ${rowsSaved} rows for ${selectedDate}${detailText}.`);
      clearFile();
      await load();
    } catch (err) {
      setMessageState(err instanceof Error ? err.message : String(err));
    } finally {
      setUploadingState(false);
    }
  };

  const uploadFeedCsv = async () =>
    uploadCsv("feed", feedUploadFile, setFeedUploading, setFeedMessage, () => setFeedUploadFile(null));

  const uploadManualCsv = async () =>
    uploadCsv("manual", manualUploadFile, setManualUploading, setManualMessage, () => setManualUploadFile(null));

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
          <h2>Uploads</h2>
          <p className="meta">Use feed upload for daily injury reports and manual upload only for explicit overrides.</p>

          <div style={{ marginTop: 12 }}>
            <h3 style={{ margin: 0, fontSize: "1rem" }}>Daily Feed</h3>
            <p className="meta">Writes the selected date to the date-scoped injuries feed snapshot.</p>
            <label className="field">
              <span>Feed Injury CSV</span>
              <input
                type="file"
                accept=".csv"
                onChange={(event) => setFeedUploadFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
              <button className="action-btn" disabled={feedUploading} onClick={uploadFeedCsv}>
                {feedUploading ? "Uploading..." : "Upload Feed CSV"}
              </button>
            </div>
            {feedMessage ? <p className="meta" style={{ marginTop: 10 }}>{feedMessage}</p> : null}
          </div>

          <div style={{ marginTop: 18 }}>
            <h3 style={{ margin: 0, fontSize: "1rem" }}>Manual Overrides</h3>
            <p className="meta">Upload `injuries_manual.csv` format to create global overrides on top of the feed.</p>
            <label className="field">
              <span>Manual Injuries CSV</span>
              <input
                type="file"
                accept=".csv"
                onChange={(event) => setManualUploadFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <div style={{ marginTop: 10, display: "flex", gap: 10 }}>
              <button className="action-btn" disabled={manualUploading} onClick={uploadManualCsv}>
                {manualUploading ? "Uploading..." : "Upload Manual CSV"}
              </button>
              <button className="ghost-btn" disabled={loading} onClick={load}>
                Refresh
              </button>
            </div>
            {manualMessage ? <p className="meta" style={{ marginTop: 10 }}>{manualMessage}</p> : null}
          </div>

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
