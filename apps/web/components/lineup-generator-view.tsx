"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type LineupJobState = {
  job_id?: string;
  status?: string;
  progress_pct?: number;
  message?: string;
  created_at_utc?: string;
  started_at_utc?: string;
  completed_at_utc?: string;
  result?: {
    lineups_generated?: number;
    warnings?: string[];
    coverage?: { coverage_pct?: number; unresolved_players?: number; conflict_players?: number };
    model?: { version_key?: string; version_label?: string };
  };
  error?: string;
};

type ArtifactItem = {
  name: string;
  filename?: string;
  content_type?: string;
  size_bytes?: number;
  download_url?: string;
};

type GenerateResponse = {
  job_id: string;
  status: string;
  status_url: string;
  artifacts_url: string;
};

function apiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
}

function todayIso(): string {
  return new Date().toISOString().slice(0, 10);
}

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let idx = 0;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(idx === 0 ? 0 : 1)} ${units[idx]}`;
}

export default function LineupJobsPage() {
  const base = useMemo(() => apiBaseUrl(), []);

  const [selectedDate, setSelectedDate] = useState<string>(todayIso());
  const [contestType, setContestType] = useState<string>("Large GPP");
  const [modelKey, setModelKey] = useState<string>("salary_efficiency_ceiling_v1");
  const [numLineups, setNumLineups] = useState<number>(150);
  const [maxSalaryLeft, setMaxSalaryLeft] = useState<number>(400);
  const [globalMaxExposurePct, setGlobalMaxExposurePct] = useState<number>(50);
  const [rotowireCookie, setRotowireCookie] = useState<string>("");

  const [submitting, setSubmitting] = useState<boolean>(false);
  const [jobId, setJobId] = useState<string>("");
  const [jobState, setJobState] = useState<LineupJobState | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactItem[]>([]);
  const [error, setError] = useState<string>("");

  const pollHandleRef = useRef<number | null>(null);

  const clearPoll = () => {
    if (pollHandleRef.current !== null) {
      window.clearTimeout(pollHandleRef.current);
      pollHandleRef.current = null;
    }
  };

  const fetchArtifacts = async (activeJobId: string) => {
    const response = await fetch(`${base}/v1/lineups/jobs/${encodeURIComponent(activeJobId)}/artifacts`, {
      cache: "no-store",
    });
    if (!response.ok) {
      return;
    }
    const payload = (await response.json()) as { artifacts?: ArtifactItem[] };
    setArtifacts(Array.isArray(payload.artifacts) ? payload.artifacts : []);
  };

  const fetchJobState = async (activeJobId: string) => {
    const response = await fetch(`${base}/v1/lineups/jobs/${encodeURIComponent(activeJobId)}`, {
      cache: "no-store",
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Status request failed (${response.status})`);
    }
    const payload = (await response.json()) as LineupJobState;
    setJobState(payload);
    const status = String(payload.status || "").toLowerCase();
    if (status === "succeeded") {
      await fetchArtifacts(activeJobId);
      clearPoll();
      return;
    }
    if (status === "failed") {
      clearPoll();
      return;
    }
    clearPoll();
    pollHandleRef.current = window.setTimeout(() => {
      fetchJobState(activeJobId).catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      });
    }, 2500);
  };

  useEffect(() => {
    return () => {
      clearPoll();
    };
  }, []);

  const submitJob = async () => {
    setSubmitting(true);
    setError("");
    setArtifacts([]);
    setJobState(null);
    setJobId("");

    try {
      const body = {
        selected_date: selectedDate,
        contest_type: contestType,
        model_key: modelKey,
        num_lineups: numLineups,
        max_salary_left: maxSalaryLeft,
        global_max_exposure_pct: globalMaxExposurePct,
        rotowire_contest_type: "Classic",
        rotowire_slate_name: "All",
      };

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (rotowireCookie.trim()) {
        headers["X-Rotowire-Cookie"] = rotowireCookie.trim();
      }

      const response = await fetch(`${base}/v1/lineups/generate`, {
        method: "POST",
        headers,
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Generate request failed (${response.status})`);
      }
      const payload = (await response.json()) as GenerateResponse;
      setJobId(payload.job_id);
      await fetchJobState(payload.job_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  };

  const status = String(jobState?.status || "idle").toLowerCase();
  const progressPct = Number(jobState?.progress_pct || 0);
  const readyArtifacts = artifacts.length > 0;

  return (
    <main className="page">
      <section className="hero">
        <h1>Lineup Jobs</h1>
        <p>
          Phase 2 UI: submit lineup generation to the async API, track progress, and download generated artifacts.
        </p>
        <div className="badge-row">
          <span className="badge">API: {base}</span>
          <span className="badge">Date: {selectedDate}</span>
          <span className="badge">Mode: Async</span>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <h2>Job Request</h2>
          <p className="meta">Submit one model run. Polling begins automatically after job creation.</p>
          <div className="form-grid">
            <label className="field">
              <span>Slate Date</span>
              <input type="date" value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} />
            </label>

            <label className="field">
              <span>Contest Type</span>
              <select value={contestType} onChange={(e) => setContestType(e.target.value)}>
                <option>Large GPP</option>
                <option>Small GPP</option>
                <option>Cash</option>
              </select>
            </label>

            <label className="field">
              <span>Model</span>
              <select value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
                <option value="salary_efficiency_ceiling_v1">Salary-Efficiency v1 (Ceiling)</option>
                <option value="spike_v2_tail">Spike v2 (High-Variance Tail)</option>
                <option value="standout_v1_capture">Standout v1 (Leverage Surge)</option>
                <option value="chalk_value_capture_v1">Chalk-Value v1 (Leverage Pivots)</option>
                <option value="standard_v1">Standard v1 (Balanced Core)</option>
              </select>
            </label>

            <label className="field">
              <span>Lineups</span>
              <input
                type="number"
                min={1}
                max={1000}
                value={numLineups}
                onChange={(e) => setNumLineups(Number(e.target.value || 1))}
              />
            </label>

            <label className="field">
              <span>Max Salary Left</span>
              <input
                type="number"
                min={0}
                max={10000}
                value={maxSalaryLeft}
                onChange={(e) => setMaxSalaryLeft(Number(e.target.value || 0))}
              />
            </label>

            <label className="field">
              <span>Global Max Exposure %</span>
              <input
                type="number"
                min={0}
                max={100}
                step={1}
                value={globalMaxExposurePct}
                onChange={(e) => setGlobalMaxExposurePct(Number(e.target.value || 0))}
              />
            </label>
          </div>

          <label className="field" style={{ marginTop: 12 }}>
            <span>RotoWire Cookie (optional)</span>
            <input
              type="password"
              placeholder="Paste member cookie if endpoint requires auth"
              value={rotowireCookie}
              onChange={(e) => setRotowireCookie(e.target.value)}
            />
          </label>

          <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
            <button className="action-btn" disabled={submitting || !selectedDate} onClick={submitJob}>
              {submitting ? "Submitting..." : "Submit Job"}
            </button>
            <a
              className="ghost-btn"
              href={`/generate-lineup/saved-runs?date=${encodeURIComponent(selectedDate)}&slate_key=main`}
            >
              View Saved Runs
            </a>
            {jobId ? (
              <button
                className="ghost-btn"
                onClick={() => {
                  fetchJobState(jobId).catch((err: unknown) => {
                    setError(err instanceof Error ? err.message : String(err));
                  });
                }}
              >
                Refresh Status
              </button>
            ) : null}
          </div>
        </article>

        <article className="panel">
          <h2>Job Status</h2>
          <p className="meta">Queued and running jobs update every 2.5 seconds.</p>
          <div className="metric-grid">
            <div className="metric">
              <p className="label">Job ID</p>
              <p className="value" style={{ fontSize: "0.95rem", wordBreak: "break-all" }}>
                {jobId || "-"}
              </p>
            </div>
            <div className="metric">
              <p className="label">State</p>
              <p className={`value ${status === "succeeded" ? "ok" : status === "failed" ? "bad" : "warn"}`}>
                {jobState?.status || "idle"}
              </p>
            </div>
            <div className="metric">
              <p className="label">Progress</p>
              <p className="value">{progressPct}%</p>
            </div>
            <div className="metric">
              <p className="label">Lineups</p>
              <p className="value">{Number(jobState?.result?.lineups_generated || 0)}</p>
            </div>
          </div>

          <div className="progress-shell">
            <div className="progress-fill" style={{ width: `${Math.max(0, Math.min(100, progressPct))}%` }} />
          </div>
          <p className="meta">{jobState?.message || "No active job."}</p>
          {jobState?.error ? <p className="error-text">{jobState.error}</p> : null}
          {error ? <p className="error-text">{error}</p> : null}
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Artifacts</h2>
        {!readyArtifacts ? (
          <p className="meta">No artifacts yet. Generate a job and wait for success.</p>
        ) : (
          <div className="artifact-list">
            {artifacts.map((artifact) => {
              const href = `${base}${artifact.download_url || ""}`;
              return (
                <div className="artifact-row" key={artifact.name}>
                  <div>
                    <p className="artifact-name">{artifact.filename || artifact.name}</p>
                    <p className="meta" style={{ margin: 0 }}>
                      {artifact.content_type || "application/octet-stream"} | {formatBytes(artifact.size_bytes)}
                    </p>
                  </div>
                  <a className="download-link" href={href} target="_blank" rel="noreferrer">
                    Download
                  </a>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </main>
  );
}

