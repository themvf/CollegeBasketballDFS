"use client";

import { useMemo, useState } from "react";
import { getApiBaseUrl } from "../lib/api";

type RotowireScraperViewProps = {
  selectedDate: string;
};

type SlateRow = {
  slate_id?: number;
  contest_type?: string;
  slate_name?: string;
  game_count?: number;
  players?: number;
};

type CoveragePayload = {
  coverage?: {
    players_total?: number;
    resolved_players?: number;
    unresolved_players?: number;
    conflict_players?: number;
    coverage_pct?: number;
  };
  unresolved_players?: Array<{
    player_name?: string;
    team_abbr?: string;
    salary?: number;
    dk_resolution_status?: string;
    dk_match_reason?: string;
  }>;
};

function intValue(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0";
  }
  return String(Math.round(value));
}

function pctValue(value: number | undefined | null): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "0.00%";
  }
  return `${value.toFixed(2)}%`;
}

export default function RotowireScraperView({ selectedDate }: RotowireScraperViewProps) {
  const base = useMemo(() => getApiBaseUrl(), []);
  const [dateValue, setDateValue] = useState(selectedDate);
  const [contestType, setContestType] = useState("Classic");
  const [slateName, setSlateName] = useState("All");
  const [slateKey, setSlateKey] = useState("main");
  const [cookieValue, setCookieValue] = useState("");
  const [loadingSlates, setLoadingSlates] = useState(false);
  const [loadingCoverage, setLoadingCoverage] = useState(false);
  const [error, setError] = useState("");
  const [slates, setSlates] = useState<SlateRow[]>([]);
  const [selectedSlateId, setSelectedSlateId] = useState<number | "">("");
  const [coverage, setCoverage] = useState<CoveragePayload | null>(null);

  async function loadSlates() {
    setLoadingSlates(true);
    setError("");
    try {
      const params = new URLSearchParams({
        selected_date: dateValue,
        contest_type: contestType,
        slate_name: slateName,
        site_id: "1",
      });
      const headers: Record<string, string> = {};
      if (cookieValue.trim()) {
        headers["X-Rotowire-Cookie"] = cookieValue.trim();
      }
      const response = await fetch(`${base}/v1/rotowire/slates?${params.toString()}`, {
        cache: "no-store",
        headers,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Failed to load slates (${response.status})`);
      }
      const payload = (await response.json()) as { slates?: SlateRow[] };
      const rows = Array.isArray(payload.slates) ? payload.slates : [];
      setSlates(rows);
      if (rows.length > 0) {
        const firstId = Number(rows[0].slate_id);
        setSelectedSlateId(Number.isFinite(firstId) ? firstId : "");
      } else {
        setSelectedSlateId("");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingSlates(false);
    }
  }

  async function loadCoverage() {
    setLoadingCoverage(true);
    setError("");
    try {
      const params = new URLSearchParams({
        selected_date: dateValue,
        slate_key: slateKey || "main",
        contest_type: contestType,
        slate_name: slateName,
        site_id: "1",
      });
      if (selectedSlateId !== "") {
        params.set("slate_id", String(selectedSlateId));
      }
      const headers: Record<string, string> = {};
      if (cookieValue.trim()) {
        headers["X-Rotowire-Cookie"] = cookieValue.trim();
      }
      const response = await fetch(`${base}/v1/registry/coverage?${params.toString()}`, {
        cache: "no-store",
        headers,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Failed to load coverage (${response.status})`);
      }
      setCoverage((await response.json()) as CoveragePayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoadingCoverage(false);
    }
  }

  const coverageStats = coverage?.coverage ?? {};
  const unresolved = coverage?.unresolved_players ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>RotoWire Scraper</h1>
        <p>Pull slate catalog and validate DK-ID registry coverage from the API.</p>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Request Controls</h2>
        <div className="form-grid">
          <label className="field">
            <span>Date</span>
            <input type="date" value={dateValue} onChange={(e) => setDateValue(e.target.value)} />
          </label>
          <label className="field">
            <span>Contest Type</span>
            <input value={contestType} onChange={(e) => setContestType(e.target.value)} />
          </label>
          <label className="field">
            <span>Slate Name</span>
            <input value={slateName} onChange={(e) => setSlateName(e.target.value)} />
          </label>
          <label className="field">
            <span>Slate Key</span>
            <input value={slateKey} onChange={(e) => setSlateKey(e.target.value)} />
          </label>
          <label className="field">
            <span>Selected Slate ID</span>
            <select
              value={selectedSlateId === "" ? "" : String(selectedSlateId)}
              onChange={(e) => {
                const next = e.target.value;
                setSelectedSlateId(next ? Number(next) : "");
              }}
            >
              <option value="">(auto by contest/name)</option>
              {slates.map((row, idx) => (
                <option key={`${row.slate_id}-${idx}`} value={String(row.slate_id ?? "")}>
                  {`${row.slate_id ?? "?"} | ${row.contest_type ?? ""} | ${row.slate_name ?? ""}`}
                </option>
              ))}
            </select>
          </label>
          <label className="field">
            <span>RotoWire Cookie (optional)</span>
            <input
              type="password"
              value={cookieValue}
              onChange={(e) => setCookieValue(e.target.value)}
              placeholder="Paste Cookie header"
            />
          </label>
        </div>

        <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
          <button className="action-btn" type="button" onClick={loadSlates} disabled={loadingSlates || !dateValue}>
            {loadingSlates ? "Loading Slates..." : "Load Slates"}
          </button>
          <button className="ghost-btn" type="button" onClick={loadCoverage} disabled={loadingCoverage || !dateValue}>
            {loadingCoverage ? "Loading Coverage..." : "Check Coverage"}
          </button>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      <section className="content-grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <h2>Slate Catalog</h2>
          {slates.length === 0 ? (
            <p className="meta">No slates loaded yet.</p>
          ) : (
            <div className="table-shell">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Slate ID</th>
                    <th>Contest</th>
                    <th>Name</th>
                    <th>Games</th>
                    <th>Players</th>
                  </tr>
                </thead>
                <tbody>
                  {slates.map((row, idx) => (
                    <tr key={`${row.slate_id}-${idx}`}>
                      <td>{intValue(row.slate_id)}</td>
                      <td>{row.contest_type ?? "-"}</td>
                      <td>{row.slate_name ?? "-"}</td>
                      <td>{intValue(row.game_count)}</td>
                      <td>{intValue(row.players)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>

        <article className="panel">
          <h2>Coverage</h2>
          {!coverage ? (
            <p className="meta">No coverage payload loaded yet.</p>
          ) : (
            <div className="metric-grid">
              <div className="metric">
                <p className="label">Players Total</p>
                <p className="value">{intValue(coverageStats.players_total)}</p>
              </div>
              <div className="metric">
                <p className="label">Resolved</p>
                <p className="value">{intValue(coverageStats.resolved_players)}</p>
              </div>
              <div className="metric">
                <p className="label">Unresolved</p>
                <p className="value">{intValue(coverageStats.unresolved_players)}</p>
              </div>
              <div className="metric">
                <p className="label">Conflicts</p>
                <p className="value">{intValue(coverageStats.conflict_players)}</p>
              </div>
              <div className="metric">
                <p className="label">Coverage %</p>
                <p className="value">{pctValue(coverageStats.coverage_pct)}</p>
              </div>
            </div>
          )}
        </article>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Unresolved Players</h2>
        {unresolved.length === 0 ? (
          <p className="meta">No unresolved players returned.</p>
        ) : (
          <div className="table-shell">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Team</th>
                  <th>Salary</th>
                  <th>Status</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {unresolved.slice(0, 80).map((row, idx) => (
                  <tr key={`${row.player_name}-${row.team_abbr}-${idx}`}>
                    <td>{row.player_name ?? "-"}</td>
                    <td>{row.team_abbr ?? "-"}</td>
                    <td>{intValue(row.salary)}</td>
                    <td>{row.dk_resolution_status ?? "-"}</td>
                    <td>{row.dk_match_reason ?? "-"}</td>
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
