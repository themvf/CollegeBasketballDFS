"use client";

import { useMemo, useState } from "react";
import { getApiBaseUrl } from "../lib/api";

type DkSlateImportViewProps = {
  selectedDate: string;
};

type ImportPayload = {
  coverage_before?: {
    players_total?: number;
    resolved_players?: number;
    unresolved_players?: number;
    conflict_players?: number;
    coverage_pct?: number;
  };
  coverage_after?: {
    players_total?: number;
    resolved_players?: number;
    unresolved_players?: number;
    conflict_players?: number;
    coverage_pct?: number;
  };
  derived_override_count?: number;
  remaining_unresolved_after_count?: number;
  remaining_unresolved_after?: Array<{
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

export default function DkSlateImportView({ selectedDate }: DkSlateImportViewProps) {
  const base = useMemo(() => getApiBaseUrl(), []);
  const [dateValue, setDateValue] = useState(selectedDate);
  const [contestType, setContestType] = useState("Classic");
  const [slateName, setSlateName] = useState("All");
  const [slateKey, setSlateKey] = useState("main");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<ImportPayload | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  async function submitImport() {
    if (!selectedFile) {
      setError("Select a DK salary CSV file first.");
      return;
    }

    setUploading(true);
    setError("");
    try {
      const params = new URLSearchParams({
        selected_date: dateValue,
        slate_key: slateKey || "main",
        contest_type: contestType,
        slate_name: slateName,
        persist: "true",
        site_id: "1",
      });
      const form = new FormData();
      form.append("file", selectedFile);
      const response = await fetch(`${base}/v1/registry/import-dk-slate?${params.toString()}`, {
        method: "POST",
        body: form,
      });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `DK slate import failed (${response.status})`);
      }
      setResult((await response.json()) as ImportPayload);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  const before = result?.coverage_before ?? {};
  const after = result?.coverage_after ?? {};
  const unresolvedAfter = result?.remaining_unresolved_after ?? [];

  return (
    <main className="page">
      <section className="hero">
        <h1>DK Slate</h1>
        <p>Upload the DraftKings salary CSV that drives the active slate. Supplemental ID mapping runs automatically when available.</p>
      </section>

      <section className="panel" style={{ marginTop: 16 }}>
        <h2>Upload</h2>
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
            <span>DK Salary CSV</span>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => {
                const file = e.target.files?.[0];
                setSelectedFile(file ?? null);
              }}
            />
          </label>
        </div>

        <div style={{ marginTop: 12 }}>
          <button className="action-btn" type="button" onClick={submitImport} disabled={uploading || !selectedFile}>
            {uploading ? "Uploading..." : "Upload DK Slate"}
          </button>
        </div>
        {error ? <p className="error-text">{error}</p> : null}
      </section>

      {!result ? null : (
        <>
          <section className="content-grid" style={{ marginTop: 16 }}>
            <article className="panel">
              <h2>Supplemental Mapping Before</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Players</p>
                  <p className="value">{intValue(before.players_total)}</p>
                </div>
                <div className="metric">
                  <p className="label">Resolved</p>
                  <p className="value">{intValue(before.resolved_players)}</p>
                </div>
                <div className="metric">
                  <p className="label">Unresolved</p>
                  <p className="value">{intValue(before.unresolved_players)}</p>
                </div>
                <div className="metric">
                  <p className="label">Coverage %</p>
                  <p className="value">{pctValue(before.coverage_pct)}</p>
                </div>
              </div>
            </article>

            <article className="panel">
              <h2>Supplemental Mapping After</h2>
              <div className="metric-grid">
                <div className="metric">
                  <p className="label">Players</p>
                  <p className="value">{intValue(after.players_total)}</p>
                </div>
                <div className="metric">
                  <p className="label">Resolved</p>
                  <p className="value">{intValue(after.resolved_players)}</p>
                </div>
                <div className="metric">
                  <p className="label">Unresolved</p>
                  <p className="value">{intValue(after.unresolved_players)}</p>
                </div>
                <div className="metric">
                  <p className="label">Coverage %</p>
                  <p className="value">{pctValue(after.coverage_pct)}</p>
                </div>
                <div className="metric">
                  <p className="label">Derived Overrides</p>
                  <p className="value">{intValue(result.derived_override_count)}</p>
                </div>
                <div className="metric">
                  <p className="label">Remaining Unresolved</p>
                  <p className="value">{intValue(result.remaining_unresolved_after_count)}</p>
                </div>
              </div>
            </article>
          </section>

          <section className="panel" style={{ marginTop: 16 }}>
            <h2>Remaining Mapping Exceptions After Import</h2>
            {unresolvedAfter.length === 0 ? (
              <p className="meta">No unresolved players remain.</p>
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
                    {unresolvedAfter.slice(0, 80).map((row, idx) => (
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
        </>
      )}
    </main>
  );
}
