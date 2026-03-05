"use client";

import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Route error:", error);
  }, [error]);

  return (
    <main className="page">
      <section className="hero">
        <h1>Something went wrong</h1>
        <p>{error.message || "Unexpected error while loading this section."}</p>
        <div className="badge-row">
          <button className="action-btn" onClick={reset} type="button">
            Try again
          </button>
        </div>
      </section>
    </main>
  );
}
