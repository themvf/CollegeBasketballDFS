export default function Loading() {
  return (
    <main className="page">
      <section className="hero">
        <div className="skeleton-line skeleton-title" />
        <div className="skeleton-line skeleton-text" />
        <div className="skeleton-line skeleton-text short" />
      </section>

      <section className="content-grid" style={{ marginTop: 16 }}>
        <article className="panel">
          <div className="skeleton-line skeleton-heading" />
          <div className="skeleton-grid">
            <div className="skeleton-card" />
            <div className="skeleton-card" />
            <div className="skeleton-card" />
            <div className="skeleton-card" />
          </div>
        </article>
        <article className="panel">
          <div className="skeleton-line skeleton-heading" />
          <div className="skeleton-line skeleton-text" />
          <div className="skeleton-line skeleton-text short" />
          <div className="skeleton-line skeleton-text short" />
        </article>
      </section>
    </main>
  );
}
