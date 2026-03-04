"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { NAV_ITEMS, getNavItem } from "../lib/navigation";

type AppShellProps = {
  children: React.ReactNode;
};

function isActive(pathname: string, href: string): boolean {
  if (href === "/") {
    return pathname === "/";
  }
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function AppShell({ children }: AppShellProps) {
  const pathname = usePathname() || "/";
  const active = getNavItem(pathname);

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="brand-block">
          <p className="brand-eyebrow">CollegeBasketballDFS</p>
          <h2>Control Room</h2>
          <p className="meta" style={{ marginTop: 8 }}>
            Streamlit parity navigation with a cleaner operational UX.
          </p>
        </div>

        <div className="nav-group">
          <p className="nav-group-label">Overview</p>
          {NAV_ITEMS.filter((item) => item.group === "overview").map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-item ${isActive(pathname, item.href) ? "active" : ""}`}
            >
              <span className="nav-title">{item.label}</span>
              <span className="nav-desc">{item.description}</span>
            </Link>
          ))}
        </div>

        <div className="nav-group">
          <p className="nav-group-label">Operations</p>
          {NAV_ITEMS.filter((item) => item.group === "operations").map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-item ${isActive(pathname, item.href) ? "active" : ""}`}
            >
              <span className="nav-title">{item.label}</span>
              <span className="nav-desc">{item.description}</span>
            </Link>
          ))}
        </div>

        <div className="nav-group">
          <p className="nav-group-label">Analysis</p>
          {NAV_ITEMS.filter((item) => item.group === "analysis").map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={`nav-item ${isActive(pathname, item.href) ? "active" : ""}`}
            >
              <span className="nav-title">{item.label}</span>
              <span className="nav-desc">{item.description}</span>
            </Link>
          ))}
        </div>
      </aside>

      <section className="app-main">
        <header className="topbar">
          <div>
            <p className="topbar-eyebrow">Active Module</p>
            <h1 className="topbar-title">{active?.label ?? "Dashboard"}</h1>
            <p className="meta" style={{ marginTop: 6 }}>
              {active?.description ?? "Migration workspace"}
            </p>
          </div>
        </header>
        <div className="app-content">{children}</div>
      </section>
    </div>
  );
}
