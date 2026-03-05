"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
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
  const [mobileOpen, setMobileOpen] = useState(false);
  const active = getNavItem(pathname);
  const crumbItems = useMemo(() => {
    if (!active || active.href === "/") {
      return [{ label: "Home", href: "/" as const }];
    }
    return [
      { label: "Home", href: "/" as const },
      { label: active.label, href: active.href },
    ];
  }, [active]);

  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  useEffect(() => {
    if (!mobileOpen) {
      document.body.style.removeProperty("overflow");
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMobileOpen(false);
      }
    };
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.body.style.removeProperty("overflow");
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [mobileOpen]);

  const renderGroup = (group: "overview" | "operations" | "analysis", label: string) => (
    <div className="nav-group">
      <p className="nav-group-label">{label}</p>
      {NAV_ITEMS.filter((item) => item.group === group).map((item) => {
        const activeState = isActive(pathname, item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`nav-item ${activeState ? "active" : ""}`}
            aria-current={activeState ? "page" : undefined}
          >
            <span className="nav-title">{item.label}</span>
            <span className="nav-desc">{item.description}</span>
          </Link>
        );
      })}
    </div>
  );

  return (
    <div className="app-shell">
      {mobileOpen ? (
        <button
          type="button"
          className="mobile-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      ) : null}

      <aside className={`app-sidebar ${mobileOpen ? "open" : ""}`} aria-label="Primary" id="primary-navigation">
        <div className="brand-block">
          <p className="brand-eyebrow">CollegeBasketballDFS</p>
          <h2>Control Room</h2>
          <p className="meta" style={{ marginTop: 8 }}>
            Streamlit parity navigation with a cleaner operational UX.
          </p>
        </div>

        {renderGroup("overview", "Overview")}
        {renderGroup("operations", "Operations")}
        {renderGroup("analysis", "Analysis")}
      </aside>

      <section className="app-main" id="main-content">
        <header className="topbar">
          <div className="topbar-main">
            <button
              type="button"
              className="mobile-nav-toggle"
              onClick={() => setMobileOpen((prev) => !prev)}
              aria-expanded={mobileOpen}
              aria-controls="primary-navigation"
              aria-label="Toggle navigation"
            >
              {mobileOpen ? "Close" : "Menu"}
            </button>

            <div>
              <nav aria-label="Breadcrumb" className="breadcrumbs">
                <ol>
                  {crumbItems.map((item, idx) => {
                    const isLast = idx === crumbItems.length - 1;
                    return (
                      <li key={`${item.label}-${idx}`}>
                        {idx > 0 ? <span className="crumb-sep">/</span> : null}
                        {isLast ? (
                          <span aria-current="page">{item.label}</span>
                        ) : (
                          <Link href={item.href}>{item.label}</Link>
                        )}
                      </li>
                    );
                  })}
                </ol>
              </nav>

              <p className="topbar-eyebrow">Active Module</p>
              <h1 className="topbar-title">{active?.label ?? "Dashboard"}</h1>
              <p className="meta" style={{ marginTop: 6 }}>
                {active?.description ?? "Migration workspace"}
              </p>
            </div>
          </div>

          <div className="topbar-actions">
            <p className="topbar-eyebrow">Active Module</p>
            <p className="meta" style={{ marginTop: 6 }}>
              Route: <code>{pathname}</code>
            </p>
          </div>
        </header>
        <div className="app-content">{children}</div>
      </section>
    </div>
  );
}
