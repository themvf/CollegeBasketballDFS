"use client";

import type { Route } from "next";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import {
  PRIMARY_NAV_ITEMS,
  canonicalizePath,
  getActiveSubNavItem,
  getPrimaryNavItem,
  getSubNavItems,
  resolvePrimaryNavKey,
} from "../lib/navigation";

type AppShellProps = {
  children: React.ReactNode;
};

function isRouteActive(pathname: string, href: string): boolean {
  const normalized = canonicalizePath(pathname);
  if (href === "/") {
    return normalized === "/";
  }
  return normalized === href || normalized.startsWith(`${href}/`);
}

export default function AppShell({ children }: AppShellProps) {
  const pathname = usePathname() || "/";
  const canonicalPath = canonicalizePath(pathname);
  const [mobileOpen, setMobileOpen] = useState(false);
  const activePrimary = getPrimaryNavItem(canonicalPath);
  const activeSubNav = getActiveSubNavItem(canonicalPath);
  const subNavItems = getSubNavItems(resolvePrimaryNavKey(canonicalPath));

  const crumbItems = useMemo(() => {
    const crumbs: Array<{ label: string; href?: string }> = [];
    crumbs.push({ label: activePrimary.label, href: activePrimary.href });
    if (activeSubNav) {
      crumbs.push({ label: activeSubNav.label });
    }
    if (crumbs.length === 0) {
      return [{ label: "Generate Lineup" }];
    }
    return crumbs;
  }, [activePrimary, activeSubNav]);

  const activeHeadline = activeSubNav?.label ?? activePrimary.label;
  const activeDescription = activeSubNav?.description ?? activePrimary.description;

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

  return (
    <div className="app-shell">
      <header className="site-header" aria-label="Main navigation">
        <div className="site-header-inner">
          <div className="brand-wrap">
            <span className="brand-mark" aria-hidden="true" />
            <p className="brand-name">CollegeBasketballDFS</p>
          </div>

          <nav className="primary-nav" aria-label="Primary">
            {PRIMARY_NAV_ITEMS.map((item) => {
              const primaryActive = item.key === activePrimary.key;
              return (
                <Link
                  key={item.key}
                  href={item.href as Route}
                  className={`primary-link ${primaryActive ? "active" : ""}`}
                  aria-current={primaryActive ? "page" : undefined}
                >
                  {item.label}
                </Link>
              );
            })}
          </nav>

          <button
            type="button"
            className="mobile-nav-toggle"
            onClick={() => setMobileOpen((prev) => !prev)}
            aria-expanded={mobileOpen}
            aria-controls="mobile-nav-drawer"
            aria-label="Toggle navigation"
          >
            {mobileOpen ? "Close" : "Menu"}
          </button>
        </div>
      </header>

      <nav className="subnav-shell" aria-label={`${activePrimary.label} sub-navigation`}>
        <div className="subnav-inner">
          <div className="subnav-scroll">
            {subNavItems.map((item) => {
              const subActive = isRouteActive(canonicalPath, item.href);
              return (
                <Link
                  key={item.key}
                  href={item.href as Route}
                  className={`subnav-link ${subActive ? "active" : ""}`}
                  aria-current={subActive ? "page" : undefined}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>

      {mobileOpen ? (
        <button
          type="button"
          className="mobile-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileOpen(false)}
        />
      ) : null}

      <aside className={`mobile-drawer ${mobileOpen ? "open" : ""}`} id="mobile-nav-drawer" aria-label="Mobile menu">
        <p className="mobile-heading">Sections</p>
        <nav className="mobile-nav-list" aria-label="Primary mobile">
          {PRIMARY_NAV_ITEMS.map((item) => {
            const primaryActive = item.key === activePrimary.key;
            return (
              <Link
                key={item.key}
                href={item.href as Route}
                className={`mobile-nav-item ${primaryActive ? "active" : ""}`}
                aria-current={primaryActive ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        <p className="mobile-heading">Current Workflow</p>
        <nav className="mobile-nav-list" aria-label="Sub-navigation mobile">
          {subNavItems.map((item) => {
            const subActive = isRouteActive(canonicalPath, item.href);
            return (
              <Link
                key={item.key}
                href={item.href as Route}
                className={`mobile-nav-item ${subActive ? "active" : ""}`}
                aria-current={subActive ? "page" : undefined}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </aside>

      <section className="app-main" id="main-content">
        <div className="page-header">
          <nav aria-label="Breadcrumb" className="breadcrumbs">
            <ol>
              {crumbItems.map((item, idx) => {
                const isLast = idx === crumbItems.length - 1;
                return (
                  <li key={`${item.label}-${idx}`}>
                    {idx > 0 ? <span className="crumb-sep">/</span> : null}
                    {item.href && !isLast ? (
                      <Link href={item.href as Route}>{item.label}</Link>
                    ) : (
                      <span aria-current={isLast ? "page" : undefined}>{item.label}</span>
                    )}
                  </li>
                );
              })}
            </ol>
          </nav>
          <h1 className="topbar-title">{activeHeadline}</h1>
          <p className="meta" style={{ marginTop: 6 }}>
            {activeDescription}
          </p>
        </div>
        <div className="app-content">{children}</div>
      </section>
    </div>
  );
}
