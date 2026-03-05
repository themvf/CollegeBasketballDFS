import type { Metadata } from "next";
import AppShell from "../components/app-shell";
import "./globals.css";

export const metadata: Metadata = {
  title: "CollegeBasketballDFS Console",
  description: "Vercel migration console for slate ingestion, registry coverage, and lineup operations."
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <a href="#main-content" className="skip-link">
          Skip to content
        </a>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
