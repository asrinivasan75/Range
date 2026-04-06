import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Range — Poker Solver & Analysis Platform",
  description:
    "Advanced game-theoretic solver for No-Limit Texas Hold'em. Train strategies with CFR/MCCFR, analyze equilibria, and explore the mathematics of poker.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-graphite-950">
        {children}
      </body>
    </html>
  );
}
