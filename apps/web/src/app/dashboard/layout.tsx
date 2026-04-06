"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/dashboard", label: "Overview", icon: "◈" },
  { href: "/dashboard/play", label: "Play vs Bot", icon: "♠" },
  { href: "/dashboard/runs", label: "Training Runs", icon: "▶" },
  { href: "/dashboard/explorer", label: "Explorer", icon: "◇" },
  { href: "/dashboard/strategy", label: "Strategy", icon: "⬡" },
  { href: "/dashboard/artifacts", label: "Artifacts", icon: "□" },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-56 shrink-0 border-r border-graphite-800/50 bg-graphite-950/80 backdrop-blur-sm">
        <div className="sticky top-0 h-screen flex flex-col">
          {/* Logo */}
          <div className="p-5 border-b border-graphite-800/50">
            <Link href="/" className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-md bg-gradient-to-br from-accent-blue to-accent-cyan flex items-center justify-center">
                <span className="text-white font-bold text-xs">R</span>
              </div>
              <span className="font-semibold text-white text-sm tracking-tight">
                Range
              </span>
              <span className="text-[10px] font-mono text-graphite-500 ml-auto">
                v0.1
              </span>
            </Link>
          </div>

          {/* Nav */}
          <nav className="flex-1 p-3 space-y-0.5">
            {navItems.map((item) => {
              const isActive =
                item.href === "/dashboard"
                  ? pathname === "/dashboard"
                  : pathname.startsWith(item.href);

              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={cn(
                    "flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-all duration-200",
                    isActive
                      ? "bg-graphite-800/60 text-white"
                      : "text-graphite-400 hover:text-graphite-200 hover:bg-graphite-800/30"
                  )}
                >
                  <span className="text-xs opacity-60 w-4 text-center font-mono">
                    {item.icon}
                  </span>
                  {item.label}
                </Link>
              );
            })}
          </nav>

          {/* Footer */}
          <div className="p-4 border-t border-graphite-800/50">
            <div className="flex items-center gap-2 text-[10px] text-graphite-600">
              <div className="w-1.5 h-1.5 rounded-full bg-accent-emerald animate-pulse" />
              <span>API Connected</span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 min-w-0">
        <div className="p-6 md:p-8 max-w-7xl">{children}</div>
      </main>
    </div>
  );
}
