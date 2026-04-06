import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(n: number, decimals = 2): string {
  if (Math.abs(n) >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (Math.abs(n) >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return n.toFixed(decimals);
}

export function formatDuration(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}m ${secs.toFixed(0)}s`;
}

export function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function statusColor(status: string): string {
  switch (status) {
    case "running":
      return "text-accent-blue";
    case "completed":
      return "text-accent-emerald";
    case "failed":
      return "text-accent-rose";
    case "cancelled":
      return "text-accent-amber";
    default:
      return "text-graphite-400";
  }
}

export function statusBadgeClass(status: string): string {
  switch (status) {
    case "running":
      return "badge-blue";
    case "completed":
      return "badge-emerald";
    case "failed":
      return "badge-rose";
    case "cancelled":
      return "badge-amber";
    default:
      return "badge bg-graphite-800/50 text-graphite-400 border border-graphite-700/50";
  }
}

export const bucketColors = [
  "#ef4444", // 0 - weakest (red)
  "#f97316", // 1
  "#f59e0b", // 2
  "#eab308", // 3
  "#84cc16", // 4
  "#22c55e", // 5
  "#10b981", // 6
  "#06b6d4", // 7 - strongest (cyan)
];

export function bucketColor(bucket: number, total = 8): string {
  const idx = Math.min(bucket, total - 1);
  return bucketColors[idx] || bucketColors[0];
}
