import { Loader2 } from "lucide-react";

export default function StatusBadge({ health }) {
  if (!health) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-muted/50 border border-muted-foreground/20 text-xs font-medium text-muted-foreground animate-pulse">
        <Loader2 className="w-3 h-3 animate-spin opacity-50" />
        Connecting...
      </div>
    );
  }

  const ok = health.status === "ok";
  const colorClass = ok
    ? "bg-green-500/10 text-green-500 border-green-500/20"
    : "bg-red-500/10 text-red-500 border-red-500/20";

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold tracking-wide ${colorClass} transition-colors`}>
      <span className="relative flex h-2 w-2">
        {ok && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>}
        <span className={`relative inline-flex rounded-full h-2 w-2 ${ok ? 'bg-green-500' : 'bg-red-500'}`}></span>
      </span>
      {ok ? "SYSTEM READY" : "SERVICE DEGRADED"}
    </div>
  );
}
