import { cn } from "@/lib/utils";

const CONFIG = {
  // quality labels
  ACCEPTABLE: { word: "PASSED", cls: "text-green-600 border-green-600/40 bg-green-50", rotate: "-rotate-[-6deg]" },
  DEGRADED: { word: "REVIEW", cls: "text-amber-600 border-amber-600/40 bg-amber-50", rotate: "rotate-3" },
  DEFECTIVE: { word: "REJECTED", cls: "text-red-600 border-red-600/40 bg-red-50", rotate: "-rotate-3" },
  // recommended actions
  PASS: { word: "PASS", cls: "text-green-600 border-green-600/40 bg-green-50", rotate: "-rotate-[-4deg]" },
  REVIEW: { word: "REVIEW", cls: "text-amber-600 border-amber-600/40 bg-amber-50", rotate: "rotate-2" },
  REJECT: { word: "REJECT", cls: "text-red-600 border-red-600/40 bg-red-50", rotate: "-rotate-2" },
};

export default function StampBadge({ label, size = "md", subtitle = null, hideSubtitle = false }) {
  const cfg = CONFIG[label] || { word: label || "UNKNOWN", cls: "text-amber-600 border-amber-600", rotate: "rotate-0" };

  const sizeClasses = {
    sm: "px-2 pb-0.5 pt-1 text-[10px] border-2 uppercase",
    md: "px-4 pb-1 pt-2 text-lg border-4 uppercase",
    lg: "px-6 pb-2 pt-3 text-3xl border-8 uppercase"
  };

  return (
    <div className="flex flex-col items-center justify-center shrink-0">
      <div
        className={cn(
          "inline-flex font-black font-serif tracking-widest rounded-sm opacity-90 shadow-sm backdrop-blur-sm",
          sizeClasses[size],
          cfg.cls,
          cfg.rotate
        )}
        aria-label={`Verdict: ${cfg.word}`}
      >
        {cfg.word}
      </div>
      {!hideSubtitle && subtitle && (
        <span className="text-[9px] font-bold tracking-widest text-muted-foreground uppercase mt-1 text-center bg-muted/50 px-1 rounded">
          {subtitle}
        </span>
      )}
    </div>
  );
}
