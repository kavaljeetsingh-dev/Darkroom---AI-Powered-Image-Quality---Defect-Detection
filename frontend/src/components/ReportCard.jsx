import { useState } from "react";
import StampBadge from "./StampBadge.jsx";
import { Card, CardHeader, CardContent, CardTitle, CardDescription } from "./ui/card.jsx";
import { Button } from "./ui/button.jsx";
import { Badge } from "./ui/badge.jsx";
import { ChevronDown, ChevronRight, Activity, Zap } from "lucide-react";

const ISSUE_LABELS = {
  blur: "Blur / soft focus",
  underexposure: "Underexposure",
  overexposure: "Overexposure",
  noise: "Noise",
  corruption: "Severe degradation / artifacts",
  potential_defect: "Potential defect (unclassified)",
};

const SEVERITY_COLORS = {
  low: "bg-blue-500/10 text-blue-500 border-blue-500/20",
  medium: "bg-amber-500/10 text-amber-500 border-amber-500/20",
  high: "bg-red-500/10 text-red-500 border-red-500/20",
};

function ScoreDial({ score }) {
  const clamped = Math.max(0, Math.min(100, score));
  const circumference = 2 * Math.PI * 42;
  const offset = circumference * (1 - clamped / 100);

  let toneClass = "text-red-500";
  if (clamped >= 78) toneClass = "text-green-500";
  else if (clamped >= 45) toneClass = "text-amber-500";

  return (
    <div className="relative w-32 h-32 flex items-center justify-center shrink-0">
      <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90 block">
        <circle cx="50" cy="50" r="42" className="stroke-muted fill-none stroke-[8px]" />
        <circle
          cx="50"
          cy="50"
          r="42"
          className={`fill-none stroke-[8px] transition-all duration-1000 ease-out ${toneClass} stroke-current`}
          style={{ strokeDasharray: circumference, strokeDashoffset: offset }}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold tracking-tighter">{clamped.toFixed(0)}</span>
        <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground mr-1">Score</span>
      </div>
    </div>
  );
}

function IssueRow({ issue }) {
  const [open, setOpen] = useState(false);
  const evidenceEntries = Object.entries(issue.evidence || {});
  const isAnomaly = issue.type === "potential_defect";
  const sevColor = SEVERITY_COLORS[issue.severity] || "bg-muted";

  return (
    <div className="border-b last:border-0 border-border group overflow-hidden">
      <button
        className="w-full flex items-center gap-4 py-4 text-left hover:bg-muted/30 transition-colors px-2 rounded-md"
        onClick={() => setOpen((v) => !v)}
      >
        <span className={`text-xs font-bold uppercase tracking-wider px-2 py-1 rounded-sm border ${sevColor}`}>
          {issue.severity}
        </span>
        <span className="font-semibold flex-1 text-foreground/90">{ISSUE_LABELS[issue.type] || issue.type}</span>

        {isAnomaly ? (
          <Badge variant="secondary" className="font-mono bg-destructive/10 text-destructive border-transparent">
            ⚠ {issue.anomaly_score?.toFixed(3) ?? "—"}
          </Badge>
        ) : (
          <span className="text-sm font-medium text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
            {Math.round((issue.confidence ?? 0) * 100)}%
          </span>
        )}

        {open ? <ChevronDown className="w-4 h-4 text-muted-foreground" /> : <ChevronRight className="w-4 h-4 text-muted-foreground" />}
      </button>

      <div className={`overflow-hidden transition-all duration-300 ${open ? 'max-h-96 opacity-100 mb-4' : 'max-h-0 opacity-0'}`}>
        <div className="px-4 py-3 bg-muted/30 rounded-lg mx-2 text-sm text-foreground/80 shadow-inner">
          <p className="mb-3">{issue.description}</p>

          {isAnomaly && issue.anomaly_score != null && (
            <div className="mb-3 text-xs p-2 bg-destructive/5 text-destructive rounded-md border border-destructive/20 flex flex-col gap-1">
              <strong>Anomaly Score: {issue.anomaly_score.toFixed(4)}</strong>
              <span>Top features deviating from clean-image distribution:</span>
            </div>
          )}

          {evidenceEntries.length > 0 && (
            <div className="grid grid-cols-2 gap-2 mt-2">
              {evidenceEntries.map(([k, v]) => (
                <div key={k} className="bg-background/50 border rounded p-2 flex flex-col">
                  <span className="text-[10px] uppercase text-muted-foreground font-semibold">{k.replace(/_/g, ' ')}</span>
                  <span className="font-mono text-sm">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ReportCard({ result }) {
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const hasHeatmap = Boolean(result.blur_heatmap_png_base64);

  return (
    <Card className="overflow-hidden shadow-xl border-border/50 bg-background/50 backdrop-blur-sm">
      <div className="bg-muted/30 border-b p-6 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 relative">
        <div className="absolute top-0 left-0 w-1 h-full bg-primary" />
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-primary/80 mb-1 flex items-center gap-2">
            <Activity className="w-3 h-3" />
            Report #{String(result.id).padStart(4, "0")}
          </p>
          <h2 className="text-2xl font-bold tracking-tight text-foreground truncate max-w-[200px] sm:max-w-xs md:max-w-md" title={result.filename}>
            {result.filename}
          </h2>
          <p className="text-xs text-muted-foreground mt-1 font-medium">
            {result.image_width && result.image_height ? `${result.image_width} × ${result.image_height}px · ` : ""}
            {new Date(result.created_at).toLocaleString()}
          </p>
        </div>

        <div className="flex flex-col items-end gap-1 shrink-0">
          <div className="flex gap-2">
            <StampBadge label={result.quality_label} subtitle="QUALITY" />
            {result.recommended_action && (
              <StampBadge label={result.recommended_action} subtitle="ACTION" />
            )}
          </div>
          {result.quality_label === "ACCEPTABLE" && result.recommended_action === "REVIEW" && (
            <span className="text-[10px] text-muted-foreground mt-1 max-w-[200px] text-right leading-tight">
              Quality is acceptable, but an anomaly was detected outside the known defect categories.
            </span>
          )}
        </div>
      </div>

      <CardContent className="p-0">
        <div className="grid grid-cols-1 md:grid-cols-3 divide-y md:divide-y-0 md:divide-x border-b">
          <div className="p-8 flex flex-col items-center justify-center bg-gradient-to-br from-background to-muted/20 relative">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-primary/5 to-transparent opacity-50" />
            <ScoreDial score={result.quality_score} />

            {hasHeatmap && (
              <Button
                variant={showHeatmap ? "secondary" : "outline"}
                size="sm"
                className="mt-6 rounded-full w-full max-w-[200px] relative z-10 font-semibold"
                onClick={() => setShowHeatmap((v) => !v)}
              >
                <Zap className={`w-4 h-4 mr-2 ${showHeatmap ? 'text-primary animate-pulse' : 'text-muted-foreground'}`} />
                {showHeatmap ? "Hide Sharpness Map" : "Show Sharpness Map"}
              </Button>
            )}
          </div>

          <div className="col-span-2 p-0 flex flex-col relative">
            {showHeatmap && hasHeatmap ? (
              <div className="h-full min-h-[300px] w-full relative animate-in fade-in duration-500 flex items-center justify-center p-6 bg-black/5">
                <img
                  src={`data:image/png;base64,${result.blur_heatmap_png_base64}`}
                  alt="Blur localization heatmap"
                  className="rounded-lg shadow-inner max-h-[400px] object-contain border border-black/10"
                />
                <Badge className="absolute top-4 right-4 bg-background/80 text-foreground backdrop-blur-md">Red = degraded focus</Badge>
              </div>
            ) : (
              <div className="p-6">
                <h3 className="text-sm font-bold uppercase tracking-wider text-muted-foreground mb-4">Detected Issues</h3>
                {result.issues.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-12 text-center h-full text-muted-foreground opacity-60">
                    <div className="w-12 h-12 bg-green-500/10 text-green-500 rounded-full flex items-center justify-center mb-3">✓</div>
                    <p className="font-semibold text-lg">Spotless!</p>
                    <p className="text-sm">No quality issues detected.</p>
                  </div>
                ) : (
                  <div className="flex flex-col">
                    {result.issues.map((issue, i) => (
                      <IssueRow key={`${issue.type}-${i}`} issue={issue} />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {result.image_stats && (
          <div className="p-6 bg-muted/10">
            <button
              className="flex items-center gap-2 text-sm font-semibold text-muted-foreground hover:text-foreground transition-colors w-full"
              onClick={() => setShowStats(!showStats)}
            >
              <Activity className="w-4 h-4" />
              Machine Vision Statistics
              {showStats ? <ChevronDown className="w-4 h-4 ml-auto" /> : <ChevronRight className="w-4 h-4 ml-auto" />}
            </button>

            {showStats && (
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3 mt-4 animate-in slide-in-from-top-2 fade-in">
                {Object.entries(result.image_stats).map(([k, v]) => (
                  <div key={k} className="bg-background border rounded-lg p-3 shadow-sm flex flex-col justify-between">
                    <span className="text-[10px] text-muted-foreground uppercase font-bold tracking-tight mb-2 truncate" title={k}>{k.replace(/_/g, ' ')}</span>
                    <span className="font-mono text-sm font-medium text-foreground">{typeof v === 'number' ? v.toFixed(4) : v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
