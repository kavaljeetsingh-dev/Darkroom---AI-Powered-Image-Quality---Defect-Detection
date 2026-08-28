import StampBadge from "./StampBadge.jsx";
import { Card } from "./ui/card.jsx";
import { Button } from "./ui/button.jsx";

export default function HistoryStrip({ items, loading, onSelect, selectedId, onClearAll, onDelete }) {
  return (
    <div className="flex flex-col h-full bg-muted/20 border rounded-xl overflow-hidden shadow-sm">
      <div className="p-4 border-b bg-background/50 backdrop-blur flex justify-between items-center">
        <h3 className="font-semibold text-sm uppercase tracking-wider text-muted-foreground">Recent Analyses</h3>
        <div className="flex items-center gap-2">
          {items.length > 0 && (
            <Button variant="outline" size="sm" className="h-6 px-2 text-[10px] uppercase font-bold" onClick={onClearAll}>
              Clear All
            </Button>
          )}
          <span className="text-xs font-mono bg-muted px-2 py-0.5 rounded-full text-foreground/70">{items.length} items</span>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
        {loading && items.length === 0 && (
          <div className="flex justify-center items-center h-40 opacity-50 space-x-2">
            <div className="w-2 h-2 rounded-full bg-primary animate-ping" />
            <span className="text-sm font-medium">Loading history...</span>
          </div>
        )}

        {!loading && items.length === 0 && (
          <div className="flex flex-col h-full justify-center items-center text-center p-6 text-muted-foreground opacity-60">
            <div className="w-12 h-12 bg-muted rounded-full flex items-center justify-center mb-3">🕒</div>
            <p className="text-sm">Nothing developed yet. Analyzed images will appear here for review.</p>
          </div>
        )}

        <ul className="space-y-3">
          {items.map((item) => (
            <li key={item.id}>
              <button
                className={`w-full text-left transition-all duration-200 rounded-lg border overflow-hidden group ${selectedId === item.id
                  ? "ring-2 ring-primary ring-offset-2 ring-offset-background border-primary bg-primary/5 shadow-md scale-[1.02] z-10 relative"
                  : "border-border bg-card hover:border-primary/50 hover:bg-accent/50"
                  }`}
                onClick={() => onSelect(item.id)}
              >
                <div className="flex justify-between items-center p-3 border-b bg-muted/10">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-muted-foreground bg-background border rounded px-1.5 font-semibold">
                      #{String(item.id).padStart(4, "0")}
                    </span>
                    <button
                      className="opacity-0 group-hover:opacity-100 transition-opacity hover:bg-destructive/20 hover:text-destructive rounded-full w-5 h-5 flex items-center justify-center text-muted-foreground"
                      title="Remove from history"
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(item.id);
                      }}
                    >
                      ✕
                    </button>
                  </div>
                  <StampBadge label={item.quality_label || item.recommended_action} size="sm" hideSubtitle />
                </div>

                <div className="p-3">
                  <p className="text-sm font-medium truncate text-foreground group-hover:text-primary transition-colors mb-2" title={item.filename}>
                    {item.filename}
                  </p>

                  <div className="flex justify-between items-center text-xs">
                    <div className="flex flex-col">
                      <span className="text-[10px] text-muted-foreground uppercase font-bold tracking-tight">Score</span>
                      <span className={`font-mono font-bold ${item.quality_score >= 78 ? 'text-green-500' : item.quality_score >= 45 ? 'text-amber-500' : 'text-red-500'}`}>
                        {item.quality_score.toFixed(0)}
                      </span>
                    </div>

                    <div className="flex flex-col items-end">
                      <span className="text-[10px] text-muted-foreground uppercase font-bold tracking-tight">Issues</span>
                      <span className="text-muted-foreground truncate max-w-[120px] font-medium text-[11px]" title={item.issues.map(i => i.type).join(", ")}>
                        {item.issues.length === 0 ? (
                          <span className="text-green-500 uppercase">Clean</span>
                        ) : (
                          item.issues.map(i => i.type.replace('_', ' ')).join(", ")
                        )}
                      </span>
                    </div>
                  </div>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
