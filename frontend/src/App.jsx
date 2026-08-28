import { useEffect, useRef, useState, useCallback } from "react";
import UploadPanel from "./components/UploadPanel.jsx";
import ReportCard from "./components/ReportCard.jsx";
import HistoryStrip from "./components/HistoryStrip.jsx";
import StatusBadge from "./components/StatusBadge.jsx";
import { analyzeImage, listResults, getResult, checkHealth, imageUrl, clearResults, deleteResult } from "./api/client.js";
import { Card } from "./components/ui/card.jsx";

export default function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const objectUrlRef = useRef(null);

  const refreshHistory = useCallback(async () => {
    setHistoryLoading(true);
    try {
      const data = await listResults({ limit: 16 });
      setHistory(data.results);
    } catch (e) {
      console.error(e);
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    // Wipe backend persistence entirely upon hard refresh per user instruction
    clearResults()
      .catch(console.error)
      .finally(() => refreshHistory());

    checkHealth().then(setHealth).catch(() => setHealth({ status: "unreachable" }));
  }, [refreshHistory]);

  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
    };
  }, []);

  const handleFileSelected = async (file) => {
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);

    if (!file) {
      setPreview(null);
      setResult(null);
      setStatus("idle");
      setErrorMsg("");
      return;
    }

    const url = URL.createObjectURL(file);
    objectUrlRef.current = url;
    setPreview(url);
    setResult(null);
    setStatus("loading");
    setErrorMsg("");
    try {
      const data = await analyzeImage(file);
      setResult(data);
      setStatus("success");
      refreshHistory();
    } catch (e) {
      setStatus("error");
      setErrorMsg(e.message || "Analysis failed.");
    }
  };

  const handleSelectHistoryItem = async (id) => {
    setStatus("loading");
    setErrorMsg("");
    try {
      const data = await getResult(id);
      setResult(data);
      setPreview(data.image_path ? imageUrl(data.image_path) : null);
      setStatus("success");
    } catch (e) {
      setStatus("error");
      setErrorMsg(e.message || "Could not load that report.");
    }
  };

  const handleClearAllHistory = async () => {
    try {
      await clearResults();
      await refreshHistory();
      if (result) await handleFileSelected(null);
    } catch (e) {
      alert("Failed to clear history: " + e.message);
      console.error(e);
    }
  };

  const handleDeleteHistoryItem = async (id) => {
    try {
      await deleteResult(id);
      await refreshHistory();
      if (result && result.id === id) {
        await handleFileSelected(null);
      }
    } catch (e) {
      alert("Failed to delete item: " + e.message);
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-background text-foreground font-sans selection:bg-primary/20">
      <div className="fixed inset-0 pointer-events-none opacity-0 mix-blend-overlay z-50 bg-background" />

      <header className="sticky top-0 z-40 w-full border-b bg-background/80 backdrop-blur-md">
        <div className="container mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-serif text-xl">
              ◐
            </div>
            <div>
              <h1 className="text-xl font-bold leading-tight tracking-tight">Darkroom</h1>
              <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-medium">Automated Inspection</p>
            </div>
          </div>
          <StatusBadge health={health} />
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 grid grid-cols-1 lg:grid-cols-12 gap-8 relative z-10 w-full max-w-7xl">
        <div className="lg:col-span-8 flex flex-col gap-8">
          <UploadPanel
            preview={preview}
            status={status}
            onFileSelected={handleFileSelected}
          />

          {status === "error" && (
            <Card className="border-destructive/50 bg-destructive/10 p-4 text-destructive flex flex-col gap-1">
              <strong className="font-semibold">Couldn't process that image.</strong>
              <span className="text-sm opacity-90">{errorMsg}</span>
            </Card>
          )}

          {result && status === "success" && (
            <div className="animate-in slide-in-from-bottom-4 duration-500 fade-in">
              <ReportCard result={result} />
            </div>
          )}
        </div>

        <aside className="lg:col-span-4 h-full">
          <HistoryStrip
            items={history}
            loading={historyLoading}
            onSelect={handleSelectHistoryItem}
            selectedId={result?.id}
            onClearAll={handleClearAllHistory}
            onDelete={handleDeleteHistoryItem}
          />
        </aside>
      </main>
    </div>
  );
}
