const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function handle(res) {
  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch (_) {
      /* ignore parse errors */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function analyzeImage(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    body: form,
  });
  return handle(res);
}

export async function listResults({ limit = 12, offset = 0, qualityLabel = null } = {}) {
  const params = new URLSearchParams({ limit, offset });
  if (qualityLabel) params.set("quality_label", qualityLabel);
  const res = await fetch(`${API_BASE}/api/results?${params.toString()}`);
  return handle(res);
}

export async function getResult(id) {
  const res = await fetch(`${API_BASE}/api/results/${id}`);
  return handle(res);
}

export async function deleteResult(id) {
  const res = await fetch(`${API_BASE}/api/results/${id}`, { method: "DELETE" });
  return handle(res);
}

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/api/health`);
  return handle(res);
}

export function imageUrl(path) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}
