# API Reference

Base URL (local): `http://localhost:8000`
Interactive docs: `/docs` (Swagger) and `/redoc` once the backend is running.

---

## `POST /api/analyze`

Upload an image and run the full quality analysis. Persists the result.

**Request**: `multipart/form-data`, field name `file`.

```bash
curl -F "file=@sample_images/good_quality.jpg" http://localhost:8000/api/analyze
```

**Response** `200 OK`:
```json
{
  "id": 1,
  "filename": "good_quality.jpg",
  "quality_score": 91.2,
  "quality_label": "ACCEPTABLE",
  "issues": [],
  "image_stats": {
    "sharpness_lap_var": 812.4,
    "sharpness_norm": 0.14,
    "brightness_mean": 118.3,
    "brightness_std": 74.9,
    "dark_pixel_ratio": 0.02,
    "bright_pixel_ratio": 0.01,
    "hist_low_mass": 0.05,
    "hist_high_mass": 0.03,
    "noise_estimate": 1.8,
    "noise_local_var": 1.2,
    "edge_density": 0.08,
    "colorfulness": 55.1,
    "saturation_mean": 88.4,
    "entropy": 7.5,
    "blockiness": 0.02,
    "channel_mean_asymmetry": 12.3
  },
  "image_width": 600,
  "image_height": 400,
  "created_at": "2026-08-27T13:53:44.788307",
  "blur_heatmap_png_base64": "iVBORw0KGgoAAAANSUhEUgAA..."
}
```

**Response with detected issues** (`quality_label` other than `ACCEPTABLE`):
```json
{
  "id": 2,
  "filename": "blurry.jpg",
  "quality_score": 17.3,
  "quality_label": "DEFECTIVE",
  "issues": [
    {
      "type": "blur",
      "severity": "high",
      "confidence": 0.999,
      "description": "Image lacks sharp edges / fine detail, consistent with defocus or motion blur.",
      "evidence": {
        "sharpness_lap_var": 3.6242,
        "sharpness_norm": 0.0012,
        "edge_density": 0.0031
      }
    }
  ],
  "image_width": 600,
  "image_height": 400,
  "created_at": "2026-08-27T13:53:44.946568"
}
```

**Errors**:
| Status | Cause |
|---|---|
| `400` | Empty file, unreadable/corrupt image, unsupported format, dimensions outside 16px–8000px, file over 15MB |
| `503` | ML model failed to load on the server |
| `500` | Unexpected inference error |

```bash
curl -F "file=@not_an_image.txt" http://localhost:8000/api/analyze
# -> 400 {"detail": "File is not a valid or is a corrupted image: ..."}
```

---

## `GET /api/results`

Paginated history of past analyses (most recent first).

**Query params**: `limit` (1–100, default 20), `offset` (default 0),
`quality_label` (optional filter: `ACCEPTABLE` | `DEGRADED` | `DEFECTIVE`).

```bash
curl "http://localhost:8000/api/results?limit=5&quality_label=DEFECTIVE"
```

```json
{
  "total": 12,
  "limit": 5,
  "offset": 0,
  "results": [
    {
      "id": 12,
      "filename": "corrupted_defective.jpg",
      "quality_score": 32.1,
      "quality_label": "DEFECTIVE",
      "issues": [ { "type": "corruption", "severity": "high", "confidence": 0.98, "description": "...", "evidence": {} } ],
      "created_at": "2026-08-27T13:53:45.616328"
    }
  ]
}
```

---

## `GET /api/results/{id}`

Fetch one previously stored analysis (full detail, including all image
stats). The blur heatmap is only returned at analyze-time (not persisted),
so `blur_heatmap_png_base64` is `null` here.

```bash
curl http://localhost:8000/api/results/1
```

**404** if no analysis with that id exists.

---

## `DELETE /api/results/{id}`

```bash
curl -X DELETE http://localhost:8000/api/results/1
# -> {"deleted": 1}
```

---

## `GET /api/health`

```bash
curl http://localhost:8000/api/health
```
```json
{
  "status": "ok",
  "model_loaded": true,
  "database_reachable": true,
  "version": "1.0.0"
}
```
