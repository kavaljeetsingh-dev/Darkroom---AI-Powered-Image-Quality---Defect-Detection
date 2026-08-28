from __future__ import annotations

import os
import time
import uuid
import logging
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text

from app.database import get_db, init_db, engine
from app.models import AnalysisResult
from app.schemas import AnalysisResponse, PaginatedResults, HealthResponse
from app.utils.image_validation import validate_and_decode, ImageValidationError
from app.ml.infer import get_model, reset_model, QualityModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("image-quality-api")

APP_VERSION = "1.0.0"

DATA_DIR = Path(os.environ.get("APP_DATA_DIR", Path(__file__).resolve().parent / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    title="AI-Powered Image Quality & Defect Detection API",
    description=(
        "Upload an image to detect blur, exposure problems, noise, "
        "corruption, and other visual defects using a hybrid "
        "engineered-features + trained-classifier pipeline."
    ),
    version=APP_VERSION,
)

# E10: CORS — don't combine allow_credentials=True with wildcard origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

_model_load_error: str | None = None


@app.on_event("startup")
def on_startup():
    init_db()
    global _model_load_error
    try:
        reset_model()  # clear stale singleton on --reload restarts
        if not QualityModel.artifacts_available():
            raise RuntimeError(
                "Trained model artifacts not found in app/ml/artifacts/. "
                "Run `python -m app.ml.train` first (see README)."
            )
        get_model()  # warm the singleton so first request isn't slow
        logger.info("ML model artifacts loaded successfully.")
    except Exception as e:  # pragma: no cover
        _model_load_error = str(e)
        logger.error(f"Failed to load ML model: {e}")


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
def health():
    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
    except Exception:
        db_ok = False
    return HealthResponse(
        status="ok" if _model_load_error is None and db_ok else "degraded",
        model_loaded=_model_load_error is None,
        database_reachable=db_ok,
        version=APP_VERSION,
    )


@app.post("/api/analyze", response_model=AnalysisResponse, tags=["analysis"])
async def analyze_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if _model_load_error:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {_model_load_error}")

    raw_bytes = await file.read()

    try:
        img_bgr = validate_and_decode(raw_bytes, file.content_type, file.filename)
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during validation/decoding")
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    try:
        model = get_model()
        result = model.analyze(img_bgr, include_heatmap=True)
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    # persist a copy of the upload for later viewing in history
    ext = Path(file.filename or "upload.jpg").suffix or ".jpg"
    stored_name = f"{uuid.uuid4().hex}{ext}"
    stored_path = UPLOAD_DIR / stored_name
    try:
        cv2.imwrite(str(stored_path), img_bgr)
    except Exception:
        stored_path = None
        logger.warning("Failed to persist uploaded image copy (continuing without it).")

    h, w = img_bgr.shape[:2]
    record = AnalysisResult(
        filename=file.filename or "unknown",
        content_type=file.content_type,
        file_size_bytes=len(raw_bytes),
        image_width=w,
        image_height=h,
        image_path=f"/static/uploads/{stored_name}" if stored_path else None,
        quality_score=result["quality_score"],
        quality_label=result["quality_label"],
        recommended_action=result["recommended_action"],  # E2
        issues=result["issues"],
        image_stats=result["image_stats"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    response = record.to_dict()
    response["blur_heatmap_png_base64"] = result.get("blur_heatmap_png_base64")
    return response


@app.get("/api/results", response_model=PaginatedResults, tags=["history"])
def list_results(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    quality_label: str | None = Query(None, description="Filter by ACCEPTABLE/DEGRADED/DEFECTIVE"),
    db: Session = Depends(get_db),
):
    q = db.query(AnalysisResult)
    if quality_label:
        q = q.filter(AnalysisResult.quality_label == quality_label.upper())
    total = q.count()
    rows = (
        q.order_by(AnalysisResult.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "results": [r.to_dict(include_stats=False) for r in rows],
    }


@app.get("/api/results/{result_id}", response_model=AnalysisResponse, tags=["history"])
def get_result(result_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"No analysis result with id={result_id}")
    d = record.to_dict()

    # E12: Regenerate heatmap from the stored image if available
    heatmap = None
    if record.image_path:
        img_path = UPLOAD_DIR.parent.parent / record.image_path.lstrip("/static/")
        # Resolve the actual path from the stored relative URL
        actual_path = DATA_DIR / "uploads" / Path(record.image_path).name
        if actual_path.exists():
            try:
                img_bgr = cv2.imread(str(actual_path))
                if img_bgr is not None:
                    model = get_model()
                    heatmap = model._render_blur_heatmap(img_bgr)
            except Exception:
                logger.warning(f"Failed to regenerate heatmap for result {result_id}")

    d["blur_heatmap_png_base64"] = heatmap
    return d


@app.delete("/api/results/{result_id}", tags=["history"])
def delete_result(result_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"No analysis result with id={result_id}")
    db.delete(record)
    db.commit()
    return {"deleted": result_id}
