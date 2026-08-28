from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


class IssueSchema(BaseModel):
    type: str
    severity: str
    confidence: Optional[float] = None  # None for anomaly-detector issues (E4)
    anomaly_score: Optional[float] = None  # raw IsolationForest score (E4)
    description: str
    evidence: dict[str, float] = {}


class AnalysisResponse(BaseModel):
    id: int
    filename: str
    quality_score: float
    quality_label: str
    recommended_action: Optional[str] = None  # PASS / REVIEW / REJECT (E2)
    image_path: Optional[str] = None  # server-stored image path (E1)
    issues: list[IssueSchema]
    image_stats: dict[str, float] = {}
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    created_at: str
    blur_heatmap_png_base64: Optional[str] = None


class AnalysisSummary(BaseModel):
    id: int
    filename: str
    quality_score: float
    quality_label: str
    recommended_action: Optional[str] = None  # E2
    image_path: Optional[str] = None  # E1
    issues: list[IssueSchema]
    created_at: str


class PaginatedResults(BaseModel):
    total: int
    limit: int
    offset: int
    results: list[AnalysisSummary]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    database_reachable: bool
    version: str
