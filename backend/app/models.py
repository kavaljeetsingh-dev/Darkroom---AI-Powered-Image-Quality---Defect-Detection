from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text

from app.database import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    image_path = Column(String(500), nullable=True)  # stored copy of the upload
    thumbnail_path = Column(String(500), nullable=True)

    quality_score = Column(Float, nullable=False)
    quality_label = Column(String(20), nullable=False)
    recommended_action = Column(String(20), nullable=True)  # PASS / REVIEW / REJECT
    issues = Column(JSON, nullable=False, default=list)
    image_stats = Column(JSON, nullable=False, default=dict)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    def to_dict(self, include_stats: bool = True) -> dict:
        d = {
            "id": self.id,
            "filename": self.filename,
            "content_type": self.content_type,
            "file_size_bytes": self.file_size_bytes,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "image_path": self.image_path,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "recommended_action": self.recommended_action,
            "issues": self.issues,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_stats:
            d["image_stats"] = self.image_stats
        return d
