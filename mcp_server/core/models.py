from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolError(BaseModel):
    error: str = "tool_error"
    message: str
    detail: Any = None


class CandidateFile(BaseModel):
    file_name: str
    file_path: str
    file_format: str
    description: str = ""
    time_range: str | None = None
    match_score: float = 0.0
    match_reason: str = ""


class TargetLocation(BaseModel):
    file_name: str
    section_id: str
    heading: str = ""
    estimated_relevance: float = Field(default=0.0, ge=0.0, le=1.0)


class EvidenceItem(BaseModel):
    evidence_id: str
    source_file: str
    source_location: str
    source_format: str
    evidence_type: str
    evidence_strength: str
    content: str
    time_period: str | None = None
    numeric_value: float | None = None
    unit: str | None = None
    topic_category: str | None = None
