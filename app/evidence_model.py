from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    TREND_ANALYSIS = "trend_analysis"
    CROSS_DOC_SUMMARY = "cross_doc_summary"
    NUMERIC_QUERY = "numeric_query"
    DEFINITION_PROCESS = "definition_process"


class EvidenceStrength(str, Enum):
    DIRECT = "direct"
    DERIVED = "derived"
    OPINION = "opinion"


class QueryPlan(BaseModel):
    original_question: str
    query_type: QueryType
    keywords: list[str]
    time_range: Optional[str] = None
    target_formats: list[str] = Field(default_factory=list)
    expected_dimensions: list[str] = Field(default_factory=list)
    initial_file_count: int = 5


class CandidateFile(BaseModel):
    file_name: str
    file_path: str
    file_format: str
    description: str = ""
    match_score: float = 0.0
    match_reason: str = ""
    time_range: Optional[str] = None


class TargetLocation(BaseModel):
    file_name: str
    section_id: str
    heading: str = ""
    estimated_relevance: float = 0.0


class Evidence(BaseModel):
    evidence_id: str
    source_file: str
    source_location: str
    source_format: str
    evidence_type: str
    evidence_strength: EvidenceStrength
    content: str
    time_period: Optional[str] = None
    numeric_value: Optional[float] = None
    unit: Optional[str] = None
    topic_category: Optional[str] = None
    dedup_key: Optional[str] = None
    extraction_quality: str = "good"
