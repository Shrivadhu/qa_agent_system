"""Database models and schemas for QA Agent system."""

from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from enum import Enum
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, 
    ForeignKey, JSON, Text, Enum as SQLEnum
)
from sqlalchemy.orm import relationship, declarative_base
from pydantic import BaseModel, Field, ConfigDict

Base = declarative_base()


def get_utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class QAVerdict(str, Enum):
    """QA verdict types."""
    SUCCESSFUL = "successful"
    PARTIAL = "partial"
    DISPUTE = "dispute"
    FAILED = "failed"


class Recommendation(str, Enum):
    """QA recommendations."""
    SETTLE = "settle"
    REVIEW = "review"
    ESCALATE = "escalate"


# SQLAlchemy Models

class LearningOutcome(Base):
    """Learning outcome database model."""
    __tablename__ = "learning_outcomes"
    
    learning_outcome_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, index=True)
    skill_taught = Column(String(200), nullable=False)
    progression_score = Column(Float, nullable=False)
    knowledge_statements_count = Column(Integer, default=0)
    question_count = Column(Integer, default=0)
    progression_graph = Column(JSON)  # Array of scores over time
    created_at = Column(DateTime, default=get_utc_now)
    
    qa_report = relationship("QAReport", back_populates="learning_outcome")


class QAReport(Base):
    """QA report database model."""
    __tablename__ = "qa_reports"
    
    qa_report_id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    qa_verdict = Column(SQLEnum(QAVerdict), nullable=False)
    overall_quality_score = Column(Float, nullable=False)
    component_scores = Column(JSON, nullable=False)  # Dict of component scores
    evidence = Column(Text)  # Summary of analysis
    recommendation = Column(SQLEnum(Recommendation), nullable=False)
    
    # Component details
    scope_completeness_score = Column(Float)
    teaching_quality_score = Column(Float)
    learning_indicators_score = Column(Float)
    feedback_authenticity_score = Column(Float)
    participant_satisfaction_score = Column(Float)
    compliance_score = Column(Float)
    
    # Foreign keys
    learning_outcome_id = Column(
        Integer, 
        ForeignKey('learning_outcomes.learning_outcome_id')
    )
    
    created_at = Column(DateTime, default=get_utc_now)
    updated_at = Column(DateTime, default=get_utc_now, onupdate=get_utc_now)
    
    # Relationships
    learning_outcome = relationship("LearningOutcome", back_populates="qa_report")


class KeywordAnalysis(Base):
    """Keyword analysis cache."""
    __tablename__ = "keyword_analyses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    extracted_keywords = Column(JSON)  # List of {keyword, frequency}
    expected_keywords = Column(JSON)  # List of expected keywords
    coverage_percentage = Column(Float)
    missing_topics = Column(JSON)  # List of missing topics
    created_at = Column(DateTime, default=get_utc_now)


class FeedbackAuthenticity(Base):
    """Feedback authenticity analysis."""
    __tablename__ = "feedback_authenticity"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(100), nullable=False, unique=True, index=True)
    feedback_text = Column(Text, nullable=True)  # Stores feedback for plagiarism checking
    sentiment = Column(String(50))  # positive/neutral/negative
    sentiment_score = Column(Float)
    consistency_score = Column(Float)
    plagiarism_score = Column(Float)
    template_match_score = Column(Float)
    human_likeness_score = Column(Float)
    evidence_match_score = Column(Float)
    overall_authenticity_score = Column(Float)
    created_at = Column(DateTime, default=get_utc_now)


# Pydantic Schemas (for API requests/responses)

class TeachingQualityDimensions(BaseModel):
    """Teaching quality dimension scores."""
    clarity: float = Field(ge=0.0, le=1.0)
    engagement: float = Field(ge=0.0, le=1.0)
    patience: float = Field(ge=0.0, le=1.0)
    structure: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)


class KeywordExtractionResult(BaseModel):
    """Keyword extraction result."""
    keywords: List[Dict[str, Any]]  # [{keyword, frequency}]
    total_keywords: int
    extraction_method: str


class CoverageReport(BaseModel):
    """Objective coverage report."""
    coverage_percentage: float
    covered_topics: List[str]
    missing_topics: List[str]
    expected_keyword_count: int
    found_keyword_count: int


class LearningProgressionResult(BaseModel):
    """Learning progression analysis result."""
    progression_score: float
    knowledge_statements_count: int
    question_count: int
    progression_graph: List[float]
    evidence: List[str]


class FeedbackAuthenticityResult(BaseModel):
    """Feedback authenticity analysis result."""
    sentiment: str
    sentiment_score: float
    consistency_score: float
    plagiarism_score: float
    template_match_score: float
    human_likeness_score: float
    evidence_match_score: float
    overall_authenticity_score: float
    flags: List[str]


class ComponentScores(BaseModel):
    """QA component scores."""
    scope_completeness: float = Field(ge=0.0, le=1.0)
    teaching_quality: float = Field(ge=0.0, le=1.0)
    learning_indicators: float = Field(ge=0.0, le=1.0)
    feedback_authenticity: float = Field(ge=0.0, le=1.0)
    participant_satisfaction: float = Field(ge=0.0, le=1.0)
    compliance: float = Field(ge=0.0, le=1.0)


class QAReportResponse(BaseModel):
    """QA report API response."""
    qa_report_id: int
    session_id: str
    qa_verdict: QAVerdict
    overall_quality_score: float
    component_scores: ComponentScores
    evidence: str
    recommendation: Recommendation
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SessionAnalysisRequest(BaseModel):
    """Request model for session analysis."""
    session_id: str
    transcript: str
    promised_skills: List[str]
    feedback_data: Dict[str, Any]
    participant_ratings: Optional[Dict[str, float]] = None
    skill_keywords: Optional[Dict[str, List[str]]] = None


class SkillsResponse(BaseModel):
    """Response model for /skills endpoint."""
    skills: List[str]
    taxonomy: Dict[str, List[str]]


class BatchSessionAnalysisRequest(BaseModel):
    """Request model for batch session analysis."""
    sessions: List[SessionAnalysisRequest]


class BatchSessionAnalysisResponse(BaseModel):
    """Response model for batch session analysis."""
    total: int
    successful: int
    failed: int
    results: List[QAReportResponse]
    errors: List[Dict[str, Any]]


class QASummaryResponse(BaseModel):
    """Response model for QA aggregate statistics."""
    total_sessions_analyzed: int
    verdict_distribution: Dict[str, int]
    recommendation_distribution: Dict[str, int]
    average_scores: Dict[str, float]
