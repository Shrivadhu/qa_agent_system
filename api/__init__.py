"""API module for QA Agent system."""

from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status, Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import (
    SessionAnalysisRequest,
    QAReportResponse,
    SkillsResponse,
    ComponentScores,
    BatchSessionAnalysisRequest,
    BatchSessionAnalysisResponse,
    QASummaryResponse
)
from .qa_service import QAService
from utils.logger import log
from config import config

# Initialize QA service
qa_service = QAService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure database tables exist on server startup."""
    try:
        qa_service.db.create_tables()
        log.info("Database tables initialized successfully on startup")
    except Exception as e:
        log.error(f"Error during database startup initialization: {e}")
    yield


# Create FastAPI application
app = FastAPI(
    title="QA Agent Post-Session Analysis API",
    description="API for automated quality assurance of teaching sessions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
origins = config.cors_origins
allow_credentials = False if "*" in origins else True
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security scheme (X-API-Key header)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Optional[str] = Security(api_key_header)) -> Optional[str]:
    """
    Verify X-API-Key header.
    If API_KEY is set in environment or config, request must provide matching key.
    If API_KEY is unset or empty, access is unrestricted.
    """
    expected_key = config.api_key
    if not expected_key:
        return None  # Unrestricted access when no API key configured
    if not api_key or api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "QA Agent Post-Session Analysis API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health")
def health_check():
    """Health check endpoint that verifies database connection."""
    db_ok = qa_service.db.check_connection()
    if not db_ok:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unhealthy", "database": "disconnected"}
        )
    return {"status": "healthy", "database": "connected"}


@app.get(
    "/skills",
    response_model=SkillsResponse
)
def get_skills(_: Optional[str] = Depends(verify_api_key)) -> SkillsResponse:
    """Get list of supported skills and taxonomy keywords."""
    taxonomy = qa_service.coverage_checker.skill_taxonomy
    return SkillsResponse(
        skills=list(taxonomy.keys()),
        taxonomy=taxonomy
    )


@app.post(
    "/analyze",
    response_model=QAReportResponse,
    status_code=status.HTTP_200_OK
)
def analyze_session(
    request: SessionAnalysisRequest,
    _: Optional[str] = Depends(verify_api_key)
) -> QAReportResponse:
    """
    Analyze a teaching session and generate QA report.
    
    Args:
        request: SessionAnalysisRequest with session data
        
    Returns:
        QAReportResponse with complete analysis
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        log.info(f"Received analysis request for session: {request.session_id}")
        
        # Perform analysis
        result = qa_service.analyze_session(request)
        
        return result
        
    except Exception as e:
        log.error(f"Error analyzing session {request.session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get(
    "/report/{session_id}",
    response_model=QAReportResponse
)
def get_report(
    session_id: str,
    _: Optional[str] = Depends(verify_api_key)
) -> QAReportResponse:
    """
    Get existing QA report for a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        QAReportResponse
        
    Raises:
        HTTPException: If report not found
    """
    try:
        report = qa_service.db.get_qa_report(session_id)
        
        if not report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report not found for session: {session_id}"
            )
        
        component_scores = ComponentScores(
            scope_completeness=report.scope_completeness_score,
            teaching_quality=report.teaching_quality_score,
            learning_indicators=report.learning_indicators_score,
            feedback_authenticity=report.feedback_authenticity_score,
            participant_satisfaction=report.participant_satisfaction_score,
            compliance=report.compliance_score
        )
        
        response = QAReportResponse(
            qa_report_id=report.qa_report_id,
            session_id=report.session_id,
            qa_verdict=report.qa_verdict,
            overall_quality_score=report.overall_quality_score,
            component_scores=component_scores,
            evidence=report.evidence,
            recommendation=report.recommendation,
            created_at=report.created_at
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Error retrieving report for {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve report: {str(e)}"
        )


@app.post(
    "/analyze/batch",
    response_model=BatchSessionAnalysisResponse,
    status_code=status.HTTP_200_OK
)
def analyze_sessions_batch(
    request: BatchSessionAnalysisRequest,
    _: Optional[str] = Depends(verify_api_key)
) -> BatchSessionAnalysisResponse:
    """
    Analyze multiple teaching sessions in a single batch request.
    Handles individual session errors gracefully so partial batches succeed.
    """
    results = []
    errors = []
    log.info(f"Processing batch analysis for {len(request.sessions)} sessions")

    for session_req in request.sessions:
        try:
            res = qa_service.analyze_session(session_req)
            results.append(res)
        except Exception as e:
            log.error(f"Batch item failed for {session_req.session_id}: {e}")
            errors.append({"session_id": session_req.session_id, "error": str(e)})

    return BatchSessionAnalysisResponse(
        total=len(request.sessions),
        successful=len(results),
        failed=len(errors),
        results=results,
        errors=errors
    )


@app.get(
    "/reports",
    response_model=List[QAReportResponse]
)
def list_reports(
    verdict: Optional[str] = None,
    recommendation: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    _: Optional[str] = Depends(verify_api_key)
) -> List[QAReportResponse]:
    """List QA reports with optional filtering by verdict or recommendation, and pagination."""
    try:
        db_reports = qa_service.db.list_qa_reports(
            verdict=verdict,
            recommendation=recommendation,
            limit=limit,
            offset=offset
        )
        responses = []
        for r in db_reports:
            component_scores = ComponentScores(
                scope_completeness=r.scope_completeness_score,
                teaching_quality=r.teaching_quality_score,
                learning_indicators=r.learning_indicators_score,
                feedback_authenticity=r.feedback_authenticity_score,
                participant_satisfaction=r.participant_satisfaction_score,
                compliance=r.compliance_score
            )
            responses.append(QAReportResponse(
                qa_report_id=r.qa_report_id,
                session_id=r.session_id,
                qa_verdict=r.qa_verdict,
                overall_quality_score=r.overall_quality_score,
                component_scores=component_scores,
                evidence=r.evidence,
                recommendation=r.recommendation,
                created_at=r.created_at
            ))
        return responses
    except Exception as e:
        log.error(f"Error listing reports: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list reports: {str(e)}"
        )


@app.get(
    "/reports/summary",
    response_model=QASummaryResponse
)
def get_reports_summary(
    _: Optional[str] = Depends(verify_api_key)
) -> QASummaryResponse:
    """Retrieve aggregate QA analytics and score distributions across all sessions."""
    try:
        summary_data = qa_service.db.get_qa_summary()
        return QASummaryResponse(**summary_data)
    except Exception as e:
        log.error(f"Error generating QA summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate QA summary: {str(e)}"
        )


def start_server():
    """Start the API server."""
    import uvicorn
    
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    workers = config.get("api.workers", 4)
    
    log.info(f"Starting QA Agent API server on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )


__all__ = ['app', 'qa_service', 'start_server', 'verify_api_key']
