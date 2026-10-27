"""Database repository for QA Agent system."""

import os
from urllib.parse import quote_plus
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from models.schemas import (
    Base,
    LearningOutcome,
    QAReport,
    KeywordAnalysis,
    FeedbackAuthenticity
)
from utils.logger import log
from config import config


class DatabaseRepository:
    """Database operations for QA system."""
    
    def __init__(self):
        """Initialize database repository."""
        # Build connection string
        db_config = config.database
        connection_string = os.getenv("DATABASE_URL")
        if not connection_string:
            # Check if postgres driver is available and valid password configured
            for driver, prefix in [("psycopg", "postgresql://"), ("psycopg2", "postgresql+psycopg2://")]:
                try:
                    __import__(driver)
                    pw = db_config.get('password')
                    if pw and str(pw) != "None" and str(pw).strip():
                        connection_string = (
                            f"{prefix}{db_config['user']}:{quote_plus(str(pw))}"
                            f"@{db_config['host']}:{db_config['port']}/{db_config['name']}"
                        )
                        break
                except ImportError:
                    continue

            if not connection_string:
                log.info("No external DATABASE_URL or postgres driver configured; using local SQLite database: sqlite:///dev.db")
                connection_string = "sqlite:///dev.db"

        # Create engine with fallback to SQLite if connection fails
        try:
            self.engine = self._create_db_engine(connection_string, db_config)
            if not connection_string.startswith("sqlite"):
                with self.engine.connect() as conn:
                    pass
        except Exception as e:
            if not connection_string.startswith("sqlite"):
                log.warning(f"Could not connect to database ({e}). Falling back to local SQLite: sqlite:///dev.db")
                connection_string = "sqlite:///dev.db"
                self.engine = self._create_db_engine(connection_string, db_config)
            else:
                raise
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,  # returned objects stay readable after commit
            bind=self.engine
        )
        
        # Auto-create tables on startup
        self.create_tables()
        log.info("DatabaseRepository initialized with tables verified")
    
    @staticmethod
    def _create_db_engine(connection_string: str, db_config: Dict[str, Any]):
        """Create SQLAlchemy engine with appropriate dialect arguments."""
        engine_kwargs = {"pool_pre_ping": True}
        if not connection_string.startswith("sqlite"):
            engine_kwargs["pool_size"] = db_config.get('pool_size', 10)
            engine_kwargs["max_overflow"] = db_config.get('max_overflow', 20)
        elif ":memory:" in connection_string:
            from sqlalchemy.pool import StaticPool
            engine_kwargs["poolclass"] = StaticPool
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        else:
            from pathlib import Path
            sqlite_path = connection_string.replace("sqlite:///", "")
            if sqlite_path and not sqlite_path.startswith(":"):
                Path(sqlite_path).resolve().parent.mkdir(parents=True, exist_ok=True)
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        return create_engine(connection_string, **engine_kwargs)

    def check_connection(self) -> bool:
        """Check if database is reachable for health checks."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            log.error(f"Database health check failed: {e}")
            return False

    def create_tables(self):
        """Create all tables in the database."""
        try:
            Base.metadata.create_all(bind=self.engine)
            log.info("Database tables created or verified successfully")
        except Exception as e:
            log.error(f"Error creating database tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """
        Context manager for database sessions.
        
        Usage:
            with repo.get_session() as session:
                # do database operations
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            log.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    # Learning Outcome operations
    
    def create_learning_outcome(
        self,
        session_id: str,
        skill_taught: str,
        progression_score: float,
        knowledge_statements_count: int,
        question_count: int,
        progression_graph: List[float]
    ) -> LearningOutcome:
        """Create learning outcome record."""
        with self.get_session() as session:
            session.query(LearningOutcome).filter(
                LearningOutcome.session_id == session_id,
                LearningOutcome.skill_taught == skill_taught
            ).delete()
            outcome = LearningOutcome(
                session_id=session_id,
                skill_taught=skill_taught,
                progression_score=progression_score,
                knowledge_statements_count=knowledge_statements_count,
                question_count=question_count,
                progression_graph=progression_graph
            )
            session.add(outcome)
            session.flush()
            session.refresh(outcome)
            
            log.info(f"Created learning outcome for session: {session_id}")
            return outcome
    
    def get_learning_outcome(
        self,
        session_id: str
    ) -> Optional[LearningOutcome]:
        """Get learning outcome by session ID."""
        with self.get_session() as session:
            return session.query(LearningOutcome).filter(
                LearningOutcome.session_id == session_id
            ).first()
    
    # QA Report operations
    
    def create_qa_report(
        self,
        report_data: Dict
    ) -> QAReport:
        """Create QA report record."""
        with self.get_session() as session:
            report = session.query(QAReport).filter(
                QAReport.session_id == report_data['session_id']
            ).first()
            if report:  # re-analysis of the same session: update in place
                for key, value in report_data.items():
                    setattr(report, key, value)
            else:
                report = QAReport(**report_data)
                session.add(report)
            session.flush()
            session.refresh(report)
            
            log.info(f"Created QA report for session: {report_data['session_id']}")
            return report
    
    def get_qa_report(
        self,
        session_id: str
    ) -> Optional[QAReport]:
        """Get QA report by session ID."""
        with self.get_session() as session:
            return session.query(QAReport).filter(
                QAReport.session_id == session_id
            ).first()
    
    def list_qa_reports(
        self,
        verdict: Optional[str] = None,
        recommendation: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[QAReport]:
        """List QA reports with optional filters and pagination."""
        with self.get_session() as session:
            query = session.query(QAReport)
            if verdict:
                query = query.filter(QAReport.qa_verdict == verdict)
            if recommendation:
                query = query.filter(QAReport.recommendation == recommendation)
            return query.order_by(QAReport.created_at.desc()).offset(offset).limit(limit).all()

    def get_qa_summary(self) -> Dict:
        """Aggregate statistical summary across all stored QA reports."""
        with self.get_session() as session:
            from sqlalchemy import func
            total = session.query(func.count(QAReport.qa_report_id)).scalar() or 0
            if total == 0:
                return {
                    "total_sessions_analyzed": 0,
                    "verdict_distribution": {},
                    "recommendation_distribution": {},
                    "average_scores": {}
                }
            
            verdict_counts = dict(
                session.query(QAReport.qa_verdict, func.count(QAReport.qa_report_id))
                .group_by(QAReport.qa_verdict).all()
            )
            rec_counts = dict(
                session.query(QAReport.recommendation, func.count(QAReport.qa_report_id))
                .group_by(QAReport.recommendation).all()
            )
            
            avg_overall = session.query(func.avg(QAReport.overall_quality_score)).scalar() or 0.0
            avg_scope = session.query(func.avg(QAReport.scope_completeness_score)).scalar() or 0.0
            avg_teaching = session.query(func.avg(QAReport.teaching_quality_score)).scalar() or 0.0
            avg_learning = session.query(func.avg(QAReport.learning_indicators_score)).scalar() or 0.0
            avg_auth = session.query(func.avg(QAReport.feedback_authenticity_score)).scalar() or 0.0
            avg_sat = session.query(func.avg(QAReport.participant_satisfaction_score)).scalar() or 0.0
            avg_comp = session.query(func.avg(QAReport.compliance_score)).scalar() or 0.0

            return {
                "total_sessions_analyzed": int(total),
                "verdict_distribution": {
                    str(k.value if hasattr(k, 'value') else k): int(v)
                    for k, v in verdict_counts.items()
                },
                "recommendation_distribution": {
                    str(k.value if hasattr(k, 'value') else k): int(v)
                    for k, v in rec_counts.items()
                },
                "average_scores": {
                    "overall_quality": round(float(avg_overall), 3),
                    "scope_completeness": round(float(avg_scope), 3),
                    "teaching_quality": round(float(avg_teaching), 3),
                    "learning_indicators": round(float(avg_learning), 3),
                    "feedback_authenticity": round(float(avg_auth), 3),
                    "participant_satisfaction": round(float(avg_sat), 3),
                    "compliance": round(float(avg_comp), 3),
                }
            }

    # Keyword Analysis operations
    
    def create_keyword_analysis(
        self,
        session_id: str,
        extracted_keywords: List[Dict],
        expected_keywords: List[str],
        coverage_percentage: float,
        missing_topics: List[str]
    ) -> KeywordAnalysis:
        """Create keyword analysis record."""
        with self.get_session() as session:
            analysis = session.query(KeywordAnalysis).filter(
                KeywordAnalysis.session_id == session_id
            ).first() or KeywordAnalysis(session_id=session_id)
            analysis.extracted_keywords = extracted_keywords
            analysis.expected_keywords = expected_keywords
            analysis.coverage_percentage = coverage_percentage
            analysis.missing_topics = missing_topics
            session.add(analysis)
            session.flush()
            session.refresh(analysis)
            
            log.info(f"Created keyword analysis for session: {session_id}")
            return analysis
    
    # Feedback Authenticity operations
    
    def create_feedback_authenticity(
        self,
        authenticity_data: Dict
    ) -> FeedbackAuthenticity:
        """Create feedback authenticity record."""
        with self.get_session() as session:
            authenticity = session.query(FeedbackAuthenticity).filter(
                FeedbackAuthenticity.session_id == authenticity_data['session_id']
            ).first()
            if authenticity:
                for key, value in authenticity_data.items():
                    setattr(authenticity, key, value)
            else:
                authenticity = FeedbackAuthenticity(**authenticity_data)
                session.add(authenticity)
            session.flush()
            session.refresh(authenticity)
            
            log.info(
                f"Created feedback authenticity for session: "
                f"{authenticity_data['session_id']}"
            )
            return authenticity
    
    # Utility operations
    
    def get_past_feedback_comments(
        self,
        session_id_to_exclude: Optional[str] = None,
        limit: int = 1000
    ) -> List[str]:
        """Get past feedback comments from database for plagiarism checking."""
        try:
            with self.get_session() as session:
                query = session.query(FeedbackAuthenticity.feedback_text).filter(
                    FeedbackAuthenticity.feedback_text.isnot(None),
                    FeedbackAuthenticity.feedback_text != ""
                )
                if session_id_to_exclude:
                    query = query.filter(FeedbackAuthenticity.session_id != session_id_to_exclude)
                
                results = query.order_by(FeedbackAuthenticity.created_at.desc()).limit(limit).all()
                comments = [r[0] for r in results if r[0]]
                log.debug(f"Retrieved {len(comments)} past feedback comments for comparison")
                return comments
        except Exception as e:
            log.warning(f"Error querying past feedback comments: {e}")
            return []
