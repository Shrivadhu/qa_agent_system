"""Main QA Service orchestrator."""

from typing import Dict, List, Optional, Tuple
from models.schemas import (
    SessionAnalysisRequest,
    ComponentScores,
    QAReportResponse,
    FeedbackAuthenticityResult
)
from core import (
    KeywordExtractor,
    CoverageChecker,
    TeachingQualityClassifier,
    QualityAggregator,
    ProgressionTracker,
    SentimentAnalyzer,
    AuthenticityChecker,
    CrossValidator,
    VerdictCalculator,
    ComplianceChecker
)
from database import DatabaseRepository
from utils.logger import log


class QAService:
    """Main QA service for analyzing teaching sessions."""
    
    def __init__(self):
        """Initialize QA service with all components."""
        log.info("Initializing QA Service...")
        
        # Initialize all components
        self.keyword_extractor = KeywordExtractor()
        self.coverage_checker = CoverageChecker()
        self.quality_classifier = TeachingQualityClassifier()
        self.quality_aggregator = QualityAggregator()
        self.progression_tracker = ProgressionTracker()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.authenticity_checker = AuthenticityChecker()
        self.cross_validator = CrossValidator()
        self.verdict_calculator = VerdictCalculator()
        self.compliance_checker = ComplianceChecker()
        
        # Database repository
        self.db = DatabaseRepository()
        
        log.info("QA Service initialized successfully")
    
    def analyze_session(
        self,
        request: SessionAnalysisRequest
    ) -> QAReportResponse:
        """
        Perform complete QA analysis on a session.
        
        This orchestrates all phases of the QA analysis:
        1. Objective verification (keyword extraction, coverage)
        2. Teaching quality assessment
        3. Learning outcome verification
        4. Feedback validation
        5. Verdict calculation
        
        Args:
            request: SessionAnalysisRequest with all session data
            
        Returns:
            QAReportResponse with complete analysis
        """
        log.info(f"Starting QA analysis for session: {request.session_id}")
        details: Dict = {}
        
        # Phase 1: Objective Verification
        log.info("Phase 1: Objective Verification")
        scope_score = self._analyze_objectives(
            request.session_id,
            request.transcript,
            request.promised_skills,
            details,
            request.skill_keywords
        )
        
        # Phase 2: Teaching Quality Assessment
        log.info("Phase 2: Teaching Quality Assessment")
        quality_score = self._analyze_teaching_quality(request.transcript, details)
        
        # Phase 3: Learning Outcome Verification
        log.info("Phase 3: Learning Outcome Verification")
        learning_score = self._analyze_learning_outcomes(
            request.session_id,
            request.transcript,
            request.promised_skills
        )
        
        # Phase 4: Feedback Validation
        log.info("Phase 4: Feedback Validation")
        feedback_score, satisfaction_score = self._analyze_feedback(
            request.session_id,
            request.feedback_data,
            request.transcript,
            request.participant_ratings,
            details
        )
        
        # Phase 5: Compliance
        log.info("Phase 5: Compliance Checking")
        compliance_score, compliance_violations = self.compliance_checker.check_compliance(
            request.transcript
        )
        if compliance_violations:
            details['compliance_flags'] = compliance_violations
        
        # Create component scores (clamped: ComponentScores enforces 0.0-1.0)
        def clamp(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        component_scores = ComponentScores(
            scope_completeness=clamp(scope_score),
            teaching_quality=clamp(quality_score),
            learning_indicators=clamp(learning_score),
            feedback_authenticity=clamp(feedback_score),
            participant_satisfaction=clamp(satisfaction_score),
            compliance=clamp(compliance_score)
        )
        
        # Phase 6: Calculate Verdict
        log.info("Phase 5: Calculating Verdict")
        qa_report_data = self.verdict_calculator.create_qa_report(
            session_id=request.session_id,
            component_scores=component_scores,
            analysis_details=details
        )
        
        # Save to database
        db_report = self.db.create_qa_report(qa_report_data)
        
        # Convert to response
        response = QAReportResponse(
            qa_report_id=db_report.qa_report_id,
            session_id=db_report.session_id,
            qa_verdict=db_report.qa_verdict,
            overall_quality_score=db_report.overall_quality_score,
            component_scores=component_scores,
            evidence=db_report.evidence,
            recommendation=db_report.recommendation,
            created_at=db_report.created_at
        )
        
        log.info(
            f"QA analysis complete for session {request.session_id}: "
            f"{response.qa_verdict.value} "
            f"(score: {response.overall_quality_score:.2f})"
        )
        
        return response
    
    def _analyze_objectives(
        self,
        session_id: str,
        transcript: str,
        promised_skills: List[str],
        details: Dict,
        custom_skill_keywords: Optional[Dict[str, List[str]]] = None
    ) -> float:
        """Analyze objective coverage."""
        # Extract keywords
        extraction_result = self.keyword_extractor.extract_keywords(
            transcript,
            max_keywords=100
        )
        
        # Generate expected keywords
        expected_keywords = self.coverage_checker.generate_expected_keywords(
            promised_skills,
            custom_skill_keywords=custom_skill_keywords
        )
        
        # Check coverage
        coverage_report = self.coverage_checker.check_coverage(
            extraction_result.keywords,
            expected_keywords,
            transcript=transcript
        )
        
        # Store analysis
        details['coverage'] = coverage_report
        details['missing_topics'] = coverage_report.missing_topics
        self.db.create_keyword_analysis(
            session_id=session_id,
            extracted_keywords=extraction_result.keywords,
            expected_keywords=expected_keywords,
            coverage_percentage=coverage_report.coverage_percentage,
            missing_topics=coverage_report.missing_topics
        )
        
        # Calculate score
        score = self.coverage_checker.calculate_scope_completeness_score(
            coverage_report
        )
        
        log.info(f"Objective coverage score: {score:.2f}")
        return score
    
    def _analyze_teaching_quality(
        self,
        transcript: str,
        details: Dict
    ) -> float:
        """Analyze teaching quality."""
        # Evaluate transcript segments
        segment_scores = self.quality_classifier.evaluate_transcript(transcript)
        
        # Aggregate scores
        quality_assessment = self.quality_aggregator.get_quality_assessment(
            segment_scores
        )
        
        # Store analysis
        details['teaching_quality'] = quality_assessment
        
        score = quality_assessment['scores']['overall']
        
        log.info(f"Teaching quality score: {score:.2f}")
        return score
    
    def _analyze_learning_outcomes(
        self,
        session_id: str,
        transcript: str,
        skills: List[str]
    ) -> float:
        """Analyze learning outcomes."""
        # Track progression for each skill
        total_score = 0.0
        
        for skill in skills:
            progression_result = self.progression_tracker.detect_progression(
                transcript,
                skill
            )
            
            # Store in database
            outcome_data = self.progression_tracker.create_learning_outcome_record(
                session_id,
                skill,
                progression_result
            )
            self.db.create_learning_outcome(**outcome_data)
            
            total_score += progression_result.progression_score
        
        # Average score across all skills
        score = total_score / len(skills) if skills else 0.0
        
        log.info(f"Learning outcomes score: {score:.2f}")
        return score
    
    def _analyze_feedback(
        self,
        session_id: str,
        feedback_data: Dict,
        transcript: str,
        ratings: Optional[Dict[str, float]],
        details: Dict
    ) -> Tuple[float, float]:
        """Analyze feedback authenticity and satisfaction."""
        feedback_text = feedback_data.get('free_text_comment', '')
        
        if not feedback_text:
            log.warning("No feedback text provided")
            return (0.5, 0.5)
        
        # Sentiment analysis
        sentiment, sentiment_score = self.sentiment_analyzer.analyze_sentiment(
            feedback_text
        )
        
        # Tone consistency check
        if ratings:
            consistency_score, inconsistent = self.sentiment_analyzer.check_tone_consistency(
                sentiment,
                sentiment_score,
                ratings
            )
        else:
            consistency_score = 0.7
            inconsistent = False
        
        # Authenticity checks
        past_comments = self.db.get_past_feedback_comments(session_id_to_exclude=session_id)
        plagiarism_score, plagiarism_flag = self.authenticity_checker.check_plagiarism(
            feedback_text,
            past_comments
        )
        
        template_score, template_flag = self.authenticity_checker.check_template_match(
            feedback_text
        )
        
        human_likeness_score, _ = self.authenticity_checker.calculate_human_likeness(
            feedback_text
        )
        
        # Cross-validation with transcript
        validation_result = self.cross_validator.validate_feedback(
            feedback_text,
            transcript
        )
        evidence_score = validation_result['overall_evidence_score']
        
        # Aggregate authenticity
        authenticity_score = self.authenticity_checker.aggregate_authenticity_checks(
            plagiarism_score,
            template_score,
            human_likeness_score,
            consistency_score,
            evidence_match_score=evidence_score
        )
        
        flags = []
        if plagiarism_flag:
            flags.append("possible copied feedback")
        if template_flag:
            flags.append("matches a common template")
        if inconsistent:
            flags.append("tone inconsistent with ratings")
        if flags:
            details['authenticity_flags'] = flags

        # Store authenticity analysis
        authenticity_data = {
            'session_id': session_id,
            'feedback_text': feedback_text,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'consistency_score': consistency_score,
            'plagiarism_score': plagiarism_score,
            'template_match_score': template_score,
            'human_likeness_score': human_likeness_score,
            'evidence_match_score': evidence_score,
            'overall_authenticity_score': authenticity_score
        }
        self.db.create_feedback_authenticity(authenticity_data)
        
        # Participant satisfaction (from ratings or sentiment)
        if ratings:
            satisfaction_score = sum(ratings.values()) / (len(ratings) * 5.0)
            satisfaction_score = max(0.0, min(1.0, satisfaction_score))
        else:
            satisfaction_score = sentiment_score
        
        log.info(
            f"Feedback authenticity: {authenticity_score:.2f}, "
            f"Satisfaction: {satisfaction_score:.2f}"
        )
        
        return (authenticity_score, satisfaction_score)
