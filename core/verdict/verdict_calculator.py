"""QA verdict calculation module."""

from typing import Dict
from models.schemas import (
    QAVerdict,
    Recommendation,
    ComponentScores
)
from utils.logger import log
from config import config


class VerdictCalculator:
    """Calculate QA verdict based on component scores."""
    
    def __init__(self):
        """Initialize verdict calculator."""
        self.thresholds = config.get("verdict.thresholds", {
            "successful": 0.90,
            "partial": 0.70,
            "dispute": 0.50
        })
        
        self.component_weights = config.get("verdict.component_weights", {
            "scope_completeness": 0.167,
            "teaching_quality": 0.167,
            "learning_indicators": 0.167,
            "feedback_authenticity": 0.167,
            "participant_satisfaction": 0.166,
            "compliance": 0.166
        })
        
        log.info("VerdictCalculator initialized")
    
    def aggregate_qa_score(
        self,
        component_scores: ComponentScores
    ) -> float:
        """
        Aggregate component scores into overall QA score.
        
        Implementation of FEAT-6.5.1:
        - Components:
          * Scope completeness (0.0-1.0)
          * Teaching quality (0.0-1.0)
          * Learning indicators (0.0-1.0)
          * Feedback authenticity (0.0-1.0)
          * Participant satisfaction (0.0-1.0)
          * Compliance (0.0-1.0)
        - Equal weighting (1/6 each) or configured weights
        - Overall QA_score = weighted average of components
        
        Args:
            component_scores: ComponentScores object
            
        Returns:
            Overall QA score (0.0-1.0)
        """
        log.info("Aggregating QA component scores")
        
        # Extract scores
        scores_dict = {
            'scope_completeness': component_scores.scope_completeness,
            'teaching_quality': component_scores.teaching_quality,
            'learning_indicators': component_scores.learning_indicators,
            'feedback_authenticity': component_scores.feedback_authenticity,
            'participant_satisfaction': component_scores.participant_satisfaction,
            'compliance': component_scores.compliance
        }
        
        # Log individual scores
        for component, score in scores_dict.items():
            log.debug(f"{component}: {score:.3f}")
        
        # Calculate weighted average
        overall_score = sum(
            score * self.component_weights.get(component, 1/6)
            for component, score in scores_dict.items()
        )
        
        log.info(f"Overall QA score: {overall_score:.3f}")
        
        return overall_score
    
    def calculate_verdict(
        self,
        qa_score: float
    ) -> QAVerdict:
        """
        Calculate verdict based on QA score.
        
        Implementation of FEAT-6.5.2:
        - QA_score >= 0.90: SUCCESSFUL
        - QA_score 0.70-0.89: PARTIAL
        - QA_score 0.50-0.69: DISPUTE
        - QA_score < 0.50: FAILED
        
        Args:
            qa_score: Overall QA score
            
        Returns:
            QA verdict
        """
        if qa_score >= self.thresholds['successful']:
            verdict = QAVerdict.SUCCESSFUL
        elif qa_score >= self.thresholds['partial']:
            verdict = QAVerdict.PARTIAL
        elif qa_score >= self.thresholds['dispute']:
            verdict = QAVerdict.DISPUTE
        else:
            verdict = QAVerdict.FAILED
        
        log.info(f"QA Verdict: {verdict.value} (score: {qa_score:.3f})")
        
        return verdict
    
    def generate_recommendation(
        self,
        verdict: QAVerdict,
        component_scores: ComponentScores
    ) -> Recommendation:
        """
        Generate recommendation based on verdict and scores.
        
        Args:
            verdict: QA verdict
            component_scores: Component scores
            
        Returns:
            Recommendation (settle/review/escalate)
        """
        # Automatic settlement for successful sessions
        if verdict == QAVerdict.SUCCESSFUL:
            return Recommendation.SETTLE
        
        # Check for any critically low scores
        critical_threshold = 0.4
        has_critical_issues = any([
            component_scores.scope_completeness < critical_threshold,
            component_scores.teaching_quality < critical_threshold,
            component_scores.feedback_authenticity < critical_threshold
        ])
        
        if verdict == QAVerdict.FAILED or has_critical_issues:
            return Recommendation.ESCALATE
        
        # Partial or dispute verdicts need review
        return Recommendation.REVIEW
    
    def generate_evidence_summary(
        self,
        verdict: QAVerdict,
        component_scores: ComponentScores,
        analysis_details: Dict[str, any] = None
    ) -> str:
        """
        Generate evidence summary for QA report.
        
        Args:
            verdict: QA verdict
            component_scores: Component scores
            analysis_details: Optional detailed analysis data
            
        Returns:
            Evidence summary string
        """
        evidence_parts = []
        
        # Overall verdict
        evidence_parts.append(
            f"Session received a {verdict.value.upper()} verdict."
        )
        
        # Component analysis
        scores_dict = {
            'Scope Completeness': component_scores.scope_completeness,
            'Teaching Quality': component_scores.teaching_quality,
            'Learning Indicators': component_scores.learning_indicators,
            'Feedback Authenticity': component_scores.feedback_authenticity,
            'Participant Satisfaction': component_scores.participant_satisfaction,
            'Compliance': component_scores.compliance
        }
        
        # Identify strengths and weaknesses
        strengths = [
            name for name, score in scores_dict.items()
            if score >= 0.8
        ]
        
        weaknesses = [
            name for name, score in scores_dict.items()
            if score < 0.6
        ]
        
        if strengths:
            evidence_parts.append(
                f"Strong performance in: {', '.join(strengths)}."
            )
        
        if weaknesses:
            evidence_parts.append(
                f"Areas needing improvement: {', '.join(weaknesses)}."
            )
        
        # Add specific findings if available
        if analysis_details:
            if 'missing_topics' in analysis_details:
                missing = analysis_details['missing_topics']
                if missing and len(missing) > 0:
                    evidence_parts.append(
                        f"Missing topics: {', '.join(missing[:3])}."
                    )
            
            if 'authenticity_flags' in analysis_details:
                flags = analysis_details['authenticity_flags']
                if flags:
                    evidence_parts.append(
                        f"Feedback concerns: {', '.join(flags)}."
                    )
            
            if 'compliance_flags' in analysis_details:
                c_flags = analysis_details['compliance_flags']
                if c_flags:
                    evidence_parts.append(
                        f"Compliance violations: {', '.join(c_flags)}."
                    )
        
        return " ".join(evidence_parts)
    
    def create_qa_report(
        self,
        session_id: str,
        component_scores: ComponentScores,
        analysis_details: Dict[str, any] = None
    ) -> Dict[str, any]:
        """
        Create complete QA report.
        
        Implementation of FEAT-6.5.3 (QA report schema):
        - qa_report_id (auto-generated by database)
        - session_id
        - qa_verdict (successful/partial/dispute/failed)
        - overall_quality_score (0.0-1.0)
        - component_scores (object)
        - evidence (string - summary of analysis)
        - recommendation (settle/review/escalate)
        
        Args:
            session_id: Session identifier
            component_scores: Component scores
            analysis_details: Optional detailed analysis
            
        Returns:
            Dictionary matching QAReport schema
        """
        log.info(f"Creating QA report for session: {session_id}")
        
        # Calculate overall score and verdict
        overall_score = self.aggregate_qa_score(component_scores)
        verdict = self.calculate_verdict(overall_score)
        recommendation = self.generate_recommendation(verdict, component_scores)
        
        # Generate evidence summary
        evidence = self.generate_evidence_summary(
            verdict,
            component_scores,
            analysis_details
        )
        
        # Create report dictionary
        report = {
            'session_id': session_id,
            'qa_verdict': verdict,
            'overall_quality_score': overall_score,
            'component_scores': component_scores.model_dump(),
            'evidence': evidence,
            'recommendation': recommendation,
            
            # Individual component scores (for database columns)
            'scope_completeness_score': component_scores.scope_completeness,
            'teaching_quality_score': component_scores.teaching_quality,
            'learning_indicators_score': component_scores.learning_indicators,
            'feedback_authenticity_score': component_scores.feedback_authenticity,
            'participant_satisfaction_score': component_scores.participant_satisfaction,
            'compliance_score': component_scores.compliance
        }
        
        log.info(
            f"QA Report created: {verdict.value} "
            f"(score: {overall_score:.2f}, "
            f"recommendation: {recommendation.value})"
        )
        
        return report
