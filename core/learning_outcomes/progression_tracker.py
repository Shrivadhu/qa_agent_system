"""Learning progression tracking module."""

from typing import List, Dict, Tuple
import numpy as np
from utils.nlp_utils import NLPUtils
from utils.logger import log
from models.schemas import LearningProgressionResult


class ProgressionTracker:
    """Track knowledge progression during learning session."""
    
    def __init__(self):
        """Initialize progression tracker."""
        self.nlp_utils = NLPUtils()
        log.info("ProgressionTracker initialized")
    
    def detect_progression(
        self,
        transcript: str,
        skill_being_taught: str
    ) -> LearningProgressionResult:
        """
        Detect knowledge progression throughout session.
        
        Implementation of FEAT-6.3.1:
        - Takes transcript and skill being taught
        - Detects:
          * Questions asked (learner confusion)
          * Statements of understanding
          * Demonstration of knowledge (examples given)
        - Tracks progression from start to end
        - Generates progression graph
        - Scores: did learner progress? (0.0-1.0)
        
        Args:
            transcript: Full session transcript
            skill_being_taught: Skill that was taught
            
        Returns:
            LearningProgressionResult with progression metrics
        """
        log.info(f"Detecting progression for skill: {skill_being_taught}")
        
        # Split transcript into time-ordered segments
        sentences = self.nlp_utils.tokenize_sentences(transcript)
        num_segments = min(10, max(3, len(sentences) // 10))
        segment_size = len(sentences) // num_segments
        
        segments = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(sentences)
            segment = " ".join(sentences[start_idx:end_idx])
            segments.append(segment)
        
        # Analyze each segment
        progression_scores = []
        questions_count = 0
        understanding_count = 0
        evidence_list = []
        
        for i, segment in enumerate(segments):
            # Detect questions (confusion indicators)
            questions = self.nlp_utils.detect_questions(segment)
            questions_count += len(questions)
            
            # Detect understanding statements
            understanding_stmts = self.nlp_utils.detect_statements_of_understanding(
                segment
            )
            understanding_count += len(understanding_stmts)
            
            # Calculate segment score
            # Early segments: questions are expected (negative contribution)
            # Later segments: understanding statements are positive
            time_factor = (i + 1) / num_segments
            
            question_penalty = len(questions) * 0.1 * (1 - time_factor)
            understanding_bonus = len(understanding_stmts) * 0.2
            
            # Base score increases over time (learning naturally progresses)
            base_score = 0.3 + (0.4 * time_factor)
            
            segment_score = base_score + understanding_bonus - question_penalty
            segment_score = max(0.0, min(1.0, segment_score))
            
            progression_scores.append(segment_score)
            
            # Collect evidence
            if understanding_stmts:
                evidence_list.append(
                    f"Segment {i+1}: Understanding expressed - "
                    f"{understanding_stmts[0][:100]}..."
                )
        
        # Calculate overall progression score
        # Check if there's upward trend
        if len(progression_scores) > 1:
            # Calculate linear regression slope
            x = np.arange(len(progression_scores))
            slope = np.polyfit(x, progression_scores, 1)[0]
            
            # Positive slope indicates progression
            progression_trend = max(0.0, min(1.0, 0.5 + slope * 5))
        else:
            progression_trend = 0.5
        
        # Combine end score and trend
        final_score = progression_scores[-1] if progression_scores else 0.0
        overall_progression = (final_score * 0.6 + progression_trend * 0.4)
        
        # Create result
        result = LearningProgressionResult(
            progression_score=overall_progression,
            knowledge_statements_count=understanding_count,
            question_count=questions_count,
            progression_graph=progression_scores,
            evidence=evidence_list[:5]  # Top 5 pieces of evidence
        )
        
        log.info(
            f"Progression detected: score={overall_progression:.2f}, "
            f"questions={questions_count}, understanding={understanding_count}"
        )
        
        return result
    
    def analyze_learner_engagement(
        self,
        transcript: str
    ) -> Dict[str, any]:
        """
        Analyze learner engagement patterns.
        
        Args:
            transcript: Session transcript
            
        Returns:
            Dictionary with engagement metrics
        """
        # Identify learner vs teacher speaking patterns
        # This would require speaker diarization in production
        
        questions = self.nlp_utils.detect_questions(transcript)
        understanding = self.nlp_utils.detect_statements_of_understanding(transcript)
        
        # Count interactive phrases
        interactive_phrases = [
            "i tried", "let me try", "i think", "my answer",
            "i would", "can i", "should i"
        ]
        
        transcript_lower = transcript.lower()
        interaction_count = sum(
            transcript_lower.count(phrase)
            for phrase in interactive_phrases
        )
        
        engagement_score = min(1.0, (
            len(questions) * 0.1 +
            len(understanding) * 0.15 +
            interaction_count * 0.05
        ))
        
        return {
            'engagement_score': engagement_score,
            'question_count': len(questions),
            'understanding_count': len(understanding),
            'interaction_count': interaction_count
        }
    
    def create_learning_outcome_record(
        self,
        session_id: str,
        skill_taught: str,
        progression_result: LearningProgressionResult
    ) -> Dict[str, any]:
        """
        Create learning outcome record for database.
        
        Implementation of FEAT-6.3.2 (schema creation).
        
        Args:
            session_id: Session identifier
            skill_taught: Skill that was taught
            progression_result: Result from detect_progression
            
        Returns:
            Dictionary matching LearningOutcome schema
        """
        return {
            'session_id': session_id,
            'skill_taught': skill_taught,
            'progression_score': progression_result.progression_score,
            'knowledge_statements_count': progression_result.knowledge_statements_count,
            'question_count': progression_result.question_count,
            'progression_graph': progression_result.progression_graph
        }
