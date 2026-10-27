"""Feedback authenticity checking module."""

from typing import List, Dict, Tuple
import re
from utils.embeddings import EmbeddingManager
from utils.nlp_utils import NLPUtils
from utils.logger import log
from config import config


class AuthenticityChecker:
    """Check authenticity of feedback text."""
    
    def __init__(self):
        """Initialize authenticity checker."""
        self.embedding_manager = EmbeddingManager()
        self.nlp_utils = NLPUtils()
        
        # Load common feedback templates
        self.feedback_templates = self._load_templates()
        
        log.info("AuthenticityChecker initialized")
    
    def _load_templates(self) -> List[str]:
        """Load common feedback templates."""
        # In production, this would load from a database
        return [
            "great session! learned a lot. very helpful and patient.",
            "excellent teacher! explained concepts clearly. highly recommend.",
            "very good session. covered all topics. thank you.",
            "helpful session. good explanations. would recommend.",
            "nice session. learned new things. teacher was patient.",
            "good teaching. clear explanations. satisfied with session.",
        ]
    
    def check_plagiarism(
        self,
        feedback_text: str,
        past_comments: List[str] = None
    ) -> Tuple[float, bool]:
        """
        Detect plagiarized feedback.
        
        Implementation of FEAT-6.4.3:
        - Takes feedback free_text_comment
        - Queries database for similar past comments
        - Uses cosine similarity on embeddings
        - Returns plagiarism score (0.0-1.0)
        - Threshold: > 0.90 flags as suspicious
        
        Args:
            feedback_text: Feedback to check
            past_comments: List of historical feedback comments
            
        Returns:
            Tuple of (plagiarism_score, is_suspicious)
        """
        log.debug("Checking for plagiarism")
        
        if not feedback_text or len(feedback_text.strip()) < 10:
            return (0.0, False)
        
        if not past_comments:
            # No historical data to compare
            log.debug("No past comments to check against")
            return (0.0, False)
        
        # Find most similar past comment
        similar_comments = self.embedding_manager.find_most_similar(
            feedback_text,
            past_comments,
            top_k=1
        )
        
        if not similar_comments:
            return (0.0, False)
        
        plagiarism_score = similar_comments[0]['score']
        threshold = config.get(
            "feedback_validation.plagiarism_threshold",
            0.90
        )
        
        is_suspicious = plagiarism_score > threshold
        
        if is_suspicious:
            log.warning(
                f"Potential plagiarism detected: "
                f"similarity={plagiarism_score:.3f} with past comment"
            )
        
        return (plagiarism_score, is_suspicious)
    
    def check_template_match(
        self,
        feedback_text: str
    ) -> Tuple[float, bool]:
        """
        Detect template-based feedback.
        
        Implementation of FEAT-6.4.4:
        - Maintains list of common feedback templates
        - Checks input against templates
        - Generates template match score
        - Flags high matches
        
        Args:
            feedback_text: Feedback to check
            
        Returns:
            Tuple of (template_score, is_templated)
        """
        log.debug("Checking for template usage")
        
        if not feedback_text or len(feedback_text.strip()) < 10:
            return (0.0, False)
        
        # Calculate similarity with each template
        similarities = [
            self.embedding_manager.cosine_similarity(
                feedback_text.lower(),
                template
            )
            for template in self.feedback_templates
        ]
        
        # Get highest similarity
        template_score = max(similarities) if similarities else 0.0
        
        threshold = config.get(
            "feedback_validation.template_threshold",
            0.85
        )
        
        is_templated = template_score > threshold
        
        if is_templated:
            log.warning(
                f"Template-based feedback detected: "
                f"match_score={template_score:.3f}"
            )
        
        return (template_score, is_templated)
    
    def calculate_human_likeness(
        self,
        feedback_text: str
    ) -> Tuple[float, Dict[str, any]]:
        """
        Score human-likeness of feedback.
        
        Implementation of FEAT-6.4.5:
        - Evaluates:
          * Sentence length variance
          * Word choice naturalness
          * Typos/grammar (natural for human)
          * Specific details mentioned
        - Generates human-likeness score (0.0-1.0)
        
        Args:
            feedback_text: Feedback to analyze
            
        Returns:
            Tuple of (human_likeness_score, metrics_dict)
        """
        log.debug("Calculating human-likeness")
        
        if not feedback_text or len(feedback_text.strip()) < 10:
            return (0.5, {})
        
        metrics = {}
        
        # 1. Sentence length variance
        variance = self.nlp_utils.calculate_sentence_length_variance(
            feedback_text
        )
        min_variance = config.get(
            "feedback_validation.human_likeness.min_sentence_length_variance",
            0.15
        )
        variance_score = min(1.0, variance / min_variance) if variance > 0 else 0.5
        metrics['sentence_variance'] = variance
        metrics['variance_score'] = variance_score
        
        # 2. Specific details count
        details_count = self.nlp_utils.count_specific_details(feedback_text)
        min_details = config.get(
            "feedback_validation.human_likeness.min_specific_details",
            2
        )
        details_score = min(1.0, details_count / max(min_details, 1))
        metrics['specific_details'] = details_count
        metrics['details_score'] = details_score
        
        # 3. Natural typos/imperfections (very basic check)
        # Human feedback often has minor imperfections
        has_lowercase_start = bool(re.search(r'\.\s+[a-z]', feedback_text))
        has_repeated_punctuation = bool(re.search(r'[!?.]{2,}', feedback_text))
        has_informal_words = any(
            word in feedback_text.lower()
            for word in ['really', 'very', 'pretty', 'kinda', 'quite']
        )
        
        naturalness_indicators = sum([
            has_lowercase_start,
            has_repeated_punctuation,
            has_informal_words,
            variance > 0.1,
            details_count > 0
        ])
        naturalness_score = naturalness_indicators / 5.0
        metrics['naturalness_score'] = naturalness_score
        
        # 4. Not too perfect (overly formal is suspicious)
        sentences = self.nlp_utils.tokenize_sentences(feedback_text)
        too_formal = all(
            s[0].isupper() and s.strip().endswith(('.', '!', '?'))
            for s in sentences if len(s.strip()) > 0
        )
        formality_penalty = 0.2 if too_formal and len(sentences) > 2 else 0.0
        
        # Calculate overall human-likeness
        human_likeness_score = (
            variance_score * 0.3 +
            details_score * 0.3 +
            naturalness_score * 0.4 -
            formality_penalty
        )
        human_likeness_score = max(0.0, min(1.0, human_likeness_score))
        
        metrics['overall'] = human_likeness_score
        
        log.debug(f"Human-likeness score: {human_likeness_score:.3f}")
        
        return (human_likeness_score, metrics)
    
    def aggregate_authenticity_checks(
        self,
        plagiarism_score: float,
        template_score: float,
        human_likeness_score: float,
        consistency_score: float,
        evidence_match_score: float = None
    ) -> float:
        """
        Aggregate all authenticity checks.
        
        Implementation of FEAT-6.4.7:
        - Aggregates:
          * Sentiment consistency score
          * Plagiarism score (inverted)
          * Template match score (inverted)
          * Human-likeness score
          * Evidence match score
        - Weighted average
        - Final authenticity_score (0.0-1.0)
        
        Args:
            plagiarism_score: Score from plagiarism check
            template_score: Score from template check
            human_likeness_score: Score from human-likeness check
            consistency_score: Score from sentiment consistency
            evidence_match_score: Score from transcript evidence cross-validation
            
        Returns:
            Overall authenticity score (0.0-1.0)
        """
        # Invert plagiarism and template scores (high is bad)
        plagiarism_authenticity = 1.0 - plagiarism_score
        template_authenticity = 1.0 - template_score
        
        # Weighted average
        if evidence_match_score is not None:
            weights = {
                'consistency': 0.20,
                'plagiarism': 0.20,
                'template': 0.20,
                'human_likeness': 0.20,
                'evidence_match': 0.20
            }
            authenticity_score = (
                consistency_score * weights['consistency'] +
                plagiarism_authenticity * weights['plagiarism'] +
                template_authenticity * weights['template'] +
                human_likeness_score * weights['human_likeness'] +
                evidence_match_score * weights['evidence_match']
            )
        else:
            weights = {
                'consistency': 0.25,
                'plagiarism': 0.25,
                'template': 0.20,
                'human_likeness': 0.30
            }
            authenticity_score = (
                consistency_score * weights['consistency'] +
                plagiarism_authenticity * weights['plagiarism'] +
                template_authenticity * weights['template'] +
                human_likeness_score * weights['human_likeness']
            )
        
        authenticity_score = max(0.0, min(1.0, authenticity_score))
        log.info(f"Overall authenticity score: {authenticity_score:.3f}")
        
        return authenticity_score
