"""Cross-validation between feedback and transcript."""

from typing import List, Dict, Tuple
from utils.embeddings import EmbeddingManager
from utils.nlp_utils import NLPUtils
from utils.logger import log


class CrossValidator:
    """Cross-validate feedback claims against transcript."""
    
    def __init__(self):
        """Initialize cross validator."""
        self.embedding_manager = EmbeddingManager()
        self.nlp_utils = NLPUtils()
        log.info("CrossValidator initialized")
    
    def extract_feedback_claims(
        self,
        feedback_text: str
    ) -> List[str]:
        """
        Extract verifiable claims from feedback.
        
        Args:
            feedback_text: Feedback text
            
        Returns:
            List of claim strings
        """
        # Split into sentences
        sentences = self.nlp_utils.tokenize_sentences(feedback_text)
        
        # Filter for factual claims (not just opinions)
        claim_indicators = [
            'explained', 'covered', 'taught', 'showed', 'demonstrated',
            'discussed', 'reviewed', 'went over', 'helped with',
            'answered', 'clarified', 'provided examples'
        ]
        
        claims = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in claim_indicators):
                claims.append(sentence)
        
        log.debug(f"Extracted {len(claims)} verifiable claims from feedback")
        
        return claims
    
    def validate_claim_against_transcript(
        self,
        claim: str,
        transcript: str,
        threshold: float = 0.6
    ) -> Tuple[float, List[str]]:
        """
        Validate a single claim against transcript.
        
        Args:
            claim: Claim to validate
            transcript: Session transcript
            threshold: Similarity threshold for evidence
            
        Returns:
            Tuple of (evidence_score, supporting_excerpts)
        """
        # Split transcript into searchable segments
        sentences = self.nlp_utils.tokenize_sentences(transcript)
        
        # Find most relevant segments
        relevant_segments = self.embedding_manager.semantic_search(
            claim,
            sentences,
            threshold=threshold
        )
        
        if not relevant_segments:
            return (0.0, [])
        
        # Calculate evidence score based on top matches
        top_scores = [seg['score'] for seg in relevant_segments[:3]]
        evidence_score = sum(top_scores) / len(top_scores) if top_scores else 0.0
        
        # Extract supporting excerpts
        supporting_excerpts = [
            seg['text']
            for seg in relevant_segments[:3]
        ]
        
        return (evidence_score, supporting_excerpts)
    
    def validate_feedback(
        self,
        feedback_text: str,
        transcript: str
    ) -> Dict[str, any]:
        """
        Validate feedback claims against transcript.
        
        Implementation of FEAT-6.4.6:
        - Takes feedback claims and transcript
        - For each feedback claim, searches transcript for evidence
        - Generates evidence_match_score (0.0-1.0) per claim
        
        Args:
            feedback_text: Feedback text with claims
            transcript: Session transcript
            
        Returns:
            Dictionary with validation results
        """
        log.info("Validating feedback against transcript")
        
        # Extract claims from feedback
        claims = self.extract_feedback_claims(feedback_text)
        
        if not claims:
            log.warning("No verifiable claims found in feedback")
            return {
                'overall_evidence_score': 0.5,
                'claim_validations': [],
                'validated_claims': 0,
                'total_claims': 0
            }
        
        # Validate each claim
        claim_validations = []
        total_score = 0.0
        
        for claim in claims:
            evidence_score, excerpts = self.validate_claim_against_transcript(
                claim,
                transcript
            )
            
            validation = {
                'claim': claim,
                'evidence_score': evidence_score,
                'is_validated': evidence_score >= 0.6,
                'supporting_excerpts': excerpts[:2]  # Top 2
            }
            
            claim_validations.append(validation)
            total_score += evidence_score
            
            log.debug(
                f"Claim validation: {claim[:50]}... "
                f"-> score={evidence_score:.2f}"
            )
        
        # Calculate overall evidence match score
        overall_score = total_score / len(claims) if claims else 0.0
        validated_count = sum(
            1 for v in claim_validations if v['is_validated']
        )
        
        result = {
            'overall_evidence_score': overall_score,
            'claim_validations': claim_validations,
            'validated_claims': validated_count,
            'total_claims': len(claims),
            'validation_rate': validated_count / len(claims) if claims else 0.0
        }
        
        log.info(
            f"Feedback validation complete: "
            f"{validated_count}/{len(claims)} claims validated "
            f"(overall score: {overall_score:.2f})"
        )
        
        return result
