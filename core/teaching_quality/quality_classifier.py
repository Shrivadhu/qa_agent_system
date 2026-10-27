"""Teaching quality classification module."""

from typing import List, Dict
import numpy as np
from utils.logger import log
from utils.nlp_utils import NLPUtils
from models.schemas import TeachingQualityDimensions
from config import config


class TeachingQualityClassifier:
    """Classify teaching quality across multiple dimensions."""
    
    def __init__(self):
        """Initialize teaching quality classifier."""
        self.nlp_utils = NLPUtils()
        
        # Load model (would be fine-tuned BERT in production)
        model_name = config.get("models.bert_base", "bert-base-uncased")
        log.info(f"Loading model for teaching quality: {model_name}")
        
        # For demo, we'll use rule-based scoring
        # In production, this would load a fine-tuned model
        self.use_ml_model = False
        
        if self.use_ml_model:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=5  # One for each dimension
            )
        
        log.info("TeachingQualityClassifier initialized")
    
    def evaluate_segment(
        self,
        segment: str
    ) -> TeachingQualityDimensions:
        """
        Evaluate teaching quality for a transcript segment.
        
        Implementation of FEAT-6.2.1:
        - Takes transcript segment as input
        - Evaluates on dimensions:
          * Clarity (0.0-1.0): Are explanations clear?
          * Engagement (0.0-1.0): Is learner engaged?
          * Patience (0.0-1.0): How patient is teacher?
          * Structure (0.0-1.0): Is lesson organized?
          * Relevance (0.0-1.0): Is teaching relevant to level?
        
        Args:
            segment: Transcript segment to evaluate
            
        Returns:
            TeachingQualityDimensions with scores
        """
        if self.use_ml_model:
            return self._evaluate_with_model(segment)
        else:
            return self._evaluate_with_rules(segment)
    
    def _evaluate_with_rules(
        self,
        segment: str
    ) -> TeachingQualityDimensions:
        """
        Rule-based evaluation (fallback/demo implementation).
        
        Args:
            segment: Transcript segment
            
        Returns:
            TeachingQualityDimensions
        """
        # Clarity indicators
        clarity_indicators = [
            "for example", "for instance", "this means", "in other words",
            "let me explain", "to clarify", "specifically", "simply put",
            "that is", "as an illustration", "in particular", "step by step",
            "to break it down", "let's look at", "lets look at"
        ]
        clarity_score = self._score_indicators(
            segment,
            clarity_indicators,
            base_score=0.7
        )
        
        # Engagement indicators
        engagement_indicators = [
            "what do you think", "can you tell me", "try it", "your turn",
            "let's practice", "lets practice", "good question", "excellent",
            "that's right", "thats right", "keep going", "how would you",
            "give it a try", "give it a shot", "notice how", "what happens if",
            "would you like to try"
        ]
        engagement_score = self._score_indicators(
            segment,
            engagement_indicators,
            base_score=0.6
        )
        
        # Patience indicators (positive reinforcement increases score, impatience reduces)
        impatience_indicators = [
            "hurry up", "quickly", "should be obvious",
            "come on", "simple", "just do it"
        ]
        patience_boost_indicators = [
            "take your time", "no rush", "don't worry", "no problem",
            "that's fine", "completely fine", "good attempt", "great try",
            "almost there", "take as much time"
        ]
        patience_penalty = self._score_indicators(
            segment,
            impatience_indicators,
            base_score=0.0,
            is_negative=True
        )
        patience_boost = self._score_indicators(
            segment,
            patience_boost_indicators,
            base_score=0.0,
            is_negative=True
        )
        patience_score = max(0.0, min(1.0, 0.8 + patience_boost - patience_penalty))
        
        # Structure indicators
        structure_indicators = [
            "first", "second", "third", "next", "then", "finally",
            "to summarize", "in conclusion", "let's start with", "lets start with",
            "moving on to", "now that we've covered", "to recap", "in overview",
            "as a next step"
        ]
        structure_score = self._score_indicators(
            segment,
            structure_indicators,
            base_score=0.65
        )
        
        # Relevance indicators (checking for jargon, appropriate examples)
        relevance_score = self._assess_relevance(segment)
        
        return TeachingQualityDimensions(
            clarity=clarity_score,
            engagement=engagement_score,
            patience=patience_score,
            structure=structure_score,
            relevance=relevance_score
        )
    
    def _score_indicators(
        self,
        text: str,
        indicators: List[str],
        base_score: float,
        is_negative: bool = False
    ) -> float:
        """Score based on presence of indicator phrases."""
        text_lower = text.lower()
        
        count = sum(1 for indicator in indicators if indicator in text_lower)
        
        if count == 0:
            return base_score
        
        # Increase score based on indicator count (diminishing returns)
        boost = min(0.3, count * 0.05)
        
        if is_negative:
            return boost
        else:
            return min(1.0, base_score + boost)
    
    def _assess_relevance(self, text: str) -> float:
        """Assess relevance of content."""
        # Check for overly technical jargon vs. clear language
        words = self.nlp_utils.tokenize_words(text.lower())
        
        # Count complex words (>10 characters)
        complex_words = sum(1 for w in words if len(w) > 10)
        complexity_ratio = complex_words / max(len(words), 1)
        
        # Moderate complexity is good (0.05-0.15)
        if 0.05 <= complexity_ratio <= 0.15:
            return 0.85
        elif complexity_ratio < 0.05:
            return 0.75  # Too simple
        else:
            return max(0.4, 0.85 - (complexity_ratio - 0.15) * 2)
    
    def _evaluate_with_model(
        self,
        segment: str
    ) -> TeachingQualityDimensions:
        """
        ML model-based evaluation (production implementation).
        
        Args:
            segment: Transcript segment
            
        Returns:
            TeachingQualityDimensions
        """
        import torch

        # Tokenize
        inputs = self.tokenizer(
            segment,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            scores = torch.sigmoid(logits).squeeze().numpy()
        
        return TeachingQualityDimensions(
            clarity=float(scores[0]),
            engagement=float(scores[1]),
            patience=float(scores[2]),
            structure=float(scores[3]),
            relevance=float(scores[4])
        )
    
    def evaluate_transcript(
        self,
        transcript: str,
        segment_size: int = 500
    ) -> List[TeachingQualityDimensions]:
        """
        Evaluate entire transcript by segments.
        
        Args:
            transcript: Full transcript
            segment_size: Number of words per segment
            
        Returns:
            List of quality dimensions for each segment
        """
        log.info("Evaluating teaching quality for entire transcript")
        
        # Split into segments
        # Whitespace split keeps contractions ("let's", "that's") intact so
        # phrase indicators can match
        words = transcript.split()
        segments = []
        
        for i in range(0, len(words), segment_size):
            segment_words = words[i:i + segment_size]
            segment = " ".join(segment_words)
            segments.append(segment)
        
        log.info(f"Split transcript into {len(segments)} segments")
        
        # Evaluate each segment
        results = []
        if not segments:
            return results
        for i, segment in enumerate(segments):
            log.debug(f"Evaluating segment {i+1}/{len(segments)}")
            quality = self.evaluate_segment(segment)
            results.append(quality)
        
        return results
