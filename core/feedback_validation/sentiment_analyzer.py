"""Sentiment analysis for feedback validation."""

from typing import Dict, Tuple
import re
from utils.logger import log
from config import config


class SentimentAnalyzer:
    """Analyze sentiment in feedback text."""
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        model_name = config.get(
            "models.sentiment_model",
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        log.info(f"Loading sentiment model: {model_name}")
        
        self.sentiment_pipeline = None
        try:
            from transformers import pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name
            )
        except Exception as e:  # missing package or no network for model download
            log.warning(
                f"Could not load sentiment model '{model_name}' ({e}). "
                f"Falling back to a simple lexicon - less accurate."
            )
        
        log.info("SentimentAnalyzer initialized")
    
    _POSITIVE = {
        "great", "excellent", "good", "clear", "clearly", "helpful", "patient",
        "perfect", "amazing", "awesome", "love", "loved", "best", "understand",
        "appreciated", "confident", "thank", "thanks", "recommend", "nice",
        "wonderful", "fantastic", "enjoyed", "solidify", "thorough", "knowledgeable",
        "informative", "engaging", "effective", "encouraging", "brilliant",
        "insightful", "supportive", "superb", "delightful", "skilled", "mastered",
        "friendly", "responsive", "valuable", "productive", "clearer", "satisfied"
    }
    _NEGATIVE = {
        "bad", "poor", "terrible", "awful", "confusing", "confused", "rude",
        "impatient", "unhelpful", "waste", "worst", "boring", "late", "unclear",
        "disappointed", "disappointing", "horrible", "useless", "never", "frustrating",
        "frustrated", "unprepared", "distracted", "struggling", "rushed",
        "incompetent", "abrupt", "dismissive", "messy", "disorganized", "lazy", "harsh"
    }
    _NEGATIONS = {
        "not", "never", "no", "hardly", "scarcely", "barely", "without",
        "wasn't", "wasnt", "isn't", "isnt", "didn't", "didnt", "cannot",
        "can't", "cant", "couldn't", "couldnt", "don't", "dont", "doesn't", "doesnt"
    }

    def _lexicon_sentiment(self, text: str) -> Dict:
        """Lexicon fallback with negation handling when transformer is unavailable."""
        words = re.findall(r"[a-z']+", text.lower())
        pos = 0
        neg = 0

        for i, word in enumerate(words):
            # Check if preceded by a negation word within 2 words
            has_negation = False
            if i > 0 and words[i - 1] in self._NEGATIONS:
                has_negation = True
            elif i > 1 and words[i - 2] in self._NEGATIONS:
                has_negation = True

            if word in self._POSITIVE:
                if has_negation:
                    neg += 1
                else:
                    pos += 1
            elif word in self._NEGATIVE:
                if has_negation:
                    pos += 1
                else:
                    neg += 1

        if pos + neg == 0:
            return {"label": "NEUTRAL", "score": 0.5}
        if pos > neg:
            score = 0.5 + 0.5 * min(1.0, (pos - neg + 1) / (pos + neg + 1))
            return {"label": "POSITIVE", "score": min(1.0, score)}
        elif neg > pos:
            score = 0.5 + 0.5 * min(1.0, (neg - pos + 1) / (pos + neg + 1))
            return {"label": "NEGATIVE", "score": min(1.0, score)}
        return {"label": "NEUTRAL", "score": 0.5}

    def analyze_sentiment(
        self,
        text: str
    ) -> Tuple[str, float]:
        """
        Analyze sentiment of feedback text.
        
        Implementation of FEAT-6.4.1:
        - Takes feedback free_text_comment
        - Outputs sentiment (positive/neutral/negative)
        - Outputs sentiment_score (0.0-1.0)
        - Uses BERT fine-tuned for sentiment
        
        Args:
            text: Feedback text to analyze
            
        Returns:
            Tuple of (sentiment, sentiment_score)
        """
        if not text or len(text.strip()) < 5:
            log.warning("Empty or very short feedback text")
            return ("neutral", 0.5)
        
        try:
            # Get sentiment prediction
            if self.sentiment_pipeline is None:
                result = self._lexicon_sentiment(text)
            else:
                result = self.sentiment_pipeline(text[:512])[0]
            
            label = result['label'].lower()
            score = result['score']
            
            # Convert to our sentiment categories
            if label == 'positive':
                sentiment = 'positive'
                sentiment_score = score
            elif label == 'negative':
                sentiment = 'negative'
                sentiment_score = 1.0 - score
            else:
                sentiment = 'neutral'
                sentiment_score = 0.5
            
            log.debug(f"Sentiment: {sentiment} (score: {sentiment_score:.3f})")
            
            return (sentiment, sentiment_score)
            
        except Exception as e:
            log.error(f"Error analyzing sentiment: {e}")
            return ("neutral", 0.5)
    
    def check_tone_consistency(
        self,
        sentiment: str,
        sentiment_score: float,
        ratings: Dict[str, float]
    ) -> Tuple[float, bool]:
        """
        Check consistency between sentiment and ratings.
        
        Implementation of FEAT-6.4.2:
        - Takes feedback ratings + sentiment
        - High ratings should correlate with positive sentiment
        - Flags inconsistencies
        - Generates consistency score (0.0-1.0)
        
        Args:
            sentiment: Detected sentiment (positive/neutral/negative)
            sentiment_score: Sentiment confidence score
            ratings: Dictionary of rating scores
            
        Returns:
            Tuple of (consistency_score, is_inconsistent)
        """
        log.debug("Checking tone consistency")
        
        if not ratings:
            log.warning("No ratings provided for consistency check")
            return (0.5, False)
        
        # Calculate average rating
        avg_rating = sum(ratings.values()) / len(ratings)
        
        # Normalize rating to 0-1 scale (assuming ratings are 0-5)
        normalized_rating = avg_rating / 5.0
        
        # Expected sentiment based on rating
        if normalized_rating >= 0.7:
            expected_sentiment = 'positive'
        elif normalized_rating <= 0.4:
            expected_sentiment = 'negative'
        else:
            expected_sentiment = 'neutral'
        
        # Check consistency
        if sentiment == expected_sentiment:
            # Perfect match
            consistency_score = 0.95
            is_inconsistent = False
        elif (sentiment == 'positive' and expected_sentiment == 'neutral') or \
             (sentiment == 'neutral' and expected_sentiment == 'positive'):
            # Minor mismatch
            consistency_score = 0.75
            is_inconsistent = False
        elif (sentiment == 'negative' and expected_sentiment == 'neutral') or \
             (sentiment == 'neutral' and expected_sentiment == 'negative'):
            # Minor mismatch
            consistency_score = 0.70
            is_inconsistent = False
        else:
            # Major inconsistency (positive rating with negative sentiment, etc.)
            consistency_score = 0.3
            is_inconsistent = True
        
        # Adjust based on magnitude of difference
        score_diff = abs(sentiment_score - normalized_rating)
        consistency_score = max(0.0, consistency_score - score_diff * 0.3)
        
        if is_inconsistent:
            log.warning(
                f"Tone inconsistency detected: sentiment={sentiment} "
                f"(score={sentiment_score:.2f}), "
                f"avg_rating={avg_rating:.2f}"
            )
        
        log.debug(f"Consistency score: {consistency_score:.3f}")
        
        return (consistency_score, is_inconsistent)
