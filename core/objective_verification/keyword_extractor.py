"""Keyword extraction module for objective verification."""

from typing import List, Dict
from utils.nlp_utils import NLPUtils
from utils.logger import log
from models.schemas import KeywordExtractionResult


class KeywordExtractor:
    """Extract keywords from transcripts using NLP techniques."""
    
    def __init__(self):
        """Initialize keyword extractor."""
        self.nlp_utils = NLPUtils()
        log.info("KeywordExtractor initialized")
    
    def extract_keywords(
        self,
        transcript: str,
        max_keywords: int = 100,
        method: str = "combined"
    ) -> KeywordExtractionResult:
        """
        Extract top keywords from transcript.
        
        Implementation of FEAT-6.1.1:
        - Uses TF-IDF or YAKE keyword extraction
        - Extracts top 100 keywords by default
        - Filters out stop words
        - Returns keywords with frequencies
        
        Args:
            transcript: Full session transcript
            max_keywords: Maximum number of keywords to extract
            method: Extraction method ('tfidf', 'yake', 'combined')
            
        Returns:
            KeywordExtractionResult with keywords and metadata
        """
        log.info(f"Extracting keywords using method: {method}")
        
        # Choose extraction method
        if method == "tfidf":
            keywords = self.nlp_utils.extract_keywords_tfidf(
                transcript,
                max_features=max_keywords
            )
        elif method == "yake":
            keywords = self.nlp_utils.extract_keywords_yake(
                transcript,
                max_keywords=max_keywords
            )
        else:  # combined
            keywords = self.nlp_utils.extract_keywords_combined(
                transcript,
                max_keywords=max_keywords
            )
        
        # Convert scores to frequencies (normalize to 0-100 scale)
        if keywords:
            max_score = max(kw['score'] for kw in keywords)
            if max_score > 0:
                for kw in keywords:
                    kw['frequency'] = int((kw['score'] / max_score) * 100)
            else:
                for kw in keywords:
                    kw['frequency'] = 0
        
        result = KeywordExtractionResult(
            keywords=keywords,
            total_keywords=len(keywords),
            extraction_method=method
        )
        
        log.info(f"Extracted {len(keywords)} keywords")
        
        return result
    
    def get_keyword_frequencies(
        self,
        keywords: List[Dict[str, any]]
    ) -> Dict[str, int]:
        """
        Convert keyword list to frequency dictionary.
        
        Args:
            keywords: List of keyword dicts
            
        Returns:
            Dictionary mapping keyword to frequency
        """
        return {
            kw['keyword']: kw.get('frequency', kw.get('score', 0))
            for kw in keywords
        }
