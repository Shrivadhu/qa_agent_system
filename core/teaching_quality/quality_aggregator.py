"""Teaching quality score aggregation module."""

from typing import List, Dict
import numpy as np
from utils.logger import log
from models.schemas import TeachingQualityDimensions
from config import config


class QualityAggregator:
    """Aggregate teaching quality scores across transcript."""
    
    def __init__(self):
        """Initialize quality aggregator."""
        self.dimension_weights = config.get(
            "teaching_quality.weights",
            {
                "clarity": 0.25,
                "engagement": 0.20,
                "patience": 0.15,
                "structure": 0.20,
                "relevance": 0.20
            }
        )
        log.info("QualityAggregator initialized")
    
    def aggregate_scores(
        self,
        segment_scores: List[TeachingQualityDimensions],
        segment_lengths: List[int] = None
    ) -> Dict[str, float]:
        """
        Aggregate quality scores across segments.
        
        Implementation of FEAT-6.2.2:
        - Calculates average score across transcript segments
        - Weights by segment length
        - Generates per-dimension score
        - Generates overall quality score
        
        Args:
            segment_scores: List of quality dimensions for each segment
            segment_lengths: Optional weights (lengths) for each segment
            
        Returns:
            Dictionary with per-dimension and overall scores
        """
        log.info(f"Aggregating scores from {len(segment_scores)} segments")
        
        if not segment_scores:
            log.warning("No segment scores provided")
            return self._get_zero_scores()
        
        # Use uniform weights if lengths not provided
        if segment_lengths is None:
            segment_lengths = [1] * len(segment_scores)
        
        # Normalize weights
        total_length = sum(segment_lengths)
        weights = [length / total_length for length in segment_lengths]
        
        # Aggregate each dimension
        dimensions = ['clarity', 'engagement', 'patience', 'structure', 'relevance']
        aggregated = {}
        
        for dim in dimensions:
            scores = [getattr(score, dim) for score in segment_scores]
            weighted_avg = sum(
                score * weight
                for score, weight in zip(scores, weights)
            )
            aggregated[dim] = weighted_avg
            log.debug(f"{dim}: {weighted_avg:.3f}")
        
        # Calculate overall quality score using dimension weights
        overall_score = sum(
            aggregated[dim] * self.dimension_weights.get(dim, 0.2)
            for dim in dimensions
        )
        
        aggregated['overall'] = overall_score
        
        log.info(f"Overall teaching quality score: {overall_score:.3f}")
        
        return aggregated
    
    def _get_zero_scores(self) -> Dict[str, float]:
        """Return zero scores for all dimensions."""
        return {
            'clarity': 0.0,
            'engagement': 0.0,
            'patience': 0.0,
            'structure': 0.0,
            'relevance': 0.0,
            'overall': 0.0
        }
    
    def get_quality_assessment(
        self,
        segment_scores: List[TeachingQualityDimensions],
        segment_lengths: List[int] = None
    ) -> Dict[str, any]:
        """
        Get comprehensive quality assessment with insights.
        
        Args:
            segment_scores: List of quality dimensions
            segment_lengths: Optional segment lengths
            
        Returns:
            Dictionary with scores, insights, and recommendations
        """
        aggregated = self.aggregate_scores(segment_scores, segment_lengths)
        
        # Identify strengths and weaknesses
        dimensions = ['clarity', 'engagement', 'patience', 'structure', 'relevance']
        dim_scores = [(dim, aggregated[dim]) for dim in dimensions]
        dim_scores.sort(key=lambda x: x[1], reverse=True)
        
        strengths = [dim for dim, score in dim_scores if score >= 0.8]
        weaknesses = [dim for dim, score in dim_scores if score < 0.6]
        
        # Generate insights
        insights = []
        if strengths:
            insights.append(f"Strong in: {', '.join(strengths)}")
        if weaknesses:
            insights.append(f"Needs improvement in: {', '.join(weaknesses)}")
        
        # Calculate score variance (consistency)
        scores = [aggregated[dim] for dim in dimensions]
        score_variance = np.var(scores)
        
        if score_variance < 0.01:
            insights.append("Very consistent teaching quality across all dimensions")
        elif score_variance > 0.05:
            insights.append("Teaching quality varies significantly across dimensions")
        
        return {
            'scores': aggregated,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'insights': insights,
            'consistency': 1.0 - min(score_variance * 10, 1.0)
        }
