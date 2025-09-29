"""
Evaluation module for Azerbaijani DeepSeek models.

This module provides comprehensive evaluation tools including:
- Model performance evaluation on various NLP tasks
- Azerbaijani-specific text quality metrics
- Comparative analysis between models
- Automated benchmarking
"""

from .evaluator import AzerbaijaniEvaluator, EvaluationResult, EvaluationSuite, create_sample_evaluation_dataset
from .metrics import AzerbaijaniTextMetrics, SemanticCoherenceMetrics, AzerbaijaniSpecificMetrics, calculate_cultural_appropriateness

__all__ = [
    'AzerbaijaniEvaluator',
    'EvaluationResult',
    'EvaluationSuite',
    'AzerbaijaniTextMetrics',
    'SemanticCoherenceMetrics',
    'AzerbaijaniSpecificMetrics',
    'calculate_cultural_appropriateness',
    'create_sample_evaluation_dataset'
]