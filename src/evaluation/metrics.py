"""
Custom evaluation metrics for Azerbaijani language models.

This module provides specialized metrics and evaluation functions
specifically designed for Azerbaijani language tasks.
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import unicodedata

import numpy as np

logger = logging.getLogger(__name__)


class AzerbaijaniTextMetrics:
    """
    Azerbaijani-specific text quality metrics.

    Provides metrics that take into account the morphological and
    syntactic characteristics of the Azerbaijani language.
    """

    def __init__(self):
        """Initialize Azerbaijani text metrics."""
        # Common Azerbaijani words for frequency analysis
        self.common_words = {
            'və', 'bir', 'ki', 'bu', 'da', 'də', 'o', 'üçün', 'ilə', 'daha',
            'ən', 'həm', 'ancaq', 'artıq', 'hələ', 'son', 'yeni', 'böyük',
            'kiçik', 'yaxşı', 'pis', 'çox', 'az', 'bütün', 'bəzi', 'hər',
            'heç', 'özü', 'onun', 'onlar', 'bizim', 'sizin', 'mənim'
        }

        # Azerbaijani suffixes for morphological analysis
        self.common_suffixes = {
            'lar', 'lər', 'dan', 'dən', 'nan', 'nən', 'da', 'də',
            'ya', 'yə', 'nı', 'ni', 'nu', 'nü', 'ın', 'in', 'un', 'ün',
            'sı', 'si', 'su', 'sü', 'dır', 'dir', 'dur', 'dür'
        }

        # Azerbaijani alphabet
        self.azerbaijani_chars = set(
            'abcçdeəfgğhıijklmnöpqrsştuüvwxyzABCÇDEƏFGĞHIİJKLMNÖPQRSŞTUÜVWXYZ'
        )

    def calculate_azerbaijani_readability(self, text: str) -> float:
        """
        Calculate readability score for Azerbaijani text.

        Args:
            text: Input text in Azerbaijani

        Returns:
            Readability score (0-100, higher is more readable)
        """
        if not text.strip():
            return 0.0

        # Basic statistics
        sentences = self._split_sentences(text)
        words = self._split_words(text)

        if len(sentences) == 0 or len(words) == 0:
            return 0.0

        # Calculate metrics
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)

        # Azerbaijani-specific adjustments
        # Shorter sentences are generally more readable in Azerbaijani
        sentence_score = max(0, 100 - (avg_sentence_length - 10) * 2)

        # Moderate word length is preferred
        word_score = max(0, 100 - abs(avg_word_length - 6) * 10)

        # Common word frequency (higher is more readable)
        common_word_ratio = sum(1 for word in words if word.lower() in self.common_words) / len(words)
        common_score = common_word_ratio * 100

        # Combine scores
        readability = (sentence_score * 0.4 + word_score * 0.3 + common_score * 0.3)

        return min(100, max(0, readability))

    def calculate_morphological_diversity(self, text: str) -> Dict[str, float]:
        """
        Calculate morphological diversity metrics for Azerbaijani text.

        Args:
            text: Input text in Azerbaijani

        Returns:
            Dictionary with morphological diversity metrics
        """
        words = self._split_words(text)

        if not words:
            return {
                'suffix_diversity': 0.0,
                'root_diversity': 0.0,
                'morphological_complexity': 0.0
            }

        # Analyze suffixes
        suffix_counts = Counter()
        roots = []

        for word in words:
            word_lower = word.lower()

            # Find potential suffixes
            for suffix in self.common_suffixes:
                if word_lower.endswith(suffix) and len(word_lower) > len(suffix):
                    suffix_counts[suffix] += 1
                    root = word_lower[:-len(suffix)]
                    roots.append(root)
                    break
            else:
                # No suffix found, consider whole word as root
                roots.append(word_lower)

        # Calculate diversity metrics
        unique_suffixes = len(suffix_counts)
        unique_roots = len(set(roots))
        total_words = len(words)

        suffix_diversity = unique_suffixes / max(1, total_words)
        root_diversity = unique_roots / max(1, total_words)

        # Morphological complexity (average morphemes per word)
        morpheme_counts = []
        for word in words:
            count = 1  # Root
            for suffix in self.common_suffixes:
                if word.lower().endswith(suffix):
                    count += 1
                    break
            morpheme_counts.append(count)

        morphological_complexity = np.mean(morpheme_counts)

        return {
            'suffix_diversity': suffix_diversity,
            'root_diversity': root_diversity,
            'morphological_complexity': morphological_complexity,
            'unique_suffixes': unique_suffixes,
            'unique_roots': unique_roots
        }

    def calculate_lexical_diversity(self, text: str) -> Dict[str, float]:
        """
        Calculate lexical diversity metrics.

        Args:
            text: Input text

        Returns:
            Dictionary with lexical diversity metrics
        """
        words = [word.lower() for word in self._split_words(text)]

        if not words:
            return {
                'ttr': 0.0,  # Type-Token Ratio
                'mattr': 0.0,  # Moving Average Type-Token Ratio
                'mtld': 0.0,  # Measure of Textual Lexical Diversity
                'hdd': 0.0,  # HD-D
            }

        unique_words = set(words)
        total_words = len(words)

        # Type-Token Ratio
        ttr = len(unique_words) / total_words

        # Moving Average Type-Token Ratio (with window size 50)
        window_size = min(50, total_words)
        if total_words >= window_size:
            ttrs = []
            for i in range(total_words - window_size + 1):
                window_words = words[i:i + window_size]
                window_unique = len(set(window_words))
                ttrs.append(window_unique / window_size)
            mattr = np.mean(ttrs)
        else:
            mattr = ttr

        # Simplified MTLD calculation
        mtld = self._calculate_mtld(words)

        # HD-D (Hypergeometric Distribution Diversity)
        hdd = self._calculate_hdd(words)

        return {
            'ttr': ttr,
            'mattr': mattr,
            'mtld': mtld,
            'hdd': hdd,
        }

    def _calculate_mtld(self, words: List[str], threshold: float = 0.72) -> float:
        """Calculate Measure of Textual Lexical Diversity."""
        if len(words) < 50:
            return 0.0

        def calculate_factor(word_list):
            unique_words = set()
            for i, word in enumerate(word_list):
                unique_words.add(word)
                ttr = len(unique_words) / (i + 1)
                if ttr <= threshold:
                    return i + 1
            return len(word_list)

        # Forward calculation
        forward_factors = []
        start = 0
        while start < len(words):
            factor_length = calculate_factor(words[start:])
            forward_factors.append(factor_length)
            start += factor_length
            if start >= len(words):
                break

        # Backward calculation
        backward_factors = []
        start = len(words) - 1
        while start >= 0:
            factor_length = calculate_factor(words[max(0, start-49):start+1][::-1])
            backward_factors.append(factor_length)
            start -= factor_length
            if start < 0:
                break

        all_factors = forward_factors + backward_factors
        return np.mean(all_factors) if all_factors else 0.0

    def _calculate_hdd(self, words: List[str], sample_size: int = 42) -> float:
        """Calculate HD-D (Hypergeometric Distribution Diversity)."""
        word_counts = Counter(words)
        total_words = len(words)

        if total_words < sample_size:
            return len(set(words)) / total_words

        # Calculate probability of each word type appearing in a sample
        probabilities = []
        for count in word_counts.values():
            # Probability that word type does NOT appear in sample
            prob_not_in_sample = 1.0
            for i in range(sample_size):
                prob_not_in_sample *= (total_words - count - i) / (total_words - i)

            prob_in_sample = 1 - prob_not_in_sample
            probabilities.append(prob_in_sample)

        return sum(probabilities)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting for Azerbaijani
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _split_words(self, text: str) -> List[str]:
        """Split text into words."""
        # Clean and split words
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.split()
        return [word for word in words if word and any(c in self.azerbaijani_chars for c in word)]


class SemanticCoherenceMetrics:
    """
    Metrics for measuring semantic coherence in Azerbaijani text.
    """

    def __init__(self):
        """Initialize semantic coherence metrics."""
        # Common discourse markers in Azerbaijani
        self.discourse_markers = {
            'həmçinin', 'bundan_başqa', 'buna_görə', 'nəticədə', 'beləliklə',
            'lakin', 'ancaq', 'amma', 'yalnız', 'sadəcə',
            'məsələn', 'yəni', 'başqa_sözlə', 'xüsusilə',
            'əvvəlcə', 'sonra', 'daha_sonra', 'nəhayət'
        }

        # Topic transition words
        self.transition_words = {
            'əvvəla', 'ikincisi', 'üçüncüsü', 'son_olaraq',
            'digər_tərəfdən', 'əksinə', 'qarşı_olaraq'
        }

    def calculate_coherence_score(self, text: str) -> Dict[str, float]:
        """
        Calculate semantic coherence score for text.

        Args:
            text: Input text

        Returns:
            Dictionary with coherence metrics
        """
        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            return {
                'coherence_score': 1.0,
                'discourse_marker_density': 0.0,
                'topic_consistency': 1.0,
                'sentence_connectivity': 1.0
            }

        # Calculate discourse marker density
        words = text.lower().split()
        marker_count = sum(1 for word in words if word in self.discourse_markers)
        marker_density = marker_count / len(words) if words else 0

        # Calculate sentence connectivity (simplified)
        connectivity_scores = []
        for i in range(len(sentences) - 1):
            curr_words = set(sentences[i].lower().split())
            next_words = set(sentences[i + 1].lower().split())

            # Lexical overlap
            overlap = len(curr_words & next_words)
            total_unique = len(curr_words | next_words)
            connectivity = overlap / total_unique if total_unique > 0 else 0
            connectivity_scores.append(connectivity)

        avg_connectivity = np.mean(connectivity_scores) if connectivity_scores else 0

        # Topic consistency (simplified based on word repetition)
        all_words = text.lower().split()
        word_counts = Counter(all_words)
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        topic_consistency = repeated_words / len(set(all_words)) if all_words else 0

        # Overall coherence score
        coherence_score = (
            marker_density * 0.3 +
            avg_connectivity * 0.4 +
            topic_consistency * 0.3
        )

        return {
            'coherence_score': coherence_score,
            'discourse_marker_density': marker_density,
            'topic_consistency': topic_consistency,
            'sentence_connectivity': avg_connectivity
        }

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class AzerbaijaniSpecificMetrics:
    """
    Collection of Azerbaijani-specific evaluation metrics.
    """

    def __init__(self):
        """Initialize Azerbaijani-specific metrics."""
        self.text_metrics = AzerbaijaniTextMetrics()
        self.coherence_metrics = SemanticCoherenceMetrics()

    def evaluate_text_quality(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive text quality evaluation for Azerbaijani.

        Args:
            text: Input text to evaluate

        Returns:
            Dictionary with all quality metrics
        """
        if not text or not text.strip():
            return self._get_empty_metrics()

        # Calculate all metrics
        results = {}

        # Basic readability
        results['readability'] = self.text_metrics.calculate_azerbaijani_readability(text)

        # Lexical diversity
        lexical_metrics = self.text_metrics.calculate_lexical_diversity(text)
        results.update({f'lexical_{k}': v for k, v in lexical_metrics.items()})

        # Morphological diversity
        morphological_metrics = self.text_metrics.calculate_morphological_diversity(text)
        results.update({f'morphological_{k}': v for k, v in morphological_metrics.items()})

        # Semantic coherence
        coherence_metrics = self.coherence_metrics.calculate_coherence_score(text)
        results.update({f'coherence_{k}': v for k, v in coherence_metrics.items()})

        # Overall quality score
        results['overall_quality'] = self._calculate_overall_quality(results)

        return results

    def compare_texts(
        self,
        reference_text: str,
        generated_text: str
    ) -> Dict[str, Any]:
        """
        Compare generated text with reference text.

        Args:
            reference_text: Reference (ground truth) text
            generated_text: Generated text to evaluate

        Returns:
            Dictionary with comparison metrics
        """
        ref_metrics = self.evaluate_text_quality(reference_text)
        gen_metrics = self.evaluate_text_quality(generated_text)

        comparison = {}

        # Calculate differences
        for key in ref_metrics:
            if isinstance(ref_metrics[key], (int, float)):
                diff = gen_metrics[key] - ref_metrics[key]
                relative_diff = diff / max(ref_metrics[key], 0.001)

                comparison[f'{key}_diff'] = diff
                comparison[f'{key}_relative_diff'] = relative_diff

        # Overall similarity score
        similarity_scores = []
        for key in ['readability', 'lexical_ttr', 'morphological_complexity', 'coherence_score']:
            if key in ref_metrics and key in gen_metrics:
                ref_val = ref_metrics[key]
                gen_val = gen_metrics[key]

                # Calculate similarity (1 - normalized absolute difference)
                max_val = max(ref_val, gen_val, 0.001)
                similarity = 1 - abs(ref_val - gen_val) / max_val
                similarity_scores.append(similarity)

        comparison['overall_similarity'] = np.mean(similarity_scores) if similarity_scores else 0

        return comparison

    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics."""
        quality_components = []

        # Readability (normalized to 0-1)
        if 'readability' in metrics:
            quality_components.append(metrics['readability'] / 100.0)

        # Lexical diversity (TTR)
        if 'lexical_ttr' in metrics:
            quality_components.append(min(metrics['lexical_ttr'], 1.0))

        # Morphological complexity (normalized)
        if 'morphological_complexity' in metrics:
            # Optimal complexity around 1.5-2.0 morphemes per word
            optimal_complexity = 1.75
            complexity_score = 1 - abs(metrics['morphological_complexity'] - optimal_complexity) / optimal_complexity
            quality_components.append(max(0, complexity_score))

        # Coherence score
        if 'coherence_score' in metrics:
            quality_components.append(min(metrics['coherence_score'], 1.0))

        return np.mean(quality_components) if quality_components else 0.0

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary."""
        return {
            'readability': 0.0,
            'lexical_ttr': 0.0,
            'lexical_mattr': 0.0,
            'lexical_mtld': 0.0,
            'lexical_hdd': 0.0,
            'morphological_suffix_diversity': 0.0,
            'morphological_root_diversity': 0.0,
            'morphological_complexity': 0.0,
            'morphological_unique_suffixes': 0,
            'morphological_unique_roots': 0,
            'coherence_score': 0.0,
            'coherence_discourse_marker_density': 0.0,
            'coherence_topic_consistency': 0.0,
            'coherence_sentence_connectivity': 0.0,
            'overall_quality': 0.0,
        }


def calculate_cultural_appropriateness(text: str) -> Dict[str, float]:
    """
    Calculate cultural appropriateness score for Azerbaijani text.

    This function evaluates how well the text reflects Azerbaijani
    cultural context and linguistic patterns.

    Args:
        text: Input text to evaluate

    Returns:
        Dictionary with cultural appropriateness metrics
    """
    # Azerbaijani cultural keywords
    cultural_keywords = {
        'təbrik', 'bayram', 'novruz', 'şəhid', 'vətən', 'milli',
        'azərbaycan', 'bakı', 'xalq', 'dövlət', 'respublika',
        'məhəbbət', 'dostluq', 'qonaq', 'ev', 'ailə', 'ana',
        'ata', 'oğul', 'qız', 'qardaş', 'bacı', 'nənə', 'baba'
    }

    words = text.lower().split()
    total_words = len(words)

    if total_words == 0:
        return {'cultural_appropriateness': 0.0, 'cultural_keyword_density': 0.0}

    # Count cultural keywords
    cultural_word_count = sum(1 for word in words if word in cultural_keywords)
    cultural_density = cultural_word_count / total_words

    # Cultural appropriateness score
    appropriateness = min(1.0, cultural_density * 10)  # Scale appropriately

    return {
        'cultural_appropriateness': appropriateness,
        'cultural_keyword_density': cultural_density,
    }