"""
Evaluation framework for Azerbaijani DeepSeek models.

This module provides comprehensive evaluation utilities for assessing model
performance on various Azerbaijani NLP tasks including text generation,
question answering, translation, and sentiment analysis.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict

import torch
from transformers import AutoTokenizer
from datasets import Dataset

# Evaluation metrics
try:
    from evaluate import load as load_metric
    from rouge_score import rouge_scorer
    from sacrebleu import sentence_bleu, BLEU
    from bert_score import score as bert_score
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    logging.warning("Some evaluation metrics not available. Install with: pip install evaluate rouge-score sacrebleu bert-score")

from ..models.deepseek_wrapper import DeepSeekModel, DeepSeekAzerbaijani

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    task: str
    metric_name: str
    score: float
    details: Dict[str, Any] = None


@dataclass
class EvaluationSuite:
    """Container for evaluation suite results."""
    model_name: str
    results: List[EvaluationResult]
    summary: Dict[str, float]
    metadata: Dict[str, Any] = None


class AzerbaijaniEvaluator:
    """
    Comprehensive evaluator for Azerbaijani DeepSeek models.

    Supports evaluation on various tasks including:
    - Text generation quality (BLEU, ROUGE, BERTScore)
    - Question answering accuracy
    - Translation quality
    - Sentiment analysis performance
    - Perplexity measurement
    """

    def __init__(
        self,
        model: Optional[Union[DeepSeekModel, DeepSeekAzerbaijani]] = None,
        model_path: Optional[str] = None,
        cache_dir: str = "./eval_cache",
    ):
        """
        Initialize evaluator.

        Args:
            model: Pre-loaded model instance
            model_path: Path to model (if model not provided)
            cache_dir: Directory for caching evaluation results
        """
        self.model = model
        self.model_path = model_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.metrics = {}
        self._load_metrics()

        # Evaluation history
        self.evaluation_history = []

    def _load_metrics(self):
        """Load evaluation metrics."""
        if not HAS_METRICS:
            logger.warning("Evaluation metrics not available")
            return

        try:
            # Load HuggingFace metrics
            self.metrics['bleu'] = load_metric('bleu')
            self.metrics['rouge'] = load_metric('rouge')
            self.metrics['meteor'] = load_metric('meteor')

            # Initialize ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'],
                use_stemmer=True
            )

            # Initialize BLEU scorer
            self.bleu_scorer = BLEU()

            logger.info("Evaluation metrics loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load some metrics: {e}")

    def _ensure_model_loaded(self):
        """Ensure model is loaded."""
        if self.model is None:
            if self.model_path is None:
                raise ValueError("No model or model_path provided")

            logger.info(f"Loading model from {self.model_path}")
            self.model = DeepSeekAzerbaijani.from_pretrained(self.model_path)

    def evaluate_text_generation(
        self,
        test_prompts: List[str],
        reference_texts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        num_beams: int = 1,
    ) -> Dict[str, float]:
        """
        Evaluate text generation quality.

        Args:
            test_prompts: List of input prompts
            reference_texts: List of reference (ground truth) texts
            max_new_tokens: Maximum tokens to generate
            temperature: Generation temperature
            num_beams: Number of beams for beam search

        Returns:
            Dictionary with evaluation metrics
        """
        self._ensure_model_loaded()

        logger.info(f"Evaluating text generation on {len(test_prompts)} samples")

        # Generate texts
        generated_texts = []
        for prompt in test_prompts:
            generated = self.model.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_beams=num_beams,
                do_sample=temperature > 0,
            )
            generated_texts.append(generated)

        # Calculate metrics
        results = {}

        # BLEU Score
        if 'bleu' in self.metrics:
            bleu_scores = []
            for generated, reference in zip(generated_texts, reference_texts):
                score = sentence_bleu([reference.split()], generated.split())
                bleu_scores.append(score.score)

            results['bleu'] = np.mean(bleu_scores)
            results['bleu_std'] = np.std(bleu_scores)

        # ROUGE Scores
        if hasattr(self, 'rouge_scorer'):
            rouge_scores = defaultdict(list)
            for generated, reference in zip(generated_texts, reference_texts):
                scores = self.rouge_scorer.score(reference, generated)
                for metric, score in scores.items():
                    rouge_scores[f'{metric}_f'].append(score.fmeasure)
                    rouge_scores[f'{metric}_p'].append(score.precision)
                    rouge_scores[f'{metric}_r'].append(score.recall)

            for metric, score_list in rouge_scores.items():
                results[f'rouge_{metric}'] = np.mean(score_list)

        # BERTScore (if available)
        try:
            P, R, F1 = bert_score(generated_texts, reference_texts, lang='other')
            results['bertscore_precision'] = P.mean().item()
            results['bertscore_recall'] = R.mean().item()
            results['bertscore_f1'] = F1.mean().item()
        except Exception as e:
            logger.warning(f"BERTScore not available: {e}")

        # Length statistics
        gen_lengths = [len(text.split()) for text in generated_texts]
        ref_lengths = [len(text.split()) for text in reference_texts]

        results['avg_generated_length'] = np.mean(gen_lengths)
        results['avg_reference_length'] = np.mean(ref_lengths)
        results['length_ratio'] = np.mean(gen_lengths) / np.mean(ref_lengths)

        logger.info(f"Text generation evaluation completed: {results}")
        return results

    def evaluate_question_answering(
        self,
        questions: List[str],
        contexts: List[str],
        answers: List[str],
        max_new_tokens: int = 256,
    ) -> Dict[str, float]:
        """
        Evaluate question answering performance.

        Args:
            questions: List of questions
            contexts: List of context passages
            answers: List of correct answers
            max_new_tokens: Maximum tokens for answers

        Returns:
            Dictionary with evaluation metrics
        """
        self._ensure_model_loaded()

        logger.info(f"Evaluating question answering on {len(questions)} samples")

        # Generate answers
        generated_answers = []
        for question, context in zip(questions, contexts):
            if isinstance(self.model, DeepSeekAzerbaijani):
                answer = self.model.answer_question(question, context)
            else:
                prompt = f"Kontekst: {context}\n\nSual: {question}\n\nCavab:"
                answer = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=0.3,
                )
            generated_answers.append(answer.strip())

        # Calculate metrics
        results = {}

        # Exact Match
        exact_matches = []
        for generated, correct in zip(generated_answers, answers):
            exact_matches.append(1 if generated.lower().strip() == correct.lower().strip() else 0)

        results['exact_match'] = np.mean(exact_matches)

        # F1 Score (token-level)
        f1_scores = []
        for generated, correct in zip(generated_answers, answers):
            gen_tokens = set(generated.lower().split())
            correct_tokens = set(correct.lower().split())

            if len(gen_tokens) == 0 and len(correct_tokens) == 0:
                f1_scores.append(1.0)
            elif len(gen_tokens) == 0 or len(correct_tokens) == 0:
                f1_scores.append(0.0)
            else:
                common = gen_tokens & correct_tokens
                precision = len(common) / len(gen_tokens)
                recall = len(common) / len(correct_tokens)

                if precision + recall == 0:
                    f1_scores.append(0.0)
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)
                    f1_scores.append(f1_score)

        results['f1_score'] = np.mean(f1_scores)

        # BLEU score for QA
        if hasattr(self, 'bleu_scorer'):
            bleu_scores = []
            for generated, correct in zip(generated_answers, answers):
                score = sentence_bleu([correct.split()], generated.split())
                bleu_scores.append(score.score)

            results['qa_bleu'] = np.mean(bleu_scores)

        logger.info(f"Question answering evaluation completed: {results}")
        return results

    def evaluate_translation(
        self,
        source_texts: List[str],
        target_texts: List[str],
        source_lang: str = 'en',
        target_lang: str = 'az',
    ) -> Dict[str, float]:
        """
        Evaluate translation quality.

        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts (ground truth)
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Dictionary with evaluation metrics
        """
        self._ensure_model_loaded()

        logger.info(f"Evaluating translation on {len(source_texts)} samples")

        # Generate translations
        generated_translations = []
        for source_text in source_texts:
            if isinstance(self.model, DeepSeekAzerbaijani):
                translation = self.model.translate_to_azerbaijani(source_text, source_lang)
            else:
                # Create translation prompt
                lang_map = {'en': 'ingilis', 'tr': 'türk', 'ru': 'rus'}
                source_name = lang_map.get(source_lang, source_lang)
                prompt = f"Aşağıdakı {source_name} mətni Azərbaycan dilinə tərcümə et:\n{source_text}\n\nTərcümə:"

                translation = self.model.generate(
                    prompt=prompt,
                    max_new_tokens=512,
                    temperature=0.3,
                )

            generated_translations.append(translation.strip())

        # Calculate metrics
        results = {}

        # BLEU Score
        bleu_scores = []
        for generated, reference in zip(generated_translations, target_texts):
            score = sentence_bleu([reference.split()], generated.split())
            bleu_scores.append(score.score)

        results['translation_bleu'] = np.mean(bleu_scores)
        results['translation_bleu_std'] = np.std(bleu_scores)

        # ROUGE Scores
        if hasattr(self, 'rouge_scorer'):
            rouge_scores = defaultdict(list)
            for generated, reference in zip(generated_translations, target_texts):
                scores = self.rouge_scorer.score(reference, generated)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)

            for metric, score_list in rouge_scores.items():
                results[f'translation_{metric}'] = np.mean(score_list)

        # Character-level metrics
        char_edit_distances = []
        for generated, reference in zip(generated_translations, target_texts):
            # Simple character-level edit distance approximation
            distance = abs(len(generated) - len(reference))
            char_edit_distances.append(distance)

        results['avg_char_edit_distance'] = np.mean(char_edit_distances)

        logger.info(f"Translation evaluation completed: {results}")
        return results

    def evaluate_perplexity(
        self,
        test_texts: List[str],
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Evaluate model perplexity on test texts.

        Args:
            test_texts: List of texts for perplexity evaluation
            batch_size: Batch size for evaluation

        Returns:
            Dictionary with perplexity metrics
        """
        self._ensure_model_loaded()

        logger.info(f"Evaluating perplexity on {len(test_texts)} samples")

        tokenizer = self.model.tokenizer
        model = self.model.peft_model or self.model.model

        model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(test_texts), batch_size):
                batch_texts = test_texts[i:i + batch_size]

                # Tokenize batch
                encodings = tokenizer(
                    batch_texts,
                    truncation=True,
                    padding=True,
                    max_length=1024,
                    return_tensors="pt",
                )

                # Move to model device
                encodings = {k: v.to(model.device) for k, v in encodings.items()}

                # Calculate loss
                outputs = model(**encodings, labels=encodings["input_ids"])
                loss = outputs.loss

                # Accumulate loss and token count
                total_loss += loss.item() * encodings["input_ids"].size(1)
                total_tokens += encodings["input_ids"].size(1)

        # Calculate perplexity
        average_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(average_loss)).item()

        results = {
            'perplexity': perplexity,
            'average_loss': average_loss,
            'total_tokens': total_tokens,
        }

        logger.info(f"Perplexity evaluation completed: {results}")
        return results

    def run_comprehensive_evaluation(
        self,
        evaluation_dataset: Dict[str, Any],
        output_dir: str = "./evaluation_results",
        save_results: bool = True,
    ) -> EvaluationSuite:
        """
        Run comprehensive evaluation on multiple tasks.

        Args:
            evaluation_dataset: Dictionary containing evaluation data for different tasks
            output_dir: Directory to save results
            save_results: Whether to save results to disk

        Returns:
            EvaluationSuite with complete results
        """
        self._ensure_model_loaded()

        logger.info("Starting comprehensive evaluation")
        results = []

        # Text Generation
        if 'text_generation' in evaluation_dataset:
            data = evaluation_dataset['text_generation']
            gen_results = self.evaluate_text_generation(
                data['prompts'], data['references']
            )
            for metric, score in gen_results.items():
                results.append(EvaluationResult(
                    task='text_generation',
                    metric_name=metric,
                    score=score
                ))

        # Question Answering
        if 'question_answering' in evaluation_dataset:
            data = evaluation_dataset['question_answering']
            qa_results = self.evaluate_question_answering(
                data['questions'], data['contexts'], data['answers']
            )
            for metric, score in qa_results.items():
                results.append(EvaluationResult(
                    task='question_answering',
                    metric_name=metric,
                    score=score
                ))

        # Translation
        if 'translation' in evaluation_dataset:
            data = evaluation_dataset['translation']
            trans_results = self.evaluate_translation(
                data['source_texts'], data['target_texts'],
                data.get('source_lang', 'en'), data.get('target_lang', 'az')
            )
            for metric, score in trans_results.items():
                results.append(EvaluationResult(
                    task='translation',
                    metric_name=metric,
                    score=score
                ))

        # Perplexity
        if 'perplexity' in evaluation_dataset:
            data = evaluation_dataset['perplexity']
            ppl_results = self.evaluate_perplexity(data['texts'])
            for metric, score in ppl_results.items():
                results.append(EvaluationResult(
                    task='perplexity',
                    metric_name=metric,
                    score=score
                ))

        # Create summary
        summary = {}
        for result in results:
            summary[f"{result.task}_{result.metric_name}"] = result.score

        # Create evaluation suite
        evaluation_suite = EvaluationSuite(
            model_name=self.model_path or "loaded_model",
            results=results,
            summary=summary,
            metadata={
                'num_tasks': len(evaluation_dataset),
                'total_metrics': len(results),
                'evaluation_date': pd.Timestamp.now().isoformat(),
            }
        )

        # Save results
        if save_results:
            self.save_evaluation_results(evaluation_suite, output_dir)

        logger.info(f"Comprehensive evaluation completed with {len(results)} metrics")
        return evaluation_suite

    def save_evaluation_results(
        self,
        evaluation_suite: EvaluationSuite,
        output_dir: str,
    ):
        """
        Save evaluation results to disk.

        Args:
            evaluation_suite: Evaluation results to save
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        results_data = {
            'model_name': evaluation_suite.model_name,
            'summary': evaluation_suite.summary,
            'metadata': evaluation_suite.metadata,
            'detailed_results': [
                {
                    'task': r.task,
                    'metric': r.metric_name,
                    'score': r.score,
                    'details': r.details,
                }
                for r in evaluation_suite.results
            ]
        }

        results_path = output_dir / 'evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # Save summary CSV
        summary_data = []
        for result in evaluation_suite.results:
            summary_data.append({
                'Task': result.task,
                'Metric': result.metric_name,
                'Score': result.score,
            })

        try:
            import pandas as pd
            df = pd.DataFrame(summary_data)
            df.to_csv(output_dir / 'evaluation_summary.csv', index=False)
        except ImportError:
            logger.warning("Pandas not available, skipping CSV export")

        logger.info(f"Evaluation results saved to {output_dir}")

    def compare_models(
        self,
        model_paths: List[str],
        evaluation_dataset: Dict[str, Any],
        output_dir: str = "./model_comparison",
    ) -> Dict[str, EvaluationSuite]:
        """
        Compare multiple models on the same evaluation dataset.

        Args:
            model_paths: List of model paths to compare
            evaluation_dataset: Evaluation dataset
            output_dir: Output directory for results

        Returns:
            Dictionary mapping model paths to evaluation results
        """
        logger.info(f"Comparing {len(model_paths)} models")

        results = {}

        for model_path in model_paths:
            logger.info(f"Evaluating model: {model_path}")

            # Create evaluator for this model
            evaluator = AzerbaijaniEvaluator(model_path=model_path)

            # Run evaluation
            model_results = evaluator.run_comprehensive_evaluation(
                evaluation_dataset,
                output_dir=f"{output_dir}/{Path(model_path).name}",
                save_results=True,
            )

            results[model_path] = model_results

        # Save comparison results
        self._save_comparison_results(results, output_dir)

        logger.info("Model comparison completed")
        return results

    def _save_comparison_results(
        self,
        comparison_results: Dict[str, EvaluationSuite],
        output_dir: str,
    ):
        """Save model comparison results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create comparison summary
        comparison_data = {}
        for model_path, suite in comparison_results.items():
            model_name = Path(model_path).name
            comparison_data[model_name] = suite.summary

        # Save as JSON
        with open(output_dir / 'model_comparison.json', 'w', encoding='utf-8') as f:
            json.dump(comparison_data, f, indent=2, ensure_ascii=False)

        # Save as CSV if pandas available
        try:
            import pandas as pd
            df = pd.DataFrame(comparison_data).T
            df.to_csv(output_dir / 'model_comparison.csv')
        except ImportError:
            pass

        logger.info(f"Comparison results saved to {output_dir}")


def create_sample_evaluation_dataset() -> Dict[str, Any]:
    """
    Create a sample evaluation dataset for testing.

    Returns:
        Sample evaluation dataset
    """
    return {
        'text_generation': {
            'prompts': [
                "Azərbaycan haqqında maraqlı fakt:",
                "Bakının tarixi:",
                "Azərbaycan mətbəxi:",
            ],
            'references': [
                "Azərbaycan dünyada ən böyük neft ehtiyatlarına malikdir.",
                "Bakı şəhəri çox qədim tarixi olan şəhərdir.",
                "Azərbaycan mətbəxi çox zəngin və dadlıdır.",
            ]
        },
        'question_answering': {
            'questions': [
                "Azərbaycanın paytaxtı hansı şəhərdir?",
                "Xəzər dənizinin ən böyük şəhəri hansıdır?",
            ],
            'contexts': [
                "Azərbaycan Respublikası Cənubi Qafqazda yerləşir. Paytaxtı Bakı şəhəridir.",
                "Bakı şəhəri Xəzər dənizi sahilində yerləşir və ölkənin ən böyük şəhəridir.",
            ],
            'answers': [
                "Bakı",
                "Bakı",
            ]
        },
        'perplexity': {
            'texts': [
                "Bu gün hava çox gözəldir. Günəş parlaqdir və göy açıqdır.",
                "Azərbaycan dilində yazmaq çox asandır və maraqlıdır.",
            ]
        }
    }