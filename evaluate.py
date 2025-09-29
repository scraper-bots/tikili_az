#!/usr/bin/env python3
"""
Model evaluation script for DeepSeek Azerbaijani models.

This script provides a command-line interface for evaluating trained models
on various Azerbaijani NLP tasks and benchmarks.

Usage:
    python evaluate.py --model ./checkpoints/deepseek_aze_v1 --dataset data/eval/test_set.json

Examples:
    # Evaluate on comprehensive benchmark
    python evaluate.py --model ./checkpoints/best_model --benchmark comprehensive

    # Evaluate specific tasks
    python evaluate.py --model ./checkpoints/best_model --tasks text_generation question_answering

    # Compare multiple models
    python evaluate.py --compare ./checkpoints/model1 ./checkpoints/model2 --dataset data/eval/test_set.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.evaluation.evaluator import AzerbaijaniEvaluator, create_sample_evaluation_dataset
from src.evaluation.metrics import AzerbaijaniSpecificMetrics
from src.models.deepseek_wrapper import DeepSeekAzerbaijani
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


def load_evaluation_dataset(dataset_path: str) -> Dict[str, Any]:
    """
    Load evaluation dataset from file.

    Args:
        dataset_path: Path to evaluation dataset

    Returns:
        Loaded evaluation dataset
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return {}

    logger.info(f"Loading evaluation dataset from: {dataset_path}")

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            if dataset_path.suffix == '.json':
                dataset = json.load(f)
            elif dataset_path.suffix == '.jsonl':
                dataset = {}
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        task = data.get('task', 'default')
                        if task not in dataset:
                            dataset[task] = {'prompts': [], 'references': []}
                        dataset[task]['prompts'].append(data.get('prompt', ''))
                        dataset[task]['references'].append(data.get('reference', ''))
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")

        logger.info(f"Loaded dataset with {len(dataset)} tasks")
        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {}


def create_benchmark_dataset(benchmark_type: str = "comprehensive") -> Dict[str, Any]:
    """
    Create standard benchmark dataset.

    Args:
        benchmark_type: Type of benchmark to create

    Returns:
        Benchmark dataset
    """
    if benchmark_type == "comprehensive":
        return {
            'text_generation': {
                'prompts': [
                    "Azərbaycan haqqında maraqlı faktlar:",
                    "Bakının tarixi və mədəniyyəti haqqında:",
                    "Azərbaycan mətbəxindən məşhur yeməklər:",
                    "Azərbaycan ədəbiyyatının böyük şairləri:",
                    "Xəzər dənizi və onun əhəmiyyəti:",
                    "Novruz bayramının ənənələri:",
                    "Azərbaycanın təbiəti və coğrafiyası:",
                    "Milli musiqi alətlərimiz haqqında:",
                    "Qarabağın tarixi əhəmiyyəti:",
                    "Azərbaycan dilinün xüsusiyyətləri:"
                ],
                'references': [
                    "Azərbaycan dünyada ən çox vulkan palçığına malik ölkədir və neft sənayesinin beşiyi sayılır.",
                    "Bakı şəhəri IX əsrdə yaranmışdır və zəngin neft ehtiyatları sayəsində XX əsrdə böyük inkişaf etmişdir.",
                    "Azərbaycan mətbəxində plov, dolma, kebab, qutab və pakhlava kimi məşhur yeməklər var.",
                    "Nizami Gəncəvi, Füzuli, Nəsimi, Sabir və Nəzim Hikmət kimi böyük şairlər Azərbaycan ədəbiyyatını zənginləşdirmişdir.",
                    "Xəzər dənizi dünyanın ən böyük gölüdür və Azərbaycan iqtisadiyyatı üçün həyati əhəmiyyət daşıyır.",
                    "Novruz baharın gəlişini qeyd edən qədim bayramdır və UNESCO tərəfindən qorunur.",
                    "Azərbaycan 9 iqlim zonasına malikdir və zəngin biologik müxtəliflik təqdim edir.",
                    "Tar, kamança, balaban və zurna Azərbaycanın əsas milli musiqi alətləridir.",
                    "Qarabağ Azərbaycanın ayrılmaz hissəsidir və zəngin mədəni irsə malikdir.",
                    "Azərbaycan dili Türk dillər ailəsindəndir və 50 milyona yaxın insan tərəfindən danışılır."
                ]
            },
            'question_answering': {
                'questions': [
                    "Azərbaycanın paytaxtı hansı şəhərdir?",
                    "Azərbaycanın ən böyük gölü hansıdır?",
                    "Azərbaycan dilində neçə hərf var?",
                    "Azərbaycanın milli valyutası nədir?",
                    "Azərbaycanın qonşu ölkələri hansilardır?",
                    "Azərbaycanın ən hündür dağı hansıdır?",
                    "Füzuli kimdir və nə ilə məşhurdur?",
                    "Azərbaycanın dövlət bayrağındakı rənglər nəyi simvollaşdırır?",
                    "Novruz bayramı neçə gün keçirilir?",
                    "Azərbaycanın ən böyük çayı hansıdır?"
                ],
                'contexts': [
                    "Azərbaycan Respublikası Cənubi Qafqazda yerləşən ölkədir. Paytaxtı Bakı şəhəridir.",
                    "Azərbaycanda çoxsaylı göllər vardır. Ən böyüyü Mingəçevir gölüdür.",
                    "Azərbaycan əlifbası latın əsaslı əlifbadır və 32 hərfdən ibarətdir.",
                    "Azərbaycanın milli valyutası manatdır və 1992-ci ildən istifadə olunur.",
                    "Azərbaycan Rusiya, Gürcüstan, Ermənistan, Türkiyə və İranla həmsərhəddir.",
                    "Azərbaycanın ən hündür dağı Bazardüzü dağıdır və hündürlüyü 4466 metrdir.",
                    "Füzuli XVI əsr Azərbaycan şairi idi və 'Leyli və Məcnun' əsəri ilə məşhurdur.",
                    "Azərbaycan bayrağında mavi, qırmızı və yaşıl rənglər vardır. Ay-ulduz da bayraqda təsvir edilir.",
                    "Novruz bayramı adətən bir həftə keçirilir və 20-21 mart tarixlərində başlayır.",
                    "Azərbaycanın ən böyük çayı Kür çayıdır və 1515 kilometr uzunluğundadır."
                ],
                'answers': [
                    "Bakı",
                    "Mingəçevir gölü",
                    "32 hərf",
                    "Manat",
                    "Rusiya, Gürcüstan, Ermənistan, Türkiyə və İran",
                    "Bazardüzü dağı",
                    "XVI əsr Azərbaycan şairi, 'Leyli və Məcnun' əsərinin müəllifi",
                    "Mavi - türkçülük, qırmızı - müasirlik, yaşıl - İslam",
                    "Bir həftə",
                    "Kür çayı"
                ]
            },
            'perplexity': {
                'texts': [
                    "Azərbaycan zəngin mədəni irsə malik ölkədir. Burada müxtəlif xalqlar yaşayır və onların hər biri öz rəngarəng ənənələrinə malikdir.",
                    "Neft sənayesi Azərbaycan iqtisadiyyatının əsas sahələrindən biridir. Xəzər dənizindəki neft yataqları ölkə büdcəsinin əsas gəlir mənbəyidir.",
                    "Azərbaycan mətbəxi şərq və qərb mətbəxlərinin sintezindən yaranmışdır. Burada ət, düyü, tərəvəz və ədviyyatlardan geniş istifadə olunur.",
                    "Bakı şəhəri öz memarlığı ilə məşhurdur. Qədim şəhər, Qız qalası və Şirvanşahlar sarayı UNESCO-nun Dünya İrsi siyahısındadır.",
                    "Azərbaycan xalq mahnıları muğam adlanır və bu musiqi növü UNESCO tərəfindən qorunur. Muğam ifaçıları xanəndə adlanır."
                ]
            }
        }
    elif benchmark_type == "basic":
        return create_sample_evaluation_dataset()
    else:
        logger.error(f"Unknown benchmark type: {benchmark_type}")
        return {}


def evaluate_single_model(
    model_path: str,
    dataset: Dict[str, Any],
    tasks: Optional[List[str]] = None,
    output_dir: str = "./evaluation_results",
) -> None:
    """
    Evaluate a single model.

    Args:
        model_path: Path to the model
        dataset: Evaluation dataset
        tasks: List of tasks to evaluate (None for all)
        output_dir: Output directory for results
    """
    logger.info(f"Evaluating model: {model_path}")

    # Filter dataset by tasks if specified
    if tasks:
        dataset = {task: data for task, data in dataset.items() if task in tasks}

    # Create evaluator
    evaluator = AzerbaijaniEvaluator(model_path=model_path)

    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(
        dataset,
        output_dir=output_dir,
        save_results=True,
    )

    # Print summary
    print(f"\nEvaluation Results for {Path(model_path).name}")
    print("=" * 50)

    for result in results.results:
        print(f"{result.task} | {result.metric_name}: {result.score:.4f}")

    print(f"\nDetailed results saved to: {output_dir}")


def compare_models(
    model_paths: List[str],
    dataset: Dict[str, Any],
    output_dir: str = "./model_comparison",
) -> None:
    """
    Compare multiple models.

    Args:
        model_paths: List of model paths
        dataset: Evaluation dataset
        output_dir: Output directory for results
    """
    logger.info(f"Comparing {len(model_paths)} models")

    # Create evaluator for comparison
    evaluator = AzerbaijaniEvaluator()

    # Run comparison
    comparison_results = evaluator.compare_models(
        model_paths,
        dataset,
        output_dir=output_dir,
    )

    # Print comparison summary
    print(f"\nModel Comparison Results")
    print("=" * 50)

    # Get common metrics for comparison
    common_metrics = set()
    for results in comparison_results.values():
        common_metrics.update(results.summary.keys())

    # Print table header
    print(f"{'Metric':<30}", end="")
    for model_path in model_paths:
        model_name = Path(model_path).name[:15]
        print(f"{model_name:>15}", end="")
    print()

    print("-" * (30 + 15 * len(model_paths)))

    # Print metrics comparison
    for metric in sorted(common_metrics):
        print(f"{metric:<30}", end="")
        for model_path in model_paths:
            score = comparison_results[model_path].summary.get(metric, 0.0)
            print(f"{score:>15.4f}", end="")
        print()

    print(f"\nDetailed comparison saved to: {output_dir}")


def analyze_text_quality(
    text_file: str,
    output_file: Optional[str] = None,
) -> None:
    """
    Analyze text quality using Azerbaijani-specific metrics.

    Args:
        text_file: Path to text file
        output_file: Optional output file for results
    """
    logger.info(f"Analyzing text quality: {text_file}")

    # Load text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Analyze quality
    metrics = AzerbaijaniSpecificMetrics()
    quality_results = metrics.evaluate_text_quality(text)

    # Print results
    print(f"\nText Quality Analysis for {Path(text_file).name}")
    print("=" * 50)

    for metric, score in quality_results.items():
        if isinstance(score, float):
            print(f"{metric:<30}: {score:.4f}")
        else:
            print(f"{metric:<30}: {score}")

    # Save results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(quality_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate DeepSeek Azerbaijani models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--model', '-m',
        type=str,
        help='Path to model for evaluation'
    )
    model_group.add_argument(
        '--compare',
        type=str,
        nargs='+',
        help='Paths to multiple models for comparison'
    )

    # Dataset arguments
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        '--dataset', '-d',
        type=str,
        help='Path to evaluation dataset'
    )
    data_group.add_argument(
        '--benchmark', '-b',
        type=str,
        choices=['comprehensive', 'basic'],
        default='basic',
        help='Use standard benchmark dataset'
    )

    # Task arguments
    parser.add_argument(
        '--tasks',
        type=str,
        nargs='+',
        choices=['text_generation', 'question_answering', 'translation', 'perplexity'],
        help='Specific tasks to evaluate'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Output directory for results'
    )

    # Text quality analysis
    parser.add_argument(
        '--analyze-text',
        type=str,
        help='Analyze quality of text file'
    )

    # Misc arguments
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Handle text quality analysis
    if args.analyze_text:
        output_file = None
        if args.output_dir != './evaluation_results':
            output_file = os.path.join(args.output_dir, 'text_quality_analysis.json')

        analyze_text_quality(args.analyze_text, output_file)
        return

    # Load or create dataset
    if args.dataset:
        dataset = load_evaluation_dataset(args.dataset)
    else:
        dataset = create_benchmark_dataset(args.benchmark)

    if not dataset:
        logger.error("No dataset available for evaluation")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        if args.compare:
            # Compare multiple models
            compare_models(args.compare, dataset, args.output_dir)
        else:
            # Evaluate single model
            evaluate_single_model(args.model, dataset, args.tasks, args.output_dir)

        logger.info("Evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()