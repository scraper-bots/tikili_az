#!/usr/bin/env python3
"""
Main training script for fine-tuning DeepSeek models on Azerbaijani data.

This script provides a command-line interface for training DeepSeek models
with various parameter-efficient fine-tuning methods including LoRA and QLoRA.

Usage:
    python train.py --model deepseek-ai/deepseek-llm-7b-base --dataset instructions --method qlora

Examples:
    # Quick start with default settings
    python train.py --model deepseek-llm-7b-base --dataset data/processed/instructions.json

    # Advanced configuration
    python train.py \\
        --model deepseek-ai/deepseek-v3 \\
        --dataset data/processed/azerbaijani_corpus.json \\
        --method lora \\
        --rank 16 \\
        --alpha 32 \\
        --epochs 5 \\
        --batch_size 4 \\
        --learning_rate 2e-4 \\
        --output_dir ./checkpoints/deepseek_aze_v1
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import wandb
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.models.deepseek_wrapper import DeepSeekModel
from src.data.dataset_builder import AzerbaijaniDatasetBuilder
from src.data.preprocessor import AzerbaijaniPreprocessor
from src.utils.config import ConfigManager
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)


class DeepSeekTrainer:
    """
    Main trainer class for DeepSeek fine-tuning on Azerbaijani data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trainer with configuration.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        self.trainer = None

        # Initialize components
        self._setup_model()
        self._setup_data()

    def _setup_model(self):
        """Setup model and tokenizer."""
        logger.info(f"Loading model: {self.config['model_name']}")

        # Initialize model with configuration
        self.model = DeepSeekModel(
            model_name=self.config['model_name'],
            cache_dir=self.config.get('cache_dir'),
            load_in_4bit=self.config.get('load_in_4bit', False),
            load_in_8bit=self.config.get('load_in_8bit', False),
        )

        self.tokenizer = self.model.tokenizer

        # Setup LoRA if specified
        if self.config.get('method') in ['lora', 'qlora']:
            self.model.setup_lora(
                rank=self.config.get('lora_rank', 16),
                alpha=self.config.get('lora_alpha', 32),
                dropout=self.config.get('lora_dropout', 0.1),
                target_modules=self.config.get('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"]),
            )

        logger.info("Model setup completed")

    def _setup_data(self):
        """Setup training and evaluation datasets."""
        logger.info(f"Loading dataset: {self.config['dataset_path']}")

        # Initialize dataset builder
        dataset_builder = AzerbaijaniDatasetBuilder(
            tokenizer=self.tokenizer,
            max_length=self.config.get('max_length', 2048),
        )

        # Determine dataset format and load accordingly
        dataset_path = Path(self.config['dataset_path'])
        if dataset_path.suffix == '.json':
            format_type = 'json'
        elif dataset_path.suffix == '.jsonl':
            format_type = 'jsonl'
        else:
            format_type = 'json'  # Default

        # Load dataset based on type
        dataset_type = self.config.get('dataset_type', 'instruction')

        if dataset_type == 'instruction':
            dataset_builder.add_instruction_dataset(
                str(dataset_path),
                format_type=format_type
            )
        elif dataset_type == 'chat':
            dataset_builder.add_chat_dataset(
                str(dataset_path),
                format_type=format_type
            )
        elif dataset_type == 'translation':
            dataset_builder.add_translation_dataset(
                str(dataset_path),
                format_type=format_type
            )
        elif dataset_type == 'generation':
            dataset_builder.add_text_generation_dataset(
                str(dataset_path),
                format_type=format_type
            )
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # Create train/validation splits
        dataset_dict = dataset_builder.create_splits(
            train_ratio=self.config.get('train_ratio', 0.9),
            val_ratio=self.config.get('val_ratio', 0.1),
            test_ratio=0.0,  # We don't need test set for training
        )

        # Tokenize datasets
        self.train_dataset = dataset_builder.tokenize_dataset(
            dataset_dict['train'],
            self.tokenizer
        )

        self.eval_dataset = dataset_builder.tokenize_dataset(
            dataset_dict['validation'],
            self.tokenizer
        )

        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Validation samples: {len(self.eval_dataset)}")

    def _create_training_arguments(self) -> TrainingArguments:
        """Create training arguments."""
        return TrainingArguments(
            output_dir=self.config['output_dir'],
            overwrite_output_dir=True,

            # Training parameters
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 4),
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 8),

            # Optimization
            learning_rate=self.config.get('learning_rate', 2e-4),
            weight_decay=self.config.get('weight_decay', 0.01),
            warmup_steps=self.config.get('warmup_steps', 100),
            max_grad_norm=self.config.get('max_grad_norm', 1.0),

            # Precision and memory
            fp16=self.config.get('fp16', True),
            bf16=self.config.get('bf16', False),
            gradient_checkpointing=self.config.get('gradient_checkpointing', True),
            dataloader_pin_memory=False,

            # Evaluation and saving
            evaluation_strategy="steps",
            eval_steps=self.config.get('eval_steps', 500),
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 1000),
            save_total_limit=self.config.get('save_total_limit', 3),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,

            # Logging
            logging_dir=os.path.join(self.config['output_dir'], 'logs'),
            logging_steps=self.config.get('logging_steps', 50),

            # Misc
            seed=self.config.get('seed', 42),
            data_seed=self.config.get('data_seed', 42),
            remove_unused_columns=False,
            label_names=["labels"],

            # Reporting
            report_to=["wandb"] if self.config.get('use_wandb', False) else [],
            run_name=self.config.get('run_name'),
        )

    def train(self):
        """Start the training process."""
        logger.info("Starting training...")

        # Setup W&B if enabled
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'deepseek-azerbaijani'),
                name=self.config.get('run_name', 'deepseek-aze-finetune'),
                config=self.config,
            )

        # Create training arguments
        training_args = self._create_training_arguments()

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model.model,
            padding=True,
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model.peft_model or self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.get('early_stopping_patience', 3)
                )
            ] if self.config.get('early_stopping', False) else [],
        )

        # Start training
        train_result = self.trainer.train()

        # Log training results
        logger.info("Training completed!")
        logger.info(f"Training loss: {train_result.training_loss:.4f}")

        # Save model
        self.save_model()

        # Finish W&B run
        if self.config.get('use_wandb', False):
            wandb.finish()

        return train_result

    def save_model(self):
        """Save the trained model."""
        logger.info(f"Saving model to {self.config['output_dir']}")

        # Save model and tokenizer
        self.model.save_model(
            output_dir=self.config['output_dir'],
            save_tokenizer=True
        )

        # Save training configuration
        config_path = os.path.join(self.config['output_dir'], 'training_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info("Model saved successfully")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on validation set."""
        if self.trainer is None:
            logger.error("Model not trained yet. Please run train() first.")
            return {}

        logger.info("Evaluating model...")
        eval_result = self.trainer.evaluate()

        logger.info("Evaluation results:")
        for key, value in eval_result.items():
            logger.info(f"{key}: {value:.4f}")

        return eval_result


def create_config_from_args(args) -> Dict[str, Any]:
    """Create configuration dictionary from command line arguments."""
    config = {
        # Model configuration
        'model_name': args.model,
        'cache_dir': args.cache_dir,
        'load_in_4bit': args.method == 'qlora',
        'load_in_8bit': args.load_in_8bit,

        # Dataset configuration
        'dataset_path': args.dataset,
        'dataset_type': args.dataset_type,
        'max_length': args.max_length,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,

        # Training configuration
        'method': args.method,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'max_grad_norm': args.max_grad_norm,

        # LoRA configuration
        'lora_rank': args.rank,
        'lora_alpha': args.alpha,
        'lora_dropout': args.dropout,
        'target_modules': args.target_modules.split(',') if args.target_modules else ["q_proj", "v_proj", "k_proj", "o_proj"],

        # Output configuration
        'output_dir': args.output_dir,
        'save_steps': args.save_steps,
        'eval_steps': args.eval_steps,
        'logging_steps': args.logging_steps,
        'save_total_limit': args.save_total_limit,

        # Optimization configuration
        'fp16': args.fp16,
        'bf16': args.bf16,
        'gradient_checkpointing': args.gradient_checkpointing,
        'early_stopping': args.early_stopping,
        'early_stopping_patience': args.early_stopping_patience,

        # Monitoring configuration
        'use_wandb': args.wandb,
        'wandb_project': args.wandb_project,
        'run_name': args.run_name,

        # Misc
        'seed': args.seed,
    }

    return config


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSeek models for Azerbaijani language",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model arguments
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument(
        '--model', '-m',
        type=str,
        default='deepseek-llm-7b-base',
        help='Model name or path (default: deepseek-llm-7b-base)'
    )
    model_group.add_argument(
        '--cache-dir',
        type=str,
        default='./model_cache',
        help='Model cache directory (default: ./model_cache)'
    )
    model_group.add_argument(
        '--load-in-8bit',
        action='store_true',
        help='Load model in 8-bit precision'
    )

    # Dataset arguments
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Path to training dataset'
    )
    data_group.add_argument(
        '--dataset-type',
        type=str,
        choices=['instruction', 'chat', 'translation', 'generation'],
        default='instruction',
        help='Type of dataset (default: instruction)'
    )
    data_group.add_argument(
        '--max-length',
        type=int,
        default=2048,
        help='Maximum sequence length (default: 2048)'
    )
    data_group.add_argument(
        '--train-ratio',
        type=float,
        default=0.9,
        help='Ratio for training split (default: 0.9)'
    )
    data_group.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Ratio for validation split (default: 0.1)'
    )

    # Training arguments
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument(
        '--method',
        type=str,
        choices=['full', 'lora', 'qlora'],
        default='qlora',
        help='Fine-tuning method (default: qlora)'
    )
    train_group.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Training batch size per device (default: 4)'
    )
    train_group.add_argument(
        '--eval-batch-size',
        type=int,
        default=4,
        help='Evaluation batch size per device (default: 4)'
    )
    train_group.add_argument(
        '--gradient-accumulation-steps',
        type=int,
        default=8,
        help='Gradient accumulation steps (default: 8)'
    )
    train_group.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4)'
    )
    train_group.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='Weight decay (default: 0.01)'
    )
    train_group.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='Number of warmup steps (default: 100)'
    )
    train_group.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping (default: 1.0)'
    )

    # LoRA arguments
    lora_group = parser.add_argument_group('LoRA Configuration')
    lora_group.add_argument(
        '--rank',
        type=int,
        default=16,
        help='LoRA rank (default: 16)'
    )
    lora_group.add_argument(
        '--alpha',
        type=int,
        default=32,
        help='LoRA alpha (default: 32)'
    )
    lora_group.add_argument(
        '--dropout',
        type=float,
        default=0.1,
        help='LoRA dropout (default: 0.1)'
    )
    lora_group.add_argument(
        '--target-modules',
        type=str,
        default='q_proj,v_proj,k_proj,o_proj',
        help='Comma-separated list of target modules (default: q_proj,v_proj,k_proj,o_proj)'
    )

    # Output arguments
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='./checkpoints',
        help='Output directory for model checkpoints (default: ./checkpoints)'
    )
    output_group.add_argument(
        '--save-steps',
        type=int,
        default=1000,
        help='Save checkpoint every N steps (default: 1000)'
    )
    output_group.add_argument(
        '--eval-steps',
        type=int,
        default=500,
        help='Evaluate every N steps (default: 500)'
    )
    output_group.add_argument(
        '--logging-steps',
        type=int,
        default=50,
        help='Log every N steps (default: 50)'
    )
    output_group.add_argument(
        '--save-total-limit',
        type=int,
        default=3,
        help='Maximum number of checkpoints to keep (default: 3)'
    )

    # Optimization arguments
    opt_group = parser.add_argument_group('Optimization Configuration')
    opt_group.add_argument(
        '--fp16',
        action='store_true',
        default=True,
        help='Use mixed precision training (default: True)'
    )
    opt_group.add_argument(
        '--bf16',
        action='store_true',
        help='Use bfloat16 precision (requires newer GPUs)'
    )
    opt_group.add_argument(
        '--gradient-checkpointing',
        action='store_true',
        default=True,
        help='Use gradient checkpointing to save memory (default: True)'
    )
    opt_group.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )
    opt_group.add_argument(
        '--early-stopping-patience',
        type=int,
        default=3,
        help='Early stopping patience (default: 3)'
    )

    # Monitoring arguments
    monitor_group = parser.add_argument_group('Monitoring Configuration')
    monitor_group.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging'
    )
    monitor_group.add_argument(
        '--wandb-project',
        type=str,
        default='deepseek-azerbaijani',
        help='W&B project name (default: deepseek-azerbaijani)'
    )
    monitor_group.add_argument(
        '--run-name',
        type=str,
        help='Name for this training run'
    )

    # Misc arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Setup logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create configuration
    config = create_config_from_args(args)

    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)

    # Log configuration
    logger.info("Training Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    try:
        # Initialize trainer
        trainer = DeepSeekTrainer(config)

        # Start training
        train_result = trainer.train()

        # Evaluate model
        eval_result = trainer.evaluate()

        logger.info("Training completed successfully!")

        # Print final results
        print(f"\\nFinal Results:")
        print(f"Training Loss: {train_result.training_loss:.4f}")
        print(f"Evaluation Loss: {eval_result.get('eval_loss', 'N/A')}")
        print(f"Model saved to: {config['output_dir']}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()