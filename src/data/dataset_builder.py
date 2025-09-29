"""
Dataset builder for Azerbaijani fine-tuning datasets.

This module provides utilities to build and manage datasets for training
DeepSeek models on Azerbaijani language tasks.
"""

import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

from .preprocessor import AzerbaijaniPreprocessor

logger = logging.getLogger(__name__)


class AzerbaijaniDatasetBuilder:
    """
    Builder class for creating and managing Azerbaijani training datasets.

    Supports various data sources and formats including instruction following,
    chat conversations, translation pairs, and general text generation.
    """

    def __init__(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 2048,
        preprocessor: Optional[AzerbaijaniPreprocessor] = None,
    ):
        """
        Initialize the dataset builder.

        Args:
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length
            preprocessor: Text preprocessor instance
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessor = preprocessor or AzerbaijaniPreprocessor()

        # Dataset registry
        self.datasets = {}
        self.combined_dataset = None

    def add_instruction_dataset(
        self,
        data_path: str,
        dataset_name: str = "instructions",
        format_type: str = "json",
    ) -> "AzerbaijaniDatasetBuilder":
        """
        Add instruction following dataset.

        Args:
            data_path: Path to instruction data
            dataset_name: Name for the dataset
            format_type: Format of the data file

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading instruction dataset from {data_path}")

        # Load data based on format
        if format_type == "json":
            raw_data = self.preprocessor.load_and_process_json(data_path)
        elif format_type == "jsonl":
            raw_data = self.preprocessor.load_and_process_jsonl(data_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Format for instruction following
        processed_data = self.preprocessor.format_instruction_data(raw_data)

        # Create dataset
        dataset = Dataset.from_list(processed_data)
        self.datasets[dataset_name] = dataset

        logger.info(f"Added instruction dataset '{dataset_name}' with {len(dataset)} samples")
        return self

    def add_chat_dataset(
        self,
        data_path: str,
        dataset_name: str = "chat",
        format_type: str = "json",
    ) -> "AzerbaijaniDatasetBuilder":
        """
        Add chat conversation dataset.

        Args:
            data_path: Path to chat data
            dataset_name: Name for the dataset
            format_type: Format of the data file

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading chat dataset from {data_path}")

        # Load data
        if format_type == "json":
            raw_data = self.preprocessor.load_and_process_json(data_path)
        elif format_type == "jsonl":
            raw_data = self.preprocessor.load_and_process_jsonl(data_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Extract conversations (assuming 'conversations' key)
        conversations = []
        for item in raw_data:
            if "conversations" in item:
                conversations.append(item["conversations"])
            elif "messages" in item:
                conversations.append(item["messages"])
            elif isinstance(item, list):
                conversations.append(item)

        # Format chat data
        processed_data = self.preprocessor.format_chat_data(conversations)

        # Create dataset
        dataset = Dataset.from_list(processed_data)
        self.datasets[dataset_name] = dataset

        logger.info(f"Added chat dataset '{dataset_name}' with {len(dataset)} samples")
        return self

    def add_translation_dataset(
        self,
        data_path: str,
        dataset_name: str = "translation",
        source_lang: str = "en",
        target_lang: str = "az",
        format_type: str = "json",
    ) -> "AzerbaijaniDatasetBuilder":
        """
        Add translation dataset.

        Args:
            data_path: Path to translation data
            dataset_name: Name for the dataset
            source_lang: Source language code
            target_lang: Target language code
            format_type: Format of the data file

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading translation dataset from {data_path}")

        # Load data
        if format_type == "json":
            raw_data = self.preprocessor.load_and_process_json(data_path)
        elif format_type == "jsonl":
            raw_data = self.preprocessor.load_and_process_jsonl(data_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Extract translation pairs
        pairs = []
        for item in raw_data:
            source_text = item.get(source_lang, item.get("source", ""))
            target_text = item.get(target_lang, item.get("target", ""))

            if source_text and target_text:
                pairs.append((source_text, target_text))

        # Format translation data
        processed_data = self.preprocessor.format_translation_data(
            pairs, source_lang, target_lang
        )

        # Create dataset
        dataset = Dataset.from_list(processed_data)
        self.datasets[dataset_name] = dataset

        logger.info(f"Added translation dataset '{dataset_name}' with {len(dataset)} samples")
        return self

    def add_text_generation_dataset(
        self,
        data_path: str,
        dataset_name: str = "generation",
        text_column: str = "text",
        format_type: str = "json",
    ) -> "AzerbaijaniDatasetBuilder":
        """
        Add general text generation dataset.

        Args:
            data_path: Path to text data
            dataset_name: Name for the dataset
            text_column: Column name containing text
            format_type: Format of the data file

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading text generation dataset from {data_path}")

        # Load data
        if format_type == "json":
            raw_data = self.preprocessor.load_and_process_json(data_path)
        elif format_type == "jsonl":
            raw_data = self.preprocessor.load_and_process_jsonl(data_path)
        elif format_type == "txt":
            # For plain text files, process as DOLLMA corpus
            texts = self.preprocessor.process_dollma_corpus(data_path)
            raw_data = [{"text": text} for text in texts]
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Extract and clean texts
        processed_data = []
        for item in raw_data:
            text = item.get(text_column, "")
            if text:
                # Clean text
                if self.preprocessor.clean_text:
                    text = self.preprocessor.clean_text_basic(text)

                # Check if text is Azerbaijani
                if self.preprocessor.is_azerbaijani_text(text):
                    processed_data.append({"text": text})

        # Create dataset
        dataset = Dataset.from_list(processed_data)
        self.datasets[dataset_name] = dataset

        logger.info(f"Added text generation dataset '{dataset_name}' with {len(dataset)} samples")
        return self

    def add_huggingface_dataset(
        self,
        dataset_name: str,
        hf_dataset_name: str,
        split: str = "train",
        text_column: str = "text",
        preprocessing_fn: Optional[Callable] = None,
    ) -> "AzerbaijaniDatasetBuilder":
        """
        Add dataset from HuggingFace Hub.

        Args:
            dataset_name: Name for the dataset
            hf_dataset_name: HuggingFace dataset name
            split: Dataset split to load
            text_column: Column containing text data
            preprocessing_fn: Optional preprocessing function

        Returns:
            Self for method chaining
        """
        logger.info(f"Loading HuggingFace dataset: {hf_dataset_name}")

        # Load dataset from HF Hub
        hf_dataset = load_dataset(hf_dataset_name, split=split)

        # Apply preprocessing if provided
        if preprocessing_fn:
            hf_dataset = hf_dataset.map(preprocessing_fn, batched=True)

        # Filter for Azerbaijani text if applicable
        if text_column in hf_dataset.column_names:
            hf_dataset = hf_dataset.filter(
                lambda x: self.preprocessor.is_azerbaijani_text(x[text_column])
            )

        self.datasets[dataset_name] = hf_dataset

        logger.info(f"Added HuggingFace dataset '{dataset_name}' with {len(hf_dataset)} samples")
        return self

    def combine_datasets(
        self,
        dataset_names: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> "AzerbaijaniDatasetBuilder":
        """
        Combine multiple datasets into a single dataset.

        Args:
            dataset_names: Names of datasets to combine (all if None)
            weights: Sampling weights for each dataset

        Returns:
            Self for method chaining
        """
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())

        datasets_to_combine = [self.datasets[name] for name in dataset_names]

        if not datasets_to_combine:
            raise ValueError("No datasets to combine")

        # Apply weights if specified
        if weights:
            weighted_datasets = []
            for name in dataset_names:
                dataset = self.datasets[name]
                weight = weights.get(name, 1.0)

                if weight != 1.0:
                    # Sample dataset according to weight
                    sample_size = int(len(dataset) * weight)
                    if sample_size > 0:
                        dataset = dataset.shuffle().select(range(min(sample_size, len(dataset))))

                weighted_datasets.append(dataset)

            datasets_to_combine = weighted_datasets

        # Combine datasets
        self.combined_dataset = concatenate_datasets(datasets_to_combine)

        logger.info(f"Combined {len(datasets_to_combine)} datasets into {len(self.combined_dataset)} samples")
        return self

    def create_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True,
    ) -> DatasetDict:
        """
        Create train/validation/test splits from the combined dataset.

        Args:
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            shuffle: Whether to shuffle before splitting

        Returns:
            DatasetDict with splits
        """
        if self.combined_dataset is None:
            # Combine all datasets first
            self.combine_datasets()

        dataset = self.combined_dataset

        if shuffle:
            dataset = dataset.shuffle()

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size

        # Create splits
        splits = DatasetDict({
            "train": dataset.select(range(train_size)),
            "validation": dataset.select(range(train_size, train_size + val_size)),
            "test": dataset.select(range(train_size + val_size, total_size)),
        })

        logger.info(f"Created splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        return splits

    def tokenize_dataset(
        self,
        dataset: Dataset,
        tokenizer: Optional[AutoTokenizer] = None,
        text_column: str = "text",
        max_length: Optional[int] = None,
    ) -> Dataset:
        """
        Tokenize a dataset for training.

        Args:
            dataset: Dataset to tokenize
            tokenizer: Tokenizer to use (uses self.tokenizer if None)
            text_column: Column containing text to tokenize
            max_length: Maximum sequence length (uses self.max_length if None)

        Returns:
            Tokenized dataset
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        if tokenizer is None:
            raise ValueError("No tokenizer provided")

        if max_length is None:
            max_length = self.max_length

        def tokenize_function(examples):
            # Tokenize the texts
            tokenized = tokenizer(
                examples[text_column],
                truncation=True,
                padding=False,  # We'll pad in the data collator
                max_length=max_length,
                return_tensors=None,
            )

            # Add labels for causal LM (same as input_ids)
            tokenized["labels"] = tokenized["input_ids"].copy()

            return tokenized

        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        logger.info(f"Tokenized dataset with {len(tokenized_dataset)} samples")
        return tokenized_dataset

    def create_data_loader(
        self,
        dataset: Dataset,
        tokenizer: Optional[AutoTokenizer] = None,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Create a DataLoader from tokenized dataset.

        Args:
            dataset: Tokenized dataset
            tokenizer: Tokenizer for data collation
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            DataLoader instance
        """
        if tokenizer is None:
            tokenizer = self.tokenizer

        if tokenizer is None:
            raise ValueError("No tokenizer provided")

        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=None,  # We don't need the model for collation
            padding=True,
            return_tensors="pt",
        )

        # Create DataLoader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=data_collator,
        )

        return data_loader

    def save_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        output_path: str,
        format_type: str = "json",
    ):
        """
        Save dataset to disk.

        Args:
            dataset: Dataset to save
            output_path: Output path
            format_type: Format to save in
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(dataset, DatasetDict):
            # Save each split separately
            for split_name, split_data in dataset.items():
                split_path = output_path.parent / f"{output_path.stem}_{split_name}.{format_type}"
                self._save_single_dataset(split_data, split_path, format_type)
        else:
            # Save single dataset
            self._save_single_dataset(dataset, output_path, format_type)

        logger.info(f"Saved dataset to {output_path}")

    def _save_single_dataset(self, dataset: Dataset, path: Path, format_type: str):
        """Save single dataset in specified format."""
        if format_type == "json":
            dataset.to_json(path)
        elif format_type == "csv":
            dataset.to_csv(path)
        elif format_type == "parquet":
            dataset.to_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def get_dataset_stats(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Get statistics about the dataset.

        Args:
            dataset: Dataset to analyze

        Returns:
            Statistics dictionary
        """
        stats = {
            "total_samples": len(dataset),
            "columns": dataset.column_names,
        }

        # Analyze text lengths if 'text' column exists
        if "text" in dataset.column_names:
            texts = dataset["text"]
            text_lengths = [len(text) for text in texts]

            stats.update({
                "avg_text_length": sum(text_lengths) / len(text_lengths),
                "min_text_length": min(text_lengths),
                "max_text_length": max(text_lengths),
                "median_text_length": sorted(text_lengths)[len(text_lengths) // 2],
            })

        return stats

    def print_sample_data(self, dataset: Dataset, num_samples: int = 3):
        """
        Print sample data from dataset.

        Args:
            dataset: Dataset to sample from
            num_samples: Number of samples to print
        """
        print(f"\n{'='*50}")
        print(f"Sample Data ({num_samples} samples)")
        print(f"{'='*50}")

        sample_indices = range(min(num_samples, len(dataset)))

        for i in sample_indices:
            sample = dataset[i]
            print(f"\nSample {i + 1}:")
            print("-" * 30)

            for key, value in sample.items():
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."

                print(f"{key}: {value}")

        print(f"\n{'='*50}\n")

    def build_instruction_dataset(
        self,
        instruction_paths: List[str],
        **kwargs
    ) -> DatasetDict:
        """
        Convenience method to build instruction dataset with train/val/test splits.

        Args:
            instruction_paths: Paths to instruction data files
            **kwargs: Additional arguments for dataset building

        Returns:
            DatasetDict with splits
        """
        # Add all instruction datasets
        for i, path in enumerate(instruction_paths):
            self.add_instruction_dataset(path, f"instructions_{i}")

        # Create splits
        return self.create_splits(**kwargs)