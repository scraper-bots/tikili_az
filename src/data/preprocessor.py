"""
Data preprocessing utilities for Azerbaijani text processing.

This module provides comprehensive text preprocessing functions specifically
designed for Azerbaijani language data preparation for fine-tuning.
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict
import unicodedata

logger = logging.getLogger(__name__)


class AzerbaijaniPreprocessor:
    """
    Comprehensive preprocessor for Azerbaijani text data.

    Handles text cleaning, normalization, and formatting for various NLP tasks
    including instruction following, text generation, and translation.
    """

    def __init__(self, clean_text: bool = True, normalize_unicode: bool = True):
        """
        Initialize the preprocessor.

        Args:
            clean_text: Whether to apply text cleaning
            normalize_unicode: Whether to normalize Unicode characters
        """
        self.clean_text = clean_text
        self.normalize_unicode = normalize_unicode

        # Common Azerbaijani patterns and replacements
        self.azerbaijani_patterns = {
            # Fix common encoding issues
            r'Ä±': 'ı',
            r'Å\x9f': 'ş',
            r'Ä\x9f': 'ğ',
            r'Ã¼': 'ü',
            r'Ã¶': 'ö',
            r'Ã§': 'ç',
            r'Ä™': 'ə',

            # Normalize quotes and punctuation
            r'[""„"]': '"',
            r"[''‚']": "'",
            r'[…]': '...',
            r'[–—]': '-',

            # Remove excessive whitespace
            r'\s+': ' ',
            r'\n+': '\n',
            r'\t+': ' ',
        }

        # Azerbaijani alphabet for validation
        self.azerbaijani_alphabet = set(
            'abcçdeəfgğhıijklmnopöpqrsştuüvwxyz'
            'ABCÇDEƏFGĞHIİJKLMNOPÖPQRSŞTUÜVWXYZ'
            '0123456789.,!?;:()[]{}"\'-/\\@#$%^&*+=_~`|<> \n\t'
        )

    def clean_text_basic(self, text: str) -> str:
        """
        Apply basic text cleaning operations.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)

        # Normalize Unicode if requested
        if self.normalize_unicode:
            text = unicodedata.normalize('NFKC', text)

        # Apply Azerbaijani-specific patterns
        for pattern, replacement in self.azerbaijani_patterns.items():
            text = re.sub(pattern, replacement, text)

        # Remove excessive punctuation
        text = re.sub(r'([.!?]){2,}', r'\1', text)

        # Fix spacing around punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-ZÇƏĞIÖŞÜ])', r'\1 \2', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def is_azerbaijani_text(self, text: str, threshold: float = 0.7) -> bool:
        """
        Check if text is predominantly Azerbaijani.

        Args:
            text: Input text
            threshold: Minimum ratio of valid characters

        Returns:
            True if text appears to be Azerbaijani
        """
        if not text:
            return False

        valid_chars = sum(1 for char in text if char in self.azerbaijani_alphabet)
        total_chars = len(text)

        return (valid_chars / total_chars) >= threshold

    def filter_by_length(
        self,
        texts: List[str],
        min_length: int = 10,
        max_length: int = 2048
    ) -> List[str]:
        """
        Filter texts by length.

        Args:
            texts: List of text strings
            min_length: Minimum character length
            max_length: Maximum character length

        Returns:
            Filtered list of texts
        """
        return [
            text for text in texts
            if min_length <= len(text) <= max_length
        ]

    def remove_duplicates(self, texts: List[str]) -> List[str]:
        """
        Remove duplicate texts while preserving order.

        Args:
            texts: List of text strings

        Returns:
            List with duplicates removed
        """
        seen = set()
        unique_texts = []

        for text in texts:
            # Use normalized text for comparison
            normalized = self.clean_text_basic(text).lower()
            if normalized not in seen:
                seen.add(normalized)
                unique_texts.append(text)

        logger.info(f"Removed {len(texts) - len(unique_texts)} duplicates")
        return unique_texts

    def format_instruction_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Format data for instruction following tasks.

        Args:
            data: List of dictionaries with instruction data

        Returns:
            Formatted instruction data
        """
        formatted_data = []

        for item in data:
            # Extract fields
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output_text = item.get('output', '')

            # Skip if essential fields are missing
            if not instruction or not output_text:
                continue

            # Clean texts
            if self.clean_text:
                instruction = self.clean_text_basic(instruction)
                input_text = self.clean_text_basic(input_text) if input_text else ''
                output_text = self.clean_text_basic(output_text)

            # Create formatted entry
            formatted_item = {
                'instruction': instruction,
                'input': input_text,
                'output': output_text,
            }

            # Create formatted text for training
            if input_text:
                formatted_text = f"### Təlimat:\n{instruction}\n\n### Giriş:\n{input_text}\n\n### Cavab:\n{output_text}"
            else:
                formatted_text = f"### Təlimat:\n{instruction}\n\n### Cavab:\n{output_text}"

            formatted_item['text'] = formatted_text
            formatted_data.append(formatted_item)

        logger.info(f"Formatted {len(formatted_data)} instruction examples")
        return formatted_data

    def format_chat_data(self, conversations: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """
        Format chat conversation data.

        Args:
            conversations: List of conversation lists, where each conversation
                          is a list of messages with 'role' and 'content' keys

        Returns:
            Formatted chat data
        """
        formatted_data = []

        for conversation in conversations:
            if not conversation:
                continue

            # Build conversation text
            conversation_parts = []
            for message in conversation:
                role = message.get('role', '').lower()
                content = message.get('content', '')

                if not content:
                    continue

                # Clean content
                if self.clean_text:
                    content = self.clean_text_basic(content)

                # Format based on role
                if role == 'user' or role == 'human':
                    conversation_parts.append(f"### İstifadəçi:\n{content}")
                elif role == 'assistant' or role == 'bot':
                    conversation_parts.append(f"### Köməkçi:\n{content}")
                elif role == 'system':
                    conversation_parts.append(f"### Sistem:\n{content}")

            if len(conversation_parts) >= 2:  # At least user + assistant
                formatted_item = {
                    'text': '\n\n'.join(conversation_parts),
                    'conversation_length': len(conversation_parts)
                }
                formatted_data.append(formatted_item)

        logger.info(f"Formatted {len(formatted_data)} conversations")
        return formatted_data

    def format_translation_data(
        self,
        pairs: List[Tuple[str, str]],
        source_lang: str = 'en',
        target_lang: str = 'az'
    ) -> List[Dict[str, str]]:
        """
        Format translation pairs for training.

        Args:
            pairs: List of (source, target) translation pairs
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Formatted translation data
        """
        lang_names = {
            'en': 'İngilis',
            'az': 'Azərbaycan',
            'tr': 'Türk',
            'ru': 'Rus',
            'ar': 'Ərəb'
        }

        source_name = lang_names.get(source_lang, source_lang.upper())
        target_name = lang_names.get(target_lang, target_lang.upper())

        formatted_data = []

        for source_text, target_text in pairs:
            if not source_text or not target_text:
                continue

            # Clean texts
            if self.clean_text:
                source_text = self.clean_text_basic(source_text)
                target_text = self.clean_text_basic(target_text)

            # Create instruction format
            instruction = f"Aşağıdakı {source_name} mətni {target_name} dilinə tərcümə et:"
            formatted_text = f"### Təlimat:\n{instruction}\n\n### Giriş:\n{source_text}\n\n### Cavab:\n{target_text}"

            formatted_item = {
                'instruction': instruction,
                'input': source_text,
                'output': target_text,
                'text': formatted_text,
                'source_lang': source_lang,
                'target_lang': target_lang
            }

            formatted_data.append(formatted_item)

        logger.info(f"Formatted {len(formatted_data)} translation pairs")
        return formatted_data

    def create_dataset_splits(
        self,
        data: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        shuffle: bool = True
    ) -> DatasetDict:
        """
        Create train/validation/test splits from data.

        Args:
            data: List of data samples
            train_ratio: Ratio for training split
            val_ratio: Ratio for validation split
            test_ratio: Ratio for test split
            shuffle: Whether to shuffle data before splitting

        Returns:
            DatasetDict with train/validation/test splits
        """
        import random

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # Shuffle if requested
        if shuffle:
            data = data.copy()
            random.shuffle(data)

        # Calculate split indices
        n_samples = len(data)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val

        # Create splits
        train_data = data[:n_train]
        val_data = data[n_train:n_train + n_val]
        test_data = data[n_train + n_val:]

        # Create HuggingFace datasets
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })

        logger.info(f"Created splits - Train: {len(train_data)}, "
                   f"Val: {len(val_data)}, Test: {len(test_data)}")

        return dataset_dict

    def load_and_process_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process JSON data file.

        Args:
            file_path: Path to JSON file

        Returns:
            Processed data list
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            data = [data]

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def load_and_process_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load and process JSONL data file.

        Args:
            file_path: Path to JSONL file

        Returns:
            Processed data list
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def save_processed_data(
        self,
        data: Union[List[Dict[str, Any]], DatasetDict],
        output_path: str,
        format_type: str = 'json'
    ):
        """
        Save processed data to file.

        Args:
            data: Processed data
            output_path: Output file path
            format_type: Format type ('json', 'jsonl', 'csv', 'parquet')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, DatasetDict):
            # Save dataset splits
            for split_name, split_data in data.items():
                split_path = output_path.parent / f"{output_path.stem}_{split_name}{output_path.suffix}"
                self._save_data_by_format(split_data.to_list(), split_path, format_type)
        else:
            # Save single dataset
            self._save_data_by_format(data, output_path, format_type)

        logger.info(f"Saved processed data to {output_path}")

    def _save_data_by_format(self, data: List[Dict[str, Any]], path: Path, format_type: str):
        """Save data in specified format."""
        if format_type == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format_type == 'jsonl':
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format_type == 'csv':
            pd.DataFrame(data).to_csv(path, index=False, encoding='utf-8')
        elif format_type == 'parquet':
            pd.DataFrame(data).to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def process_dollma_corpus(self, file_path: str, chunk_size: int = 1000) -> List[str]:
        """
        Process DOLLMA corpus text file.

        Args:
            file_path: Path to DOLLMA corpus file
            chunk_size: Size of text chunks

        Returns:
            List of processed text chunks
        """
        chunks = []

        with open(file_path, 'r', encoding='utf-8') as f:
            current_chunk = ''
            for line in f:
                line = line.strip()

                if not line:
                    continue

                # Clean line
                if self.clean_text:
                    line = self.clean_text_basic(line)

                # Check if line is Azerbaijani
                if not self.is_azerbaijani_text(line):
                    continue

                # Add to chunk
                current_chunk += line + ' '

                # If chunk is large enough, save it
                if len(current_chunk) >= chunk_size:
                    if len(current_chunk.strip()) >= 100:  # Minimum chunk size
                        chunks.append(current_chunk.strip())
                    current_chunk = ''

            # Add final chunk
            if len(current_chunk.strip()) >= 100:
                chunks.append(current_chunk.strip())

        logger.info(f"Processed DOLLMA corpus into {len(chunks)} chunks")
        return chunks

    def validate_processed_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate processed data quality.

        Args:
            data: Processed data list

        Returns:
            Validation statistics
        """
        stats = {
            'total_samples': len(data),
            'valid_samples': 0,
            'avg_text_length': 0,
            'min_text_length': float('inf'),
            'max_text_length': 0,
            'empty_samples': 0,
            'non_azerbaijani_samples': 0,
        }

        total_length = 0

        for item in data:
            text = item.get('text', '')

            if not text or not text.strip():
                stats['empty_samples'] += 1
                continue

            text_length = len(text)
            total_length += text_length

            stats['min_text_length'] = min(stats['min_text_length'], text_length)
            stats['max_text_length'] = max(stats['max_text_length'], text_length)

            if not self.is_azerbaijani_text(text):
                stats['non_azerbaijani_samples'] += 1
                continue

            stats['valid_samples'] += 1

        if stats['valid_samples'] > 0:
            stats['avg_text_length'] = total_length / stats['valid_samples']

        if stats['min_text_length'] == float('inf'):
            stats['min_text_length'] = 0

        logger.info(f"Validation stats: {stats}")
        return stats