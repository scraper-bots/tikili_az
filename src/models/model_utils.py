"""
Model utility functions for DeepSeek fine-tuning.

This module contains helper functions for model operations, memory management,
and optimization utilities.
"""

import gc
import torch
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import logging

logger = logging.getLogger(__name__)


def get_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Calculate model size and memory usage.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size information
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Estimate memory usage (rough approximation)
    param_size_mb = param_count * 4 / (1024 * 1024)  # Assuming float32

    return {
        "total_params": param_count,
        "trainable_params": trainable_params,
        "non_trainable_params": param_count - trainable_params,
        "trainable_percentage": (trainable_params / param_count) * 100,
        "estimated_size_mb": param_size_mb,
        "estimated_size_gb": param_size_mb / 1024,
    }


def print_model_info(model: torch.nn.Module, model_name: str = "Model"):
    """
    Print detailed model information.

    Args:
        model: PyTorch model
        model_name: Name of the model for display
    """
    info = get_model_size(model)

    print(f"\n{'='*50}")
    print(f"{model_name} Information")
    print(f"{'='*50}")
    print(f"Total Parameters: {info['total_params']:,}")
    print(f"Trainable Parameters: {info['trainable_params']:,}")
    print(f"Non-trainable Parameters: {info['non_trainable_params']:,}")
    print(f"Trainable Percentage: {info['trainable_percentage']:.2f}%")
    print(f"Estimated Model Size: {info['estimated_size_gb']:.2f} GB")
    print(f"{'='*50}\n")


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory usage information
    """
    # System memory
    system_memory = psutil.virtual_memory()

    memory_info = {
        "system_total_gb": system_memory.total / (1024**3),
        "system_available_gb": system_memory.available / (1024**3),
        "system_used_gb": system_memory.used / (1024**3),
        "system_percentage": system_memory.percent,
    }

    # GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            total = gpu_memory.total_memory / (1024**3)

            memory_info.update({
                f"gpu_{i}_total_gb": total,
                f"gpu_{i}_allocated_gb": allocated,
                f"gpu_{i}_reserved_gb": reserved,
                f"gpu_{i}_free_gb": total - reserved,
            })

    return memory_info


def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize model for inference by disabling gradients and setting eval mode.

    Args:
        model: PyTorch model

    Returns:
        Optimized model
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    return model


def find_optimal_batch_size(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    sample_text: str,
    max_length: int = 512,
    start_batch_size: int = 1,
    max_batch_size: int = 32,
) -> int:
    """
    Find the optimal batch size for the given model and hardware.

    Args:
        model: PyTorch model
        tokenizer: Tokenizer
        sample_text: Sample text for testing
        max_length: Maximum sequence length
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to test

    Returns:
        Optimal batch size
    """
    model.eval()

    # Prepare sample input
    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    ).to(model.device)

    optimal_batch_size = start_batch_size

    for batch_size in range(start_batch_size, max_batch_size + 1):
        try:
            # Create batch
            batch_inputs = {
                key: tensor.repeat(batch_size, 1)
                for key, tensor in inputs.items()
            }

            # Test forward pass
            with torch.no_grad():
                _ = model(**batch_inputs)

            optimal_batch_size = batch_size
            logger.info(f"Batch size {batch_size} successful")

            # Clean up
            cleanup_memory()

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at batch size {batch_size}")
            cleanup_memory()
            break
        except Exception as e:
            logger.error(f"Error at batch size {batch_size}: {e}")
            break

    logger.info(f"Optimal batch size: {optimal_batch_size}")
    return optimal_batch_size


def get_model_layers(model: torch.nn.Module) -> List[str]:
    """
    Get list of all layers in the model.

    Args:
        model: PyTorch model

    Returns:
        List of layer names
    """
    return [name for name, _ in model.named_modules()]


def find_linear_layers(model: torch.nn.Module) -> List[str]:
    """
    Find all linear/dense layers in the model for LoRA targeting.

    Args:
        model: PyTorch model

    Returns:
        List of linear layer names
    """
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            linear_layers.append(name)

    return linear_layers


def get_lora_target_modules(model: torch.nn.Module, model_type: str = "llama") -> List[str]:
    """
    Get recommended LoRA target modules for different model architectures.

    Args:
        model: PyTorch model
        model_type: Type of model architecture

    Returns:
        List of recommended target modules
    """
    # Common patterns for different architectures
    target_patterns = {
        "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "deepseek": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "qwen": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "general": ["query", "key", "value", "dense"],
    }

    patterns = target_patterns.get(model_type.lower(), target_patterns["general"])

    # Find matching modules in the model
    available_modules = [name for name, _ in model.named_modules()]
    target_modules = []

    for pattern in patterns:
        matching_modules = [name for name in available_modules if pattern in name.split('.')[-1]]
        target_modules.extend(matching_modules)

    # Remove duplicates and return unique modules
    return list(set(target_modules))


def estimate_training_time(
    dataset_size: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1,
    seconds_per_step: float = 1.0,
) -> Dict[str, float]:
    """
    Estimate training time based on dataset and hardware.

    Args:
        dataset_size: Number of training samples
        batch_size: Batch size per device
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Gradient accumulation steps
        seconds_per_step: Estimated seconds per training step

    Returns:
        Dictionary with time estimates
    """
    effective_batch_size = batch_size * gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch_size
    total_steps = steps_per_epoch * num_epochs

    total_seconds = total_steps * seconds_per_step
    total_minutes = total_seconds / 60
    total_hours = total_minutes / 60

    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_seconds": total_seconds,
        "estimated_minutes": total_minutes,
        "estimated_hours": total_hours,
        "estimated_days": total_hours / 24,
    }


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration dictionary.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = ["model_name", "batch_size", "learning_rate"]

    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required config key: {key}")
            return False

    # Validate value ranges
    if config.get("batch_size", 0) <= 0:
        logger.error("Batch size must be positive")
        return False

    if config.get("learning_rate", 0) <= 0:
        logger.error("Learning rate must be positive")
        return False

    return True


def save_model_metadata(
    model_path: str,
    metadata: Dict[str, Any],
    filename: str = "model_metadata.json"
):
    """
    Save model metadata to JSON file.

    Args:
        model_path: Path to model directory
        metadata: Metadata dictionary
        filename: Filename for metadata file
    """
    import json
    import os

    metadata_path = os.path.join(model_path, filename)

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Model metadata saved to {metadata_path}")


def load_model_metadata(model_path: str, filename: str = "model_metadata.json") -> Dict[str, Any]:
    """
    Load model metadata from JSON file.

    Args:
        model_path: Path to model directory
        filename: Filename of metadata file

    Returns:
        Metadata dictionary
    """
    import json
    import os

    metadata_path = os.path.join(model_path, filename)

    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}

    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    return metadata