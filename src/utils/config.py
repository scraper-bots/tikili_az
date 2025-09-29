"""
Configuration management utilities.

This module provides utilities for managing configuration files, environment
variables, and training parameters for DeepSeek fine-tuning.
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration dataclass."""
    name: str = "deepseek-ai/deepseek-llm-7b-base"
    cache_dir: str = "./model_cache"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration dataclass."""
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training configuration dataclass."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    # Precision and memory
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False

    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Logging
    logging_steps: int = 50

    # Early stopping
    early_stopping: bool = False
    early_stopping_patience: int = 3

    # Seeds
    seed: int = 42
    data_seed: int = 42


@dataclass
class DataConfig:
    """Data configuration dataclass."""
    dataset_path: str = ""
    dataset_type: str = "instruction"  # instruction, chat, translation, generation
    text_column: str = "text"
    max_length: int = 2048
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    preprocessing_num_workers: int = 4

    # Data filtering
    min_length: int = 10
    max_length_filter: int = 4096
    filter_duplicates: bool = True
    filter_non_azerbaijani: bool = True
    azerbaijani_threshold: float = 0.7


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration."""
    use_wandb: bool = False
    wandb_project: str = "deepseek-azerbaijani"
    wandb_entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list = field(default_factory=list)

    use_tensorboard: bool = True
    log_dir: str = "./logs"

    save_model_metadata: bool = True
    track_memory: bool = True


@dataclass
class DeploymentConfig:
    """Deployment configuration dataclass."""
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    workers: int = 1
    reload: bool = False
    log_level: str = "info"

    # Model serving
    max_concurrent_requests: int = 10
    timeout: int = 30
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50


@dataclass
class FullConfig:
    """Complete configuration containing all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)


class ConfigManager:
    """
    Configuration manager for DeepSeek fine-tuning.

    Handles loading, saving, and merging configurations from various sources
    including YAML files, environment variables, and command line arguments.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = FullConfig()

        # Load environment variables
        load_dotenv()

        # Load configuration if path provided
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> FullConfig:
        """
        Load configuration from file.

        Args:
            config_path: Path to configuration file

        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return self.config

        logger.info(f"Loading configuration from: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_path.suffix}")

            # Update configuration with loaded data
            self._update_config_from_dict(config_data)

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

        return self.config

    def save_config(self, output_path: str, format_type: str = "yaml"):
        """
        Save current configuration to file.

        Args:
            output_path: Output file path
            format_type: Format to save in ('yaml' or 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(self.config)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format_type.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif format_type.lower() == 'json':
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported format: {format_type}")

            logger.info(f"Configuration saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def _update_config_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary."""
        if 'model' in config_data:
            self._update_dataclass(self.config.model, config_data['model'])

        if 'lora' in config_data:
            self._update_dataclass(self.config.lora, config_data['lora'])

        if 'training' in config_data:
            self._update_dataclass(self.config.training, config_data['training'])

        if 'data' in config_data:
            self._update_dataclass(self.config.data, config_data['data'])

        if 'monitoring' in config_data:
            self._update_dataclass(self.config.monitoring, config_data['monitoring'])

        if 'deployment' in config_data:
            self._update_dataclass(self.config.deployment, config_data['deployment'])

    def _update_dataclass(self, target_config: Any, update_dict: Dict[str, Any]):
        """Update dataclass instance with dictionary values."""
        for key, value in update_dict.items():
            if hasattr(target_config, key):
                setattr(target_config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

    def update_from_env(self, prefix: str = "DEEPSEEK_"):
        """
        Update configuration from environment variables.

        Args:
            prefix: Environment variable prefix
        """
        env_mappings = {
            # Model configuration
            f"{prefix}MODEL": ("model", "name"),
            f"{prefix}CACHE_DIR": ("model", "cache_dir"),
            f"{prefix}LOAD_IN_4BIT": ("model", "load_in_4bit"),
            f"{prefix}LOAD_IN_8BIT": ("model", "load_in_8bit"),

            # Training configuration
            f"{prefix}BATCH_SIZE": ("training", "per_device_train_batch_size"),
            f"{prefix}LEARNING_RATE": ("training", "learning_rate"),
            f"{prefix}NUM_EPOCHS": ("training", "num_train_epochs"),
            f"{prefix}OUTPUT_DIR": ("training", "output_dir"),
            f"{prefix}GRADIENT_ACCUMULATION_STEPS": ("training", "gradient_accumulation_steps"),
            f"{prefix}MAX_LENGTH": ("data", "max_length"),

            # LoRA configuration
            f"{prefix}LORA_RANK": ("lora", "rank"),
            f"{prefix}LORA_ALPHA": ("lora", "alpha"),
            f"{prefix}LORA_DROPOUT": ("lora", "dropout"),

            # Data configuration
            f"{prefix}DATA_PATH": ("data", "dataset_path"),

            # Monitoring configuration
            f"{prefix}USE_WANDB": ("monitoring", "use_wandb"),
            f"{prefix}WANDB_PROJECT": ("monitoring", "wandb_project"),
            f"{prefix}RUN_NAME": ("monitoring", "run_name"),

            # Deployment configuration
            f"{prefix}API_HOST": ("deployment", "api_host"),
            f"{prefix}API_PORT": ("deployment", "api_port"),
        }

        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                section_config = getattr(self.config, section)

                # Convert value to appropriate type
                current_value = getattr(section_config, key)
                if isinstance(current_value, bool):
                    value = value.lower() in ['true', '1', 'yes', 'on']
                elif isinstance(current_value, int):
                    value = int(value)
                elif isinstance(current_value, float):
                    value = float(value)
                elif isinstance(current_value, list):
                    value = value.split(',') if value else []

                setattr(section_config, key, value)
                logger.debug(f"Updated {section}.{key} = {value} from {env_var}")

    def update_from_dict(self, updates: Dict[str, Any]):
        """
        Update configuration from dictionary.

        Args:
            updates: Dictionary with configuration updates
        """
        self._update_config_from_dict(updates)

    def get_training_args_dict(self) -> Dict[str, Any]:
        """
        Get training arguments as dictionary for HuggingFace Trainer.

        Returns:
            Dictionary with training arguments
        """
        training_dict = asdict(self.config.training)

        # Add logging directory
        training_dict['logging_dir'] = os.path.join(
            training_dict['output_dir'], 'logs'
        )

        # Add W&B configuration
        if self.config.monitoring.use_wandb:
            training_dict['report_to'] = ["wandb"]
            training_dict['run_name'] = self.config.monitoring.run_name
        else:
            training_dict['report_to'] = []

        # Clean up fields that aren't TrainingArguments parameters
        fields_to_remove = ['early_stopping', 'early_stopping_patience']
        for field in fields_to_remove:
            training_dict.pop(field, None)

        return training_dict

    def validate_config(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Validate model configuration
            if not self.config.model.name:
                logger.error("Model name cannot be empty")
                return False

            # Validate training configuration
            if self.config.training.per_device_train_batch_size <= 0:
                logger.error("Batch size must be positive")
                return False

            if self.config.training.learning_rate <= 0:
                logger.error("Learning rate must be positive")
                return False

            if self.config.training.num_train_epochs <= 0:
                logger.error("Number of epochs must be positive")
                return False

            # Validate data configuration
            if not self.config.data.dataset_path:
                logger.error("Dataset path cannot be empty")
                return False

            if not Path(self.config.data.dataset_path).exists():
                logger.error(f"Dataset path does not exist: {self.config.data.dataset_path}")
                return False

            # Validate split ratios
            total_ratio = (self.config.data.train_split +
                          self.config.data.val_split +
                          self.config.data.test_split)
            if abs(total_ratio - 1.0) > 1e-6:
                logger.error(f"Data split ratios must sum to 1.0, got {total_ratio}")
                return False

            # Validate LoRA configuration
            if self.config.lora.rank <= 0:
                logger.error("LoRA rank must be positive")
                return False

            if self.config.lora.alpha <= 0:
                logger.error("LoRA alpha must be positive")
                return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def create_sample_config(self, output_path: str):
        """
        Create a sample configuration file.

        Args:
            output_path: Path to save sample configuration
        """
        sample_config = FullConfig()

        # Update with some example values
        sample_config.data.dataset_path = "data/processed/azerbaijani_instructions.json"
        sample_config.monitoring.use_wandb = True
        sample_config.monitoring.run_name = "deepseek-aze-experiment"
        sample_config.monitoring.tags = ["azerbaijani", "deepseek", "lora"]

        # Save to file
        config_dict = asdict(sample_config)

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

        logger.info(f"Sample configuration saved to: {output_path}")

    def print_config(self):
        """Print current configuration in a readable format."""
        print("\n" + "="*50)
        print("Current Configuration")
        print("="*50)

        sections = [
            ("Model", self.config.model),
            ("LoRA", self.config.lora),
            ("Training", self.config.training),
            ("Data", self.config.data),
            ("Monitoring", self.config.monitoring),
            ("Deployment", self.config.deployment),
        ]

        for section_name, section_config in sections:
            print(f"\n{section_name}:")
            print("-" * 20)

            section_dict = asdict(section_config)
            for key, value in section_dict.items():
                print(f"  {key}: {value}")

        print("\n" + "="*50 + "\n")


def load_config_from_path(config_path: str) -> FullConfig:
    """
    Load configuration from file path.

    Args:
        config_path: Path to configuration file

    Returns:
        Loaded configuration
    """
    config_manager = ConfigManager()
    return config_manager.load_config(config_path)


def create_default_config() -> FullConfig:
    """
    Create default configuration.

    Returns:
        Default configuration
    """
    return FullConfig()