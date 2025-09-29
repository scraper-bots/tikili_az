"""
Logging utilities for DeepSeek fine-tuning project.

This module provides logging setup and utilities for consistent logging
across the entire project.
"""

import os
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )

        return super().format(record)


def setup_logger(
    name: str = "deepseek_aze",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: str = "logs",
    use_colors: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Log file name (if None, uses timestamp)
        log_dir: Directory for log files
        use_colors: Whether to use colored output for console
        format_string: Custom format string

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Default format string
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_colors and sys.stdout.isatty():
        console_formatter = ColoredFormatter(format_string)
    else:
        console_formatter = logging.Formatter(format_string)

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (if log directory is writable)
    try:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"deepseek_aze_{timestamp}.log"

        log_file_path = log_dir_path / log_file

        # Use rotating file handler to prevent large log files
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(level)

        file_formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file_path}")

    except (OSError, PermissionError) as e:
        logger.warning(f"Could not setup file logging: {e}")

    return logger


def setup_training_logger(
    output_dir: str,
    experiment_name: str = "experiment",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup logger specifically for training runs.

    Args:
        output_dir: Training output directory
        experiment_name: Name of the experiment
        level: Logging level

    Returns:
        Configured training logger
    """
    log_dir = Path(output_dir) / "logs"
    log_file = f"{experiment_name}.log"

    return setup_logger(
        name=f"deepseek_aze.training.{experiment_name}",
        level=level,
        log_file=log_file,
        log_dir=str(log_dir),
    )


class TrainingLogger:
    """
    Enhanced logger for training with metrics tracking and progress reporting.
    """

    def __init__(
        self,
        name: str = "training",
        output_dir: str = "./logs",
        log_metrics: bool = True,
    ):
        """
        Initialize training logger.

        Args:
            name: Logger name
            output_dir: Output directory for logs
            log_metrics: Whether to log metrics to file
        """
        self.name = name
        self.output_dir = Path(output_dir)
        self.log_metrics = log_metrics

        # Setup main logger
        self.logger = setup_training_logger(str(self.output_dir), name)

        # Metrics storage
        self.metrics_history = []
        self.step_count = 0

        # Create metrics file if needed
        if self.log_metrics:
            self.metrics_file = self.output_dir / "logs" / f"{name}_metrics.jsonl"
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

    def log_step(
        self,
        step: int,
        metrics: Dict[str, Any],
        prefix: str = "train",
    ):
        """
        Log training step with metrics.

        Args:
            step: Training step number
            metrics: Dictionary of metrics
            prefix: Prefix for metric names
        """
        self.step_count = step

        # Format metrics for logging
        metric_strings = []
        for key, value in metrics.items():
            if isinstance(value, float):
                metric_strings.append(f"{key}: {value:.4f}")
            else:
                metric_strings.append(f"{key}: {value}")

        metrics_str = " | ".join(metric_strings)
        self.logger.info(f"Step {step} | {prefix} | {metrics_str}")

        # Save metrics to file
        if self.log_metrics:
            metric_entry = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "prefix": prefix,
                **metrics
            }
            self.metrics_history.append(metric_entry)

            with open(self.metrics_file, 'a', encoding='utf-8') as f:
                import json
                f.write(json.dumps(metric_entry) + '\n')

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        eval_metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Log epoch completion with metrics.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics (optional)
        """
        self.logger.info(f"Epoch {epoch} completed")
        self.log_step(self.step_count, train_metrics, "train_epoch")

        if eval_metrics:
            self.log_step(self.step_count, eval_metrics, "eval_epoch")

    def log_model_info(self, model_info: Dict[str, Any]):
        """
        Log model information.

        Args:
            model_info: Dictionary with model information
        """
        self.logger.info("Model Information:")
        for key, value in model_info.items():
            self.logger.info(f"  {key}: {value}")

    def log_config(self, config: Dict[str, Any]):
        """
        Log training configuration.

        Args:
            config: Configuration dictionary
        """
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"    {sub_key}: {sub_value}")
            else:
                self.logger.info(f"  {key}: {value}")

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset information.

        Args:
            dataset_info: Dictionary with dataset information
        """
        self.logger.info("Dataset Information:")
        for key, value in dataset_info.items():
            self.logger.info(f"  {key}: {value}")

    def log_training_start(self):
        """Log training start."""
        self.logger.info("="*50)
        self.logger.info("Training Started")
        self.logger.info("="*50)

    def log_training_end(self, final_metrics: Dict[str, Any]):
        """
        Log training completion.

        Args:
            final_metrics: Final training metrics
        """
        self.logger.info("="*50)
        self.logger.info("Training Completed")
        self.logger.info("="*50)

        self.logger.info("Final Metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {key}: {value:.4f}")
            else:
                self.logger.info(f"  {key}: {value}")

    def get_metrics_history(self) -> list:
        """
        Get complete metrics history.

        Returns:
            List of metric entries
        """
        return self.metrics_history.copy()

    def save_metrics_summary(self, output_path: Optional[str] = None):
        """
        Save metrics summary to file.

        Args:
            output_path: Output file path (optional)
        """
        if output_path is None:
            output_path = self.output_dir / "logs" / f"{self.name}_metrics_summary.json"

        summary = {
            "experiment_name": self.name,
            "total_steps": self.step_count,
            "total_entries": len(self.metrics_history),
            "metrics_history": self.metrics_history,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(summary, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Metrics summary saved to: {output_path}")


def create_experiment_logger(
    experiment_name: str,
    output_dir: str = "./experiments",
    level: int = logging.INFO,
) -> TrainingLogger:
    """
    Create logger for a specific experiment.

    Args:
        experiment_name: Name of the experiment
        output_dir: Output directory for experiment logs
        level: Logging level

    Returns:
        Configured training logger
    """
    experiment_dir = Path(output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    return TrainingLogger(
        name=experiment_name,
        output_dir=str(experiment_dir),
        log_metrics=True,
    )


def setup_quiet_logger(name: str = "deepseek_aze") -> logging.Logger:
    """
    Setup quiet logger that only shows warnings and errors.

    Args:
        name: Logger name

    Returns:
        Configured quiet logger
    """
    return setup_logger(
        name=name,
        level=logging.WARNING,
        use_colors=False,
    )


def disable_transformers_logging():
    """Disable verbose transformers logging."""
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    datasets_logger = logging.getLogger("datasets")
    datasets_logger.setLevel(logging.WARNING)


def enable_debug_logging():
    """Enable debug logging for all loggers."""
    logging.getLogger().setLevel(logging.DEBUG)

    # Also enable for specific libraries
    for logger_name in ["deepseek_aze", "transformers", "datasets"]:
        logging.getLogger(logger_name).setLevel(logging.DEBUG)