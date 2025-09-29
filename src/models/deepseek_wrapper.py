"""
DeepSeek Model Wrapper for Azerbaijani Fine-tuning

This module provides wrapper classes for DeepSeek models with support for
LoRA/QLoRA fine-tuning and Azerbaijani language processing.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

warnings.filterwarnings("ignore")


class DeepSeekModel:
    """
    Wrapper class for DeepSeek models with fine-tuning capabilities.

    Supports various DeepSeek model variants and parameter-efficient fine-tuning
    methods like LoRA and QLoRA.
    """

    SUPPORTED_MODELS = {
        "deepseek-llm-7b-base": "deepseek-ai/deepseek-llm-7b-base",
        "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat",
        "deepseek-llm-67b-base": "deepseek-ai/deepseek-llm-67b-base",
        "deepseek-llm-67b-chat": "deepseek-ai/deepseek-llm-67b-chat",
        "deepseek-coder-6.7b-base": "deepseek-ai/deepseek-coder-6.7b-base",
        "deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    }

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
    ):
        """
        Initialize DeepSeek model wrapper.

        Args:
            model_name: Name or path of the model
            cache_dir: Directory to cache downloaded models
            device_map: Device mapping strategy
            torch_dtype: PyTorch data type for model weights
            load_in_4bit: Whether to load model in 4-bit precision
            load_in_8bit: Whether to load model in 8-bit precision
            trust_remote_code: Whether to trust remote code execution
        """
        self.model_name = self._resolve_model_name(model_name)
        self.cache_dir = cache_dir
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code

        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.generation_config = None

        self._load_model_and_tokenizer()
        self._setup_generation_config()

    def _resolve_model_name(self, model_name: str) -> str:
        """Resolve model name to full HuggingFace model path."""
        if model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[model_name]
        return model_name

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer."""
        print(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )

        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Configure quantization
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.torch_dtype,
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            quantization_config=quantization_config,
            trust_remote_code=self.trust_remote_code,
        )

        print(f"Model loaded successfully. Parameters: {self.get_parameter_count()}")

    def _setup_generation_config(self):
        """Setup default generation configuration."""
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    def setup_lora(
        self,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
    ) -> "DeepSeekModel":
        """
        Setup LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.

        Args:
            rank: Rank of the adaptation
            alpha: Alpha parameter for LoRA scaling
            dropout: Dropout rate for LoRA layers
            target_modules: List of modules to apply LoRA to
            bias: Bias handling strategy

        Returns:
            Self for method chaining
        """
        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        # Prepare model for k-bit training if quantized
        if self.load_in_4bit or self.load_in_8bit:
            self.model = prepare_model_for_kbit_training(self.model)

        # Configure LoRA
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )

        # Apply LoRA to model
        self.peft_model = get_peft_model(self.model, lora_config)

        # Print trainable parameters
        self.print_trainable_parameters()

        return self

    def print_trainable_parameters(self):
        """Print the number of trainable parameters."""
        if self.peft_model is not None:
            self.peft_model.print_trainable_parameters()
        else:
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in self.model.parameters())
            print(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || "
                  f"Trainable%: {100 * trainable_params / all_params:.2f}")

    def get_parameter_count(self) -> str:
        """Get formatted parameter count."""
        if hasattr(self.model, "num_parameters"):
            params = self.model.num_parameters()
        else:
            params = sum(p.numel() for p in self.model.parameters())

        if params >= 1e9:
            return f"{params / 1e9:.1f}B"
        elif params >= 1e6:
            return f"{params / 1e6:.1f}M"
        else:
            return f"{params:,}"

    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        **kwargs
    ) -> str:
        """
        Generate text using the model.

        Args:
            prompt: Input prompt text
            max_length: Maximum total length of generated sequence
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            **kwargs: Additional generation arguments

        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                max_new_tokens=max_new_tokens or self.generation_config.max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode and return
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )

        return generated_text.strip()

    def save_model(self, output_dir: str, save_tokenizer: bool = True):
        """
        Save the model (and optionally tokenizer) to directory.

        Args:
            output_dir: Output directory path
            save_tokenizer: Whether to save tokenizer
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.peft_model is not None:
            # Save LoRA adapter
            self.peft_model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)

        if save_tokenizer:
            self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")

    def load_adapter(self, adapter_path: str) -> "DeepSeekModel":
        """
        Load a LoRA adapter.

        Args:
            adapter_path: Path to the adapter

        Returns:
            Self for method chaining
        """
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.peft_model = self.model
        print(f"Adapter loaded from {adapter_path}")
        return self

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ) -> "DeepSeekModel":
        """
        Load a pre-trained model, optionally with adapter.

        Args:
            model_path: Path to the base model
            adapter_path: Optional path to LoRA adapter
            **kwargs: Additional arguments for model initialization

        Returns:
            Initialized model instance
        """
        instance = cls(model_path, **kwargs)

        if adapter_path:
            instance.load_adapter(adapter_path)

        return instance


class DeepSeekAzerbaijani(DeepSeekModel):
    """
    Specialized wrapper for Azerbaijani fine-tuned DeepSeek models.

    Includes Azerbaijani-specific prompt templates and generation strategies.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_azerbaijani_config()

    def _setup_azerbaijani_config(self):
        """Setup Azerbaijani-specific configuration."""
        # Update generation config for Azerbaijani
        if self.generation_config:
            self.generation_config.max_new_tokens = 1024
            self.generation_config.temperature = 0.8
            self.generation_config.repetition_penalty = 1.05

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """
        Format prompt for Azerbaijani instruction following.

        Args:
            instruction: Instruction in Azerbaijani
            input_text: Optional input context

        Returns:
            Formatted prompt
        """
        if input_text:
            return f"""### Təlimat:
{instruction}

### Giriş:
{input_text}

### Cavab:
"""
        else:
            return f"""### Təlimat:
{instruction}

### Cavab:
"""

    def generate_azerbaijani(
        self,
        instruction: str,
        input_text: str = "",
        **kwargs
    ) -> str:
        """
        Generate Azerbaijani text based on instruction.

        Args:
            instruction: Instruction in Azerbaijani
            input_text: Optional input context
            **kwargs: Additional generation arguments

        Returns:
            Generated Azerbaijani text
        """
        prompt = self.format_prompt(instruction, input_text)
        return self.generate(prompt, **kwargs)

    def translate_to_azerbaijani(self, text: str, source_lang: str = "en") -> str:
        """
        Translate text to Azerbaijani.

        Args:
            text: Source text to translate
            source_lang: Source language code

        Returns:
            Translated Azerbaijani text
        """
        source_lang_map = {
            "en": "ingilis",
            "tr": "türk",
            "ru": "rus",
            "ar": "ərəb",
        }

        source_name = source_lang_map.get(source_lang, source_lang)
        instruction = f"Aşağıdakı {source_name} mətni Azərbaycan dilinə tərcümə et:"

        return self.generate_azerbaijani(instruction, text)

    def answer_question(self, question: str, context: str = "") -> str:
        """
        Answer a question in Azerbaijani.

        Args:
            question: Question in Azerbaijani
            context: Optional context information

        Returns:
            Answer in Azerbaijani
        """
        instruction = "Aşağıdakı suala ətraflı cavab ver:"
        return self.generate_azerbaijani(instruction, question)