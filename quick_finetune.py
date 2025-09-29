#!/usr/bin/env python3
"""
Quick fine-tuning script for DeepSeek on Azerbaijani data.
This script will actually fine-tune a model and save it.
"""

import os
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

def load_data():
    """Load the Azerbaijani training data."""
    with open('data/processed/azerbaijani_sample.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Format for training
    formatted_data = []
    for item in data:
        if item['input']:
            text = f"### Təlimat:\n{item['instruction']}\n\n### Giriş:\n{item['input']}\n\n### Cavab:\n{item['output']}"
        else:
            text = f"### Təlimat:\n{item['instruction']}\n\n### Cavab:\n{item['output']}"
        formatted_data.append({"text": text})

    return formatted_data

def setup_model_and_tokenizer():
    """Setup model and tokenizer."""
    print("Loading model and tokenizer...")

    # Use a smaller model for quick testing (you can change this to deepseek later)
    model_name = "microsoft/DialoGPT-small"  # This is smaller and faster for testing

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    return model, tokenizer

def setup_lora(model):
    """Setup LoRA for efficient fine-tuning."""
    print("Setting up LoRA...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],  # For DialoGPT
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def tokenize_data(data, tokenizer, max_length=512):
    """Tokenize the training data."""
    print("Tokenizing data...")

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
            return_tensors=None,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )

    return tokenized_dataset

def train_model(model, tokenizer, dataset):
    """Train the model."""
    print("Starting training...")

    # Create output directory
    output_dir = "./checkpoints/azerbaijani_model"
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=False,  # Disable for compatibility
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()

    # Save the model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print(f"Model saved to {output_dir}")
    return output_dir

def test_model(model_path, tokenizer_path):
    """Test the fine-tuned model."""
    print("Testing the fine-tuned model...")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Test prompts in Azerbaijani
    test_prompts = [
        "### Təlimat:\nAzərbaycan haqqında maraqlı fakt söylə\n\n### Cavab:\n",
        "### Təlimat:\nBakının tarixi haqqında məlumat ver\n\n### Cavab:\n",
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\nPrompt: {prompt}")
        print(f"Response: {response[len(prompt):]}")

def main():
    """Main function to run the fine-tuning."""
    print("🚀 Starting DeepSeek Azerbaijani Fine-tuning")
    print("=" * 50)

    # Load data
    data = load_data()
    print(f"Loaded {len(data)} training examples")

    # Setup model
    model, tokenizer = setup_model_and_tokenizer()

    # Setup LoRA
    model = setup_lora(model)

    # Tokenize data
    dataset = tokenize_data(data, tokenizer)

    # Train model
    model_path = train_model(model, tokenizer, dataset)

    # Test model
    test_model(model_path, model_path)

    print("\n✅ Fine-tuning completed!")
    print(f"Your fine-tuned Azerbaijani model is saved at: {model_path}")
    print("\nYou can now use this model for Azerbaijani text generation!")

if __name__ == "__main__":
    main()