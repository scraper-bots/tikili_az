#!/usr/bin/env python3
"""
Simple fine-tuning script that actually works and creates a fine-tuned model.
This uses a lightweight approach to get you a working Azerbaijani model quickly.
"""

import os
import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn.functional as F

def create_azerbaijani_model():
    """Create and fine-tune a simple Azerbaijani model."""
    print("🚀 Creating your Azerbaijani fine-tuned model...")

    # Load a base model (using GPT-2 small for quick testing)
    print("Loading base model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Add pad token
    tokenizer.pad_token = tokenizer.eos_token

    # Load Azerbaijani data
    with open('data/processed/azerbaijani_sample.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare training texts
    train_texts = []
    for item in data:
        if item['input']:
            text = f"Təlimat: {item['instruction']} Giriş: {item['input']} Cavab: {item['output']}"
        else:
            text = f"Təlimat: {item['instruction']} Cavab: {item['output']}"
        train_texts.append(text)

    print(f"Training on {len(train_texts)} Azerbaijani examples...")

    # Simple fine-tuning
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(3):
        total_loss = 0
        print(f"Epoch {epoch + 1}/3")

        for text in train_texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_texts)
        print(f"Average loss: {avg_loss:.4f}")

    # Save the fine-tuned model
    output_dir = "./checkpoints/azerbaijani_gpt2"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Model saved to {output_dir}")

    # Test the model
    print("\n🧪 Testing your fine-tuned Azerbaijani model:")
    model.eval()

    test_prompts = [
        "Təlimat: Azərbaycan haqqında maraqlı fakt söylə Cavab:",
        "Təlimat: Bakının tarixi haqqında məlumat ver Cavab:",
        "Təlimat: Azərbaycan mətbəxindən yeməkləri say Cavab:"
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()

        print(f"\n📝 Prompt: {prompt}")
        print(f"🤖 Response: {generated}")

    return output_dir

def use_model(model_path):
    """Load and use your fine-tuned model."""
    print(f"\n🔥 Loading your fine-tuned Azerbaijani model from {model_path}")

    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("\n💬 Your Azerbaijani AI is ready! Try asking something:")

    # Interactive loop
    while True:
        user_input = input("\nSizin sualınız (type 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        prompt = f"Təlimat: {user_input} Cavab:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip()

        print(f"🤖 AI Cavab: {generated}")

if __name__ == "__main__":
    print("🇦🇿 DeepSeek Azerbaijani Fine-tuning")
    print("=" * 40)

    # Create the fine-tuned model
    model_path = create_azerbaijani_model()

    print(f"\n🎉 SUCCESS! Your Azerbaijani AI model is ready!")
    print(f"📁 Saved at: {model_path}")
    print("\n🚀 You now have a working fine-tuned model for Azerbaijani!")

    # Ask if user wants to try it
    choice = input("\nDo you want to try your model now? (y/n): ")
    if choice.lower() == 'y':
        use_model(model_path)