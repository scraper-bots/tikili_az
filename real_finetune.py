#!/usr/bin/env python3
"""
REAL fine-tuning - actual neural network training with gradient descent.
This will download a model and actually train it on Azerbaijani data.
"""

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
import json
import os
from tqdm import tqdm

def real_finetune():
    """Actually fine-tune a neural network on Azerbaijani data."""
    print("🔥 REAL FINE-TUNING - Training Neural Network Weights")
    print("=" * 60)

    # Check device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download and load ACTUAL model
    print("📥 Downloading actual pre-trained model...")
    model_name = "gpt2"  # Real neural network
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Move to device
    model.to(device)

    # Load training data
    print("📚 Loading Azerbaijani training data...")
    with open('data/processed/extended_azerbaijani.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare training texts
    training_texts = []
    for item in data:
        if item['input']:
            text = f"Sual: {item['instruction']} Giriş: {item['input']} Cavab: {item['output']}<|endoftext|>"
        else:
            text = f"Sual: {item['instruction']} Cavab: {item['output']}<|endoftext|>"
        training_texts.append(text)

    print(f"Training on {len(training_texts)} examples")

    # Tokenize training data
    print("🔤 Tokenizing training data...")
    tokenized_texts = []
    for text in training_texts:
        tokens = tokenizer.encode(text, truncation=True, max_length=512)
        tokenized_texts.append(torch.tensor(tokens))

    # Set up optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # ACTUAL TRAINING LOOP WITH REAL GRADIENT DESCENT
    print("🧠 Starting REAL neural network training...")
    model.train()

    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Progress bar
        pbar = tqdm(tokenized_texts, desc="Training")

        for batch_tokens in pbar:
            # Move to device
            batch_tokens = batch_tokens.to(device)

            # Create input and target (shift by one for language modeling)
            inputs = batch_tokens[:-1].unsqueeze(0)  # All tokens except last
            targets = batch_tokens[1:].unsqueeze(0)  # All tokens except first

            # Skip if too short
            if inputs.size(1) < 2:
                continue

            # Forward pass
            outputs = model(inputs)
            logits = outputs.logits

            # Calculate loss (cross-entropy)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Backward pass (REAL GRADIENT DESCENT)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(tokenized_texts)
        print(f"Average loss: {avg_loss:.4f}")

    # Save the ACTUALLY TRAINED model
    output_dir = "./checkpoints/real_azerbaijani_gpt2"
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ REAL fine-tuned model saved to: {output_dir}")

    # Test the trained model
    print("\n🧪 Testing REAL fine-tuned model:")
    model.eval()

    test_prompts = [
        "Sual: Azərbaycanda neçə iqlim qurşağı var? Cavab:",
        "Sual: Bakının tarixi haqqında danış Cavab:",
        "Sual: Azərbaycan mətbəxindən nələr var? Cavab:"
    ]

    with torch.no_grad():
        for prompt in test_prompts:
            inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

            # Generate with the TRAINED neural network
            outputs = model.generate(
                inputs,
                max_length=inputs.size(1) + 100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = response[len(prompt):].strip()

            print(f"\n📝 Prompt: {prompt}")
            print(f"🤖 Neural Network Response: {generated}")

    return output_dir

if __name__ == "__main__":
    try:
        model_path = real_finetune()
        print(f"\n🎉 SUCCESS! You now have a REAL fine-tuned neural network!")
        print(f"📁 Model saved at: {model_path}")
        print("🧠 This model actually learned through gradient descent!")
        print("🔥 The weights were actually updated through backpropagation!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("This requires downloading models and actual training.")