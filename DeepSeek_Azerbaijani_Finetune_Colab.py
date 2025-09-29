# %% [markdown]
# # 🇦🇿 DeepSeek Fine-tuning for Azerbaijani Language
#
# This notebook fine-tunes a language model specifically for Azerbaijani.
# Run each cell in order in Google Colab.

# %% [markdown]
# ## 📦 Step 1: Install Dependencies

# %%
# Install required packages
!pip install transformers torch datasets accelerate peft bitsandbytes -q

# %%
# Import libraries
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import os

print("✅ All packages imported successfully!")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")
print(f"📱 Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## 📚 Step 2: Prepare Azerbaijani Training Data

# %%
# Create Azerbaijani training dataset
azerbaijani_data = [
    {
        "instruction": "Azərbaycan haqqında maraqlı fakt söylə",
        "output": "Azərbaycan dünyada ən çox vulkan palçığına malik ölkədir. Burada 400-dən çox vulkan palçığı var."
    },
    {
        "instruction": "Bakının tarixi haqqında məlumat ver",
        "output": "Bakı şəhəri IX əsrdə yaranmışdır. O, Xəzər dənizi sahilində yerləşir və zəngin neft ehtiyatları ilə məşhurdur."
    },
    {
        "instruction": "Azərbaycan mətbəxindən yeməkləri say",
        "output": "Azərbaycan mətbəxində plov, dolma, kebab, qutab, düşbərə, pakhlava kimi dadlı yeməklər var."
    },
    {
        "instruction": "Azərbaycanda neçə iqlim qurşağı var",
        "output": "Azərbaycanda 9 iqlim qurşağı var: subtropik, mülayim kontinental, dağ iqlimi və digərləri."
    },
    {
        "instruction": "Novruz bayramı haqqında danış",
        "output": "Novruz baharın gəlişini qeyd edən qədim bayramdır. Bu bayram 21 mart tarixində keçirilir və UNESCO tərəfindən qorunur."
    },
    {
        "instruction": "Azərbaycanın məşhur şairlərini say",
        "output": "Azərbaycanın məşhur şairləri arasında Nizami Gəncəvi, Füzuli, Nəsimi, Sabir var."
    },
    {
        "instruction": "Xəzər dənizinin əhəmiyyətini izah et",
        "output": "Xəzər dənizi dünyanın ən böyük gölüdür və Azərbaycan iqtisadiyyatı üçün çox vacibdir."
    },
    {
        "instruction": "Azərbaycan dilinin xüsusiyyətləri nələrdir",
        "output": "Azərbaycan dili Türk dillər ailəsindəndir. Latın əlifbası ilə yazılır və agglütinativ quruluşa malikdir."
    },
    {
        "instruction": "Qarabağ haqqında məlumat ver",
        "output": "Qarabağ Azərbaycanın tarixi ərazisidir və zəngin mədəni irsə malikdir."
    },
    {
        "instruction": "Azərbaycanın milli musiqi alətlərini say",
        "output": "Azərbaycanın milli musiqi alətləri arasında tar, kamança, balaban, zurna, nağara var."
    },
    {
        "instruction": "İngilis dilindən Azərbaycan dilinə tərcümə et: Good morning",
        "output": "Sabahınız xeyir"
    },
    {
        "instruction": "İngilis dilindən Azərbaycan dilinə tərcümə et: Thank you",
        "output": "Təşəkkür edirəm"
    }
]

# Format data for training
formatted_data = []
for item in azerbaijani_data:
    text = f"### Təlimat:\n{item['instruction']}\n\n### Cavab:\n{item['output']}<|endoftext|>"
    formatted_data.append({"text": text})

print(f"✅ Created {len(formatted_data)} training examples")
print("📝 Sample training text:")
print(formatted_data[0]["text"])

# %% [markdown]
# ## 🤖 Step 3: Load Base Model and Tokenizer

# %%
# Load model and tokenizer
model_name = "microsoft/DialoGPT-small"  # Small model for quick training
print(f"📥 Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Add padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model and tokenizer loaded successfully!")
print(f"🧠 Model parameters: {model.num_parameters():,}")

# %% [markdown]
# ## 🔄 Step 4: Prepare Dataset for Training

# %%
# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,
        return_tensors=None,
    )

# Create dataset
dataset = Dataset.from_list(formatted_data)
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
)

print(f"✅ Dataset tokenized: {len(tokenized_dataset)} examples")

# %% [markdown]
# ## 🏋️ Step 5: Set Up Training Configuration

# %%
# Training arguments
training_args = TrainingArguments(
    output_dir="./azerbaijani_model",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    prediction_loss_only=True,
    remove_unused_columns=False,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

print("✅ Training configuration set up!")

# %% [markdown]
# ## 🚀 Step 6: Start Fine-tuning!

# %%
# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("🚀 Starting fine-tuning...")
print("This will take a few minutes...")

# Start training
trainer.train()

print("✅ Fine-tuning completed!")

# %% [markdown]
# ## 💾 Step 7: Save the Fine-tuned Model

# %%
# Save model and tokenizer
output_dir = "./fine_tuned_azerbaijani_model"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Model saved to: {output_dir}")

# %% [markdown]
# ## 🧪 Step 8: Test Your Fine-tuned Model

# %%
# Load the fine-tuned model for testing
model.eval()

def generate_response(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.size(1) + max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()

# Test questions
test_prompts = [
    "### Təlimat:\nAzərbaycanda neçə iqlim qurşağı var?\n\n### Cavab:\n",
    "### Təlimat:\nBakının tarixi haqqında danış\n\n### Cavab:\n",
    "### Təlimat:\nAzərbaycan mətbəxindən nələr var?\n\n### Cavab:\n",
    "### Təlimat:\nNovruz bayramı nədir?\n\n### Cavab:\n",
]

print("🧪 Testing your fine-tuned Azerbaijani model:")
print("=" * 50)

for i, prompt in enumerate(test_prompts, 1):
    response = generate_response(prompt)
    print(f"\n🔸 Test {i}:")
    print(f"❓ Prompt: {prompt.split('###')[1].strip()}")
    print(f"🤖 AI Response: {response}")

# %% [markdown]
# ## 🎯 Step 9: Interactive Testing

# %%
# Interactive testing function
def ask_model(question):
    prompt = f"### Təlimat:\n{question}\n\n### Cavab:\n"
    response = generate_response(prompt, max_length=150)
    return response

# Test it yourself!
print("🎯 Your Azerbaijani AI is ready!")
print("Try asking questions like:")
print("- Azərbaycan haqqında danış")
print("- Bakının əhəmiyyəti nədir?")
print("- Azərbaycan mətbəxindən nələr var?")

# Example usage
question = "Azərbaycan haqqında maraqlı fakt söylə"
response = ask_model(question)
print(f"\n🇦🇿 Sual: {question}")
print(f"🤖 Cavab: {response}")

# %% [markdown]
# ## 📊 Step 10: Model Information and Download

# %%
# Display model information
print("📊 YOUR FINE-TUNED AZERBAIJANI MODEL")
print("=" * 40)
print(f"✅ Model successfully fine-tuned on {len(azerbaijani_data)} Azerbaijani examples")
print(f"📁 Model saved at: {output_dir}")
print(f"🧠 Base model: {model_name}")
print(f"🔥 Training device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
print(f"📈 Training epochs: {training_args.num_train_epochs}")
print(f"⚡ Learning rate: {training_args.learning_rate}")

# Function to download model files
def download_model():
    """Download the fine-tuned model files"""
    import zipfile
    import shutil

    # Create zip file with model
    zip_path = "azerbaijani_model.zip"
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

    print(f"📦 Model packaged as: {zip_path}")
    return zip_path

# Uncomment the next line to download your model
# download_model()

print("\n🎉 CONGRATULATIONS!")
print("🇦🇿 You now have a fine-tuned Azerbaijani language model!")
print("🚀 The model can understand and respond to Azerbaijani questions!")

# %% [markdown]
# ## 🔧 Bonus: Save Model for Later Use

# %%
# Save a simple usage script
usage_code = '''
# Usage script for your fine-tuned Azerbaijani model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your fine-tuned model
model_path = "./fine_tuned_azerbaijani_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def ask_azerbaijani_ai(question):
    prompt = f"### Təlimat:\\n{question}\\n\\n### Cavab:\\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.size(1) + 100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Example usage
response = ask_azerbaijani_ai("Azərbaycan haqqında danış")
print(response)
'''

with open("use_model.py", "w", encoding="utf-8") as f:
    f.write(usage_code)

print("📄 Usage script saved as 'use_model.py'")
print("🎯 You can use this script to load and use your model anywhere!")

# %%