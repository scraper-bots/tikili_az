#!/usr/bin/env python3
"""
Update your fine-tuned model with new data.
"""

from instant_model import AzerbaijaniModel

def update_model():
    """Update the model with extended training data."""
    print("🔄 Updating your Azerbaijani model with climate information...")

    # Create new model with extended data
    model = AzerbaijaniModel()

    # Use extended dataset
    import json
    with open('data/processed/extended_azerbaijani.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load the data
    for item in data:
        instruction = item['instruction'].lower()
        response = item['output']

        # Store direct mappings
        model.responses[instruction] = response

        # Extract keywords for pattern matching
        keywords = model._extract_keywords(instruction)
        for keyword in keywords:
            if keyword not in model.patterns:
                model.patterns[keyword] = []
            model.patterns[keyword].append(response)

    # Save updated model
    model_path = "./checkpoints/azerbaijani_model.pkl"
    model.save(model_path)

    print(f"✅ Model updated and saved to: {model_path}")

    # Test the updated model
    print("\n🧪 Testing updated model:")

    test_questions = [
        "nece iqlim qursagi var?",
        "Azərbaycanda neçə iqlim qurşağı var?",
        "Azərbaycanın iqlimi necədir?",
        "təbiət haqqında danış"
    ]

    for question in test_questions:
        response = model.generate(question)
        print(f"\n📝 Sual: {question}")
        print(f"🤖 Cavab: {response}")

if __name__ == "__main__":
    update_model()