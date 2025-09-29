#!/usr/bin/env python3
# Usage script for your fine-tuned Azerbaijani model

from instant_model import AzerbaijaniModel

# Load your fine-tuned model
model = AzerbaijaniModel.load('./checkpoints/azerbaijani_model.pkl')

# Use the model
while True:
    question = input("Sualınız: ")
    if question.lower() == 'quit':
        break
    response = model.generate(question)
    print(f"Cavab: {response}")
