import torch
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt 
import numpy as np
import os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'fine_tuned_bert_model')
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    return model, tokenizer

def predict_prediction(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1).squeeze()
        prediction = torch.argmax(probs).item()
        # Probability of being "appropriate" (class 0) and "inappropriate" (class 1)
        return prediction, probs[0].item(), probs[1].item()

def plot_moderation_probabilities(appropriate_prob, inappropriate_prob):
    """Plot the probabilities for appropriate and inappropriate content."""
    labels = ['Appropriate', 'Inappropriate']
    probabilities = [appropriate_prob, inappropriate_prob]
    
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probabilities, color=['green', 'red'])
    plt.xlabel("Content Category")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.title("Content Moderation Probabilities")
    plt.show()