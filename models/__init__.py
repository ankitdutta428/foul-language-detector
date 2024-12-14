# models/__init__.py

from .model import load_model

# Load the model and tokenizer on package import
model, tokenizer = load_model()
