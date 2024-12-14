from flask import Flask, render_template, request, jsonify
from models import model, tokenizer  # Import model and tokenizer from models package
from models.model import predict_prediction  # Import predict_moderation function
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import torch



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/moderate', methods=['POST'])
def moderate():
    content = request.form['text']
    
    # Use model and tokenizer to get moderation prediction
    response = classify_text(model, tokenizer, content)

    # Get the percentage values for appropriate and inappropriate
    appropriate_percentage = response["probabilities"]["appropriate"] * 100
    inappropriate_percentage = response["probabilities"]["inappropriate"] * 100
    
    return jsonify({
        'moderated': 'Yes' if response['predicted_class'] == 'Appropriate' else 'No',
        'appropriate_prob': response["probabilities"]["appropriate"],
        'inappropriate_prob': response["probabilities"]["inappropriate"],
        'appropriate_percentage': appropriate_percentage,
        'inappropriate_percentage': inappropriate_percentage
    })

    
stop_words = set(stopwords.words('english'))
stop_words.add("rt") # adding rt to remove retweet in dataset

# Removing Emojis
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# Replacing user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "", raw_text)
    return text

# Removing URLs
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)
    return text

# Removing Unnecessary Symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')
    text = text.replace(".", '')
    text = text.replace(",", '')
    text = text.replace("#", '')
    text = text.replace(":", '')
    text = text.replace("?", '')
    return text

# Stemming
def stemming(raw_text):
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in raw_text.split()]
    return ' '.join(words)

# Removing stopwords
def remove_stopwords(raw_text):
    tokenize = word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = ' '.join(text)
    return text

def preprocess(data):
    clean = []
    clean = [text.lower() for text in data]
    clean = [change_user(text) for text in clean]
    clean = [remove_entity(text) for text in clean]
    clean = [remove_url(text) for text in clean]
    clean = [remove_noise_symbols(text) for text in clean]
    clean = [stemming(text) for text in clean]
    clean = [remove_stopwords(text) for text in clean]

    return clean

def classify_text(model, tokenizer, text):

    # Preprocess text
    text_list = [text]
    preprocessed_text = preprocess(text_list)[0]

    # Tokenize the text
    tokenized_input = tokenizer(preprocessed_text, return_tensors='pt')

    # Get model predictions
    output = model(**tokenized_input)
    logits = output.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    if predicted_label==0:
        predicted_label = "Appropriate"
    else:
        predicted_label = "Inappropriate"

    probability_class_0 = probabilities[0][0].item()
    probability_class_1 = probabilities[0][1].item()

    # Formulate response
    response = {
        "text": text,
        "predicted_class": predicted_label,
        "probabilities": {
            "appropriate": probability_class_0,
            "inappropriate": probability_class_1
        }
    }
    return response
