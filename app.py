import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

MODEL_PATH = "model/pytorch_model.bin"
MODEL_URL = "https://huggingface.co/username/your-model/resolve/main/pytorch_model.bin"

# Download model if not exists
os.makedirs("model", exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("model", num_labels=2)
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({'prediction': 'Real' if predicted_class == 1 else 'Fake'})

