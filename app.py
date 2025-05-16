import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load from Hugging Face model repo
model_name = "ASH4787/bert-fake-news"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits).item()
    result = "Real" if predicted_class == 1 else "Fake"
    return jsonify({'prediction': result})

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
