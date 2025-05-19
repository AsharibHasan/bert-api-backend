import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load model and tokenizer from Hugging Face
MODEL_NAME = "ASH4787/bert-fake-news"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

@app.route('/')
def home():
    return "✅ BERT Fake News Detector API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"📥 Received request: {data}")  # Log input payload

        if not data or "text" not in data:
            print("⚠️ No 'text' key found in payload.")
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Text is empty"}), 400

        # Tokenize and predict
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)

        label = "Real" if prediction.item() == 1 else "Fake"
        confidence_score = round(confidence.item(), 4)

        print(f"✅ Prediction: {label}, Confidence: {confidence_score}")

        return jsonify({
            "label": label,
            "confidence": confidence_score
        })

    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return jsonify({"error": "Internal Server Error", "details": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
