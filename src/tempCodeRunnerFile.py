from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
import os

app = Flask(__name__)

# 1️⃣ Load the fine-tuned (federated) model
MODEL_PATH = "models/aggregated_model"

# Use pre-trained tokenizer from Hugging Face hub
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load your fine-tuned model weights
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model path '{MODEL_PATH}' does not exist.")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# 2️⃣ Define predict route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    sentiment = "positive" if prediction == 1 else "negative"

    return jsonify({
        "text": text,
        "prediction": sentiment
    })

# 3️⃣ Root route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Federated Sentiment Analysis API is running!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
