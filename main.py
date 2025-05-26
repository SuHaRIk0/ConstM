from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from flask_cors import CORS

vectorizer = joblib.load("vectorizer_1.pkl")
article_encoder = joblib.load("article_encoder_1.pkl")
model = load_model("constitution_model_1.h5")
law = pd.read_csv("law.csv")

app = Flask(__name__)
CORS(app)

def preprocess(text):
    tfidf_vector = vectorizer.transform([text])
    return tfidf_vector.toarray().astype(np.float32)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    vector = preprocess(text)
    verdict_pred, confidence_pred, article_pred = model.predict(vector)

    # Вердикт
    verdict_class = int(np.argmax(verdict_pred[0]))
    verdict_label = "Правда" if verdict_class == 1 else "Неправда"

    # Довіра
    confidence = float(confidence_pred[0][0])

    # Назва статті (декодуємо з індексу)
    article_index = int(np.argmax(article_pred[0]))
    article_label = article_encoder.inverse_transform([article_index])[0]

    # Текст статті за назвою
    row = law[law['section'] == article_label]
    if not row.empty:
        description = row.iloc[0]['text']
    else:
        description = "Відповідну статтю не знайдено."

    return jsonify({
        "verdict": verdict_label,
        "article": article_label,
        "description": description,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=False, port=5000)
