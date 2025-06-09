import nltk
nltk.download('stopwords')

from flask import Flask, request, jsonify
from model_utils import model_svm, model_nb, model_rf, vectorizer, clean_text, extract_email_body
import os


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    email = data.get('email')
    model_choice = data.get('model')

    if not email:
        return jsonify({'error': 'Aucun contenu d\'email reçu'}), 400

    email_body = extract_email_body(email)
    cleaned = clean_text(email_body)
    X = vectorizer.transform([cleaned])

    if model_choice == 'SVM':
        prediction = model_svm.predict(X)[0]
    elif model_choice == 'Naïve Bayes':
        prediction = model_nb.predict(X)[0]
    elif model_choice == 'Random Forest':
        prediction = model_rf.predict(X)[0]
    else:
        return jsonify({'error': 'Modèle non reconnu'}), 400

    result = "SPAM" if prediction == 1 else "NON SPAM"
    return jsonify({'model': model_choice.upper(), 'prediction': result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
