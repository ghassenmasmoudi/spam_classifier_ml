import joblib
import re
from email import message_from_string
import nltk
from nltk.corpus import stopwords
import os

# Ajouter le chemin du dossier nltk_data local (dans le même dossier que model_utils.py)
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
nltk.data.path.append(nltk_data_path)


# Charger les stopwords depuis le dossier local
stop_words = set(stopwords.words('english'))
# Charger le modèle SVM et le CountVectorizer

model_svm = joblib.load("model_svm.pkl")
model_nb = joblib.load("model_nb.pkl")
model_rf = joblib.load("model_rf.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def extract_email_body(text):
    msg = message_from_string(text)
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                body += part.get_payload(decode=True).decode(errors='ignore')
    else:
        body = msg.get_payload(decode=True).decode(errors='ignore')
    return body

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return " ".join([w for w in text.split() if w not in stop_words])
