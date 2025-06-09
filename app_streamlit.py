import streamlit as st
import requests

st.title("Détecteur de SPAM")

# Saisie email
email_input = st.text_area("Copie/colle ton email ici")

# Choix modèle
model = st.selectbox("Choisis le modèle", options=['SVM', 'Naïve Bayes', 'Random Forest'])

if st.button("Prédire"):
    if not email_input.strip():
        st.error("Merci de saisir un email.")
    else:
        # Appel API Flask
        url = "https://flask-api-oz82.onrender.com/predict"
        payload = {'email': email_input, 'model': model}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            st.success(f"Résultat avec {data['model']} : {data['prediction']}")
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur API : {e}")
