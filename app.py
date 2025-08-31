# app.py
import streamlit as st
import pickle
import pandas as pd
import re

def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['vectorizer']

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]
    
    return {
        'prediction': 'Positive' if prediction == 1 else 'Negative',
        'confidence': max(probability),
        'probabilities': {
            'Negative': probability[0],
            'Positive': probability[1]
        }
    }

# Streamlit app
st.title("🎭 Sentiment Analysis App")
st.write("Analyze the sentiment of movie reviews using machine learning!")

# Load model
model, vectorizer = load_model()

# Input text
user_input = st.text_area("Enter your review:", height=100)

if st.button("Analyze Sentiment"):
    if user_input:
        result = predict_sentiment(user_input, model, vectorizer)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", result['prediction'])
        with col2:
            st.metric("Confidence", f"{result['confidence']:.2%}")
        
        # Probability chart
        prob_df = pd.DataFrame({
            'Sentiment': ['Negative', 'Positive'],
            'Probability': [result['probabilities']['Negative'], 
                          result['probabilities']['Positive']]
        })
        st.bar_chart(prob_df.set_index('Sentiment'))