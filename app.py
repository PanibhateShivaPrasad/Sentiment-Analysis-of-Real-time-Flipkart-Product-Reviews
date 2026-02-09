import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("Flipkart Review Sentiment Analyzer")
st.subheader("Product: YONEX MAVIS 350 Nylon Shuttle")

review = st.text_area("Enter your review")

if st.button("Analyze"):
    cleaned = clean_text(review)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        st.success("😊 Positive Review")
    else:
        st.error("😡 Negative Review")
