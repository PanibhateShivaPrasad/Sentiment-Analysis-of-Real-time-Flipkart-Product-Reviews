import pandas as pd
import nltk
import re
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv(r"D:\Intership Tasks\ML task\Datasets\reviews_badminton\data.csv")

# Create Sentiment column from Ratings
df['Sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)

# Text cleaning setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Clean reviews
df["cleaned_review"] = df["Review text"].apply(clean_text)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df["cleaned_review"])
y = df["Sentiment"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("F1 Score:", f1_score(y_test, y_pred))

# Save model
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model training completed and files saved successfully!")
