"""
Trains the customer review sentiment model and saves the fitted TF-IDF
vectorizer and logistic regression classifier for the Streamlit app.

    python train_and_save_model.py
"""

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

print("Loading dataset...")
df = pd.read_csv('amazon_alexa.csv', sep='\t')
df = df.dropna(subset=['verified_reviews']).reset_index(drop=True)


def rating_to_sentiment(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'


df['sentiment'] = df['rating'].apply(rating_to_sentiment)

X = df['verified_reviews']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training model...")
clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_train_tfidf, y_train)

print(classification_report(y_test, clf.predict(X_test_tfidf)))

joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(clf, 'sentiment_model.joblib')

print("Saved: vectorizer.joblib, sentiment_model.joblib")
