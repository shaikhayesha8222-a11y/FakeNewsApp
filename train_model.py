import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Synthetic Dataset for Cloud Building
data = {
    'text': [
        "Government announces new education policy for universities.",
        "Breaking: Alien spacecraft landed in Mumbai yesterday night!",
        "Scientists discover new species of deep sea fish.",
        "Drink hot lemon water to cure all diseases instantly!",
        "Stock market reaches new high following economic report.",
        "Miracle cure found that removes 100% wrinkles overnight!"
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = Real, 0 = Fake
}

df = pd.DataFrame(data)

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# Save .pkl files inside cloud environment
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("Model trained successfully on Cloud!")
