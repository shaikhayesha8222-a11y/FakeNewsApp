import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("Downloading full 10,000+ Kaggle news dataset on Render...")

# Direct URL stream for full datasets
fake_url = "https://raw.githubusercontent.com/ZenEnti/fake-news-analysis/main/Fake.csv"
true_url = "https://raw.githubusercontent.com/ZenEnti/fake-news-analysis/main/True.csv"

try:
    fake_df = pd.read_csv(fake_url)
    true_df = pd.read_csv(true_url)

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Title + Text context combining for high precision
    df['combined_text'] = df['title'].fillna('') + " " + df['text'].fillna('')

    X = df['combined_text']
    y = df['label']

    print(f"Successfully loaded {len(df)} news items!")

except Exception as e:
    print("Download error, using local fallback:", e)
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    fake_df['label'] = 0
    true_df['label'] = 1
    df = pd.concat([fake_df, true_df], ignore_index=True)
    X = df['title'].fillna('') + " " + df['text'].fillna('')
    y = df['label']

print("Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

print("Training Logistic Regression model on 10,000+ samples...")
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

# Save artifacts
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("10k Dataset Model Training Completed!")
