import requests
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Fallback ML model setup
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    model, vectorizer = None, None

# Paste your copied Google Fact Check API key here
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

def verify_claim_via_api(query_text):
    """
    Live API call to Google Fact Check database
    """
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "AIzaSyBSrAqkpdcm_dfxSjrY2pvC9DooARrBoiQ":
        return None

    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query_text}&key={GOOGLE_API_KEY}"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "claims" in data and len(data["claims"]) > 0:
                claim = data["claims"][0]
                review = claim["claimReview"][0]
                rating = review.get("textualRating", "Verified Fact Check")
                publisher = review.get("publisher", {}).get("name", "Fact Checker")
                
                rating_lower = rating.lower()
                if "false" in rating_lower or "fake" in rating_lower or "misleading" in rating_lower or "incorrect" in rating_lower:
                    return f"{rating} (Verified by {publisher}) ⚠️"
                else:
                    return f"{rating} (Verified by {publisher}) ✅"
    except Exception as e:
        print("API Lookup error:", e)
        
    return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form.get('news') or ""
        
        if not news_text.strip():
            return render_template('index.html', prediction_text="Kripya text enter karein.")

        # 1. Primary Layer: Real-Time Dynamic API Fact-Check
        api_result = verify_claim_via_api(news_text)
        
        if api_result:
            result = f"Live Fact Check: {api_result}"
        else:
            # 2. Secondary Layer: Machine Learning Pattern Prediction (Fallback)
            if model and vectorizer:
                data = [news_text]
                vect = vectorizer.transform(data)
                prediction = model.predict(vect)
                result = "Real News ✅" if prediction[0] == 1 else "Fake / Misleading News ⚠️"
            else:
                result = "Analysis Completed"

        return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run()
