import requests
from flask import Flask, render_template, request

app = Flask(__name__)

def check_fact_api(query):
    # Google Fact Check Tools API
    url = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={query}&key=YOUR_GOOGLE_API_KEY"
    response = requests.get(url).json()

    if "claims" in response:
        rating = response["claims"][0]["claimReview"][0]["textualRating"]
        return f"Fact Check Result: {rating}"
    return None

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get('news') or ""

    # 1. First check Live Fact-Checking API
    api_result = check_fact_api(news_text)
    if api_result:
        return render_template('index.html', prediction_text=f"Live Fact-Check: {api_result}")

    # 2. Fallback to Machine Learning Model
    # (Your ML Model code here)
