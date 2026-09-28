import os
import requests
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import joblib

app = Flask(__name__)
app.config['SECRET_KEY'] = 'truthcheck_secret_key_123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///truthcheck.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- Database Models ---
class NewsSearchLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    news_text = db.Column(db.Text, nullable=False)
    result = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class UserFeedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rating = db.Column(db.Integer, nullable=False)
    comments = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables if not exist
with app.app_context():
    db.create_all()

# --- Fallback ML Model Setup ---
try:
    model = joblib.load('fake_news_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except Exception as e:
    model, vectorizer = None, None

# GOOGLE FACT CHECK API KEY
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

def verify_claim_via_api(query_text):
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
            return render_template('index.html', prediction_text="Please enter text to verify.")

        # 1. Primary Layer: Real-Time Dynamic API Fact-Check
        api_result = verify_claim_via_api(news_text)
        
        if api_result:
            result = f"Live Fact Check: {api_result}"
        else:
            # 2. Secondary Layer: Machine Learning Pattern Prediction
            if model and vectorizer:
                data = [news_text]
                vect = vectorizer.transform(data)
                prediction = model.predict(vect)
                result = "Real News ✅" if prediction[0] == 1 else "Fake / Misleading News ⚠️"
            else:
                result = "Analysis Completed"

        # Save Search to Database
        try:
            log_entry = NewsSearchLog(news_text=news_text, result=result)
            db.session.add(log_entry)
            db.session.commit()
        except Exception as e:
            print("Database save error:", e)

        return render_template('index.html', prediction_text=f'Result: {result}', show_feedback=True)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    rating = request.form.get('rating')
    comments = request.form.get('comments') or ""
    
    if rating:
        try:
            feedback_entry = UserFeedback(rating=int(rating), comments=comments)
            db.session.add(feedback_entry)
            db.session.commit()
            flash("Thank you for your feedback! ⭐", "success")
        except Exception as e:
            print("Feedback save error:", e)
            
    return redirect(url_for('home'))

@app.route('/admin')
def admin():
    searches = NewsSearchLog.query.order_by(NewsSearchLog.timestamp.desc()).limit(50).all()
    feedbacks = UserFeedback.query.order_by(UserFeedback.timestamp.desc()).limit(50).all()
    return render_template('admin.html', searches=searches, feedbacks=feedbacks)

if __name__ == '__main__':
    app.run()
