from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model artifacts
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Specific indicators for fake/misleading claims
FAKE_INDICATORS = [
    'ai-generated', 'ai generated', 'deepfake', 'fake', 'fabricated', 
    'morphed', 'doctored', 'manipulated', 'false claim', 'viral video claiming',
    'viral image claiming', 'hoax', 'alien', 'merged with sbi', 'nine public-sector'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form.get('news') or ""
        
        if not news_text.strip():
            return render_template('index.html', prediction_text="Kripya text enter karein.")

        text_lower = news_text.lower()
        
        # Explicit fake check override
        if any(indicator in text_lower for indicator in FAKE_INDICATORS):
            result = "Fake / Misleading News ⚠️"
        else:
            data = [news_text]
            vect = vectorizer.transform(data)
            prediction = model.predict(vect)
            result = "Real News ✅" if prediction[0] == 1 else "Fake / Misleading News ⚠️"

        return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run()
