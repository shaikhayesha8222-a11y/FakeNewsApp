from flask import Flask, render_template, request
import joblib
from pyngrok import ngrok

app = Flask(__name__)

# Load Model and Vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form.get('news_text', '').strip()
        
        if not news_text:
            return render_template('index.html', prediction="Kripya koi text enter karein.")
            
        lower_text = news_text.lower()
        
        # Rule-based detection for viral claims
        fake_triggers = [
            "15 days of total darkness", "darkness", "aliens", "miracle herb", 
            "secret potion", "cures all", "drinking hot water", "raw onions",
            "flying human", "100% guaranteed", "earth stop rotating", "mind control"
        ]
        
        if any(trigger in lower_text for trigger in fake_triggers):
            prediction = 'FAKE'
        else:
            vectorized_text = vectorizer.transform([news_text])
            prediction = model.predict(vectorized_text)[0]

        return render_template('index.html', prediction=prediction, text=news_text)

if __name__ == '__main__':
    app.run()
     