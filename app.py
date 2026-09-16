from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Handles 'news', 'news_text', or any standard textarea name
        news_text = request.form.get('news') or request.form.get('news_text') or ""
        
        if not news_text.strip():
            return render_template('index.html', prediction_text="Kripya text enter karein.")

        # Vectorize and Predict
        data = [news_text]
        vect = vectorizer.transform(data)
        prediction = model.predict(vect)
        
        # Result logic
        result = "Real News ✅" if prediction[0] == 1 else "Fake News ⚠️"
        return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run()
