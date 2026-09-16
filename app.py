from flask import Flask, render_template, request
from duckduckgo_search import DDGS

app = Flask(__name__)

def check_live_fact(query):
    """
    Searches web for existing fact-checks on the input query.
    """
    try:
        with DDGS() as ddgs:
            # Fact-check sources search query
            search_query = f"{query} fact check fake or real"
            results = list(ddgs.text(search_query, max_results=3))
            
            if results:
                combined_text = " ".join([r['title'] + " " + r['body'] for r in results]).lower()
                
                # Check for strong debunks in web snippets
                if any(word in combined_text for word in ['fake', 'hoax', 'false', 'debunked', 'misleading', 'untrue', 'rumor']):
                    return "Fake / Misleading News ⚠️"
                elif any(word in combined_text for word in ['true', 'confirmed', 'verified', 'official report']):
                    return "Real News ✅"
    except Exception as e:
        print("Search API Error:", e)
    
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

        # 1. Direct Live Web Fact-Check Verification
        live_result = check_live_fact(news_text)
        
        if live_result:
            final_output = f"Result: {live_result} (Verified via Live Fact-Check Search)"
        else:
            # Fallback output if no online records exist
            final_output = "Result: Needs Verification / Unconfirmed News ⚠️"

        return render_template('index.html', prediction_text=final_output)

if __name__ == '__main__':
    app.run()
