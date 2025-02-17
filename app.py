from flask import Flask, request, make_response ,render_template,jsonify
import re
import joblib

cv = joblib.load('vectorizer.joblib')
model = joblib.load('logistic_regression_model.joblib')


app = Flask(__name__,template_folder='templates')


@app.route('/')
def index():
    return render_template('main.html')

def preprocessor(e):
    e = re.sub(r'[^a-zA-Z0-9]', ' ', e) 
    e = e.lower()
    return e

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email_text = data.get("email", "")
    
    if not email_text:
        return jsonify({"error": "No email content provided"}), 400
    
    # Preprocess input and transform with vectorizer
    processed_text = preprocessor(email_text)
    transformed_text = cv.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(transformed_text)
    if prediction[0] == 0:
        return jsonify({"spam": False})
    else:
        return jsonify({"spam": True})

if __name__ == '__main__':
    app.run(debug=True)

