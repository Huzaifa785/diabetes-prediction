from flask import Flask, render_template, request, jsonify
from joblib import load
import numpy as np
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure PaLM API
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

# Load diabetes prediction model and scaler
model = load("Diabetes_model(KNN).joblib")
scaler = load("Diabetes_model_scalar.joblib")

# Load diet recommendation model from PaLM API
diet_model = genai.GenerativeModel('gemini-pro')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_diabetes():
    if request.method == 'GET':
        return render_template('predict.html')
    else:
        # Get user data from request
        user_data = request.json.get('user')

        # Extract 'user' data from JSON
        # Ensure user data is provided and in the correct format
        if user_data is None or not isinstance(user_data, list) or len(user_data) != 7:
            return jsonify({'error': 'Invalid user data format. Expected a list of 7 elements.'}), 400

        try:
            # Reshape and scale user data
            np_user = np.asarray(user_data)
            re_user = np_user.reshape(1, -1)
            # Transform using pre-trained scaler
            scaled_user = scaler.transform(re_user)
            # Make prediction
            prediction = model.predict(scaled_user)
            # Check if the person has diabetes
            has_diabetes = int(prediction[0])

            if has_diabetes:
                # If the person has diabetes, generate diet recommendation
                diet_response = genai.chat(
                    context="Recommend a diet for a person with diabetes.",
                    messages="What diet would you recommend for someone with diabetes?"
                )
                diet_recommendation = diet_response.last
            else:
                diet_recommendation = "No specific diet recommendation needed."

            # Return prediction and diet recommendation
            return jsonify({'prediction': has_diabetes, 'diet_recommendation': diet_recommendation})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)