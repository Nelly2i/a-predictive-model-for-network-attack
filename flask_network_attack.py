from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and preprocessing components
model = load_model('Leslie_network_attack_model.h5')
#scaler = joblib.load('scaler.pkl')
#encoded_columns = joblib.load('encoded_columns.pkl')  # Ensure this is aligned with the training set

# Root endpoint for health check or API info
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Network Attack Prediction API is running!",
        "endpoints": {
            "/predict": "POST endpoint to predict network attacks. Provide network log data in JSON format."
        }
    })


# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict_attack():
    try:
        # Step 1: Receive JSON input from Postman
        data = request.get_json()

        # Extract the input data for prediction
        input_data = data['input']  # Expecting input as a list of 10 features
        if len(input_data) != 10:
            raise ValueError("Input must contain exactly 10 features.")

        # Step 2: Convert the input to a NumPy array
        input_array = np.array(input_data).reshape(1, 1, -1)  # Reshape to match model input shape

        # Step 3: Predict using the loaded model
        predictions = model.predict(input_array)

        # Step 4: Decode the predictions
        labels = ['normal', 'ipsweep', 'satan', 'portsweep', 'back']
        decoded_prediction = labels[np.argmax(predictions[0])]

        # Step 5: Return the prediction result
        return jsonify({"prediction": decoded_prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)