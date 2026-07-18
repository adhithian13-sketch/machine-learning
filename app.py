import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')
scaler = joblib.load('standard_scaler.joblib')

# Get the feature names used during training for consistent input order
# Assuming X from the previous step is still available in the kernel state
feature_names = X.columns.tolist()

print("Model and Scaler loaded successfully.")
print(f"Expected features for input: {feature_names}")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(force=True)

    if not data or not isinstance(data, list):
        return jsonify({"error": "Input data must be a list of dictionaries"}), 400

    try:
        # Convert input data to DataFrame, ensuring correct column order
        input_df = pd.DataFrame(data)
        input_df = input_df[feature_names] # Ensure column order matches training data

        # Scale the input features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        results = []
        for i in range(len(prediction)):
            results.append({
                "prediction": int(prediction[i]),
                "probability_class_0": float(prediction_proba[i][0]),
                "probability_class_1": float(prediction_proba[i][1])
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

print("Flask app and '/predict' endpoint defined.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)



