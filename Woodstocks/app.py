from flask import Flask, request, jsonify
import pandas as pd
import pickle
import json

app = Flask(__name__)

# Load model and metadata
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

@app.route('/')
def home():
    return "Woodstocks Prediction API - Use POST /predict endpoint"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get required features from metadata
        required_features = [f for f in metadata['features'] 
                          if f not in ['Date', 'next_day_change']]
        
        # Validate all required features are present
        missing = [f for f in required_features if f not in data]
        if missing:
            return jsonify({
                "error": "Missing required features",
                "missing": missing
            }), 400
        
        # Create input DataFrame with correct feature order
        input_df = pd.DataFrame({f: [data[f]] for f in required_features})
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            "prediction": float(prediction),
            "direction": "UP" if prediction > 0 else "DOWN",
            "confidence": min(99, abs(prediction)*100),
            "model": metadata['model_name']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)