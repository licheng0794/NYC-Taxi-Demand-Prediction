from flask import Flask, request, jsonify
import xgboost as xgb
import mlflow.xgboost
import pandas as pd

app = Flask(__name__)

# Load the model from MLflow (update the model URI as needed)
model = mlflow.xgboost.load_model("models:/TaxiDemandPrediction/Production")

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON input with feature values, e.g.:
    # {"PULocationID": 1, "Month": 1, "day": 15, "hour": 8, ...}
    data = request.json
    # Convert input to DataFrame (you might also perform necessary preprocessing)
    df_input = pd.DataFrame([data])
    # Ensure any categorical features are encoded consistently
    # (If using a feature store, you might join the latest feature data here)
    
    # Create DMatrix for prediction
    dmatrix = xgb.DMatrix(df_input)
    prediction = model.predict(dmatrix)
    return jsonify({"predicted_trips": float(prediction[0])})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
