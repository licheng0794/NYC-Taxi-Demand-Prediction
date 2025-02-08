import hopsworks
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# Step 1: Connect to Hopsworks and Retrieve Features
# ---------------------------
# Log in to your Hopsworks project
project = hopsworks.login()  # This will prompt for credentials if needed.
fs = project.get_feature_store()

# Retrieve the feature group containing your taxi demand features.
# Adjust the name and version according to your configuration.
feature_group = fs.get_feature_group(name="taxi_demand_features", version=1)

# Read the features into a DataFrame.
# This DataFrame should contain columns like 'PULocationID', 'Month', 'day', 'hour', 'trips', etc.
df_features = feature_group.read()

# ---------------------------
# Step 2: Preprocess the Data
# ---------------------------
# For example, assume we predict "trips" and all other columns are features.
# If necessary, perform additional preprocessing (e.g., one-hot encoding).

# Split into train/validation/test sets.
train = df_features[df_features['month'] <= 8]  # Training (First 8 months)
valid = df_features[(df_features['month'] > 8) & (df_features['month'] <= 10)]  # Validation (9th and 10th month)
holdout = df_features[df_features['month'] >= 11]  # Holdout (Final month)

X_train, y_train = train.drop(columns=['current_trips', 'PULocationID']), train['current_trips']
X_valid, y_valid = valid.drop(columns=['current_trips', 'PULocationID']), valid['current_trips']
X_test, y_test = holdout.drop(columns=['current_trips', 'PULocationID']), holdout['current_trips']
# ---------------------------
# Step 3: Train the XGBoost Model and Log with MLflow
# ---------------------------
# Convert training sets to XGBoost's DMatrix format.
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest  = xgb.DMatrix(X_test, label=y_test)

# Define hyperparameters for regression.
params = {
    'objective': 'reg:squarederror',
    'max_depth': 4,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
num_round = 1000

# Start an MLflow experiment run.
mlflow.set_experiment("Taxi Demand Prediction")
with mlflow.start_run():
    # Log the feature version used.
    mlflow.log_param("feature_version", "v1.0")
    
    # Log hyperparameters.
    for key, value in params.items():
        mlflow.log_param(key, value)
    mlflow.log_param("num_round", num_round)
    
    # Train the XGBoost model.
    model = xgb.train(params, dtrain, num_round, evals=[(dvalid, "Validation")])
    
    # Predict on the test set.
    y_pred = model.predict(dtest)
    
    # Evaluate the model.
    mse = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mlflow.log_metric("MAE", mse)
    mlflow.log_metric("R2", r2)
    
    # Log the trained model.
    mlflow.xgboost.log_model(model, "model")
    
    print("Training complete. MSE: {:.2f}, R2: {:.2f}".format(mse, r2))
