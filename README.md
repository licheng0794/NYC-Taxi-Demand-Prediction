## Project Description

### the structure of the project

<img width="557" alt="Project file structure" src="https://github.com/user-attachments/assets/b634e9fa-ce0c-4508-81f9-66e889c2d2cc" />



#### 1. src/feature_store.py: store key features in Hopworks and feature versioning

#### 2. src train_xgb.py: XGBoost Model Training with CI/CD (MLflow Tracking). 
- access the features and versioning from Hopsworks
- split training, validation and test data based on time
- loging the parameters of xgboost and metrics (MAE and R2) with MLflow
  
#### 3. .github/workflows/train.yml: CI/CD Pipeline Using GitHub Actions. This workflow automatically triggers model retraining when code is pushed or on a schedule.

#### 4. src/predict_api.py. This Flask API serves predictions from the trained model in real time. Deploy this API on a cloud service (like Heroku, AWS, or using Docker containers) so that it is accessible publicly.

#### 5. src/dashboard.py. Visualize model performance and opertional insights. 
- I used the predicted demand from offline xgboost model for visualization. In real cases, it allow users to input parameters for real-time predictions (which are sent to the prediction API endpoint in step 4). 
- I assume we already got model versions and metrics for convience. In real cases, we can access logs and model performance in MLflow

#### 6. notebook/Taxi Demand Prediction.ipynb includes: 
- explicit data analysis and data processing
- model builing, comparison and conclusion 
