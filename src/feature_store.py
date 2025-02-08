import hopsworks
import pandas as pd

def get_feature_store():
    project = hopsworks.login()  # Logs into your Hopsworks instance
    fs = project.get_feature_store()
    return fs

def create_and_ingest_features(file_path, feature_group_name="taxi_demand_features", version=1):
    fs = get_feature_store()
    feature_group = fs.get_or_create_feature_group(
        name=feature_group_name,
        version=version,
        primary_key=["PULocationID", "Month", "day", "hour", "fare_amount", "passenger_count", "trip_distance"],
        description="Key features for predicting NYC taxi demand",
        online_enabled=True
    )
    df_features = pd.read_csv(file_path)
    # Insert features; note: in a real project, error handling and versioning logic would be added.
    feature_group.insert(df_features, write_options={"wait_for_job": True})
    return feature_group

if __name__ == "__main__":
    # Example call: ingest features from CSV located in the data/ folder
    create_and_ingest_features("../data/taxi_features.csv")