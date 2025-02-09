import mlflow
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt

# """
# ###################### this part can be used to connect to MLflow #############
# # Set the MLflow tracking URI if not using the default local file-based store:
# # mlflow.set_tracking_uri("http://your-tracking-server:5000")

# # Specify your MLflow experiment ID or name.
# # For example, assume your experiment ID is "0"
# experiment_id = "0"

# # Retrieve run data from MLflow (you can filter or sort as needed)
# runs_df = mlflow.search_runs(experiment_ids=[experiment_id])

# # If you want to sort runs by start time (most recent first)
# runs_df = runs_df.sort_values(by="start_time", ascending=False)

# st.title("MLflow Model Performance Dashboard")

# # Display a table of run metrics (customize which metrics you want to display)
# st.subheader("Recent Runs")
# if not runs_df.empty:
#     # Select a few key columns
#     display_cols = ['run_id', 'metrics.MSE', 'metrics.R2', 'params.max_depth', 'params.eta']
#     st.dataframe(runs_df[display_cols])
# else:
#     st.write("No runs found for experiment id:", experiment_id)

# # Display the latest run's performance
# if not runs_df.empty:
#     latest_run = runs_df.iloc[0]
#     st.subheader("Latest Run Performance")
#     st.write(f"**Run ID:** {latest_run['run_id']}")
#     st.write(f"**MSE:** {latest_run['metrics.MSE']}")
#     st.write(f"**R²:** {latest_run['metrics.R2']}")
#     st.write(f"**Max Depth:** {latest_run['params.max_depth']}")
#     st.write(f"**Learning Rate (eta):** {latest_run['params.eta']}")

#     # You could also show additional details or even load artifacts if needed.
# else:
#     st.info("No MLflow run data available.")

# # Optionally, you can create charts from the runs data.
# if not runs_df.empty:
#     # For example, plot MSE over time
#     runs_df['start_time'] = pd.to_datetime(runs_df['start_time'], unit='ms')
#     mse_chart_data = runs_df.sort_values("start_time")[["start_time", "metrics.MSE"]].set_index("start_time")
#     st.line_chart(mse_chart_data)

# ############################################################################################################
# """


df = pd.read_csv('./data/PredictedTrips.csv')

df['predict_trips'] = df['predict_trips'].round().astype(int)

dfZone = pd.read_csv("./data/taxi_zone_lookup.csv")

df = df.merge(dfZone[['LocationID', 'Zone']], on='LocationID', how='left')

# ----------------------------
# Sidebar: Global Time Filter and Page Navigation
# ----------------------------
st.sidebar.header("Time Filter")
month = st.sidebar.number_input("Month", min_value=9, max_value=11, value=9)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=1)
hour = st.sidebar.number_input("Hour", min_value=0, max_value=23, value=0)

# Filter the data by time parameters
filtered_df = df[(df['month'] == month) & (df['day'] == day) & (df['hour'] == hour)]

# Sidebar: Page Selection
page = st.sidebar.radio("Select Page", ["Overall Map", "Top Locations", "Specific Location"])

# ----------------------------
# Page 1: Overall Map
# ----------------------------
if page == "Overall Map":
    st.title("Overall Ride Demand Prediction Map")
    if not filtered_df.empty:
        # Rename for mapping: st.pydeck_chart expects 'lat' and 'lon'
        map_df = filtered_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        view_state = pdk.ViewState(
            longitude=map_df['lon'].mean(),
            latitude=map_df['lat'].mean(),
            zoom=11,
            pitch=0
        )
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_radius="predict_trips",  # you might apply a scaling factor if needed
            get_fill_color="[255, 0, 0, 160]",
            pickable=True
        )
        deck = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip={
                "html": "<b>Zone:</b> {Zone}<br/><b>Predicted Ride Demand:</b> {predict_trips}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
        )
        st.pydeck_chart(deck)
    else:
        st.warning("No data found for the selected time parameters.")

# ----------------------------
# Page 2: Top Locations (Bar Plot)
# ----------------------------
elif page == "Top Locations":
    st.title("Top Locations by Estimated Ride Demand and Revenue")
    if not filtered_df.empty:
        # Sort by trips descending and get up to 10 locations
        top_locations = filtered_df.sort_values(by="predict_trips", ascending=False).head(10)
        
        # Create a bar plot using Matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        # ax1.bar(top_locations['PULocationID'].astype(str), top_locations['predict_trips'], color="skyblue")
        ax1.bar(top_locations['Zone'], top_locations['predict_trips'], color="skyblue")
        # ax1.set_xticklabels(ax2.get_xticks(), rotation = 90)
        ax1.set_xticklabels(top_locations['Zone'], rotation=90)
        ax1.set_xlabel("Zone")
        ax1.set_ylabel("Trips")
        ax1.set_title("Top LocationIDs by Estimated Ride Demand")

        filtered_df['totalFare'] = filtered_df['predict_trips'] * filtered_df['fare_amount']
        top_fare = filtered_df.sort_values(by="totalFare", ascending=False).head(10)
        # ax2.bar(top_locations['PULocationID'].astype(str), top_fare['totalFare'], color="khaki")
        ax2.bar(top_fare['Zone'], top_fare['totalFare'], color="khaki")
        ax2.set_xticklabels(top_fare['Zone'], rotation=90)
        ax2.set_xlabel("Zone")
        ax2.set_ylabel("Trips")
        ax2.set_title("Top LocationIDs by Estimated Revenue")
        st.pyplot(fig)

    else:
        st.warning("No data found for the selected time parameters.")

# ----------------------------
# Page 3: Specific Location
# ----------------------------
elif page == "Specific Location":
    st.title("Ride Demand for a Specific Location")
    if not filtered_df.empty:
        # Create a dropdown list from unique LocationIDs in the filtered data
        unique_locations = filtered_df['Zone'].unique()
        selected_location = st.selectbox("Select Zone", unique_locations)
        loc_df = filtered_df[filtered_df['Zone'] == selected_location]
        
        if not loc_df.empty:
            trips_value = loc_df.iloc[0]['predict_trips']
            # st.write(f"**Trips for LocationID {selected_location}:** {trips_value}")
            
            # Prepare map data and show a focused map for the selected location
            map_df = loc_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
            view_state = pdk.ViewState(
                longitude=map_df['lon'].iloc[0],
                latitude=map_df['lat'].iloc[0],
                zoom=14,
                pitch=0
            )
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position=["lon", "lat"],
                get_radius=200,  # Fixed radius for clarity
                get_fill_color="[0, 255, 0, 200]",
                pickable=True
            )
            deck = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>Zone:</b> {Zone}<br/><b>Estimated Ride Demand:</b> {predict_trips}",
                    "style": {"backgroundColor": "darkgreen", "color": "white"}
                }
            )
            st.pydeck_chart(deck)
        else:
            st.info("No data found for the selected LocationID.")
    else:
        st.info("No data available for the selected time parameters.")

# ----------------------------
# Page 4: Model performace
# ----------------------------
elif page == "Model monitoring":
    st.title("XGBoost Model Versions and Performance Metrics")

    # we assume a value for runs first. In real cases, we receive it from MLflow
    runs = {'metrics.MAE': [14.0, 8.0, 6.8], 'metrics.R2': [0.54, 0.53, 0.46]}

    # Display the DataFrame
    st.dataframe(runs)

    # Plot MAE and R² for different model versions
    st.line_chart(runs[["metrics.MAE", "metrics.R2"]])
