import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import pandas as pd
from .utils import adjusted_r2


def train_log_models(df, dataset_name):
    #Train and log regression models with MLflow and save metrics.
    mlflow.set_tracking_uri("http://mlflow1:5000")
    mlflow.set_experiment("Anomaly_Detection1")

    X = df[['voltage']]
    y = df['discharge_capacity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Linear Regression": LinearRegression(),
        "XGBoost": xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    # Define the output directory for saving the CSV file
    output_dir = "/opt/airflow/outputs"
    os.makedirs(output_dir, exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name} - {dataset_name}"):

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            adj_r2 = adjusted_r2(r2, len(y_test), X_train.shape[1])

            # Log metrics to MLflow
            mlflow.set_tag("Dataset Type", dataset_name)
            mlflow.log_params({
                "Model": name,
                "Dataset": dataset_name,
                "Test_Size": 0.2,
                "Random_State": 42
            })
            mlflow.log_metrics({
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Adjusted_R2": adj_r2
            })

            # Save metrics to CSV for Power BI
            results = {
                "Model": name,
                "Dataset": dataset_name,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Adjusted_R2": adj_r2
            }

            # Define the CSV file path
            csv_path = os.path.join(output_dir, "combined_metrics.csv")

            # If the CSV doesn't exist, create it with headers
            if not os.path.exists(csv_path):
                results_df = pd.DataFrame([results])
                results_df.to_csv(csv_path, index=False)
            else:
                # If the CSV exists, append the new results
                results_df = pd.DataFrame([results])
                results_df.to_csv(csv_path, mode='a', header=False, index=False)

            # Log model with MLflow
            if name == "XGBoost":
                mlflow.xgboost.log_model(model, f"{name}_model")
            else:
                mlflow.sklearn.log_model(model, f"{name}_model")
