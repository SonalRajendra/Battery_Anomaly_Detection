import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mlflow

def preprocess_data(df, stratified_sample):
    #Perform data preprocessing steps and log results with MLflow.
    mlflow.set_tracking_uri("http://mlflow1:5000")
    mlflow.set_experiment("Anomaly_Detection1")

    with mlflow.start_run(run_name="Data Preprocessing"):

        # Null checks
        mlflow.log_param("nulls_in_full_data", df.isnull().sum().sum())
        mlflow.log_param("nulls_in_stratified_sample", stratified_sample.isnull().sum().sum())
        
        # Time Series Analysis
        # Voltage Trend
        plt.figure(figsize=(10, 5))
        plt.plot(df['test_time'], df['voltage'], label='Voltage')
        plt.xlabel('Test Time')
        plt.ylabel('Voltage')
        plt.legend()
        plt.title('Voltage Trend Over Time')
        plot_path = "/opt/airflow/outputs/voltage_trend_full.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        # Analysing in a Very small sample of 5000 points
        sampled_indices = np.linspace(0, len(df) - 1, 5000, dtype=int)
        df_downsampled = df.iloc[sampled_indices]

        plt.figure(figsize=(10, 5))
        plt.scatter(df_downsampled['test_time'], df_downsampled['voltage'], label='Voltage', marker='o')
        plt.xlabel('Test Time')
        plt.ylabel('Voltage')
        plt.title('Downsampled Voltage Trend Over Time')
        plt.legend()
        downsample_plot = "/opt/airflow/outputs/voltage_trend_downsampled.png"
        plt.savefig(downsample_plot)
        mlflow.log_artifact(downsample_plot)
        plt.close()

        # Correlation Heatmap (Stratified Sample)
        analysis_data = stratified_sample.drop(columns=['cell_index', 'index', 'cycle_index'])
        correlation_matrix = analysis_data.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix")
        heatmap_path = "/opt/airflow/outputs/correlation_matrix.png"
        plt.savefig(heatmap_path)
        mlflow.log_artifact(heatmap_path)
        plt.close()

        mlflow.log_param("timestamp_ignored", True)

