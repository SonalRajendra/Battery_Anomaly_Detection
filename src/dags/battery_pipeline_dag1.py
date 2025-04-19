# Updated DAG with XCom for data passing and explicit task order
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add pipeline folder to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline')))

# Import custom functions from the pipeline directory
from pipeline.data_loader import load_data, stratified_sample
from pipeline.preprocessing import preprocess_data
from pipeline.anomalies import remove_zscore_outliers, clean_with_isolation_forest
from pipeline.modeling import train_log_models

# DAG Definition
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 4, 1),
    'retries': 1,
}
# Instantiate the DAG
with DAG(
    dag_id='battery_anomaly_detection_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='Battery anomaly detection pipeline with XCom and MLflow',
) as dag:
    # Task 1: Load Data
    def task_load_data(ti):
        #Loads the battery data into a pandas DataFrame and pushes it to XCom.
        df = load_data()
        ti.xcom_push(key='df', value=df)
        
    # Task 2: Stratified Sampling
    def task_stratified_sample(ti):
        #Pulls the DataFrame from XCom, performs stratified sampling, and pushes the sample to XCom.
        df = ti.xcom_pull(key='df', task_ids='load_data')
        sample = stratified_sample(df)
        ti.xcom_push(key='strat_sample', value=sample)
        
    # Task 3: Preprocess Data
    def task_preprocess(ti):
        #Pulls the original DataFrame and the stratified sample from XCom and performs preprocessing.
        df = ti.xcom_pull(key='df', task_ids='load_data')
        sample = ti.xcom_pull(key='strat_sample', task_ids='stratified_sample')
        preprocess_data(df, sample)
    
    # Task 4: Outliers Removal using Z-score
    def task_outlier_removal(ti):
        #Pulls the stratified sample from XCom, removes outliers using the Z-score method, and pushes the cleaned DataFrame to XCom.
        sample = ti.xcom_pull(key='strat_sample', task_ids='stratified_sample')
        cleaned = remove_zscore_outliers(
            sample,
            ['voltage', 'discharge_capacity', 'current', 'internal_resistance', 'temperature']
        )
        ti.xcom_push(key='cleaned_df', value=cleaned)
        
     # Task 5: Clean with Isolation Forest
    def task_isolation_forest(ti):
        #Pulls the Z-score cleaned DataFrame from XCom, further cleans it using Isolation Forest, and pushes the final cleaned DataFrame           to XCom.
        cleaned = ti.xcom_pull(key='cleaned_df', task_ids='remove_outliers_zscore')
        final_df = clean_with_isolation_forest(
            cleaned,
            ['voltage', 'discharge_capacity', 'current', 'internal_resistance', 'temperature']
        )
        ti.xcom_push(key='final_df', value=final_df)
        
     # Task 6: Train model on raw data
    def task_model_raw(ti):
        #Pulls the stratified sample from XCom and trains a regression model on it.
        sample = ti.xcom_pull(key='strat_sample', task_ids='stratified_sample')
        train_log_models(sample, "Raw Data")
        
     # Task 7: Train model on clean data
    def task_model_clean(ti):
        #Pulls the final cleaned DataFrame from XCom and trains a regression model on it.
        final_df = ti.xcom_pull(key='final_df', task_ids='clean_with_isolation_forest')
        train_log_models(final_df, "Cleaned Data")

    # Define all tasks
    load_task = PythonOperator(
        task_id='load_data',
        python_callable=task_load_data
    )

    strat_sample_task = PythonOperator(
        task_id='stratified_sample',
        python_callable=task_stratified_sample
    )

    preprocessing_task = PythonOperator(
        task_id='data_preprocessing',
        python_callable=task_preprocess
    )

    remove_outliers_task = PythonOperator(
        task_id='remove_outliers_zscore',
        python_callable=task_outlier_removal
    )

    isolation_forest_task = PythonOperator(
        task_id='clean_with_isolation_forest',
        python_callable=task_isolation_forest
    )

    model_raw_task = PythonOperator(
        task_id='train_log_model_raw_data',
        python_callable=task_model_raw
    )

    model_clean_task = PythonOperator(
        task_id='train_log_model_cleaned_data',
        python_callable=task_model_clean
    )

    # task execution order
    # Define the order in which the tasks should be executed
    load_task >> strat_sample_task >> preprocessing_task
    preprocessing_task >> [model_raw_task, remove_outliers_task]
    remove_outliers_task >> isolation_forest_task >> model_clean_task
