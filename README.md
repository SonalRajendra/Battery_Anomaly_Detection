
# 🔋 Battery Anomaly Detection
This project implements a pipeline for detecting anomalies in battery data. The pipeline encompasses data preprocessing, model training, and evaluation, utilizing tools like MLflow for experiment tracking, Airflow for orchestration, and Power BI for visualization.

## 🚀 Project Overview
The battery anomaly detection pipeline includes the following steps:

- Data Preprocessing: Stratified sampling, voltage trend visualization, and correlation matrix generation.

- Outlier Removal: Cleaning and transforming data to enhance model accuracy.

- Modeling: Training various regression models (Random Forest, XGBoost, Decision Tree, and Linear Regression) and logging evaluation metrics such as MSE, RMSE, R², and Adjusted R².

- Experiment Tracking: Using MLflow to log results, including parameters, metrics, and artifacts.

- Orchestration: Employing Airflow for workflow management.

- Visualization: Displaying results using Power BI.


### 1.0 Project Structure
```arduino
Battery_Anomaly_Detection/
├── data/                                  # Contains links or references to raw data
│   └── README.md                          # Information or link to download dataset
│
├── results/                               # Model outputs and visualizations
│   ├── figures/                           # Visualizations and plots
│   └── battery_anomaly_detection.pbix     # Power BI report file
│
├── src/                                   # Source code and pipeline logic
│   ├── dags/                              # Airflow DAG definitions
│   │   └── battery_pipeline_dag.py        # Main pipeline DAG
│   │
│   ├── pipeline/                          # Core pipeline modules
│   │   ├── __init__.py                    # Package initializer
│   │   ├── preprocessing.py               # Data cleaning and transformation logic
│   │   ├── data_loader.py                 # Data loading logic
│   │   ├── utils.py                       # Utility functions
│   │   ├── anomalies.py                   # Anomaly detection logic
│   │   └── modeling.py                    # Model training and evaluation logic
│   │
│   ├── Dockerfile.airflow1                # Dockerfile for Airflow setup
│   ├── Dockerfile.mlflow1                 # Dockerfile for MLflow setup
│   └── docker-compose.yml                 # Docker Compose configuration for multi-container setup
│
├── Sphinx Documentation/                  # Project documentation
│   ├── index.html                         # Entry point for generated HTML docs
|   ├── genindex.html                      # Generated index of all the symbols
│   ├── thesis.html                        # Custom thesis page 
│   └── search.html                        # Search page for documentation
│
├── requirements.txt                       # Python dependencies
├── README.md                              # Project overview and setup guide



```

### 2.0 How to run the code
In this section we will understand how to run the code.
### 2.1 Prerequisites
All required Python libraries and their specific versions are listed in `requirements.txt`.

Before you run this project, ensure you have the following installed:

- **Docker** and **Docker Compose** for managing Airflow in a containerized environment.
- **Python 3.8+** for running the preprocessing scripts and other tasks.
- **Apache Airflow** (packaged in Docker) is used for workflow orchestration.



### 2.2 Running the Project with Airflow
This project uses Apache Airflow for orchestrating the tasks. It is packaged using Docker for containerized execution. You can easily start and run Airflow locally using Docker Compose. **
1. **Clone the repository**
   ```bash
   git clone https://github.com/SonalRajendra/Battery_Anomaly_Detection.git
   cd Battery_Anomaly_Detection
   ```

2. **Install Dependencies**  
   Open a command prompt (or terminal) in your project folder and run:
   ```bash
   pip install -r requirements.txt
3. **Start Docker**

   Make sure Docker is running on your system
4. **Open Command Prompt**

    Navigate to the src directory 
    ```bash
    cd src
5. **Run Airflow**

    - Start the Airflow services using Docker Compose. From the src directory, run:
    ```bash
    docker-compose up
    ```
    This command will start all Docker containers, including those for Airflow and MLflow.
6. **Verify Docker Containers**

     Open your Docker dashboard (e.g., Docker Desktop) to verify that the containers are running.
    Look for the container associated with the thesis project.
    Ensure that the Airflow container is running.
7. **Access Airflow UI**

    Access the Airflow UI by navigating to `http://localhost:8080` in your browser.
     The default username is `admin1` and the default password is also `admin1`.
8. **Trigger the DAG**

      Once you have logged into the Airflow UI, find the `battery_anomaly_detection_pipeline` DAG which you can trigger manually.

### 2.3 Access to MLflow
  - MLflow is used to log parameters, metrics, and artifacts. You can view the logged results in the MLflow UI, which is running as part of the Docker Compose setup.
  - Access the MLflow UI by navigating to 'http://127.0.0.1:5000' in your browser in new tab to view the experiment details and logs for each run.
  - MLflow runs are logged under the experiment 'Anomaly_Detection1'.


### 2.4 How to generate results
- After triggering the DAG in Airflow, the data_preprocessing task will execute, performing the preprocessing steps.
- The task will log the parameters and results in MLflow.
- The generated plots (voltage_trends_full, voltage_trend_downsampled, and correlation_matrix ) are saved in results.
- After training various models (Random Forest, XGBoost, Decision Tree, Linear Regression) on both raw and clean data, a file named `combined_metrics.csv` will be  generated, which includes:Dataset type: [raw, clean], Evaluation metrics: [MSE, RMSE, R², Adjusted R²], Models: [Random Forest, XGBoost, Decision Tree, Linear Regression].
 - This CSV is used in Power BI [`battery_anomaly_detection.pbix`] to create interactive visualizations comparing model performance across different datasets and metrics.







