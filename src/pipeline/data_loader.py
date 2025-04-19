import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def load_data(path="/opt/airflow/case_study.csv"):
    #Load data from CSV
    df = pd.read_csv(path)
    df['cycle_index'] = df['cycle_index'].astype(int)
    return df

def stratified_sample(df, sample_size=50000):
    #Perform stratified sampling based on 'cycle_index'
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size / len(df), random_state=42)
    for _, idx in sss.split(df, df['cycle_index']):
        return df.iloc[idx]







