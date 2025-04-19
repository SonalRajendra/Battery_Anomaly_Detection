from scipy.stats import zscore
from sklearn.ensemble import IsolationForest


def remove_zscore_outliers(df, features):
    #Remove anomalies based on Z-score.
    z_scores = zscore(df[features])
    mask = (abs(z_scores) > 3).any(axis=1)
    return df[~mask]

def clean_with_isolation_forest(df, features, contamination=0.05):
    #Remove anomalies using Isolation Forest.
    model = IsolationForest(contamination=contamination, random_state=42)
    df['anomaly'] = model.fit_predict(df[features])
    return df[df['anomaly'] == 1]





