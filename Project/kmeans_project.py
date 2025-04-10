from Utils.databasify import PostgresDB
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import warnings
import joblib

def clean_and_prepare_for_kmeans(df, exclude_cols=["genetic_disorder", "patient_id", "num"]):
    df = df.dropna(axis=1, how='all').copy() # drop all NaN values 
    df = df.loc[:, df.nunique() > 1] # drop columns with only 1 unique value (we don't need it cause it adds no variance in our dataset)
    df.drop(columns=exclude_cols, errors='ignore', inplace=True)

    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean()) # replace NaN with means

    return df

def run_kmeans(df, n_clusters=10):
    df = clean_and_prepare_for_kmeans(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)
    print(f"\n📊 Silhouette Score for {n_clusters} clusters: {score:.4f}")

    joblib.dump(kmeans, "models/mlproj/kmeans_model.pkl")
    print("💾 KMeans model saved to 'models/mlproj/kmeans_model.pkl'")

    return kmeans, labels

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    db = PostgresDB()
    df = db.fetch_dataframe("SELECT * FROM raw_ml_data", table_name="raw_ml_data")
    run_kmeans(df, n_clusters=14)