import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from preprocess import *
from model_params import CLUSTER_PARAMS
from pipelines import cluster_pipeline

# File paths
FOLDER = "oem_id_predictor"
train_path = os.path.join(FOLDER, "global_parts_data.csv")
pre_path = os.path.join(FOLDER, "cluster_preprocessed.csv")
pipeline_path = os.path.join(FOLDER, "cluster_pipeline.pkl")
results_path = os.path.join(FOLDER, "clusters.csv")

# Settings
xgb_params = CLUSTER_PARAMS
create_pipeline = cluster_pipeline


def main():
    df = load_df(train_path)
    if not os.path.exists(pre_path):
        print("Preprocessing data...")
        cluster_pipeline(
            df,
            pre_path,
            pipeline_path,
        )
        print("Finished preprocessing data")

    # Load preprocessed data
    print("Loading preprocessed data...")
    pre_df = load_df(pre_path)
    print("Finished loading data")

    # Load pipeline
    print("Loading pipeline...")
    pipeline = load_pkl(pipeline_path)
    print("Pipeline loaded")
    for trans in pipeline:
        if trans[0] == "target_encoder":
            target_col = trans[1]

    X = pre_df.drop(columns=[target_col, "Train Mask"]).astype(float).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cluster
    clusterer = HDBSCAN(**CLUSTER_PARAMS)
    clusterer.fit(X_scaled)
    df["Cluster"] = clusterer.labels_
    df["Confidence"] = clusterer.probabilities_

    # Save results
    df.to_csv(results_path, index=False)
    print("Saved results")


if __name__ == "__main__":
    main()
