import os
import numpy as np
from sklearn.utils import compute_class_weight
from xgboost import XGBClassifier
from preprocess import *
from model_params import OEM_PARAMS, CC1_PARAMS, DC_PARAMS, CC_PARAMS
from pipelines import oem_pipeline, cc1_pipeline, dc_pipeline, cc_pipeline

# File paths
FOLDER = "indirect_cc_predictor"
train_path = os.path.join(FOLDER, "train_data.csv")
pre_path = os.path.join(FOLDER, "preprocessed.csv")
pipeline_path = os.path.join(FOLDER, "pipeline.pkl")
model_path = os.path.join(FOLDER, "xgb_model.pkl")
results_path = os.path.join(FOLDER, "results.csv")

# Settings
xgb_params = CC_PARAMS
create_pipeline = cc_pipeline
use_class_weights = False


def main():
    df = load_df(train_path)

    # Preprocess data
    if not os.path.exists(pre_path):
        print("Preprocessing data...")
        create_pipeline(df, pre_path, pipeline_path)
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
            target_encoder = trans[2]
            target_col = trans[1]

    # Extract train mask and split data into X and y
    train_mask = extract_train_mask(pre_df)
    y = pre_df[target_col].values
    X = pre_df.drop(columns=[target_col, "Train Mask"]).astype(float).values

    # Class weights
    if use_class_weights:
        print("Calculating class weights...")
        classes = np.unique(y[train_mask])
        class_weights = compute_class_weight(
            "balanced", classes=classes, y=y[train_mask]
        )
        weight_map = dict(zip(classes, class_weights))
        sample_weights = np.array([weight_map[label] for label in y[train_mask]])
        print("Weights have been calculated")
    else:
        sample_weights = np.ones_like(y[train_mask], dtype=float)

    # Train
    print("Training XGB predictor...")
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X[train_mask], y[train_mask], sample_weight=sample_weights)
    print("Finished training")

    # Save model
    print("Saving model...")
    save_pkl(xgb, model_path)
    print("Model saved")

    # Predict
    print("Predicting...")
    preds = xgb.predict(X)
    probs = xgb.predict_proba(X)
    print("Finished predicting")

    # Save results
    print("Saving results...")
    df["Train Mask"] = train_mask
    df["Prediction"] = target_encoder.inverse_transform(preds)
    for i, cls in enumerate(target_encoder.classes_):
        df[f"prob_{cls}"] = probs[:, i]
    save_df(df, results_path)
    print("Saved results")


if __name__ == "__main__":
    main()
