import os
import pandas as pd
from preprocess import *

# Paths
FOLDER = "oem_id_predictor"
TEST_PATH = os.path.join(FOLDER, "global_parts_data.csv")
PIPELINE_PATH = os.path.join(FOLDER, "pipeline.pkl")
MODEL_PATH = os.path.join(FOLDER, "xgb_model.pkl")
RESULTS_PATH = os.path.join(FOLDER, "results2.csv")


def main():

    # Load pipeline and model
    pipeline = load_pkl(PIPELINE_PATH)
    model = load_pkl(MODEL_PATH)

    # Load unseen test data
    df = load_df(TEST_PATH)

    dfs = []

    # transform data per pipeline
    for name, cols, transformer in pipeline:
        if name == "target_encoder":
            target_encoder = transformer
            dfs.append(pd.DataFrame())  # ensure dfs structure remains consistent
        elif name == "transformer_emb":
            block, _ = get_text_embeddings(df, cols, transformer)
            dfs.append(block)
        elif name == "char_freq":
            block, _ = get_char_freq(df, cols)
            dfs.append(block)
        elif name == "tf-idf":
            block, _ = get_tfidf(df, cols, vectorizer=transformer)
            dfs.append(block)
        elif name == "ordinal_encoder":
            block, _ = encode_ordinal(df, cols, encoder=transformer)
            dfs.append(block)
        elif name == "hashing_encoder":
            block, _ = encode_hashing(df, cols, encoder=transformer)
            dfs.append(block)
        elif name == "scaler":
            scaled, _ = scale_features(dfs, cols, scaler=transformer)
            dfs[cols] = scaled
        elif name == "pca":
            reduced, _ = apply_pca(dfs, cols, pca=transformer)
            dfs[cols] = reduced
        elif name == "get_columns":
            block, _ = get_columns(df, cols)
            dfs.append(block)
        elif name == "grouped_ratio":
            block, _ = get_grouped_value_ratios(df, cols, transformer)
            dfs.append(block)

    # Merge all features
    X = combine_features(dfs).astype(float).values

    # Make predictions
    preds = model.predict(X)
    probs = model.predict_proba(X)

    # Attach predictions and confidences back to original data
    df["Prediction"] = target_encoder.inverse_transform(preds)
    for i, cls in enumerate(target_encoder.classes_):
        df[f"prob_{cls}"] = probs[:, i]

    # Save the results
    save_df(df, RESULTS_PATH)
    print(f"Predictions saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
