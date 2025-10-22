# auto_labeller.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import load_df, save_df, get_text_embeddings  # your helpers

# --- File paths
FOLDER = "indirect_cc_predictor"
input_path = os.path.join(FOLDER, "auto_label.csv")
output_path = os.path.join(FOLDER, "auto_label.csv")

# --- Configure these
feature_columns = [
    "Part Description - Cleaned",
    "Category L1 - Cleaned",
    "Category L2 - Cleaned",
    "Category L3 - Cleaned",
    "Category L4 - Cleaned",
]

# L2 Labels
# possible_labels = [
#     "general",
#     "automation mechanical",
#     "gasses",
#     "chemicals oils lubricants",
#     "office supplies",
#     "abrasives",
#     "automation electrical",
#     "automation welding",
#     "cutting tools",
#     "ppe",
#     "safety",
#     "tools",
# ]

possible_labels = [
    "accident prevention",
    "acetylene",
    "adhesive sealants and tapes",
    "adhesive cementing material glue",
    "argon",
    "barriers",
    "bearings",
    "belts",
    "blasting agents",
    "broaching tools",
    "chains",
    "chemical resistant gloves",
    "chemicals",
    "coated and dipped gloves",
    "cotton and string knit gloves",
    "cut resistant gloves",
    "disposable gloves",
    "dressing tools",
    "drills",
    "electric installation material and consumable materials",
    "electro engineering & measurement technique",
    "electrode",
    "eye protection",
    "fasteners hardware",
    "first aid",
    "foot guard",
    "furniture",
    "grinding",
    "hand tools",
    "headgear",
    "hearing protection",
    "helium",
    "hobbing",
    "hvac and refrigeration",
    "hydraulic",
    "inserts",
    "janitorial",
    "lacquer paint varnish",
    "lubricants",
    "machine tools",
    "material handling",
    "microprocessor technology",
    "nitrogen",
    "oils",
    "plumbing",
    "pneumatic",
    "power tools",
    "propane",
    "respiratory protection",
    "sensors technology",
    "sleeves",
    "supplies",
    "weld wire",
]

# --- Load data
df = load_df(input_path)

# --- Guardrails
if not feature_columns:
    raise ValueError("feature_columns is empty. Add at least one text column to embed.")
if not possible_labels:
    raise ValueError(
        "possible_labels is empty. Provide a list of candidate label strings."
    )
for col in feature_columns:
    if col not in df.columns:
        raise ValueError(f"feature column '{col}' not found in input dataframe.")

# --- Embed labels once (consistent embedding space)
label_df = pd.DataFrame({"label_text": possible_labels})
label_emb_df, _ = get_text_embeddings(label_df, "label_text", embedding_type="ST")
L = label_emb_df.values  # (K, D)

# --- Compute per-column distances to labels, then average across columns
# Each D_col is shape (N, K) with cosine distances (1 - cosine_similarity)
dist_mats = []
for col in feature_columns:
    emb_df, _ = get_text_embeddings(df, col, embedding_type="ST")  # (N, D)
    S_col = cosine_similarity(emb_df.values, L)  # (N, K)
    D_col = 1.0 - S_col  # cosine distance
    dist_mats.append(D_col)

# --- Define weights for each feature column (must sum to 1)
# Example: emphasize part description, downweight category hierarchy
feature_weights = {
    "Part Description - Cleaned": 1,
    "Category L1 - Cleaned": 1,
    "Category L2 - Cleaned": 1,
    "Category L3 - Cleaned": 1,
    "Category L4 - Cleaned": 1,
}

# --- Compute weighted average distances
if len(dist_mats) == 1:
    D_avg = dist_mats[0]
else:
    # Collect weights in correct order
    w = np.array(
        [feature_weights.get(col, 1.0 / len(dist_mats)) for col in feature_columns]
    )
    w = w / w.sum()  # normalize to sum=1 for safety

    # Stack and apply weighted average
    # dist_mats shape: (num_cols, N, K)
    D_stack = np.stack(dist_mats, axis=0)
    D_avg = np.tensordot(w, D_stack, axes=(0, 0))  # (N, K)

# --- Choose label with MIN averaged distance
best_idx = D_avg.argmin(axis=1)  # (N,)
predicted_label = [possible_labels[i] for i in best_idx]
predicted_distance = D_avg[np.arange(D_avg.shape[0]), best_idx]
predicted_similarity = (
    1.0 - predicted_distance
)  # convenience: cosine sim of chosen label

# --- Optional: top-k by smallest distance (keep if useful)
k = min(3, len(possible_labels))
topk_idx = np.argsort(D_avg, axis=1)[:, :k]  # ascending distance
topk_labels = [[possible_labels[j] for j in row] for row in topk_idx]
topk_distances = [D_avg[i, row].tolist() for i, row in enumerate(topk_idx)]
topk_similarities = [[1.0 - d for d in row] for row in topk_distances]

# --- Output: original input + predictions
out = df.copy()
out["predicted_label"] = predicted_label
out["predicted_distance"] = predicted_distance
out["predicted_similarity"] = predicted_similarity
# CSV-friendly compact top-k (optional)
out["topk_labels"] = [", ".join(lbls) for lbls in topk_labels]
out["topk_distances"] = [
    ", ".join(f"{d:.4f}" for d in dists) for dists in topk_distances
]
out["topk_similarities"] = [
    ", ".join(f"{s:.4f}" for s in sims) for sims in topk_similarities
]

save_df(out, output_path)
print(f"Saved auto-label predictions to: {output_path}")
