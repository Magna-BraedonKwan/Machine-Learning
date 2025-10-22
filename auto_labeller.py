# auto_labeller.py
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import load_df, save_df, get_text_embeddings  # your helpers

# --- File paths
FOLDER = "indirect_cc_predictor"
input_path = os.path.join(FOLDER, "parts_data.csv")
output_path = os.path.join(FOLDER, "auto_label.csv")

# --- Configure these
feature_columns = [
    "Part Description - Cleaned",
    "Category L1 - Cleaned",
    "Category L2 - Cleaned",
    "Category L3 - Cleaned",
    "Category L4 - Cleaned",
]

possible_labels = {
    "Abrasives": "abrasives blasting agents dressing grinding other abrasive honing wheels technical brushed",
    "Automation Electrical": "automation electrical electric installation material consumable materials electro engineering measurement technique microprocessor technology sensors technology devices electrical installations maintenance including wires connectors fuses devices programmable logic controllers plcs sensors other automated systems devices detect respond changes monitoring control manufacturing process",
    "Automation Mechanical": "automation mechanical bearings belts chains hydraulic pneumatic reduce friction wear between moving parts machinery ball bearings roller bearings tapered roller transmit power between rotating shafts machinery v belts timing belts flat lifting pulling transmitting power machinery components use fluid power perform work pumps motors valves hoses actuators components use compressed air perform work pumps motors valves hoses actuators",
    "Automation Welding": "automation welding electrode weld wire material welding conduct current through workpiece weld wire",
    "Chemicals, Oils & Lubricants": "chemicals oils lubricants adhesive cementing material glue chemicals lacquer paint varnish lubricants oils",
    "Cutting Tools": "cutting broaching drills hobbing inserts process removes material toothed push pull keyway surface create holes materials cutting gears splines",
    "General": "general adhesive sealants tapes fasteners hardware hvac refrigeration janitorial material handling plumbing materials bonding sealing securing",
    "Gasses": "gasses acetylene argon helium nitrogen propane",
    "Office Supplies": "office supplies furniture supplies furnish equip workspaces writing utensils paper toner",
    "PPE": "ppe personal protective chemical resistant gloves coated dipped gloves cotton string knit gloves cut resistant gloves disposable gloves eye protection foot guard headgear hearing protection respiratory protection sleeves protect hands hazardous chemicals coatings enhanced grip protection made cotton string knit materials general hand protection designed protect hands cuts abrasions single use gloves protection against contaminants designed protect eyes hazards flying debris chemicals protective gear worn head prevent injuries protective coverings limbs damage injury",
    "Safety": "safety accident prevention barriers first aid garments prevent accidents workplace garments restrict access hazardous areas protect personnel danger garments provide initial medical treatment case of injury illness",
    "Tools": "hand machine power manual various maintenance repair task lathe milling shaping powered by an external source electricity batteries various tasks",
}

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

# --- Prepare labels (names) and their descriptions (to embed)
# possible_labels: dict[str, str]  e.g., {"Abrasives": "abrasives blasting ...", ...}
labels = list(possible_labels.keys())  # what we'll output
label_desc = list(possible_labels.values())  # what we'll embed

# --- Embed label DESCRIPTIONS once (consistent embedding space)
label_df = pd.DataFrame({"label_text": label_desc})
label_emb_df, _ = get_text_embeddings(label_df, "label_text", embedding_type="ST")
L = label_emb_df.values  # (K, D)

# --- Compute per-column distances to labels, then weighted average across columns
dist_mats = []
for col in feature_columns:
    emb_df, _ = get_text_embeddings(df, col, embedding_type="ST")  # (N, D)
    S_col = cosine_similarity(emb_df.values, L)  # (N, K)
    D_col = 1.0 - S_col  # cosine distance
    dist_mats.append(D_col)

# --- Define weights for each feature column (must sum to 1)
feature_weights = {
    "Part Description - Cleaned": 0.2,
    "Category L1 - Cleaned": 0.2,
    "Category L2 - Cleaned": 0.2,
    "Category L3 - Cleaned": 0.2,
    "Category L4 - Cleaned": 0.2,
}

if len(dist_mats) == 1:
    D_avg = dist_mats[0]
else:
    # Collect weights in the same order as feature_columns; default to equal share if missing
    w = np.array(
        [feature_weights.get(col, 1.0 / len(dist_mats)) for col in feature_columns],
        dtype=float,
    )
    w = w / w.sum()  # normalize just in case
    D_stack = np.stack(dist_mats, axis=0)  # (num_cols, N, K)
    D_avg = np.tensordot(w, D_stack, axes=(0, 0))  # (N, K)

# --- Choose label with MIN averaged distance; output LABEL NAMES (not descriptions)
best_idx = D_avg.argmin(axis=1)  # (N,)
predicted_label = [labels[i] for i in best_idx]
predicted_distance = D_avg[np.arange(D_avg.shape[0]), best_idx]
predicted_similarity = 1.0 - predicted_distance

# --- Optional: top-k by smallest distance (k limited by number of labels)
k = min(3, len(labels))
topk_idx = np.argsort(D_avg, axis=1)[:, :k]  # ascending distance
topk_labels = [[labels[j] for j in row] for row in topk_idx]
topk_distances = [D_avg[i, row].tolist() for i, row in enumerate(topk_idx)]
topk_similarities = [[1.0 - d for d in row] for row in topk_distances]

# --- Output
out = df.copy()
out["predicted_label"] = predicted_label
out["predicted_distance"] = predicted_distance
out["predicted_similarity"] = predicted_similarity
out["topk_labels"] = [", ".join(lbls) for lbls in topk_labels]
out["topk_distances"] = [
    ", ".join(f"{d:.4f}" for d in dists) for dists in topk_distances
]
out["topk_similarities"] = [
    ", ".join(f"{s:.4f}" for s in sims) for sims in topk_similarities
]

save_df(out, output_path)
print(f"Saved auto-label predictions to: {output_path}")
