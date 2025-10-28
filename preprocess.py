import pandas as pd
import numpy as np
import torch
import pickle
from tqdm import tqdm
from category_encoders import HashingEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel



def split_data_time_series(df, val_size):
    n = len(df)
    if isinstance(val_size, float):
        val_size = int(n * val_size)
    val_size = max(0, min(val_size, n))
    mask = np.ones(n, dtype=bool)
    if val_size > 0:
        mask[-val_size:] = False
    return pd.DataFrame(mask, columns=["Train Mask"])

def split_data(df, target_col, val_size, random_state=42):
    idx = np.arange(len(df))
    mask = np.ones(len(df), dtype=bool)
    if val_size > 0:
        train_idx, val_idx = train_test_split(
            idx, test_size=val_size, random_state=random_state, stratify=df[target_col]
        )
        mask[val_idx] = False
    return pd.DataFrame(mask, columns=["Train Mask"])


def extract_train_mask(df):
    train_mask = df["Train Mask"].copy().values
    return train_mask


def encode_target(df, target_col, encoder=None):
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(df[target_col])
    encoded = encoder.transform(df[target_col])
    encoded = pd.DataFrame({target_col: encoded})
    return encoded, ("target_encoder", target_col, encoder)


def encode_ordinal(df, cols, train_mask=None, encoder=None):
    if encoder is None and train_mask is not None:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoder.fit(df.loc[train_mask, cols])
    encoded = encoder.transform(df[cols])
    return pd.DataFrame(encoded, columns=cols), ("ordinal_encoder", cols, encoder)


def encode_hashing(df, cols, n_components=8, encoder=None):
    if encoder is None:
        encoder = HashingEncoder(
            cols=cols, n_components=n_components, drop_invariant=False
        )
        encoder.fit(df[cols])
    arr = encoder.transform(df[cols]).values
    col_names = [f"Hash_{cols[0]}..._{i+1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=col_names), ("hashing_encoder", cols, encoder)


# TODO: Need to fix for data leakage
def get_grouped_value_ratios(df, cols, target_col):

    # Count occurrences of each target value per group
    counts = df.groupby(cols + [target_col]).size().reset_index(name="_cnt")

    # Pivot to wide format: one column per target value
    pivot = counts.pivot_table(
        index=cols, columns=target_col, values="_cnt", fill_value=0
    )

    # Compute row totals and normalize to get ratios
    total = pivot.sum(axis=1)
    ratios = pivot.div(total, axis=0)

    # Rename ratio columns
    ratios.columns = [f"{val}_ratio" for val in ratios.columns]
    ratios_reset = ratios.reset_index()

    # Identify new ratio column names
    ratio_cols = [c for c in ratios_reset.columns if c not in cols]

    # Merge ratios back to the original group keys
    merged = pd.merge(df[cols], ratios_reset, on=cols, how="left", sort=False)

    return merged[ratio_cols], ("grouped_ratio", cols, target_col)


def get_text_embeddings(
    df,
    text_col,
    embedding_type,
    batch_size=32,
    max_length=128,
):

    texts = df[text_col].fillna("").astype(str).tolist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device.type}")

    # Create sentence transformer embeddings
    if embedding_type == "ST":
        model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True)

    # Create bert embeddings
    else:

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert_model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        bert_model.eval()

        chunks = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
                batch = texts[i : i + batch_size]

                # Tokenize batch
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                ids, mask = enc["input_ids"].to(device), enc["attention_mask"].to(
                    device
                )

                # Get model outputs
                out = bert_model(input_ids=ids, attention_mask=mask).last_hidden_state

                # Mean pooling
                m = mask.unsqueeze(-1).expand_as(out).float()
                summed = (out * m).sum(1)
                counts = m.sum(1).clamp(min=1e-9)
                chunks.append((summed / counts).cpu().numpy())

        emb = np.vstack(chunks)

    # Add column headers and return embeddings
    cols = [f"Emb_{text_col}_{i+1}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, columns=cols), (
        "transformer_emb",
        text_col,
        embedding_type,
    )


def get_char_freq(df, text_col):
    ALLOWED_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 ")
    texts = df[text_col].fillna("").astype(str).tolist()
    matrix = np.array([[txt.count(ch) for ch in ALLOWED_CHARS] for txt in texts])
    cols = [
        f"Char_{text_col}_{ch}" if ch != " " else "Char_{text_col}_space"
        for ch in ALLOWED_CHARS
    ]
    return pd.DataFrame(matrix, columns=cols), ("char_freq", text_col, None)

# TODO: remove stop words
def get_tfidf(
    df,
    text_col,
    min_df=1,
    max_df=1.0,
    sublinear_tf=False,
    norm="l2",
    max_features=None,
    train_mask=None,
    vectorizer=None,
):
    texts = df[text_col].fillna("").astype(str)
    if vectorizer is None and train_mask is not None:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
        )
        vectorizer.fit(texts[train_mask])
    arr = vectorizer.transform(texts).toarray()
    cols = [f"Tfidf_{text_col}_{i+1}" for i in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=cols), ("tf-idf", text_col, vectorizer)


def get_column_names(df, prefix):
    return [col for col in df.columns if col.startswith(prefix)]


def get_columns(df, cols):
    selected_df = df[cols].copy()
    return selected_df, ("get_columns", cols, None)


def scale_features(dfs, index, train_mask=None, scaler=None):
    df = dfs[index]
    if scaler is None and train_mask is not None:
        scaler = StandardScaler().fit(df.loc[train_mask])
    scaled = scaler.transform(df)
    cols = df.columns
    return pd.DataFrame(scaled, columns=cols), (
        "scaler",
        index - 1,
        scaler,
    )  # index - 1 b/c dfs[0] is the train mask which is later dropped


def apply_pca(
    dfs,
    index,
    n_components=None,
    train_mask=None,
    pca=None,
    random_state=42,
):
    df = dfs[index]
    if pca is None and train_mask is not None:
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(df.loc[train_mask])
    pcs = pca.transform(df)
    cols = [f"PC_{df.columns[0]}..._{i+1}" for i in range(pcs.shape[1])]
    return pd.DataFrame(pcs, columns=cols), (
        "pca",
        index - 1,
        pca,
    )  # index - 1 b/c dfs[0] is the train mask which is later dropped


def combine_features(dfs):
    return pd.concat(dfs, axis=1)


def save_df(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_pkl(file, path):
    with open(path, "wb") as f:
        pickle.dump(file, f)


def load_df(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def load_pkl(path):
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline
