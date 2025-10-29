from preprocess import *


def init_pipeline(df, target_col, val_split, dfs, pipeline):

    # Train/Val split
    train_mask_col = split_data(df, target_col, val_split)
    dfs.append(train_mask_col)
    train_mask = extract_train_mask(train_mask_col)

    # Encode target
    processed_df, trans = encode_target(df, target_col)
    pipeline.append(trans)
    dfs.append(processed_df)

    return train_mask


def oem_pipeline(df, pre_path, pipeline_path):
    pipeline = []
    dfs = []

    # Split data into train/validation sets and encode target
    train_mask = init_pipeline(df, "OEM ID", 0.0, dfs, pipeline)

    # Embed part description (cleaned)
    processed_df, trans = get_text_embeddings(
        df,
        "Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Char frequency of part description (cleaned)
    processed_df, trans = get_char_freq(df, "Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)

    # TF-IDF of part description (cleaned)
    processed_df, trans = get_tfidf(
        df, "Cleaned", max_features=150, train_mask=train_mask, min_df=2, max_df=0.2
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Save combined features and pipeline
    pre_df = combine_features(dfs)
    save_df(pre_df, pre_path)
    save_pkl(pipeline, pipeline_path)


def cc1_pipeline(df, pre_path, pipeline_path):
    pipeline = []
    dfs = []

    # Split data into train/validation sets and encode target
    train_mask = init_pipeline(df, "Category L1", 0.2, dfs, pipeline)

    # Embed part description (cleaned)
    processed_df, trans = get_text_embeddings(
        df,
        "Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Char frequency of part description (cleaned)
    processed_df, trans = get_char_freq(df, "Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)

    # TF-IDF of part description (cleaned)
    processed_df, trans = get_tfidf(
        df, "Cleaned", min_df=50, max_df=0.1, max_features=300, train_mask=train_mask
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Scale embedded features
    dfs[2], trans = scale_features(dfs, 2, train_mask=train_mask)
    pipeline.append(trans)

    # Apply PCA to embedded features
    dfs[2], trans = apply_pca(dfs, 2, 100, train_mask=train_mask)
    pipeline.append(trans)

    # Pivot features
    processed_df, trans = get_grouped_value_ratios(
        df, ["Divisionname (Hyperion Code)"], "Category L1"
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_grouped_value_ratios(
        df, ["Supplier Group Name (DUNS)"], "Category L1"
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_grouped_value_ratios(
        df,
        ["Divisionname (Hyperion Code)", "Supplier Group Name (DUNS)"],
        "Category L1",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Save combined features and pipeline
    pre_df = combine_features(dfs)
    save_df(pre_df, pre_path)
    save_pkl(pipeline, pipeline_path)


def dc_pipeline(df, pre_path, pipeline_path):
    pipeline = []
    dfs = []

    # Split data into train/validation sets and encode target
    train_mask = init_pipeline(df, "Directed/Controllable", 0.2, dfs, pipeline)

    # Embed part description (cleaned)
    processed_df, trans = get_text_embeddings(
        df,
        "Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Char frequency of part description (cleaned)
    processed_df, trans = get_char_freq(df, "Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)

    # TF-IDF of part description (cleaned)
    processed_df, trans = get_tfidf(
        df, "Cleaned", min_df=50, max_df=0.1, max_features=200, train_mask=train_mask
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Scale embedded features
    dfs[2], trans = scale_features(dfs, 2, train_mask=train_mask)
    pipeline.append(trans)

    # Apply PCA to embedded features
    dfs[2], trans = apply_pca(dfs, 2, 100, train_mask=train_mask)
    pipeline.append(trans)

    # Pivot features
    processed_df, trans = get_grouped_value_ratios(
        df, ["Divisionname (Hyperion Code)"], "Directed/Controllable"
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_grouped_value_ratios(
        df, ["Supplier Group Name (DUNS)"], "Directed/Controllable"
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_grouped_value_ratios(
        df,
        ["Category L1"],
        "Directed/Controllable",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Save combined features and pipeline
    pre_df = combine_features(dfs)
    save_df(pre_df, pre_path)
    save_pkl(pipeline, pipeline_path)


def cluster_pipeline(df, pre_path, pipeline_path):
    pipeline = []
    dfs = []

    # Must init_pipeline even for unsupervised learning
    train_mask = init_pipeline(df, "Category L1", 0.0, dfs, pipeline)

    # TF-IDF of part description (cleaned)
    processed_df, trans = get_tfidf(
        df,
        "Part Description",
        min_df=10,
        max_df=0.5,
        max_features=100,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Embed part description (cleaned)
    processed_df, trans = get_text_embeddings(
        df,
        "Part Description",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # TF-IDF of part description (cleaned)
    processed_df, trans = get_tfidf(
        df,
        "Category L1",
        min_df=10,
        max_df=0.5,
        max_features=100,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Embed part description (cleaned)
    processed_df, trans = get_text_embeddings(
        df,
        "Category L1",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # TF-IDF of part description (cleaned)
    processed_df, trans = get_tfidf(
        df,
        "Category L2",
        min_df=10,
        max_df=0.5,
        max_features=100,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Embed part description (cleaned)
    processed_df, trans = get_text_embeddings(
        df,
        "Category L2",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # TF-IDF of part description (cleaned)
    processed_df, trans = get_tfidf(
        df,
        "Category L3",
        min_df=10,
        max_df=0.5,
        max_features=100,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Embed part description (cleaned)
    processed_df, trans = get_text_embeddings(
        df,
        "Category L3",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Save combined features and pipeline
    pre_df = combine_features(dfs)
    save_df(pre_df, pre_path)
    save_pkl(pipeline, pipeline_path)


def indirect_cc_pipeline(df, pre_path, pipeline_path):
    pipeline = []
    dfs = []

    # Split data into train/validation sets and encode target
    train_mask = init_pipeline(df, "target", 0.2, dfs, pipeline)

    # Feature engineering for part description
    processed_df, trans = get_text_embeddings(
        df,
        "Part Description - Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    dfs[2], trans = scale_features(dfs, 2, train_mask=train_mask)
    pipeline.append(trans)
    dfs[2], trans = apply_pca(dfs, 2, 100, train_mask=train_mask)
    pipeline.append(trans)
    processed_df, trans = get_char_freq(df, "Part Description - Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_tfidf(
        df,
        "Part Description - Cleaned",
        min_df=5,
        max_df=0.1,
        max_features=250,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Feature engineering for category L1
    processed_df, trans = get_text_embeddings(
        df,
        "Category L1 - Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    dfs[5], trans = scale_features(dfs, 5, train_mask=train_mask)
    pipeline.append(trans)
    dfs[5], trans = apply_pca(dfs, 5, 100, train_mask=train_mask)
    pipeline.append(trans)
    processed_df, trans = get_char_freq(df, "Category L1 - Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_tfidf(
        df,
        "Category L1 - Cleaned",
        min_df=10,
        max_df=0.2,
        max_features=50,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Feature engineering for category L2
    processed_df, trans = get_text_embeddings(
        df,
        "Category L2 - Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    dfs[8], trans = scale_features(dfs, 8, train_mask=train_mask)
    pipeline.append(trans)
    dfs[8], trans = apply_pca(dfs, 8, 100, train_mask=train_mask)
    pipeline.append(trans)
    processed_df, trans = get_char_freq(df, "Category L2 - Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_tfidf(
        df,
        "Category L2 - Cleaned",
        min_df=10,
        max_df=0.2,
        max_features=50,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Feature engineering for category L3
    processed_df, trans = get_text_embeddings(
        df,
        "Category L3 - Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    dfs[11], trans = scale_features(dfs, 11, train_mask=train_mask)
    pipeline.append(trans)
    dfs[11], trans = apply_pca(dfs, 11, 100, train_mask=train_mask)
    pipeline.append(trans)
    processed_df, trans = get_char_freq(df, "Category L3 - Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_tfidf(
        df,
        "Category L3 - Cleaned",
        min_df=10,
        max_df=0.2,
        max_features=50,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Feature engineering for category L4
    processed_df, trans = get_text_embeddings(
        df,
        "Category L4 - Cleaned",
        "ST",
    )
    pipeline.append(trans)
    dfs.append(processed_df)
    dfs[14], trans = scale_features(dfs, 14, train_mask=train_mask)
    pipeline.append(trans)
    dfs[14], trans = apply_pca(dfs, 14, 100, train_mask=train_mask)
    pipeline.append(trans)
    processed_df, trans = get_char_freq(df, "Category L4 - Cleaned")
    pipeline.append(trans)
    dfs.append(processed_df)
    processed_df, trans = get_tfidf(
        df,
        "Category L4 - Cleaned",
        min_df=10,
        max_df=0.2,
        max_features=50,
        train_mask=train_mask,
    )
    pipeline.append(trans)
    dfs.append(processed_df)

    # Save combined features and pipeline
    pre_df = combine_features(dfs)
    save_df(pre_df, pre_path)
    save_pkl(pipeline, pipeline_path)
