# -*- coding: utf-8 -*-
#Author: Liuxin YANG
#Date: 2025-05-31

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from config.configuration import ModelConfig


def generate_embedding(cfg: ModelConfig, df: pd.DataFrame, model_path, text_cols: str, include_y_in_x: bool, existing_label_map = None) -> None:
    st_model = SentenceTransformer(model_path)

    UNK = "<UNK>"
    ######################### Text variables #########################
    if text_cols is not None:
        desc_embeddings = st_model.encode(
            df[text_cols].fillna(""),
            convert_to_tensor=False, 
            batch_size=8,
            device="cuda" if st_model.device.type == "cuda" else "cpu",
            show_progress_bar=True
        )
    else:
        desc_embeddings = np.array([])

    ######################### Numerical variables #########################
    value_scaled = []
    if cfg.value_cols is not None:
        scaler = StandardScaler()
        for col in cfg.value_cols:
            value_scaled.append(scaler.fit_transform(df[[col]]))
        value_scaled = np.hstack(value_scaled)
    else:
        value_scaled = np.array([])

    ######################### Categorical variables #########################
    category_onehot = []
    if cfg.category_cols is not None:
        for col in cfg.category_cols:
            onehot = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            category_onehot.append(onehot.fit_transform(df[[col]].astype(str)))
        category_onehot = np.hstack(category_onehot)
    else:
        category_onehot = np.array([])

    ######################### Labels #########################
    label_map = {} if existing_label_map is None else {k: dict(v) for k, v in existing_label_map.items()}
    reverse_label_map = {}
    
    ### y
    y = None
    if cfg.hierarchy_cols is not None:
        for col in cfg.hierarchy_cols:
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' not found in DataFrame.")
            
            if existing_label_map is None:
                values = [UNK] + [v for v in pd.Series(df[col].unique()).dropna().tolist() if v != UNK]
                label_map[col] = {v: i for i, v in enumerate(values)}
            
            reverse_label_map[col] = {v: k for k, v in label_map[col].items()}
            unk_id = label_map[col][UNK]
            df[col + "_idx"] = df[col].fillna(UNK).map(label_map[col]).fillna(unk_id).astype(int)

        y = df[[col + "_idx" for col in cfg.hierarchy_cols]].values
    else:
        y = None

    ### ylabels
    ylabels = None
    if cfg.correct_hierarchy_cols is not None:
        for col in cfg.correct_hierarchy_cols:
            if col not in df.columns:
                raise ValueError(f"Correct label column '{col}' not found in DataFrame.")
        
            base = col.replace("correct_", "")
            unk_id = label_map[base][UNK]
            df[col + "_idx"] = df[col].fillna(UNK).map(label_map[base]).fillna(unk_id).astype(int)

        ylabels = df[[col + "_idx" for col in cfg.correct_hierarchy_cols]].values
    else:
        ylabels = None

    label = None
    if cfg.label_cols is not None:
        for col in cfg.label_cols:
            if col not in df.columns:
                raise ValueError(f"Label column '{col}' not found in DataFrame.")
        
            label_map[col] = {v: i for i, v in enumerate(df[col].unique())}
            reverse_label_map[col] = {v: k for k, v in label_map[col].items()}
        label = df[cfg.label_cols].astype(bool).values
    else:
        label = None

    ######################### Hierarchy Path Embedding #########################
    if cfg.hierarchy_cols is not None:
        hierarchy_path = df[cfg.hierarchy_cols].fillna("").astype(str).agg(" > ".join, axis=1)
        hierarchy_path_embeddings = st_model.encode(
            hierarchy_path,
            convert_to_tensor=False, 
            batch_size=8,
            device="cuda" if st_model.device.type == "cuda" else "cpu",
            show_progress_bar=True
        )
    else:
        hierarchy_path_embeddings = np.array([])

    ######################### Combine all features #########################
    feature_parts = []
    for part in [desc_embeddings, value_scaled, category_onehot, hierarchy_path_embeddings]:
        if isinstance(part, (np.ndarray, torch.Tensor)) and part.size > 0:
            feature_parts.append(part)

    if include_y_in_x and y is not None:
        if isinstance(y, (np.ndarray, torch.Tensor)) and y.size > 0:
            feature_parts.append(y.astype(float))

    X = np.hstack(feature_parts)

    return X, y, ylabels, label, label_map, reverse_label_map