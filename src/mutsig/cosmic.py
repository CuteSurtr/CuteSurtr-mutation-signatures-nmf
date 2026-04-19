from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from mutsig.features import SBS96_CHANNELS

def load_cosmic_sbs(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t', index_col=0)
    df = df.reindex(SBS96_CHANNELS)
    if df.isna().any().any():
        missing = df.index[df.isna().any(axis=1)].tolist()
        raise ValueError(f'COSMIC file is missing channels: {missing[:5]} ...')
    return df

def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-12)
    return A_norm.T @ B_norm

def match_to_cosmic(W: np.ndarray, cosmic: pd.DataFrame, min_similarity: float=0.85) -> pd.DataFrame:
    sims = cosine_similarity_matrix(W, cosmic.values)
    row_ind, col_ind = linear_sum_assignment(-sims)
    records = []
    for k, c in zip(row_ind, col_ind):
        best_sim = sims[k, c]
        label = cosmic.columns[c] if best_sim >= min_similarity else f'novel_{k}'
        records.append({'recovered_k': int(k), 'cosmic_match': label, 'cosine_similarity': float(best_sim)})
    return pd.DataFrame(records).sort_values('recovered_k').reset_index(drop=True)
