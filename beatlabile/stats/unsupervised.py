"""Unsupervised population comparison across cohorts.

PCA projection and cross-cohort Spearman feature correlations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def population_pca(
    cohort_dfs: dict[str, pd.DataFrame],
    feat_cols: list[str],
    n_components: int = 2,
) -> dict:
    """PCA on pooled cohort feature matrices.

    Parameters
    ----------
    cohort_dfs : dict mapping cohort name → DataFrame (must contain feat_cols)
    feat_cols   : feature column names
    n_components: number of PCs to retain

    Returns
    -------
    dict with keys:
        scores     : DataFrame with columns PC1, PC2, ..., cohort
        explained  : list of explained variance ratios
        loadings   : DataFrame (features × PCs) for feature interpretation
        silhouette : float — sklearn silhouette score between cohort clusters
    """
    frames = []
    for name, df in cohort_dfs.items():
        sub = df[feat_cols].copy()
        sub["cohort"] = name
        frames.append(sub)

    pool = pd.concat(frames, ignore_index=True)
    X = pool[feat_cols].values.astype(float)
    cohort_labels = pool["cohort"].values

    # Drop rows with NaN
    valid = ~np.isnan(X).any(axis=1)
    X = X[valid]
    cohort_labels = cohort_labels[valid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(X_scaled)

    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=pc_cols)
    scores_df["cohort"] = cohort_labels

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feat_cols,
        columns=pc_cols,
    )

    # Silhouette score (only if >1 unique cohort)
    unique_cohorts = np.unique(cohort_labels)
    sil = float("nan")
    if len(unique_cohorts) > 1:
        try:
            from sklearn.metrics import silhouette_score
            # Encode cohort labels as ints
            label_map = {c: i for i, c in enumerate(unique_cohorts)}
            y_sil = np.array([label_map[c] for c in cohort_labels])
            sil = float(silhouette_score(X_scaled, y_sil))
        except Exception:
            pass

    return {
        "scores": scores_df,
        "explained": pca.explained_variance_ratio_.tolist(),
        "loadings": loadings_df,
        "silhouette": sil,
    }


def cross_cohort_correlations(
    cohort_dfs: dict[str, pd.DataFrame],
    feat_cols: list[str],
) -> dict:
    """Spearman correlation matrix per cohort + direction consistency.

    Parameters
    ----------
    cohort_dfs : dict cohort name → DataFrame
    feat_cols  : feature columns

    Returns
    -------
    dict with keys:
        corr_matrices : dict cohort → DataFrame (Spearman r matrix)
        direction_consistency : DataFrame (feature_pair, pct_consistent, mean_abs_r)
            pct_consistent = fraction of cohort pairs with same correlation sign
    """
    import itertools

    rmatrices: dict[str, pd.DataFrame] = {}
    for name, df in cohort_dfs.items():
        sub = df[feat_cols].dropna()
        if len(sub) < 10:
            continue
        corr, _ = spearmanr(sub.values)
        # spearmanr returns scalar when n_features==2, otherwise array
        if np.ndim(corr) == 0:
            corr = np.array([[1.0, float(corr)], [float(corr), 1.0]])
        rmatrices[name] = pd.DataFrame(corr, index=feat_cols, columns=feat_cols)

    # Build direction consistency for each feature pair
    cohort_names = list(rmatrices.keys())
    n_feats = len(feat_cols)
    rows = []
    for i, fi in enumerate(feat_cols):
        for j, fj in enumerate(feat_cols):
            if j <= i:
                continue
            signs = []
            abs_rs = []
            for name in cohort_names:
                r = rmatrices[name].loc[fi, fj]
                signs.append(np.sign(r))
                abs_rs.append(abs(r))

            if len(signs) < 2:
                continue

            # Fraction of cohort pairs with same sign
            pairs = list(itertools.combinations(range(len(signs)), 2))
            same = sum(signs[a] == signs[b] for a, b in pairs)
            pct = same / len(pairs) if pairs else float("nan")

            rows.append(
                {
                    "feature_a": fi,
                    "feature_b": fj,
                    "pct_sign_consistent": pct,
                    "mean_abs_r": float(np.mean(abs_rs)),
                    **{f"r_{name}": float(rmatrices[name].loc[fi, fj]) for name in cohort_names},
                }
            )

    consistency_df = pd.DataFrame(rows).sort_values(
        "pct_sign_consistent", ascending=False
    )

    return {
        "corr_matrices": rmatrices,
        "direction_consistency": consistency_df,
    }
