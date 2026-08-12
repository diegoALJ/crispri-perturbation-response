from typing import List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def compute_single_cell_statistics(
    X,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate mean expression, variance and dropout rate
    for every gene without unnecessarily densifying a
    sparse single-cell matrix.
    """
    if sparse.issparse(X):

        mean = np.asarray(
            X.mean(axis=0)
        ).ravel()

        X_sq = X.copy()
        X_sq.data **= 2

        mean_sq = np.asarray(
            X_sq.mean(axis=0)
        ).ravel()

        variance = mean_sq - mean**2

        n_cells = X.shape[0]

        nnz_per_gene = np.diff(
            X.tocsc().indptr
        )

        dropout = 1.0 - (
            nnz_per_gene / n_cells
        )

    else:

        X = np.asarray(X)

        mean = X.mean(axis=0)
        variance = X.var(axis=0)
        dropout = (X == 0).mean(axis=0)

    # Numerical precision can occasionally produce
    # very small negative variances.
    variance = np.maximum(variance, 0)

    return (
        mean.astype(np.float32),
        variance.astype(np.float32),
        dropout.astype(np.float32),
    )


def compute_gene_pca_embeddings(
    adata,
    n_gene_pcs: int = 30,
    n_pca_components: int = 50,
) -> Tuple[np.ndarray, List[str]]:
    """
    Calculate PCA on the single-cell expression matrix
    and return PCA loadings for each gene.

    The loadings provide a compact representation of how
    genes contribute to major transcriptomic variation
    across cells.
    """
    n_pca_components = max(
        n_pca_components,
        n_gene_pcs,
    )

    if (
        "X_pca" not in adata.obsm
        or "PCs" not in adata.varm
        or adata.varm["PCs"].shape[1] < n_gene_pcs
    ):
        sc.tl.pca(
            adata,
            n_comps=n_pca_components,
            svd_solver="arpack",
        )

    gene_pcs = (
        adata.varm["PCs"][:, :n_gene_pcs]
        .astype(np.float32)
    )

    pc_cols = [
        f"gpc_{i + 1}"
        for i in range(n_gene_pcs)
    ]

    return gene_pcs, pc_cols


def build_gene_feature_table(
    adata,
    n_gene_pcs: int = 30,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the complete gene-level feature table.

    Features
    --------
    - single-cell mean expression
    - single-cell variance
    - dropout rate
    - detection rate
    - PCA gene loadings
    """
    gene_mean, gene_var, gene_dropout = (
        compute_single_cell_statistics(
            adata.X
        )
    )

    gene_features = pd.DataFrame(
        {
            "gene": adata.var_names.astype(str),
            "sc_mean": gene_mean,
            "sc_var": gene_var,
            "sc_dropout": gene_dropout,
            "sc_detect": 1.0 - gene_dropout,
        }
    )

    gene_pcs, pc_cols = (
        compute_gene_pca_embeddings(
            adata,
            n_gene_pcs=n_gene_pcs,
        )
    )

    gene_pc_df = pd.DataFrame(
        gene_pcs,
        columns=pc_cols,
    )

    gene_pc_df["gene"] = (
        adata.var_names.astype(str)
    )

    gene_features = gene_features.merge(
        gene_pc_df,
        on="gene",
        how="left",
    )

    feature_cols = [
        "sc_mean",
        "sc_var",
        "sc_dropout",
        "sc_detect",
    ] + pc_cols

    return gene_features, feature_cols


def build_training_features(
    pert_symbols,
    gene_features: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series]:
    """
    Convert training perturbation gene symbols into
    model input features.

    Missing values are imputed using medians calculated
    from the training perturbations.
    """
    feature_map = (
        gene_features
        .set_index("gene")[feature_cols]
    )

    X_df = pd.DataFrame(
        {"pert_symbol": pert_symbols}
    )

    X_df = X_df.join(
        feature_map,
        on="pert_symbol",
    )

    train_medians = (
        X_df[feature_cols]
        .median()
    )

    X_df[feature_cols] = (
        X_df[feature_cols]
        .fillna(train_medians)
    )

    X = (
        X_df[feature_cols]
        .values
        .astype(np.float32)
    )

    return X, X_df, train_medians


def genes_to_features(
    genes,
    gene_features: pd.DataFrame,
    feature_cols: List[str],
    train_medians: pd.Series,
) -> np.ndarray:
    """
    Transform unseen perturbation genes into the exact
    same feature space used during training.
    """
    feature_map = (
        gene_features
        .set_index("gene")[feature_cols]
    )

    df = pd.DataFrame(
        {"pert_symbol": genes}
    )

    df = df.join(
        feature_map,
        on="pert_symbol",
    )

    df[feature_cols] = (
        df[feature_cols]
        .fillna(train_medians)
    )

    return (
        df[feature_cols]
        .values
        .astype(np.float32)
    )
