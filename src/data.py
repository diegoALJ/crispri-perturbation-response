from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc


def load_competition_data(
    path_means: str,
    path_pert_ids: str,
    path_cells: str,
    path_gt: str,
) -> Dict:
    """
    Load the main competition datasets.

    Returns
    -------
    dict
        Dictionary containing:
        - means
        - pert_ids
        - ground_truth
        - adata
    """
    data = {
        "means": pd.read_csv(Path(path_means)),
        "pert_ids": pd.read_csv(Path(path_pert_ids)),
        "ground_truth": pd.read_csv(Path(path_gt)),
        "adata": sc.read_h5ad(Path(path_cells)),
    }

    return data


def build_delta_matrix(
    df_means: pd.DataFrame,
    control_label: str = "non-targeting",
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.Index,
    pd.DataFrame,
]:
    """
    Construct perturbation-induced delta expression matrix.

    Delta expression is calculated as:

        perturbation mean expression - non-targeting expression

    Parameters
    ----------
    df_means:
        training_data_means.csv
    control_label:
        Label identifying the non-targeting control.

    Returns
    -------
    delta_matrix:
        Shape (n_perturbations, n_genes).

    pert_symbols:
        Gene symbol targeted by each perturbation.

    baseline:
        Non-targeting expression vector.

    gene_cols:
        Ordered list of target genes.

    df_train:
        Training dataframe excluding the control.
    """
    if "pert_symbol" not in df_means.columns:
        raise ValueError("Expected column 'pert_symbol' not found.")

    gene_cols = df_means.columns.drop("pert_symbol")

    control_rows = df_means[
        df_means["pert_symbol"] == control_label
    ]

    if len(control_rows) != 1:
        raise ValueError(
            f"Expected exactly one '{control_label}' row, "
            f"found {len(control_rows)}."
        )

    baseline = (
        control_rows
        .iloc[0][gene_cols]
        .astype(np.float32)
        .values
    )

    df_train = (
        df_means[df_means["pert_symbol"] != control_label]
        .reset_index(drop=True)
    )

    expression = (
        df_train[gene_cols]
        .astype(np.float32)
        .values
    )

    delta_matrix = expression - baseline

    pert_symbols = (
        df_train["pert_symbol"]
        .astype(str)
        .values
    )

    return (
        delta_matrix,
        pert_symbols,
        baseline,
        gene_cols,
        df_train,
    )


def compute_mean_delta(
    delta_matrix: np.ndarray,
) -> np.ndarray:
    """
    Competition baseline: arithmetic mean of the training
    perturbation delta-expression vectors.
    """
    return (
        np.asarray(delta_matrix)
        .mean(axis=0)
        .astype(np.float32)
    )
