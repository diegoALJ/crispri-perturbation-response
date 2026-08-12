from typing import List

import numpy as np
import pandas as pd
import torch


def build_validation_mapping(
    df_pert_ids: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract validation perturbation mapping.
    """
    required = {
        "pert_id",
        "pert",
        "class",
    }

    missing = (
        required
        - set(df_pert_ids.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return (
        df_pert_ids[
            df_pert_ids["class"] == "val"
        ][["pert_id", "pert"]]
        .copy()
    )


def predict_latent(
    model,
    X: np.ndarray,
    scaler,
    device=None,
) -> np.ndarray:
    """
    Predict PCA latent transcriptional responses.
    """
    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    X_scaled = scaler.transform(X)

    tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
    ).to(device)

    model.eval()

    with torch.no_grad():

        prediction = (
            model(tensor)
            .cpu()
            .numpy()
        )

    return prediction


def reconstruct_delta(
    latent_predictions: np.ndarray,
    pca,
) -> np.ndarray:
    """
    Convert latent predictions back to the original
    5,127-gene expression space.
    """
    return (
        pca.inverse_transform(
            latent_predictions
        )
        .astype(np.float32)
    )


def blend_with_baseline(
    model_predictions: np.ndarray,
    mean_delta: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    """
    Blend model predictions with the competition baseline.

    prediction =
        (1 - alpha) * baseline
        + alpha * model_prediction
    """
    if not 0 <= alpha <= 1:
        raise ValueError(
            "alpha must be between 0 and 1."
        )

    baseline = mean_delta.reshape(
        1,
        -1,
    )

    return (
        (1.0 - alpha) * baseline
        + alpha * model_predictions
    ).astype(np.float32)


def build_submission(
    val_predictions: np.ndarray,
    gene_cols: List[str],
    mean_delta: np.ndarray,
    n_rows: int = 120,
    n_validation: int = 60,
) -> pd.DataFrame:
    """
    Construct Kaggle submission.

    Validation predictions fill pert_1 ... pert_60.

    Remaining perturbations are initialized with the
    competition mean-delta baseline until their identities
    become available.
    """
    if (
        val_predictions.shape[0]
        != n_validation
    ):
        raise ValueError(
            "Unexpected number of validation predictions."
        )

    if (
        val_predictions.shape[1]
        != len(gene_cols)
    ):
        raise ValueError(
            "Prediction columns do not match gene_cols."
        )

    matrix = np.tile(
        mean_delta.reshape(1, -1),
        (n_rows, 1),
    ).astype(np.float32)

    matrix[
        :n_validation
    ] = val_predictions

    submission = pd.DataFrame(
        matrix,
        columns=gene_cols,
    )

    submission.insert(
        0,
        "pert_id",
        [
            f"pert_{i}"
            for i in range(
                1,
                n_rows + 1,
            )
        ],
    )

    return submission


def validate_submission(
    submission: pd.DataFrame,
    gene_cols,
    expected_rows: int = 120,
) -> None:
    """
    Basic submission integrity checks.
    """
    assert (
        len(submission)
        == expected_rows
    )

    assert (
        submission.columns[0]
        == "pert_id"
    )

    assert (
        list(
            submission.columns[1:]
        )
        == list(gene_cols)
    )

    assert not (
        submission
        .iloc[:, 1:]
        .isna()
        .any()
        .any()
    )

    assert (
        submission["pert_id"].iloc[0]
        == "pert_1"
    )
