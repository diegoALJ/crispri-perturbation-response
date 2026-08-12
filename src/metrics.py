import numpy as np


def mae_latent(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Mean Absolute Error in latent PCA space.
    """
    return float(
        np.mean(
            np.abs(y_true - y_pred)
        )
    )


def cosine_similarity_flat(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Cosine similarity between two flattened matrices.

    Used as a lightweight validation proxy.
    """
    a = y_true.reshape(-1)
    b = y_pred.reshape(-1)

    numerator = np.dot(a, b)

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
        + eps
    )

    return float(
        numerator / denominator
    )


def validation_proxy_score(
    latent_true: np.ndarray,
    latent_pred: np.ndarray,
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    cosine_weight: float = 0.10,
) -> float:
    """
    Proxy metric used during MLP optimization.

    Lower is better.
    """
    mae = mae_latent(
        latent_true,
        latent_pred,
    )

    cosine = cosine_similarity_flat(
        delta_true,
        delta_pred,
    )

    return (
        mae
        - cosine_weight * cosine
    )


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
) -> float:
    """
    Weighted Mean Absolute Error used by the competition.
    """
    return float(
        np.mean(
            weights
            * np.abs(y_true - y_pred)
        )
    )


def weighted_cosine(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    right: float = 0.3,
    eps: float = 1e-12,
) -> float:
    """
    Weighted cosine similarity following the competition's
    smooth-gating formulation.
    """
    a = np.asarray(
        y_true,
        dtype=np.float64,
    ).reshape(-1)

    b = np.asarray(
        y_pred,
        dtype=np.float64,
    ).reshape(-1)

    x = np.maximum(
        np.abs(a),
        np.abs(b),
    )

    t = np.clip(
        x / right,
        0.0,
        1.0,
    )

    # smoothstep
    weights = (
        t**2
        * (3.0 - 2.0 * t)
    )

    weights_sq = weights**2

    numerator = np.sum(
        weights_sq * a * b
    )

    denominator = np.sqrt(
        np.sum(weights_sq * a**2)
        * np.sum(weights_sq * b**2)
    )

    if denominator < eps:
        return 0.0

    return float(
        numerator / denominator
    )
