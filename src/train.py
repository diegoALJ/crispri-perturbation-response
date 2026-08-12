import os
import random
from copy import deepcopy
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from .metrics import validation_proxy_score
from .model import SmallMLP


class TabularDataset(Dataset):
    """
    PyTorch dataset for perturbation-level tabular features.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.X = torch.tensor(
            X,
            dtype=torch.float32,
        )

        self.y = torch.tensor(
            y,
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return (
            self.X[index],
            self.y[index],
        )


def seed_everything(
    seed: int = 42,
) -> None:
    """
    Reproducibility configuration.
    """
    random.seed(seed)
    np.random.seed(seed)

    os.environ[
        "PYTHONHASHSEED"
    ] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Simple validation-based early stopping.
    """

    def __init__(
        self,
        patience: int = 35,
        min_delta: float = 1e-6,
    ):
        self.patience = patience
        self.min_delta = min_delta

        self.best = None
        self.counter = 0

    def step(
        self,
        metric: float,
    ) -> bool:

        if (
            self.best is None
            or metric
            < self.best - self.min_delta
        ):
            self.best = metric
            self.counter = 0
            return False

        self.counter += 1

        return (
            self.counter
            >= self.patience
        )


def train_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx,
    val_idx,
    pca,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.25,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 220,
    patience: int = 35,
    clip_grad: float = 1.0,
    scheduler_type: str = "onecycle",
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Train one cross-validation fold.

    Input scaling is fitted exclusively on the training
    fold to prevent validation leakage.
    """
    seed_everything(seed)

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X[train_idx]
    )

    X_val = scaler.transform(
        X[val_idx]
    )

    train_dataset = TabularDataset(
        X_train,
        y[train_idx],
    )

    val_dataset = TabularDataset(
        X_val,
        y[val_idx],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = SmallMLP(
        in_dim=X.shape[1],
        out_dim=y.shape[1],
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_fn = nn.L1Loss()

    if scheduler_type == "onecycle":

        scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=max_epochs,
                steps_per_epoch=max(
                    1,
                    len(train_loader),
                ),
                pct_start=0.1,
                div_factor=10.0,
                final_div_factor=100.0,
            )
        )

        scheduler_per_batch = True

    elif scheduler_type == "cosine":

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
            )
        )

        scheduler_per_batch = False

    else:

        scheduler = None
        scheduler_per_batch = False

    use_amp = (
        device.type == "cuda"
    )

    amp_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    early_stopping = EarlyStopping(
        patience=patience,
    )

    best_score = np.inf
    best_state = None

    for _ in range(max_epochs):

        model.train()

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):
                prediction = model(xb)

                loss = loss_fn(
                    prediction,
                    yb,
                )

            amp_scaler.scale(
                loss
            ).backward()

            amp_scaler.unscale_(
                optimizer
            )

            nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad,
            )

            amp_scaler.step(
                optimizer
            )

            amp_scaler.update()

            if (
                scheduler is not None
                and scheduler_per_batch
            ):
                scheduler.step()

        # Validation
        model.eval()

        predictions = []
        targets = []

        with torch.no_grad():

            for xb, yb in val_loader:

                xb = xb.to(device)

                pred = (
                    model(xb)
                    .cpu()
                    .numpy()
                )

                predictions.append(pred)
                targets.append(
                    yb.numpy()
                )

        latent_pred = np.concatenate(
            predictions
        )

        latent_true = np.concatenate(
            targets
        )

        delta_pred = pca.inverse_transform(
            latent_pred
        )

        delta_true = pca.inverse_transform(
            latent_true
        )

        score = validation_proxy_score(
            latent_true,
            latent_pred,
            delta_true,
            delta_pred,
        )

        if (
            scheduler is not None
            and not scheduler_per_batch
        ):
            scheduler.step()

        if score < best_score:

            best_score = score

            best_state = deepcopy(
                model.state_dict()
            )

        if early_stopping.step(score):
            break

    return {
        "score": float(best_score),
        "state_dict": best_state,
        "scaler": scaler,
    }


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    params: Dict,
    seed: int = 42,
    max_epochs: int = 450,
    device: Optional[torch.device] = None,
):
    """
    Train the selected model on the complete training set.
    """
    seed_everything(seed)

    if device is None:
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    dataset = TabularDataset(
        X_scaled,
        y,
    )

    loader = DataLoader(
        dataset,
        batch_size=params.get(
            "batch_size",
            16,
        ),
        shuffle=True,
    )

    model = SmallMLP(
        in_dim=X.shape[1],
        out_dim=y.shape[1],
        hidden_dim=params.get(
            "hidden_dim",
            64,
        ),
        n_layers=params.get(
            "n_layers",
            2,
        ),
        dropout=params.get(
            "dropout",
            0.25,
        ),
    ).to(device)

    lr = params.get(
        "lr",
        2e-3,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=params.get(
            "weight_decay",
            1e-4,
        ),
    )

    loss_fn = nn.L1Loss()

    scheduler_type = params.get(
        "scheduler",
        "onecycle",
    )

    if scheduler_type == "onecycle":

        scheduler = (
            torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=max_epochs,
                steps_per_epoch=max(
                    1,
                    len(loader),
                ),
                pct_start=0.1,
                div_factor=10.0,
                final_div_factor=100.0,
            )
        )

        scheduler_per_batch = True

    elif scheduler_type == "cosine":

        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_epochs,
            )
        )

        scheduler_per_batch = False

    else:

        scheduler = None
        scheduler_per_batch = False

    use_amp = (
        device.type == "cuda"
    )

    amp_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    model.train()

    for _ in range(max_epochs):

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp,
            ):

                prediction = model(xb)

                loss = loss_fn(
                    prediction,
                    yb,
                )

            amp_scaler.scale(
                loss
            ).backward()

            amp_scaler.unscale_(
                optimizer
            )

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            amp_scaler.step(
                optimizer
            )

            amp_scaler.update()

            if (
                scheduler is not None
                and scheduler_per_batch
            ):
                scheduler.step()

        if (
            scheduler is not None
            and not scheduler_per_batch
        ):
            scheduler.step()

    return model, scaler
