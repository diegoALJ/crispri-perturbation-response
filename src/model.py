import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """
    Lightweight multilayer perceptron for predicting
    latent transcriptional responses from gene-level
    perturbation features.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()

        layers = []
        current_dim = in_dim

        for _ in range(n_layers):

            layers.extend(
                [
                    nn.Linear(
                        current_dim,
                        hidden_dim,
                    ),
                    nn.LayerNorm(
                        hidden_dim
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        dropout
                    ),
                ]
            )

            current_dim = hidden_dim

        layers.append(
            nn.Linear(
                current_dim,
                out_dim,
            )
        )

        self.network = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)
