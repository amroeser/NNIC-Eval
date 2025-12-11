from typing import Any, Dict, List

import torch
from torch import nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    return nn.ReLU()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: List[int],
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_get_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_mlp(input_dim: int, num_classes: int, config: Dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    hidden_layers = model_cfg.get("hidden_layers", [128, 64])
    activation = model_cfg.get("activation", "relu")
    dropout = float(model_cfg.get("dropout", 0.0))
    return MLP(input_dim=input_dim, num_classes=num_classes, hidden_layers=hidden_layers, activation=activation, dropout=dropout)
