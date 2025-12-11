from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from nnic.models.nn_models import build_mlp


class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader | None:
    if X is None or y is None:
        return None
    dataset = NumpyDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _select_device(training_cfg: Dict[str, Any]) -> torch.device:
    device_str = str(training_cfg.get("device", "cpu")).lower()
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_criterion(training_cfg: Dict[str, Any]) -> nn.Module:
    loss_name = str(training_cfg.get("loss", "cross_entropy")).lower()
    if loss_name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss()
    if loss_name in {"mse", "mse_loss"}:
        return nn.MSELoss()
    # Fallback auf CrossEntropy, falls Unbekanntes konfiguriert wurde
    return nn.CrossEntropyLoss()


def _build_optimizer(model: nn.Module, training_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(training_cfg.get("optimizer", "adam")).lower()
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))

    if optimizer_name == "sgd":
        momentum = float(training_cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name in {"adamw"}:
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Default: Adam
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def train_nn_model(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    config: Dict[str, Any],
    output_dirs: Dict[str, Path],
) -> Tuple[nn.Module, torch.device, Dict[str, list]]:
    X_train, y_train = splits["train"]
    X_val, y_val = splits.get("val", (None, None))

    input_dim = X_train.shape[1]
    num_classes = int(len(np.unique(y_train)))

    training_cfg = config.get("training", {})
    batch_size = int(training_cfg.get("batch_size", 64))
    num_epochs = int(training_cfg.get("num_epochs", 20))

    device = _select_device(training_cfg)

    model = build_mlp(input_dim=input_dim, num_classes=num_classes, config=config)
    model.to(device)

    train_loader = _create_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = _create_dataloader(X_val, y_val, batch_size=batch_size, shuffle=False) if X_val is not None else None

    criterion = _build_criterion(training_cfg)
    optimizer = _build_optimizer(model, training_cfg)

    history: Dict[str, list] = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for _ in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            epoch_val_loss = val_loss / val_total if val_total > 0 else 0.0
            epoch_val_acc = val_correct / val_total if val_total > 0 else 0.0
            history["val_loss"].append(epoch_val_loss)
            history["val_acc"].append(epoch_val_acc)
        else:
            history["val_loss"].append(0.0)
            history["val_acc"].append(0.0)

    return model, device, history
