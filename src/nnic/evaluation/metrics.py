from typing import Any, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset


def model_predict(model: torch.nn.Module, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    """Batched Vorhersagen eines Torch-Modells für ein NumPy-Array X.

    Dient als gemeinsame Utility-Funktion, um auf CPU/GPU effiziente Inferenz
    mit konsistenter Batch-Größe durchzuführen.
    """
    model.eval()
    X_tensor = torch.from_numpy(X).float()
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            preds.append(predicted.cpu().numpy())
    return np.concatenate(preds, axis=0)


def basic_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Berechnet Basis-Klassifikationsmetriken für y_true/y_pred.

    Neben Accuracy, Precision, Recall, F1 und Confusion-Matrix werden auch
    die absolute Fehleranzahl und die Fehlerrate (1 - Accuracy) zurückgegeben,
    damit Fehlerreduktion durch NNIC explizit ausgewiesen werden kann.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    n_samples = int(y_true.shape[0]) if y_true.size > 0 else 0
    num_errors = int((y_true != y_pred).sum()) if n_samples > 0 else 0
    error_rate = float(num_errors / n_samples) if n_samples > 0 else 0.0

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "num_errors": num_errors,
        "error_rate": error_rate,
    }
