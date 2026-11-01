"""A small PyTorch neural network (multi-layer perceptron) for the repurchase task.

It exists to show a hand-written training loop, and to be compared honestly with gradient boosting.
This is the ONLY file that imports torch, and only the offline training run uses it: the deployed
app never imports torch (it reads the saved scores).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class MLP(nn.Module):
    """input -> 64 -> 32 -> 1 logit. ReLU adds non-linearity; dropout randomly zeroes 20% of the
    units during training so the network cannot memorise the few thousand training rows."""

    def __init__(self, n_features: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers, width = [], n_features
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Xtr, ytr, Xva, yva, seed: int = 42, max_epochs: int = 200, patience: int = 15,
            lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256):
    """Train with mini-batches and stop when validation ROC-AUC has not improved for `patience` epochs.

    Returns (predict_function, history). predict_function maps a feature matrix to probabilities.
    Inputs must already be log-transformed and standardised.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    rng = np.random.default_rng(seed)

    Xtr_t = torch.tensor(np.asarray(Xtr), dtype=torch.float32)
    ytr_t = torch.tensor(np.asarray(ytr), dtype=torch.float32)
    Xva_t = torch.tensor(np.asarray(Xva), dtype=torch.float32)

    model = MLP(Xtr_t.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy in one stable op
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_state, bad_epochs, history = -1.0, None, 0, []
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(len(Xtr_t))
        total = 0.0
        for start in range(0, len(order), batch_size):
            idx = torch.tensor(order[start:start + batch_size])
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val_auc = roc_auc_score(yva, torch.sigmoid(model(Xva_t)).numpy())
        history.append({"epoch": epoch + 1, "train_loss": total / len(order), "val_auc": float(val_auc)})
        if val_auc > best_auc + 1e-5:
            best_auc, best_state, bad_epochs = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    def predict(X) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(model(torch.tensor(np.asarray(X), dtype=torch.float32))).numpy()

    return predict, history
