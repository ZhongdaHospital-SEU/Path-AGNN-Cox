"""DeepSurv / Cox-nnet style MLP survival models (torch + Cox loss)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from path_agnn_cox.loss import cox_ph_loss


class CoxMLP(nn.Module):
    def __init__(self, n_features: int, hidden: list[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _train(model, X, time, event, epochs=150, lr=1e-3, batch_size=64,
           patience=15, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    tt = torch.tensor(time, dtype=torch.float32)
    et = torch.tensor(event, dtype=torch.float32)
    loader = DataLoader(TensorDataset(Xt, tt, et), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best, bad = float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, tb, eb in loader:
            opt.zero_grad()
            loss = cox_ph_loss(model(xb), tb, eb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = cox_ph_loss(model(Xt), tt, et).item()
        if vloss < best:
            best, bad = vloss, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return model


def deepsurv(X, time, event, hidden=(32, 16), **kw):
    model = CoxMLP(X.shape[1], list(hidden))
    _train(model, X, time, event, **kw)
    return _RiskPredictor(model)


def cox_nnet(X, time, event, hidden=(64,), dropout=0.0, **kw):
    model = CoxMLP(X.shape[1], list(hidden), dropout=dropout)
    _train(model, X, time, event, **kw)
    return _RiskPredictor(model)


class _RiskPredictor:
    def __init__(self, model):
        self.model = model

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(np.asarray(X, dtype=float), dtype=torch.float32)).numpy()
