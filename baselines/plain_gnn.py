"""Plain GNN survival baseline: same backbone, no pathway constraint.

Uses an identity / fully-connected adjacency and global pooling, i.e. the
Path-AGNN-Cox model without the pathway prior (ablation: -Pathway).
"""
from __future__ import annotations
import numpy as np
import torch
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_risk


def plain_gnn_survival(X, time, event, X_val=None, time_val=None, event_val=None,
                       adj_mode: str = "identity", **train_kw):
    """adj_mode: 'identity' (no edges) or 'full' (fully connected)."""
    n = X.shape[1]
    if adj_mode == "identity":
        adj = torch.eye(n)
    else:
        adj = torch.ones(n, n)
    model = PathAGNNCox(n, adj=adj, pathway_ids=None)
    train_model(model, X, time, event, X_val, time_val, event_val, **train_kw)
    return _GNNPredictor(model)


class _GNNPredictor:
    def __init__(self, model):
        self.model = model

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return predict_risk(self.model, X)
