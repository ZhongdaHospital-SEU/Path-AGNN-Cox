"""Training loop for Path-AGNN-Cox with early stopping and CV helpers."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from .loss import total_loss
from .evaluate import c_index


def _to_tensor(X: np.ndarray, time: np.ndarray, event: np.ndarray, device):
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(time, dtype=torch.float32, device=device),
            torch.tensor(event, dtype=torch.float32, device=device))


def train_epoch(model, loader, optimizer, device, lambda_sparse, lambda_consist,
                l2, dropout_seed_view=True):
    model.train()
    total = 0.0
    for xb, tb, eb in loader:
        optimizer.zero_grad()
        risk = model(xb)
        if lambda_consist > 0 and dropout_seed_view:
            # second stochastic view for consistency regularization
            with torch.no_grad():
                risk2 = model(xb)
        else:
            risk2 = risk
        sp = model.layers[-1].sparsity_penalty()
        alpha = sp if sp is not None else torch.zeros_like(risk)
        loss = total_loss(risk, tb, eb, risk2, alpha,
                          model, l2=l2, lambda_sparse=lambda_sparse,
                          lambda_consist=lambda_consist)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(xb)
    return total / max(len(loader.dataset), 1)


def train_model(model, X: np.ndarray, time: np.ndarray, event: np.ndarray,
                X_val=None, time_val=None, event_val=None,
                epochs: int = 200, lr: float = 1e-3, batch_size: int = 64,
                l2: float = 1e-4, lambda_sparse: float = 0.0,
                lambda_consist: float = 0.0, patience: int = 20,
                seed: int = 0, device: str = "cpu", verbose: bool = False):
    """Train with optional early stopping on validation C-index."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    Xt, tt, et = _to_tensor(X, time, event, device)
    ds = TensorDataset(Xt, tt, et)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    best = -1.0
    bad = 0
    best_state = None
    for ep in range(epochs):
        train_epoch(model, loader, optimizer, device,
                    lambda_sparse, lambda_consist, l2)
        if X_val is not None:
            risk = predict_risk(model, X_val, device)
            score = c_index(risk, time_val, event_val)
            if score > best:
                best = score
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
        elif ep == epochs - 1:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_risk(model, X: np.ndarray, device="cpu") -> np.ndarray:
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        risk = model(x).cpu().numpy()
    return risk

@torch.no_grad()
def predict_with_alpha(model, X: np.ndarray, device="cpu"):
    """Return (risk, alpha, src, dst) with per-sample edge weights.

    alpha has shape (B, E) where E is the number of pathway edges; alpha[b, e]
    is the attention weight of edge e (src[e] -> dst[e]) for sample b, i.e.
    row-normalized over the pathway-masked neighbor set. For static models
    (use_adaptive=False) the fixed normalized adjacency weights are returned
    (identical across samples, serving as the H4 negative control).
    """
    model.eval()
    model = model.to(device)
    x = torch.tensor(X, dtype=torch.float32, device=device)
    risk, alpha, src, dst = model(x, return_alpha=True)
    if alpha is None:  # static mode: use fixed normalized edge weights
        layer = model.layers[-1]
        alpha = layer.adj_weight.unsqueeze(0).expand(x.shape[0], -1)
        src, dst = layer.src, layer.dst
    return risk.cpu().numpy(), alpha.cpu().numpy(), src.cpu().numpy(), dst.cpu().numpy()
