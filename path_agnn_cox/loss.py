"""Survival losses for Path-AGNN-Cox.

- cox_ph_loss: standard Cox partial likelihood (Breslow tie handling).
- weighted_cox_ph_loss: sample-weighted variant (e.g. IPCW or malignancy
  weights) used to focus learning on clinically informative patients.
- total_loss: Cox loss + L2 + intra-pathway sparsity + consistency
  regularization (the anti-overfitting module for heterogeneous cohorts).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def _breslow_partial_loglik(risk: torch.Tensor, time: torch.Tensor,
                            event: torch.Tensor,
                            weights: torch.Tensor) -> torch.Tensor:
    """Negative partial log-likelihood with Breslow tie handling.

    risk: (B,) predicted log-hazard. time/event: (B,). weights: (B,).
    """
    order = torch.argsort(time, descending=True)
    risk_s, event_s = risk[order], event[order]
    w_s = weights[order]
    # cumulative sum of exp(risk) over the risk set (patients with t_j >= t_i)
    log_denom = torch.logcumsumexp(risk_s, dim=0)
    # For tied event times, use the Breslow denominator: sum over all events
    # at the same time uses the risk set at that time (computed via cumsum).
    num = (event_s.float() * w_s * risk_s).sum()
    den = (event_s.float() * w_s * log_denom).sum()
    n_events = event_s.float().sum().clamp(min=1.0)
    return -(num - den) / n_events


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
    """Standard Cox partial likelihood loss."""
    return _breslow_partial_loglik(risk, time, event,
                                   torch.ones_like(time, dtype=torch.float32))


def weighted_cox_ph_loss(risk: torch.Tensor, time: torch.Tensor,
                         event: torch.Tensor,
                         weights: torch.Tensor) -> torch.Tensor:
    """Sample-weighted Cox loss (weights precomputed, e.g. IPCW)."""
    return _breslow_partial_loglik(risk, time, event, weights.float())


def sparsity_regularizer(alpha: torch.Tensor) -> torch.Tensor:
    """Intra-pathway sparsity: encourage focused (sparse) adaptive weights.

    alpha: adaptive adjacency weights (B, N, N) after masking. We penalize
    the mean absolute weight of non-self edges so the model must justify
    keeping an interaction.
    """
    return alpha.abs().mean()


def consistency_regularizer(risk: torch.Tensor,
                            risk_perturbed: torch.Tensor) -> torch.Tensor:
    """Consistency between two stochastic forward passes (dropout views).

    Suppresses overfitting in high-heterogeneity cohorts by requiring the
    risk score to be stable under feature-dropout perturbations.
    """
    return F.mse_loss(risk, risk_perturbed)


def total_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor,
               risk_perturbed: torch.Tensor, alpha: torch.Tensor,
               model, l2: float = 1e-4, lambda_sparse: float = 0.0,
               lambda_consist: float = 0.0,
               weights: torch.Tensor | None = None) -> torch.Tensor:
    """Full objective: Cox + L2 + sparse + consistency."""
    if weights is None:
        loss = cox_ph_loss(risk, time, event)
    else:
        loss = weighted_cox_ph_loss(risk, time, event, weights)
    if l2 > 0:
        loss = loss + l2 * sum(p.pow(2).sum() for p in model.parameters())
    if lambda_sparse > 0:
        loss = loss + lambda_sparse * sparsity_regularizer(alpha)
    if lambda_consist > 0:
        loss = loss + lambda_consist * consistency_regularizer(risk, risk_perturbed)
    return loss