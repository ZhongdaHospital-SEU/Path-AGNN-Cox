"""Penalized Cox models: LASSO / Ridge / Elastic-Net via R glmnet.

Runs `glmnet` (alpha = 1 / 0 / 0.5) through Rscript on the exact same
pathway-mapped feature matrix as all other models. glmnet handles p > n and
returns sparse coefficient vectors, avoiding lifelines\'s O(n^2) Hessian.
"""
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RSCRIPT = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
GLMNET_R = ROOT / "data" / "scripts" / "cox_glmnet.R"


class PenalizedCox:
    """Wrapper around R glmnet with l1_ratio (alpha) and penalizer (lambda)."""

    def __init__(self, l1_ratio: float = 0.0, penalizer: float = 0.1):
        self.l1_ratio = l1_ratio
        self.penalizer = penalizer
        self.coef = None

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray, seed: int = 42, **kw):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            xf, tf, ef, of = (td / "X.csv"), (td / "t.txt"), (td / "e.txt"), (td / "coef.csv")
            np.savetxt(xf, X, delimiter=",")
            np.savetxt(tf, np.asarray(time, dtype=float))
            np.savetxt(ef, np.asarray(event, dtype=int))
            r = subprocess.run(
                [RSCRIPT, str(GLMNET_R), str(xf), str(tf), str(ef),
                 str(self.l1_ratio), str(seed), str(of)],
                capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"glmnet failed: {r.stderr[-800:]}")
            self.coef = np.loadtxt(of).ravel()
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coef


def lasso_cox(X, time, event, penalizer=0.05, **kw):
    return PenalizedCox(l1_ratio=1.0, penalizer=penalizer).fit(X, time, event)


def ridge_cox(X, time, event, penalizer=0.1, **kw):
    return PenalizedCox(l1_ratio=0.0, penalizer=penalizer).fit(X, time, event)


def elastic_net_cox(X, time, event, l1_ratio=0.5, penalizer=0.1, **kw):
    return PenalizedCox(l1_ratio=l1_ratio, penalizer=penalizer).fit(X, time, event)
