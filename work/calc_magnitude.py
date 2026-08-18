# Compute per-patient rewiring magnitude from alpha.npy (LUAD + BRCA) in parallel subprocesses
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
DS = sys.argv[1]
alpha = np.load(ROOT / "results" / "rewiring" / DS / "alpha.npy")   # (n, n_edges)
print(DS, "alpha", alpha.shape, alpha.dtype, flush=True)
mu = alpha.mean(axis=0, keepdims=True)
mag = np.abs(alpha - mu).sum(axis=1)
risk = pd.read_csv(ROOT / "results" / "rewiring" / DS / "risk_scores.csv")
risk["sample_id"] = risk["sample_id"].str[:12]
risk = risk.drop_duplicates(subset="sample_id", keep="first")
risk["rewiring_magnitude"] = mag[: len(risk)]
risk.to_csv(ROOT / "results" / "rewiring" / DS / "rewiring_magnitude.csv", index=False)
print(DS, "magnitude saved", risk.shape, "mean mag:", mag.mean(), flush=True)
