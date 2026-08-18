"""Minimal end-to-end demo on synthetic data (no real data required)."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from path_agnn_cox.pathway import load_gmt, build_pathway_adjacency
from path_agnn_cox.models import PathAGNNCox
from path_agnn_cox.train import train_model, predict_risk
from path_agnn_cox.evaluate import full_report

# --- synthetic pathway prior (3 pathways x 20 genes) ---
rng = np.random.default_rng(0)
genes = [f"G{i}" for i in range(60)]
pathway_dict = {
    "PATH_A": genes[0:20], "PATH_B": genes[20:40], "PATH_C": genes[40:60],
}
with open("_tmp.gmt", "w") as fh:
    for pid, gs in pathway_dict.items():
        fh.write(pid + "\tna\t" + "\t".join(gs) + "\n")
pathway_dict = load_gmt("_tmp.gmt")

# --- synthetic cohort: 300 patients, 60 genes, 35% events ---
n = 300
X = rng.normal(0, 1, (n, len(genes)))
# risk driven by pathway A/B expression
z = X[:, :20].mean(1) * 0.8 + X[:, 20:40].mean(1) * 0.5
time = np.exp(2.5 - 0.7 * z + rng.normal(0, 0.6, n))
time = np.clip(time, 0.1, None)
event = (rng.random(n) < 0.35).astype(int)
time = np.where(event == 1, time, np.minimum(time, rng.uniform(1, 5, n)))

adj, mem, gene_order = build_pathway_adjacency(genes, pathway_dict)
Xf = X[:, [genes.index(g) for g in gene_order]]
ids = torch.tensor([list(mem.columns).index(mem.loc[g].idxmax()) for g in gene_order])

# --- train/test split ---
idx = rng.permutation(n)
tr, va = idx[:240], idx[240:]
model = PathAGNNCox(n_genes=len(gene_order), adj=torch.tensor(adj), pathway_ids=ids)
train_model(model, Xf[tr], time[tr], event[tr], Xf[va], time[va], event[va],
            epochs=60, patience=10, lambda_sparse=0.001, lambda_consist=0.1)
risk = predict_risk(model, Xf[va])
print("Validation C-index:", round(full_report(risk, time[va], event[va])["c_index"], 3))
Path("_tmp.gmt").unlink(missing_ok=True)