"""Generate synthetic processed cohorts to validate the benchmark pipeline."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.dataset_manifest import load_datasets

rng = np.random.default_rng(7)
pathway_dict = {}
genes_all = []
for i in range(6):
    genes = [f"G{i}_{j}" for j in range(30)]
    pathway_dict[f"PATH_{i}"] = genes
    genes_all += genes
genes_all = list(dict.fromkeys(genes_all))

# write GMT
gmt_dir = ROOT / "data" / "pathways"
gmt_dir.mkdir(parents=True, exist_ok=True)
with open(gmt_dir / "KEGG_2021_Human.gmt", "w") as fh:
    for pid, gs in pathway_dict.items():
        fh.write(pid + "\tna\t" + "\t".join(gs) + "\n")

def make_cohort(n, seed, effect=1.0):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, (n, len(genes_all)))
    z = X[:, :30].mean(1) * effect + X[:, 30:60].mean(1) * 0.4 * effect
    time = np.exp(2.2 - 0.6 * z + r.normal(0, 0.6, n))
    event = (r.random(n) < 0.4).astype(int)
    time = np.where(event == 1, time, np.minimum(time, r.uniform(0.5, 5, n)))
    df = pd.DataFrame(X, columns=genes_all)
    df.insert(0, "OS_event", event)
    df.insert(0, "OS_time", time)
    df.insert(0, "sample_id", [f"S{i}" for i in range(n)])
    return df

for ds in load_datasets():
    name = ds["name"]
    out_dir = ROOT / "data" / "processed" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    make_cohort(300, seed=hash(name) % 1000).to_csv(out_dir / "train.csv", index=False)
    ext_dir = out_dir / "external"
    ext_dir.mkdir(exist_ok=True)
    for k, gse in enumerate(ds["external"]):
        make_cohort(120, seed=(hash(name) + k) % 1000, effect=0.8).to_csv(
            ext_dir / f"{gse}.csv", index=False)
    print("synthetic:", name)
print("done")