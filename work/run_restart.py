# -*- coding: utf-8 -*-
"""Restart crashed jobs with bounded torch threads (OMP/MKL=4) to avoid
memory exhaustion / 0xC0000005. Stagger: 4 jobs first, BRCA rewiring after."""
import os, subprocess
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "4"
env["MKL_NUM_THREADS"] = "4"
env["OPENBLAS_NUM_THREADS"] = "4"

def spawn(args, tag):
    out = open(os.path.join(root, "work", tag + ".log"), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", tag + ".err"), "w", encoding="utf-8")
    p = subprocess.Popen(args, cwd=root, env=env, stdout=out, stderr=err)
    print("started", tag, p.pid, flush=True)
    return p, out, err

jobs = [
    ([py, "-m", "benchmark.rewiring_analysis", "--dataset", "LUAD",
      "--train-csv", os.path.join(root, "data", "processed", "LUAD", "train.csv"),
      "--gmt", os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"),
      "--clinical-csv", os.path.join(root, "data", "processed", "rewiring", "clinical_LUAD.csv"),
      "--out", os.path.join(root, "results", "rewiring", "LUAD"),
      "--known-pathways-file", os.path.join(root, "data", "pathways", "luad_known_pathways.txt")],
     "rw3_LUAD"),
    ([py, os.path.join(root, "work", "diag_configB.py"), "LUAD", "path_agnn_cox"], "diagB2_full"),
    ([py, os.path.join(root, "work", "diag_configB.py"), "LUAD", "path_agnn_cox_static"], "diagB2_static"),
    ([py, os.path.join(root, "work", "diag_configB.py"), "LUAD", "path_agnn_cox_noreg"], "diagB2_noreg"),
]
procs = []
for args, tag in jobs:
    p, out, err = spawn(args, tag)
    procs.append((tag, p, out, err))
for tag, p, out, err in procs:
    rc = p.wait()
    out.close(); err.close()
    print("done", tag, "exit", rc, flush=True)
# BRCA rewiring after the batch (memory headroom)
args = [py, "-m", "benchmark.rewiring_analysis", "--dataset", "BRCA",
        "--train-csv", os.path.join(root, "data", "processed", "BRCA", "train.csv"),
        "--gmt", os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"),
        "--clinical-csv", os.path.join(root, "data", "processed", "rewiring", "clinical_BRCA.csv"),
        "--out", os.path.join(root, "results", "rewiring", "BRCA")]
p, out, err = spawn(args, "rw3_BRCA")
rc = p.wait()
out.close(); err.close()
print("done BRCA exit", rc, flush=True)
print("ALL_RESTART_DONE", flush=True)
