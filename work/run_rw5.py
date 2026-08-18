# -*- coding: utf-8 -*-
"""Sequential: finish LUAD rewiring outputs, then full BRCA rewiring. OMP=3, batch=64."""
import os, subprocess
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "3"
env["MKL_NUM_THREADS"] = "3"
env["PATH_AGNN_BATCH_SIZE"] = "64"

def run(args, tag):
    out = open(os.path.join(root, "work", tag + ".log"), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", tag + ".err"), "w", encoding="utf-8")
    p = subprocess.Popen(args, cwd=root, env=env, stdout=out, stderr=err)
    print("started", tag, p.pid, flush=True)
    rc = p.wait()
    out.close(); err.close()
    print("done", tag, "exit", rc, flush=True)
    return rc

rc1 = run([py, os.path.join(root, "work", "finish_rewiring.py")], "rw5_LUAD_finish")
rc2 = run([py, "-m", "benchmark.rewiring_analysis", "--dataset", "BRCA",
           "--train-csv", os.path.join(root, "data", "processed", "BRCA", "train.csv"),
           "--gmt", os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"),
           "--clinical-csv", os.path.join(root, "data", "processed", "rewiring", "clinical_BRCA.csv"),
           "--out", os.path.join(root, "results", "rewiring", "BRCA")],
          "rw5_BRCA")
print("ALL_RW5_DONE", flush=True)
