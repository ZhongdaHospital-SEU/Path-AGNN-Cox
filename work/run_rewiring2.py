# -*- coding: utf-8 -*-
import os, subprocess, sys, time
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
jobs = [
    ("LUAD", ["--known-pathways-file", os.path.join(root, "data", "pathways", "luad_known_pathways.txt")]),
    ("BRCA", []),
]
procs = []
for ds, extra in jobs:
    cmd = [py, "-m", "benchmark.rewiring_analysis",
           "--dataset", ds,
           "--train-csv", os.path.join(root, "data", "processed", ds, "train.csv"),
           "--gmt", os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"),
           "--clinical-csv", os.path.join(root, "data", "processed", "rewiring", "clinical_%s.csv" % ds),
           "--out", os.path.join(root, "results", "rewiring", ds)] + extra
    out = open(os.path.join(root, "work", "rewiring2_%s.log" % ds), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", "rewiring2_%s.err" % ds), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=root, stdout=out, stderr=err)
    procs.append((ds, p, out, err))
    print("started", ds, "pid", p.pid, flush=True)
for ds, p, out, err in procs:
    rc = p.wait()
    out.close(); err.close()
    print("done", ds, "exit", rc, flush=True)
print("ALL_REWIRING2_DONE", flush=True)
