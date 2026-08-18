# -*- coding: utf-8 -*-
"""Parallel: finish LUAD + finish BRCA rewiring outputs (clinical corr + static null)."""
import os, subprocess
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "3"
env["MKL_NUM_THREADS"] = "3"
env["PATH_AGNN_BATCH_SIZE"] = "64"
procs = []
for ds in ["LUAD", "BRCA"]:
    out = open(os.path.join(root, "work", "rw6_%s.log" % ds), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", "rw6_%s.err" % ds), "w", encoding="utf-8")
    p = subprocess.Popen([py, os.path.join(root, "work", "finish_rewiring.py"), "--dataset", ds],
                         cwd=root, env=env, stdout=out, stderr=err)
    procs.append((ds, p, out, err))
    print("started", ds, p.pid, flush=True)
for ds, p, out, err in procs:
    rc = p.wait()
    out.close(); err.close()
    print("done", ds, "exit", rc, flush=True)
print("ALL_RW6_DONE", flush=True)
