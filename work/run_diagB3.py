# -*- coding: utf-8 -*-
"""diagB config-B workers: OMP=3, batch=64."""
import os, subprocess
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "3"
env["MKL_NUM_THREADS"] = "3"
env["PATH_AGNN_BATCH_SIZE"] = "64"
procs = []
for m in ["path_agnn_cox", "path_agnn_cox_static", "path_agnn_cox_noreg"]:
    cmd = [py, os.path.join(root, "work", "diag_configB.py"), "LUAD", m]
    out = open(os.path.join(root, "work", "diagB3_%s.log" % m), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", "diagB3_%s.err" % m), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=root, env=env, stdout=out, stderr=err)
    procs.append((m, p, out, err))
    print("started", m, p.pid, flush=True)
for m, p, out, err in procs:
    rc = p.wait()
    out.close(); err.close()
    print("done", m, "exit", rc, flush=True)
print("ALL_DIAGB3_DONE", flush=True)
