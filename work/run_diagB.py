# -*- coding: utf-8 -*-
import os, subprocess, sys
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
models = ["path_agnn_cox", "path_agnn_cox_static", "path_agnn_cox_noreg", "plain_gnn"]
procs = []
for m in models:
    cmd = [py, os.path.join(root, "work", "diag_configB.py"), "LUAD", m]
    out = open(os.path.join(root, "work", "diagB_%s.log" % m), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", "diagB_%s.err" % m), "w", encoding="utf-8")
    p = subprocess.Popen(cmd, cwd=root, stdout=out, stderr=err)
    procs.append((m, p, out, err))
    print("started", m, p.pid, flush=True)
for m, p, out, err in procs:
    rc = p.wait()
    out.close(); err.close()
    print("done", m, "exit", rc, flush=True)
print("ALL_DIAG_DONE", flush=True)
