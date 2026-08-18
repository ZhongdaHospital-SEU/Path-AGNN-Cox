# -*- coding: utf-8 -*-
import os, subprocess
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "3"
env["MKL_NUM_THREADS"] = "3"
out = open(os.path.join(root, "work", "perm2.log"), "w", encoding="utf-8")
err = open(os.path.join(root, "work", "perm2.err"), "w", encoding="utf-8")
p = subprocess.Popen([py, os.path.join(root, "work", "permutation_test.py")], cwd=root, env=env, stdout=out, stderr=err)
print("started perm2 pid", p.pid, flush=True)
rc = p.wait()
out.close(); err.close()
print("done perm2 exit", rc, flush=True)
