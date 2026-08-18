# -*- coding: utf-8 -*-
"""plain_gnn external-only re-run (post-fix), 2 workers."""
import os, subprocess, sys, time, threading
ROOT = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
PY = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
LOG = os.path.join(ROOT, "work", "plaingnn.log")
DATASETS = ["LUAD", "LUSC", "BRCA", "COAD", "STAD", "LIHC", "KIRC", "HNSC", "BLCA", "OV", "GBM"]
N_WORKERS = 2

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)

def run_job(ds):
    out_dir = os.path.join(ROOT, "results", "par_plain_ext", ds)
    cmd = [PY, "-m", "benchmark.run_benchmark", "--datasets", ds,
           "--models", "plain_gnn", "--out", out_dir, "--external-only"]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"; env["MKL_NUM_THREADS"] = "4"
    out = open(os.path.join(ROOT, "work", "plaingnn_%s.log" % ds), "w", encoding="utf-8")
    err = open(os.path.join(ROOT, "work", "plaingnn_%s.err" % ds), "w", encoding="utf-8")
    p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=err)
    out.close(); err.close()
    return p.returncode

def worker(queue):
    while True:
        try:
            ds = queue.pop(0)
        except IndexError:
            return
        log("  [worker] start plain_gnn ext %s" % ds)
        rc = run_job(ds)
        log("  [worker] done plain_gnn ext %s rc=%d" % (ds, rc))

log("PLAIN_GNN RUNNER START (2 workers)")
queue = list(DATASETS)
threads = []
for i in range(N_WORKERS):
    t = threading.Thread(target=worker, args=(queue,)); t.start(); threads.append(t)
for t in threads:
    t.join()
log("PLAIN_GNN ALL DONE")
open(os.path.join(ROOT, "work", "plaingnn_done.flag"), "w").write("done")
