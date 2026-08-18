# -*- coding: utf-8 -*-
"""Orchestrator: fixed-model re-runs (phase1 internal+external for full/noreg;
phase2 external-only for static/plain/deepsurv/cox_nnet), then merge + summarize
+ rewiring with corrected results."""
import os, subprocess, sys, time, json

ROOT = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
PY = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
LOG = os.path.join(ROOT, "work", "rerun.log")

DATASETS = ["LUAD", "LUSC", "BRCA", "COAD", "STAD", "LIHC", "KIRC",
            "HNSC", "BLCA", "OV", "GBM"]
PHASE1_MODELS = "path_agnn_cox,path_agnn_cox_noreg"
PHASE2_MODELS = "path_agnn_cox_static,plain_gnn,deepsurv,cox_nnet"
N_WORKERS = 3

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)

def run_job(ds, models, out_dir, external_only=False):
    cmd = [PY, "-m", "benchmark.run_benchmark", "--datasets", ds,
           "--models", models, "--out", out_dir]
    if external_only:
        cmd.append("--external-only")
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    tag = "ext" if external_only else "full"
    out = open(os.path.join(ROOT, "work", "rerun_%s_%s.log" % (tag, ds)), "w", encoding="utf-8")
    err = open(os.path.join(ROOT, "work", "rerun_%s_%s.err" % (tag, ds)), "w", encoding="utf-8")
    p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=err)
    out.close(); err.close()
    return p.returncode

def merge():
    import pandas as pd
    main_csv = os.path.join(ROOT, "results", "benchmark_results.csv")
    df = pd.read_csv(main_csv)
    # drop rows to be replaced
    drop = (df["model"].isin(["path_agnn_cox", "path_agnn_cox_noreg"])) | \
           ((df["model"].isin(["path_agnn_cox_static", "plain_gnn", "deepsurv", "cox_nnet"])) & (df["split"] == "external"))
    df = df[~drop]
    frames = [df]
    for ds in DATASETS:
        for tag, d in (("full", "par_fixed"), ("ext", "par_fixext")):
            cp = os.path.join(ROOT, "results", d, ds, "benchmark_results.csv")
            if os.path.exists(cp):
                f = pd.read_csv(cp)
                if len(f):
                    frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out = out.drop_duplicates(subset=["dataset", "split", "model", "cohort"])
    out.to_csv(main_csv, index=False)
    log("merged %d -> %d rows" % (before, len(out)))
    # sanity: every dataset x model should have 5 cv folds
    miss = out[(out["split"] == "cv")].groupby(["dataset", "model"]).size()
    bad = miss[miss != 5]
    log("datasets x models with !=5 cv folds: %d" % len(bad))
    return out

log("ORCHESTRATOR START (phase1 full+noreg, phase2 ext-only)")

def worker(queue, result):
    while True:
        try:
            job = queue.pop(0)
        except IndexError:
            return
        ds, models, out_dir, ext = job
        log("  [worker] start %s %s %s" % (ds, "ext" if ext else "full", models))
        rc = run_job(ds, models, out_dir, ext)
        log("  [worker] done %s rc=%d" % (ds, rc))
        result.append((ds, ext, rc))

# phase 1
queue = [(ds, PHASE1_MODELS, os.path.join(ROOT, "results", "par_fixed", ds), False)
         for ds in DATASETS]
result = []
threads = []
import threading
for i in range(N_WORKERS):
    t = threading.Thread(target=worker, args=(queue, result))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
log("PHASE1 COMPLETE")

# phase 2
queue = [(ds, PHASE2_MODELS, os.path.join(ROOT, "results", "par_fixext", ds), True)
         for ds in DATASETS]
result = []
threads = []
for i in range(N_WORKERS):
    t = threading.Thread(target=worker, args=(queue, result))
    t.start()
    threads.append(t)
for t in threads:
    t.join()
log("PHASE2 COMPLETE")

merge()

log("running summarize")
p = subprocess.run([PY, "-m", "benchmark.summarize"], cwd=ROOT,
                   stdout=open(os.path.join(ROOT, "work", "rerun_summarize.log"), "w"),
                   stderr=open(os.path.join(ROOT, "work", "rerun_summarize.err"), "w"))
log("summarize exit=%d" % p.returncode)

for ds, extra in [("LUAD", ["--known-pathways-file", os.path.join(ROOT, "data", "pathways", "luad_known_pathways.txt")]),
                  ("BRCA", [])]:
    cmd = [PY, "-m", "benchmark.rewiring_analysis", "--dataset", ds,
           "--train-csv", os.path.join(ROOT, "data", "processed", ds, "train.csv"),
           "--gmt", os.path.join(ROOT, "data", "pathways", "kegg_cancer_core.gmt"),
           "--clinical-csv", os.path.join(ROOT, "data", "processed", "rewiring", "clinical_%s.csv" % ds),
           "--out", os.path.join(ROOT, "results", "rewiring", ds)] + extra
    log("rewiring %s start" % ds)
    p = subprocess.run(cmd, cwd=ROOT,
                       stdout=open(os.path.join(ROOT, "work", "rerun_rewiring_%s.log" % ds), "w"),
                       stderr=open(os.path.join(ROOT, "work", "rerun_rewiring_%s.err" % ds), "w"))
    log("rewiring %s exit=%d" % (ds, p.returncode))

open(os.path.join(ROOT, "work", "rerun_done.flag"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
log("ALL_RERUN_DONE")
