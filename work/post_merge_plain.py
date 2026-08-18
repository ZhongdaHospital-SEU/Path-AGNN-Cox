# -*- coding: utf-8 -*-
"""Post-merge: wait for orchestrator + plain_gnn runner, then insert
plain_gnn external rows into the merged CSV and re-run summarize."""
import os, subprocess, sys, time
import pandas as pd
ROOT = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
PY = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
LOG = os.path.join(ROOT, "work", "postmerge.log")
DATASETS = ["LUAD", "LUSC", "BRCA", "COAD", "STAD", "LIHC", "KIRC", "HNSC", "BLCA", "OV", "GBM"]

def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")
    print(msg, flush=True)

log("POSTMERGE WAITER START")
while True:
    if os.path.exists(os.path.join(ROOT, "work", "rerun_done.flag")) and \
       os.path.exists(os.path.join(ROOT, "work", "plaingnn_done.flag")):
        break
    time.sleep(120)
log("both runners done; inserting plain_gnn external rows")

main_csv = os.path.join(ROOT, "results", "benchmark_results.csv")
df = pd.read_csv(main_csv)
before = len(df)
# drop any stale plain_gnn external rows
df = df[~((df["model"] == "plain_gnn") & (df["split"] == "external"))]
frames = [df]
for ds in DATASETS:
    cp = os.path.join(ROOT, "results", "par_plain_ext", ds, "benchmark_results.csv")
    if os.path.exists(cp):
        f = pd.read_csv(cp)
        f = f[f["split"] == "external"]
        if len(f):
            frames.append(f)
out = pd.concat(frames, ignore_index=True)
out = out.drop_duplicates(subset=["dataset", "split", "model", "cohort"])
out.to_csv(main_csv, index=False)
log("merged %d -> %d rows (plain_gnn external inserted)" % (before, len(out)))
pg = out[(out["model"] == "plain_gnn") & (out["split"] == "external")]
log("plain_gnn external rows: %d" % len(pg))

log("re-running summarize")
p = subprocess.run([PY, "-m", "benchmark.summarize"], cwd=ROOT,
                   stdout=open(os.path.join(ROOT, "work", "postmerge_summarize.log"), "w"),
                   stderr=open(os.path.join(ROOT, "work", "postmerge_summarize.err"), "w"))
log("summarize exit=%d" % p.returncode)
open(os.path.join(ROOT, "work", "postmerge_done.flag"), "w").write("done")
log("POSTMERGE DONE")
