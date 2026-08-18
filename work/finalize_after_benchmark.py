
import os, subprocess, time
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
logf = os.path.join(root, "work", "finalize.log")
csvp = os.path.join(root, "results", "benchmark_results.csv")
par_csvs = {ds: os.path.join(root, "results", "par_%s", "benchmark_results.csv") % ds for ds in ["OV", "STAD", "BLCA"]}
par_logs = {ds: os.path.join(root, "work", "benchmark_par_%s.log" % ds) for ds in ["OV", "STAD", "BLCA"]}
BLCA_FLAG = os.path.join(root, "work", "blca_done.flag")

def log(msg):
    with open(logf, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")

ALL_MODELS = ["path_agnn_cox", "path_agnn_cox_static", "path_agnn_cox_noreg",
              "lasso_cox", "ridge_cox", "elastic_net", "rsf", "deepsurv",
              "cox_nnet", "plain_gnn"]

def lusc_done():
    import pandas as pd
    try:
        df = pd.read_csv(csvp)
        l = df[df["dataset"] == "LUSC"]
        return len(l) >= 70 and set(l["model"].unique()) == set(ALL_MODELS)
    except Exception:
        return False

def par_done():
    for ds in ["OV", "STAD"]:
        if not os.path.exists(par_logs[ds]):
            return False
        with open(par_logs[ds], encoding="utf-8", errors="replace") as f:
            if "Saved" not in f.read():
                return False
    # BLCA runs via the auto-launcher; its completion is signalled by the flag
    if not os.path.exists(BLCA_FLAG):
        return False
    return True

log("FINALIZE STARTED v3 (parallel-aware)")
last_state = None
stagnant_since = None
while True:
    state = (lusc_done(), par_done())
    if all(state):
        log("all datasets complete: %s" % str(state))
        break
    if state != last_state:
        last_state = state
        stagnant_since = None
        log("waiting... LUSC=%s par=%s" % state)
    else:
        if stagnant_since is None:
            stagnant_since = time.time()
        elif time.time() - stagnant_since > 43200:  # 12h hard cap
            log("stagnant 12h; proceeding")
            break
    time.sleep(120)

log("merging results")
import pandas as pd
frames = [pd.read_csv(csvp)]
for ds, cp in par_csvs.items():
    if os.path.exists(cp):
        f = pd.read_csv(cp)
        if len(f):
            log("  par %s rows=%d" % (ds, len(f)))
            frames.append(f)
df = pd.concat(frames, ignore_index=True)
before = len(df)
df = df.drop_duplicates(subset=["dataset", "split", "model", "cohort"])
df.to_csv(csvp, index=False)
log("merged+deduped %d -> %d rows" % (before, len(df)))
print("datasets in final CSV:", sorted(set(df["dataset"])))

log("running summarize")
out = open(os.path.join(root, "work", "finalize_summarize.log"), "w", encoding="utf-8")
err = open(os.path.join(root, "work", "finalize_summarize.err"), "w", encoding="utf-8")
p = subprocess.run([py, "-m", "benchmark.summarize"], cwd=root, stdout=out, stderr=err)
out.close(); err.close()
log("summarize exit=%d" % p.returncode)

jobs = [
    ("LUAD", ["--known-pathways-file", os.path.join(root, "data", "pathways", "luad_known_pathways.txt")]),
    ("BRCA", []),
]
for ds, extra in jobs:
    log("rewiring %s start" % ds)
    cmd = [py, "-m", "benchmark.rewiring_analysis",
           "--dataset", ds,
           "--train-csv", os.path.join(root, "data", "processed", ds, "train.csv"),
           "--gmt", os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"),
           "--clinical-csv", os.path.join(root, "data", "processed", "rewiring", "clinical_%s.csv" % ds),
           "--out", os.path.join(root, "results", "rewiring", ds)] + extra
    out = open(os.path.join(root, "work", "rewiring_%s.log" % ds), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", "rewiring_%s.err" % ds), "w", encoding="utf-8")
    p = subprocess.run(cmd, cwd=root, stdout=out, stderr=err)
    out.close(); err.close()
    log("rewiring %s exit=%d" % (ds, p.returncode))

log("ALL_FINALIZE_DONE")
