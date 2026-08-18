
import os, subprocess, time
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
csvp = os.path.join(root, "results", "benchmark_results.csv")
flag = os.path.join(root, "work", "blca_done.flag")
logf = os.path.join(root, "work", "blca_launcher.log")
def log(m):
    with open(logf, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + m + "\n")
log("launcher started")
while True:
    try:
        import pandas as pd
        df = pd.read_csv(csvp)
        l = df[df["dataset"] == "LUSC"]
        if len(l) >= 70:
            break
    except Exception:
        pass
    time.sleep(90)
log("LUSC complete, launching BLCA")
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "4"
env["MKL_NUM_THREADS"] = "4"
out = open(os.path.join(root, "work", "benchmark_par_BLCA.log"), "w", encoding="utf-8")
err = open(os.path.join(root, "work", "benchmark_par_BLCA.err"), "w", encoding="utf-8")
p = subprocess.run([py, "-m", "benchmark.run_benchmark", "--datasets", "BLCA", "--folds", "5",
                    "--out", os.path.join(root, "results", "par_BLCA")],
                   cwd=root, env=env, stdout=out, stderr=err)
log("BLCA exit=%d" % p.returncode)
open(flag, "w").write("done")
log("blca_done flag written")
