
import subprocess, os, sys, time
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
datasets = ["COAD","GBM","HNSC","KIRC","LIHC","LUSC","OV","STAD","BLCA"]
log = open(os.path.join(root,"work","benchmark_all9.log"),"w",encoding="utf-8")
err = open(os.path.join(root,"work","benchmark_all9.err"),"w",encoding="utf-8")
for ds in datasets:
    log.write(f"===== {ds} start {time.strftime('%H:%M:%S')} =====\n"); log.flush()
    p = subprocess.run([py,"-m","benchmark.run_benchmark","--datasets",ds,"--folds","5"],
                       cwd=root, stdout=log, stderr=err)
    log.write(f"===== {ds} exit={p.returncode} {time.strftime('%H:%M:%S')} =====\n"); log.flush()
log.write("ALL_DONE\n"); log.close(); err.close()
