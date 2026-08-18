
import subprocess, sys, os
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
r = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
jobs = [("GSE17260","data/processed/OV/external/GSE17260.csv"),("GSE7696","data/processed/GBM/external/GSE7696.csv"),("GSE84437","data/processed/STAD/external/GSE84437.csv"),("GSE8894","data/processed/LUSC/external/GSE8894.csv")]
log = open(os.path.join(root,"work","geo_preprocess_new.log"),"w",encoding="utf-8")
err = open(os.path.join(root,"work","geo_preprocess_new.err"),"w",encoding="utf-8")
for gse, out in jobs:
    log.write("===== "+gse+" =====\n"); log.flush()
    p = subprocess.run([r,"data/scripts/04_preprocess_geo.R",gse,out],cwd=root,stdout=log,stderr=err)
    log.write("exit="+str(p.returncode)+"\n"); log.flush()
log.write("DONE\n"); log.close(); err.close()
