
import subprocess, os
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
r = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
jobs = [("GSE29609","data/processed/KIRC/external/GSE29609.csv"),("GSE65858","data/processed/HNSC/external/GSE65858.csv"),("GSE32894","data/processed/BLCA/external/GSE32894.csv"),("GSE13507","data/processed/BLCA/external/GSE13507.csv"),("GSE32062","data/processed/OV/external/GSE32062.csv"),("GSE14520","data/processed/LIHC/external/GSE14520.csv"),("GSE116174","data/processed/LIHC/external/GSE116174.csv")]
log = open(os.path.join(root,"work","geo_batchA.log"),"w",encoding="utf-8")
err = open(os.path.join(root,"work","geo_batchA.err"),"w",encoding="utf-8")
for gse, out in jobs:
    log.write("===== "+gse+" =====\n"); log.flush()
    p = subprocess.run([r,"data/scripts/04_preprocess_geo.R",gse,out],cwd=root,stdout=log,stderr=err)
    log.write("exit="+str(p.returncode)+"\n"); log.flush()
log.write("DONE\n"); log.close(); err.close()
