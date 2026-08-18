
import subprocess, os
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
jobs = [("GSE62254_ACRG_Clinical_Data.txt.gz","data/raw/GSE62254/GSE62254_ACRG_Clinical_Data.txt.gz","https://ftp.ncbi.nlm.nih.gov/geo/series/GSE62nnn/GSE62254/suppl/GSE62254_ACRG_Clinical_Data.txt.gz"),("GSE16011_clinical_data_GBM.txt.gz","data/raw/GSE16011/GSE16011_clinical_data_GBM.txt.gz","https://ftp.ncbi.nlm.nih.gov/geo/series/GSE16nnn/GSE16011/suppl/GSE16011_clinical_data_GBM.txt.gz"),("GPL8542.annot.gz","data/raw/GPL/GPL8542.annot.gz","https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL85nnn/GPL8542/annot/GPL8542.annot.gz"),("GSE40435_series_matrix.txt.gz","data/raw/GSE40435/GSE40435_series_matrix.txt.gz","https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40435/matrix/GSE40435_series_matrix.txt.gz")]
log = open(os.path.join(root,"work","dl_batchB.log"),"w",encoding="utf-8")
for name, rel, url in jobs:
    outp = os.path.join(root, rel)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    for a in range(4):
        p = subprocess.run(["curl.exe","-L","-s","--ssl-no-revoke","--retry","8","--retry-delay","5","--connect-timeout","30","-o",outp,url], capture_output=True)
        sz = os.path.getsize(outp) if os.path.exists(outp) else 0
        log.write(f"{name} attempt {a}: rc={p.returncode} size={sz}\n"); log.flush()
        if p.returncode == 0 and sz > 10000 and not open(outp,"rb").read(200).startswith(b"<"):
            log.write(f"{name} DONE\n"); log.flush()
            break
        if os.path.exists(outp): os.remove(outp)
log.write("ALL_DONE\n"); log.close()
