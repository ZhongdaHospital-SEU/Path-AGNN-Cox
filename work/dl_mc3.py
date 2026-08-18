
import subprocess, os
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
outp = os.path.join(root, "data", "raw", "mc3.v0.2.8.PUBLIC.maf.gz")
logf = os.path.join(root, "work", "dl_mc3.log")
url = "https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc"
with open(logf, "w", encoding="utf-8") as f:
    f.write("start\n"); f.flush()
    for a in range(8):
        p = subprocess.run(["curl.exe","-L","-s","--ssl-no-revoke","--retry","10","--retry-delay","5","--connect-timeout","30","-o",outp,url], capture_output=True)
        sz = os.path.getsize(outp) if os.path.exists(outp) else 0
        f.write(f"attempt {a}: rc={p.returncode} size={sz}\n"); f.flush()
        if p.returncode == 0 and sz > 50000000:
            f.write("DONE\n"); f.flush(); break
        if os.path.exists(outp): os.remove(outp)
