
import subprocess, os
out = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis\data\raw\gencode.v23.annotation.gtf.gz"
logf = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis\work\dl_gencode.log"
url = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_23/gencode.v23.annotation.gtf.gz"
with open(logf, "w") as f:
    f.write("start\n"); f.flush()
    for a in range(5):
        p = subprocess.run(["curl.exe","-L","-s","--ssl-no-revoke","--retry","5","--retry-delay","5","--connect-timeout","30","-o",out,url], capture_output=True)
        sz = os.path.getsize(out) if os.path.exists(out) else 0
        f.write(f"attempt {a}: rc={p.returncode} size={sz}\n"); f.flush()
        if p.returncode == 0 and sz > 30000000:
            f.write("DONE\n"); f.flush(); break
        os.remove(out) if os.path.exists(out) else None
