# -*- coding: utf-8 -*-
"""Serial download with speed-limit protection: ICB cohorts then TCGA-SKCM."""
import gzip, subprocess, time
from pathlib import Path

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
LOG = ROOT / "work" / "dl_serial.log"
FILES = [
    ("data/raw/ICB/GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE91nnn/GSE91061/suppl/GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz", 3_000_000),
    ("data/raw/ICB/GSE78220_series_matrix.txt.gz",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/matrix/GSE78220_series_matrix.txt.gz", 2_000),
    ("data/raw/ICB/GSE78220_PatientFPKM.xlsx",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/suppl/GSE78220_PatientFPKM.xlsx", 6_000_000),
    ("data/raw/ICB/GSE100797_series_matrix.txt.gz",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100797/matrix/GSE100797_series_matrix.txt.gz", 2_000),
    ("data/raw/ICB/GSE100797_ProcessedData.txt.gz",
     "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100797/suppl/GSE100797_ProcessedData.txt.gz", 1_000_000),
    ("data/raw/TCGA-xena/TCGA-SKCM.star_tpm.tsv.gz",
     "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.star_tpm.tsv.gz", 10_000_000),
    ("data/raw/TCGA-xena/TCGA-SKCM.survival.tsv.gz",
     "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-SKCM.survival.tsv.gz", 2_000),
]

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def good(path: Path, min_size: int) -> bool:
    if not path.exists() or path.stat().st_size < min_size:
        return False
    if path.name.endswith(".gz"):
        try:
            total = 0
            with gzip.open(path, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    total += len(chunk)
            return total > 20_000
        except Exception:
            return False
    return True

def download_one(rel, url, min_size):
    out = ROOT / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 8):
        if good(out, min_size):
            log(f"OK   {rel} size={out.stat().st_size}")
            return True
        log(f"DL   {rel} attempt {attempt}")
        r = subprocess.run(
            ["curl.exe", "-L", "-C", "-", "--ssl-no-revoke", "--retry", "5",
             "--retry-delay", "5", "--retry-all-errors", "--connect-timeout", "30",
             "--speed-limit", "1024", "--speed-time", "120",
             "-o", str(out), url],
            capture_output=True, text=True)
        if good(out, min_size):
            log(f"OK   {rel} size={out.stat().st_size}")
            return True
        sz = out.stat().st_size if out.exists() else 0
        log(f"BAD  {rel} attempt {attempt} rc={r.returncode} size={sz}")
        try:
            out.unlink()
        except OSError:
            pass
        time.sleep(2)
    log(f"FAIL {rel}")
    return False

if __name__ == "__main__":
    for rel, url, ms in FILES:
        download_one(rel, url, ms)
    log("ALL_DONE")
