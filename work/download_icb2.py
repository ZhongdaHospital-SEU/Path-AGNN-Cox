# -*- coding: utf-8 -*-
"""Download ICB cohort expression + series matrices with gzip validation."""
import gzip, subprocess, time
from pathlib import Path

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
FILES = {
    "GSE91061_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE91nnn/GSE91061/matrix/GSE91061_series_matrix.txt.gz",
    "GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE91nnn/GSE91061/suppl/GSE91061_BMS038109Sample.hg19KnownGene.fpkm.csv.gz",
    "GSE78220_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/matrix/GSE78220_series_matrix.txt.gz",
    "GSE78220_PatientFPKM.xlsx": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE78nnn/GSE78220/suppl/GSE78220_PatientFPKM.xlsx",
    "GSE100797_series_matrix.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100797/matrix/GSE100797_series_matrix.txt.gz",
    "GSE100797_ProcessedData.txt.gz": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100797/suppl/GSE100797_ProcessedData.txt.gz",
}
LOG = ROOT / "work" / "icb_dl2.log"

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
            return total > 50_000
        except Exception:
            return False
    return True

def download_one(fname, url, min_size):
    out = ROOT / "data" / "raw" / "ICB" / fname
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 6):
        if good(out, min_size):
            log(f"OK   {fname} size={out.stat().st_size}")
            return True
        log(f"DL   {fname} attempt {attempt}")
        r = subprocess.run(
            ["curl.exe", "-L", "-C", "-", "--ssl-no-revoke", "--retry", "5",
             "--retry-delay", "5", "--retry-all-errors", "--connect-timeout", "30",
             "-o", str(out), url],
            capture_output=True, text=True)
        if good(out, min_size):
            log(f"OK   {fname} size={out.stat().st_size}")
            return True
        log(f"BAD  {fname} attempt {attempt} rc={r.returncode} size={(out.stat().st_size if out.exists() else 0)}")
        try:
            out.unlink()
        except OSError:
            pass
        time.sleep(2)
    log(f"FAIL {fname}")
    return False

if __name__ == "__main__":
    ok = True
    for fname, url in FILES.items():
        min_size = 3_000_000 if "fpkm" in fname else (6_000_000 if fname.endswith("xlsx") else (1_000_000 if "ProcessedData" in fname else 5_000))
        ok &= download_one(fname, url, min_size)
    log("ALL DONE")
