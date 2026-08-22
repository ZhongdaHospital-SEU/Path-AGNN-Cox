# -*- coding: utf-8 -*-
"""Download ICB immune-checkpoint cohorts (series matrices) with gzip validation & resume."""
import concurrent.futures as cf
import gzip, subprocess, time
from pathlib import Path

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GSES = ["GSE91061", "GSE78220", "GSE100797", "GSE135222", "GSE120795"]
MAX_WORKERS = 3
LOG = ROOT / "work" / "icb_dl.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def gzip_ok(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size < 100_000:
            return False
        total = 0
        with gzip.open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
        return total > 500_000
    except Exception:
        return False

def download_one(g):
    d = ROOT / "data" / "raw" / g
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{g}_series_matrix.txt.gz"
    digits = g[3:]
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/GSE{digits[:-3]}nnn/{g}/matrix/{g}_series_matrix.txt.gz"
    for attempt in range(1, 6):
        if gzip_ok(out):
            log(f"OK   {g} size={out.stat().st_size}")
            return True
        log(f"DL   {g} attempt {attempt}")
        r = subprocess.run(
            ["curl.exe", "-L", "-C", "-", "--ssl-no-revoke", "--retry", "5",
             "--retry-delay", "5", "--retry-all-errors", "--connect-timeout", "30",
             "-o", str(out), url],
            capture_output=True, text=True)
        if gzip_ok(out):
            log(f"OK   {g} size={out.stat().st_size}")
            return True
        log(f"BAD  {g} attempt {attempt} rc={r.returncode} size={(out.stat().st_size if out.exists() else 0)}")
        try:
            out.unlink()
        except OSError:
            pass
        time.sleep(3)
    log(f"FAIL {g} after 5 attempts")
    return False

if __name__ == "__main__":
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = list(ex.map(download_one, GSES))
    for g, ok in zip(GSES, results):
        log(f"RESULT {g}: {'OK' if ok else 'FAIL'}")
