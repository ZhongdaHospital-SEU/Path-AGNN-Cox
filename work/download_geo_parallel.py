# Parallel GEO series-matrix downloader with per-file gzip validation & resume.
import concurrent.futures as cf
import gzip, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
GSES = ['GSE37745', 'GSE4573', 'GSE20685', 'GSE21653', 'GSE7390', 'GSE14333', 'GSE17536', 'GSE39582', 'GSE62254', 'GSE26942', 'GSE15459', 'GSE14520', 'GSE116174', 'GSE22541', 'GSE29609', 'GSE65858', 'GSE41613', 'GSE32894', 'GSE13507', 'GSE9891', 'GSE32062', 'GSE26712', 'GSE16011']
MAX_WORKERS = 6
LOG = ROOT / "work" / "geo_dl3.log"

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def gzip_ok(path: Path) -> bool:
    """Validate by fully decompressing; truncated gzips raise EOFError."""
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
        return total > 1_000_000  # series-matrix files decompress to >1 MB
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
            log(f"OK   {g} size={out.stat().st_size} (already complete)")
            return True
        log(f"DL   {g} attempt {attempt} -> {url}")
        r = subprocess.run(
            ["curl.exe", "-L", "-C", "-", "--ssl-no-revoke", "--retry", "5",
             "--retry-delay", "5", "--retry-all-errors", "--connect-timeout", "30",
             "-o", str(out), url],
            capture_output=True, text=True)
        if gzip_ok(out):
            log(f"OK   {g} size={out.stat().st_size}")
            return True
        log(f"BAD  {g} attempt {attempt} rc={r.returncode} size={(out.stat().st_size if out.exists() else 0)} -> retry")
        try:
            out.unlink()
        except OSError:
            pass
        time.sleep(3)
    log(f"FAIL {g} after 5 attempts")
    return False

def main():
    log(f"== parallel GEO download start: {len(GSES)} series, workers={MAX_WORKERS} ==")
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(download_one, g): g for g in GSES}
        for fut in cf.as_completed(futs):
            g = futs[fut]
            try:
                if fut.result():
                    ok += 1
            except Exception as e:
                log(f"EXC  {g}: {e}")
    log(f"== DONE {ok}/{len(GSES)} complete ==")

if __name__ == "__main__":
    main()
