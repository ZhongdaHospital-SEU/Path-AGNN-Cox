import subprocess, sys, time, os
RS = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
SCRIPT = "data/scripts/04_preprocess_geo.R"
LOG = "work/geo_preprocess.log"
# gse -> cancer external dir
JOBS = [
    ("GSE21653", "BRCA"), ("GSE7390", "BRCA"),
    ("GSE14333", "COAD"), ("GSE17536", "COAD"), ("GSE39582", "COAD"),
    ("GSE37745", "LUSC"),
    ("GSE41613", "HNSC"),
    ("GSE15459", "STAD"),
    ("GSE26712", "OV"),
    ("GSE29609", "KIRC"),
]
def log(msg):
    line = time.strftime("[%H:%M:%S] ") + msg
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
for gse, cancer in JOBS:
    out = os.path.join("data/processed", cancer, "external", gse + ".csv")
    if os.path.exists(out) and os.path.getsize(out) > 10_000_000:
        log(f"SKIP {gse} (exists {os.path.getsize(out)})")
        continue
    log(f"RUN  {gse} -> {out}")
    r = subprocess.run([RS, SCRIPT, gse, out], capture_output=True, text=True, timeout=600)
    tail = (r.stdout + r.stderr).strip().splitlines()
    for line in tail[-3:]:
        log("   " + line)
    if r.returncode != 0:
        log(f"FAIL {gse} rc={r.returncode}")
log("GEO_PREPROCESS_ALL_DONE")
