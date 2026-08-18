import gzip, sys

p = sys.argv[1]
try:
    with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
        d = f.read()
    lines = d.splitlines()
    if len(d) < 500:
        print("EMPTY")
        sys.exit(0)
    hdr = lines[0].split("	")
    ok = "sample" in hdr and "OS.time" in hdr and len(lines) > 100
    print("OK" if ok else "BADFMT")
except Exception:
    print("GZERR")
