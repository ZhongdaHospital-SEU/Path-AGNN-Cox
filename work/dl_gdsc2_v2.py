import urllib.request, os, sys, time
url = "https://osf.io/download/temyk/"
dst = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis\work\pkg\GDSC2_DataFiles.zip"
tmp = dst + ".part"
for attempt in range(1, 7):
    try:
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have > 0:
            req.add_header("Range", "bytes=%d-" % have)
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "ab") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        total = os.path.getsize(tmp)
        print("attempt %d DONE %d bytes" % (attempt, total), flush=True)
        if total > 60000000:
            os.replace(tmp, dst)
            print("FINAL DONE", flush=True)
            break
    except Exception as e:
        print("attempt %d fail: %s" % (attempt, e), flush=True)
        time.sleep(10)
else:
    print("GAVE_UP", flush=True)
    sys.exit(1)
