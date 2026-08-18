import urllib.request, sys, time
url = "https://osf.io/download/temyk/"
dst = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis\work\pkg\GDSC2_DataFiles.zip"
tmp = dst + ".part"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
import os
resume = os.path.exists(tmp) and os.path.getsize(tmp) > 0
if resume:
    req.add_header("Range", f"bytes={os.path.getsize(tmp)}-")
t0 = time.time()
last = [0.0]
def report(block, bs, total):
    now = time.time()
    if now - last[0] > 5 or block * bs >= total:
        last[0] = now
        done = block * bs
        speed = done / max(now - t0, 1e-9) / 1024
        print(f"{done/1024/1024:.1f} MB / {total/1024/1024:.1f} MB @ {speed:.0f} KB/s", flush=True)
try:
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "ab") as f:
        if not resume:
            total = int(resp.headers.get("Content-Length", 0))
        else:
            total = os.path.getsize(tmp) + int(resp.headers.get("Content-Length", 0))
        count = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            count += len(chunk)
            now = time.time()
            if now - last[0] > 5:
                last[0] = now
                done = os.path.getsize(tmp)
                speed = done / max(now - t0, 1e-9) / 1024
                print(f"{done/1024/1024:.1f} MB / {total/1024/1024:.1f} MB @ {speed:.0f} KB/s", flush=True)
    os.replace(tmp, dst)
    print(f"DONE {os.path.getsize(dst)} bytes", flush=True)
except Exception as e:
    print(f"FAIL {e}", flush=True)
    sys.exit(1)
