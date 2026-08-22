# -*- coding: utf-8 -*-
import urllib.request, re, time
IDS = ["1609.02907", "1710.10903", "1912.01703", "1201.0490", "2604.24371"]
for aid in IDS:
    try:
        req = urllib.request.Request("https://arxiv.org/abs/" + aid, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        with urllib.request.urlopen(req, timeout=40) as r:
            txt = r.read().decode("utf-8", errors="replace")
        t = re.search(r'<title>(.*?)</title>', txt, re.S)
        title = re.sub(r"\s+", " ", t.group(1)).strip() if t else "?"
        y = re.search(r'\[Submitted on\s+([^\]]*)\]', txt)
        print("OK ", aid, "|", (y.group(1).strip()[:30] if y else "?"), "|", title[:100])
    except Exception as e:
        print("FAIL", aid, str(e)[:80])
    time.sleep(3)
