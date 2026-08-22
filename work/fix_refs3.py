# -*- coding: utf-8 -*-
import json, urllib.request, urllib.parse, time, re

def get(url, ua="Mozilla/5.0"):
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()

print("=== NCBI esummary PMC9516820 ===")
try:
    raw = get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pmc&id=9516820&retmode=json")
    j = json.loads(raw)
    r = j.get("result", {}).get("9516820", {})
    print("title:", r.get("title"))
    print("journal:", r.get("fulljournalname"), r.get("pubdate"))
    print("doi:", [x["value"] for x in r.get("articleids", []) if x["idtype"] == "doi"])
except Exception as e:
    print("ERR", e)

print("=== arXiv retry ===")
time.sleep(5)
try:
    raw = get("http://export.arxiv.org/api/query?id_list=1609.02907,1710.10903,1912.01703,1201.0490,2604.24371&max_results=10", ua="Mozilla/5.0 (research refcheck)")
    txt = raw.decode("utf-8", errors="replace")
    for m in re.finditer(r"<entry>.*?</entry>", txt, re.S):
        e = m.group(0)
        aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e).group(1)
        title = re.search(r"<title>([^<]+)</title>", e).group(1).strip().replace("\n ", "")
        year = re.search(r"<published>(\d{4})", e).group(1)
        print("ARXIV", aid, "|", year, "|", title[:90])
except Exception as ex:
    print("ERR", ex)
