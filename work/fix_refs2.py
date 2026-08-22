# -*- coding: utf-8 -*-
import json, urllib.request, urllib.parse, time, re

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "path-agnn-refcheck/1.0 (mailto:test@example.com)"})
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()

DOIS = [
    ("imvigor2018", "10.1038/nature25501"),
    ("bmj_validating2009", "10.1136/bmj.b605"),
    ("eraslan2019", "10.1038/s41576-019-0122-6"),
    ("wainberg2018", "10.1038/nbt.4233"),
    ("ching2018_interface", "10.1098/rsif.2017.0387"),
    ("miotto2018", "10.1093/bib/bbx044"),
    ("zitnik2018", "10.1093/bioinformatics/bty294"),
    ("davis_geoquery2007", "10.1093/bioinformatics/btm254"),
    ("leek2012", "10.1093/bioinformatics/bts034"),
    ("curtis_metabric2012", "10.1038/nature10983"),
    ("dingo2015", "10.1093/bioinformatics/btv406"),
    ("delfuente2010", "10.1016/j.tig.2010.05.001"),
]
for key, doi in DOIS:
    try:
        raw = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
        m = json.loads(raw)["message"]
        print("OK ", key, doi, "|", (m.get("issued",{}).get("date-parts",[[None]])[0][0]), "|", (m.get("title") or [""])[0][:70], "|", (m.get("container-title") or [""])[0][:40])
    except Exception as e:
        print("FAIL", key, doi, str(e)[:70])
    time.sleep(0.5)

print("=== Europe PMC PathGNN ===")
try:
    raw = get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:9516820%20AND%20SRC:PMC&resultType=core&format=json")
    j = json.loads(raw)
    for r in j.get("resultList", {}).get("result", [])[:2]:
        print("PMC9516820:", r.get("title"), "|", r.get("authorString", "")[:80], "|", r.get("journalTitle"), r.get("pubYear"), "| DOI:", r.get("doi"))
except Exception as e:
    print("ERR", e)

print("=== arXiv check ===")
try:
    raw = get("http://export.arxiv.org/api/query?id_list=1609.02907,1710.10903,1912.01703,1201.0490,2604.24371&max_results=10")
    txt = raw.decode("utf-8", errors="replace")
    for m in re.finditer(r"<entry>.*?</entry>", txt, re.S):
        e = m.group(0)
        aid = re.search(r"<id>http://arxiv.org/abs/([^<]+)</id>", e).group(1)
        title = re.search(r"<title>([^<]+)</title>", e).group(1).strip().replace("\n ", "")
        year = re.search(r"<published>(\d{4})", e).group(1)
        print("ARXIV", aid, "|", year, "|", title[:80])
except Exception as ex:
    print("ERR", ex)
