# -*- coding: utf-8 -*-
import json, urllib.request, urllib.parse, time

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "path-agnn-refcheck/1.0 (mailto:test@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

SEARCHES = [
    ("imvigor2018", "TGFbeta attenuates tumour response to PD-L1 blockade Mariathasan"),
    ("royston2013", "External validation of a Cox prognostic model lessons prospective cohort"),
    ("dingo2015", "DINGO differential network analysis genomics"),
    ("delfuente2010", "differential expression differential networking dysfunctional regulatory networks diseases"),
    ("pathgnn2022", "PathGNN pathway graph neural network cancer survival"),
    ("li2024_pathway", "pathway graph neural network survival prediction KEGG"),
    ("metabric", "The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups"),
]
for key, q in SEARCHES:
    try:
        data = get("https://api.crossref.org/works?rows=4&query.bibliographic=" + urllib.parse.quote(q))
        for i, m in enumerate(data["message"]["items"]):
            print(">>", key, i, m.get("DOI"), (m.get("title") or [""])[0][:80], (m.get("container-title") or [""])[0][:40], (m.get("issued",{}).get("date-parts",[[None]])[0][0]))
    except Exception as e:
        print("ERR", key, e)
    time.sleep(1)
