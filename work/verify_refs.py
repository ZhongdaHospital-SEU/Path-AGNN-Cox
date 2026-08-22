# -*- coding: utf-8 -*-
"""Verify candidate DOIs via CrossRef; print verified metadata."""
import json, urllib.request, urllib.parse, time

CANDIDATES = [
    # (key, doi or None, title search query)
    ("cox1972", "10.1111/j.2517-6161.1972.tb00899.x", None),
    ("glmnet2010", "10.18637/jss.v033.i01", None),
    ("glmnet_cox2011", "10.18637/jss.v039.i05", None),
    ("deepsurv2018", "10.1186/s12874-018-0482-1", None),
    ("coxnnet2018", "10.1371/journal.pcbi.1006076", None),
    ("deephit2018", "10.1609/aaai.v32i1.11842", None),
    ("rsf2008", "10.1214/08-AOAS169", None),
    ("gcn2017", None, "semi-supervised classification with graph convolutional networks"),
    ("gat2018", None, "graph attention networks"),
    ("tcga_pancan2013", "10.1038/ng.2764", None),
    ("xena2020", "10.1038/s41587-020-0546-8", None),
    ("geo2002", "10.1093/nar/30.1.207", None),
    ("geo2013", "10.1093/nar/gks1193", None),
    ("kegg2023", "10.1093/nar/gkac963", None),
    ("bh1995", "10.1111/j.2517-6161.1995.tb02031.x", None),
    ("harrell1996", "10.1002/(SICI)1097-0258(19960229)15:4<361::AID-SIM168>3.0.CO;2-4", None),
    ("pencina2004", "10.1002/sim.1802", None),
    ("uno2011", "10.1002/sim.4154", None),
    ("heagerty2005", "10.1111/j.0006-341X.2005.030814.x", None),
    ("dca2006", "10.1177/0272989X06295361", None),
    ("crowson2016", "10.1177/0962280213497434", None),
    ("royston2013", "10.1002/sim.5953", None),
    ("imvigor2018", "10.1038/s41586-018-0100-z", None),
    ("cibersort2015", "10.1038/nmeth.3337", None),
    ("estimate2013", "10.1038/ncomms3612", None),
    ("purity2015", "10.1038/ncomms9971", None),
    ("tmb2017", "10.1186/s13073-017-0424-2", None),
    ("gdsc2013", "10.1093/nar/gks1111", None),
    ("oncopredict2021", "10.1093/bib/bbab260", None),
    ("limma2015", "10.1093/nar/gkv007", None),
    ("clusterprofiler2012", "10.1089/omi.2011.0118", None),
    ("gsea2005", "10.1073/pnas.0506580102", None),
    ("dingo2015", "10.1093/bioinformatics/btv452", None),
    ("delfuente2010", "10.1016/j.tig.2010.04.002", None),
    ("gill2010", "10.1186/1471-2105-11-95", None),
    ("pytorch2019", None, "PyTorch an imperative style high-performance deep learning library"),
    ("scikitlearn2011", None, "Scikit-learn machine learning in Python"),
    ("lifelines2019", "10.21105/joss.01317", None),
    ("tcga_biobiolinks2016", "10.1093/nar/gkv1507", None),
    ("coxpath2024", "10.1145/3698587.3701397", None),
    ("priorgnn2024", "10.1093/bib/bbae184", None),
    ("pathmoG2026", None, "PathMoG pathway-centric modular graph neural network multi-omics survival"),
    ("pathgnn2022", None, "pathway graph neural network prognosis cancer"),
    ("gsva2013", "10.1186/1471-2105-14-7", None),
    ("bindea2013", "10.1016/j.immuni.2013.10.003", None),
    ("therneau2000", "10.1007/978-1-4757-3294-8", None),
    ("string2023", "10.1093/nar/gkac1000", None),
    ("hallmark2015", "10.1016/j.cels.2015.12.004", None),
    ("xtile2004", "10.1158/1078-0432.CCR-04-0713", None),
    ("li2024_pathway", None, "pathway-constrained graph neural network survival"),
]

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "path-agnn-refcheck/1.0 (mailto:test@example.com)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

out = []
for key, doi, query in CANDIDATES:
    rec = {"key": key, "doi": None, "title": None, "container": None, "year": None, "authors": [], "vol": None, "issue": None, "page": None}
    try:
        if doi:
            data = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))
            m = data["message"]
        elif query:
            data = get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(query))
            m = data["message"]["items"][0]
        else:
            continue
        rec["doi"] = m.get("DOI")
        rec["title"] = (m.get("title") or [""])[0]
        rec["container"] = (m.get("container-title") or [""])[0]
        rec["year"] = (m.get("issued", {}).get("date-parts", [[None]])[0][0])
        for a in m.get("author", [])[:12]:
            rec["authors"].append((a.get("family", ""), a.get("given", "")))
        rec["vol"] = m.get("volume")
        rec["issue"] = m.get("issue")
        rec["page"] = m.get("page")
        out.append(rec)
        print("OK  %-20s %s | %s | %s | %s" % (key, rec["doi"], rec["year"], (rec["title"] or "")[:60], rec["container"][:40]))
    except Exception as e:
        print("FAIL %-20s %s | %s" % (key, doi or query, str(e)[:80]))
    time.sleep(0.3)

json.dump(out, open("work/refs_verified.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("TOTAL OK:", len(out))
