
import gzip, os, time
import pandas as pd
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
maf = os.path.join(root, "data", "raw", "mc3.v0.2.8.PUBLIC.maf.gz")
outd = os.path.join(root, "data", "processed", "rewiring")
os.makedirs(outd, exist_ok=True)
outp = os.path.join(outd, "tmb_by_sample.csv")
NONSYN = {"Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
          "Frame_Shift_Del", "Frame_Shift_Ins", "In_Frame_Del", "In_Frame_Ins",
          "Splice_Site", "Translation_Start_Site"}
counts = {}
t0 = time.time()
with gzip.open(maf, "rt", errors="replace") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    ci = hdr.index("Tumor_Sample_Barcode")
    cv = hdr.index("Variant_Classification")
    n = 0
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) <= max(ci, cv):
            continue
        sid = p[ci][:15]
        vc = p[cv]
        c = counts.setdefault(sid, [0, 0])
        c[1] += 1
        if vc in NONSYN:
            c[0] += 1
        n += 1
        if n % 1000000 == 0:
            print(n, "rows", round(time.time() - t0), "s", flush=True)
df = pd.DataFrame([(k, v[0], v[1]) for k, v in counts.items()],
                  columns=["sample_id", "tmb_nonsyn", "total_coding"])
df.to_csv(outp, index=False)
print("DONE", df.shape, "->", outp, flush=True)
