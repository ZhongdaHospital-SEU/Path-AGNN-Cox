# -*- coding: utf-8 -*-
"""Aggregate the P0 addendum results into one summary for the manuscript."""
import pandas as pd
from scipy.stats import spearmanr
RW = r"results/rewiring"

gv = pd.read_csv(f"{RW}/global_vs_specific_summary.csv")
ov = pd.read_csv(f"{RW}/cross_cohort_overlap.csv")
an = pd.read_csv(f"{RW}/anchor_rewiring_summary.csv")

tabs = {d: pd.read_csv(f"{RW}/{d}/pathway_specific_decomposition.csv").set_index("pathway")
        for d in ["LUAD", "BRCA", "KIRC"]}
b = tabs["BRCA"].rename(columns={"d_adj": "dB", "q_adj": "qB"})[["dB", "qB"]]
k = tabs["KIRC"].rename(columns={"d_adj": "dK", "q_adj": "qK"})[["dK", "qK"]]
both = b.join(k)
sigB = set(both.index[both["qB"] < 0.05]); sigK = set(both.index[both["qK"] < 0.05])
shared = both.loc[sorted(sigB & sigK)]
concord = float(((shared["dB"] > 0) == (shared["dK"] > 0)).mean())
r_d, p_d = spearmanr(shared["dB"], shared["dK"])
d_adj_bk = {"n_shared_sig": len(shared), "concordance": concord,
            "spearman_d_adj": r_d, "spearman_p": p_d,
            "spearman_d_adj_all": float(spearmanr(both["dB"], both["dK"])[0])}

rows = {"global_shift": {}, "specific": {}, "anchors": {}, "direction": d_adj_bk}
for _, r in gv.iterrows():
    rows["global_shift"][r["dataset"]] = {"d": r["global_d"], "p": r["global_p"]}
    rows["specific"][r["dataset"]] = {"n_sig": int(r["n_sig_specific_q005"]),
                                      "n_pos": int(r["n_direction_pos"]),
                                      "n_neg": int(r["n_direction_neg"])}
for _, r in an.iterrows():
    rows["anchors"].setdefault(r["dataset"], {})[r["anchor"]] = {
        "n_sig": None if pd.isna(r["n_sig_q005"]) else int(r["n_sig_q005"]),
        "n_hi": int(r["n_hi"]), "n_lo": int(r["n_lo"])}
pd.to_pickle(rows, f"{RW}/addendum_summary.pkl")

# readable CSV for the manuscript table
flat = []
for d in ["LUAD", "BRCA", "KIRC"]:
    g = rows["global_shift"][d]; s = rows["specific"][d]
    a = rows["anchors"].get(d, {})
    flat.append({"dataset": d,
                 "global_d": g["d"], "global_p": g["p"],
                 "n_sig_raw_perm": {"LUAD": 2, "BRCA": 43, "KIRC": 52}[d],
                 "n_sig_specific_q005": s["n_sig"],
                 "n_sig_anchor_mki67": a.get("mki67", {}).get("n_sig"),
                 "n_sig_anchor_tmb": a.get("tmb", {}).get("n_sig"),
                 "n_sig_anchor_stage": a.get("stage", {}).get("n_sig")})
pd.DataFrame(flat).to_csv(f"{RW}/addendum_table.csv", index=False)
print(pd.DataFrame(flat).to_string(index=False))
print("BK direction:", d_adj_bk)
print("ADDENDUM_DONE")
