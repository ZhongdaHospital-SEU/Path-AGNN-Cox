# -*- coding: utf-8 -*-
"""KIRC pure-null matched-control simulation (fills Table 6 n.a. for KIRC)."""
import os, sys
sys.path.insert(0, r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis")
os.environ.setdefault("OMP_NUM_THREADS", "2")
from work.sim_matched_control import run_ds, SEED
if __name__ == "__main__":
    import pandas as pd
    summ = run_ds("KIRC", SEED + 2)
    print(summ.to_string())
    print("MC_NULL_KIRC_DONE")
