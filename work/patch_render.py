# -*- coding: utf-8 -*-
import io
p = "manuscript/render_manuscript.py"
s = io.open(p, encoding="utf-8").read()

# 1) fix P<1e-9 in sensitivity sentence
old = '"the risk-score association in IMvigor210 (rho = %s-%s, all P<1e-9)"'
new = '"the risk-score association in IMvigor210 (rho = %s-%s, all P<0.001)"'
assert old in s
s = s.replace(old, new)

# 2) add table_hyper + table_sens before def main():
anchor = "def main():"
add = '''def table_hyper() -> str:
    """Model hyperparameters and baseline configurations."""
    rows = [
        ("Model", "Configuration"),
        ("Path-AGNN-Cox", "hidden 32, layers 2, mlp 32, dropout 0.1, epochs 100, lr 0.001, batch 128, patience 15, L2 0.0001, lambda_sparse 0.001, lambda_consist 0.1"),
        ("\\u2212Adaptive (static)", "same backbone with fixed normalized adjacency"),
        ("\\u2212Regularization", "lambda_sparse = 0, lambda_consist = 0"),
        ("Plain GNN", "identity adjacency, global pooling"),
        ("LASSO-Cox", "penalizer 0.05, 10-fold internal CV"),
        ("Ridge-Cox", "penalizer 0.1"),
        ("Elastic-Net-Cox", "l1_ratio 0.5, penalizer 0.1"),
        ("Random Survival Forest", "500 trees, min_samples_leaf 15"),
        ("DeepSurv", "hidden [32, 16]"),
        ("Cox-nnet", "hidden [64], dropout 0.0"),
    ]
    return "\\n".join("| " + " | ".join(r) + " |" for r in rows)


def table_sens() -> str:
    """Sensitivity of clinical anchors to the rewiring-magnitude definition."""
    p = ROOT / "results" / "rewiring" / "sensitivity_magnitude.csv"
    if not p.exists():
        return "| Cohort | Definition | rho(risk) | P(risk) | rho(Ki-67) | P(Ki-67) | n(Ki-67) | rho(TMB) | P(TMB) | n(TMB) |\\n|---|---|---|---|---|---|---|---|---|---|"
    t = pd.read_csv(p)
    lines = ["| Cohort | Definition | rho(risk) | P(risk) | rho(Ki-67) | P(Ki-67) | n(Ki-67) | rho(TMB) | P(TMB) | n(TMB) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in ("LUAD", "BRCA", "IMvigor210"):
        sub = t[t["dataset"] == ds].sort_values("definition")
        for _, r in sub.iterrows():
            def f(v, nd=2):
                try:
                    x = float(v)
                    return fmt2(x) if nd == 2 else fmt_p(x)
                except (TypeError, ValueError):
                    return "\\u2014"
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                ds, r["definition"], f(r["rho_risk"]), f(r["P_risk"], 0), f(r["rho_ki67"]),
                f(r["P_ki67"], 0), f(r["n_ki67"]), f(r["rho_tmb"]), f(r["P_tmb"], 0), f(r["n_tmb"])))
    return "\\n".join(lines)


def main():'''
assert anchor in s
s = s.replace(anchor, add, 1)

# 3) register new tables
old = '''        "CALIBRATION": table_cal(),
    }'''
new = '''        "CALIBRATION": table_cal(),
        "HYPERPARAM": table_hyper(),
        "SENSITIVITY": table_sens(),
    }'''
assert old in s
s = s.replace(old, new, 1)

io.open(p, "w", encoding="utf-8", newline="\n").write(s)
print("render_manuscript.py patched")
