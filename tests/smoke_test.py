"""Smoke test: run every baseline + Path-AGNN-Cox on synthetic data."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.cox_penalized import lasso_cox, ridge_cox, elastic_net_cox
from baselines.rsf import RSFModel
from baselines.deepsurv import deepsurv, cox_nnet
from baselines.plain_gnn import plain_gnn_survival
from path_agnn_cox.evaluate import c_index, time_dependent_auc, calibration_slope

rng = np.random.default_rng(1)
n, p = 200, 40
X = rng.normal(0, 1, (n, p))
z = X[:, :10].mean(1)
time = np.exp(2.0 - 0.6 * z + rng.normal(0, 0.5, n))
event = (rng.random(n) < 0.4).astype(int)

def check(name, fn):
    risk = fn()
    ci = c_index(risk, time, event)
    print(f"{name:14s} C-index={ci:.3f}")
    assert ci > 0.5, name

check("lasso_cox", lambda: lasso_cox(X, time, event).predict_risk(X))
check("ridge_cox", lambda: ridge_cox(X, time, event).predict_risk(X))
check("elastic_net", lambda: elastic_net_cox(X, time, event).predict_risk(X))
check("rsf", lambda: RSFModel(n_estimators=100).fit(X, time, event).predict_risk(X))
check("deepsurv", lambda: deepsurv(X, time, event, epochs=60).predict_risk(X))
check("cox_nnet", lambda: cox_nnet(X, time, event, epochs=60).predict_risk(X))
check("plain_gnn", lambda: plain_gnn_survival(
    X, time, event, epochs=40, patience=5).predict_risk(X))

auc = time_dependent_auc(X[:, 0], time, event)
cal = calibration_slope(X[:, 0], time, event)
print("td-AUC:", auc.round(3).to_dict("records"))
print("calibration:", cal)
print("SMOKE TEST PASSED")