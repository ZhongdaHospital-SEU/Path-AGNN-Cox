"""Random Survival Forest (scikit-survival)."""
from __future__ import annotations
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from path_agnn_cox.data import make_structured_survival


class RSFModel:
    def __init__(self, n_estimators: int = 500, min_samples_leaf: int = 15,
                 max_features: str = "sqrt", random_state: int = 0):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features=max_features, random_state=random_state, n_jobs=-1)

    def fit(self, X: np.ndarray, time: np.ndarray, event: np.ndarray):
        y = make_structured_survival(time, event)
        self.model.fit(X, y)
        return self

    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        # negative survival function at last time point ~ higher risk
        surv = self.model.predict_survival_function(np.asarray(X, dtype=float), return_array=True)
        return -surv[:, -1]
