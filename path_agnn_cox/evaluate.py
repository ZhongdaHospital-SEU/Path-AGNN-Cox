"""Evaluation metrics: Harrell C-index, time-dependent AUC, calibration.

All metrics are computed from predicted risk scores (higher = worse prognosis)
plus observed (time, event) pairs. time_dependent_auc requires scikit-survival.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scipy import stats
from .data import make_structured_survival


def c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell's concordance index (higher = better).

    Convention: higher risk score = worse prognosis (sksurv convention,
    matching the paper's risk-score semantics).
    """
    from sksurv.metrics import concordance_index_censored
    c, _, _, _, _ = concordance_index_censored(event.astype(bool), time, risk)
    return float(c)


def time_dependent_auc(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                       times: list[float] | np.ndarray | None = None
                       ) -> pd.DataFrame:
    """Cumulative dynamic AUC at evaluation times (requires scikit-survival).

    Returns a DataFrame with columns [time, auc].
    """
    from sksurv.metrics import cumulative_dynamic_auc
    y = make_structured_survival(time, event)
    if times is None:
        q = np.percentile(time[event > 0], [25, 50, 75])
        times = np.unique(q)
    auc, _ = cumulative_dynamic_auc(y, y, risk, times)
    return pd.DataFrame({"time": times, "auc": auc})


def calibration_slope(risk: np.ndarray, time: np.ndarray, event: np.ndarray
                      ) -> dict:
    """Calibration slope: Cox regression of observed survival on risk score.

    slope ~ 1 indicates good calibration; < 1 overfitting, > 1 underfitting.
    """
    try:
        from lifelines import CoxPHFitter
    except Exception:
        return {"slope": np.nan, "p": np.nan}
    if np.std(risk) < 1e-12 or len(np.unique(time)) < 3:
        return {"slope": np.nan, "p": np.nan}
    try:
        df = pd.DataFrame({"T": time, "E": event.astype(bool), "risk": risk})
        cph = CoxPHFitter().fit(df, duration_col="T", event_col="E")
        p = cph.params_["risk"]
        ci = cph.confidence_intervals_.loc["risk"]
        se = (ci.iloc[1] - ci.iloc[0]) / 1.96
        return {"slope": float(p), "se": float(se)}
    except Exception:
        return {"slope": np.nan, "p": np.nan}


def full_report(risk: np.ndarray, time: np.ndarray, event: np.ndarray,
                times: list[float] | None = None) -> dict:
    """One-stop evaluation summary."""
    report = {"c_index": c_index(risk, time, event)}
    try:
        auc_df = time_dependent_auc(risk, time, event, times)
        report["auc_mean"] = float(auc_df["auc"].mean())
        report["auc_times"] = auc_df
    except Exception:
        report["auc_mean"] = np.nan
        report["auc_times"] = None
    report.update(calibration_slope(risk, time, event))
    return report

