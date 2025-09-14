from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss

def ece(probs, y, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins+1)
    idx = np.digitize(probs, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = idx==b
        if mask.sum()==0: continue
        conf = probs[mask].mean()
        acc  = y[mask].mean()
        ece += (mask.sum()/len(y)) * abs(acc - conf)
    return ece

def reliability_plot(ax, y_true, y_prob, n_bins=10, label="model"):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    ax.plot([0,1],[0,1], "--", linewidth=1)
    ax.plot(mean_pred, frac_pos, marker="o", label=f"{label}")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False)

def prob_scores(y_true, y_prob):
    return {
        "logloss": float(log_loss(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": float(ece(y_prob, y_true))
    }
