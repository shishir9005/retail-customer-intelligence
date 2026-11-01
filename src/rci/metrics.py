"""Evaluation metrics, written to answer a marketing question: "if I can only contact some customers,
how good is the list?" Plus a bootstrap to say how much to trust each number.
"""
from __future__ import annotations

from math import ceil

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def top_fraction_capture(y: np.ndarray, score: np.ndarray, fraction: float = 0.20) -> float:
    """Share of all real repurchasers found inside the top `fraction` of customers by score.

    Contact the top 20% of the list: what % of the people who would have bought again are in it?
    A random list captures 20%. A perfect list captures 100% (when repurchasers are <= 20%).
    """
    y, score = np.asarray(y), np.asarray(score)
    n_top = max(1, ceil(fraction * len(y)))
    order = np.argsort(-score, kind="stable")
    return float(y[order[:n_top]].sum() / max(1, y.sum()))


def top_fraction_precision(y: np.ndarray, score: np.ndarray, fraction: float = 0.20) -> float:
    """Of the top `fraction` of customers by score, what share really buy again?
    Compare with the overall repurchase rate: that gap is what the model adds."""
    y, score = np.asarray(y), np.asarray(score)
    n_top = max(1, ceil(fraction * len(y)))
    order = np.argsort(-score, kind="stable")
    return float(y[order[:n_top]].mean())


METRIC_KEYS = ("roc_auc", "pr_auc", "top20_capture", "top20_precision")


def gain_curve(y: np.ndarray, score: np.ndarray, points: int = 51) -> list[list[float]]:
    """[(fraction of customers contacted, fraction of repurchasers captured), ...] for plotting."""
    fractions = np.linspace(0, 1, points)
    return [[float(f), 0.0 if f == 0 else top_fraction_capture(y, score, float(f))] for f in fractions]


def point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "top20_capture": top_fraction_capture(y, score, 0.20),
        "top20_precision": top_fraction_precision(y, score, 0.20),
    }


def bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], baseline: str, n_boot: int = 1000, seed: int = 42):
    """Re-draw the test customers with replacement `n_boot` times and recompute every metric.

    The spread of those recomputed numbers is the uncertainty from having only this many customers.
    All models see the SAME resamples, so differences between models are paired and fair.
    Returns {model: {metric: (low, high)}} and {model: (low, high)} for ROC-AUC minus the baseline's.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = {m: {k: [] for k in METRIC_KEYS} for m in scores}
    diffs = {m: [] for m in scores if m != baseline}
    while min(len(v["roc_auc"]) for v in draws.values()) < n_boot:
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():  # need both classes to compute AUC
            continue
        aucs = {}
        for m, s in scores.items():
            r = point_metrics(y[idx], np.asarray(s)[idx])
            for k, v in r.items():
                draws[m][k].append(v)
            aucs[m] = r["roc_auc"]
        for m in diffs:
            diffs[m].append(aucs[m] - aucs[baseline])
    ci = {m: {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in d.items()} for m, d in draws.items()}
    diff_ci = {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in diffs.items()}
    return ci, diff_ci


def roc_points(y, score, grid: int = 101) -> list[list[float]]:
    fpr, tpr, _ = roc_curve(y, score)
    xs = np.linspace(0, 1, grid)
    return [[float(x), float(t)] for x, t in zip(xs, np.interp(xs, fpr, tpr))]


def pr_points(y, score, grid: int = 101) -> list[list[float]]:
    precision, recall, _ = precision_recall_curve(y, score)
    order = np.argsort(recall)
    xs = np.linspace(0, 1, grid)
    return [[float(x), float(p)] for x, p in zip(xs, np.interp(xs, recall[order], precision[order]))]
