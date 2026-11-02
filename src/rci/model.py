"""Will this customer buy again in the next 90 days?  Honest comparison of scorers.

  1. Recency rule        score = -days since last order (the simplest thing a marketer would do)
  2. Recent-orders rule  orders in the last 90 days, a tie going to the more recent customer
  3. Logistic regression (recency + frequency)   the simplest learned baseline
  4. Logistic regression a linear baseline on all 13 features
  5. Gradient boosting   LightGBM if available, else scikit-learn's HistGradientBoosting (same idea)
  6. PyTorch MLP         a small neural network (only if torch is installed)

Splitting is by TIME, never random: train on earlier "pretend todays", tune on a later one, and
report on the latest one. See config.SPLITS and tests/test_no_leakage.py. models 1-5 are also scored on
three successive test quarters (config.BACKTEST_FOLDS); the MLP is trained for the latest quarter only.

Run:  python -m rci.model
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rci import config, db, features, metrics

RECENCY_RULE = "Recency rule"
RECENT_ORDERS_RULE = "Recent-orders rule"
LOGREG_2 = "Logistic regression (recency + frequency)"
LOGREG = "Logistic regression"
MLP_NAME = "PyTorch MLP"
RECENCY_FREQUENCY = ["recency_days", "frequency"]
BACKTEST_TITLE = "Three-quarter backtest (MLP not included)"
LATEST_QUARTER_LABEL = "latest quarter only (Sep–Dec 2011)"


# ── split safety ─────────────────────────────────────────────────────────────
def check_split_chain(splits: dict[str, list[str]], horizon: int = config.HORIZON_DAYS) -> None:
    """Raise if a split's label window overlaps the next split's period (that would leak the future).

    Order must be train < valid < test. For each consecutive pair, the LAST label window end of the
    earlier split (latest cutoff + horizon) must be on or before the FIRST cutoff of the later split.
    """
    order = ["train", "valid", "test"]
    for a, b in zip(order, order[1:]):
        last_label_end = max(pd.Timestamp(c) for c in splits[a]) + pd.Timedelta(days=horizon)
        next_start = min(pd.Timestamp(c) for c in splits[b])
        if last_label_end > next_start:
            raise ValueError(f"{a} labels run until {last_label_end.date()}, after {b} starts {next_start.date()}")


def check_labels_observable(splits: dict[str, list[str]], last_day: dt.date, horizon: int = config.HORIZON_DAYS) -> None:
    """Raise if any cutoff's label window runs past the end of the data.

    Such a snapshot has no real labels yet. Failing here, with the cutoff named, is better than
    discovering it later as NaN labels.
    """
    for split, cutoffs in splits.items():
        for c in cutoffs:
            cutoff = pd.Timestamp(c).date()
            if not features.label_is_observable(last_day, cutoff, horizon):
                end = cutoff + dt.timedelta(days=horizon)
                raise ValueError(
                    f"{split} cutoff {c}: its {horizon}-day label window ends {end}, but the data ends {last_day}. "
                    "Move the cutoff earlier."
                )


def labels_as_int(df: pd.DataFrame, split: str) -> np.ndarray:
    """The label column as 0/1 integers, refusing anything that is not an observed label.

    `Series.to_numpy(dtype=int)` turns NaN into a huge negative number with only a RuntimeWarning, so
    a snapshot with unobserved labels would train or score on garbage without any error. This check
    makes that impossible.
    """
    label = df["label"]
    if len(label) == 0:
        raise ValueError(f"{split}: the split has no rows.")
    n_nan = int(label.isna().sum())
    if n_nan:
        raise ValueError(
            f"{split}: {n_nan} of {len(label):,} labels are NaN, meaning their label window is not fully observed. "
            "Refusing to cast them to integers."
        )
    bad = label[~label.isin([0, 1])]
    if len(bad):
        raise ValueError(f"{split}: labels must be 0 or 1, found {sorted(bad.unique())[:5]}.")
    return label.to_numpy(dtype=int)


# ── models ───────────────────────────────────────────────────────────────────
def fit_logreg(X: pd.DataFrame, y: np.ndarray):
    """log(1+x) -> standardise -> logistic regression. Scaling matters for a linear model."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    pipe.fit(features.log_transform(X), y)
    return pipe


def recent_orders_score(Z: pd.DataFrame) -> np.ndarray:
    """Orders in the last 90 days, minus recency / 10,000 as a tie-break (it stays below 1 for any recency under
    10,000 days, so it can never outrank a whole order)."""
    return Z["orders_last_90d"].to_numpy() - Z["recency_days"].to_numpy() / 10_000


def fit_logreg_recency_frequency(X: pd.DataFrame, y: np.ndarray):
    """The simplest learned baseline: log(1+x) of recency and frequency only -> standardise -> logistic regression.
    Returns a function that scores a feature table (it reads no other column)."""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    log_two = lambda Z: np.log1p(Z[RECENCY_FREQUENCY].clip(lower=0))  # noqa: E731
    pipe.fit(log_two(X), y)
    return lambda Z: pipe.predict_proba(log_two(Z))[:, 1]


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except (ImportError, OSError):  # macOS without libomp raises OSError
        return False


def fit_gbm(Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray):
    """Gradient-boosted trees: many small trees, each one correcting the errors of the previous.

    The number of trees is chosen on the VALIDATION snapshot (early stopping), never on the test one.
    Returns (model, display_name).
    """
    if lightgbm_available():
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=config.SEED, verbose=-1,
        )
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        return model, "LightGBM"  # n_trees: model.best_iteration_

    params = dict(learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0,
                  early_stopping=False, random_state=config.SEED)
    probe = HistGradientBoostingClassifier(max_iter=400, **params).fit(Xtr, ytr)
    val_auc = [roc_auc_score(yva, p[:, 1]) for p in probe.staged_predict_proba(Xva)]
    best_iter = int(np.argmax(val_auc)) + 1
    model = HistGradientBoostingClassifier(max_iter=best_iter, **params).fit(Xtr, ytr)
    return model, "HistGradientBoosting"


def fit_mlp_scorer(Xtr, ytr, Xva, yva):
    """Standardise log features with TRAIN statistics only, then train the PyTorch network."""
    from rci import mlp  # imports torch; raises ImportError when torch is not installed

    scaler = StandardScaler().fit(features.log_transform(Xtr))
    prep = lambda X: scaler.transform(features.log_transform(X))  # noqa: E731
    predict, history = mlp.fit_mlp(prep(Xtr), ytr, prep(Xva), yva, seed=config.SEED)
    return (lambda X: predict(prep(X))), history


def fit_scorers(X: dict, y: dict) -> tuple[dict, object, str]:
    """The five scorers that are not neural, fitted on the train rows (boosting stops early on the valid rows).

    The main table and every backtest fold both use this, so they can never be built differently.
    Returns ({name: score function}, the boosting model, its display name).
    """
    lr = fit_logreg(X["train"], y["train"])
    gbm, gbm_name = fit_gbm(X["train"], y["train"], X["valid"], y["valid"])
    scorers = {
        RECENCY_RULE: lambda Z: -Z["recency_days"].to_numpy(),
        RECENT_ORDERS_RULE: recent_orders_score,
        LOGREG_2: fit_logreg_recency_frequency(X["train"], y["train"]),
        LOGREG: lambda Z: lr.predict_proba(features.log_transform(Z))[:, 1],
        gbm_name: lambda Z: gbm.predict_proba(Z)[:, 1],
    }
    return scorers, gbm, gbm_name


def backtest(dataset, cols: list[str], folds: dict[str, dict[str, list[str]]] = config.BACKTEST_FOLDS) -> dict:
    """ROC-AUC of every non-neural scorer on three successive test quarters, each fold trained only on earlier data.

    `dataset(cutoffs)` returns the stacked snapshots (feature columns and `label`) for those cutoffs. The MLP is not
    part of this: it is trained for the latest quarter only, so it must never be averaged or ranked against these means.
    """
    results = []
    for label, fold in folds.items():
        data = {s: dataset(cutoffs) for s, cutoffs in fold.items()}
        X = {s: d[cols] for s, d in data.items()}
        y = {s: labels_as_int(d, f"{label} {s}") for s, d in data.items()}
        scorers, _, gbm_name = fit_scorers(X, y)
        results.append({
            "label": label, "splits": fold, "test_customers": int(len(y["test"])),
            "repurchase_rate": round(float(y["test"].mean()), 4),
            "roc_auc": {m: float(roc_auc_score(y["test"], score(X["test"]))) for m, score in scorers.items()},
        })
    names = list(results[0]["roc_auc"])
    gap = [r["roc_auc"][gbm_name] - r["roc_auc"][LOGREG] for r in results]
    return {
        "title": BACKTEST_TITLE, "metric": "roc_auc", "models": names, "folds": results,
        "mean": {m: float(np.mean([r["roc_auc"][m] for r in results])) for m in names},
        "gbm_minus_logreg": {"per_fold": gap, "mean": float(np.mean(gap))},
    }


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    check_split_chain(config.SPLITS)
    con = db.connect()
    last_day = db.data_end(con)
    for fold in (config.SPLITS, *config.BACKTEST_FOLDS.values()):
        check_split_chain(fold)
        check_labels_observable(fold, last_day)
    cols = features.feature_columns(con)

    snapshots: dict[str, pd.DataFrame] = {}      # one snapshot per cutoff, shared by the main table and the backtest

    def dataset(cutoffs: list[str]) -> pd.DataFrame:
        for c in cutoffs:
            if c not in snapshots:
                snapshots[c] = features.build_snapshot(con, c)
        return pd.concat([snapshots[c] for c in cutoffs], ignore_index=True)

    data = {s: dataset(cut) for s, cut in config.SPLITS.items()}
    X = {s: d[cols] for s, d in data.items()}
    y = {s: labels_as_int(d, s) for s, d in data.items()}
    for s in data:
        print(f"{s:<6} rows={len(data[s]):>6,}  repurchase rate={y[s].mean():.3f}  cutoffs={config.SPLITS[s]}")

    scorers, gbm, gbm_name = fit_scorers(X, y)
    mlp_history = None
    try:
        mlp_predict, mlp_history = fit_mlp_scorer(X["train"], y["train"], X["valid"], y["valid"])
        scorers[MLP_NAME] = lambda Z: mlp_predict(Z)
    except ImportError:
        print("torch not installed: skipping the MLP")

    test_scores = {m: f(X["test"]) for m, f in scorers.items()}
    valid_scores = {m: f(X["valid"]) for m, f in scorers.items()}

    point = {m: metrics.point_metrics(y["test"], s) for m, s in test_scores.items()}
    ci, diff_ci = metrics.bootstrap(y["test"], test_scores, baseline=LOGREG)
    print(f"\n{'model':<24}{'ROC-AUC':>9}{'PR-AUC':>9}{'top-20% captures':>18}{'top-20% precision':>19}")
    for m in scorers:
        p = point[m]
        print(f"{m:<24}{p['roc_auc']:>9.3f}{p['pr_auc']:>9.3f}{p['top20_capture']:>17.1%}{p['top20_precision']:>19.1%}")

    imp = permutation_importance(gbm, X["valid"], y["valid"], scoring="roc_auc", n_repeats=15,
                                 random_state=config.SEED)
    importance = sorted(
        ({"feature": c, "label": c.replace("_", " "),
          "mean": float(m), "std": float(s)} for c, m, s in zip(cols, imp.importances_mean, imp.importances_std)),
        key=lambda r: -r["mean"],
    )

    bt = backtest(dataset, cols)
    print(f"\n{BACKTEST_TITLE}: ROC-AUC")
    print(f"{'model':<44}" + "".join(f"{f['label']:>9}" for f in bt["folds"]) + f"{'mean':>9}")
    for m in bt["models"]:
        print(f"{m:<44}" + "".join(f"{f['roc_auc'][m]:>9.3f}" for f in bt["folds"]) + f"{bt['mean'][m]:>9.3f}")
    print(f"{'GBM minus logistic regression':<44}" + "".join(f"{g:>+9.3f}" for g in bt["gbm_minus_logreg"]["per_fold"])
          + f"{bt['gbm_minus_logreg']['mean']:>+9.3f}")

    horizon = config.HORIZON_DAYS
    report = {
        "gbm_model": gbm_name,
        "lightgbm_used": gbm_name == "LightGBM",
        "gbm_trees": int(getattr(gbm, "best_iteration_", None) or getattr(gbm, "n_iter_", 0)),
        "features": cols,
        "horizon_days": horizon,
        "splits": {
            s: {
                "cutoffs": config.SPLITS[s],
                "label_window_ends": [str((pd.Timestamp(c) + pd.Timedelta(days=horizon) - pd.Timedelta(days=1)).date())
                                      for c in config.SPLITS[s]],
                "rows": int(len(data[s])),
                "repurchase_rate": round(float(y[s].mean()), 4),
            } for s in data
        },
        "last_data_day": str(last_day),
        "test_customers": int(len(y["test"])),
        "models": {
            m: {k: {"value": point[m][k], "ci_low": ci[m][k][0], "ci_high": ci[m][k][1]} for k in point[m]}
            for m in scorers
        },
        "latest_quarter_only": [MLP_NAME] if MLP_NAME in scorers else [],      # trained for the latest quarter only: not in the backtest
        "latest_quarter_label": LATEST_QUARTER_LABEL,
        "backtest": bt,
        "auc_diff_vs_logreg": {m: {"ci_low": lo, "ci_high": hi, "value": point[m]["roc_auc"] - point[LOGREG]["roc_auc"]}
                               for m, (lo, hi) in diff_ci.items()},
        "curves": {m: {"gain": metrics.gain_curve(y["test"], s), "roc": metrics.roc_points(y["test"], s),
                       "pr": metrics.pr_points(y["test"], s)} for m, s in test_scores.items()},
        "importance": importance,
        "base_rate_test": round(float(y["test"].mean()), 4),
        "mlp_history": mlp_history,
        "notes": [
            "Time-based split: no customer-period in training overlaps the test period.",
            "The test window (Sep-Dec 2011) includes the Christmas peak, so absolute repurchase rates "
            "are higher than in a quiet quarter. Model ranking, not the absolute rate, is the point.",
            "Metrics use customers who had already ordered before the cutoff; brand-new customers are not scored.",
        ],
    }
    (config.PROCESSED_DIR / "metrics.json").write_text(json.dumps(report, indent=2))

    out = data["test"][["customer_id", "label"]].copy()
    for m, s in test_scores.items():
        out[f"score_{m}"] = s
    out.to_parquet(config.PROCESSED_DIR / "test_scores.parquet", index=False, compression="zstd")

    # Score everyone as of "today" (the day after the last order). The label is unknown for these rows.
    today = last_day + dt.timedelta(days=1)
    now = features.build_snapshot(con, today)
    scored = now[["customer_id"]].assign(p_repurchase_90d=gbm.predict_proba(now[cols])[:, 1].round(4))
    scored = scored.merge(now[["customer_id", "recency_days", "frequency", "monetary"]], on="customer_id")
    scored.to_parquet(config.PROCESSED_DIR / "current_scores.parquet", index=False, compression="zstd")

    print("\nAUC difference vs logistic regression (95% bootstrap CI):")
    for m, (lo, hi) in diff_ci.items():
        print(f"  {m:<24}{point[m]['roc_auc'] - point[LOGREG]['roc_auc']:+.3f}  [{lo:+.3f}, {hi:+.3f}]")
    print("\nTop features (permutation importance on validation):")
    for r in importance[:6]:
        print(f"  {r['label']:<28}{r['mean']:.4f}")


if __name__ == "__main__":
    main()
