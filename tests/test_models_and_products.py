"""Model fitting helpers, the PyTorch MLP and product-category helpers, on small synthetic data."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from rci import features, model, products


def synthetic(n=1500, seed=0):
    """Customers whose repurchase depends mostly on recency and frequency (as in the real data)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({c: rng.gamma(2.0, 50.0, n) for c in features.BASE_FEATURES})
    X["is_uk"] = rng.integers(0, 2, n)
    logit = -0.03 * X["recency_days"] + 0.03 * X["frequency"] + 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit * 0.4))).astype(int)
    return X, y


def test_logistic_regression_beats_chance():
    X, y = synthetic()
    lr = model.fit_logreg(X.iloc[:1000], y[:1000])
    auc = roc_auc_score(y[1000:], lr.predict_proba(features.log_transform(X.iloc[1000:]))[:, 1])
    assert auc > 0.65


def test_gradient_boosting_fits_and_scores():
    X, y = synthetic()
    gbm, name = model.fit_gbm(X.iloc[:1000], y[:1000], X.iloc[1000:1250], y[1000:1250])
    assert name in {"LightGBM", "HistGradientBoosting"}
    assert roc_auc_score(y[1250:], gbm.predict_proba(X.iloc[1250:])[:, 1]) > 0.65


def test_recency_rule_is_just_negative_recency():
    X, _ = synthetic(50)
    scorer = lambda Z: -Z["recency_days"].to_numpy()  # noqa: E731  (the same rule used in model.main)
    assert np.argmax(scorer(X)) == np.argmin(X["recency_days"].to_numpy())


def test_recent_orders_rule_ranks_by_orders_in_the_last_90_days_then_by_recency():
    X = pd.DataFrame({"orders_last_90d": [2, 2, 1, 0], "recency_days": [10, 5, 1, 3]})
    score = model.recent_orders_score(X)
    assert score[0] == pytest.approx(2 - 10 / 10_000)
    assert np.argsort(-score).tolist() == [1, 0, 2, 3]            # more recent orders first; a tie goes to the shorter recency
    far = pd.DataFrame({"orders_last_90d": [1, 2], "recency_days": [0, 9_999]})
    assert model.recent_orders_score(far)[1] > model.recent_orders_score(far)[0]   # recency can never outrank a whole order


def test_two_feature_logistic_regression_uses_only_recency_and_frequency():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = synthetic()
    predict = model.fit_logreg_recency_frequency(X.iloc[:1000], y[:1000])
    two = ["recency_days", "frequency"]
    by_hand = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(np.log1p(X.iloc[:1000][two]), y[:1000])
    np.testing.assert_allclose(predict(X.iloc[1000:]), by_hand.predict_proba(np.log1p(X.iloc[1000:][two]))[:, 1])
    scrambled = X.iloc[1000:].copy()
    for c in set(X.columns) - set(two):
        scrambled[c] = scrambled[c].sample(frac=1, random_state=1).to_numpy()
    np.testing.assert_allclose(predict(scrambled), predict(X.iloc[1000:]))     # no other column is read
    assert roc_auc_score(y[1000:], predict(X.iloc[1000:])) > 0.65


def test_backtest_scores_every_model_but_the_mlp_in_every_fold():
    X, y = synthetic()
    frame = X.assign(label=y)
    bt = model.backtest(lambda cutoffs: frame, features.BASE_FEATURES)
    gbm = next(m for m in bt["models"] if m in {"LightGBM", "HistGradientBoosting"})
    assert bt["models"] == [model.RECENCY_RULE, model.RECENT_ORDERS_RULE, model.LOGREG_2, model.LOGREG, gbm]
    assert bt["title"] == "Three-quarter backtest (MLP not included)"
    assert [f["label"] for f in bt["folds"]] == ["2011-03", "2011-06", "2011-09"]
    for fold in bt["folds"]:
        assert list(fold["roc_auc"]) == bt["models"] and model.MLP_NAME not in fold["roc_auc"]
    assert model.MLP_NAME not in bt["mean"] and model.MLP_NAME not in bt["gbm_minus_logreg"]
    for name in bt["models"]:
        assert bt["mean"][name] == pytest.approx(np.mean([f["roc_auc"][name] for f in bt["folds"]]))
    gap = [f["roc_auc"][gbm] - f["roc_auc"][model.LOGREG] for f in bt["folds"]]
    assert bt["gbm_minus_logreg"]["per_fold"] == pytest.approx(gap)
    assert bt["gbm_minus_logreg"]["mean"] == pytest.approx(np.mean(gap))


def test_mlp_learns_a_simple_pattern_and_is_deterministic():
    pytest.importorskip("torch")
    from rci import mlp

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1200, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.3, 1200) > 0).astype(int)
    predict_a, hist = mlp.fit_mlp(X[:800], y[:800], X[800:1000], y[800:1000], seed=7, max_epochs=40)
    predict_b, _ = mlp.fit_mlp(X[:800], y[:800], X[800:1000], y[800:1000], seed=7, max_epochs=40)
    assert roc_auc_score(y[1000:], predict_a(X[1000:])) > 0.9
    assert np.allclose(predict_a(X[1000:]), predict_b(X[1000:]), atol=1e-5)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


# ── products ─────────────────────────────────────────────────────────────────
def test_description_cleaning_removes_colour_size_and_filler_words():
    assert products.clean_description("JUMBO BAG RED RETROSPOT") == "bag retrospot"
    assert products.clean_description("SET OF 3 LARGE PINK CANDLES") == "candles"
    assert products.clean_description("RED") == "red"            # never returns an empty string


def test_cluster_names_use_distinctive_words_and_are_unique():
    texts = ["candle holder", "scented candle", "candle tin", "tea mug", "coffee mug", "mug cup"] * 5
    labels = np.array([0, 0, 0, 1, 1, 1] * 5)
    names = products.name_clusters(texts, labels)
    assert "Candle" in names[0] and "Mug" in names[1]
    assert names[0] != names[1]


def test_plural_and_singular_are_not_both_used_in_a_name():
    texts = ["candle", "candles", "candle holder"] * 4 + ["mug"] * 12
    labels = np.array([0] * 12 + [1] * 12)
    name = products.name_clusters(texts, labels)[0]
    assert not ("Candle" in name and "Candles" in name)


def test_clustering_picks_k_from_the_grid():
    rng = np.random.default_rng(0)
    centres = np.eye(8)[:4] * 5
    E = np.vstack([c + rng.normal(0, 0.3, (60, 8)) for c in centres])
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    km, k, scores = products.cluster(E, k_grid=(2, 4, 6))
    assert k == 4 and set(scores) == {2, 4, 6}
