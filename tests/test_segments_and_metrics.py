"""RFM segments, evaluation metrics and the bootstrap."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from rci import metrics, segments


# ── segments ─────────────────────────────────────────────────────────────────
def blobs(seed=0, n=200):
    """Four clearly separated customer groups in R/F/M space."""
    rng = np.random.default_rng(seed)
    specs = [(10, 25, 15000), (15, 3, 800), (250, 5, 1800), (400, 1, 250)]   # recency, orders, spend
    rows = []
    for r, f, m in specs:
        rows.append(pd.DataFrame({
            "recency_days": np.clip(rng.normal(r, r * 0.1, n), 1, None),
            "frequency": np.clip(rng.normal(f, f * 0.1, n).round(), 1, None),
            "monetary": np.clip(rng.normal(m, m * 0.1, n), 10, None),
        }))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "customer_id", range(len(df)))
    return df


def test_k_is_chosen_inside_the_allowed_range():
    Z = segments.StandardScaler().fit_transform(segments.log_matrix(blobs()))
    k, scores = segments.select_k(Z)
    assert 3 <= k <= 6
    assert set(scores) == set(range(2, 9))


def test_fit_recovers_four_named_segments():
    df = blobs()
    model, k, _ = segments.fit(df)
    labels = segments.assign(model, df)
    assert k == 4
    assert set(labels) == {"Champions", "Promising", "At risk", "Hibernating"}
    group = labels.groupby(np.arange(len(df)) // 200).agg(lambda s: s.mode()[0])
    assert group.tolist() == ["Champions", "Promising", "At risk", "Hibernating"]


def test_segment_names_are_unique_even_when_clusters_collide():
    z = np.array([[-1.0, 1.0, 1.0], [-0.5, 0.6, 0.6], [-0.2, 0.2, 0.2], [1.0, -1.0, -1.0]])
    names = segments.name_clusters(z)
    assert len(set(names.values())) == 4
    assert names[0] == "Champions"                      # the most engaged keeps the plain name


def test_naming_rule_quadrants():
    z = np.array([[-1, 1, 1], [-1, -1, -1], [1, 1, 1], [1, -1, -1]], dtype=float)
    names = segments.name_clusters(z)
    assert [names[i] for i in range(4)] == ["Champions", "Promising", "At risk", "Hibernating"]


def test_assign_uses_the_fitted_model_not_a_refit():
    df = blobs()
    model, *_ = segments.fit(df)
    newcomer = pd.DataFrame({"recency_days": [5], "frequency": [40], "monetary": [30000]})
    assert segments.assign(model, newcomer).iloc[0] == "Champions"


def test_summary_shares_add_up():
    df = blobs()
    model, *_ = segments.fit(df)
    rows = segments.summarize(df, segments.assign(model, df))
    assert sum(r["customer_share"] for r in rows) == pytest.approx(1, abs=1e-3)
    assert sum(r["revenue_share"] for r in rows) == pytest.approx(1, abs=1e-3)


# ── metrics ──────────────────────────────────────────────────────────────────
def test_top_fraction_capture_perfect_and_random_lists():
    y = np.array([1] * 10 + [0] * 90)
    perfect = np.where(y == 1, 1.0, 0.0)
    assert metrics.top_fraction_capture(y, perfect, 0.10) == 1.0
    assert metrics.top_fraction_capture(y, perfect, 0.20) == 1.0
    worst = 1 - perfect
    assert metrics.top_fraction_capture(y, worst, 0.10) == 0.0


def test_top_fraction_hand_example():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # top 30% = customers 0, 1, 2 -> contains 2 of the 3 positives, and 2 of the 3 picks are positive
    assert metrics.top_fraction_capture(y, score, 0.30) == pytest.approx(2 / 3)
    assert metrics.top_fraction_precision(y, score, 0.30) == pytest.approx(2 / 3)


def test_point_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = y * 0.4 + rng.random(500)
    m = metrics.point_metrics(y, s)
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["pr_auc"] == pytest.approx(average_precision_score(y, s))


def test_gain_curve_is_monotone_and_ends_at_one():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    g = metrics.gain_curve(y, y + rng.random(300))
    ys = [p[1] for p in g]
    assert ys[0] == 0.0 and ys[-1] == pytest.approx(1.0)
    assert all(b >= a - 1e-12 for a, b in zip(ys, ys[1:]))


def test_bootstrap_interval_contains_the_estimate_and_shrinks_with_more_data():
    rng = np.random.default_rng(3)

    def run(n):
        y = rng.integers(0, 2, n)
        good = y * 0.8 + rng.random(n)
        weak = y * 0.2 + rng.random(n)
        ci, diff = metrics.bootstrap(y, {"good": good, "weak": weak}, baseline="weak", n_boot=200)
        point = metrics.point_metrics(y, good)["roc_auc"]
        return point, ci["good"]["roc_auc"], diff["good"]

    point, (lo, hi), (dlo, dhi) = run(400)
    assert lo < point < hi
    assert dlo > 0                                       # 'good' is clearly better than 'weak', paired
    _, (lo2, hi2), _ = run(4000)
    assert (hi2 - lo2) < (hi - lo)
