"""The A/B maths is hand-written, so it is cross-checked against statsmodels."""
import pytest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest

from rci import abtest
from rci.ratelimit import SlidingWindowLimiter


@pytest.mark.parametrize("p1,uplift", [(0.10, 0.02), (0.30, 0.05), (0.45, 0.05), (0.45, 0.10), (0.05, 0.01)])
def test_sample_size_agrees_with_statsmodels(p1, uplift):
    ours = abtest.sample_size_per_group(p1, uplift)
    effect = proportion_effectsize(p1 + uplift, p1)
    theirs = NormalIndPower().solve_power(effect_size=effect, alpha=0.05, power=0.8, ratio=1.0)
    assert ours == pytest.approx(theirs, rel=0.03)       # two standard formulas, so allow 3%


def test_known_textbook_value():
    # 10% -> 12% at alpha 0.05, power 0.80 needs about 3,800 customers per group
    assert 3700 <= abtest.sample_size_per_group(0.10, 0.02) <= 3900


def test_power_at_the_required_size_is_about_80_percent():
    n = abtest.sample_size_per_group(0.30, 0.05)
    assert abtest.power_for_n(0.30, 0.05, n) == pytest.approx(0.80, abs=0.01)
    assert abtest.power_for_n(0.30, 0.05, n // 4) < 0.5


def test_minimum_detectable_uplift_inverts_sample_size():
    n = abtest.sample_size_per_group(0.45, 0.07)
    assert abtest.minimum_detectable_uplift(0.45, n) == pytest.approx(0.07, abs=0.002)


def test_more_ambitious_effects_need_fewer_customers():
    assert abtest.sample_size_per_group(0.3, 0.10) < abtest.sample_size_per_group(0.3, 0.05)


def test_halving_the_effect_roughly_quadruples_the_sample():
    ratio = abtest.sample_size_per_group(0.3, 0.025) / abtest.sample_size_per_group(0.3, 0.05)
    assert 3.6 < ratio < 4.4


def test_ztest_matches_statsmodels():
    r = abtest.two_proportion_ztest(300, 1000, 345, 1000)
    z, p = proportions_ztest([345, 300], [1000, 1000])
    assert r["p_value"] == pytest.approx(p, rel=1e-6)
    assert r["z"] == pytest.approx(z, rel=1e-6)
    assert r["uplift_abs"] == pytest.approx(0.045)
    assert r["ci_low"] < 0.045 < r["ci_high"]
    assert r["significant"]


def test_identical_groups_are_not_significant_and_ci_covers_zero():
    r = abtest.two_proportion_ztest(300, 1000, 300, 1000)
    assert r["p_value"] == pytest.approx(1.0)
    assert not r["significant"]
    assert r["ci_low"] < 0 < r["ci_high"]


@pytest.mark.parametrize("p,u", [(0.0, 0.1), (1.0, 0.1), (0.5, 0.0), (0.5, -0.1), (0.95, 0.06)])
def test_invalid_inputs_raise(p, u):
    with pytest.raises(ValueError):
        abtest.sample_size_per_group(p, u)


# ── rate limiter ─────────────────────────────────────────────────────────────
def test_rate_limiter_blocks_then_recovers():
    now = [0.0]
    lim = SlidingWindowLimiter(max_calls=2, window_s=10, clock=lambda: now[0])
    assert lim.try_acquire() and lim.try_acquire()
    assert not lim.try_acquire()                         # third call inside the window is refused
    now[0] = 10.5
    assert lim.try_acquire()                             # the old calls have left the window


def test_refused_calls_are_not_recorded():
    now = [0.0]
    lim = SlidingWindowLimiter(max_calls=1, window_s=10, clock=lambda: now[0])
    assert lim.try_acquire()
    for _ in range(5):
        assert not lim.try_acquire()
    now[0] = 10.1
    assert lim.try_acquire()
