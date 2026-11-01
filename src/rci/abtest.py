"""A/B test planning and reading, written by hand with only the standard library.

Scenario: a win-back email to the "At risk" segment.
  unit       = one customer (randomly assigned: half get the email, half do not)
  metric     = bought again within 90 days (yes / no)
  hypothesis = the email raises that 90-day repurchase rate

Two questions an analyst must answer:
  1. BEFORE the test: how many customers per group do I need?  -> sample_size_per_group()
  2. AFTER the test: is the difference real or luck?            -> two_proportion_ztest()
"""
from __future__ import annotations

from math import ceil, sqrt
from statistics import NormalDist

_N = NormalDist()  # the standard normal curve


def sample_size_per_group(p_control: float, uplift_abs: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Customers needed in EACH group to detect an absolute uplift of `uplift_abs`.

    alpha = chance of a false alarm (calling a useless email 'a winner'); 5% is the convention.
    power = chance of catching a real effect of the size we care about; 80% is the convention.

        n = (z_a * sqrt(2 * p_bar * (1 - p_bar)) + z_b * sqrt(p1 (1 - p1) + p2 (1 - p2)))^2 / (p2 - p1)^2
    """
    _check(p_control, uplift_abs)
    p1, p2 = p_control, p_control + uplift_abs
    p_bar = (p1 + p2) / 2
    z_a = _N.inv_cdf(1 - alpha / 2)   # two-sided test
    z_b = _N.inv_cdf(power)
    numerator = z_a * sqrt(2 * p_bar * (1 - p_bar)) + z_b * sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return ceil(numerator**2 / (p2 - p1) ** 2)


def power_for_n(p_control: float, uplift_abs: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Chance of detecting the uplift when each group has `n_per_group` customers."""
    _check(p_control, uplift_abs)
    p1, p2 = p_control, p_control + uplift_abs
    p_bar = (p1 + p2) / 2
    z_a = _N.inv_cdf(1 - alpha / 2)
    se_null = sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
    se_alt = sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n_per_group)
    return _N.cdf((abs(p2 - p1) - z_a * se_null) / se_alt)


def minimum_detectable_uplift(p_control: float, n_per_group: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute uplift we could reliably detect with `n_per_group` customers per group."""
    lo, hi = 1e-6, 1 - p_control - 1e-6
    for _ in range(60):  # bisection: power rises with uplift, so halve the search range each step
        mid = (lo + hi) / 2
        if power_for_n(p_control, mid, n_per_group, alpha) < power:
            lo = mid
        else:
            hi = mid
    return hi


def two_proportion_ztest(conv_a: int, n_a: int, conv_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Read a finished test. A = control, B = treatment.

    p-value: IF the email did nothing, how often would we see a gap at least this big by luck alone?
    A small p-value (< alpha) means 'unlikely to be luck'. It is NOT the probability that the email
    works, and it says nothing about whether the gap is big enough to be worth the cost.
    The confidence interval shows the plausible size of the true uplift.
    """
    p_a, p_b = conv_a / n_a, conv_b / n_b
    pooled = (conv_a + conv_b) / (n_a + n_b)
    se_pooled = sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se_pooled if se_pooled > 0 else 0.0
    p_value = 2 * (1 - _N.cdf(abs(z)))
    se_diff = sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    margin = _N.inv_cdf(1 - alpha / 2) * se_diff
    return {
        "rate_control": p_a,
        "rate_treatment": p_b,
        "uplift_abs": p_b - p_a,
        "z": z,
        "p_value": p_value,
        "ci_low": p_b - p_a - margin,
        "ci_high": p_b - p_a + margin,
        "significant": p_value < alpha,
    }


def _check(p_control: float, uplift_abs: float) -> None:
    if not 0 < p_control < 1:
        raise ValueError("p_control must be between 0 and 1")
    if uplift_abs <= 0 or p_control + uplift_abs >= 1:
        raise ValueError("uplift must be positive and keep the treatment rate below 100%")
