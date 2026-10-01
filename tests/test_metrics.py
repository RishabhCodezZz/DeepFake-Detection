"""The NumPy metric reimplementations replace sklearn on purpose (see CLAUDE.md),
so they are checked here against definitions simple enough to trust."""
import numpy as np
import pytest

from crossfuse_v5 import (
    balanced_accuracy_score, bootstrap_ci, bootstrap_stable_threshold,
    expected_calibration_error, fit_temperature, roc_auc_score, roc_curve,
    youden_threshold,
)


def brute_force_auc(y, s):
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))


@pytest.mark.parametrize("seed", range(20))
def test_roc_auc_matches_pairwise_definition_with_heavy_ties(seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 2, 60)
    if y.min() == y.max():
        y[0] = 1 - y[0]
    s = rng.randint(0, 6, 60) / 5.0  # only 6 distinct scores, so ties are everywhere
    assert roc_auc_score(y, s) == pytest.approx(brute_force_auc(y, s), abs=1e-12)


def test_roc_auc_perfect_inverted_and_single_class():
    y = np.array([0, 0, 1, 1])
    assert roc_auc_score(y, [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc_score(y, [0.9, 0.8, 0.2, 0.1]) == 0.0
    with pytest.raises(ValueError):
        roc_auc_score([1, 1, 1], [0.1, 0.2, 0.3])


def test_roc_curve_starts_at_origin_ends_at_one_and_is_monotonic():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    s = rng.rand(200)
    fpr, tpr, _ = roc_curve(y, s)
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0)
    area = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0)
    assert area == pytest.approx(roc_auc_score(y, s), abs=1e-12)


def test_balanced_accuracy_is_mean_per_class_recall():
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 0, 1, 1, 0]
    assert balanced_accuracy_score(y_true, y_pred) == pytest.approx((3 / 4 + 1 / 2) / 2)


def test_youden_threshold_separates_a_separable_set():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = youden_threshold(y, p)
    assert 0.3 < thr <= 0.7
    assert youden_threshold([1, 1], [0.2, 0.9], default=0.42) == 0.42


def test_bootstrap_threshold_is_deterministic_for_a_seed():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, 150)
    p = np.clip(y * 0.3 + rng.rand(150) * 0.7, 0, 1)
    assert bootstrap_stable_threshold(y, p, seed=7) == bootstrap_stable_threshold(y, p, seed=7)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.RandomState(2)
    y = rng.randint(0, 2, 300)
    s = y * 0.8 + rng.rand(300)
    lo, hi = bootstrap_ci(y, s, roc_auc_score, n_boot=500)
    assert lo < roc_auc_score(y, s) < hi
    assert 0.5 < lo and hi <= 1.0


def test_fit_temperature_recovers_a_known_overconfidence():
    rng = np.random.RandomState(3)
    true_logit = rng.randn(4000) * 2.0
    y = (rng.rand(4000) < 1 / (1 + np.exp(-true_logit))).astype(float)
    overconfident = true_logit * 3.0  # model is 3x too sure of itself
    assert fit_temperature(overconfident, y) == pytest.approx(3.0, rel=0.15)


def test_expected_calibration_error_is_zero_for_a_perfectly_calibrated_toy():
    probs = np.array([1.0, 1.0, 0.0, 0.0])
    labels = np.array([1, 1, 0, 0])
    assert expected_calibration_error(probs, labels) == pytest.approx(0.0)
