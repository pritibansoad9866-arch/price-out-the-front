import numpy as np

from potf.config import LADDER, RUNG_INDEX
from potf.stats.mmd import mmd_test


def test_honest_population_stays_inside_budgets(honest_run):
    s = honest_run.summary
    assert not s.is_front.any()
    # Ville: P(ever reach 1/alpha) <= alpha. Small slack for a finite simulation.
    ev = np.maximum.accumulate(honest_run.log_ev, axis=1)[:, -1]
    for name, slack in [("prompt", 2.0), ("delay", 5.0)]:
        a = LADDER[RUNG_INDEX[name]].alpha
        assert (ev >= np.log(1 / a)).mean() <= max(slack * a, 3 / len(s)), name
    assert not s.rung.isin(["check", "hold", "exit"]).any()


def test_benign_changes_do_not_escalate(sim_run):
    s = sim_run.summary
    benign = s[(~s.is_front) & (s.benign_change != "")]
    assert len(benign) > 50
    assert not benign.rung.isin(["check", "hold", "exit"]).any()


def test_recursive_residuals_are_standardised(honest_run):
    z = honest_run.baseline.z
    assert abs(np.nanmean(z)) < 0.05
    assert 0.85 < np.nanstd(z) < 1.2


def test_mmd_detects_shift_only():
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(30, 2)), rng.normal(size=(40, 2))
    assert mmd_test(x, y, n_perm=200)[1] > 0.05
    assert mmd_test(x + 1.5, y, n_perm=200)[1] < 0.05
