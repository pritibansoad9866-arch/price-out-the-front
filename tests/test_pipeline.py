import numpy as np

from potf.config import DEFAULT
from potf.pipeline import metrics, run
from potf.policy.ladder import inclusion_lr, propose_rung
from potf.simulate import simulate


def test_fronts_are_priced_out(sim_run):
    m = metrics(sim_run)
    assert m["fronts_reached"]["delay"]["share_of_fronts"] >= 0.6
    assert m["fronts_reached"]["exit"]["share_of_fronts"] >= 0.4
    linked, total = m["syndicate_linked"]
    assert linked >= total - 1


def test_no_honest_merchant_exits(sim_run):
    s = sim_run.summary
    assert not ((~s.is_front) & (s.rung == "exit")).any()


def test_heavy_rungs_need_a_human(sim_run):
    s = sim_run.summary
    heavy = s[s.rung.isin(["check", "hold", "exit"])]
    reviewed = set(sim_run.reviews.merchant_id)
    assert set(heavy.index) <= reviewed


def test_transparency_only_upon_review(sim_run):
    s = sim_run.summary
    assert (s.reasons_visible == s.rung.isin(["check", "hold", "exit"])).all()


def test_shadow_mode_applies_nothing():
    p = simulate(n=400, days=90, seed=1)
    r = run(p, mode="shadow")
    assert (r.applied == 0).all()
    assert (r.proposed > 0).any()


def test_inclusion_rule_is_an_e_value_and_never_a_conviction():
    pr = DEFAULT.programme
    lr_in, lr_out = inclusion_lr(np.array([True, False]), DEFAULT)
    # mean over honest merchants is exactly 1, so E x LR is still an e-value
    assert abs(pr.uptake_legit * lr_in + (1 - pr.uptake_legit) * lr_out - 1) < 1e-12
    # with no behavioural evidence, declining the programme alone reaches nothing that costs anything
    rung = propose_rung(np.log(lr_out), np.log(0.01 / 0.99), DEFAULT)
    assert rung <= 1


def test_fleet_explained_together(sim_run):
    s = sim_run.summary
    syn = s[s.syndicate >= 0]
    assert (syn.explanation == "betting").mean() >= 0.75
