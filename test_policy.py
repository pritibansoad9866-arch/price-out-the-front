import numpy as np

from potf.config import DEFAULT, LADDER, RUNG_INDEX
from potf.game import play
from potf.policy.economics import Levers, best_volume
from potf.policy.programme import no_exploit_curve


def test_ladder_days_match_the_deck():
    days = {r.name: np.log(1 / r.alpha) / 0.5 for r in LADDER[1:]}
    assert round(days["delay"], 1) == 13.8
    assert round(days["hold"], 1) == 19.8
    assert round(days["exit"], 1) == 24.4


def test_budgets_set_thresholds():
    for r in LADDER:
        if np.isfinite(r.budget_per_million):
            assert abs(r.alpha - r.budget_per_million / 1e6) < 1e-12


def test_repeated_game_ends_with_the_syndicate_leaving():
    rounds = play()
    assert [r.response for r in rounds] == ["big_fronts", "fleet", "probe", "fake_trust", "leave"]
    last = rounds[-1].costs
    assert min(v for k, v in last.items() if k != "leave") > DEFAULT.economics.c_alt


def test_status_quo_is_profitable_for_fronts():
    assert best_volume(5e6, Levers(), DEFAULT)["total"] < DEFAULT.economics.c_alt


def test_no_exploit_condition_holds_at_every_tenure():
    tau, G, F = no_exploit_curve(DEFAULT)
    assert (G < F).all()
