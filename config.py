"""System-wide parameters.

Every number here is illustrative and is meant to be re-estimated from real data.
They match the values used in the deck so the prototype and the slides agree.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rung:
    name: str
    alpha: float            # Ville bound: P(a legitimate merchant ever reaches this rung) <= alpha
    budget_per_million: float  # wrongful escalations allowed per 1M merchants a month
    cost_fp: float          # cost (Rs) of wrongly applying this rung to a legitimate merchant
    needs_human: bool
    friction: float         # share of a flagged merchant's volume lost to this rung's friction


# The graded ladder. Thresholds come from the error budgets: alpha = budget / 1e6.
# Watch and prompt carry no budget because they cost a legitimate merchant ~nothing.
LADDER: tuple[Rung, ...] = (
    Rung("none", 1.0, float("inf"), 0.0, False, 0.0),
    Rung("watch", 0.05, float("inf"), 0.0, False, 0.0),
    Rung("prompt", 0.01, float("inf"), 20.0, False, 0.001),
    Rung("delay", 1e-3, 1000.0, 400.0, False, 0.004),
    Rung("check", 2e-4, 200.0, 1500.0, True, 0.006),
    Rung("hold", 5e-5, 50.0, 8000.0, True, 0.02),
    Rung("exit", 5e-6, 5.0, 60000.0, True, 1.0),
)

RUNG_INDEX = {r.name: i for i, r in enumerate(LADDER)}


@dataclass
class Economics:
    c_alt: float = 0.03            # syndicate's cost per Rs of its next-best channel (2-3%)
    margin: float = 1.5            # win line multiplier: we win when c > margin * c_alt
    account_cost: float = 5000.0   # A: cost to rent one merchant account (Rs)
    setup_cost: float = 1700.0     # k: device, SIM, dressing per front (Rs)
    operating: float = 0.008       # o: base operating cost per Rs (mule cuts, cash-out)
    hold_share: float = 0.35       # h: share of in-flight volume lost when a hold lands

    @property
    def win_line(self) -> float:
        return self.margin * self.c_alt


@dataclass
class Envelope:
    # Monthly volume an unverified merchant can settle at normal speed, by scale band (Rs).
    unverified: dict = field(default_factory=lambda: {"small": 3e5, "medium": 1.2e6, "large": 5e6})
    verified_multiplier: float = 2.5   # verified merchants get a wider envelope (grows with tenure)
    over_envelope_delay_days: int = 3  # settlement delay applied above the envelope


@dataclass
class Programme:
    """Opt-in verified programme. Benefits are told upfront; rebates vest slowly."""
    stake: float = 2000.0              # refundable stake / documents effort (F at tenure 0)
    rebate_per_month: float = 900.0    # rebate that accrues each month and vests slowly
    vest_months: int = 12
    uptake_legit: float = 0.55         # share of legitimate merchants that opt in
    uptake_front: float = 0.05         # share of fronts that opt in (most avoid scrutiny)

    def forfeit(self, tenure_months: float) -> float:
        """F(tau): what a member loses if caught exploiting at tenure tau."""
        return self.stake + self.rebate_per_month * min(tenure_months, self.vest_months)

    def inclusion_lr(self) -> float:
        """Likelihood ratio of *not* joining: P(decline | front) / P(decline | legit)."""
        return (1 - self.uptake_front) / (1 - self.uptake_legit)


@dataclass
class Config:
    economics: Economics = field(default_factory=Economics)
    envelope: Envelope = field(default_factory=Envelope)
    programme: Programme = field(default_factory=Programme)
    base_rate: float = 0.01        # prior share of fronts in a node (re-estimated per node)
    cost_fn: float = 40000.0       # cost of a missed front for one month (litigation exposure)
    flag_p: float = 0.01           # first-flag (F-test-esque) significance
    window_days: int = 14          # rolling window for the first flag and MMD
    fleet_n0: float = 5.0          # N0 in lambda(N) = lambda0 (1 + N / N0)
    fleet_overlap: float = 0.08    # Jaccard payer overlap that links two merchants
    mode: str = "full"             # shadow | light | full  (rollout stage)


DEFAULT = Config()
