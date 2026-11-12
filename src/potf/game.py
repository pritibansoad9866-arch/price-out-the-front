"""The repeated game.

PayU is the leader: it commits to one policy for everyone. The syndicate is
the follower: each round it reads the policy and best-responds with the
cheapest way to move its money, or leaves when nothing beats its outside
option. PayU then retunes against the response it saw. Normal businesses are
the third player; the policy is only acceptable while their friction stays
near zero.

    We cap volume, they split into fleets.
    We link the fleet, they probe the thresholds.
    We rotate thresholds, they fake trust.
    We vest the rewards, they leave.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .config import DEFAULT, Config
from .policy.economics import V_PEER, Levers, best_volume

PROBE_COST = 15000.0   # Rs per account burnt while learning fixed thresholds
FAKE_COST = 25000.0    # Rs per account to fake the programme's low-friction proofs (invoices, repeat buyers)

STRATEGIES = {
    "big_fronts": "A few big fronts, no limits respected",
    "fleet": "Split into a fleet of small fronts inside the envelope",
    "probe": "Probe the thresholds and sit just under them",
    "fake_trust": "Join the verified programme, behave, then exploit",
    "leave": "Leave PayU for the next-best channel",
}

# PayU's retune: which lever answers which response.
COUNTER = {"big_fronts": ("envelope", "ladder"), "fleet": ("link",), "probe": ("rotate",), "fake_trust": ("vest",)}

PAYU_MOVE = {
    ("envelope", "ladder"): "Cap volume with an envelope and start the graded ladder",
    ("link",): "Link fleets through shared payers",
    ("rotate",): "Rotate thresholds at random",
    ("vest",): "Vest the rewards slowly; scrutiny never fully drops",
}


@dataclass
class RoundResult:
    round: int
    levers: dict
    payu_move: str
    costs: dict          # strategy -> cost per Rs
    response: str
    response_text: str
    on_payu: bool


def strategy_costs(lv: Levers, cfg: Config, M: float) -> dict:
    env_small = cfg.envelope.unverified["small"]
    big = best_volume(M, lv, cfg, lo=M / 5, hi=M)["total"]
    fleet = best_volume(M, lv, cfg, lo=0.2 * env_small, hi=env_small)["total"]
    # probing burns accounts to learn where fixed thresholds sit
    probe = best_volume(M, lv, cfg, lo=0.6 * V_PEER, hi=1.0 * V_PEER, probing=True, extra_fixed=PROBE_COST)["total"]
    # fake trust: stake + months of honest behaviour, then exploit with verified headroom
    fake = best_volume(M, lv, cfg, lo=0.6 * V_PEER, hi=2.5 * V_PEER, verified=True,
                       extra_fixed=FAKE_COST)["total"]
    if lv.vest:
        fake += cfg.programme.forfeit(3) / (M / 3)   # forfeited vested rebates, spread over the volume moved
    return {"big_fronts": big, "fleet": fleet, "probe": probe, "fake_trust": fake,
            "leave": cfg.economics.c_alt}


def play(rounds: int = 6, cfg: Config = DEFAULT, M: float = 5e6, start: Levers | None = None):
    """Play the repeated game. M = money the syndicate wants to move each month (Rs)."""
    lv = start or Levers()
    out = []
    move = "Status quo: KYC once, score each payment"
    for r in range(rounds):
        costs = strategy_costs(lv, cfg, M)
        resp = min(costs, key=costs.get)
        out.append(RoundResult(r, lv.as_dict(), move, costs, resp, STRATEGIES[resp], resp != "leave"))
        if resp == "leave":
            break
        levers = COUNTER[resp]
        if all(getattr(lv, k) for k in levers):
            break                         # no counter left: equilibrium with the front on PayU
        lv = replace(lv, **{k: True for k in levers})
        move = PAYU_MOVE[levers]
    return out
