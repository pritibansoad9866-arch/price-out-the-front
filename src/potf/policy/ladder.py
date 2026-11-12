"""The graded ladder: watch, prompt, delay, check, hold, exit.

A rung is reached only when BOTH hold:
  1. evidence:  E_t x LR_inclusion >= 1 / alpha_rung     (error budget, via Ville)
  2. economics: posterior odds >= cost_fp(rung) / cost_fn  (worth acting on)

The inclusion rule (declining the opt-in programme) has a higher impact:
its likelihood ratio multiplies the evidence. Because that ratio has mean 1
over honest merchants, the product is still an e-value, so budgets hold.
It is never a conviction on its own: with no behavioural evidence (E_t ~ 1)
the product cannot reach a rung that costs a merchant anything.

Termination differs by type: betting and mule fronts exit (with a report);
an undeclared but legitimate adjacent line is re-categorised, not exited.
"""
from __future__ import annotations

import numpy as np

from ..config import LADDER, RUNG_INDEX, Config

ACTION_BY_TYPE = {
    "betting": "Exit and file a suspicious transaction report",
    "mule": "Exit, freeze the suspect amount, file a report",
    "adjacent": "Re-categorise the declared business (no exit)",
    "growth": "Widen the envelope (no exit)",
    "decline": "No action",
    "none": "No action",
}

RUNG_ACTION = {
    "none": "None",
    "watch": "Monitor (invisible to the merchant)",
    "prompt": "Ask for one low-friction proof",
    "delay": "Delay settlement on the suspect amount",
    "check": "Human review",
    "hold": "Hold the suspect amount pending review",
}


def rung_action(rung: str, explanation: str) -> str:
    benign = explanation in ("adjacent", "growth", "decline", "none")
    if rung in ("check", "hold") and benign:
        return f"{RUNG_ACTION[rung]}; likely {ACTION_BY_TYPE[explanation].lower()}"
    if rung == "exit":
        return ACTION_BY_TYPE.get(explanation, "Exit")
    return RUNG_ACTION[rung]


MERCHANT_MESSAGE = {
    "none": "",
    "watch": "",
    "prompt": "Quick check: please confirm your shop details in the app.",
    "delay": "Some settlements take an extra day while we run a routine check.",
    "check": "Under review. A reviewer will explain what we saw and what you can send.",
    "hold": "Under review. The amount under review is held; the rest settles normally.",
    "exit": "Account closed after review.",
}


def inclusion_lr(opted_in: np.ndarray, cfg: Config) -> np.ndarray:
    pr = cfg.programme
    lr_decline = (1 - pr.uptake_front) / (1 - pr.uptake_legit)
    lr_join = pr.uptake_front / pr.uptake_legit
    return np.where(opted_in, lr_join, lr_decline)


def thresholds(cfg: Config) -> np.ndarray:
    """log(1/alpha) for each rung."""
    return np.array([np.log(1 / r.alpha) for r in LADDER])


def propose_rung(log_ev: np.ndarray, log_prior_odds: np.ndarray, cfg: Config, flagged: np.ndarray | None = None):
    """Highest rung whose evidence and economics conditions both hold. Shapes broadcast."""
    th = thresholds(cfg)
    out = np.zeros(np.broadcast(log_ev, log_prior_odds).shape, dtype=int)
    for k, r in enumerate(LADDER):
        if k == 0:
            continue
        need_odds = np.log(max(r.cost_fp, 1e-9) / cfg.cost_fn) if r.cost_fp > 0 else -np.inf
        ok = (log_ev >= th[k]) & (log_prior_odds + log_ev >= need_odds)
        out = np.where(ok, k, out)
    if flagged is not None:
        out = np.where(flagged & (out < RUNG_INDEX["watch"]), RUNG_INDEX["watch"], out)
    return out


def cap_for_mode(rung: np.ndarray, mode: str) -> np.ndarray:
    if mode == "shadow":
        return np.zeros_like(rung)
    if mode == "light":
        return np.minimum(rung, RUNG_INDEX["delay"])
    return rung
