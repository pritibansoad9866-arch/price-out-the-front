"""Which explanation? Bayes factors across benign and illicit explanations.

Festival and salary-day effects are already removed by the same-day peer
comparison. What is left is explained as: nothing, growth, decline, an
adjacent line (all fine), betting or mule pass-through (fronts).
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from ..nodes import HYPOTHESES, Node

S_GRID = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0])


def _loglik(v: np.ndarray, n: float, d: np.ndarray | None) -> float:
    if d is None:
        return -0.5 * n * float((v**2).sum())
    ll = -0.5 * n * ((v[None] - S_GRID[:, None] * d[None]) ** 2).sum(axis=1)
    return float(logsumexp(ll) - np.log(len(S_GRID)))


def explain(node: Node, delta: np.ndarray, n_eff: float, u: np.ndarray,
            scale_change: np.ndarray | None = None, scale_level: np.ndarray | None = None) -> dict:
    """delta: change in z units since onboarding (precision n_eff); u: onboarding level in z units."""
    delta, u = np.nan_to_num(delta), np.nan_to_num(u)
    logp = {}
    for h in HYPOTHESES:
        dc = None if h == "none" else node.signature(h, scale_change)
        dl = None if h == "none" else node.signature(h, scale_level)
        prior = node.priors.get(h, 0.02)
        logp[h] = np.log(prior) + _loglik(delta, max(n_eff, 1.0), dc) + _loglik(u, 1.0, dl)
    vals = np.array(list(logp.values()))
    post = dict(zip(logp, np.exp(vals - logsumexp(vals))))
    base = logp["none"] - np.log(node.priors["none"])
    log_bf = {h: logp[h] - np.log(node.priors.get(h, 0.02)) - base for h in HYPOTHESES if h != "none"}
    top = max(post, key=post.get)
    return {"posterior": post, "log_bf": log_bf, "top": top}
