"""Calibrated merchant population simulator.

Used so the prototype runs out of the box. It produces the same Panel shape as
the real-data loaders. Legitimate merchants are not static: some grow, some
add an auxiliary or adjacent line, all feel festivals, salary days and
weekends. Fronts have real documents (they look normal at onboarding or
were legitimate shops whose account got rented) and some are run as a fleet
by one syndicate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT, Config
from .nodes import F, load_all
from .panel import Panel, build_calendar

GEOS = ["Area A", "Area B", "Area C", "Area D"]
SCALES = ["small", "medium", "large"]
SCALE_P = [0.6, 0.3, 0.1]
NODE_P = {"kirana": 0.45, "restaurant": 0.25, "sports_bar": 0.15, "jewellery": 0.15}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n: int = 1200, days: int = 120, front_rate: float = 0.025, syndicate_size: int = 8,
             start: str = "2026-03-01", seed: int = 7, cfg: Config = DEFAULT,
             node_mix: dict | None = None) -> Panel:
    rng = np.random.default_rng(seed)
    nodes = load_all()
    mix = node_mix or NODE_P
    names = list(mix)
    dates = pd.date_range(start, periods=days, freq="D")
    cal = build_calendar(dates, seed=seed)
    fest, sal, wkd, match = (cal[c].to_numpy() for c in ["festival", "salary", "weekend", "match"])

    node_of = rng.choice(names, size=n, p=np.array([mix[k] for k in names]) / sum(mix.values()))
    geo_of = rng.choice(GEOS, size=n)
    scale_of = rng.choice(SCALES, size=n, p=SCALE_P)

    # --- who is a front -------------------------------------------------------
    is_front = np.zeros(n, bool)
    n_fronts = max(int(round(front_rate * n)), syndicate_size + (2 if front_rate > 0 else 0))
    syn_candidates = np.flatnonzero((node_of == "kirana") & (scale_of == "small"))
    syn_members = rng.choice(syn_candidates, size=syndicate_size, replace=False) if syndicate_size else np.array([], int)
    others = rng.choice(np.setdiff1d(np.arange(n), syn_members), size=n_fronts - syndicate_size, replace=False)
    is_front[syn_members] = True
    is_front[others] = True
    syndicate = np.full(n, -1)
    syndicate[syn_members] = 0

    front_type = np.array([""] * n, dtype=object)
    front_type[is_front] = np.where(rng.random(is_front.sum()) < 0.75, "betting", "mule")
    front_type[syn_members] = "betting"
    onset = np.full(n, -1)
    rented = rng.random(n) < 0.6
    onset[is_front] = np.where(rented[is_front], rng.integers(35, 95, n)[is_front], 0)
    onset[syn_members] = rng.integers(40, 60, syndicate_size)
    intensity = np.where(is_front, rng.uniform(0.05, 0.25, n), 0.0)
    intensity[syn_members] = rng.uniform(0.03, 0.05, syndicate_size)

    # --- benign changes among legitimate merchants ---------------------------
    benign = np.array([""] * n, dtype=object)
    u = rng.random(n)
    benign[(~is_front) & (u < 0.20)] = "growth"
    benign[(~is_front) & (u >= 0.20) & (u < 0.30)] = "adjacent"
    change_day = rng.integers(20, days - 15, n)

    opted_in = np.where(is_front, rng.random(n) < cfg.programme.uptake_front,
                        rng.random(n) < cfg.programme.uptake_legit)
    opted_in[syn_members] = False

    X = np.empty((n, days, F))
    t = np.arange(days)
    for m in range(n):
        nd = nodes[node_of[m]]
        c = nd.calendar
        cal_mult = (1 + c["festival"] * fest) * (1 + c["salary"] * sal) * (1 + c["weekend"] * wkd) * (1 + c["match"] * match)
        lam = nd.capacity[scale_of[m]] * np.exp(rng.normal(0, 0.25)) * cal_mult
        ticket = nd.ticket_mean * np.exp(rng.normal(0, nd.ticket_sd_log)) * np.ones(days)
        base = {k: _sigmoid(_logit(nd.shares[k]) + rng.normal(0, 0.25)) for k in nd.shares}
        sh = {k: np.full(days, v) for k, v in base.items()}
        if c["match"] > 0.3:
            sh["night"] = np.clip(sh["night"] + 0.10 * match, 0, 0.95)

        if benign[m] == "growth":
            ramp = np.clip((t - change_day[m]) / 40, 0, 1)
            lam = lam * np.exp(0.5 * ramp)
            sh["new_payer"] = sh["new_payer"] + 0.04 * ramp
        elif benign[m] == "adjacent":
            on = (t >= change_day[m]).astype(float)
            a = nd.adjacent
            lam = lam * (1 + a.get("txn", 0) * on)
            ticket = ticket * (1 + a.get("ticket", 0) * on)
            sh["round"] = sh["round"] + a.get("round", 0) * on

        if is_front[m]:
            s = intensity[m] * (t >= onset[m])
            if front_type[m] == "betting":
                k = np.where(match > 0, 1.8, 0.6) * s
                lam = lam * (1 + 1.0 * k)
                ticket = ticket * (1 + 0.8 * k)
                sh["round"] = sh["round"] + 0.40 * k
                sh["night"] = sh["night"] + 0.25 * k
                sh["new_payer"] = sh["new_payer"] + 0.35 * k
                sh["top_payer"] = sh["top_payer"] + 0.04 * k
            else:
                lam = lam * (1 + 0.2 * s)
                ticket = ticket * (1 + 2.5 * s)
                sh["round"] = sh["round"] + 0.30 * s
                sh["night"] = sh["night"] + 0.08 * s
                sh["new_payer"] = sh["new_payer"] - 0.05 * s
                sh["top_payer"] = sh["top_payer"] + 0.45 * s

        ntx = np.maximum(rng.poisson(lam), 1)
        X[m, :, 0] = np.log1p(ntx)
        X[m, :, 1] = np.log(ticket) + rng.normal(0, 0.06 + 0.5 / np.sqrt(ntx))
        for j, k in enumerate(["round", "night", "new_payer", "top_payer"]):
            p = np.clip(sh[k], 0.002, 0.95)
            X[m, :, 2 + j] = rng.binomial(ntx, p) / ntx

    # --- payer sets for the fleet graph ---------------------------------------
    pool = 4000
    geo_idx = {g: i for i, g in enumerate(GEOS)}
    syn_pool_start = len(GEOS) * pool
    own_pool_start = syn_pool_start + 10_000
    payers = []
    for m in range(n):
        g0 = geo_idx[geo_of[m]] * pool
        local = rng.choice(pool, size=120, replace=False) + g0
        if is_front[m] and syndicate[m] >= 0:
            local = np.concatenate([local[:48], rng.choice(220, size=72, replace=False) + syn_pool_start])
        elif is_front[m]:
            local = np.concatenate([local[:48], rng.choice(5000, size=72, replace=False) + own_pool_start + 5000 * m])
        payers.append(np.unique(local))

    merchants = pd.DataFrame({
        "node": node_of, "geo": geo_of, "scale": scale_of, "opted_in": opted_in.astype(bool),
        "is_front": is_front, "front_type": front_type, "onset": onset, "intensity": intensity.round(3),
        "syndicate": syndicate, "benign_change": benign, "change_day": np.where(benign != "", change_day, -1),
    }, index=pd.Index([f"M{m:05d}" for m in range(n)], name="merchant_id"))
    return Panel(merchants=merchants, X=X, dates=dates, calendar=cal, payers=payers)
