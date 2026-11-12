"""End to end: sense, estimate, act, observe.

Five questions, asked in order, each cheaper than the next is costly:
  1. Anything different?   F-test-esque first flag against peers (screen everyone)
  2. What moved?           kernel MMD per channel (amount, timing, payers)
  3. Which explanation?    Bayes factors: growth / adjacent line vs betting / mule
  4. Is it a fleet?        payer-overlap graph, corroborated evidence
  5. Which rung?           error budgets (Ville) + cost ratio + human for heavy rungs

The rollout mode decides what is applied: shadow (log only), light (up to
delay, inside strict error budgets), full (heavy rungs only with a human).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DEFAULT, LADDER, RUNG_INDEX, Config
from .nodes import load_all
from .panel import Panel
from .policy.ladder import MERCHANT_MESSAGE, cap_for_mode, inclusion_lr, propose_rung, rung_action
from .stats.evalue import e_values
from .stats.explain import explain
from .stats.flag import first_flag
from .stats.fleet import find_fleets, fleet_factor
from .stats.peers import Baseline, peer_baseline

RUNGS = [r.name for r in LADDER]
HUMAN_FROM = RUNG_INDEX["check"]


@dataclass
class Result:
    panel: Panel
    cfg: Config
    baseline: Baseline
    flags: dict
    log_level: np.ndarray
    log_change: np.ndarray
    log_ev: np.ndarray          # [M, T] log evidence incl. inclusion rule
    log_fleet: np.ndarray       # [M, T] corroborated fleet evidence (-inf if not in a fleet)
    proposed: np.ndarray        # [M, T] rung the evidence supports
    applied: np.ndarray         # [M, T] rung actually applied (mode + human review)
    fleets: dict
    reviews: pd.DataFrame
    summary: pd.DataFrame
    explanations: dict = field(default_factory=dict)

    def merchant_index(self, merchant_id: str) -> int:
        return int(np.flatnonzero(self.panel.merchants.index == merchant_id)[0])


def _human_review(rng, is_front: bool | None, p_confirm_front=0.92, p_confirm_legit=0.05) -> str:
    if is_front is None:
        return "pending"
    return "confirmed" if rng.random() < (p_confirm_front if is_front else p_confirm_legit) else "cleared"


def run(panel: Panel, cfg: Config = DEFAULT, burn_in: int = 28, mode: str | None = None, seed: int = 0,
        inflation: float = 1.1) -> Result:
    mode = mode or cfg.mode
    rng = np.random.default_rng(seed)
    m = panel.merchants
    M, T = panel.M, panel.T
    nodes = load_all()

    b = peer_baseline(panel, burn_in=burn_in, inflation=inflation)
    flags = first_flag(b, window=cfg.window_days, p_cut=cfg.flag_p)
    lL, lC, lE = e_values(panel, b)
    log_ev = lE + np.log(inclusion_lr(m["opted_in"].to_numpy(), cfg))[:, None]
    log_ev[:, :b.burn_in] = 0.0

    if panel.payers:
        fl = find_fleets(panel.payers, m, threshold=cfg.fleet_overlap)
    else:
        fl = {"fleet_id": np.full(M, -1), "size": np.ones(M, int), "cross_geo": np.zeros(M, bool), "graph": None}
    base_rate = np.array([nodes[n].base_rate for n in m["node"]])
    log_prior = np.log(base_rate / (1 - base_rate)) + np.log(fleet_factor(fl["size"], cfg.fleet_n0))

    truth = m["is_front"].to_numpy() if panel.has_truth() else None
    reset = np.zeros(M)
    excluded_from_fleet = np.zeros(M, bool)
    run_own = np.zeros(M)
    run_fleet = np.full(M, -np.inf)
    approved = np.zeros(M, int)               # highest human-approved rung
    proposed = np.zeros((M, T), int)
    applied = np.zeros((M, T), int)
    log_fleet = np.full((M, T), -np.inf)
    reviews = []
    fid = fl["fleet_id"]

    for t in range(T):
        ev = log_ev[:, t] - reset
        run_own = np.maximum(run_own, ev)
        if (fid >= 0).any():
            members = (fid >= 0) & ~excluded_from_fleet
            tot = pd.Series(np.where(members, ev, 0.0)).groupby(fid).transform("sum").to_numpy()
            fev = np.where(members, tot, -np.inf)
            run_fleet = np.maximum(run_fleet, fev)
            log_fleet[:, t] = fev
        r_own = propose_rung(run_own, log_prior, cfg, flags["flag"][:, t])
        r_fleet = np.where(np.isfinite(run_fleet), propose_rung(np.where(np.isfinite(run_fleet), run_fleet, 0), log_prior, cfg), 0)
        prop = np.maximum(r_own, r_fleet)
        if t < b.burn_in:
            prop[:] = 0
        proposed[:, t] = prop

        want = cap_for_mode(prop, mode)
        need = (want >= HUMAN_FROM) & (want > approved)
        for i in np.flatnonzero(need):
            verdict = _human_review(rng, None if truth is None else bool(truth[i]))
            reviews.append({"merchant_id": m.index[i], "day": t, "date": panel.dates[t], "rung": RUNGS[want[i]],
                            "verdict": verdict, "is_front": None if truth is None else bool(truth[i])})
            if verdict == "confirmed":
                approved[i] = want[i]
            elif verdict == "cleared":
                reset[i] = log_ev[i, t]           # evidence restarts from here (still a valid e-process)
                run_own[i] = 0.0
                excluded_from_fleet[i] = True
        heavy_ok = approved >= want
        allowed = np.where(want >= HUMAN_FROM, np.where(heavy_ok, want, np.minimum(want, HUMAN_FROM - 1)), want)
        applied[:, t] = 0 if mode == "shadow" else allowed

    # ---- explanations, actions, transparency ---------------------------------
    # explain at the moment that matters: the day the merchant first reached its highest
    # proposed rung (or today, if nothing was proposed)
    win = 21
    top_r = proposed.max(axis=1)
    t_eval = np.where(top_r > 0, (proposed >= top_r[:, None]).argmax(axis=1), T - 1)
    t_eval = np.maximum(t_eval, b.burn_in + 1)
    expl, top, ptop = {}, [], []
    for i in range(M):
        lo = max(b.burn_in, t_eval[i] - win + 1)
        w_i = t_eval[i] + 1 - lo
        delta = np.nanmean(b.z_raw[i, lo:t_eval[i] + 1], axis=0) - np.nanmean(b.z_raw[i, :b.burn_in], axis=0)
        n_eff = 1.0 / (1.0 / w_i + 1.0 / b.burn_in)
        e = explain(nodes[m["node"].iloc[i]], delta, n_eff, b.u[i], b.sigma_w[i], b.sigma_lv[i])
        e.update(day=int(t_eval[i]), delta=delta, n_eff=n_eff, source="own")
        expl[m.index[i]] = e
    # fleets are explained together: members' shifts are pooled (corroboration)
    for f in np.unique(fid[fid >= 0]):
        idx = np.flatnonzero(fid == f)
        d = np.mean([expl[m.index[i]]["delta"] for i in idx], axis=0)
        n = sum(expl[m.index[i]]["n_eff"] for i in idx)
        uu = np.mean(b.u[idx], axis=0) * np.sqrt(len(idx))
        i0 = idx[0]
        e = explain(nodes[m["node"].iloc[i0]], d, n, uu, b.sigma_w[idx].mean(0), b.sigma_lv[idx].mean(0))
        for i in idx:
            own = expl[m.index[i]]
            if own["top"] in ("none", "decline") or own["posterior"][own["top"]] < e["posterior"][e["top"]]:
                expl[m.index[i]] = {**e, "day": own["day"], "delta": own["delta"], "n_eff": own["n_eff"],
                                    "source": f"fleet {f} ({len(idx)} merchants)"}
    for i in range(M):
        e = expl[m.index[i]]
        top.append(e["top"])
        ptop.append(e["posterior"][e["top"]])

    final = applied[:, -1]
    maxprop = proposed.max(axis=1)
    first_day = {}
    for k, name in enumerate(RUNGS[1:], start=1):
        hit = (applied >= k).any(axis=1)
        first_day[name] = np.where(hit, (applied >= k).argmax(axis=1), -1)

    summary = m.copy()
    summary["fleet_id"] = fid
    summary["fleet_size"] = np.where(fid >= 0, fl["size"], 1)
    summary["log_evidence"] = np.round(log_ev[:, -1] - reset, 2)
    summary["proposed_rung"] = [RUNGS[k] for k in maxprop]
    summary["rung"] = [RUNGS[k] for k in final]
    summary["explanation"] = top
    summary["p_explanation"] = np.round(ptop, 3)
    summary["explained_by"] = [expl[k]["source"] for k in m.index]
    summary["action"] = [rung_action(RUNGS[k], tp) for tp, k in zip(top, final)]
    # transparency only upon review: reasons are shown from the check rung up
    summary["merchant_sees"] = [MERCHANT_MESSAGE[RUNGS[k]] for k in final]
    summary["reasons_visible"] = final >= HUMAN_FROM
    for name, d in first_day.items():
        summary[f"day_{name}"] = d

    return Result(panel=panel, cfg=cfg, baseline=b, flags=flags, log_level=lL, log_change=lC, log_ev=log_ev,
                  log_fleet=log_fleet, proposed=proposed, applied=applied, fleets=fl,
                  reviews=pd.DataFrame(reviews), summary=summary, explanations=expl)


def metrics(res: Result) -> dict:
    """Score the run against ground truth (simulated or injected data only)."""
    s = res.summary
    if "is_front" not in s:
        return {"rung_counts": s["rung"].value_counts().to_dict()}
    T, burn = res.panel.T, res.baseline.burn_in
    months = max((T - burn) / 30.0, 1e-9)
    legit, fronts = s[~s.is_front], s[s.is_front]
    out = {"merchants": len(s), "fronts": len(fronts), "months_monitored": round(months, 2)}
    wrong = {}
    for k, r in enumerate(LADDER[1:], start=1):
        reached = (legit[f"day_{r.name}"] >= 0).sum()
        per_million = reached / (len(legit) * months) * 1e6
        wrong[r.name] = {"legit_reached": int(reached), "per_1M_merchant_months": round(per_million, 1),
                         "budget_per_1M": r.budget_per_million}
    out["wrongful_escalations"] = wrong
    caught = {}
    for name in ["delay", "check", "hold", "exit"]:
        d = fronts[f"day_{name}"]
        start = np.maximum(fronts["onset"], burn)
        lag = (d - start)[d >= 0]
        caught[name] = {"share_of_fronts": round(float((d >= 0).mean()), 3),
                        "median_days_after_onset": float(np.median(lag)) if len(lag) else None}
    out["fronts_reached"] = caught
    fl = fronts[fronts.syndicate >= 0]
    out["syndicate_linked"] = int((fl.fleet_id >= 0).sum()), int(len(fl))
    out["reviews"] = res.reviews["verdict"].value_counts().to_dict() if len(res.reviews) else {}
    return out
