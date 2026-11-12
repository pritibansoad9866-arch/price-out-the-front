"""Command line: potf <command>.  Run `potf -h` for the list."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT, LADDER


def _print(obj):
    print(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def cmd_simulate(a):
    from .panel import save_panel
    from .simulate import simulate
    p = simulate(n=a.merchants, days=a.days, seed=a.seed, syndicate_size=a.syndicate)
    save_panel(p, a.out)
    print(f"Saved {p.M} merchants x {p.T} days to {a.out} ({int(p.merchants.is_front.sum())} fronts)")


def _load_or_simulate(a):
    from .panel import load_panel
    from .simulate import simulate
    if getattr(a, "panel", None):
        return load_panel(a.panel)
    return simulate(n=a.merchants, days=a.days, seed=a.seed)


def cmd_run(a):
    from .pipeline import metrics, run
    p = _load_or_simulate(a)
    res = run(p, mode=a.mode, seed=a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.summary.to_csv(out / "summary.csv")
    res.reviews.to_csv(out / "reviews.csv", index=False)
    _print(metrics(res))
    print(f"\nWrote {out/'summary.csv'} and {out/'reviews.csv'}")


def cmd_rollout(a):
    """Shadow mode first, then light rungs inside strict error budgets, then heavy rungs with a human."""
    from .pipeline import metrics, run
    p = _load_or_simulate(a)
    rows = []
    for mode in ["shadow", "light", "full"]:
        res = run(p, mode=mode, seed=a.seed)
        mt = metrics(res)
        s = res.summary
        prop = s.loc[~s.is_front, "proposed_rung"].value_counts().to_dict()
        rows.append({"stage": mode,
                     "honest merchants actually affected (prompt+)": int((~s.is_front & ~s.rung.isin(["none", "watch"])).sum()),
                     "honest merchants the evidence would escalate (delay+)": int(sum(v for k, v in prop.items() if k in ["delay", "check", "hold", "exit"])),
                     "fronts exited": mt["fronts_reached"]["exit"]["share_of_fronts"],
                     "fronts delayed or worse": mt["fronts_reached"]["delay"]["share_of_fronts"]})
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_game(a):
    from .game import play
    rows = []
    for r in play(rounds=a.rounds, M=a.money):
        rows.append({"round": r.round, "PayU moves": r.payu_move, "syndicate responds": r.response_text,
                     **{f"c[{k}] %": round(float(v) * 100, 2) for k, v in r.costs.items()}})
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_economics(a):
    from .policy.economics import Levers, cost_curve
    vols = np.array([1e5, 2e5, 3e5, 5e5, 1e6, 2e6, 5e6])
    full = Levers(True, True, True, True, True)
    rows = []
    for lv, name in [(Levers(), "status quo"), (full, "our system")]:
        for c in cost_curve(a.money, lv, DEFAULT, volumes=vols):
            rows.append({"policy": name, "Rs per front per month": int(c["V"]), "fronts": round(c["N"], 1),
                         "lifetime days": round(c["life_days"], 1), "cost per Rs %": round(c["total"] * 100, 2)})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nOutside option c_alt = {DEFAULT.economics.c_alt:.1%}; win line = {DEFAULT.economics.win_line:.1%}")


def cmd_real(a):
    from .loaders import bustabit, dunnhumby, ipl
    from .pipeline import metrics, run
    md = ipl.match_days(a.ipl) if a.ipl else None
    panel, agg = dunnhumby.load_panel(a.dunnhumby, days=a.days, match_days=md)
    if a.bustabit:
        bets = bustabit.load_bets(a.bustabit, bet_to_rs=a.bet_to_rs)
        payers = {mid: p for mid, p in zip(panel.merchants.index, panel.payers)}
        panel = bustabit.inject(agg, panel.merchants, bets, n_fronts=a.fronts, payers=payers, match_days=md)
    res = run(panel, mode=a.mode)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res.summary.to_csv(out / "summary_real.csv")
    _print(metrics(res))


def cmd_calibrate(a):
    out = {}
    if a.recruit_visits and a.recruit_dates:
        from .loaders import recruit
        out["restaurant (Recruit)"] = recruit.calendar_uplift(a.recruit_visits, a.recruit_dates)
    if a.favorita_train and a.favorita_holidays:
        from .loaders import favorita
        out["grocery (Favorita)"] = favorita.holiday_uplift(a.favorita_train, a.favorita_holidays, max_chunks=a.max_chunks)
    if a.ipl:
        from .loaders import ipl
        out["IPL match days"] = len(ipl.match_days(a.ipl))
    _print(out or {"note": "pass dataset paths; see data/README.md"})


def cmd_elliptic(a):
    from .loaders import elliptic
    _print(elliptic.fleet_benchmark(a.classes, a.edges))


def cmd_ladder(a):
    rows = [{"rung": r.name, "alpha": r.alpha, "evidence needed (E >=)": round(1 / r.alpha),
             "days at D=0.5/day": round(np.log(1 / r.alpha) / 0.5, 1), "budget per 1M/month": r.budget_per_million,
             "human": r.needs_human} for r in LADDER[1:]]
    print(pd.DataFrame(rows).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="potf", description="Pricing Out the Front: prototype")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def sim_args(p):
        p.add_argument("--merchants", type=int, default=1200)
        p.add_argument("--days", type=int, default=120)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--panel", help="load a saved panel instead of simulating")

    p = sub.add_parser("simulate", help="simulate a merchant population and save it")
    p.add_argument("--merchants", type=int, default=1200)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--syndicate", type=int, default=8)
    p.add_argument("--out", default="outputs/panel")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("run", help="run the full pipeline and score it")
    sim_args(p)
    p.add_argument("--mode", choices=["shadow", "light", "full"], default="full")
    p.add_argument("--out", default="outputs")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("rollout", help="shadow -> light -> full, side by side")
    sim_args(p)
    p.set_defaults(fn=cmd_rollout)

    p = sub.add_parser("game", help="play the repeated game")
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--money", type=float, default=5e6, help="Rs the syndicate moves per month")
    p.set_defaults(fn=cmd_game)

    p = sub.add_parser("economics", help="front cost per Rs, status quo vs our system")
    p.add_argument("--money", type=float, default=5e6)
    p.set_defaults(fn=cmd_economics)

    p = sub.add_parser("ladder", help="print the graded ladder and its error budgets")
    p.set_defaults(fn=cmd_ladder)

    p = sub.add_parser("real", help="run on real stores (dunnhumby) with real bets injected (Bustabit)")
    p.add_argument("--dunnhumby", required=True, help="path to transaction_data.csv")
    p.add_argument("--bustabit", help="path to bustabit.csv")
    p.add_argument("--ipl", help="path to IPL matches.csv")
    p.add_argument("--fronts", type=int, default=10)
    p.add_argument("--bet-to-rs", type=float, default=1.0)
    p.add_argument("--days", type=int, default=120)
    p.add_argument("--mode", choices=["shadow", "light", "full"], default="full")
    p.add_argument("--out", default="outputs")
    p.set_defaults(fn=cmd_real)

    p = sub.add_parser("calibrate", help="calendar effects from Recruit / Favorita / IPL")
    p.add_argument("--recruit-visits")
    p.add_argument("--recruit-dates")
    p.add_argument("--favorita-train")
    p.add_argument("--favorita-holidays")
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--ipl")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("elliptic", help="fleet premise check on the Elliptic graph")
    p.add_argument("--classes", required=True)
    p.add_argument("--edges", required=True)
    p.set_defaults(fn=cmd_elliptic)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
