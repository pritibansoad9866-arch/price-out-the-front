"""Pricing Out the Front: interactive prototype.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from potf.config import DEFAULT, LADDER
from potf.game import play
from potf.nodes import FEATURES, load_all
from potf.pipeline import RUNGS, metrics, run
from potf.policy.economics import Levers, cost_curve
from potf.policy.programme import BENEFITS_UPFRONT, LOW_FRICTION_STEPS, no_exploit_curve
from potf.simulate import simulate
from potf.stats.mmd import what_moved

# palette (validated: blue vs orange passes lightness, chroma, CVD and contrast checks)
HONEST, FRONT = "#1F5FAD", "#D9722B"
INK, MUTED, GRID = "#0D1B2A", "#5B6B7F", "#E3EAF3"
FEATURE_LABEL = {"log_txn": "Payments per day (log)", "log_ticket": "Average ticket (log Rs)",
                 "round_share": "Share in round amounts", "night_share": "Share at night (22:00-04:00)",
                 "new_payer_share": "Share from new payers", "top_payer_share": "Share of value from top 5 payers"}

st.set_page_config(page_title="Pricing Out the Front", layout="wide")


def style(fig, height=360, ytitle=None, xtitle=None):
    fig.update_layout(template="simple_white", height=height, margin=dict(l=10, r=10, t=44, b=10),
                      font=dict(family="Nunito, Arial", color=INK, size=14), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="top", y=-0.22, x=0))
    fig.update_xaxes(showgrid=False, title=xtitle, linecolor=MUTED)
    fig.update_yaxes(gridcolor=GRID, title=ytitle, linecolor=MUTED, zeroline=False)
    return fig


@st.cache_data(show_spinner="Simulating merchants and running the pipeline...")
def get_run(n, days, seed, front_rate, syndicate, mode):
    p = simulate(n=n, days=days, seed=seed, front_rate=front_rate, syndicate_size=syndicate)
    r = run(p, mode=mode, seed=seed)
    return r


with st.sidebar:
    st.header("Population")
    n = st.slider("Merchants", 300, 3000, 1200, 100)
    days = st.slider("Days", 60, 180, 120, 10)
    front_rate = st.slider("Share of fronts", 0.01, 0.06, 0.025, 0.005)
    syndicate = st.slider("Fleet size of the syndicate", 0, 15, 8)
    seed = st.number_input("Seed", 0, 9999, 7)
    st.header("Rollout stage")
    mode = st.radio("Mode", ["shadow", "light", "full"], index=2,
                    captions=["Log only, nothing applied", "Up to delay, inside error budgets", "Heavy rungs with a human"])

res = get_run(n, days, int(seed), front_rate, syndicate, mode)
s = res.summary
mt = metrics(res)

st.title("Pricing Out the Front")
st.caption("Keep fronts OFF PayU, then identify them. Game theory is the architecture, statistics the mechanics, "
           "technology the enabler. Simulated data; see the README for running on real Kaggle datasets.")

tabs = st.tabs(["Overview", "Merchant", "Fleets", "Economics", "Repeated game", "Nodes"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(4)
    c[0].metric("Merchants", f"{len(s):,}", f"{int(s.is_front.sum())} fronts", delta_color="off")
    c[1].metric("Fronts exited", f"{mt['fronts_reached']['exit']['share_of_fronts']:.0%}",
                f"median {mt['fronts_reached']['exit']['median_days_after_onset']} days after onset", delta_color="off")
    honest_hit = int((~s.is_front & s.rung.isin(["delay", "check", "hold", "exit"])).sum())
    c[2].metric("Honest merchants delayed or worse", honest_hit, f"of {int((~s.is_front).sum()):,}", delta_color="off")
    c[3].metric("Syndicate members linked", f"{mt['syndicate_linked'][0]} / {mt['syndicate_linked'][1]}")

    left, right = st.columns([3, 2])
    with left:
        cnt = pd.crosstab(s["rung"], s["is_front"]).reindex(RUNGS, fill_value=0)
        share = cnt / cnt.sum()
        fig = go.Figure()
        fig.add_bar(x=RUNGS, y=share.get(False, 0), name="Honest merchants", marker_color=HONEST,
                    customdata=cnt.get(False, 0), hovertemplate="%{y:.1%} (%{customdata})")
        fig.add_bar(x=RUNGS, y=share.get(True, 0), name="Fronts", marker_color=FRONT,
                    customdata=cnt.get(True, 0), hovertemplate="%{y:.1%} (%{customdata})")
        fig.update_layout(title="Where everyone ends up on the ladder", bargap=0.25, bargroupgap=0.08)
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(style(fig, ytitle="Share of group"), width="stretch")
    with right:
        st.subheader("Error budgets")
        rows = []
        for name, v in mt["wrongful_escalations"].items():
            b = v["budget_per_1M"]
            rows.append({"Rung": name, "Honest merchants reached": v["legit_reached"],
                         "Per 1M merchant-months": v["per_1M_merchant_months"],
                         "Budget per 1M": "none (costs ~0)" if b == float("inf") else str(int(b))})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Budgets set the thresholds: alpha = budget / 1M. Ville's inequality bounds how often an honest "
                   "merchant's evidence ever crosses 1/alpha. Watch and prompt cost an honest merchant nothing.")
    st.subheader("Merchants on a rung")
    show = s[s.rung != "none"].sort_values("log_evidence", ascending=False)
    cols = ["node", "geo", "scale", "rung", "explanation", "p_explanation", "explained_by", "action", "merchant_sees",
            "fleet_id", "is_front", "front_type"]
    st.dataframe(show[cols], width="stretch", height=320)

# ---------------------------------------------------------------- merchant
with tabs[1]:
    fronts = s[s.is_front].index.tolist()
    order = fronts + [i for i in s.index if i not in fronts]
    mid = st.selectbox("Merchant", order, format_func=lambda k: f"{k} · {s.loc[k, 'node']} · {s.loc[k, 'rung']}"
                       + (" · FRONT" if s.loc[k, "is_front"] else ""))
    i = res.merchant_index(mid)
    row = s.loc[mid]
    a, b = st.columns([2, 3])
    with a:
        st.markdown(f"**Declared business:** {load_all()[row.node].declared}")
        st.markdown(f"**Peer group:** {res.baseline.group[i]}  \n**Opted in to the programme:** {bool(row.opted_in)}")
        st.markdown(f"**Rung now:** `{row.rung}`  ·  **Action:** {row.action}")
        st.markdown(f"**Merchant sees:** {row.merchant_sees or '(nothing)'}")
        if not row.reasons_visible:
            st.caption("Transparency is only upon review: reasons are shown from the check rung up.")
        if row.is_front:
            st.warning(f"Ground truth (simulation only): {row.front_type} front from day {row.onset}, "
                       f"intensity {row.intensity}" + (f", syndicate fleet" if row.syndicate >= 0 else ""))
        elif row.benign_change:
            st.info(f"Ground truth (simulation only): honest, with a benign change ({row.benign_change}) "
                    f"from day {row.change_day}")
        e = res.explanations[mid]
        post = e["posterior"]
        fig = go.Figure(go.Bar(x=list(post.values()), y=list(post.keys()), orientation="h",
                               marker_color=[FRONT if k in ("betting", "mule") else HONEST for k in post],
                               hovertemplate="%{x:.1%}<extra></extra>"))
        fig.update_layout(title=f"Which explanation? (day {e['day']}, {e['source']})")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(style(fig, 280), width="stretch")
    with b:
        ev = np.maximum.accumulate(res.log_ev[i]) / np.log(10)
        fig = go.Figure()
        fig.add_scatter(x=res.panel.dates, y=res.log_ev[i] / np.log(10), name="Evidence log10 E", line=dict(color=INK, width=2))
        if np.isfinite(res.log_fleet[i]).any():
            fig.add_scatter(x=res.panel.dates, y=np.where(np.isfinite(res.log_fleet[i]), res.log_fleet[i], np.nan) / np.log(10),
                            name="Fleet evidence (corroborated)", line=dict(color=FRONT, width=2, dash="dot"))
        for r in LADDER[1:]:
            y = np.log10(1 / r.alpha)
            fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash="dash"), annotation_text=r.name,
                          annotation_position="top left", annotation_font=dict(color=MUTED, size=12))
        fig.update_yaxes(range=[-1.5, 7])
        if row.is_front and row.onset >= 0:
            fig.add_vline(x=res.panel.dates[max(int(row.onset), 0)], line=dict(color=FRONT, width=1))
        fig.update_layout(title="Evidence against the ladder thresholds (axis capped at E = 10^7)")
        st.plotly_chart(style(fig, 330, "log10 E"), width="stretch")
        feat = st.selectbox("Feature vs peers", FEATURES, format_func=FEATURE_LABEL.get, index=2)
        j = FEATURES.index(feat)
        fig = go.Figure()
        fig.add_scatter(x=res.panel.dates, y=res.baseline.med[i, :, j], name="Peer median", line=dict(color=MUTED, width=2))
        fig.add_scatter(x=res.panel.dates, y=res.panel.X[i, :, j], name="This merchant",
                        line=dict(color=FRONT if row.is_front else HONEST, width=2))
        match = res.panel.calendar["match"].to_numpy() > 0
        fig.add_scatter(x=res.panel.dates[match], y=res.panel.X[i, match, j], mode="markers", name="IPL match day",
                        marker=dict(size=8, color="white", line=dict(color=INK, width=1.5)))
        fig.update_layout(title=FEATURE_LABEL[feat])
        st.plotly_chart(style(fig, 300), width="stretch")
    if st.button("What moved? (kernel MMD per channel vs peers)"):
        peers = np.flatnonzero(res.baseline.group == res.baseline.group[i])
        t = e["day"]
        wm = what_moved(res.baseline.z_raw, i, peers, t)
        st.dataframe(pd.DataFrame([{"Channel": k, "MMD^2": round(v[0], 4), "p-value": round(v[1], 3)} for k, v in wm.items()]),
                     hide_index=True)

# ---------------------------------------------------------------- fleets
with tabs[2]:
    fid = res.fleets["fleet_id"]
    if (fid >= 0).sum() == 0:
        st.info("No fleets found in this population.")
    else:
        fl = s[s.fleet_id >= 0].groupby("fleet_id").agg(members=("node", "size"), areas=("geo", "nunique"),
                                                          fronts=("is_front", "sum"), top_rung=("rung", lambda r: max(r, key=RUNGS.index)))
        st.dataframe(fl, width="stretch")
        pick = st.selectbox("Fleet", fl.index.tolist())
        idx = np.flatnonzero(fid == pick)
        G = res.fleets["graph"].subgraph(idx)
        pos = nx.spring_layout(G, seed=1)
        ex, ey = [], []
        for u, v in G.edges():
            ex += [pos[u][0], pos[v][0], None]
            ey += [pos[u][1], pos[v][1], None]
        fig = go.Figure()
        fig.add_scatter(x=ex, y=ey, mode="lines", line=dict(color=GRID, width=2), hoverinfo="skip", showlegend=False)
        for truth, colr, name in [(True, FRONT, "Front"), (False, HONEST, "Honest")]:
            nodes = [k for k in G.nodes() if bool(s.iloc[k].is_front) == truth]
            if nodes:
                fig.add_scatter(x=[pos[k][0] for k in nodes], y=[pos[k][1] for k in nodes], mode="markers+text",
                                marker=dict(size=18, color=colr, line=dict(color="white", width=2)), name=name,
                                text=[s.index[k] for k in nodes], textposition="top center",
                                hovertext=[f"{s.index[k]} · {s.iloc[k].geo} · {s.iloc[k].rung}" for k in nodes],
                                hoverinfo="text")
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(title="Shared payers link these merchants (often across areas)")
        st.plotly_chart(style(fig, 420), width="stretch")
        st.caption(f"lambda(N) = lambda0 (1 + N / N0) with N0 = {DEFAULT.fleet_n0:g}: a fleet of {len(idx)} raises the prior "
                   f"{1 + len(idx) / DEFAULT.fleet_n0:.1f}x, and members' evidence multiplies (corroboration).")

# ---------------------------------------------------------------- economics
with tabs[3]:
    money = st.slider("Money the syndicate moves per month (Rs lakh)", 10, 200, 50, 10) * 1e5
    vols = np.geomspace(5e4, 5e6, 60)
    full = Levers(True, True, True, True, True)
    fig = go.Figure()
    for lv, name, colr in [(Levers(), "Status quo", FRONT), (full, "Our system", HONEST)]:
        cc = cost_curve(money, lv, DEFAULT, volumes=vols)
        fig.add_scatter(x=vols / 1e5, y=[c["total"] for c in cc], name=name, line=dict(color=colr, width=2),
                        hovertemplate="%{y:.1%}")
    fig.add_hline(y=DEFAULT.economics.c_alt, line=dict(color=MUTED, dash="dash", width=1),
                  annotation_text="outside option 3%", annotation_position="top left")
    fig.add_hline(y=DEFAULT.economics.win_line, line=dict(color=INK, dash="dot", width=1),
                  annotation_text="win line 4.5%", annotation_position="top left")
    env_l = DEFAULT.envelope.unverified["small"] / 1e5
    fig.add_scatter(x=[env_l, env_l], y=[0, 0.12], mode="lines", name="Envelope (Rs 3 lakh)",
                    line=dict(color=MUTED, width=1), hoverinfo="skip")
    fig.update_xaxes(type="log", title="Rs lakh per front per month", tickvals=[0.5, 1, 2, 3, 5, 10, 20, 50])
    fig.update_yaxes(tickformat=".0%", range=[0, 0.12])
    fig.update_layout(title="Front's cost per Rs moved through PayU")
    st.plotly_chart(style(fig, 420, "cost per Rs"), width="stretch")
    st.caption("c = (A + k) / (V T) + h rho (hold days / lifetime) + o + envelope friction. "
               "Small fronts pay a high fixed cost per Rs; big ones get caught fast. Fleets are linked.")
    a, b = st.columns(2)
    with a:
        tau, G_, F_ = no_exploit_curve(DEFAULT)
        fig = go.Figure()
        fig.add_scatter(x=tau, y=G_, name="G: gain from exploiting", line=dict(color=FRONT, width=2))
        fig.add_scatter(x=tau, y=F_, name="F: forfeited if caught", line=dict(color=HONEST, width=2))
        fig.update_layout(title="No-exploit condition G(tau) < F(tau)")
        st.plotly_chart(style(fig, 320, "Rs", "tenure (months)"), width="stretch")
    with b:
        st.markdown("**Verified programme (opt in; benefits told upfront)**")
        st.markdown("\n".join(f"- {x}" for x in BENEFITS_UPFRONT))
        st.markdown("**Low-friction steps a front can't replicate cheaply**")
        st.markdown("\n".join(f"- {x}" for x in LOW_FRICTION_STEPS))

# ---------------------------------------------------------------- game
with tabs[4]:
    money_g = st.slider("Syndicate's monthly volume (Rs lakh)", 10, 200, 50, 10, key="g") * 1e5
    rounds = play(M=money_g)
    rows = [{"Round": r.round, "PayU moves": r.payu_move, "Syndicate responds": r.response_text,
             "Cheapest cost per Rs": f"{min(v for k, v in r.costs.items() if k != 'leave'):.2%}"} for r in rounds]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    fig = go.Figure()
    strat = [k for k in rounds[0].costs if k != "leave"]
    xs = [r.round for r in rounds]
    for k, dash in zip(strat, ["solid", "dot", "dash", "dashdot"]):
        ys = [r.costs[k] for r in rounds]
        fig.add_scatter(x=xs, y=ys, name=k.replace("_", " "),
                        line=dict(color=INK if k == "big_fronts" else HONEST, width=2, dash=dash), mode="lines+markers",
                        marker=dict(size=8), hovertemplate="%{y:.1%}")
        off = {"fake_trust": 8, "probe": -8}.get(k, 0)
        fig.add_annotation(x=xs[-1], y=min(ys[-1], 0.068), text=k.replace("_", " ") + (" (off scale)" if ys[-1] > 0.07 else ""),
                           showarrow=False, xanchor="left", xshift=10, yshift=off, font=dict(color=INK, size=12))
    chosen = [r.costs[r.response] for r in rounds]
    fig.add_scatter(x=xs, y=chosen, mode="markers", name="syndicate's choice",
                    marker=dict(size=16, color="rgba(0,0,0,0)", line=dict(color=FRONT, width=2.5)), hoverinfo="skip")
    fig.add_hline(y=DEFAULT.economics.c_alt, line=dict(color=MUTED, width=1, dash="dash"),
                  annotation_text="outside option 3%: leave", annotation_position="top left")
    fig.update_yaxes(tickformat=".0%", range=[0, 0.07])
    fig.update_xaxes(dtick=1, range=[-0.3, len(rounds) - 0.2])
    fig.update_layout(title="Each round PayU retunes; the syndicate's cheapest route gets dearer until it leaves")
    st.plotly_chart(style(fig, 400, "cost per Rs", "round"), width="stretch")

# ---------------------------------------------------------------- nodes
with tabs[5]:
    st.markdown("One core system; every business type plugs in as a node with its own spec. "
                "A match-night spike at a sports bar is normal; at a kirana it is evidence.")
    nodes = load_all()
    pick = st.selectbox("Node", list(nodes), index=list(nodes).index("kirana"))
    nd = nodes[pick]
    st.code(yaml.safe_dump(nd.__dict__, sort_keys=False, allow_unicode=True), language="yaml")
