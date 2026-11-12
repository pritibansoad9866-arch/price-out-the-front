"""Elliptic Bitcoin dataset: a real, labelled illicit transaction graph.

Files: elliptic_txs_classes.csv (txId, class: 1 illicit, 2 licit, unknown),
       elliptic_txs_edgelist.csv (txId1, txId2)
Used to check the fleet module's premise: illicit actors cluster (homophily),
so linking by shared counterparties raises the prior on the whole cluster.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from ._io import read_csv_ci


def fleet_benchmark(classes_path, edges_path, min_size: int = 3) -> dict:
    c = read_csv_ci(classes_path, ["txId", "class"])
    e = read_csv_ci(edges_path, ["txId1", "txId2"])
    lab = c.set_index("txId")["class"].astype(str)
    e = e[e["txId1"].isin(lab.index) & e["txId2"].isin(lab.index)]
    a, b = lab.reindex(e["txId1"]).to_numpy(), lab.reindex(e["txId2"]).to_numpy()
    known = (a != "unknown") & (b != "unknown")
    a, b = a[known], b[known]
    p_illicit = float((lab == "1").sum() / ((lab == "1") | (lab == "2")).sum())
    ill_edges = (a == "1") | (b == "1")
    homophily = float(((a == "1") & (b == "1")).sum() / max(ill_edges.sum(), 1))
    G = nx.Graph()
    ill = e[known][(a == "1") & (b == "1")]
    G.add_edges_from(zip(ill["txId1"], ill["txId2"]))
    sizes = sorted((len(cc) for cc in nx.connected_components(G)), reverse=True)
    in_fleets = sum(s for s in sizes if s >= min_size)
    return {
        "base_rate_illicit": round(p_illicit, 4),
        "P(neighbour illicit | node illicit)": round(homophily, 4),
        "lift": round(homophily / p_illicit, 2) if p_illicit else np.nan,
        "illicit_components_ge_min": int(sum(s >= min_size for s in sizes)),
        "share_illicit_in_fleets": round(in_fleets / max((lab == "1").sum(), 1), 3),
        "largest_components": sizes[:5],
    }
