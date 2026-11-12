"""Is it a fleet? Payer-overlap graph across merchants.

Honest shops share a few local customers. Fronts run by one syndicate share
the same bettors, often across areas. Linked merchants form a fleet; a fleet
raises the prior (lambda(N) = lambda0 (1 + N / N0)) and members' evidence
corroborates (the product of independent e-values is an e-value).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse


def payer_jaccard(payers: list) -> sparse.csr_matrix:
    rows = np.concatenate([np.full(len(p), i) for i, p in enumerate(payers)])
    cols = np.concatenate(payers)
    _, cols = np.unique(cols, return_inverse=True)
    A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)))
    inter = (A @ A.T).tocoo()
    sizes = np.asarray(A.sum(axis=1)).ravel()
    mask = inter.row < inter.col
    r, c, v = inter.row[mask], inter.col[mask], inter.data[mask]
    j = v / (sizes[r] + sizes[c] - v)
    return sparse.coo_matrix((j, (r, c)), shape=(len(payers), len(payers))).tocsr()


def find_fleets(payers: list, merchants: pd.DataFrame, threshold: float = 0.08, min_size: int = 3):
    M = len(payers)
    J = payer_jaccard(payers).tocoo()
    keep = J.data >= threshold
    G = nx.Graph()
    G.add_nodes_from(range(M))
    G.add_weighted_edges_from(zip(J.row[keep], J.col[keep], J.data[keep]))
    fleet_id = np.full(M, -1)
    size = np.ones(M, int)
    cross_geo = np.zeros(M, bool)
    geos = merchants["geo"].to_numpy()
    fid = 0
    for comp in nx.connected_components(G):
        comp = sorted(comp)
        if len(comp) < min_size:
            continue
        fleet_id[comp] = fid
        size[comp] = len(comp)
        cross_geo[comp] = len(set(geos[comp])) > 1
        fid += 1
    return {"fleet_id": fleet_id, "size": size, "cross_geo": cross_geo, "graph": G}


def fleet_factor(size: np.ndarray, n0: float) -> np.ndarray:
    """lambda(N) / lambda0 for fleet members (1 for singletons)."""
    return np.where(size >= 3, 1 + size / n0, 1.0)


def fleet_log_evidence(logE: np.ndarray, fleet_id: np.ndarray) -> np.ndarray:
    """Corroboration: sum of members' log e-values (product of e-values)."""
    out = np.full_like(logE, -np.inf)
    for f in np.unique(fleet_id[fleet_id >= 0]):
        idx = fleet_id == f
        out[idx] = logE[idx].sum(axis=0, keepdims=True)
    return out
