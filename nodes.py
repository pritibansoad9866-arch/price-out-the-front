"""Business-type nodes.

One core system; every business type plugs in as a node described by a spec
(YAML in ``potf/nodes``). A spec says what the declared business normally
looks like, which calendar effects are normal for it, which auxiliary or
adjacent lines are fine, and which crimes are plausible.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import yaml

# Daily features every merchant is described by. Each is compared with peers.
FEATURES = ["log_txn", "log_ticket", "round_share", "night_share", "new_payer_share", "top_payer_share"]
F = len(FEATURES)
CHANNELS = {"amount": [0, 1, 2], "timing": [3], "payers": [4, 5]}

# Signatures in RAW feature units (change in log count, log ticket, shares).
# Benign changes and illicit activity move a merchant in different directions;
# that is what lets the system tell "growth / a new side line" apart from "bets".
# The pipeline converts them into z-space with each merchant's own noise scale.
# Illicit signatures are where real labelled data plugs in (Bustabit for betting).
_BASE_SIGNATURES = {
    "growth": [1.0, 0.0, 0.0, 0.0, 0.08, -0.03],
    "decline": [-1.0, 0.0, 0.0, 0.0, -0.04, 0.03],
    "betting": [1.0, 0.8, 0.40, 0.25, 0.35, 0.04],
    "mule": [0.2, 1.25, 0.30, 0.08, -0.05, 0.45],
}
BENIGN = ["growth", "decline", "adjacent"]
ILLICIT = ["betting", "mule"]
HYPOTHESES = ["none"] + BENIGN + ILLICIT


def _unit(v) -> np.ndarray:  # noqa: D103
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


@dataclass
class Node:
    name: str
    label: str
    declared: str
    capacity: dict
    ticket_mean: float
    ticket_sd_log: float
    shares: dict
    calendar: dict
    adjacent: dict
    crimes: list
    hardest_to_fake: str
    base_rate: float
    priors: dict

    def signature_raw(self, kind: str) -> np.ndarray:
        """Effect of one unit of a benign or illicit explanation, in raw feature units."""
        if kind == "adjacent":
            a = self.adjacent
            return np.array([np.log1p(a.get("txn", 0)), np.log1p(a.get("ticket", 0)), a.get("round", 0), 0.0, 0.02, 0.0])
        sig = np.array(_BASE_SIGNATURES[kind], dtype=float)
        if kind == "betting" and self.calendar.get("match", 0) > 0.3:
            # Where late nights are normal (sports bar), night share carries less weight.
            sig[3] *= 0.35
        return sig

    def signature(self, kind: str, scale: np.ndarray | None = None) -> np.ndarray:
        """Unit direction in z-space. ``scale`` is the noise sd per feature ([F] or [m, F])."""
        raw = self.signature_raw(kind)
        if scale is None:
            return _unit(raw)
        v = raw / np.asarray(scale)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)

    def benign_matrix(self, scale: np.ndarray | None = None) -> np.ndarray:
        """[F, k] or [m, F, k] benign directions."""
        return np.stack([self.signature(k, scale) for k in BENIGN], axis=-1)

    def illicit_directions(self, scale: np.ndarray | None = None) -> np.ndarray:
        """[K, F] or [m, K, F] illicit directions."""
        return np.stack([self.signature(k, scale) for k in ILLICIT], axis=-2)


def _spec_dir() -> Path:
    return Path(str(resources.files("potf") / "nodes"))


def load_node(name_or_path: str) -> Node:
    p = Path(name_or_path)
    if not p.exists():
        p = _spec_dir() / f"{name_or_path}.yaml"
    d = yaml.safe_load(p.read_text())
    return Node(
        name=d["name"], label=d["label"], declared=d["declared"],
        capacity=d["capacity_txn_per_day"], ticket_mean=float(d["ticket_mean"]),
        ticket_sd_log=float(d["ticket_sd_log"]), shares=d["shares"], calendar=d["calendar"],
        adjacent=d["adjacent"], crimes=list(d["crimes"]), hardest_to_fake=d["hardest_to_fake"],
        base_rate=float(d["base_rate"]), priors=d["priors"],
    )


def load_all() -> dict[str, Node]:
    return {p.stem: load_node(str(p)) for p in sorted(_spec_dir().glob("*.yaml"))}
