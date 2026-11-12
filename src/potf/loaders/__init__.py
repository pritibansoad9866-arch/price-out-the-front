"""Real-data loaders. Each maps a public dataset onto the system's shapes.

    dunnhumby  -> legitimate grocery stores (the kirana-like node), as a Panel
    bustabit   -> real betting flows, injected into real stores as semi-real fronts
    recruit    -> restaurant calendar effects (holiday and weekend uplift)
    favorita   -> grocery holiday uplift (festival calibration)
    ipl        -> real match calendar (match-night effects)
    elliptic   -> labelled illicit transaction graph (fleet / homophily benchmark)

Download the files from Kaggle (see data/README.md) and pass their paths.
"""
from .aggregates import AGG_COLS, features_from_aggregates, panel_from_aggregates  # noqa: F401
