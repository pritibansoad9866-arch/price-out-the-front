from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv_ci(path, needed: list[str], **kw) -> pd.DataFrame:
    """Read a CSV keeping only `needed` columns, matching names case-insensitively."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. See data/README.md for where to download it.")
    head = pd.read_csv(path, nrows=0, **{k: v for k, v in kw.items() if k in ("sep",)})
    lower = {c.lower(): c for c in head.columns}
    missing = [n for n in needed if n.lower() not in lower]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; found {list(head.columns)}")
    cols = [lower[n.lower()] for n in needed]
    df = pd.read_csv(path, usecols=cols, **kw)
    return df.rename(columns={lower[n.lower()]: n for n in needed})
