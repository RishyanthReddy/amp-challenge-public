"""Portable historical ordering for finite numeric scores.

NumPy's accelerated float quicksort can permute equal scores differently on
x86 and ARM. Object-key quicksort uses the generic comparison path, preserving
the historical ARM ordering without changing values or switching to a new
stable-sort policy. The original dataframe and its numeric dtypes are retained.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sort_descending_portable(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if not frame.index.is_unique:
        raise ValueError('Portable score sorting requires a unique dataframe index')
    keys = pd.to_numeric(frame[column], errors='raise')
    if not np.isfinite(keys.to_numpy(dtype=float)).all():
        raise ValueError(f'Score column {column} contains non-finite values')
    order = keys.astype(object).sort_values(ascending=False, kind='quicksort').index
    return frame.loc[order]
