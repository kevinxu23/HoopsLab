from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterator, Tuple

def season_label(dt: pd.Timestamp) -> int:
    return dt.year if dt.month >= 10 else dt.year - 1

def walk_forward_splits(df: pd.DataFrame, date_col="GAME_DATE",
                        n_folds=4, min_train_seasons=3) -> Iterator[Tuple[np.ndarray,np.ndarray]]:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    seasons = sorted(d[date_col].apply(season_label).unique())
    seasons = [s for s in seasons if s >= seasons[0] + min_train_seasons]
    for s in seasons[-n_folds:]:
        train_idx = d.index[d[date_col].apply(season_label) < s].to_numpy()
        test_idx  = d.index[d[date_col].apply(season_label) == s].to_numpy()
        yield train_idx, test_idx
