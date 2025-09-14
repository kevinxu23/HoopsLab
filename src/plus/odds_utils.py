from __future__ import annotations
import pandas as pd
import numpy as np

def american_to_prob(odds: float) -> float:
    o = float(odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)

def pair_no_vig_probs(home_odds: float, away_odds: float) -> tuple[float,float]:
    p_h = american_to_prob(home_odds)
    p_a = american_to_prob(away_odds)
    s = p_h + p_a
    if s <= 0:
        return (np.nan, np.nan)
    return (p_h / s, p_a / s)

def consensus_home_prob(odds_rows: list[tuple[float,float]]) -> float:
    fair_ps = [pair_no_vig_probs(h,a)[0] for (h,a) in odds_rows if pd.notnull(h) and pd.notnull(a)]
    if not fair_ps:
        return np.nan
    return float(np.median(fair_ps))

def attach_odds_features(games_df: pd.DataFrame, odds_df: pd.DataFrame,
                         key_cols=("GAME_DATE", "HOME_TEAM_ABBREV", "AWAY_TEAM_ABBREV")) -> pd.DataFrame:
    o = odds_df.copy()
    o["pair"] = list(zip(o["home_ml"], o["away_ml"]))
    agg = (o.groupby(list(key_cols))["pair"]
             .apply(lambda x: consensus_home_prob(list(x)))
             .reset_index()
             .rename(columns={"pair":"fair_home_prob"}))
    return games_df.merge(agg, on=list(key_cols), how="left")
