from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class EloConfig:
    k_base: float = 20.0
    mov_mult: float = 0.006
    home_court: float = 60.0
    regress_to_mean: float = 0.25
    base: float = 1500.0

class Elo:
    def __init__(self, cfg: EloConfig = EloConfig()):
        self.cfg = cfg
        self.ratings: dict[str, float] = {}

    def _ensure(self, team: str):
        if team not in self.ratings:
            self.ratings[team] = self.cfg.base

    def expected(self, h_elo, a_elo):
        return 1.0 / (1.0 + 10.0 ** ((a_elo - h_elo) / 400.0))

    def update_game(self, home, away, home_pts, away_pts):
        self._ensure(home); self._ensure(away)
        h = self.ratings[home] + self.cfg.home_court
        a = self.ratings[away]
        exp = self.expected(h, a)

        s = 1.0 if home_pts > away_pts else 0.0
        mov = abs(home_pts - away_pts)
        k = self.cfg.k_base * (1 + self.cfg.mov_mult * mov)

        delta = k * (s - exp)
        self.ratings[home] += delta
        self.ratings[away] -= delta

    def preseason_regress(self):
        mean = np.mean(list(self.ratings.values())) if self.ratings else self.cfg.base
        for t in list(self.ratings):
            self.ratings[t] = (1 - self.cfg.regress_to_mean) * self.ratings[t] + self.cfg.regress_to_mean * mean

def compute_daily_elo(games_df: pd.DataFrame,
                      home_col="HOME_TEAM_ABBREV", away_col="AWAY_TEAM_ABBREV",
                      home_pts_col="PTS_home", away_pts_col="PTS_away",
                      date_col="GAME_DATE",
                      cfg: EloConfig = EloConfig()) -> pd.DataFrame:
    df = games_df.copy().sort_values(date_col)
    elo = Elo(cfg)
    rows = []
    cur_season = None
    for _, r in df.iterrows():
        dt = pd.to_datetime(r[date_col])
        season = dt.year if dt.month >= 10 else dt.year - 1
        if cur_season is None:
            cur_season = season
        if season != cur_season:
            elo.preseason_regress()
            cur_season = season

        h, a = str(r[home_col]), str(r[away_col])
        elo._ensure(h); elo._ensure(a)
        h_pre, a_pre = elo.ratings[h] + cfg.home_court, elo.ratings[a]
        rows.append((r["GAME_ID"], h_pre, a_pre, (h_pre - a_pre)))

        if pd.notnull(r.get(home_pts_col)) and pd.notnull(r.get(away_pts_col)):
            elo.update_game(h, a, int(r[home_pts_col]), int(r[away_pts_col]))

    out = pd.DataFrame(rows, columns=["GAME_ID", "elo_home", "elo_away", "elo_diff"])
    return out

def fit_bt_strengths(games_df: pd.DataFrame,
                     home_team_col="HOME_TEAM_ABBREV",
                     away_team_col="AWAY_TEAM_ABBREV",
                     home_win_col="home_team_wins") -> pd.Series:
    import statsmodels.api as sm
    df = games_df[[home_team_col, away_team_col, home_win_col]].dropna().copy()
    teams = sorted(set(df[home_team_col]) | set(df[away_team_col]))
    for t in teams:
        df[f"T_{t}_home"] = (df[home_team_col] == t).astype(int)
        df[f"T_{t}_away"] = (df[away_team_col] == t).astype(int)
    diffs = []
    for t in teams[:-1]:
        diffs.append(df[f"T_{t}_home"] - df[f"T_{t}_away"])
    import numpy as np
    X = np.vstack(diffs).T
    X = sm.add_constant(X)
    y = df[home_win_col].astype(int).values
    model = sm.Logit(y, X).fit(disp=False)
    beta = model.params
    strengths = pd.Series(beta[1:], index=teams[:-1])
    strengths[teams[-1]] = 0.0
    strengths -= strengths.mean()
    strengths.name = "bt_strength"
    strengths.attrs["hca_intercept"] = float(beta[0])
    return strengths
