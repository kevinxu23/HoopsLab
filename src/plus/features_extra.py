from __future__ import annotations
import pandas as pd
import numpy as np
from .utils import haversine_km  # for travel distance

# ---------- Flexible column resolver ----------
_ALIASES = {
    "FG":  ["FG","FGM","FIELD_GOALS_MADE"],
    "FGA": ["FGA","FGA_ATT","FIELD_GOALS_ATTEMPTED"],
    "3P":  ["3P","3PM","FG3M","THREES_MADE","3PTM","FG3-MADE"],
    "FTA": ["FTA","FT_ATT","FREE_THROWS_ATTEMPTED"],
    "TOV": ["TOV","TO","TURNOVERS"],
    "ORB": ["ORB","OREB","OFFENSIVE_REBOUNDS"],
    "DRB": ["DRB","DREB","DEFENSIVE_REBOUNDS"],
    "PTS": ["PTS","POINTS","SCORE"]
}

def _find_col(df: pd.DataFrame, stat_key: str, side: str) -> pd.Series|None:
    side = side.lower(); assert side in ("home","away")
    side_caps = "HOME" if side=="home" else "AWAY"
    aliases = _ALIASES.get(stat_key, [stat_key])

    # candidates to try
    candidates = []
    for a in aliases:
        candidates += [f"{a}_{side}", f"{a}_{side_caps}", f"{side}_{a}", f"{side_caps}_{a}",
                       f"{a}{'H' if side=='home' else 'A'}"]
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return df[lower_map[c.lower()]]
    # fallback: contains search
    for col in df.columns:
        cl = col.lower()
        if any(a.lower() in cl for a in aliases) and (side in cl or side_caps.lower() in cl):
            return df[col]
    return None

# ---------- Four Factors ----------
def _four_factors_block(df: pd.DataFrame, side: str, opp: str, prefix: str):
    FG  = _find_col(df, "FG",  side)
    FGA = _find_col(df, "FGA", side)
    P3  = _find_col(df, "3P",  side)
    FTA = _find_col(df, "FTA", side)
    TOV = _find_col(df, "TOV", side)
    ORB = _find_col(df, "ORB", side)
    DRB_opp = _find_col(df, "DRB", opp)
    if any(v is None for v in (FG,FGA,P3,FTA,TOV,ORB,DRB_opp)):
        return pd.DataFrame(index=df.index)  # graceful no-op

    def sdiv(num, den):
        den = den.replace(0, np.nan) if hasattr(den, "replace") else den
        return num / den

    efg = sdiv(FG + 0.5 * P3, FGA)
    tov = sdiv(TOV, (FGA + 0.44 * FTA + TOV))
    orb = sdiv(ORB, (ORB + DRB_opp))
    ftr = sdiv(FTA, FGA)
    return pd.DataFrame({
        f"{prefix}_eFG": efg, f"{prefix}_TOVp": tov, f"{prefix}_ORBp": orb, f"{prefix}_FTr": ftr
    }, index=df.index)

def add_four_factors(df_games: pd.DataFrame) -> pd.DataFrame:
    df = df_games.copy()
    home = _four_factors_block(df, "home", "away", "home")
    away = _four_factors_block(df, "away", "home", "away")
    if home.empty and away.empty:
        return df
    return pd.concat([df, home, away], axis=1)

# ---------- Possessions / Pace ----------
def add_possessions_and_pace(df_games: pd.DataFrame) -> pd.DataFrame:
    df = df_games.copy()
    FGA_h = _find_col(df, "FGA", "home"); FGA_a = _find_col(df, "FGA", "away")
    FTA_h = _find_col(df, "FTA", "home"); FTA_a = _find_col(df, "FTA", "away")
    ORB_h = _find_col(df, "ORB", "home"); ORB_a = _find_col(df, "ORB", "away")
    TOV_h = _find_col(df, "TOV", "home"); TOV_a = _find_col(df, "TOV", "away")
    if None in (FGA_h,FTA_h,ORB_h,TOV_h,FGA_a,FTA_a,ORB_a,TOV_a):
        return df
    poss_home = (FGA_h + 0.44*FTA_h - ORB_h + TOV_h)
    poss_away = (FGA_a + 0.44*FTA_a - ORB_a + TOV_a)
    df["poss_est"] = 0.5 * (poss_home + poss_away)
    df["pace_est"] = df["poss_est"]
    return df

# ---------- Rolling windows ----------
def add_rolling_features(df_games: pd.DataFrame, windows=(3,5,10,20), by_home_away=True) -> pd.DataFrame:
    df = df_games.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    base_candidates = ["PTS","FG","FGA","3P","FTA","TOV","ORB","DRB","AST","TRB"]

    def available_stats(side):
        return [b for b in base_candidates if _find_col(df, b, side) is not None]

    base_cols = sorted(set(available_stats("home")) | set(available_stats("away")))

    def to_long(side):
        rows = {"GAME_ID": df["GAME_ID"], "GAME_DATE": df["GAME_DATE"],
                "TEAM_ABBREV": df[f"{side.upper()}_TEAM_ABBREV"], "IS_HOME": int(side=="home")}
        for b in base_cols:
            col = _find_col(df, b, side)
            if col is not None:
                rows[b] = col
        return pd.DataFrame(rows, index=df.index)

    long_df = pd.concat([to_long("home"), to_long("away")], axis=0, ignore_index=True).sort_values(["TEAM_ABBREV","GAME_DATE"])

    for w in windows:
        roll = (long_df.groupby("TEAM_ABBREV")[base_cols]
                .apply(lambda g: g.shift(1).rolling(w, min_periods=max(1,w//2)).mean())
                .reset_index(level=0, drop=True))
        roll.columns = [f"roll{w}_{c}" for c in roll.columns]
        long_df = pd.concat([long_df, roll], axis=1)

    def back(side):
        sub = long_df[long_df["IS_HOME"].eq(1 if side=="home" else 0)].drop(columns=["IS_HOME"])
        suff = "_home" if side=="home" else "_away"
        sub = sub.add_suffix(suff).rename(columns={f"TEAM_ABBREV{suff}": f"{side.upper()}_TEAM_ABBREV",
                                                   f"GAME_ID{suff}": "GAME_ID"})
        return sub

    home_w = back("home"); away_w = back("away")
    return (df.merge(home_w, on=["GAME_ID","HOME_TEAM_ABBREV"], how="left")
              .merge(away_w, on=["GAME_ID","AWAY_TEAM_ABBREV"], how="left"))

# ---------- Opponent-adjusted diffs ----------
def opponent_adjusted(df: pd.DataFrame, stat_prefix="roll10", cols=("PTS","eFG","TOVp","ORBp","FTr")) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c == "eFG":
            h, a = f"{stat_prefix}_eFG_home", f"{stat_prefix}_eFG_away"
        elif c == "TOVp":
            h, a = f"{stat_prefix}_TOVp_home", f"{stat_prefix}_TOVp_away"
        elif c == "ORBp":
            h, a = f"{stat_prefix}_ORBp_home", f"{stat_prefix}_ORBp_away"
        elif c == "FTr":
            h, a = f"{stat_prefix}_FTr_home", f"{stat_prefix}_FTr_away"
        else:
            h, a = f"{stat_prefix}_{c}_home", f"{stat_prefix}_{c}_away"
        if h in out.columns and a in out.columns:
            out[f"{stat_prefix}_{c}_diff"] = out[h] - out[a]
    return out

# ---------- Rest & Travel ----------
def add_rest_travel(df_games: pd.DataFrame, team_coords: pd.DataFrame) -> pd.DataFrame:
    """
    Adds for each game:
      - days_rest_home / days_rest_away
      - is_b2b_home / is_b2b_away
      - travel_km_home / travel_km_away  (distance from previous opponent arena to current opponent arena)
      - altitude_home_m (if provided in team_coords)
    Requires: GAME_ID, GAME_DATE, HOME_TEAM_ABBREV, AWAY_TEAM_ABBREV in df_games,
              TEAM_ABBREV, arena_lat, arena_lon[, altitude_m] in team_coords.
    """
    req = {"GAME_ID","GAME_DATE","HOME_TEAM_ABBREV","AWAY_TEAM_ABBREV"}
    if not req.issubset(df_games.columns) or not {"TEAM_ABBREV","arena_lat","arena_lon"}.issubset(team_coords.columns):
        return df_games

    df = df_games.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    def per_team_features(team_side):
        team_col = f"{team_side}_TEAM_ABBREV"
        opp_col  = "AWAY_TEAM_ABBREV" if team_side == "HOME" else "HOME_TEAM_ABBREV"
        tmp = df[["GAME_ID","GAME_DATE", team_col, opp_col]].copy().sort_values([team_col,"GAME_DATE"])

        tmp["prev_date"] = tmp.groupby(team_col)["GAME_DATE"].shift(1)
        tmp[f"days_rest_{team_side.lower()}"] = (tmp["GAME_DATE"] - tmp["prev_date"]).dt.days
        tmp[f"is_b2b_{team_side.lower()}"] = (tmp[f"days_rest_{team_side.lower()}"] == 1).astype(int)

        # previous opponent location -> current opponent location
        tc_opp = team_coords.rename(columns={"TEAM_ABBREV": opp_col, "arena_lat":"opp_lat", "arena_lon":"opp_lon"})
        tmp = tmp.merge(tc_opp[[opp_col,"opp_lat","opp_lon"]], on=opp_col, how="left")
        tmp["prev_lat"] = tmp.groupby(team_col)["opp_lat"].shift(1)
        tmp["prev_lon"] = tmp.groupby(team_col)["opp_lon"].shift(1)
        tmp[f"travel_km_{team_side.lower()}"] = np.where(tmp["prev_lat"].notna(),
            [haversine_km(tmp["prev_lat"].iloc[i], tmp["prev_lon"].iloc[i], tmp["opp_lat"].iloc[i], tmp["opp_lon"].iloc[i])
             for i in range(len(tmp))], np.nan)

        return tmp[["GAME_ID", f"days_rest_{team_side.lower()}",
                    f"is_b2b_{team_side.lower()}", f"travel_km_{team_side.lower()}"]]

    h = per_team_features("HOME")
    a = per_team_features("AWAY")

    out = (df.merge(h, on="GAME_ID", how="left")
             .merge(a, on="GAME_ID", how="left"))

    if "altitude_m" in team_coords.columns:
        out = out.merge(team_coords[["TEAM_ABBREV","altitude_m"]]
                        .rename(columns={"TEAM_ABBREV":"HOME_TEAM_ABBREV","altitude_m":"altitude_home_m"}),
                        on="HOME_TEAM_ABBREV", how="left")
    return out
