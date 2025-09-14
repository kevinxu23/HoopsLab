#!/usr/bin/env python
from __future__ import annotations
import os, re, gc, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from typing import Iterable, Tuple
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import log_loss

# ----------------- Config (override via env) -----------------
DATA_GAMES = os.environ.get("DATA_GAMES", "data/games.csv")
ODDS_CSV    = os.environ.get("ODDS_CSV", "")     # optional
DATE_MIN    = os.environ.get("DATE_MIN", "2012-10-01")
N_FOLDS     = int(os.environ.get("N_FOLDS", "3"))
N_TREES     = int(os.environ.get("N_TREES", "800"))
CAL_FRAC    = float(os.environ.get("CAL_FRAC", "0.2"))
SEED        = int(os.environ.get("SEED", "42"))

# ----------------- Schema helpers -----------------
ID2ABBR = {
    1610612737:"ATL",1610612738:"BOS",1610612739:"CLE",1610612740:"NOP",
    1610612741:"CHI",1610612742:"DAL",1610612743:"DEN",1610612744:"GSW",
    1610612745:"HOU",1610612746:"LAC",1610612747:"LAL",1610612748:"MIA",
    1610612749:"MIL",1610612750:"MIN",1610612751:"BKN",1610612752:"NYK",
    1610612753:"ORL",1610612754:"IND",1610612755:"PHI",1610612756:"PHX",
    1610612757:"POR",1610612758:"SAC",1610612759:"SAS",1610612760:"OKC",
    1610612761:"TOR",1610612762:"UTA",1610612763:"MEM",1610612764:"WAS",
    1610612765:"DET",1610612766:"CHA",
}

def pick_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    for c in df.columns:
        lc = c.lower()
        if any(k.lower() in lc for k in candidates): return c
    return None

def normalize_games_schema(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Date
    date_col = pick_col(df, ["GAME_DATE","GAME_DATE_EST","date","game_date","commence_time","start_time","DATE"])
    if not date_col: raise ValueError("No date column in games.csv")
    df = df.rename(columns={date_col:"GAME_DATE"})
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.tz_localize(None)

    # IDs
    h_id = pick_col(df, ["HOME_TEAM_ID","home_team_id"])
    a_id = pick_col(df, ["AWAY_TEAM_ID","VISITOR_TEAM_ID","away_team_id","visitor_team_id"])
    if not (h_id and a_id): raise ValueError("Need HOME_TEAM_ID and VISITOR/AWAY_TEAM_ID")
    df = df.rename(columns={h_id:"HOME_TEAM_ID", a_id:"AWAY_TEAM_ID"})
    df["HOME_TEAM_ABBREV"] = df["HOME_TEAM_ID"].apply(lambda x: ID2ABBR.get(int(x)) if pd.notnull(x) else None)
    df["AWAY_TEAM_ABBREV"] = df["AWAY_TEAM_ID"].apply(lambda x: ID2ABBR.get(int(x)) if pd.notnull(x) else None)

    # Target
    wins_col = pick_col(df, ["HOME_TEAM_WINS","home_team_wins","HOME_WIN","home_win"])
    if wins_col:
        df["home_team_wins"] = pd.to_numeric(df[wins_col], errors="coerce").astype("Int64")
    else:
        hp = pick_col(df, ["PTS_home","HOME_PTS","home_score"])
        ap = pick_col(df, ["PTS_away","AWAY_PTS","visitor_pts","away_score"])
        if hp and ap:
            df = df.rename(columns={hp:"PTS_home", ap:"PTS_away"})
            df["home_team_wins"] = (pd.to_numeric(df["PTS_home"], errors="coerce") >
                                    pd.to_numeric(df["PTS_away"], errors="coerce")).astype("Int64")
        else:
            raise SystemExit("No HOME_TEAM_WINS or points available for target.")

    # GAME_ID
    gid = pick_col(df, ["GAME_ID","game_id","id","gamecode"])
    if gid and gid != "GAME_ID": df = df.rename(columns={gid:"GAME_ID"})
    if "GAME_ID" not in df.columns:
        df["GAME_ID"] = (df["GAME_DATE"].dt.strftime("%Y%m%d") + "_" +
                         df["AWAY_TEAM_ABBREV"] + "@" + df["HOME_TEAM_ABBREV"] + "_" +
                         df.reset_index().index.astype(str))
    return df

# ----------------- No-leak feature builders -----------------
STAT_MAP = {  # stat name → (home_col, away_col)
    "PTS": ("PTS_home","PTS_away"),
    "FG_PCT": ("FG_PCT_home","FG_PCT_away"),
    "FT_PCT": ("FT_PCT_home","FT_PCT_away"),
    "FG3_PCT": ("FG3_PCT_home","FG3_PCT_away"),
    "AST": ("AST_home","AST_away"),
    "REB": ("REB_home","REB_away"),
}

def add_shifted_rolling(df: pd.DataFrame, windows=(3,5,10)) -> pd.DataFrame:
    # Build long team-series with lag1 to exclude current game, then rolling on lagged values
    keep = ["GAME_ID","GAME_DATE","HOME_TEAM_ABBREV","AWAY_TEAM_ABBREV"]
    have = {k: v for k,v in STAT_MAP.items() if v[0] in df.columns and v[1] in df.columns}
    base = df[keep + [v for pair in have.values() for v in pair]].copy()

    def side_frame(side: str):
        rows = {
            "TEAM": base[f"{side.upper()}_TEAM_ABBREV"],
            "GAME_ID": base["GAME_ID"],
            "GAME_DATE": base["GAME_DATE"],
        }
        for stat,(hcol,acol) in have.items():
            col = hcol if side=="home" else acol
            rows[stat] = pd.to_numeric(base[col], errors="coerce")
        return pd.DataFrame(rows)

    long = pd.concat([side_frame("home"), side_frame("away")], axis=0, ignore_index=True)
    long = long.sort_values(["TEAM","GAME_DATE"]).reset_index(drop=True)

    # lag1 per team then rolling means over the lagged series
    for stat in have.keys():
        long[f"{stat}_lag1"] = long.groupby("TEAM")[stat].shift(1)
        for w in windows:
            long[f"{stat}_roll{w}"] = (
                long.groupby("TEAM")[stat]
                    .shift(1)                                    # exclude current game
                    .rolling(w, min_periods=max(1,w//2))        # require some history
                    .mean()
                    .reset_index(level=0, drop=True)
            )

    # Join back for both teams
    home_join = long.add_prefix("home_")
    away_join = long.add_prefix("away_")
    df = df.merge(home_join, left_on=["HOME_TEAM_ABBREV","GAME_ID"],
                  right_on=["home_TEAM","home_GAME_ID"], how="left")
    df = df.merge(away_join, left_on=["AWAY_TEAM_ABBREV","GAME_ID"],
                  right_on=["away_TEAM","away_GAME_ID"], how="left")

    # Clean merge helper cols
    drop_cols = [c for c in df.columns if c.endswith(("home_TEAM","home_GAME_ID","home_GAME_DATE","away_TEAM","away_GAME_ID","away_GAME_DATE"))]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Make diffs for rolled stats only (not current-game stats)
    for stat in have.keys():
        for w in windows:
            h = f"home_{stat}_roll{w}"
            a = f"away_{stat}_roll{w}"
            if h in df.columns and a in df.columns:
                df[f"{stat}_roll{w}_diff"] = df[h] - df[a]
    return df

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    last_date = {}
    rest_h, rest_a = [], []
    for _, r in df.iterrows():
        d = r["GAME_DATE"]
        h, a = r["HOME_TEAM_ABBREV"], r["AWAY_TEAM_ABBREV"]
        rest_h.append((d - last_date[h]).days if h in last_date else np.nan)
        rest_a.append((d - last_date[a]).days if a in last_date else np.nan)
        last_date[h], last_date[a] = d, d
    df["rest_home"] = rest_h
    df["rest_away"] = rest_a
    df["rest_diff"] = df["rest_home"].fillna(0) - df["rest_away"].fillna(0)
    return df

def add_pre_game_elo(df: pd.DataFrame, k: float = 20.0) -> pd.DataFrame:
    # Simple Elo computed sequentially; attach PRE-game ratings (no leakage)
    df = df.sort_values("GAME_DATE").reset_index(drop=True)
    r = {t:1500.0 for t in pd.unique(df[["HOME_TEAM_ABBREV","AWAY_TEAM_ABBREV"]].values.ravel())}
    elo_pre_h, elo_pre_a = [], []
    for _, row in df.iterrows():
        h, a = row["HOME_TEAM_ABBREV"], row["AWAY_TEAM_ABBREV"]
        eh, ea = r[h], r[a]
        elo_pre_h.append(eh); elo_pre_a.append(ea)
        p = 1.0/(1.0 + 10**((ea-eh)/400.0))
        y = float(row["home_team_wins"])
        r[h] = eh + k*(y - p)
        r[a] = ea + k*((1.0-y) - (1.0-p))
    df["elo_pre_home"] = np.array(elo_pre_h, dtype="float32")
    df["elo_pre_away"] = np.array(elo_pre_a, dtype="float32")
    df["elo_pre_diff"] = df["elo_pre_home"] - df["elo_pre_away"]
    return df

def attach_odds_features(df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
    """Optional: expects either GAME_ID join or (GAME_DATE, HOME_TEAM_ID, AWAY_TEAM_ID) with moneyline/prob."""
    o = odds_df.copy()
    if "GAME_ID" in o.columns:
        key = ["GAME_ID"]
    else:
        if "GAME_DATE" in o.columns:
            o["GAME_DATE"] = pd.to_datetime(o["GAME_DATE"]).dt.tz_localize(None)
        if {"GAME_DATE","HOME_TEAM_ID","AWAY_TEAM_ID"}.issubset(o.columns):
            key = ["GAME_DATE","HOME_TEAM_ID","AWAY_TEAM_ID"]
        else:
            return df
    def ml_to_prob(ml):
        try:
            ml = float(ml)
            return 100./(ml+100.) if ml>0 else (-ml)/(-ml+100.)
        except: return np.nan
    for c in ["home_ml","HOME_ML","moneyline_home","home_moneyline"]:
        if c in o.columns: o["fair_home_prob"] = o[c].apply(ml_to_prob)
    for c in ["home_prob","HOME_PROB","implied_home_prob"]:
        if c in o.columns: o["fair_home_prob"] = pd.to_numeric(o[c], errors="coerce")
    cols = key + ([ "fair_home_prob" ] if "fair_home_prob" in o.columns else [])
    if len(cols) == len(key): return df
    o = o[cols].drop_duplicates()
    return df.merge(o, on=key, how="left")

# ----------------- CV splitter -----------------
def walk_forward_splits(df: pd.DataFrame, date_col="GAME_DATE", n_folds=2) -> Iterable[Tuple[np.ndarray,np.ndarray]]:
    df = df.sort_values(date_col).reset_index(drop=True)
    borders = np.linspace(0.6, 0.9, n_folds)
    idx = np.arange(len(df))
    for b in borders:
        cut = int(len(df)*b)
        te_end = min(len(df), cut + max(1, int(len(df)*0.1)))
        tr = idx[:cut]; te = idx[cut:te_end]
        if len(tr)==0 or len(te)==0: continue
        yield tr, te

# add here
def tune_alpha(y, p_model, p_odds):
    """Choose alpha∈[0,1] to minimize log-loss on rows that have odds."""
    import numpy as np
    mask = ~np.isnan(p_odds)
    if not np.any(mask):
        return 0.0
    y_ = y[mask]; pm = p_model[mask]; po = p_odds[mask]
    best = (1e9, 0.0)
    for a in np.linspace(0.0, 1.0, 21):  # 0.05 steps
        p = np.clip(a*po + (1-a)*pm, 1e-6, 1-1e-6)
        ll = log_loss(y_, p)
        if ll < best[0]: best = (ll, a)
    return float(best[1])

def pick_cal_slice_with_odds(tr_idx, odds_series, min_odds=100, frac=0.2):
    """
    Take a tail slice of train indices but ensure it contains at least `min_odds` rows with odds.
    If not enough, expand the slice size (up to 50% of train) to find enough odds rows.
    """
    import numpy as np
    n = len(tr_idx)
    cal_size = max(1, int(n*frac))
    # expand until min_odds or cap at 50%
    while cal_size < int(0.5*n):
        idx = tr_idx[-cal_size:]
        has = (~np.isnan(odds_series.iloc[idx])).sum()
        if has >= min_odds:
            return tr_idx[:-cal_size], tr_idx[-cal_size:]
        cal_size = int(cal_size * 1.5)  # grow window
    # final attempt
    idx = tr_idx[-cal_size:]
    return tr_idx[:-cal_size], tr_idx[-cal_size:]

# ----------------- Training -----------------
def main():
    g = normalize_games_schema(DATA_GAMES)
    g = g[g["home_team_wins"].notna()].copy()
    g = g[g["GAME_DATE"] >= pd.to_datetime(DATE_MIN)].reset_index(drop=True)

    # STRICTLY pre-game features you already had
    g = add_shifted_rolling(g, windows=(3,5,10))
    g = add_rest_days(g)
    g = add_pre_game_elo(g)

    # ---- Attach odds once (expects a column 'fair_home_prob' after attach) ----
    if ODDS_CSV and os.path.exists(ODDS_CSV):
        try:
            g = attach_odds_features(g, pd.read_csv(ODDS_CSV))
        except Exception as e:
            print(f"[warn] odds attach failed: {e}")
            g["fair_home_prob"] = np.nan
    else:
        g["fair_home_prob"] = np.nan

    # >>> paste this right after you attach odds
    odds_dates = g.loc[g["fair_home_prob"].notna(), "GAME_DATE"]
    if len(odds_dates):
        DATE_MAX = odds_dates.max()
    g = g[g["GAME_DATE"] <= DATE_MAX].reset_index(drop=True)
    print(f"[info] Clipped games to <= {DATE_MAX.date()} so every test fold has odds.")
    # Show overall odds coverage for sanity
    cov = g["fair_home_prob"].notna().mean()
    print(f"[info] Odds coverage overall: {cov:.1%}")

    # ---- Build X: ONLY your team/rolling diffs; DO NOT include odds here ----
    # We’ll use odds purely as a post-hoc blend prior to avoid leakage/ffill issues.
    feat = [c for c in g.columns if c.endswith("_diff") and ("roll" in c or c=="rest_diff" or c=="elo_pre_diff")]
    X = g[feat].astype("float32").copy()
    # fill only non-odds features (since odds not included in X)
    X = X.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    y = g["home_team_wins"].astype(int).values

    results, y_true_all, y_prob_all = [], [], []
    alphas = []

    for fold,(tr,te) in enumerate(walk_forward_splits(g, n_folds=N_FOLDS), start=1):
        print(f"[Fold {fold}/{N_FOLDS}] train={len(tr)} test={len(te)}")

        # --- choose a calibration tail that has enough odds rows ---
        tr_idx, cal_idx = pick_cal_slice_with_odds(tr, g["fair_home_prob"], min_odds=100, frac=CAL_FRAC)
        n_cal_odds = (~np.isnan(g["fair_home_prob"].iloc[cal_idx])).sum()
        n_te_odds  = (~np.isnan(g["fair_home_prob"].iloc[te])).sum()
        print(f"   cal_size={len(cal_idx)} (with_odds={n_cal_odds}) | test_with_odds={n_te_odds}/{len(te)}")

        mdl = LGBMClassifier(
            n_estimators=N_TREES, num_leaves=63, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.8, random_state=SEED, n_jobs=2,
            min_data_in_leaf=25
        )
        mdl.fit(X.iloc[tr_idx], y[tr_idx])

        cal = CalibratedClassifierCV(estimator=mdl, method="isotonic", cv="prefit")
        cal.fit(X.iloc[cal_idx], y[cal_idx])

        # Model probabilities
        p_te  = cal.predict_proba(X.iloc[te])[:,1]
        p_cal = cal.predict_proba(X.iloc[cal_idx])[:,1]
        y_te  = y[te]
        y_cal = y[cal_idx]

        # ---- Blend with odds (alpha tuned on calibration slice) ----
        p_odds_cal = g["fair_home_prob"].iloc[cal_idx].to_numpy()
        p_odds_te  = g["fair_home_prob"].iloc[te].to_numpy()

        alpha = tune_alpha(y_cal, p_cal, p_odds_cal)
        # Fallback: if no odds in cal but there are odds in test, use a conservative default
        if np.isnan(p_odds_cal).all() and (~np.isnan(p_odds_te)).any():
            alpha = 0.7  # reasonable default weighting towards odds given your odds-only AUC≈0.74

        has_odds_te = ~np.isnan(p_odds_te)
        p_blend = p_te.copy()
        p_blend[has_odds_te] = alpha*p_odds_te[has_odds_te] + (1-alpha)*p_te[has_odds_te]

        # Metrics: model vs blended vs odds-only (on rows that have odds)
        auc_model = roc_auc_score(y_te, p_te)
        auc_blend = roc_auc_score(y_te, p_blend)
        acc_blend = accuracy_score(y_te, (p_blend>=0.5).astype(int))
        msg = f"   alpha={alpha:.2f}  odds_used={has_odds_te.mean():.1%}  AUC(model)={auc_model:.3f}  AUC(blend)={auc_blend:.3f}  ACC(blend)={acc_blend:.3f}"

        # Optional: compute odds-only AUC on the test fold for reference
        if has_odds_te.any():
            from sklearn.metrics import roc_auc_score as _auc
            msg += f"  |  AUC(odds_only)={_auc(y_te[has_odds_te], p_odds_te[has_odds_te]):.3f}"
        print(msg)

        results.append({"fold":fold, "auc":auc_blend, "acc":acc_blend})
        y_true_all.append(y_te); y_prob_all.append(p_blend)
        alphas.append(alpha)

        del mdl, cal; gc.collect()

    res = pd.DataFrame(results)
    print(res); print("Mean:", res.mean(numeric_only=True).to_dict())
    print("Chosen alphas by fold:", [round(a,2) for a in alphas], " | mean alpha:", round(float(np.mean(alphas)),2))

    # Reliability plot
    try:
        from src.plus.metrics_extra import reliability_plot
        fig, ax = plt.subplots(figsize=(5,5))
        all_true = np.concatenate(y_true_all); all_prob = np.concatenate(y_prob_all)
        reliability_plot(ax, all_true, all_prob, n_bins=10, label="Calibrated LGBM (blended)")
        Path("models").mkdir(exist_ok=True)
        fig.tight_layout(); fig.savefig("models/reliability.png", dpi=160)
    except Exception as e:
        print(f"[warn] reliability plot skipped: {e}")

    Path("models/FINISHED.txt").write_text("finished\n")
    print("✅ TRAINING FINISHED — saved models/FINISHED.txt (and reliability.png if plotting succeeded)")
    print("new")

if __name__ == "__main__":
    main()
