# utils.py — analytics helpers
import numpy as np
import pandas as pd
from config import STRIKE_STEP, NEAR_STRIKES

def nearest_strike(price: float, step: int = STRIKE_STEP) -> int:
    return int(round(price / step) * step)

def classify_buildup(oi_change: float, ltp_change: float) -> str:
    if oi_change > 0 and ltp_change > 0:
        return "Long Buildup"
    if oi_change > 0 and ltp_change < 0:
        return "Short Buildup"
    if oi_change < 0 and ltp_change < 0:
        return "Long Unwinding"
    if oi_change < 0 and ltp_change > 0:
        return "Short Covering"
    return "Neutral"

def enrich_with_prev(curr: pd.DataFrame, prev: pd.DataFrame | None) -> pd.DataFrame:
    df = curr.copy()
    if prev is None or prev.empty:
        df["prev_ltp"] = df["ltp"]
        df["prev_oi"] = df["oi"]
    else:
        m = prev[["symbol","strike","option_type","ltp","oi"]].rename(columns={"ltp":"prev_ltp","oi":"prev_oi"})
        df = df.merge(m, on=["symbol","strike","option_type"], how="left")
        df["prev_ltp"] = df["prev_ltp"].fillna(df["ltp"])
        df["prev_oi"] = df["prev_oi"].fillna(df["oi"])
    df["oi_chg"] = df["oi"] - df["prev_oi"]
    df["ltp_chg"] = df["ltp"] - df["prev_ltp"]
    df["oi_chg_pct"] = np.where(df["prev_oi"]>0, 100*df["oi_chg"]/df["prev_oi"], 0.0)
    df["ltp_chg_pct"] = np.where(df["prev_ltp"]>0, 100*df["ltp_chg"]/df["prev_ltp"], 0.0)
    df["buildup"] = [classify_buildup(o, p) for o, p in zip(df["oi_chg"], df["ltp_chg"])]
    df["above_vwap"] = df["ltp"] > df["vwap"]
    return df

def select_near_atm(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    if df.empty:
        return df
    atm = nearest_strike(spot, STRIKE_STEP)
    lo, hi = atm - NEAR_STRIKES*STRIKE_STEP, atm + NEAR_STRIKES*STRIKE_STEP
    return df[(df["strike"]>=lo) & (df["strike"]<=hi)].copy()

def compute_crossover(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (symbol, strike), g in df.groupby(["symbol","strike"]):
        ce = g[g.option_type=="CE"]["ltp"].values
        pe = g[g.option_type=="PE"]["ltp"].values
        if ce.size and pe.size:
            ce_val, pe_val = ce[0], pe[0]
            out.append({
                "symbol": symbol,
                "strike": int(strike),
                "ce_gt_pe": bool(ce_val > pe_val),
                "pe_gt_ce": bool(pe_val > ce_val),
                "diff_pct": float(((ce_val - pe_val)/max(1e-6, pe_val))*100),
            })
    return pd.DataFrame(out)
