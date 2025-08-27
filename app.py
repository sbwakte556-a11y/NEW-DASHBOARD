# app.py — NIFTY/BANKNIFTY Options Live Dashboard (Streamlit Cloud ready)

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import pytz
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Local modules
from nse_fetch import fetch_option_chain  # must return a DataFrame with columns listed below
from config import STRIKE_STEP, NEAR_STRIKES, REFRESH_SECS, TIMEZONE

# ------------------------
# Utilities
# ------------------------

IST = pytz.timezone(TIMEZONE if TIMEZONE else "Asia/Kolkata")

def safe_val(row, col, default=0.0):
    """
    Safely extract a scalar float from row[col].
    Supports dict, pandas Series, and single-row DataFrames.
    Returns `default` if missing/NaN; if a Series/array is found, uses the first non-NaN scalar.
    """
    try:
        if row is None:
            return float(default)
        # dict-like
        if isinstance(row, dict):
            v = row.get(col, default)
        # pandas Series
        elif hasattr(row, "index") and not hasattr(row, "columns"):
            v = row[col] if col in row.index else default
        # pandas DataFrame (take the first row's value for this column)
        elif hasattr(row, "columns"):
            if col in row.columns and len(row) > 0:
                v = row.iloc[0][col]
            else:
                v = default
        else:
            v = getattr(row, col, default)
        # If v is Series/array, take first valid
        if hasattr(v, "__array__") or str(type(v)).endswith("Series'>"):
            try:
                v = next((x for x in np.ravel(v) if pd.notna(x)), default)
            except Exception:
                v = default
        return float(v) if (v is not None and pd.notna(v)) else float(default)
    except Exception:
        return float(default)

def get_latest_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the latest timestamp slice (if 'ts' exists); else return df."""
    if df is None or df.empty:
        return pd.DataFrame()
    if "ts" in df.columns:
        tmax = pd.to_datetime(df["ts"]).max()
        return df[pd.to_datetime(df["ts"]) == tmax].copy()
    return df.copy()

def get_expiries(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "expiry" not in df.columns:
        return []
    exp = sorted(pd.Series(df["expiry"].astype(str).unique()).tolist())
    return exp

def get_atm_strike(spot: float, step: int) -> int | float:
    try:
        return int(round(spot / step) * step)
    except Exception:
        return np.nan

def subset_atm_band(df_wide: pd.DataFrame, atm: float, k: int, step: int) -> pd.DataFrame:
    if pd.isna(atm) or "strike" not in df_wide.columns:
        return df_wide.copy()
    lo, hi = atm - k * step, atm + k * step
    return df_wide[(df_wide["strike"] >= lo) & (df_wide["strike"] <= hi)].copy()

def build_atm_band_table(wide_latest: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strike", "ce_oi", "pe_oi", "ce_oi_chg", "pe_oi_chg",
        "ce_ltp", "pe_ltp", "ce_buildup", "pe_buildup"
    ]
    have = [c for c in cols if c in wide_latest.columns]
    tbl = wide_latest[have].copy()

    # per-strike PCR
    if "ce_oi" in tbl.columns and "pe_oi" in tbl.columns:
        tbl["pcr"] = (tbl["pe_oi"] / tbl["ce_oi"]).replace([np.inf, -np.inf], np.nan)

    # Pretty names
    rename_map = {
        "ce_oi": "CE OI", "pe_oi": "PE OI",
        "ce_oi_chg": "CE OI Δ", "pe_oi_chg": "PE OI Δ",
        "ce_ltp": "CE LTP", "pe_ltp": "PE LTP",
        "ce_buildup": "CE Buildup", "pe_buildup": "PE Buildup",
        "pcr": "PCR"
    }
    disp = tbl.rename(columns=rename_map)

    totals = {"strike": "TOTAL"}
    for k in ["CE OI", "PE OI", "CE OI Δ", "PE OI Δ"]:
        if k in disp.columns:
            totals[k] = disp[k].sum()
    disp_total = pd.concat([disp, pd.DataFrame([totals])], ignore_index=True)
    return disp_total

def top_trending_strikes(wide_latest: pd.DataFrame, n=8) -> pd.DataFrame:
    needed = {"strike", "ce_oi_chg", "pe_oi_chg"}
    if not needed.issubset(set(wide_latest.columns)):
        return pd.DataFrame()
    t = wide_latest[["strike", "ce_oi_chg", "pe_oi_chg"]].copy()
    t["abs_oi_chg"] = t["ce_oi_chg"].fillna(0).abs() + t["pe_oi_chg"].fillna(0).abs()
    t = t.sort_values("abs_oi_chg", ascending=False).head(n)
    return t.rename(columns={"ce_oi_chg": "CE OI Δ", "pe_oi_chg": "PE OI Δ", "abs_oi_chg": "Total |ΔOI|"})

def strikewise_pcr(wide_latest: pd.DataFrame) -> pd.DataFrame:
    needed = {"strike", "ce_oi", "pe_oi"}
    if not needed.issubset(set(wide_latest.columns)):
        return pd.DataFrame()
    df = wide_latest[["strike", "ce_oi", "pe_oi"]].copy()
    df["PCR"] = (df["pe_oi"] / df["ce_oi"]).replace([np.inf, -np.inf], np.nan)
    return df.rename(columns={"ce_oi": "CE OI", "pe_oi": "PE OI"})

def session_vwap(history_wide: pd.DataFrame, side="ce") -> pd.Series:
    price_col = f"{side}_ltp"
    vol_col = f"{side}_vol" if f"{side}_vol" in history_wide.columns else None
    if vol_col is None or price_col not in history_wide.columns:
        return pd.Series(dtype=float)
    hv = history_wide.dropna(subset=["strike"]).copy()
    hv["pv"] = hv[price_col].fillna(0) * hv[vol_col].fillna(0)
    grp = hv.groupby("strike", as_index=True)
    vwap = grp["pv"].sum() / grp[vol_col].sum().replace(0, np.nan)
    return vwap

def last_3min_delta(history_wide: pd.DataFrame) -> dict:
    if history_wide is None or history_wide.empty or "ts" not in history_wide.columns:
        return {}
    h = history_wide.sort_values("ts")
    last = h["ts"].max()
    prev = h[h["ts"] < last]["ts"].max()
    if pd.isna(prev):
        return {}
    d = {}
    for col in ["ce_oi", "pe_oi", "ce_oi_chg", "pe_oi_chg"]:
        if col in h.columns:
            v_last = h[h["ts"] == last][col].sum()
            v_prev = h[h["ts"] == prev][col].sum()
            d[col] = float(v_last - v_prev)
    return d

def score_buyer_seller(buildup: str) -> tuple[float, float]:
    """
    Simple scoring:
      - Long Buildup    -> Buyers +1.0
      - Short Covering  -> Buyers +0.5
      - Short Buildup   -> Sellers +1.0
      - Long Unwinding  -> Sellers +0.5
    """
    if not isinstance(buildup, str):
        return 0.0, 0.0
    b = buildup.strip().lower()
    if "long" in b and "buildup" in b:
        return 1.0, 0.0
    if "short" in b and "cover" in b:
        return 0.5, 0.0
    if "short" in b and "buildup" in b:
        return 0.0, 1.0
    if "long" in b and "unwind" in b:
        return 0.0, 0.5
    return 0.0, 0.0

# ------------------------
# Sidebar / Settings
# ------------------------

st.set_page_config(page_title="NSE Options Live", layout="wide")

st.sidebar.title("Settings")
SYMBOL = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
refresh_secs = st.sidebar.number_input("Auto-refresh (seconds)", min_value=30, max_value=900, value=REFRESH_SECS, step=30)
near_strikes = st.sidebar.slider("Strikes near ATM (±)", 1, 6, NEAR_STRIKES)
oi_alert_pct = st.sidebar.slider("Exceptional OI% threshold", 5, 500, 50)
st.sidebar.markdown("---")
st.sidebar.button("🔄 Refresh now", on_click=lambda: st.cache_data.clear())
autorefresh_on = st.sidebar.toggle("Auto refresh", value=True)
if autorefresh_on:
    st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh")

# ------------------------
# Title / Market hours
# ------------------------

st.title(f"📈 {SYMBOL} Options Live — Online Dashboard")
_now_ist = dt.datetime.now(IST).time()
if not (dt.time(9, 14) <= _now_ist <= dt.time(15, 31)):
    st.info("Market appears closed. Data may be stale or unchanged.")

status = st.empty()

# ------------------------
# Fetch
# ------------------------

@st.cache_data(ttl=30)
def _fetch(symbol: str) -> pd.DataFrame:
    df = fetch_option_chain(symbol)
    # Ensure proper dtypes
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    for c in ["strike", "ce_oi", "pe_oi", "ce_oi_chg", "pe_oi_chg", "ce_ltp", "pe_ltp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

df_all = _fetch(SYMBOL)

if df_all is None or df_all.empty:
    st.error("No data received from NSE. Try refreshing.")
    st.stop()

# Expiry selection
expiries = get_expiries(df_all)
if expiries:
    default_idx = max(0, len(expiries) - 1)  # pick latest by default
    chosen_exp = st.sidebar.selectbox("Expiry", expiries, index=default_idx)
    df_exp = df_all[df_all["expiry"].astype(str) == str(chosen_exp)].copy()
else:
    chosen_exp = None
    df_exp = df_all.copy()

# Use only the latest timestamp slice to build the live picture
wide_latest = get_latest_slice(df_exp)

# Maintain session history for VWAP & 3-min deltas
if "history_wide" not in st.session_state:
    st.session_state["history_wide"] = pd.DataFrame()
st.session_state["history_wide"] = pd.concat([st.session_state["history_wide"], wide_latest], ignore_index=True)
history_wide = st.session_state["history_wide"]

# Spot & ATM
spot = float(wide_latest["spot"].iloc[0]) if "spot" in wide_latest.columns and not wide_latest.empty else np.nan
atm = get_atm_strike(spot, STRIKE_STEP)

# ------------------------
# Top summary (quick 3-min take)
# ------------------------

deltas = last_3min_delta(history_wide)
if deltas:
    line1 = f"CE OI Δ (≈3m): {int(deltas.get('ce_oi', 0)):,} | PE OI Δ (≈3m): {int(deltas.get('pe_oi', 0)):,}"
    line2 = f"CE |ΔOI| (≈3m): {int(abs(deltas.get('ce_oi_chg', 0))):,} | PE |ΔOI| (≈3m): {int(abs(deltas.get('pe_oi_chg', 0))):,}"
    bias = (
        "Bullish" if deltas.get("pe_oi", 0) > deltas.get("ce_oi", 0)
        else "Bearish" if deltas.get("ce_oi", 0) > deltas.get("pe_oi", 0)
        else "Neutral"
    )
    line3 = f"Bias: {bias} (based on relative OI increase)"
    st.markdown(f"**Quick 3-min Take:**  {line1}  •  {line2}  •  {line3}")
else:
    st.caption("Waiting for at least two snapshots to compute ≈3-minute deltas.")

# ------------------------
# ATM±k table with totals + Trending OI for the band
# ------------------------

band = subset_atm_band(wide_latest, atm, near_strikes, STRIKE_STEP)
st.subheader(f"ATM±{near_strikes} Strikes Summary (ATM ≈ {atm})")
atm_tbl = build_atm_band_table(band)
if "CE OI Δ" in atm_tbl.columns and "PE OI Δ" in atm_tbl.columns and not atm_tbl.empty:
    trending_oi_band = float(atm_tbl.iloc[:-1]["CE OI Δ"].abs().sum() + atm_tbl.iloc[:-1]["PE OI Δ"].abs().sum())
    st.caption(f"Trending OI (band): {int(trending_oi_band):,}")
st.dataframe(atm_tbl, use_container_width=True)

# ------------------------
# Top Trending Strikes (table + bar chart)
# ------------------------

st.subheader("Top Trending Strikes (by |ΔOI|)")
top_tr = top_trending_strikes(wide_latest, n=8)
if not top_tr.empty:
    st.dataframe(top_tr, use_container_width=True)
    fig = go.Figure()
    fig.add_bar(x=top_tr["strike"].astype(str), y=top_tr["Total |ΔOI|"], name="|ΔOI| (CE+PE)")
    fig.update_layout(height=320, xaxis_title="Strike", yaxis_title="Total |ΔOI|")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Insufficient fields to compute trending strikes.")

# ------------------------
# Strike-wise PCR (table + line chart)
# ------------------------

st.subheader("Strike-wise PCR")
pcr_tbl = strikewise_pcr(wide_latest)
if not pcr_tbl.empty:
    st.dataframe(pcr_tbl, use_container_width=True)
    fig_p = go.Figure()
    fig_p.add_scatter(x=pcr_tbl["strike"], y=pcr_tbl["PCR"], mode="lines+markers", name="PCR")
    fig_p.update_layout(height=320, xaxis_title="Strike", yaxis_title="PCR")
    st.plotly_chart(fig_p, use_container_width=True)
else:
    st.info("Insufficient fields for PCR.")

# ------------------------
# User-selected Trending OI (pick 5–6 strikes)
# ------------------------

st.subheader("Custom Trending OI — your selected strikes")
all_strikes = sorted(wide_latest["strike"].dropna().unique().tolist()) if "strike" in wide_latest.columns else []
sel_strikes = st.multiselect("Choose 5–6 strikes", all_strikes, max_selections=6)
if sel_strikes:
    sel_df = wide_latest[wide_latest["strike"].isin(sel_strikes)].copy()
    ce_d = sel_df["ce_oi_chg"].fillna(0).abs().sum() if "ce_oi_chg" in sel_df else 0.0
    pe_d = sel_df["pe_oi_chg"].fillna(0).abs().sum() if "pe_oi_chg" in sel_df else 0.0
    st.write(f"**Trending OI (selected):** {int(ce_d + pe_d):,}  •  CE |ΔOI|: {int(ce_d):,}  •  PE |ΔOI|: {int(pe_d):,}")
else:
    st.caption("Pick up to 6 strikes to compute your custom Trending OI.")

# ------------------------
# CE vs PE with Session VWAP (Above/Below markers)
# ------------------------

st.subheader("CE vs PE with Session VWAP (Above/Below)")
vw_ce = session_vwap(history_wide, side="ce")
vw_pe = session_vwap(history_wide, side="pe")
need_cols = {"strike", "ce_ltp", "pe_ltp"}
if need_cols.issubset(set(wide_latest.columns)):
    combo = wide_latest[["strike", "ce_ltp", "pe_ltp"]].copy()
    combo["CE VWAP"] = combo["strike"].map(vw_ce)
    combo["PE VWAP"] = combo["strike"].map(vw_pe)
    combo["CE vs VWAP"] = np.where(combo["ce_ltp"] > combo["CE VWAP"], "Above", "Below")
    combo["PE vs VWAP"] = np.where(combo["pe_ltp"] > combo["PE VWAP"], "Above", "Below")
    st.dataframe(combo.rename(columns={"ce_ltp": "CE LTP", "pe_ltp": "PE LTP"}), use_container_width=True)
    st.caption("Strikes marked **Above** may indicate strength vs. session average price.")
else:
    st.info("Need CE/PE LTP columns to compute VWAP markers.")

# ------------------------
# Buyer vs Seller Control Strength (side-by-side)
# ------------------------

st.subheader("Buyer vs Seller Control Strength")
buyers, sellers = 0.0, 0.0
for side in ["ce", "pe"]:
    col_name = f"{side}_buildup"
    if col_name in wide_latest.columns:
        b, s = zip(*[score_buyer_seller(x) for x in wide_latest[col_name].fillna("")])
        buyers += sum(b)
        sellers += sum(s)

total = buyers + sellers if (buyers + sellers) > 0 else 1.0
c1, c2 = st.columns(2)
with c1:
    st.metric("Buyers (score)", f"{buyers:.1f}", help="Aggregated from buildup types: Long Buildup (+1), Short Covering (+0.5)")
    st.progress(min(1.0, buyers / total))
with c2:
    st.metric("Sellers (score)", f"{sellers:.1f}", help="Aggregated from buildup types: Short Buildup (+1), Long Unwinding (+0.5)")
    st.progress(min(1.0, sellers / total))

# ------------------------
# Footer / debug
# ------------------------

st.caption(
    f"Data ts: {pd.to_datetime(wide_latest['ts'].iloc[0]).strftime('%Y-%m-%d %H:%M:%S') if 'ts' in wide_latest.columns and not wide_latest.empty else 'NA'}"
)
