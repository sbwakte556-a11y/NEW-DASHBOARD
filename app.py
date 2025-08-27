# app.py — NIFTY/BANKNIFTY Options Live Dashboard (keeps earlier features + your new asks)
# - Symbol switch (NIFTY / BANKNIFTY)
# - Expiry mode: default = *Nearest/Recent only*, or let user pick any expiry
# - ATM±k table with totals + Trending OI (band)
# - Top Trending Strikes (table + bar chart)
# - Strike-wise PCR (table + line chart)
# - User-selected Trending OI (5–6 strikes) with GREEN/RED coloring
# - CE vs PE with Session VWAP (Above/Below markers)
# - Buyer vs Seller Control Strength (side-by-side)
# - Quick 3-min Take based on recent OI deltas
# - Auto-refresh toggle + market-hours notice

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import pytz
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Your local modules
from nse_fetch import fetch_option_chain
from config import STRIKE_STEP, NEAR_STRIKES, REFRESH_SECS, TIMEZONE

# ------------------------
# Utilities
# ------------------------

IST = pytz.timezone(TIMEZONE if TIMEZONE else "Asia/Kolkata")

def safe_val(row, col, default=0.0):
    """
    Safely extract a scalar float from row[col].
    Supports dict, pandas Series, and single-row DataFrames.
    Returns `default` if missing/NaN; if Series/array, uses first non-NaN scalar.
    """
    try:
        if row is None:
            return float(default)
        if isinstance(row, dict):
            v = row.get(col, default)
        elif hasattr(row, "index") and not hasattr(row, "columns"):  # Series
            v = row[col] if col in row.index else default
        elif hasattr(row, "columns"):  # DataFrame
            if col in row.columns and len(row) > 0:
                v = row.iloc[0][col]
            else:
                v = default
        else:
            v = getattr(row, col, default)
        if hasattr(v, "__array__") or str(type(v)).endswith("Series'>"):
            try:
                v = next((x for x in np.ravel(v) if pd.notna(x)), default)
            except Exception:
                v = default
        return float(v) if (v is not None and pd.notna(v)) else float(default)
    except Exception:
        return float(default)

def get_latest_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "ts" in df.columns:
        tmax = pd.to_datetime(df["ts"]).max()
        return df[pd.to_datetime(df["ts"]) == tmax].copy()
    return df.copy()

def get_expiries(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "expiry" not in df.columns:
        return []
    return sorted(pd.Series(df["expiry"].astype(str).unique()).tolist())

def nearest_expiry(exp_list: list[str]) -> str | None:
    if not exp_list:
        return None
    # Parse to dates; choose the soonest >= today; else the min
    today = pd.Timestamp.now(tz=IST).normalize()
    parsed = pd.to_datetime(pd.Series(exp_list), errors="coerce")
    future = parsed[parsed >= today]
    chosen = (future.min() if not future.empty else parsed.min())
    if pd.isna(chosen):
        return str(exp_list[-1])
    # map back to original string format
    # find the original string whose parsed equals chosen
    for s in exp_list:
        if pd.to_datetime(s, errors="coerce") == chosen:
            return s
    return str(chosen.date())

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
# Page + Sidebar
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
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
    for c in ["strike", "ce_oi", "pe_oi", "ce_oi_chg", "pe_oi_chg", "ce_ltp", "pe_ltp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "expiry" in df.columns:
        df["expiry"] = df["expiry"].astype(str)
    return df

df_all = _fetch(SYMBOL)
if df_all is None or df_all.empty:
    st.error("No data received from NSE. Try refreshing.")
    st.stop()

# ------------------------
# Expiry selection (RECENT by default)
# ------------------------

expiries = get_expiries(df_all)
recent_only = st.sidebar.toggle("Recent expiry only (recommended)", value=True)
if expiries:
    if recent_only:
        exp_default = nearest_expiry(expiries)
        df_exp = df_all[df_all["expiry"] == str(exp_default)].copy()
        st.sidebar.write(f"Using expiry: **{exp_default}**")
    else:
        default_idx = max(0, len(expiries) - 1)  # latest by sort order
        chosen_exp = st.sidebar.selectbox("Select expiry", expiries, index=default_idx)
        df_exp = df_all[df_all["expiry"] == str(chosen_exp)].copy()
else:
    df_exp = df_all.copy()

# Use only the latest timestamp slice for the live picture
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
# Quick 3-min Take (kept from earlier)
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
# ATM±k table with totals + Trending OI (band)
# ------------------------

def build_atm_band_table(df_band: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "strike", "ce_oi", "pe_oi", "ce_oi_chg", "pe_oi_chg",
        "ce_ltp", "pe_ltp", "ce_buildup", "pe_buildup"
    ]
    have = [c for c in cols if c in df_band.columns]
    tbl = df_band[have].copy()
    if "ce_oi" in tbl.columns and "pe_oi" in tbl.columns:
        tbl["pcr"] = (tbl["pe_oi"] / tbl["ce_oi"]).replace([np.inf, -np.inf], np.nan)
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

band = subset_atm_band(wide_latest, atm, near_strikes, STRIKE_STEP)
st.subheader(f"ATM±{near_strikes} Strikes Summary (ATM ≈ {atm})")
atm_tbl = build_atm_band_table(band)

# Favorability color rule (GREEN if CE-favourable, RED if PE-favourable)
# Simple heuristic: if (CE OI Δ) > (PE OI Δ) -> CE-favourable (green), else PE-favourable (red). Skip TOTAL row.
def _favor_colour(row):
    if row.get("strike") == "TOTAL":
        return [""] * len(row)
    ce = row.get("CE OI Δ", 0) or 0
    pe = row.get("PE OI Δ", 0) or 0
    color = "background-color: rgba(0, 200, 0, 0.15)" if ce > pe else "background-color: rgba(255, 0, 0, 0.15)"
    styles = []
    for c in atm_tbl.columns:
        if c in ["CE OI Δ", "PE OI Δ", "CE OI", "PE OI", "strike", "PCR", "CE LTP", "PE LTP", "CE Buildup", "PE Buildup"]:
            styles.append(color)
        else:
            styles.append("")
    return styles

if "CE OI Δ" in atm_tbl.columns and "PE OI Δ" in atm_tbl.columns and not atm_tbl.empty:
    trending_oi_band = float(atm_tbl.iloc[:-1]["CE OI Δ"].abs().sum() + atm_tbl.iloc[:-1]["PE OI Δ"].abs().sum())
    st.caption(f"Trending OI (band): {int(trending_oi_band):,}")
    st.dataframe(atm_tbl.style.apply(_favor_colour, axis=1), use_container_width=True)
else:
    st.dataframe(atm_tbl, use_container_width=True)

# ------------------------
# Top Trending Strikes (table + bar chart)
# ------------------------

st.subheader("Top Trending Strikes (by |ΔOI|)")
def top_trending_strikes(wide_latest: pd.DataFrame, n=8) -> pd.DataFrame:
    needed = {"strike", "ce_oi_chg", "pe_oi_chg"}
    if not needed.issubset(set(wide_latest.columns)):
        return pd.DataFrame()
    t = wide_latest[["strike", "ce_oi_chg", "pe_oi_chg"]].copy()
    t["abs_oi_chg"] = t["ce_oi_chg"].fillna(0).abs() + t["pe_oi_chg"].fillna(0).abs()
    t = t.sort_values("abs_oi_chg", ascending=False).head(n)
    return t.rename(columns={"ce_oi_chg": "CE OI Δ", "pe_oi_chg": "PE OI Δ", "abs_oi_chg": "Total |ΔOI|"})

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
def strikewise_pcr(wide_latest: pd.DataFrame) -> pd.DataFrame:
    needed = {"strike", "ce_oi", "pe_oi"}
    if not needed.issubset(set(wide_latest.columns)):
        return pd.DataFrame()
    df = wide_latest[["strike", "ce_oi", "pe_oi"]].copy()
    df["PCR"] = (df["pe_oi"] / df["ce_oi"]).replace([np.inf, -np.inf], np.nan)
    return df.rename(columns={"ce_oi": "CE OI", "pe_oi": "PE OI"})

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
# User-selected Trending OI (5–6 strikes) with GREEN/RED colouring
# ------------------------

st.subheader("Custom Trending OI — your selected strikes")
all_strikes = sorted(wide_latest["strike"].dropna().unique().tolist()) if "strike" in wide_latest.columns else []
sel_strikes = st.multiselect("Choose up to 6 strikes", all_strikes, max_selections=6)

def _color_selected(row):
    ce = row.get("CE OI Δ", 0) or 0
    pe = row.get("PE OI Δ", 0) or 0
    color = "background-color: rgba(0, 200, 0, 0.2)" if ce > pe else "background-color: rgba(255, 0, 0, 0.2)"
    return [color] * len(row)

if sel_strikes:
    sel_df = wide_latest[wide_latest["strike"].isin(sel_strikes)].copy()
    show_cols = [c for c in ["strike","ce_oi","pe_oi","ce_oi_chg","pe_oi_chg"] if c in sel_df.columns]
    view = sel_df[show_cols].rename(columns={
        "ce_oi":"CE OI","pe_oi":"PE OI",
        "ce_oi_chg":"CE OI Δ","pe_oi_chg":"PE OI Δ"
    })
    ce_d = view["CE OI Δ"].fillna(0).abs().sum() if "CE OI Δ" in view else 0.0
    pe_d = view["PE OI Δ"].fillna(0).abs().sum() if "PE OI Δ" in view else 0.0
    st.write(f"**Trending OI (selected):** {int(ce_d + pe_d):,}  •  CE |ΔOI|: {int(ce_d):,}  •  PE |ΔOI|: {int(pe_d):,}")
    st.dataframe(view.style.apply(_color_selected, axis=1), use_container_width=True)
else:
    st.caption("Pick up to 6 strikes to compute your custom Trending OI (green = CE-favourable, red = PE-favourable).")

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
        b, s = zip(*[score_buyer_seller(x) for x in wide_latest[col_name].fillna("")]) if not wide_latest.empty else ([], [])
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
# Footer
# ------------------------

st.caption(
    f"Data ts: {pd.to_datetime(wide_latest['ts'].iloc[0]).strftime('%Y-%m-%d %H:%M:%S') if 'ts' in wide_latest.columns and not wide_latest.empty else 'NA'}"
)
