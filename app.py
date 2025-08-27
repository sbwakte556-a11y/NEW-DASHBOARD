# NIFTY Live Option Chain — Streamlit (ONLINE, auto-refresh every 3 min)
# ---------------------------------------------------------------------
# Features (parity with your offline app):
# - Live fetch from NSE option-chain (no CSVs needed)
# - Auto-refresh every 180s (3 minutes) via streamlit_autorefresh
# - Spot/ATM/PCR, Market Sentiment, Buyer vs Seller strength
# - Top strikes (OI & Price), Buildup table with colors
# - OI Distribution, Straddle, CE–PE crossover around ATM
# - Trending OI (intraday) by keeping in-memory history during the session
# - Seller Shifting (Support/Resistance) + Big Add/Drop
#
# Notes:
# - History resets when the Streamlit session restarts (Streamlit Cloud sessions are ephemeral).
# - If NSE temporarily blocks, the app will show a warning; just wait for next auto-refresh
#   or hit the "Refresh now" button.

from __future__ import annotations
import time
from typing import Optional, List, Dict
import requests
import pandas as pd
import numpy as np
import datetime as dt
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

# Optional: st_aggrid (used for a colored Short Buildup table). Fallback to plain dataframe if not installed.
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, JsCode  # type: ignore
    HAS_AGGRID = True
except Exception:
    HAS_AGGRID = False

# ------------------------
# CONFIG / DEFAULTS
# ------------------------
st.set_page_config(layout="wide", page_title="NIFTY Options Live — Online", page_icon="📈")

DEFAULT_SYMBOL = "NIFTY"
# Sidebar symbol selector
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
# Keep legacy references working
SYMBOL = symbol
TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(TIMEZONE)
REFRESH_SECS_DEFAULT = 180  # 3 minutes
STRIKE_STEP = 50
NEAR_STRIKES_DEFAULT = 3
MAX_HISTORY_POINTS = 480  # ~24h if refreshed every 3 min

# ------------------------
# HTTP / NSE HELPERS
# ------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "accept-language": "en-US,en;q=0.9",
        "accept": "application/json, text/plain, */*",
        "referer": "https://www.nseindia.com/",
        "connection": "keep-alive",
    })
    try:
        s.get("https://www.nseindia.com", timeout=8)
    except Exception:
        pass
    return s

@st.cache_data(show_spinner=False, ttl=10)
def fetch_option_chain(symbol: str = SYMBOL, tries: int = 5, backoff: float = 1.5) -> pd.DataFrame:
    """Fetch NSE option-chain for an index symbol and return tidy long DataFrame.
    Columns: symbol, strike, option_type, ltp, oi, volume, iv, ts, spot
    """
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    s = make_session()
    headers = {"accept": "application/json, text/plain, */*", "referer": "https://www.nseindia.com/option-chain"}

    data: Dict = {}
    for attempt in range(tries):
        try:
            r = s.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                data = r.json()
                break
            time.sleep(backoff * (attempt + 1))
        except Exception:
            time.sleep(backoff * (attempt + 1))
    if not data:
        return pd.DataFrame()

    records = data.get("records", {})
    ts = dt.datetime.now(IST)
    spot = float(records.get("underlyingValue") or 0.0)

    rows: List[Dict] = []
    for item in records.get("data", []):
        strike = item.get("strikePrice")
        if strike is None:
            continue
        for side in ("CE", "PE"):
            leg = item.get(side)
            if not isinstance(leg, dict):
                continue
            rows.append({
                "symbol": symbol,
                "strike": int(strike),
                "option_type": side,
                "ltp": float(leg.get("lastPrice") or 0.0),
                "oi": float(leg.get("openInterest") or 0.0),
                "volume": float(leg.get("totalTradedVolume") or 0.0),
                "iv": float(leg.get("impliedVolatility") or 0.0),
                "vwap": np.nan,  # not provided by NSE OC endpoint
                "ts": ts,
                "spot": float(spot or 0.0),
            })

    return pd.DataFrame(rows)

# ------------------------
# ANALYTICS HELPERS
# ------------------------
def nearest_strike(price: float, step: int = STRIKE_STEP) -> int:
    return int(round(price / step) * step)


def classify_buildup(oi_change: float, ltp_change: float) -> str:
    if pd.isna(oi_change) or pd.isna(ltp_change):
        return "Neutral"
    if oi_change > 0 and ltp_change > 0:
        return "Long Buildup"
    if oi_change > 0 and ltp_change < 0:
        return "Short Buildup"
    if oi_change < 0 and ltp_change < 0:
        return "Long Unwinding"
    if oi_change < 0 and ltp_change > 0:
        return "Short Covering"
    return "Neutral"


def enrich_with_prev(curr: pd.DataFrame, prev: Optional[pd.DataFrame]) -> pd.DataFrame:
    if curr.empty:
        return curr
    df = curr.copy()
    if prev is None or prev.empty:
        df["prev_ltp"], df["prev_oi"] = df["ltp"], df["oi"]
    else:
        m = prev[["symbol", "strike", "option_type", "ltp", "oi"]].rename(columns={"ltp": "prev_ltp", "oi": "prev_oi"})
        df = df.merge(m, on=["symbol", "strike", "option_type"], how="left")
        df["prev_ltp"] = df["prev_ltp"].fillna(df["ltp"])
        df["prev_oi"] = df["prev_oi"].fillna(df["oi"])

    df["oi_chg"] = df["oi"] - df["prev_oi"]
    df["ltp_chg"] = df["ltp"] - df["prev_ltp"]
    df["oi_chg_pct"] = np.where(df["prev_oi"] > 0, 100 * df["oi_chg"] / df["prev_oi"], 0.0)
    df["ltp_chg_pct"] = np.where(df["prev_ltp"] > 0, 100 * df["ltp_chg"] / df["prev_ltp"], 0.0)
    df["buildup"] = [classify_buildup(o, p) for o, p in zip(df["oi_chg"], df["ltp_chg"])]
    df["above_vwap"] = df["ltp"] > df["vwap"]
    return df


def select_near_atm(df: pd.DataFrame, spot: float, n: int = NEAR_STRIKES_DEFAULT) -> pd.DataFrame:
    if df.empty:
        return df
    atm = nearest_strike(spot)
    lo, hi = atm - n * STRIKE_STEP, atm + n * STRIKE_STEP
    return df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()


def compute_crossover(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    if df.empty:
        return pd.DataFrame(out)
    for (symbol, strike), g in df.groupby(["symbol", "strike"], as_index=False):
        ce = g[g.option_type == "CE"]["ltp"].values
        pe = g[g.option_type == "PE"]["ltp"].values
        if ce.size and pe.size:
            ce_val, pe_val = float(ce[0]), float(pe[0])
            out.append({
                "symbol": symbol,
                "strike": int(strike),
                "ce_gt_pe": bool(ce_val > pe_val),
                "pe_gt_ce": bool(pe_val > ce_val),
                "diff_pct": float((ce_val - pe_val) / max(1e-6, pe_val) * 100),
            })
    return pd.DataFrame(out)

# Wide-format utilities for history (to replicate offline features)

def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    keep = ["strike", "option_type", "oi", "ltp", "spot", "ts"]
    df = df_long[keep].copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df["oi"] = pd.to_numeric(df["oi"], errors="coerce")
    df["ltp"] = pd.to_numeric(df["ltp"], errors="coerce")
    df = df.dropna(subset=["strike"])
    piv_oi = df.pivot_table(index=["ts", "spot", "strike"], columns="option_type", values="oi", aggfunc="sum")
    piv_ltp = df.pivot_table(index=["ts", "spot", "strike"], columns="option_type", values="ltp", aggfunc="mean")
    wide = pd.concat([
        piv_oi.rename(columns={"CE": "ce_oi", "PE": "pe_oi"}),
        piv_ltp.rename(columns={"CE": "ce_ltp", "PE": "pe_ltp"}),
    ], axis=1).reset_index()
    for c in ["ce_oi", "pe_oi", "ce_ltp", "pe_ltp"]:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide.sort_values(["ts", "strike"]).reset_index(drop=True)


def infer_atm_strike(wide_latest: pd.DataFrame) -> float:
    spot = float(wide_latest["spot"].median()) if "spot" in wide_latest.columns and wide_latest["spot"].notna().any() else np.nan
    if np.isfinite(spot):
        diffs = (wide_latest["strike"] - spot).abs()
        return float(wide_latest.loc[diffs.idxmin(), "strike"])
    ssum = (wide_latest["ce_ltp"].fillna(0) + wide_latest["pe_ltp"].fillna(0))
    return float(wide_latest.loc[ssum.idxmin(), "strike"]) if not wide_latest.empty else np.nan


# ------------------------
# SIDEBAR / SETTINGS
# ------------------------
st.sidebar.title("Settings")
refresh_secs = st.sidebar.number_input("Auto-refresh (seconds)", min_value=30, max_value=900, value=REFRESH_SECS_DEFAULT, step=30)
near_strikes = st.sidebar.slider("Strikes near ATM (±)", 1, 6, NEAR_STRIKES_DEFAULT)
oi_alert_pct = st.sidebar.slider("Exceptional OI% threshold", 5, 500, 50)
st.sidebar.markdown("---")
st.sidebar.button("🔄 Refresh now", on_click=lambda: st.cache_data.clear())

# Client-side auto refresh
autorefresh_on = st.sidebar.toggle("Auto refresh every n seconds", value=True)
if autorefresh_on:
    st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh")

# ------------------------
# FETCH & ENRICH
# ------------------------
st.title(f"📈 {SYMBOL} Options Live — Online Dashboard")
# Market hours notice (IST)
_now_ist = dt.datetime.now(pytz.timezone("Asia/Kolkata")).time()
if not (dt.time(9, 14) <= _now_ist <= dt.time(15, 31)):
    st.info("Market appears closed. Data may be stale or unchanged.")
status = st.empty()

if "prev_snapshot" not in st.session_state:
    st.session_state.prev_snapshot = None
if "history_wide" not in st.session_state:
    st.session_state.history_wide = pd.DataFrame()  # accumulated wide format per fetch

try:
    status.info("Fetching option-chain from NSE...")
    curr = fetch_option_chain(SYMBOL)
    if curr.empty:
        status.error("Failed to fetch data from NSE (empty). Try again shortly.")
        st.stop()
    status.success("Fetched live option chain.")
except Exception as e:
    status.error(f"Fetch failed: {e}")
    st.stop()

# Enrich with previous snapshot for 1-interval changes
prev = st.session_state.prev_snapshot
df_en = enrich_with_prev(curr, prev)
st.session_state.prev_snapshot = curr[["symbol", "strike", "option_type", "ltp", "oi"]].copy()

# Build/append to in-memory history for intraday trending
wide_latest = make_wide(df_en)

if not wide_latest.empty:
    hist = st.session_state.history_wide
    st.session_state.history_wide = pd.concat([hist, wide_latest], ignore_index=True)
    # Keep last N points to limit memory
    if len(st.session_state.history_wide) > MAX_HISTORY_POINTS:
        st.session_state.history_wide = st.session_state.history_wide.tail(MAX_HISTORY_POINTS).reset_index(drop=True)

# ------------------------
# HEADER METRICS & SENTIMENT
# ------------------------
spot = float(df_en["spot"].dropna().iloc[0]) if "spot" in df_en.columns and not df_en["spot"].dropna().empty else 0.0
snapshot_time = pd.to_datetime(df_en["ts"].iloc[0]) if "ts" in df_en.columns else dt.datetime.now(IST)

# Buyer/Seller strength proxy using |ΔLTP|*Volume
ce = df_en[df_en.option_type == "CE"].copy()
pe = df_en[df_en.option_type == "PE"].copy()
ce["score"] = ce["volume"].fillna(0) * ce["ltp_chg"].abs().fillna(0)
pe["score"] = pe["volume"].fillna(0) * pe["ltp_chg"].abs().fillna(0)
ce_strength = float(ce["score"].sum())
pe_strength = float(pe["score"].sum())

total_strength = (ce_strength + pe_strength) or 1
buyer_pct = 100 * ce_strength / total_strength
seller_pct = 100 * pe_strength / total_strength

# directional counts for sentiment
up_count = ((df_en["buildup"] == "Long Buildup") | (df_en["buildup"] == "Short Covering")).sum()
down_count = ((df_en["buildup"] == "Short Buildup") | (df_en["buildup"] == "Long Unwinding")).sum()

sent_score = 0.6 * (buyer_pct - seller_pct) + 0.4 * (up_count - down_count)
if sent_score > 15:
    sentiment_label, sentiment_color = "Bullish", "#1b5e20"
elif sent_score < -15:
    sentiment_label, sentiment_color = "Bearish", "#b71c1c"
else:
    sentiment_label, sentiment_color = "Neutral", "#263238"

# PCR (current snapshot)
try:
    ce_oi_total = float(df_en[df_en.option_type == "CE"]["oi"].sum())
    pe_oi_total = float(df_en[df_en.option_type == "PE"]["oi"].sum())
    pcr_val = pe_oi_total / ce_oi_total if ce_oi_total > 0 else np.nan
except Exception:
    pcr_val = np.nan

# header layout
c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
with c1:
    st.markdown(f"**Spot (approx)**  \n:large_blue_circle: **{spot:.2f}**")
    st.markdown(f"**Snapshot**  \n{snapshot_time.strftime('%Y-%m-%d %H:%M:%S')}")
with c2:
    st.metric("Buyer % (CE proxy)", f"{buyer_pct:.1f}%")
with c3:
    st.metric("Seller % (PE proxy)", f"{seller_pct:.1f}%")
with c4:
    st.metric("PCR", f"{pcr_val:.2f}" if np.isfinite(pcr_val) else "—")
    st.markdown(
        f"""
        <div style=\"margin-top:6px;padding:10px;border-radius:10px;background:{sentiment_color};color:white;text-align:center\">
        <strong>Market Sentiment</strong><br><span style=\"font-size:18px\">{sentiment_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ------------------------
# NEAR-ATM VIEW
# ------------------------
near = select_near_atm(df_en, spot, n=near_strikes)
st.subheader(f"Strikes around ATM (±{near_strikes * STRIKE_STEP} points) — showing {len(near)//2} strikes")

if near.empty:
    st.warning("No near-ATM data available.")
else:
    show_cols = ["strike", "option_type", "ltp", "iv", "oi", "oi_chg_pct", "ltp_chg_pct", "buildup"]
    st.dataframe(near.sort_values(["strike", "option_type"])[show_cols], use_container_width=True)

    cross = compute_crossover(near)
    st.markdown("**CE vs PE crossover (near ATM)**")
    if not cross.empty:
        st.dataframe(cross.sort_values("strike"), use_container_width=True)
    else:
        st.info("Crossover table will appear once both CE & PE premiums are available.")

# ------------------------
# TRENDING OI for a selectable strike (1-interval view)
# ------------------------
st.markdown("---")
st.subheader("🔎 Trending OI — pick a strike to inspect (interval vs previous)")

available_strikes = sorted(df_en["strike"].unique()) if not df_en.empty else []
sel_str_index = len(available_strikes) // 2 if available_strikes else 0
sel_str = st.selectbox("Select strike", options=available_strikes, index=sel_str_index)

if sel_str:
    s = df_en[(df_en.strike == sel_str)]
    ce_row = s[s.option_type == "CE"].squeeze() if not s[s.option_type == "CE"].empty else None
    pe_row = s[s.option_type == "PE"].squeeze() if not s[s.option_type == "PE"].empty else None

    
def safe_val(row, col, default=0.0):
    """Safely extract a scalar float from row[col].
    Supports dict, pandas Series, and single-row DataFrames.
    Returns `default` if missing/NaN; if a Series is found, uses the first non-NaN scalar.
    """
    try:
        if row is None:
            return float(default)
        # dict-like
        if isinstance(row, dict):
            v = row.get(col, default)
        # pandas Series
        elif hasattr(row, 'index') and not hasattr(row, 'columns'):
            v = row[col] if col in row.index else default
        # pandas DataFrame (take the first row's value for this column)
        elif hasattr(row, 'columns'):
            if col in row.columns and len(row) > 0:
                v = row.iloc[0][col]
            else:
                v = default
        else:
            # fallback: attribute access
            v = getattr(row, col, default)
        # If v is a Series or array, take first valid
        if hasattr(v, '__array__') or str(type(v)).endswith("Series'>"):
            try:
                v = next((x for x in np.ravel(v) if pd.notna(x)), default)
            except Exception:
                v = default
        return float(v) if (v is not None and pd.notna(v)) else float(default)
    except Exception:
        return float(default)

    ce_oi = safe_val(ce_row, "oi");        pe_oi = safe_val(pe_row, "oi")
    ce_prev_oi = safe_val(ce_row, "prev_oi"); pe_prev_oi = safe_val(pe_row, "prev_oi")
    ce_oi_chg = ce_oi - ce_prev_oi;          pe_oi_chg = pe_oi - pe_prev_oi
    ce_oi_chg_pct = (100 * ce_oi_chg / ce_prev_oi) if ce_prev_oi > 0 else (100 if ce_oi_chg > 0 else 0)
    pe_oi_chg_pct = (100 * pe_oi_chg / pe_prev_oi) if pe_prev_oi > 0 else (100 if pe_oi_chg > 0 else 0)

    ce_ltp = safe_val(ce_row, "ltp");      pe_ltp = safe_val(pe_row, "ltp")
    ce_prev_ltp = safe_val(ce_row, "prev_ltp"); pe_prev_ltp = safe_val(pe_row, "prev_ltp")
    ce_ltp_chg = ce_ltp - ce_prev_ltp;       pe_ltp_chg = pe_ltp - pe_prev_ltp
    ce_ltp_chg_pct = (100 * ce_ltp_chg / ce_prev_ltp) if ce_prev_ltp > 0 else 0
    pe_ltp_chg_pct = (100 * pe_ltp_chg / pe_prev_ltp) if pe_prev_ltp > 0 else 0

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#e8f5e9'><strong>CE OI</strong><br><span style='font-size:20px'>{int(ce_oi):,}</span><br><small>Δ {int(ce_oi_chg):+,}</small></div>", unsafe_allow_html=True)
    with a2:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#ffebee'><strong>PE OI</strong><br><span style='font-size:20px'>{int(pe_oi):,}</span><br><small>Δ {int(pe_oi_chg):+,}</small></div>", unsafe_allow_html=True)
    with a3:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#e3f2fd'><strong>CE LTP</strong><br><span style='font-size:20px'>{ce_ltp:.2f}</span><br><small>Δ {ce_ltp_chg:+.2f} ({ce_ltp_chg_pct:+.1f}%)</small></div>", unsafe_allow_html=True)
    with a4:
        st.markdown(f"<div style='padding:10px;border-radius:8px;background:#fff8e1'><strong>PE LTP</strong><br><span style='font-size:20px'>{pe_ltp:.2f}</span><br><small>Δ {pe_ltp_chg:+.2f} ({pe_ltp_chg_pct:+.1f}%)</small></div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Summary Table", "Charts"])
    with tab1:
        table = {
            "side": ["CE", "PE"],
            "oi": [ce_oi, pe_oi],
            "oi_prev": [ce_prev_oi, pe_prev_oi],
            "oi_chg": [ce_oi_chg, pe_oi_chg],
            "oi_chg_pct": [round(ce_oi_chg_pct, 2), round(pe_oi_chg_pct, 2)],
            "ltp": [ce_ltp, pe_ltp],
            "ltp_prev": [ce_prev_ltp, pe_prev_ltp],
            "ltp_chg": [round(ce_ltp_chg, 4), round(pe_ltp_chg, 4)],
            "ltp_chg_pct": [round(ce_ltp_chg_pct, 2), round(pe_ltp_chg_pct, 2)],
            "buildup": [ce_row.get("buildup", "NA") if ce_row is not None else "NA", pe_row.get("buildup", "NA") if pe_row is not None else "NA"],
        }
        df_table = pd.DataFrame(table)
        st.dataframe(df_table, use_container_width=True)

    with tab2:
        fig1 = go.Figure(data=[go.Bar(name="CE OI", x=["CE", "PE"], y=[ce_oi, pe_oi])])
        fig1.update_layout(title_text=f"Total OI at strike {sel_str}", height=320, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure(data=[go.Bar(name="OI change %", x=["CE", "PE"], y=[ce_oi_chg_pct, pe_oi_chg_pct])])
        fig2.update_layout(title_text="OI change % vs previous snapshot", height=320, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(name="LTP", x=["CE", "PE"], y=[ce_ltp, pe_ltp]))
        fig3.update_layout(title_text="Premium (LTP) — CE vs PE", height=320, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------
# TOP MOVERS PANEL
# ------------------------
st.markdown("---")
st.subheader("Top movers (last interval) — by LTP % change")
if "ltp_chg_pct" in df_en.columns:
    top = df_en.assign(pct=df_en["ltp_chg_pct"]).sort_values("pct", ascending=False).head(10)
    st.dataframe(top[["strike", "option_type", "ltp", "ltp_chg_pct", "oi", "oi_chg_pct"]], use_container_width=True)
else:
    st.info("First snapshot — come back after next refresh.")

# ------------------------
# MASTER/INTRADAY HISTORY VIEWS (replicates offline charts)
# ------------------------
if not st.session_state.history_wide.empty:
    hist = st.session_state.history_wide.copy()
    # Calculate diffs per strike across history
    for col in ["ce_oi", "pe_oi", "ce_ltp", "pe_ltp"]:
        hist[f"{col}_chg"] = hist.groupby("strike")[col].diff()

    latest_ts = hist["ts"].max()
    latest_df = hist[hist["ts"] == latest_ts].copy()

    # === Header block 2: ATM, Support/Resistance by max OI ===
    atm = infer_atm_strike(latest_df)
    spot_latest = float(latest_df["spot"].median()) if latest_df["spot"].notna().any() else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("ATM", f"{atm:.0f}" if np.isfinite(atm) else "—")
    c2.metric("Spot (median)", f"{spot_latest:,.2f}" if np.isfinite(spot_latest) else "—")
    c3.metric("Snapshots stored", str(hist["ts"].nunique()))

    # === Top Strikes ===
    st.subheader("🏆 Top Strikes — OI & Price")
    top_n = st.slider("Top N", 5, 15, 10)
    t1, t2 = st.columns(2)
    with t1:
        st.write("Top CE OI")
        st.dataframe(latest_df.nlargest(top_n, "ce_oi")[ ["strike", "ce_oi", "ce_oi_chg"] ])
        st.write("Top CE Price")
        st.dataframe(latest_df.nlargest(top_n, "ce_ltp")[ ["strike", "ce_ltp", "ce_ltp_chg"] ])
    with t2:
        st.write("Top PE OI")
        st.dataframe(latest_df.nlargest(top_n, "pe_oi")[ ["strike", "pe_oi", "pe_oi_chg"] ])
        st.write("Top PE Price")
        st.dataframe(latest_df.nlargest(top_n, "pe_ltp")[ ["strike", "pe_ltp", "pe_ltp_chg"] ])

    # === Buildup Table ===
    st.subheader("🧮 Buildup Table (latest vs previous snapshot)")
    buildup_df = latest_df[["strike", "ce_oi_chg", "ce_ltp_chg", "pe_oi_chg", "pe_ltp_chg"]].copy()
    buildup_df["CE Buildup"] = [
        classify_buildup(o, p) for o, p in zip(buildup_df["ce_oi_chg"], buildup_df["ce_ltp_chg"])
    ]
    buildup_df["PE Buildup"] = [
        classify_buildup(o, p) for o, p in zip(buildup_df["pe_oi_chg"], buildup_df["pe_ltp_chg"])
    ]
    st.dataframe(buildup_df)

    # === OI Distribution ===
    st.subheader("📊 OI Distribution")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=latest_df["ce_oi"], name="CE OI"))
    fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=-latest_df["pe_oi"], name="PE OI"))
    fig_dist.update_layout(barmode="relative")
    st.plotly_chart(fig_dist, use_container_width=True)

    # === Straddle ===
    st.subheader("🎯 Straddle (CE+PE)")
    latest_df["straddle"] = latest_df["ce_ltp"].fillna(0) + latest_df["pe_ltp"].fillna(0)
    fig_straddle = go.Figure()
    fig_straddle.add_trace(go.Scatter(x=latest_df["strike"], y=latest_df["straddle"], mode="lines+markers"))
    if np.isfinite(atm):
        fig_straddle.add_vline(x=atm, line_dash="dash", annotation_text=f"ATM {atm:.0f}")
    if np.isfinite(spot_latest):
        fig_straddle.add_vline(x=spot_latest, line_dash="dot", annotation_text=f"Spot {spot_latest:.0f}")
    st.plotly_chart(fig_straddle, use_container_width=True)

    # === Trending OI (ATM ± N Strikes across the session) ===
    st.subheader("⏱️ Trending OI (ATM ± Strikes — intraday from session history)")
    # Filter history for market hours only (approx; assumes India market):
    def within_market_hours(ts: pd.Timestamp) -> bool:
        if not isinstance(ts, (pd.Timestamp, dt.datetime)):
            return False
        t = pd.Timestamp(ts).tz_localize(None).time() if isinstance(ts, pd.Timestamp) and ts.tzinfo else pd.Timestamp(ts).time()
        return dt.time(9, 15) <= t <= dt.time(15, 30)

    day_hist = hist[hist["ts"].apply(within_market_hours)]
    if not day_hist.empty:
        first_ts = day_hist["ts"].min()
        base_atm = infer_atm_strike(day_hist[day_hist["ts"] == first_ts])
        strikes_window = [base_atm + i * STRIKE_STEP for i in range(-near_strikes, near_strikes + 1)] if np.isfinite(base_atm) else []
        day_win = day_hist[day_hist["strike"].isin(strikes_window)] if strikes_window else pd.DataFrame()
        if not day_win.empty:
            tot_by_ts = day_win.groupby("ts", as_index=False).agg(ce_oi=("ce_oi", "sum"), pe_oi=("pe_oi", "sum"))
            fig_day = go.Figure()
            fig_day.add_trace(go.Scatter(x=tot_by_ts["ts"], y=tot_by_ts["ce_oi"], name="CE OI"))
            fig_day.add_trace(go.Scatter(x=tot_by_ts["ts"], y=tot_by_ts["pe_oi"], name="PE OI"))
            st.plotly_chart(fig_day, use_container_width=True)

    # === Seller Shifting — Support/Resistance + Big Add/Drop ===
    st.subheader("🧭 Seller Shifting — Support/Resistance + Big Add/Drop")
    timestamps_sorted = sorted(hist["ts"].unique())
    if len(timestamps_sorted) >= 2:
        ts_prev, ts_last = timestamps_sorted[-2], timestamps_sorted[-1]
        prev = hist[hist["ts"] == ts_prev]
        last = hist[hist["ts"] == ts_last]

        def top_strike(df: pd.DataFrame, col: str) -> Optional[float]:
            sub = df[["strike", col]].dropna()
            if sub.empty:
                return None
            return float(sub.loc[sub[col].idxmax(), "strike"])  # type: ignore

        prev_ce = top_strike(prev, "ce_oi"); last_ce = top_strike(last, "ce_oi")
        prev_pe = top_strike(prev, "pe_oi"); last_pe = top_strike(last, "pe_oi")

        st.markdown(f"**Call OI Top (Resistance):** {prev_ce:.0f} ➝ {last_ce:.0f}" if prev_ce and last_ce else "—")
        st.markdown(f"**Put OI Top (Support):** {prev_pe:.0f} ➝ {last_pe:.0f}" if prev_pe and last_pe else "—")

        # Biggest changes (sellers’ activity)
        ce_big_add = last.loc[last["ce_oi_chg"].idxmax()][["strike", "ce_oi_chg"]] if last["ce_oi_chg"].notna().any() else None
        ce_big_drop = last.loc[last["ce_oi_chg"].idxmin()][["strike", "ce_oi_chg"]] if last["ce_oi_chg"].notna().any() else None
        pe_big_add = last.loc[last["pe_oi_chg"].idxmax()][["strike", "pe_oi_chg"]] if last["pe_oi_chg"].notna().any() else None
        pe_big_drop = last.loc[last["pe_oi_chg"].idxmin()][["strike", "pe_oi_chg"]] if last["pe_oi_chg"].notna().any() else None

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Calls (CE)**")
            st.write(f"Big Add: {'—' if ce_big_add is None else f'{ce_big_add.strike:.0f} (+{ce_big_add.ce_oi_chg:,.0f})'}")
            st.write(f"Big Drop: {'—' if ce_big_drop is None else f'{ce_big_drop.strike:.0f} ({ce_big_drop.ce_oi_chg:,.0f})'}")
        with c2:
            st.markdown("**Puts (PE)**")
            st.write(f"Big Add: {'—' if pe_big_add is None else f'{pe_big_add.strike:.0f} (+{pe_big_add.pe_oi_chg:,.0f})'}")
            st.write(f"Big Drop: {'—' if pe_big_drop is None else f'{pe_big_drop.strike:.0f} ({pe_big_drop.pe_oi_chg:,.0f})'}")

        if last_pe and last_ce:
            if last_pe > prev_pe and last_ce > prev_ce:
                st.success("⚡ Support & Resistance both moving UP → bullish setup.")
            elif last_pe < prev_pe and last_ce < prev_ce:
                st.error("⚠️ Support & Resistance both moving DOWN → bearish setup.")
            else:
                st.info("🔄 Mixed shift → rangebound/volatile setup.")
    else:
        st.info("Not enough snapshots in this session to detect seller shifting.")

    # === CE vs PE Premium Crossover around ATM ===
    st.subheader("📈 CE–PE Premium Crossover (ATM ± N Strikes)")
    atm_cross = infer_atm_strike(latest_df)
    strikes_window = [atm_cross + i * STRIKE_STEP for i in range(-near_strikes, near_strikes + 1)] if np.isfinite(atm_cross) else []
    cross_df = latest_df[latest_df["strike"].isin(strikes_window)].copy() if strikes_window else pd.DataFrame()

    if cross_df.empty:
        st.warning("No strikes found in ATM window.")
    else:
        cross_df["premium_diff"] = cross_df["ce_ltp"].fillna(0) - cross_df["pe_ltp"].fillna(0)
        fig_cross = go.Figure()
        fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["ce_ltp"], mode="lines+markers", name="CE Premium"))
        fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["pe_ltp"], mode="lines+markers", name="PE Premium"))
        fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["premium_diff"], mode="lines+markers", name="CE-PE Diff"))
        fig_cross.update_layout(title=f"CE vs PE Premium Crossover (ATM {atm_cross:.0f} ± {near_strikes})", xaxis_title="Strike", yaxis_title="Premium")
        st.plotly_chart(fig_cross, use_container_width=True)
        if cross_df["premium_diff"].notna().any():
            max_strike = cross_df.loc[cross_df["premium_diff"].idxmax(), "strike"]
            min_strike = cross_df.loc[cross_df["premium_diff"].idxmin(), "strike"]
            st.info(f"➡️ CE stronger vs PE near {max_strike:.0f}, while PE premium dominates near {min_strike:.0f}.")

    # === Short Buildup near ATM (CE) — highlighted table ===
    st.subheader("⚠️ Short Buildup Near ATM (CE)")
    if np.isfinite(atm):
        short_ce_near_atm = latest_df[
            (latest_df["strike"].between(atm - 250, atm + 250)) &
            ((latest_df["ce_oi_chg"] > 0) & (latest_df["ce_ltp_chg"] < 0))
        ][["strike", "ce_oi", "ce_oi_chg", "ce_ltp", "ce_ltp_chg"]].copy()
        if not short_ce_near_atm.empty:
            if HAS_AGGRID:
                # AgGrid coloring for the Buildup column
                short_ce_near_atm["Buildup"] = "Short Buildup"
                cell_style_jscode = JsCode(
                    """
                    function(params) {
                        if (params.value == "Short Buildup") {return {color: 'white', backgroundColor: 'red'};}
                        return {color: 'black', backgroundColor: 'white'};
                    }
                    """
                )
                gb = GridOptionsBuilder.from_dataframe(short_ce_near_atm)
                gb.configure_columns(["Buildup"], cellStyle=cell_style_jscode)
                gridOptions = gb.build()
                AgGrid(short_ce_near_atm, gridOptions=gridOptions, fit_columns_on_grid_load=True)
            else:
                st.dataframe(short_ce_near_atm.style.apply(
                    lambda s: ["background-color: red; color: white" for _ in s] if s.name == "Buildup" else ["" for _ in s],
                    axis=0
                ))
        else:
            st.info("No Short Buildup detected near ATM in the latest snapshot.")

# ------------------------
# RAW SNAPSHOT (debug/inspection)
# ------------------------
with st.expander("🔎 Raw Latest Snapshot"):
    st.dataframe(df_en.sort_values(["strike", "option_type"]))

st.markdown("---")
st.caption("This online dashboard fetches NSE option-chain live and refreshes automatically. Intraday trends are from in-session history only. Use with caution.")



# ---------- Feature helpers ----------
def get_atm_strike(spot, step):
    try:
        return int(round(spot / step) * step)
    except Exception:
        return np.nan

def subset_atm_band(df_wide, atm, k, step):
    if pd.isna(atm):
        return df_wide.copy()
    lo = atm - k*step
    hi = atm + k*step
    return df_wide[(df_wide['strike'] >= lo) & (df_wide['strike'] <= hi)].copy()

def build_atm_band_table(wide_latest, k, step):
    cols = ['strike','ce_oi','pe_oi','ce_oi_chg','pe_oi_chg','ce_ltp','pe_ltp','ce_buildup','pe_buildup']
    have = [c for c in cols if c in wide_latest.columns]
    tbl = wide_latest[have].copy()
    # per-strike totals and PCR
    if all(c in tbl.columns for c in ['ce_oi','pe_oi']):
        tbl['pcr'] = (tbl['pe_oi'] / tbl['ce_oi']).replace([np.inf, -np.inf], np.nan)
    # rename for display
    disp = tbl.rename(columns={
        'ce_oi':'CE OI','pe_oi':'PE OI',
        'ce_oi_chg':'CE OI Δ','pe_oi_chg':'PE OI Δ',
        'ce_ltp':'CE LTP','pe_ltp':'PE LTP',
        'ce_buildup':'CE Buildup','pe_buildup':'PE Buildup'
    })
    # totals
    totals = {
        'strike':'TOTAL',
        'CE OI': disp['CE OI'].sum() if 'CE OI' in disp else np.nan,
        'PE OI': disp['PE OI'].sum() if 'PE OI' in disp else np.nan,
        'CE OI Δ': disp['CE OI Δ'].sum() if 'CE OI Δ' in disp else np.nan,
        'PE OI Δ': disp['PE OI Δ'].sum() if 'PE OI Δ' in disp else np.nan,
    }
    disp_total = pd.concat([disp, pd.DataFrame([totals])], ignore_index=True)
    return disp_total

def top_trending_strikes(wide_latest, n=5):
    # rank by absolute OI change on both sides
    if not all(c in wide_latest.columns for c in ['ce_oi_chg','pe_oi_chg','strike']):
        return pd.DataFrame()
    t = wide_latest[['strike','ce_oi_chg','pe_oi_chg']].copy()
    t['abs_oi_chg'] = (t['ce_oi_chg'].fillna(0).abs() + t['pe_oi_chg'].fillna(0).abs())
    t = t.sort_values('abs_oi_chg', ascending=False).head(n)
    return t[['strike','ce_oi_chg','pe_oi_chg','abs_oi_chg']].rename(columns={'ce_oi_chg':'CE OI Δ','pe_oi_chg':'PE OI Δ','abs_oi_chg':'Total |ΔOI|'})

def strikewise_pcr(wide_latest):
    if not all(c in wide_latest.columns for c in ['ce_oi','pe_oi','strike']):
        return pd.DataFrame()
    df = wide_latest[['strike','ce_oi','pe_oi']].copy()
    df['PCR'] = (df['pe_oi'] / df['ce_oi']).replace([np.inf, -np.inf], np.nan)
    return df[['strike','ce_oi','pe_oi','PCR']].rename(columns={'ce_oi':'CE OI','pe_oi':'PE OI'})

def session_vwap(history_wide, side='ce'):
    # compute per-strike VWAP using session history: sum(price*vol)/sum(vol)
    price_col = f"{side}_ltp"
    vol_col = f"{side}_vol" if f"{side}_vol" in history_wide.columns else None
    if vol_col is None or price_col not in history_wide.columns:
        return pd.Series(dtype=float)
    hv = history_wide.dropna(subset=['strike']).copy()
    hv['pv'] = hv[price_col].fillna(0) * hv[vol_col].fillna(0)
    grp = hv.groupby('strike', as_index=True)
    vwap = grp['pv'].sum() / grp[vol_col].sum().replace(0, np.nan)
    return vwap

def last_3min_delta(history_wide):
    # Use last two snapshots spaced by ~3 minutes; if timestamps exist, pick two latest
    if history_wide.empty or 'ts' not in history_wide.columns:
        return {}
    h = history_wide.sort_values('ts')
    last = h['ts'].max()
    prev = h[h['ts'] < last]['ts'].max()
    if pd.isna(prev):
        return {}
    d = {}
    for col in ['ce_oi','pe_oi','ce_oi_chg','pe_oi_chg']:
        if col in h.columns:
            v_last = h[h['ts']==last][col].sum()
            v_prev = h[h['ts']==prev][col].sum()
            d[col] = float(v_last - v_prev)
    return d




# ---------- New Feature Sections ----------
try:
    # Determine ATM and band subset
    step = STRIKE_STEP if 'STRIKE_STEP' in globals() else 50
    k = near_strikes if 'near_strikes' in globals() else 3
    spot = float(wide_latest['spot'].iloc[0]) if 'spot' in wide_latest.columns else np.nan
    atm = get_atm_strike(spot, step)
    band = subset_atm_band(wide_latest, atm, k, step)

    # 1) ATM±k table with totals (Trending OI = sum of |CE OI Δ| + |PE OI Δ| in band)
    st.subheader(f"ATM±{k} Strikes Summary (ATM ≈ {atm})")
    atm_tbl = build_atm_band_table(band, k, step)
    if 'CE OI Δ' in atm_tbl.columns and 'PE OI Δ' in atm_tbl.columns:
        trending_oi = float(atm_tbl.iloc[:-1]['CE OI Δ'].abs().sum() + atm_tbl.iloc[:-1]['PE OI Δ'].abs().sum())
        st.caption(f"Trending OI (band): {int(trending_oi):,}")
    st.dataframe(atm_tbl, use_container_width=True)

    # 2) Top trending strikes by |ΔOI|
    st.subheader("Top Trending Strikes (by |ΔOI|)")
    top_tr = top_trending_strikes(wide_latest, n=8)
    if not top_tr.empty:
        st.dataframe(top_tr, use_container_width=True)
    else:
        st.info("Insufficient fields to compute trending strikes.")

    # 3) Strike-wise PCR
    st.subheader("Strike-wise PCR")
    pcr_tbl = strikewise_pcr(wide_latest)
    if not pcr_tbl.empty:
        st.dataframe(pcr_tbl, use_container_width=True)
    else:
        st.info("Insufficient fields for PCR.")

    # 4) CE/PE side-by-side with session VWAP and line chart markers (above/below VWAP)
    st.subheader("CE vs PE with Session VWAP (above/below)")
    # Compute VWAP from session history if available
    vw_ce = session_vwap(st.session_state.get('history_wide', pd.DataFrame()), side='ce')
    vw_pe = session_vwap(st.session_state.get('history_wide', pd.DataFrame()), side='pe')
    combo = wide_latest[['strike','ce_ltp','pe_ltp']].copy() if set(['strike','ce_ltp','pe_ltp']).issubset(wide_latest.columns) else pd.DataFrame()
    if not combo.empty:
        combo['CE VWAP'] = combo['strike'].map(vw_ce)
        combo['PE VWAP'] = combo['strike'].map(vw_pe)
        combo['CE vs VWAP'] = np.where(combo['ce_ltp'] > combo['CE VWAP'], 'Above', 'Below')
        combo['PE vs VWAP'] = np.where(combo['pe_ltp'] > combo['PE VWAP'], 'Above', 'Below')
        st.dataframe(combo.rename(columns={'ce_ltp':'CE LTP','pe_ltp':'PE LTP'}), use_container_width=True)
        st.caption("Tip: Strikes marked 'Above' may indicate strength vs. session average price.")
    else:
        st.info("Need LTP columns to compute VWAP markers.")

    # 5) Three-line detail from last ~3 minutes OI delta
    deltas = last_3min_delta(st.session_state.get('history_wide', pd.DataFrame()))
    if deltas:
        line1 = f"CE OI Δ (≈3m): {int(deltas.get('ce_oi',0)):,} | PE OI Δ (≈3m): {int(deltas.get('pe_oi',0)):,}"
        line2 = f"CE |ΔOI| (≈3m): {int(abs(deltas.get('ce_oi_chg',0))):,} | PE |ΔOI| (≈3m): {int(abs(deltas.get('pe_oi_chg',0))):,}"
        bias = 'Bullish' if deltas.get('pe_oi',0) > deltas.get('ce_oi',0) else 'Bearish' if deltas.get('ce_oi',0) > deltas.get('pe_oi',0) else 'Neutral'
        line3 = f"Bias: {bias} (based on relative OI increase)"
        st.markdown(f"**Quick 3-min Take:**  "+line1+"  "+line2+"  "+line3)
    else:
        st.caption("Waiting for at least two snapshots to compute 3‑minute deltas.")

except Exception as e:
    st.warning(f"Feature section error: {e}")

