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

# Optional: st_aggrid (only used if available)
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
SYMBOL = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(TIMEZONE)
REFRESH_SECS_DEFAULT = 180
STRIKE_STEP = 50
NEAR_STRIKES_DEFAULT = 3
MAX_HISTORY_POINTS = 480
ATM_TOP_STRIKE_SPAN = 5   # Top strikes limited to ATM ± 5

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
    Columns: symbol, strike, option_type, ltp, oi, volume, iv, ts, spot, expiry
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
            expiry = leg.get("expiryDate") or item.get("expiryDate")
            rows.append({
                "symbol": symbol,
                "strike": int(strike),
                "option_type": side,
                "ltp": float(leg.get("lastPrice") or 0.0),
                "oi": float(leg.get("openInterest") or 0.0),
                "volume": float(leg.get("totalTradedVolume") or 0.0),
                "iv": float(leg.get("impliedVolatility") or 0.0),
                "vwap": np.nan,  # not in this endpoint
                "ts": ts,
                "spot": float(spot or 0.0),
                "expiry": str(expiry) if expiry else None,
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
    return df

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
    # one-interval changes
    wide = wide.sort_values(["ts", "strike"]).reset_index(drop=True)
    wide["ce_oi_chg"] = wide.groupby("strike")["ce_oi"].diff()
    wide["pe_oi_chg"] = wide.groupby("strike")["pe_oi"].diff()
    wide["ce_ltp_chg"] = wide.groupby("strike")["ce_ltp"].diff()
    wide["pe_ltp_chg"] = wide.groupby("strike")["pe_ltp"].diff()
    return wide

def infer_atm_strike(wide_latest: pd.DataFrame) -> float:
    spot = float(wide_latest["spot"].median()) if "spot" in wide_latest.columns and wide_latest["spot"].notna().any() else np.nan
    if np.isfinite(spot):
        diffs = (wide_latest["strike"] - spot).abs()
        return float(wide_latest.loc[diffs.idxmin(), "strike"])
    ssum = (wide_latest["ce_ltp"].fillna(0) + wide_latest["pe_ltp"].fillna(0))
    return float(wide_latest.loc[ssum.idxmin(), "strike"]) if not wide_latest.empty else np.nan

# ---------- Expiry helper ----------
def _pick_default_expiry(expiries: List[str]) -> Optional[str]:
    """Choose the nearest upcoming expiry (today or later)."""
    if not expiries:
        return None
    ex = pd.to_datetime(pd.Series(expiries), errors="coerce")
    today = pd.Timestamp.now(tz=IST).normalize()
    try:
        future_mask = ex.dt.date >= today.date()
        if future_mask.any():
            return str(pd.Series(expiries)[future_mask].iloc[0])
    except Exception:
        pass
    order = ex.sort_values(kind="mergesort").index
    return str(pd.Series(expiries).iloc[order[0]])

def split_weekly_monthly(expiries: List[str]) -> Dict[str, List[str]]:
    """Last expiry date of a month = Monthly, others = Weekly."""
    if not expiries:
        return {"weekly": [], "monthly": [], "all": []}
    ex = pd.to_datetime(pd.Series(expiries), errors="coerce")
    df = pd.DataFrame({"raw": expiries, "dt": ex}).dropna(subset=["dt"])
    df["ym"] = df["dt"].dt.to_period("M")
    last_per_month = df.groupby("ym")["dt"].transform("max")
    df["is_monthly"] = df["dt"].eq(last_per_month)
    monthly = df.loc[df["is_monthly"], "raw"].astype(str).tolist()
    weekly = df.loc[~df["is_monthly"], "raw"].astype(str).tolist()
    return {"weekly": weekly, "monthly": monthly, "all": list(pd.Series(expiries).astype(str).unique())}

# ---------- Styling helpers (Green/Red for bullish/bearish) ----------
BULLISH_BUILDS = {"Long Buildup", "Short Covering"}
BEARISH_BUILDS = {"Short Buildup", "Long Unwinding"}

def style_oi_change(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    cols_to_color = [c for c in ["oi_chg", "oi_chg_pct", "CE OI Δ", "PE OI Δ", "CE LTP Δ", "PE LTP Δ"] if c in df.columns]
    def _row_style(row):
        # pick any available buildup marker on the row
        b = row.get("buildup", row.get("CE Buildup", row.get("PE Buildup", "Neutral")))
        color = ""
        if b in BULLISH_BUILDS:
            color = "background-color:#e8f5e9;color:#1b5e20;"
        elif b in BEARISH_BUILDS:
            color = "background-color:#ffebee;color:#b71c1c;"
        return [color if c in cols_to_color else "" for c in df.columns]
    return df.style.apply(_row_style, axis=1)

# ------------------------
# SIDEBAR / SETTINGS
# ------------------------
st.sidebar.title("Settings")
refresh_secs = st.sidebar.number_input("Auto-refresh (seconds)", min_value=30, max_value=900, value=REFRESH_SECS_DEFAULT, step=30)
near_strikes = st.sidebar.slider("Strikes near ATM (±)", 1, 6, NEAR_STRIKES_DEFAULT)
trending_k = st.sidebar.slider("Trending OI window (ATM ± strikes)", 5, 7, 5)
oi_alert_pct = st.sidebar.slider("Exceptional OI% threshold", 5, 500, 50)
st.sidebar.markdown("---")
# client-side auto refresh
# (Use st.toggle if your Streamlit version doesn't support st.sidebar.toggle)
if getattr(st.sidebar, "toggle", None):
    if st.sidebar.toggle("Auto refresh every n seconds", value=True):
        st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh")
else:
    if st.checkbox("Auto refresh every n seconds", value=True):
        st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh")

st.sidebar.button("🔄 Refresh now", on_click=lambda: st.cache_data.clear())

# ------------------------
# FETCH & ENRICH
# ------------------------
st.title(f"📈 {SYMBOL} Options Live — Online Dashboard")
_now_ist = dt.datetime.now(IST).time()
if not (dt.time(9, 14) <= _now_ist <= dt.time(15, 31)):
    st.info("Market appears closed. Data may be stale or unchanged.")
status = st.empty()

if "prev_snapshot" not in st.session_state:
    st.session_state.prev_snapshot = None
if "history_wide" not in st.session_state:
    st.session_state.history_wide = pd.DataFrame()

try:
    status.info("Fetching option-chain from NSE…")
    curr = fetch_option_chain(SYMBOL)

    # ---- Expiry selector (with Weekly/Monthly toggle) ----
    if ("expiry" in curr.columns) and (not curr.empty):
        _exps_all = (
            pd.Series(curr["expiry"].dropna().astype(str).unique())
            .sort_values(key=lambda s: pd.to_datetime(s, errors="coerce"))
            .tolist()
        )
        buckets = split_weekly_monthly(_exps_all)
        exp_type = st.sidebar.radio("Expiry type", ["All", "Weekly", "Monthly"], index=0, horizontal=True)
        if exp_type == "Weekly":
            _exps = buckets["weekly"]
        elif exp_type == "Monthly":
            _exps = buckets["monthly"]
        else:
            _exps = buckets["all"]
        if not _exps:
            _exps = _exps_all
        _default = _pick_default_expiry(_exps) or (_exps[0] if _exps else None)
        _default_idx = _exps.index(_default) if _default in _exps else 0
        chosen_expiry = st.sidebar.selectbox("Expiry", _exps, index=_default_idx)
        curr = curr[curr["expiry"].astype(str) == str(chosen_expiry)].copy()
    else:
        st.sidebar.caption("Expiry filter unavailable (no 'expiry' column).")

    if curr.empty:
        status.error("Failed to fetch data (empty). Try again shortly.")
        st.stop()
    status.success("Fetched live option chain.")
except Exception as e:
    status.error(f"Fetch failed: {e}")
    st.stop()

# enrich vs previous snapshot
prev = st.session_state.prev_snapshot
df_en = enrich_with_prev(curr, prev)
st.session_state.prev_snapshot = curr[["symbol", "strike", "option_type", "ltp", "oi"]].copy()

# accumulate history in wide format
wide_latest = make_wide(df_en)
if not wide_latest.empty:
    hist = st.session_state.history_wide
    st.session_state.history_wide = pd.concat([hist, wide_latest], ignore_index=True)
    if len(st.session_state.history_wide) > MAX_HISTORY_POINTS:
        st.session_state.history_wide = st.session_state.history_wide.tail(MAX_HISTORY_POINTS).reset_index(drop=True)

# ------------------------
# HEADER METRICS & SENTIMENT BASICS
# ------------------------
spot = float(df_en["spot"].dropna().iloc[0]) if "spot" in df_en.columns and not df_en["spot"].dropna().empty else 0.0
snapshot_time = pd.to_datetime(df_en["ts"].iloc[0]) if "ts" in df_en.columns else dt.datetime.now(IST)

ce = df_en[df_en.option_type == "CE"].copy()
pe = df_en[df_en.option_type == "PE"].copy()
ce_strength = float((ce["volume"].fillna(0) * ce["ltp_chg"].abs().fillna(0)).sum())
pe_strength = float((pe["volume"].fillna(0) * pe["ltp_chg"].abs().fillna(0)).sum())

total_strength = (ce_strength + pe_strength) or 1
buyer_pct = 100 * ce_strength / total_strength
seller_pct = 100 * pe_strength / total_strength

up_count = ((df_en["buildup"] == "Long Buildup") | (df_en["buildup"] == "Short Covering")).sum()
down_count = ((df_en["buildup"] == "Short Buildup") | (df_en["buildup"] == "Long Unwinding")).sum()

sent_score = 0.6 * (buyer_pct - seller_pct) + 0.4 * (up_count - down_count)
if sent_score > 15:
    sentiment_label, sentiment_color = "Bullish", "#1b5e20"
elif sent_score < -15:
    sentiment_label, sentiment_color = "Bearish", "#b71c1c"
else:
    sentiment_label, sentiment_color = "Neutral", "#263238"

try:
    ce_oi_total = float(df_en[df_en.option_type == "CE"]["oi"].sum())
    pe_oi_total = float(df_en[df_en.option_type == "PE"]["oi"].sum())
    pcr_val = pe_oi_total / ce_oi_total if ce_oi_total > 0 else np.nan
except Exception:
    pcr_val = np.nan

c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
with c1:
    # Two-line markdown; close the string properly to avoid unterminated f-string errors.
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
        <div style="margin-top:6px;padding:10px;border-radius:10px;background:{sentiment_color};color:white;text-align:center">
        <strong>Market Sentiment</strong><br><span style="font-size:18px">{sentiment_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ------------------------
# NEAR-ATM VIEW (colored by buildup)
# ------------------------
def select_near_atm(df: pd.DataFrame, spot: float, n: int = NEAR_STRIKES_DEFAULT) -> pd.DataFrame:
    if df.empty:
        return df
    atm_ = nearest_strike(spot)
    lo, hi = atm_ - n * STRIKE_STEP, atm_ + n * STRIKE_STEP
    return df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()

near = select_near_atm(df_en, spot, n=near_strikes)
st.subheader(f"Strikes around ATM (±{near_strikes * STRIKE_STEP} points)")
if near.empty:
    st.warning("No near-ATM data available.")
else:
    show_cols = ["strike", "option_type", "ltp", "iv", "oi", "oi_chg_pct", "ltp_chg_pct", "buildup"]
    _near_tbl = near.sort_values(["strike", "option_type"])[show_cols]
    try:
        st.dataframe(style_oi_change(_near_tbl), use_container_width=True)
    except Exception:
        st.dataframe(_near_tbl, use_container_width=True)

# ------------------------
# TRENDING OI — PICK ONE STRIKE (one-interval view)
# ------------------------
st.markdown("---")
st.subheader("🔎 Trending OI — pick a strike (interval vs previous)")

available_strikes = sorted(df_en["strike"].unique()) if not df_en.empty else []
sel_str_index = len(available_strikes) // 2 if available_strikes else 0
sel_str = st.selectbox("Select strike", options=available_strikes, index=sel_str_index)

if sel_str:
    s = df_en[(df_en.strike == sel_str)]
    ce_row = s[s.option_type == "CE"].squeeze() if not s[s.option_type == "CE"].empty else None
    pe_row = s[s.option_type == "PE"].squeeze() if not s[s.option_type == "PE"].empty else None

    def safe_val(row, col, default=0.0):
        try:
            if row is None:
                return float(default)
            if isinstance(row, dict):
                v = row.get(col, default)
            elif hasattr(row, 'index') and not hasattr(row, 'columns'):
                v = row[col] if col in row.index else default
            elif hasattr(row, 'columns'):
                if col in row.columns and len(row) > 0:
                    v = row.iloc[0][col]
                else:
                    v = default
            else:
                v = getattr(row, col, default)
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
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;background:#e8f5e9'>"
            f"<strong>CE OI</strong><br><span style='font-size:20px'>{int(ce_oi):,}</span>"
            f"<br><small>Δ {int(ce_oi_chg):+,}</small></div>",
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;background:#ffebee'>"
            f"<strong>PE OI</strong><br><span style='font-size:20px'>{int(pe_oi):,}</span>"
            f"<br><small>Δ {int(pe_oi_chg):+,}</small></div>",
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;background:#e3f2fd'>"
            f"<strong>CE LTP</strong><br><span style='font-size:20px'>{ce_ltp:.2f}</span>"
            f"<br><small>Δ {ce_ltp_chg:+.2f} ({ce_ltp_chg_pct:+.1f}%)</small></div>",
            unsafe_allow_html=True,
        )
    with a4:
        st.markdown(
            f"<div style='padding:10px;border-radius:8px;background:#fff8e1'>"
            f"<strong>PE LTP</strong><br><span style='font-size:20px'>{pe_ltp:.2f}</span>"
            f"<br><small>Δ {pe_ltp_chg:+.2f} ({pe_ltp_chg_pct:+.1f}%)</small></div>",
            unsafe_allow_html=True,
        )

# ------------------------
# MASTER/INTRADAY HISTORY VIEWS
# ------------------------
if not st.session_state.history_wide.empty:
    hist = st.session_state.history_wide.copy()

    latest_ts = hist["ts"].max()
    latest_df = hist[hist["ts"] == latest_ts].copy()

    # === Header block 2: ATM, spot, snapshots ===
    atm = infer_atm_strike(latest_df)
    spot_latest = float(latest_df["spot"].median()) if latest_df["spot"].notna().any() else np.nan

    c1, c2, c3 = st.columns(3)
    c1.metric("ATM", f"{atm:.0f}" if np.isfinite(atm) else "—")
    c2.metric("Spot (median)", f"{spot_latest:,.2f}" if np.isfinite(spot_latest) else "—")
    c3.metric("Snapshots stored", str(hist["ts"].nunique()))

    # Restrict Top Strikes to ATM ± ATM_TOP_STRIKE_SPAN
    if np.isfinite(atm):
        allowed = [atm + i * STRIKE_STEP for i in range(-ATM_TOP_STRIKE_SPAN, ATM_TOP_STRIKE_SPAN + 1)]
        latest_df_win = latest_df[latest_df["strike"].isin(allowed)].copy()
    else:
        latest_df_win = latest_df.copy()

    # === Top Strikes (ATM window) ===
    st.subheader("🏆 Top Strikes — OI & Price (ATM window)")
    top_n = st.slider("Top N", 5, 15, 10)
    t1, t2 = st.columns(2)
    with t1:
        st.write(f"Top CE OI (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "ce_oi")[ ["strike", "ce_oi", "ce_oi_chg"] ])
        st.write(f"Top CE Price (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "ce_ltp")[ ["strike", "ce_ltp", "ce_ltp_chg"] ])
    with t2:
        st.write(f"Top PE OI (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "pe_oi")[ ["strike", "pe_oi", "pe_oi_chg"] ])
        st.write(f"Top PE Price (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "pe_ltp")[ ["strike", "pe_ltp", "pe_ltp_chg"] ])

    # === Buildup (ATM ±10, PE ◀︎ | ▶︎ CE) ===
    st.subheader("🧮 Buildup (ATM ±10) — PE ◀︎ | ▶︎ CE")
    band10 = [atm + i * STRIKE_STEP for i in range(-10, 11)] if np.isfinite(atm) else []
    bt = latest_df[latest_df["strike"].isin(band10)].copy() if band10 else pd.DataFrame()
    if not bt.empty:
        bt["PCR"] = (bt["pe_oi"] / bt["ce_oi"]).replace([np.inf, -np.inf], np.nan)
        bt["PE Buildup"] = [classify_buildup(o, p) for o, p in zip(bt["pe_oi_chg"], bt["pe_ltp_chg"])]
        bt["CE Buildup"] = [classify_buildup(o, p) for o, p in zip(bt["ce_oi_chg"], bt["ce_ltp_chg"])]
        display_cols = [
            "strike",
            "pe_oi", "pe_oi_chg", "pe_ltp_chg", "PE Buildup",
            "PCR",
            "ce_oi", "ce_oi_chg", "ce_ltp_chg", "CE Buildup",
        ]
        bt_disp = bt[display_cols].rename(columns={
            "pe_oi": "PE OI", "pe_oi_chg": "PE OI Δ", "pe_ltp_chg": "PE LTP Δ",
            "ce_oi": "CE OI", "ce_oi_chg": "CE OI Δ", "ce_ltp_chg": "CE LTP Δ",
        })

        def style_buildup_dual(df: pd.DataFrame) -> pd.io.formats.style.Styler:
            def _row_style(row):
                ce_b = row.get("CE Buildup", "Neutral")
                pe_b = row.get("PE Buildup", "Neutral")
                styles = []
                for col in df.columns:
                    color = ""
                    if col in ("PE OI Δ", "PE LTP Δ"):
                        if pe_b in BULLISH_BUILDS:
                            color = "background-color:#e8f5e9;color:#1b5e20;"
                        elif pe_b in BEARISH_BUILDS:
                            color = "background-color:#ffebee;color:#b71c1c;"
                    if col in ("CE OI Δ", "CE LTP Δ"):
                        if ce_b in BULLISH_BUILDS:
                            color = "background-color:#e8f5e9;color:#1b5e20;"
                        elif ce_b in BEARISH_BUILDS:
                            color = "background-color:#ffebee;color:#b71c1c;"
                    styles.append(color)
                return styles
            return df.style.apply(_row_style, axis=1)

        try:
            st.dataframe(style_buildup_dual(bt_disp), use_container_width=True)
        except Exception:
            st.dataframe(bt_disp, use_container_width=True)
    else:
        st.info("No strikes in ATM ±10 window yet.")

    # === OI Distribution (Green=PE, Red=CE) ===
    st.subheader("📊 OI Distribution (Green=PE writers, Red=CE writers)")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=latest_df["ce_oi"], name="CE OI", marker_color="#e53935", text=latest_df["ce_oi"], textposition="auto"))
    fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=latest_df["pe_oi"], name="PE OI", marker_color="#43a047", text=latest_df["pe_oi"], textposition="auto"))
    fig_dist.update_layout(barmode="group", xaxis_title="Strike", yaxis_title="Open Interest")
    st.plotly_chart(fig_dist, use_container_width=True)

    tot_ce = float(latest_df["ce_oi"].sum()) if "ce_oi" in latest_df else 0.0
    tot_pe = float(latest_df["pe_oi"].sum()) if "pe_oi" in latest_df else 0.0
    d1, d2, d3 = st.columns(3)
    d1.metric("Total CE OI", f"{int(tot_ce):,}")
    d2.metric("Total PE OI", f"{int(tot_pe):,}")
    diff = tot_pe - tot_ce
    d3.metric("PE − CE (OI)", f"{int(diff):,}", delta=f"{int(diff):,}")

    diff_tbl = latest_df[["strike", "ce_oi", "pe_oi"]].copy()
    diff_tbl["PE − CE"] = diff_tbl["pe_oi"].fillna(0) - diff_tbl["ce_oi"].fillna(0)
    st.dataframe(diff_tbl.sort_values("strike"), use_container_width=True)

    # === CE–PE Premium Crossover around ATM ===
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

    # === Trending OI (ATM ± user-selected strikes — intraday from session history) ===
    st.subheader("⏱️ Trending OI (ATM ± user-selected strikes)")
    def _within_mkt(ts: pd.Timestamp) -> bool:
        if not isinstance(ts, (pd.Timestamp, dt.datetime)):
            return False
        t = pd.Timestamp(ts).tz_localize(None).time() if isinstance(ts, pd.Timestamp) and ts.tzinfo else pd.Timestamp(ts).time()
        return dt.time(9, 15) <= t <= dt.time(15, 30)

    day_hist = hist[hist["ts"].apply(_within_mkt)]
    if not day_hist.empty:
        first_ts = day_hist["ts"].min()
        base_atm = infer_atm_strike(day_hist[day_hist["ts"] == first_ts])
        strikes_window = [base_atm + i * STRIKE_STEP for i in range(-trending_k, trending_k + 1)] if np.isfinite(base_atm) else []
        day_win = day_hist[day_hist["strike"].isin(strikes_window)] if strikes_window else pd.DataFrame()
        if not day_win.empty:
            tot_by_ts = day_win.groupby("ts", as_index=False).agg(ce_oi=("ce_oi", "sum"), pe_oi=("pe_oi", "sum"))
            fig_day = go.Figure()
            fig_day.add_trace(go.Scatter(x=tot_by_ts["ts"], y=tot_by_ts["ce_oi"], name="CE OI"))
            fig_day.add_trace(go.Scatter(x=tot_by_ts["ts"], y=tot_by_ts["pe_oi"], name="PE OI"))
            st.plotly_chart(fig_day, use_container_width=True)

# ------------------------
# RAW SNAPSHOT (debug/inspection)
# ------------------------
with st.expander("🔎 Raw Latest Snapshot"):
    st.dataframe(df_en.sort_values(["strike", "option_type"]))

st.markdown("---")
st.caption("This dashboard fetches NSE option-chain live and refreshes automatically. Intraday trends are from in-session history only. Use with caution.")
