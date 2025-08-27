# app.py
from __future__ import annotations
import math
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
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

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(layout="wide", page_title="NIFTY Options Live — Online", page_icon="📈")

# ========================
# CONSTANTS / DEFAULTS
# ========================
TIMEZONE = "Asia/Kolkata"
IST = pytz.timezone(TIMEZONE)

DEFAULT_SYMBOL = "NIFTY"
STRIKE_STEP = 50
REFRESH_SECS_DEFAULT = 180
NEAR_STRIKES_DEFAULT = 3           # for alerting (ATM ±3)
PANEL_STRIKES_DEFAULT = 7          # for CE/PE panel (ATM ±7)
MAX_HISTORY_POINTS = 2000
ATM_TOP_STRIKE_SPAN = 5            # top table window
DEFAULT_RISK_FREE = 0.0675         # 6.75% annual
DEFAULT_DIV_YIELD = 0.0

BULLISH_BUILDS = {"Long Buildup", "Short Covering"}
BEARISH_BUILDS = {"Short Buildup", "Long Unwinding"}

# ========================
# PERSISTENT SETTINGS (Telegram)
# ========================
TG_SETTINGS_PATH = Path.home() / ".streamlit" / "tg_settings.json"

def load_tg_settings() -> Dict[str, str | bool]:
    try:
        if TG_SETTINGS_PATH.exists():
            return json.loads(TG_SETTINGS_PATH.read_text())
    except Exception:
        pass
    return {"enable": False, "token": "", "chat_id": ""}

def save_tg_settings(data: Dict[str, str | bool]) -> bool:
    try:
        TG_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TG_SETTINGS_PATH.write_text(json.dumps(data))
        return True
    except Exception as e:
        st.sidebar.warning(f"Couldn't save Telegram settings: {e}")
        return False

# ========================
# UTILS
# ========================
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
def fetch_option_chain(symbol: str) -> pd.DataFrame:
    """Return tidy long DataFrame with: symbol, strike, option_type, ltp, oi, volume_total, iv, ts, spot, expiry"""
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    s = make_session()
    headers = {"accept": "application/json, text/plain, */*", "referer": "https://www.nseindia.com/option-chain"}
    data: Dict = {}
    for attempt in range(5):
        try:
            r = s.get(url, headers=headers, timeout=12)
            if r.status_code == 200:
                data = r.json()
                break
        except Exception:
            pass
        time.sleep(1.2 * (attempt + 1))
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
                "volume_total": float(leg.get("totalTradedVolume") or 0.0),  # cumulative
                "iv": float(leg.get("impliedVolatility") or 0.0),
                "ts": ts,
                "spot": float(spot or 0.0),
                "expiry": str(expiry) if expiry else None,
            })
    return pd.DataFrame(rows)

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

# Robust enrich_with_prev (handles absence of volume_total in prev)
def enrich_with_prev(curr: pd.DataFrame, prev: Optional[pd.DataFrame]) -> pd.DataFrame:
    if curr.empty:
        return curr

    df = curr.copy()
    curr_vol_col = "volume_total" if "volume_total" in df.columns else ("volume" if "volume" in df.columns else None)

    if prev is None or getattr(prev, "empty", True):
        df["prev_ltp"] = df["ltp"]
        df["prev_oi"] = df["oi"]
        df["prev_volume_total"] = df[curr_vol_col] if curr_vol_col else 0.0
    else:
        prev_vol_col = "volume_total" if "volume_total" in prev.columns else ("volume" if "volume" in prev.columns else None)
        merge_cols = ["symbol", "strike", "option_type", "ltp", "oi"]
        if prev_vol_col:
            merge_cols.append(prev_vol_col)
        m = prev[merge_cols].rename(columns={"ltp": "prev_ltp", "oi": "prev_oi"})
        if prev_vol_col:
            m.rename(columns={prev_vol_col: "prev_volume_total"}, inplace=True)
        else:
            m["prev_volume_total"] = 0.0
        df = df.merge(m, on=["symbol", "strike", "option_type"], how="left")
        df["prev_ltp"] = df["prev_ltp"].fillna(df["ltp"])
        df["prev_oi"] = df["prev_oi"].fillna(df["oi"])
        if curr_vol_col:
            df["prev_volume_total"] = df["prev_volume_total"].fillna(df[curr_vol_col]).fillna(0.0)
        else:
            df["prev_volume_total"] = df["prev_volume_total"].fillna(0.0)

    df["oi_chg"] = df["oi"] - df["prev_oi"]
    df["ltp_chg"] = df["ltp"] - df["prev_ltp"]
    df["oi_chg_pct"] = np.where(df["prev_oi"] > 0, 100 * df["oi_chg"] / df["prev_oi"], np.where(df["oi_chg"] > 0, 100, 0))
    df["ltp_chg_pct"] = np.where(df["prev_ltp"] > 0, 100 * df["ltp_chg"] / df["prev_ltp"], 0.0)
    df["vol_flow"] = (df[curr_vol_col] - df["prev_volume_total"]).clip(lower=0) if curr_vol_col else 0.0
    df["buildup"] = [classify_buildup(o, p) for o, p in zip(df["oi_chg"], df["ltp_chg"])]
    return df

def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    keep = ["ts", "spot", "strike", "option_type", "oi", "ltp", "iv", "vol_flow", "volume_total"]
    df = df_long[keep].copy()
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=["strike"])

    piv_oi = df.pivot_table(index=["ts", "spot", "strike"], columns="option_type", values="oi", aggfunc="sum")
    piv_ltp = df.pivot_table(index=["ts", "spot", "strike"], columns="option_type", values="ltp", aggfunc="mean")
    piv_iv = df.pivot_table(index=["ts", "spot", "strike"], columns="option_type", values="iv", aggfunc="mean")
    piv_vflow = df.pivot_table(index=["ts", "spot", "strike"], columns="option_type", values="vol_flow", aggfunc="sum")
    piv_vtot = df.pivot_table(index=["ts", "spot", "strike"], columns="option_type", values="volume_total", aggfunc="sum")

    wide = pd.concat([
        piv_oi.rename(columns={"CE": "ce_oi", "PE": "pe_oi"}),
        piv_ltp.rename(columns={"CE": "ce_ltp", "PE": "pe_ltp"}),
        piv_iv.rename(columns={"CE": "ce_iv", "PE": "pe_iv"}),
        piv_vflow.rename(columns={"CE": "ce_vflow", "PE": "pe_vflow"}),
        piv_vtot.rename(columns={"CE": "ce_vtot", "PE": "pe_vtot"}),
    ], axis=1).reset_index()

    for c in ["ce_oi", "pe_oi", "ce_ltp", "pe_ltp", "ce_iv", "pe_iv", "ce_vflow", "pe_vflow", "ce_vtot", "pe_vtot"]:
        if c not in wide.columns:
            wide[c] = np.nan

    wide = wide.sort_values(["strike", "ts"]).reset_index(drop=True)
    wide["ce_oi_chg"] = wide.groupby("strike")["ce_oi"].diff()
    wide["pe_oi_chg"] = wide.groupby("strike")["pe_oi"].diff()
    wide["ce_ltp_chg"] = wide.groupby("strike")["ce_ltp"].diff()
    wide["pe_ltp_chg"] = wide.groupby("strike")["pe_ltp"].diff()
    return wide

def split_weekly_monthly(expiries: List[str]) -> Dict[str, List[str]]:
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

def pick_default_expiry(expiries: List[str]) -> Optional[str]:
    if not expiries:
        return None
    ex = pd.to_datetime(pd.Series(expiries), errors="coerce")
    today = pd.Timestamp.now(tz=IST).normalize()
    mask = ex.dt.date >= today.date()
    if mask.any():
        return str(pd.Series(expiries)[mask].iloc[0])
    order = ex.sort_values(kind="mergesort").index
    return str(pd.Series(expiries).iloc[order[0]])

def style_oi_change(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    cols_to_color = [c for c in ["oi_chg", "oi_chg_pct", "CE OI Δ", "PE OI Δ", "CE LTP Δ", "PE LTP Δ"] if c in df.columns]
    def _row_style(row):
        b = row.get("buildup", row.get("CE Buildup", row.get("PE Buildup", "Neutral")))
        color = ""
        if b in BULLISH_BUILDS:
            color = "background-color:#e8f5e9;color:#1b5e20;"
        elif b in BEARISH_BUILDS:
            color = "background-color:#ffebee;color:#b71c1c;"
        return [color if c in cols_to_color else "" for c in df.columns]
    return df.style.apply(_row_style, axis=1)

# ========================
# BLACK–SCHOLES
# ========================
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_theoretical(spot: float, strike: float, t_years: float, r: float, q: float, iv_pct: float, call: bool) -> float:
    if t_years <= 0 or iv_pct <= 0 or spot <= 0 or strike <= 0:
        return 0.0
    sigma = iv_pct / 100.0
    try:
        d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t_years) / (sigma * math.sqrt(t_years))
        d2 = d1 - sigma * math.sqrt(t_years)
        if call:
            return spot * math.exp(-q * t_years) * _norm_cdf(d1) - strike * math.exp(-r * t_years) * _norm_cdf(d2)
        else:
            return strike * math.exp(-r * t_years) * _norm_cdf(-d2) - spot * math.exp(-q * t_years) * _norm_cdf(-d1)
    except Exception:
        return 0.0

def years_to_expiry(expiry_str: str, now_ts: pd.Timestamp) -> float:
    try:
        d = pd.to_datetime(expiry_str)
        expiry_dt = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=15, minute=30, tz=IST)
        delta = max((expiry_dt - now_ts).total_seconds(), 0.0)
        return delta / (365.0 * 24 * 3600.0)
    except Exception:
        return 0.0

# ========================
# TELEGRAM
# ========================
def send_telegram(token: str, chat_id: str, text: str) -> Tuple[bool, str]:
    if not token or not chat_id:
        return False, "Token/Chat ID not set"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": chat_id, "text": text, "parse_mode": "HTML"}, timeout=10)
        if r.status_code == 200:
            return True, "ok"
        return False, f"HTTP {r.status_code}: {r.text[:120]}"
    except Exception as e:
        return False, str(e)

# ========================
# SIDEBAR
# ========================
st.sidebar.title("Settings")
SYMBOL = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"], index=0)
refresh_secs = st.sidebar.number_input("Auto-refresh (seconds)", min_value=30, max_value=900, value=REFRESH_SECS_DEFAULT, step=30)
near_strikes = st.sidebar.slider("Strikes near ATM for Alerts (±)", 1, 5, NEAR_STRIKES_DEFAULT)
panel_strikes = st.sidebar.slider("Strikes for CE/PE Panel (±)", 3, 10, PANEL_STRIKES_DEFAULT)
oi_alert_pct = st.sidebar.slider("Exceptional OI% threshold (3-min window)", 5, 500, 80)
risk_free = st.sidebar.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.2, value=DEFAULT_RISK_FREE, step=0.0025, format="%.4f")
div_yield = st.sidebar.number_input("Dividend yield (annual)", min_value=0.0, max_value=0.1, value=DEFAULT_DIV_YIELD, step=0.0025, format="%.4f")

st.sidebar.markdown("---")
enable_auto = st.sidebar.toggle("Auto refresh on", value=True)
if enable_auto:
    st_autorefresh(interval=int(refresh_secs * 1000), key="auto_refresh")

# Telegram (persistent)
st.sidebar.markdown("### Telegram Alerts")
_saved = load_tg_settings()
if "tg_settings" not in st.session_state:
    st.session_state.tg_settings = _saved.copy()

tg_enable = st.sidebar.toggle("Enable Telegram Alerts (every ~3 minutes)", value=st.session_state.tg_settings.get("enable", False), key="tg_enable")
tg_token = st.sidebar.text_input("Bot Token", type="password", value=st.session_state.tg_settings.get("token", ""), key="tg_token")
tg_chat = st.sidebar.text_input("Chat ID", value=st.session_state.tg_settings.get("chat_id", ""), key="tg_chat")

csa, csb = st.sidebar.columns(2)
if csa.button("💾 Save Telegram"):
    saved = {"enable": st.session_state.tg_enable, "token": st.session_state.tg_token, "chat_id": st.session_state.tg_chat}
    if save_tg_settings(saved):
        st.session_state.tg_settings = saved
        st.sidebar.success("Telegram settings saved.")
if csb.button("🧹 Clear Telegram"):
    if save_tg_settings({"enable": False, "token": "", "chat_id": ""}):
        st.session_state.tg_settings = {"enable": False, "token": "", "chat_id": ""}
        st.session_state.tg_enable = False
        st.session_state.tg_token = ""
        st.session_state.tg_chat = ""
        st.sidebar.success("Telegram settings cleared.")

st.sidebar.button("🔄 Refresh now", on_click=lambda: st.cache_data.clear())

# ========================
# SESSION STATE
# ========================
if "prev_snapshot" not in st.session_state:
    st.session_state.prev_snapshot = None
if "history_wide" not in st.session_state:
    st.session_state.history_wide = pd.DataFrame()
if "last_alert_ts" not in st.session_state:
    st.session_state.last_alert_ts = None
if "last_spike_ids" not in st.session_state:
    st.session_state.last_spike_ids = set()

# ========================
# FETCH
# ========================
st.title(f"📈 {SYMBOL} Options Live — Online Dashboard")
_now_ist = dt.datetime.now(IST).time()
if not (dt.time(9, 14) <= _now_ist <= dt.time(15, 31)):
    st.info("Market appears closed. Data may be stale or unchanged.")

status = st.empty()
try:
    status.info("Fetching option-chain from NSE…")
    curr = fetch_option_chain(SYMBOL)

    # Expiry selection with Weekly/Monthly
    chosen_expiry = None
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
        _default = pick_default_expiry(_exps) or (_exps[0] if _exps else None)
        _default_idx = _exps.index(_default) if (_default and _default in _exps) else 0
        chosen_expiry = st.sidebar.selectbox("Expiry", _exps, index=_default_idx)
        curr = curr[curr["expiry"].astype(str) == str(chosen_expiry)].copy()
    else:
        st.sidebar.caption("Expiry filter unavailable (no 'expiry').")

    if curr.empty:
        status.error("Failed to fetch data (empty). Try again.")
        st.stop()
    status.success("Fetched live option chain.")
except Exception as e:
    status.error(f"Fetch failed: {e}")
    st.stop()

# ========================
# ENRICH & HISTORY
# ========================
prev = st.session_state.prev_snapshot
df_en = enrich_with_prev(curr, prev)

# Save robust prev_snapshot that ALWAYS has volume_total
_vol_col = "volume_total" if "volume_total" in curr.columns else ("volume" if "volume" in curr.columns else None)
cols = ["symbol", "strike", "option_type", "ltp", "oi"] + ([_vol_col] if _vol_col else [])
prev_snap = curr[cols].copy()
if _vol_col:
    prev_snap.rename(columns={_vol_col: "volume_total"}, inplace=True)
else:
    prev_snap["volume_total"] = 0.0
st.session_state.prev_snapshot = prev_snap

wide_latest = make_wide(df_en)
if not wide_latest.empty:
    hist = st.session_state.history_wide
    st.session_state.history_wide = pd.concat([hist, wide_latest], ignore_index=True)
    if len(st.session_state.history_wide) > MAX_HISTORY_POINTS:
        st.session_state.history_wide = st.session_state.history_wide.tail(MAX_HISTORY_POINTS).reset_index(drop=True)

# ================= TRENDING OI TAB (self-contained) =================
hist = st.session_state.history_wide.copy()
main_tab, trend_tab = st.tabs(["📍 Main", "📊 Trending OI"])
with trend_tab:
    st.markdown("### 📊 Trending OI — Select strikes and track every refresh (~3 min)")
    if hist.empty:
        st.info("Trending OI will appear after a couple of snapshots are captured.")
    else:
        latest_ts_tab = hist["ts"].max()
        latest_df_tab = hist[hist["ts"] == latest_ts_tab].copy()

        def _infer_atm_for_defaults(df: pd.DataFrame) -> Optional[int]:
            try:
                if "spot" in df.columns and df["spot"].notna().any():
                    spot_m = float(df["spot"].median())
                    diffs = (df["strike"] - spot_m).abs()
                    return int(df.loc[diffs.idxmin(), "strike"])
            except Exception:
                pass
            return None

        all_strikes = sorted(latest_df_tab["strike"].dropna().unique().tolist())
        default_sel: List[int] = []
        _atm0 = _infer_atm_for_defaults(latest_df_tab)
        if _atm0 is not None:
            default_sel = [_atm0 + i * STRIKE_STEP for i in range(-3, 3 + 1) if (_atm0 + i * STRIKE_STEP) in all_strikes]
        else:
            default_sel = all_strikes[:7]

        user_strikes = st.multiselect("Choose strikes to track", options=all_strikes, default=default_sel, key="trending_strikes")
        if not user_strikes:
            st.info("Select one or more strikes to begin.")
        else:
            def _within_mkt(ts: pd.Timestamp) -> bool:
                if not isinstance(ts, (pd.Timestamp, dt.datetime)):
                    return False
                p = pd.Timestamp(ts)
                t = (p.tz_localize(None).time() if p.tzinfo else p.time())
                return dt.time(9, 15) <= t <= dt.time(15, 30)

            day_hist = hist[hist["ts"].apply(_within_mkt)]
            dh = day_hist[day_hist["strike"].isin(user_strikes)].copy()
            if dh.empty:
                st.warning("No intraday history yet for the selected strikes.")
            else:
                agg = dh.groupby("ts", as_index=False).agg(
                    ce_oi=("ce_oi", "sum"),
                    pe_oi=("pe_oi", "sum"),
                    ce_ltp=("ce_ltp", "mean"),
                    pe_ltp=("pe_ltp", "mean"),
                ).sort_values("ts")

                agg["ce_oi_delta"] = agg["ce_oi"].diff()
                agg["pe_oi_delta"] = agg["pe_oi"].diff()
                agg["ce_px_delta"] = agg["ce_ltp"].diff()
                agg["pe_px_delta"] = agg["pe_ltp"].diff()

                disp = agg.rename(columns={
                    "ts": "Time",
                    "ce_oi": "Total CE OI",
                    "ce_oi_delta": "Δ CE OI",
                    "ce_px_delta": "Δ CE Price",
                    "pe_oi": "Total PE OI",
                    "pe_oi_delta": "Δ PE OI",
                    "pe_px_delta": "Δ PE Price",
                })
                for c in ["Total CE OI", "Δ CE OI", "Total PE OI", "Δ PE OI"]:
                    disp[c] = disp[c].round(0).astype("Int64")
                for c in ["Δ CE Price", "Δ PE Price"]:
                    disp[c] = disp[c].round(2)

                def _style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
                    def color_row(row):
                        styles = []
                        for col in df.columns:
                            color = ""
                            if col in ("Δ CE OI", "Δ CE Price"):
                                v = row[col]
                                if pd.notna(v):
                                    color = ("background-color:#e8f5e9;color:#1b5e20;" if v > 0
                                             else "background-color:#ffebee;color:#b71c1c;" if v < 0 else "")
                            if col in ("Δ PE OI", "Δ PE Price"):
                                v = row[col]
                                if pd.notna(v):
                                    color = ("background-color:#ffebee;color:#b71c1c;" if v > 0
                                             else "background-color:#e8f5e9;color:#1b5e20;" if v < 0 else "")
                            styles.append(color)
                        return styles
                    return df.style.apply(color_row, axis=1)

                st.caption(f"Selected strikes: {', '.join(map(str, user_strikes))}")
                try:
                    st.dataframe(_style(disp), use_container_width=True)
                except Exception:
                    st.dataframe(disp, use_container_width=True)

                st.markdown(
                    "<small>Totals are across selected strikes; Δ columns are per-refresh changes. Table keeps all prior intervals so you can compare 9:15, 10:00, 12:00, 13:00, etc.</small>",
                    unsafe_allow_html=True,
                )
# =============== END TRENDING OI TAB ===============

# ========================
# HEADER METRICS & SENTIMENT
# ========================
spot = float(df_en["spot"].dropna().iloc[0]) if "spot" in df_en.columns and not df_en["spot"].dropna().empty else 0.0
snapshot_time = pd.to_datetime(df_en["ts"].iloc[0]) if "ts" in df_en.columns else dt.datetime.now(IST)

ce = df_en[df_en.option_type == "CE"].copy()
pe = df_en[df_en.option_type == "PE"].copy()
ce_strength = float((ce["vol_flow"].fillna(0) * ce["ltp"].abs().fillna(0)).sum())
pe_strength = float((pe["vol_flow"].fillna(0) * pe["ltp"].abs().fillna(0)).sum())
total_strength = (ce_strength + pe_strength) or 1.0
buyer_pct = 100.0 * ce_strength / total_strength
seller_pct = 100.0 * pe_strength / total_strength

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
    ce_oi_total = float(ce["oi"].sum())
    pe_oi_total = float(pe["oi"].sum())
    pcr_val = pe_oi_total / ce_oi_total if ce_oi_total > 0 else np.nan
except Exception:
    pcr_val = np.nan

c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1.2])
with c1:
    st.markdown("**Spot (approx)**  \n:large_blue_circle: **{:.2f}**".format(spot))
    st.markdown("**Snapshot**  \n{}".format(snapshot_time.strftime('%Y-%m-%d %H:%M:%S')))
with c2:
    st.metric("Buyer % (CE proxy)", f"{buyer_pct:.1f}%")
with c3:
    st.metric("Seller % (PE proxy)", f"{seller_pct:.1f}%")
with c4:
    st.metric("PCR", f"{pcr_val:.2f}" if np.isfinite(pcr_val) else "—")
    st.markdown(
        """
        <div style="margin-top:6px;padding:10px;border-radius:10px;background:%s;color:white;text-align:center">
        <strong>Market Sentiment</strong><br><span style="font-size:18px">%s</span>
        </div>
        """ % (sentiment_color, sentiment_label),
        unsafe_allow_html=True,
    )

st.markdown("---")

# ========================
# NEAR-ATM TABLE
# ========================
def select_near_atm(df: pd.DataFrame, spot_price: float, n: int) -> pd.DataFrame:
    if df.empty:
        return df
    atm_ = nearest_strike(spot_price)
    lo, hi = atm_ - n * STRIKE_STEP, atm_ + n * STRIKE_STEP
    return df[(df["strike"] >= lo) & (df["strike"] <= hi)].copy()

near_view = select_near_atm(df_en, spot, n=near_strikes)
st.subheader(f"Strikes around ATM (±{near_strikes * STRIKE_STEP} points)")
if near_view.empty:
    st.warning("No near-ATM data available.")
else:
    show_cols = ["strike", "option_type", "ltp", "iv", "oi", "oi_chg", "oi_chg_pct", "ltp_chg", "ltp_chg_pct", "buildup"]
    tbl = near_view.sort_values(["strike", "option_type"])[show_cols]
    try:
        st.dataframe(style_oi_change(tbl), use_container_width=True)
    except Exception:
        st.dataframe(tbl, use_container_width=True)

# ========================
# HISTORY SNAPSHOT
# ========================
hist2 = st.session_state.history_wide.copy()
if not hist2.empty:
    latest_ts = hist2["ts"].max()
    latest_df = hist2[hist2["ts"] == latest_ts].copy()

    def infer_atm_strike(wide_latest: pd.DataFrame) -> float:
        spot_m = float(wide_latest["spot"].median()) if "spot" in wide_latest.columns and wide_latest["spot"].notna().any() else np.nan
        if np.isfinite(spot_m):
            diffs = (wide_latest["strike"] - spot_m).abs()
            return float(wide_latest.loc[diffs.idxmin(), "strike"])
        ssum = (wide_latest["ce_ltp"].fillna(0) + wide_latest["pe_ltp"].fillna(0))
        return float(wide_latest.loc[ssum.idxmin(), "strike"]) if not wide_latest.empty else np.nan

    atm = infer_atm_strike(latest_df)
    spot_latest = float(latest_df["spot"].median()) if latest_df["spot"].notna().any() else np.nan
    c1, c2, c3 = st.columns(3)
    c1.metric("ATM", f"{atm:.0f}" if np.isfinite(atm) else "—")
    c2.metric("Spot (median)", f"{spot_latest:,.2f}" if np.isfinite(spot_latest) else "—")
    c3.metric("Snapshots stored", str(hist2["ts"].nunique()))

    # 3-MINUTE WINDOW LOGIC
    def pick_3min_ago_ts(h: pd.DataFrame, ref_ts: pd.Timestamp) -> Optional[pd.Timestamp]:
        hts = sorted(h["ts"].unique())
        if not hts:
            return None
        target = pd.Timestamp(ref_ts) - pd.Timedelta(minutes=3)
        candidates = [t for t in hts if (ref_ts - pd.Timestamp(t)) >= pd.Timedelta(minutes=1)]
        if not candidates:
            return None
        best = min(candidates, key=lambda t: abs((pd.Timestamp(t) - target).total_seconds()))
        if abs((pd.Timestamp(best) - target).total_seconds()) > 5 * 60:
            return None
        return pd.Timestamp(best)

    ts_3m = pick_3min_ago_ts(hist2, latest_ts)
    three_min_df = pd.DataFrame()
    if ts_3m is not None:
        last = hist2[hist2["ts"] == latest_ts].copy()
        old = hist2[hist2["ts"] == ts_3m].copy()
        mcols = ["strike", "ce_oi", "pe_oi", "ce_ltp", "pe_ltp", "ce_vtot", "pe_vtot"]
        last = last[["strike"] + mcols[1:]]
        old = old[["strike"] + mcols[1:]]
        three_min_df = last.merge(old, on="strike", suffixes=("", "_old"))
        for side in ("ce", "pe"):
            three_min_df[f"{side}_oi_3m"] = three_min_df[f"{side}_oi"] - three_min_df[f"{side}_oi_old"]
            three_min_df[f"{side}_ltp_3m"] = three_min_df[f"{side}_ltp"] - three_min_df[f"{side}_ltp_old"]
            three_min_df[f"{side}_oi_3m_pct"] = np.where(
                three_min_df[f"{side}_oi_old"] > 0,
                100 * three_min_df[f"{side}_oi_3m"] / three_min_df[f"{side}_oi_old"],
                np.where(three_min_df[f"{side}_oi_3m"] > 0, 100, 0)
            )
        three_min_df["spike_flag"] = (three_min_df["ce_oi_3m_pct"].abs() >= oi_alert_pct) | (three_min_df["pe_oi_3m_pct"].abs() >= oi_alert_pct)

    # TOP STRIKES
    if np.isfinite(atm):
        allowed = [atm + i * STRIKE_STEP for i in range(-ATM_TOP_STRIKE_SPAN, ATM_TOP_STRIKE_SPAN + 1)]
        latest_df_win = latest_df[latest_df["strike"].isin(allowed)].copy()
    else:
        latest_df_win = latest_df.copy()

    st.subheader("🏆 Top Strikes — OI & Price (ATM window)")
    top_n = st.slider("Top N", 5, 15, 10, key="topn")
    t1, t2 = st.columns(2)
    with t1:
        st.write(f"Top CE OI (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "ce_oi")[["strike", "ce_oi", "ce_oi_chg"]])
        st.write(f"Top CE Price (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "ce_ltp")[["strike", "ce_ltp", "ce_ltp_chg"]])
    with t2:
        st.write(f"Top PE OI (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "pe_oi")[["strike", "pe_oi", "pe_oi_chg"]])
        st.write(f"Top PE Price (ATM ± {ATM_TOP_STRIKE_SPAN})")
        st.dataframe(latest_df_win.nlargest(top_n, "pe_ltp")[["strike", "pe_ltp", "pe_ltp_chg"]])

    # BUILDUP TABLE ATM ±10
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

    # OI DISTRIBUTION
    st.subheader("📊 OI Distribution (Green=PE writers, Red=CE writers)")
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=latest_df["ce_oi"], name="CE OI", marker_color="#e53935"))
    fig_dist.add_trace(go.Bar(x=latest_df["strike"], y=latest_df["pe_oi"], name="PE OI", marker_color="#43a047"))
    fig_dist.update_layout(barmode="group", xaxis_title="Strike", yaxis_title="Open Interest")
    st.plotly_chart(fig_dist, use_container_width=True, key=f"dist_{SYMBOL}_{latest_ts}")

    # CE/PE PANEL — ATM ±7: LTP vs VWAP + Theoretical
    st.subheader("🎛️ CE (left) & PE (right) Panel — LTP vs VWAP & Theoretical Premium (ATM ± selected)")
    if np.isfinite(atm):
        panel_strikes_list = [atm + i * STRIKE_STEP for i in range(-panel_strikes, panel_strikes + 1)]
    else:
        panel_strikes_list = sorted(latest_df["strike"].unique())[: (2 * panel_strikes + 1)]

    def build_vwap_series(side: str, strike_val: int) -> pd.DataFrame:
        dfp = hist2[(hist2["strike"] == strike_val)].sort_values("ts").copy()
        px = dfp[f"{side}_ltp"].fillna(method="ffill")
        vflow = dfp[f"{side}_vflow"].fillna(0).clip(lower=0)
        cum_notional = (px * vflow).cumsum()
        cum_vol = vflow.cumsum().replace(0, np.nan)
        vwap = (cum_notional / cum_vol).fillna(method="bfill").fillna(method="ffill")
        return pd.DataFrame({"ts": dfp["ts"], "ltp": px, "vwap": vwap})

    t_years = years_to_expiry(str(chosen_expiry) if chosen_expiry else "", pd.Timestamp(snapshot_time))

    ce_col, pe_col = st.columns(2)
    with ce_col:
        st.markdown("**CE Strikes**")
        for s in panel_strikes_list:
            series = build_vwap_series("ce", s)
            if series.dropna().empty:
                continue
            last_row = latest_df.loc[latest_df["strike"] == s]
            ce_iv = float(last_row["ce_iv"].iloc[0]) if not last_row.empty and pd.notna(last_row["ce_iv"].iloc[0]) else 0.0
            theo = bs_theoretical(spot, s, t_years, risk_free, div_yield, ce_iv, call=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["ts"], y=series["ltp"], name=f"{s} CE LTP"))
            fig.add_trace(go.Scatter(x=series["ts"], y=series["vwap"], name="VWAP"))
            fig.update_layout(title=f"{s} CE — Theoretical: {theo:.2f}", xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True, key=f"ce_panel_{SYMBOL}_{s}_{latest_ts}")

    with pe_col:
        st.markdown("**PE Strikes**")
        for s in panel_strikes_list:
            series = build_vwap_series("pe", s)
            if series.dropna().empty:
                continue
            last_row = latest_df.loc[latest_df["strike"] == s]
            pe_iv = float(last_row["pe_iv"].iloc[0]) if not last_row.empty and pd.notna(last_row["pe_iv"].iloc[0]) else 0.0
            theo = bs_theoretical(spot, s, t_years, risk_free, div_yield, pe_iv, call=False)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=series["ts"], y=series["ltp"], name=f"{s} PE LTP"))
            fig.add_trace(go.Scatter(x=series["ts"], y=series["vwap"], name="VWAP"))
            fig.update_layout(title=f"{s} PE — Theoretical: {theo:.2f}", xaxis_title="Time", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True, key=f"pe_panel_{SYMBOL}_{s}_{latest_ts}")

    # CE–PE Premium Crossover around ATM
    st.subheader("📈 CE–PE Premium Crossover (ATM ± near strikes)")
    if np.isfinite(atm):
        strikes_window = [atm + i * STRIKE_STEP for i in range(-near_strikes, near_strikes + 1)]
        cross_df = latest_df[latest_df["strike"].isin(strikes_window)].copy()
    else:
        cross_df = pd.DataFrame()
    if cross_df.empty:
        st.warning("No strikes found in ATM window.")
    else:
        cross_df["premium_diff"] = cross_df["ce_ltp"].fillna(0) - cross_df["pe_ltp"].fillna(0)
        fig_cross = go.Figure()
        fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["ce_ltp"], mode="lines+markers", name="CE Premium"))
        fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["pe_ltp"], mode="lines+markers", name="PE Premium"))
        fig_cross.add_trace(go.Scatter(x=cross_df["strike"], y=cross_df["premium_diff"], mode="lines+markers", name="CE-PE Diff"))
        fig_cross.update_layout(title=f"CE vs PE Premium Crossover (ATM {atm:.0f} ± {near_strikes})", xaxis_title="Strike", yaxis_title="Premium")
        st.plotly_chart(fig_cross, use_container_width=True, key=f"cross_{SYMBOL}_{latest_ts}")

    # 3-MINUTE OI CHANGE SECTION + ALERTS
    st.markdown("---")
    st.subheader("⏱️ 3-Minute OI & Price Movement (auto)")

    if ts_3m is not None and not three_min_df.empty:
        if np.isfinite(atm):
            alert_strikes = set([atm + i * STRIKE_STEP for i in range(-near_strikes, near_strikes + 1)])
        else:
            alert_strikes = set(three_min_df["strike"].unique()[: (2 * near_strikes + 1)])

        disp = three_min_df.copy()
        disp["CE Buildup (3m)"] = [classify_buildup(o, p) for o, p in zip(disp["ce_oi_3m"], disp["ce_ltp_3m"])]
        disp["PE Buildup (3m)"] = [classify_buildup(o, p) for o, p in zip(disp["pe_oi_3m"], disp["pe_ltp_3m"])]
        show_cols = [
            "strike",
            "ce_oi_3m", "ce_oi_3m_pct", "ce_ltp_3m", "CE Buildup (3m)",
            "pe_oi_3m", "pe_oi_3m_pct", "pe_ltp_3m", "PE Buildup (3m)",
        ]
        st.caption(f"Comparing {latest_ts} vs {ts_3m} (~3 minutes apart)")
        st.dataframe(disp[show_cols].sort_values("strike"), use_container_width=True)

        spikes = disp[(disp["spike_flag"])].copy()
        st.subheader("⚡ Sudden OI Spike (≥ threshold in 3-min window)")
        if spikes.empty:
            st.info("No spike detected in last ~3 minutes.")
        else:
            st.dataframe(spikes[["strike", "ce_oi_3m_pct", "pe_oi_3m_pct", "ce_oi_3m", "pe_oi_3m"]].sort_values("strike"),
                         use_container_width=True)

        # Telegram alerts (use persisted settings)
        def maybe_send_alerts():
            key = (str(ts_3m), str(latest_ts))
            if st.session_state.last_alert_ts == key:
                return
            st.session_state.last_alert_ts = key

            settings = st.session_state.tg_settings
            if not settings.get("enable") or not settings.get("token") or not settings.get("chat_id"):
                return

            # ATM ± near_strikes alert
            alert_rows = disp[disp["strike"].isin(alert_strikes)].sort_values("strike")
            if not alert_rows.empty:
                lines = ["<b>ATM±{} OI Update ({} → {})</b>".format(near_strikes, ts_3m.strftime("%H:%M"), latest_ts.strftime("%H:%M")),
                         f"Symbol: {SYMBOL}, ATM: {int(atm) if np.isfinite(atm) else '-'}"]
                for _, r in alert_rows.iterrows():
                    lines.append(
                        f"{int(r['strike'])}: CE ΔOI {int(r['ce_oi_3m']):+}, {r['ce_oi_3m_pct']:+.1f}% | "
                        f"PE ΔOI {int(r['pe_oi_3m']):+}, {r['pe_oi_3m_pct']:+.1f}% "
                        f"| CE {r['CE Buildup (3m)']} / PE {r['PE Buildup (3m)']}"
                    )
                send_telegram(settings["token"], settings["chat_id"], "\n".join(lines))

            # Spike alerts (dedup)
            if not spikes.empty:
                new_spikes = []
                for _, r in spikes.iterrows():
                    sid = f"{int(r['strike'])}_{ts_3m.strftime('%H%M')}_{latest_ts.strftime('%H%M')}"
                    if sid not in st.session_state.last_spike_ids:
                        new_spikes.append(r)
                        st.session_state.last_spike_ids.add(sid)
                if new_spikes:
                    lines = ["<b>⚡ Sudden OI Spike Alert</b>",
                             f"{SYMBOL} ({ts_3m.strftime('%H:%M')} → {latest_ts.strftime('%H:%M')}) | Threshold: {oi_alert_pct}%"]
                    for r in new_spikes:
                        lines.append(
                            f"{int(r['strike'])}: CE {r['ce_oi_3m_pct']:+.1f}% | PE {r['pe_oi_3m_pct']:+.1f}% "
                            f"(Δ CE {int(r['ce_oi_3m']):+}, Δ PE {int(r['pe_oi_3m']):+})"
                        )
                    send_telegram(settings["token"], settings["chat_id"], "\n".join(lines))

        maybe_send_alerts()
    else:
        st.info("Need a few snapshots to compute 3-minute changes.")

# ========================
# RAW SNAPSHOT
# ========================
with st.expander("🔎 Raw Latest Snapshot"):
    st.dataframe(df_en.sort_values(["strike", "option_type"]), use_container_width=True)

st.markdown("---")
st.caption("This dashboard fetches NSE option-chain live and refreshes automatically. Intraday trends are from in-session history only. Use with caution.")
