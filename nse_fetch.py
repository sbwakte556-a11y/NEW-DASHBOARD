# nse_fetch.py — Download NIFTY option chain from NSE with a robust session
import os
import time
import json
import math
import pytz
import numpy as np
import pandas as pd
import datetime as dt
import requests
from typing import Tuple
from config import SYMBOL, DATA_DIR, TIMEZONE

IST = pytz.timezone(TIMEZONE)

def _session() -> requests.Session:
    s = requests.Session()
    # Browser-like headers
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    # Warm-up to set cookies
    warm_urls = [
        "https://www.nseindia.com/",
        "https://www.nseindia.com/option-chain",
    ]
    for u in warm_urls:
        try:
            s.get(u, timeout=10)
        except Exception:
            pass
    return s

def _api_url(symbol: str) -> str:
    # Indices endpoint for NIFTY/BANKNIFTY
    return f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"

def fetch_option_chain(symbol: str = SYMBOL) -> pd.DataFrame:
    s = _session()
    url = _api_url(symbol)
    # Ajax-style headers for API
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.nseindia.com/option-chain",
    }
    for attempt in range(5):
        try:
            r = s.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                break
            time.sleep(2 + attempt)
        except Exception:
            time.sleep(2 + attempt)
    else:
        # All attempts failed
        return pd.DataFrame(columns=["symbol","strike","option_type","ltp","oi","volume","iv","vwap","ts","spot"])

    records = data.get("records", {})
    rows = []
    ts = dt.datetime.now(IST)
    # spot may appear under records or filtered
    spot = records.get("underlyingValue") or data.get("filtered", {}).get("underlyingValue") or 0

    for item in records.get("data", []):
        strike = item.get("strikePrice")
        if strike is None:
            continue
        for side in ("CE", "PE"):
            if side in item and isinstance(item[side], dict):
                leg = item[side]
                rows.append({
                    "symbol": symbol,
                    "strike": int(strike),
                    "option_type": side,
                    "ltp": float(leg.get("lastPrice") or 0.0),
                    "oi": float(leg.get("openInterest") or 0.0),
                    "volume": float(leg.get("totalTradedVolume") or 0.0),
                    "iv": float(leg.get("impliedVolatility") or 0.0),
                    "vwap": np.nan,  # not provided by NSE option chain endpoint
                    "ts": ts.isoformat(),
                    "spot": float(spot or 0.0),
                })
    df = pd.DataFrame(rows)
    return df

def save_snapshot(df: pd.DataFrame) -> str:
    if df.empty:
        raise ValueError("Empty dataframe — cannot save snapshot.")
    os.makedirs(DATA_DIR, exist_ok=True)
    # Use first row timestamp
    t = pd.to_datetime(df["ts"].iloc[0])
    fname = os.path.join(DATA_DIR, f"snap_{t.strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(fname, index=False)
    return fname

if __name__ == "__main__":
    d = fetch_option_chain(SYMBOL)
    if d.empty:
        print("Fetch failed or empty response.")
    else:
        f = save_snapshot(d)
        print("Saved:", f)
