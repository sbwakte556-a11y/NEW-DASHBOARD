# fetch_loop.py — run the NSE fetcher every REFRESH_SECS (3 minutes) with expiry selection
import time
import argparse
import pandas as pd
from datetime import datetime
from config import REFRESH_SECS, SYMBOL
from nse_fetch import fetch_option_chain, save_snapshot

def sleep_to_next_interval(interval: int):
    now = time.time()
    delay = interval - (int(now) % interval)
    time.sleep(delay)

def choose_expiry(df: pd.DataFrame, mode: str | None) -> str | None:
    """
    mode:
      - None or "nearest": choose soonest expiry >= today (IST), else earliest
      - "latest": choose the maximum (farthest) expiry
      - exact string: match that expiry if present
    Returns the chosen expiry string or None if not available.
    """
    if df is None or df.empty or "expiry" not in df.columns:
        return None

    exps = pd.Series(df["expiry"].dropna().astype(str).unique())
    if exps.empty:
        return None

    if not mode or mode.lower() in ("nearest", "recent"):
        today = pd.Timestamp.now(tz="Asia/Kolkata").normalize()
        parsed = pd.to_datetime(exps, errors="coerce")
        future = parsed[parsed >= today]
        chosen = future.min() if not future.empty else parsed.min()
        if pd.isna(chosen):
            return str(exps.iloc[0])
        # map back to original string that equals chosen
        for s in exps:
            if pd.to_datetime(s, errors="coerce") == chosen:
                return s
        return str(chosen.date())

    if mode.lower() == "latest":
        parsed = pd.to_datetime(exps, errors="coerce")
        chosen = parsed.max()
        if pd.isna(chosen):
            return str(exps.iloc[-1])
        for s in exps:
            if pd.to_datetime(s, errors="coerce") == chosen:
                return s
        return str(chosen.date())

    # exact string mode
    mode_str = str(mode)
    if mode_str in set(exps.tolist()):
        return mode_str

    # try tolerant match (date-like)
    try:
        target = pd.to_datetime(mode_str)
        for s in exps:
            if pd.to_datetime(s, errors="coerce") == target:
                return s
    except Exception:
        pass

    return None

def main():
    parser = argparse.ArgumentParser(description="NSE option chain fetch loop with expiry selection")
    parser.add_argument(
        "--expiry",
        default="nearest",
        help="Expiry selection: 'nearest' (default), 'latest', or an exact expiry string (e.g., 2025-08-28).",
    )
    args = parser.parse_args()
    mode = args.expiry

    print(f"Starting fetch loop for {SYMBOL} every {REFRESH_SECS}s ...")
    print(f"Expiry mode: {mode}")

    while True:
        try:
            df_all = fetch_option_chain(SYMBOL)
            if df_all.empty:
                print(datetime.now().strftime("%H:%M:%S"), "empty response (retry next interval)")
            else:
                expiry = choose_expiry(df_all, mode)
                if expiry and "expiry" in df_all.columns:
                    df = df_all[df_all["expiry"].astype(str) == str(expiry)].copy()
                    if df.empty:
                        print(datetime.now().strftime("%H:%M:%S"), f"no rows for expiry '{expiry}' (available={sorted(df_all['expiry'].astype(str).unique().tolist())})")
                    else:
                        path = save_snapshot(df)
                        print(datetime.now().strftime("%H:%M:%S"), f"saved {path} (expiry={expiry}, rows={len(df)})")
                else:
                    # Fallback: save all if no expiry column or cannot choose
                    path = save_snapshot(df_all)
                    print(datetime.now().strftime("%H:%M:%S"), f"saved {path} (no/invalid expiry filter)")
        except Exception as e:
            print(datetime.now().strftime("%H:%M:%S"), "error:", e)
        sleep_to_next_interval(REFRESH_SECS)

if __name__ == "__main__":
    main()
