# fetch_loop.py — run the NSE fetcher every REFRESH_SECS (3 minutes)
import time
from datetime import datetime
from config import REFRESH_SECS, SYMBOL
from nse_fetch import fetch_option_chain, save_snapshot

def sleep_to_next_interval(interval: int):
    now = time.time()
    delay = interval - (int(now) % interval)
    time.sleep(delay)

if __name__ == "__main__":
    print(f"Starting fetch loop for {SYMBOL} every {REFRESH_SECS}s ...")
    while True:
        try:
            df = fetch_option_chain(SYMBOL)
            if not df.empty:
                path = save_snapshot(df)
                print(datetime.now().strftime("%H:%M:%S"), "saved", path)
            else:
                print(datetime.now().strftime("%H:%M:%S"), "empty response (retry next interval)")
        except Exception as e:
            print(datetime.now().strftime("%H:%M:%S"), "error:", e)
        sleep_to_next_interval(REFRESH_SECS)
