# NIFTY Options Dashboard (NSE-only, auto-update every 3 minutes)

This project fetches the NSE **NIFTY** option chain directly from nseindia.com (no Zerodha needed), 
saves time-stamped CSV snapshots every 3 minutes, and shows a Streamlit dashboard with strike-wise analytics.

## Project Structure
```text
nifty_nse_dashboard/
├── app.py               # Streamlit dashboard (reads latest CSV from ./data)
├── config.py            # Settings (symbol, refresh secs, paths, etc.)
├── nse_fetch.py         # Robust NSE option chain downloader
├── fetch_loop.py        # Auto-run fetcher every 3 minutes
├── utils.py             # Analytics helpers (buildup, strength, crossover)
├── requirements.txt
├── README.md
├── run_fetch.bat        # Windows helper to run fetch loop
├── run_fetch.sh         # Linux/macOS helper to run fetch loop
└── data/                # CSV snapshots (created automatically)
```

## 1) Install dependencies
```bash
pip install -r requirements.txt
```

## 2) Start the auto-downloader (every 3 minutes)
Choose ONE of the following:
- **Windows (CMD/PowerShell)**:
  ```bat
  run_fetch.bat
  ```
  (Or: `python fetch_loop.py`)

- **Linux/macOS**:
  ```bash
  bash run_fetch.sh
  ```
  (Or: `python fetch_loop.py`)

## 3) Launch the dashboard
In a separate terminal:
```bash
streamlit run app.py
```
The app reads the **latest** CSV in `./data`. It auto-refreshes every 3 minutes to pick up new files.

## Windows Task Scheduler (optional)
You can also schedule `python fetch_loop.py` to run on login or market hours.

## Notes / Tips
- NSE blocks aggressive scraping. This fetcher:
  - Uses a **Session** with browser-like headers
  - Warming GET to load cookies
  - Retries on temporary failures
- If you see 403/HTTP errors, wait a few minutes, change IP, or slow down requests.
- CSV columns include: symbol, strike, option_type, ltp, oi, volume, iv, vwap(NA), ts, spot.
- Dashboard computes 1-interval % changes by comparing latest vs previous snapshot **automatically**.
