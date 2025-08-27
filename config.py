# config.py
SYMBOL = "NIFTY"         # Only indices endpoint is used here; change to "BANKNIFTY" if needed
DATA_DIR = "data"        # Where CSV snapshots are stored
STRIKE_STEP = 50         # NIFTY strike step
NEAR_STRIKES = 3         # +/- how many strikes around ATM to analyze
REFRESH_SECS = 180       # 3 minutes
TIMEZONE = "Asia/Kolkata"
TG_TOKEN = ""            # Optional Telegram; keep empty to disable
TG_CHAT_ID = ""          # Optional Telegram chat id
