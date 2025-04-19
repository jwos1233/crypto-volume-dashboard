"""
Configuration file for shared settings across all scripts.
"""

# CoinGecko API settings
API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": API_KEY}

# Token filtering
MIN_MARKET_CAP = 200_000_000  # $200M
IGNORE_SYMBOLS = {
    "XSOLVBTC", "USDT", "FDUSD", "USDC", "WBTC", "WETH",
    "USDD", "LBTC", "TBTC", "USDT0", "SOLVBTC", "CLBTC"
}

# Strategy parameters
LOOKBACK_DAYS = 730
TRAILING_WINDOW = 30
Z_THRESHOLD = 3.0
SLEEP_BETWEEN_CALLS = 0.25

# Trading parameters
INITIAL_CAPITAL = 50000
TRADE_SIZE = 10000 