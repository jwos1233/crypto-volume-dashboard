import asyncio
import aiohttp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from math import log, sqrt

API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": API_KEY}
TOKEN_ID = "decentraland"
VS_CURRENCY = "usd"
LOOKBACK_DAYS = 730
TRAILING_WINDOW = 365
Z_THRESHOLD = 2.0

# Fetch historical market chart data for 2 years
async def fetch_market_data():
    url = f"https://pro-api.coingecko.com/api/v3/coins/{TOKEN_ID}/market_chart"
    params = {"vs_currency": VS_CURRENCY, "days": LOOKBACK_DAYS}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=HEADERS, params=params) as resp:
            data = await resp.json()
            return data

# Compute daily returns and forward volatility
def realized_vol(log_returns, window):
    return log_returns.rolling(window).std() * sqrt(365)

# Run full Z-score analysis
async def run_backtest():
    print(f"Fetching {LOOKBACK_DAYS} days of price and volume data for {TOKEN_ID}...")
    data = await fetch_market_data()
    prices = data["prices"]
    volumes = data["total_volumes"]

    df = pd.DataFrame({
        "timestamp": [datetime.utcfromtimestamp(x[0] / 1000).date() for x in prices],
        "price": [x[1] for x in prices],
        "volume": [x[1] for x in volumes]
    })

    df.set_index("timestamp", inplace=True)
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["volume_avg"] = df["volume"].rolling(TRAILING_WINDOW).mean()
    df["volume_std"] = df["volume"].rolling(TRAILING_WINDOW).std()
    df["zscore_volume"] = (df["volume"] - df["volume_avg"]) / df["volume_std"]

    # Forward realized volatility (annualized)
    df["vol_7d"] = realized_vol(df["log_return"], 7).shift(-7)
    df["vol_14d"] = realized_vol(df["log_return"], 14).shift(-14)
    df["vol_30d"] = realized_vol(df["log_return"], 30).shift(-30)

    # Filter to only Z-score valid region
    df = df[df["volume_std"].notnull()]

    # Split into high Z-score and all days
    high_z = df[df["zscore_volume"] > Z_THRESHOLD]
    rest = df[df["zscore_volume"] <= Z_THRESHOLD]

    print(f"\n📈 High Z-score days (> {Z_THRESHOLD}): {len(high_z)}")
    print(f"📊 Normal days: {len(rest)}")

    for horizon in [7, 14, 30]:
        mean_high = high_z[f"vol_{horizon}d"].mean()
        mean_rest = rest[f"vol_{horizon}d"].mean()
        print(f"\n⏱ Forward {horizon}D Volatility:")
        print(f"   After High Z-score: {mean_high:.2%}")
        print(f"   Baseline:           {mean_rest:.2%}")

    # Optional: Plot
    plt.figure(figsize=(12, 5))
    plt.hist(high_z["vol_7d"], bins=30, alpha=0.6, label="High Z-score")
    plt.hist(rest["vol_7d"], bins=30, alpha=0.6, label="Normal")
    plt.title("7D Forward Realized Volatility Distribution")
    plt.xlabel("Volatility")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    asyncio.run(run_backtest())
