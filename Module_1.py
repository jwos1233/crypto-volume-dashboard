import requests
import pandas as pd
from datetime import datetime, timedelta
import sys

# === CONFIG ===
COINGECKO_API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": COINGECKO_API_KEY}
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

# === STABLECOIN FLOWS ===
def get_market_cap_series(coin_id):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": 31,
        "interval": "daily"
    }

    try:
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        data = res.json()
        df = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df = df.resample("D").mean()
        return df["market_cap"]
    except Exception as e:
        print(f"❌ Error fetching data for {coin_id}: {e}")
        sys.exit(1)

def combine_stablecoin_flows():
    usdt = get_market_cap_series("tether")
    usdc = get_market_cap_series("usd-coin")
    combined = usdt.add(usdc, fill_value=0)

    current_date = combined.index.max()
    t7 = current_date - timedelta(days=7)
    t30 = current_date - timedelta(days=30)

    def lookup(date):
        subset = combined[combined.index <= date]
        return subset.iloc[-1] if not subset.empty else None

    current = lookup(current_date)
    t7_val = lookup(t7)
    t30_val = lookup(t30)

    change_7d = (current - t7_val) / t7_val * 100 if t7_val else None
    change_30d = (current - t30_val) / t30_val * 100 if t30_val else None

    return {
        "Metric": "Stablecoin Flows",
        "Current": f"${current:,.0f}",
        "Change 7d": f"{change_7d:+.2f}%" if change_7d is not None else "N/A",
        "Change 30d": f"{change_30d:+.2f}%" if change_30d is not None else "N/A"
    }

# === FUNDING RATE HISTORY ===
def get_binance_funding(symbol, limit=1000):
    params = {"symbol": symbol, "limit": limit}
    try:
        res = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params=params)
        res.raise_for_status()
        data = res.json()

        df = pd.DataFrame(data)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
        df.set_index("fundingTime", inplace=True)
        df["fundingRate"] = df["fundingRate"].astype(float) * 100  # percent
        return df[["fundingRate"]]
    except Exception as e:
        print(f"⚠️ Error fetching funding for {symbol}: {e}")
        return pd.DataFrame()

def summarize_funding(symbol):
    df = get_binance_funding(symbol)
    if df.empty or len(df) < 90:
        return {
            "Metric": f"{symbol[:-4]} Funding Rate (bps)",
            "Current": "N/A",
            "Change 7d": "N/A",
            "Change 30d": "N/A"
        }

    # Convert % to basis points
    current = df["fundingRate"].iloc[-1] * 100
    t7_ago = df["fundingRate"].iloc[-21] * 100
    t30_ago = df["fundingRate"].iloc[-90] * 100

    change_7d = current - t7_ago
    change_30d = current - t30_ago

    return {
        "Metric": f"{symbol[:-4]} Funding Rate (bps)",
        "Current": f"{current:.2f}",
        "Change 7d": f"{change_7d:+.2f}" if t7_ago is not None else "N/A",
        "Change 30d": f"{change_30d:+.2f}" if t30_ago is not None else "N/A"
    }

# === FINAL OUTPUT ===
def get_all_metrics():
    return pd.DataFrame([
        combine_stablecoin_flows(),
        summarize_funding("BTCUSDT"),
        summarize_funding("ETHUSDT")
    ])

if __name__ == "__main__":
    df = get_all_metrics()
    print(df.to_string(index=False))
