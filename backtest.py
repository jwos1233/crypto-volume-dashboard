import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
from math import sqrt
from tqdm.asyncio import tqdm_asyncio
import matplotlib.pyplot as plt
from config import *

API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": API_KEY}

IGNORE_SYMBOLS = {
    "XSOLVBTC", "USDT", "FDUSD", "USDC", "WBTC", "WETH",
    "USDD", "LBTC", "TBTC", "USDT0", "SOLVBTC", "CLBTC"
}

LOOKBACK_DAYS = 730
TRAILING_WINDOW = 30
Z_THRESHOLD = 3.0
SLEEP_BETWEEN_CALLS = 0.25

INITIAL_CAPITAL = 50000
TRADE_SIZE = 10000

token_rows = []
btc_signal_map = {}

# --- Fetch BTC market data and compute 50DMA ---
async def fetch_btc_signal():
    url = "https://pro-api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": LOOKBACK_DAYS}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=HEADERS, params=params) as resp:
            data = await resp.json()
            df = pd.DataFrame({
                "timestamp": [datetime.utcfromtimestamp(x[0] / 1000).date() for x in data["prices"]],
                "close": [x[1] for x in data["prices"]]
            })
            df["btc_50dma"] = df["close"].rolling(50).mean()
            df["btc_above_50dma"] = df["close"] > df["btc_50dma"]
            return df.set_index("timestamp")["btc_above_50dma"].to_dict()

# --- Fetch top 300 tokens ---
async def fetch_tokens():
    all_tokens = []
    async with aiohttp.ClientSession() as session:
        for page in range(1, 4):  # Fetch 3 pages of 100 tokens each
            url = "https://pro-api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": page,
                "sparkline": False
            }
            
            print(f"Fetching page {page} of tokens...")
            async with session.get(url, headers=HEADERS, params=params) as resp:
                if resp.status != 200:
                    print(f"Error fetching tokens on page {page}: {resp.status}")
                    continue
                    
                data = await resp.json()
                if not data or not isinstance(data, list):
                    print(f"Invalid response format on page {page}")
                    continue
                    
                all_tokens.extend(data)
                await asyncio.sleep(SLEEP_BETWEEN_CALLS)  # Rate limiting
    
    print(f"Total tokens received from CoinGecko: {len(all_tokens)}")
    if all_tokens and len(all_tokens) > 0:
        print(f"First token example: {all_tokens[0]}")
    
    filtered_tokens = []
    for d in all_tokens:
        if not isinstance(d, dict):
            continue
        market_cap = d.get("market_cap", 0)
        symbol = d.get("symbol", "").upper()
        token_id = d.get("id")
        
        if (market_cap >= MIN_MARKET_CAP and 
            symbol not in IGNORE_SYMBOLS and 
            token_id):
            filtered_tokens.append({
                "id": token_id,
                "symbol": symbol,
                "market_cap": market_cap
            })
    
    filtered_tokens.sort(key=lambda x: x["market_cap"], reverse=True)
    print(f"\nFound {len(filtered_tokens)} tokens with market cap > ${MIN_MARKET_CAP/1e6}M")
    print("\nTop 10 tokens by market cap:")
    for t in filtered_tokens[:10]:
        print(f"{t['symbol']}: ${t['market_cap']/1e9:.2f}B")
        
    return filtered_tokens

# --- Fetch market data ---
async def fetch_market_data(token_id):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {"vs_currency": "usd", "days": LOOKBACK_DAYS}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=HEADERS, params=params) as resp:
            if resp.status != 200:
                return None
            return await resp.json()

# --- Compute z-scores + forward vol ---
def compute_vol_stats(df):
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))
    df["volume_avg"] = df["volume"].rolling(TRAILING_WINDOW).mean()
    df["volume_std"] = df["volume"].rolling(TRAILING_WINDOW).std()
    df["zscore_volume"] = (df["volume"] - df["volume_avg"]) / df["volume_std"]
    for h in [7, 14, 30]:
        df[f"vol_{h}d"] = df["log_return"].rolling(h).std().shift(-h) * sqrt(365)
    return df

# --- Analyze token + store daily rows ---
async def analyze_token(token):
    await asyncio.sleep(SLEEP_BETWEEN_CALLS)
    data = await fetch_market_data(token["id"])
    if data is None or "prices" not in data or "total_volumes" not in data:
        return None

    df = pd.DataFrame({
        "timestamp": [datetime.utcfromtimestamp(x[0] / 1000).date() for x in data["prices"]],
        "price": [x[1] for x in data["prices"]],
        "volume": [x[1] for x in data["total_volumes"]],
    }).set_index("timestamp")

    df = compute_vol_stats(df)
    df = df[df["volume_std"].notnull()]
    df["symbol"] = token["symbol"]
    df = df[df["vol_7d"].notnull() & df["vol_14d"].notnull() & df["vol_30d"].notnull()]
    token_rows.append(df)

    high_z = df[df["zscore_volume"] > Z_THRESHOLD]
    rest = df[df["zscore_volume"] <= Z_THRESHOLD]

    if len(high_z) < 3:
        return None

    return {
        "symbol": token["symbol"],
        "count_high_z": len(high_z),
        "vol_7d_highz": high_z["vol_7d"].mean(),
        "vol_7d_rest": rest["vol_7d"].mean(),
        "vol_14d_highz": high_z["vol_14d"].mean(),
        "vol_14d_rest": rest["vol_14d"].mean(),
        "vol_30d_highz": high_z["vol_30d"].mean(),
        "vol_30d_rest": rest["vol_30d"].mean(),
    }

# --- Full analysis ---
async def run_analysis():
    global btc_signal_map
    btc_signal_map = await fetch_btc_signal()
    tokens = await fetch_tokens()
    results = await tqdm_asyncio.gather(*(analyze_token(token) for token in tokens))
    results = [r for r in results if r is not None]
    df_summary = pd.DataFrame(results)
    df_summary["uplift_7d"] = df_summary["vol_7d_highz"] / df_summary["vol_7d_rest"]
    df_summary["uplift_14d"] = df_summary["vol_14d_highz"] / df_summary["vol_14d_rest"]
    df_summary["uplift_30d"] = df_summary["vol_30d_highz"] / df_summary["vol_30d_rest"]
    df_all = pd.concat(token_rows).reset_index()
    return df_summary.sort_values(by="uplift_7d", ascending=False), df_all

# --- Strategy simulation ---
def run_trading_strategy(df_all, hold_days=7):
    df = df_all.copy()
    df.sort_values(by=["symbol", "timestamp"], inplace=True)

    trades = []
    open_positions = {}

    for symbol, group in df.groupby("symbol"):
        group = group.reset_index()
        for i in range(len(group) - hold_days):
            row = group.iloc[i]
            entry_date = row["timestamp"]
            exit_date = group.iloc[i + hold_days]["timestamp"]

            if symbol in open_positions and entry_date <= open_positions[symbol]:
                continue

            if row["zscore_volume"] > Z_THRESHOLD and btc_signal_map.get(entry_date, False):
                entry_price = row["price"]
                exit_price = group.iloc[i + hold_days]["price"]
                if entry_price <= 0 or exit_price <= 0 or np.isnan(entry_price) or np.isnan(exit_price):
                    continue

                pnl_pct = (exit_price - entry_price) / entry_price
                pnl_usd = pnl_pct * TRADE_SIZE

                trades.append({
                    "symbol": symbol,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd
                })

                open_positions[symbol] = exit_date

    df_trades = pd.DataFrame(trades)
    if df_trades.empty:
        return df_trades, {}, pd.DataFrame()

    df_ts = df_trades.groupby("exit_date")["pnl_usd"].sum().cumsum().reset_index()
    df_ts.columns = ["date", "cumulative_pnl"]
    df_ts["date"] = pd.to_datetime(df_ts["date"])
    df_ts["net_liq"] = df_ts["cumulative_pnl"] + INITIAL_CAPITAL
    df_ts = df_ts.set_index("date").resample("D").ffill().fillna(method="ffill")

    returns = df_trades["pnl_usd"] / TRADE_SIZE
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    df_ts["peak"] = df_ts["net_liq"].cummax()
    df_ts["drawdown"] = df_ts["net_liq"] - df_ts["peak"]
    df_ts["drawdown_pct"] = df_ts["drawdown"] / df_ts["peak"].replace(0, np.nan)
    max_drawdown = df_ts["drawdown"].min()
    max_drawdown_pct = df_ts["drawdown_pct"].min() * 100

    stats = {
        "Total Trades": len(df_trades),
        "Final Net Liquidity ($)": round(df_ts["net_liq"].iloc[-1], 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown ($)": round(max_drawdown, 2),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Win Rate": round((df_trades["pnl_usd"] > 0).mean() * 100, 2)
    }

    pnl_per_token = df_trades.groupby("symbol")["pnl_usd"].agg(["count", "sum", "mean"])
    pnl_per_token.columns = ["trades", "total_pnl_usd", "avg_pnl_usd"]
    pnl_per_token = pnl_per_token.sort_values(by="total_pnl_usd", ascending=False)

    return df_trades, stats, pnl_per_token

# --- Entrypoint ---
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    print("Running strategy with BTC 50DMA condition on tokens with market cap > $200M...")
    df_summary, df_all = asyncio.run(run_analysis())

    print("\n📊 Top Tokens by 7D Volatility Uplift:")
    print(df_summary[["symbol", "count_high_z", "uplift_7d", "uplift_14d", "uplift_30d"]].head(10))

    for hold in range(1, 8):
        print(f"\n--- Holding Period: {hold} days ---")
        df_trades, stats, pnl_per_token = run_trading_strategy(df_all, hold_days=hold)

        if stats:
            for k, v in stats.items():
                print(f"{k}: {v}")
        else:
            print("No trades executed.")
