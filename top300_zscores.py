import asyncio
import aiohttp
import pandas as pd
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
from statistics import mean, stdev
from scipy.stats import percentileofscore

nest_asyncio.apply()

API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": API_KEY}
MIN_MARKET_CAP = 300_000_000  # $300M
IGNORE_SYMBOLS = {"XSOLVBTC","USDT", "FDUSD", "USDC", "WBTC", "WETH", "USDD", "LBTC", "TBTC", "USDT0", "SOLVBTC", "CLBTC"}

# Get top 300 tokens by market cap
async def get_top_300_tokens(session):
    tokens = []
    for page in range(1, 4):
        try:
            url = "https://pro-api.coingecko.com/api/v3/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 100,
                "page": page,
                "sparkline": "false"  # String instead of boolean
            }
            print(f"Fetching page {page} of tokens...")
            async with session.get(url, headers=HEADERS, params=params) as resp:
                if resp.status != 200:
                    print(f"Error on page {page}: {resp.status}")
                    continue
                data = await resp.json()
                if not isinstance(data, list):
                    print(f"Invalid response format on page {page}")
                    continue
                
                for t in data:
                    try:
                        market_cap = float(t.get("market_cap", 0))
                        symbol = str(t.get("symbol", "")).upper()
                        token_id = str(t.get("id", ""))
                        
                        if (market_cap >= MIN_MARKET_CAP and 
                            symbol and symbol not in IGNORE_SYMBOLS and 
                            token_id):
                            tokens.append({
                                "id": token_id,
                                "symbol": symbol,
                                "market_cap": market_cap
                            })
                    except (ValueError, TypeError) as e:
                        print(f"Error processing token: {str(e)}")
                        continue
                
                await asyncio.sleep(0.25)  # Rate limiting
                
        except Exception as e:
            print(f"Error fetching page {page}: {str(e)}")
            continue
    
    print(f"\nProcessed tokens:")
    for t in tokens[:5]:
        print(f"{t['symbol']}: ${t['market_cap']/1e9:.2f}B")
    return tokens

# Fetch 365d daily volume data
async def get_volume_history(session, token_id):
    try:
        url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart"
        params = {"vs_currency": "usd", "days": 365}
        async with session.get(url, headers=HEADERS, params=params) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            volumes = data.get("total_volumes", [])
            if len(volumes) < 2:
                return None
            return [float(v[1]) for v in volumes]
    except Exception as e:
        print(f"Error fetching volume for {token_id}: {str(e)}")
        return None

# Compute volume stats + volume/market_cap ratio + volume acceleration
async def compute_volume_stats(session, tokens):
    results = []
    print(f"\nAnalyzing volume data for {len(tokens)} tokens...")

    tasks = []
    for token in tokens:
        tasks.append((token["id"], token["symbol"], token["market_cap"], get_volume_history(session, token["id"])))

    volume_data = await tqdm_asyncio.gather(*(t[3] for t in tasks))

    for (token_id, symbol, market_cap, _), volumes in zip(tasks, volume_data):
        if volumes is None or len(volumes) < 10:
            continue
        try:
            current_volume = float(volumes[-1])
            previous_volume = float(volumes[-2])
            hist_volumes = [float(v) for v in volumes[:-1]]
            if len(hist_volumes) < 10 or previous_volume == 0 or market_cap == 0:
                continue

            avg_volume_7d = mean(hist_volumes[-7:])
            mu = mean(hist_volumes)
            sigma = stdev(hist_volumes)
            z = (current_volume - mu) / sigma if sigma != 0 else 0
            pctl = percentileofscore(hist_volumes, current_volume)
            dod_change = (current_volume - previous_volume) / previous_volume * 100
            vol_mcap_ratio = current_volume / market_cap
            volume_accel = current_volume / avg_volume_7d if avg_volume_7d != 0 else 0

            results.append({
                "symbol": symbol,
                "zscore_volume": round(z, 2),
                "percentile_volume": round(pctl, 2),
                "dod_change_pct": round(dod_change, 2),
                "volume_acceleration": round(volume_accel, 2),
                "current_volume": f"{current_volume / 1e6:.1f}M",
                "avg_volume": f"{mu / 1e6:.1f}M",
                "market_cap": f"{market_cap / 1e9:.2f}B",
                "volume_to_mcap": round(vol_mcap_ratio, 4)
            })
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    df = pd.DataFrame(results)
    if len(df) == 0:
        print("No valid results found!")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    df_zsorted = df.sort_values(by="zscore_volume", ascending=False)
    df_liquidity = df.sort_values(by="volume_to_mcap", ascending=False)
    df_accel = df.sort_values(by="volume_acceleration", ascending=False)
    return df_zsorted, df_liquidity, df_accel

# Run the script
async def main():
    print("Starting volume analysis for top 300 tokens...")
    async with aiohttp.ClientSession() as session:
        print("\nFetching top 300 tokens ($300M+ market cap, filtered)...")
        tokens = await get_top_300_tokens(session)
        print(f"\n{len(tokens)} tokens matched after ignoring: {', '.join(IGNORE_SYMBOLS)}")

        zscore_df, liquidity_df, accel_df = await compute_volume_stats(session, tokens)
        
        if len(zscore_df) > 0:
            print("\n🔥 Top Volume Spike Candidates (Z-score)")
            print(zscore_df.head(10)[["symbol", "zscore_volume", "percentile_volume", "dod_change_pct", "current_volume", "avg_volume"]])

            print("\n💧 Highest Volume/Market Cap Ratios")
            print(liquidity_df.head(10)[["symbol", "volume_to_mcap", "current_volume", "market_cap"]])

            print("\n🚀 Top Volume Accelerators (Current vs 7d Avg)")
            print(accel_df.head(10)[["symbol", "volume_acceleration", "current_volume", "avg_volume"]])
        else:
            print("\nNo valid results to display. Please check the error messages above.")

if __name__ == "__main__":
    asyncio.run(main())
