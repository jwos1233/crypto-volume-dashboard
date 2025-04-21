import asyncio
import aiohttp
import pandas as pd
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
from statistics import mean, stdev
from scipy.stats import percentileofscore
import time
from datetime import datetime, timedelta

nest_asyncio.apply()

API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": API_KEY}
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0

# Define all sectors and their constituents
SECTORS = {
    "Layer 1": {
        "solana": "SOL",
        "binancecoin": "BNB",
        "ethereum": "ETH",
        "tron": "TRX",
        "cardano": "ADA",
        "avalanche-2": "AVAX",
        "the-open-network": "TON",
        "hedera-hashgraph": "HBAR",
        "sui": "SUI"
    },
    "Layer 2": {
        "mantle": "MNT",
        "matic-network": "MATIC",
        "arbitrum": "ARB",
        "optimism": "OP",
        "blockstack": "STX",
        "starknet": "STRK",
        "zksync": "ZK",
        "celo": "CELO",
        "metis-token": "METIS"
    },
    "Meme": {
        "shiba-inu": "SHIB",
        "dogecoin": "DOGE",
        "pepe": "PEPE",
        "trump-coin": "TRUMP",
        "bonk": "BONK",
        "fartcoin": "FARTCOIN",
        "floki": "FLOKI",
        "dogwifhat": "WIF",
        "spx6900": "SPX"
    },
    "DeFi": {
        "chainlink": "LINK",
        "uniswap": "UNI",
        "ondo-finance": "ONDO",
        "aave": "AAVE",
        "maker": "MKR",
        "jupiter": "JUP",
        "ethena": "ENA",
        "curve-dao-token": "CRV",
        "lido-dao": "LDO",
        "raydium": "RAY",
        "pendle": "PENDLE"
    },
    "AI": {
        "bittensor": "TAO",
        "render-token": "RENDER",
        "fetch-ai": "FET",
        "grass": "GRASS",
        "virtual-protocol": "VIRTUAL",
        "akash-network": "AKT",
        "io": "IO",
        "the-graph": "GRT"
    },
    "DePIN": {
        "bittensor": "TAO",
        "render-token": "RENDER",
        "filecoin": "FIL",
        "bittorrent": "BTT",
        "livepeer": "LPT",
        "iotex": "IOTX",
        "aethir": "ATH"
    },
    "Gaming": {
        "immutable-x": "IMX",
        "gala": "GALA",
        "axie-infinity": "AXS",
        "beam": "BEAM",
        "ronin": "RON",
        "stepn": "GMT",
        "yield-guild-games": "YGG",
        "magic": "MAGIC"
    },
    "USA": {
        "solana": "SOL",
        "ripple": "XRP",
        "dogecoin": "DOGE",
        "cardano": "ADA",
        "chainlink": "LINK",
        "avalanche-2": "AVAX",
        "stellar": "XLM",
        "hedera-hashgraph": "HBAR",
        "sui": "SUI",
        "litecoin": "LTC",
        "uniswap": "UNI"
    },
    "High FDV": {
        "pi-network": "PI",
        "worldcoin": "WLD",
        "trump-coin": "TRUMP",
        "movement": "MOVE",
        "internet-computer": "ICP",
        "grass": "GRASS",
        "jupiter": "JUP",
        "ondo-finance": "ONDO",
        "sui": "SUI",
        "filecoin": "FIL"
    }
}

async def make_request(session, url, params=None):
    for retry in range(MAX_RETRIES):
        try:
            if retry > 0:
                await asyncio.sleep(RATE_LIMIT_DELAY * (retry + 1))
            async with session.get(url, headers=HEADERS, params=params) as resp:
                if resp.status == 429:  # Rate limited
                    if retry < MAX_RETRIES - 1:
                        continue
                    return None
                elif resp.status != 200:
                    print(f"Error: {resp.status} for {url}")
                    return None
                return await resp.json()
        except Exception as e:
            print(f"Error making request to {url}: {str(e)}")
            if retry < MAX_RETRIES - 1:
                continue
            return None
    return None

async def get_token_volume_history(session, token_id):
    try:
        url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart"
        params = {"vs_currency": "usd", "days": "365"}
        data = await make_request(session, url, params)
        
        if not data or "total_volumes" not in data:
            return None
            
        volumes = data.get("total_volumes", [])
        if len(volumes) < 2:
            return None
            
        # Convert to daily volumes
        daily_volumes = {}
        for timestamp, volume in volumes:
            date = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
            daily_volumes[date] = daily_volumes.get(date, 0) + float(volume)
            
        return daily_volumes
    except Exception as e:
        print(f"Error fetching volume for {token_id}: {str(e)}")
        return None

async def process_tokens(session, tokens, chunk_size=5):  # Increased chunk size
    all_volumes = {}
    
    # Process tokens in chunks to avoid rate limiting
    token_ids = list(tokens.keys())
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i:i + chunk_size]
        tasks = [get_token_volume_history(session, token_id) for token_id in chunk]
        chunk_volumes = await asyncio.gather(*tasks)
        
        # Aggregate volumes by date
        for token_id, volumes in zip(chunk, chunk_volumes):
            if volumes is None:
                continue
            for date, volume in volumes.items():
                all_volumes[date] = all_volumes.get(date, 0) + volume
                
        await asyncio.sleep(RATE_LIMIT_DELAY)  # Rate limiting between chunks
        
    return all_volumes

def format_number(num, suffix='B'):
    if num >= 1e9:
        return f"{num/1e9:.2f}{suffix}"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num:.2f}"

async def get_token_stats(session, token_id, symbol):
    try:
        url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart"
        params = {"vs_currency": "usd", "days": "365"}
        data = await make_request(session, url, params)
        
        if not data or "total_volumes" not in data:
            return None
            
        volumes = data.get("total_volumes", [])
        if len(volumes) < 2:
            return None
            
        # Convert to daily volumes
        daily_volumes = {}
        for timestamp, volume in volumes:
            date = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
            daily_volumes[date] = daily_volumes.get(date, 0) + float(volume)
            
        # Calculate statistics
        sorted_dates = sorted(daily_volumes.keys())
        volumes_list = [daily_volumes[date] for date in sorted_dates]
        
        current_volume = volumes_list[-1]
        mu = mean(volumes_list)
        sigma = stdev(volumes_list)
        z = (current_volume - mu) / sigma if sigma != 0 else 0
        
        return {
            "symbol": symbol,
            "current_volume": current_volume,
            "avg_volume": mu,
            "zscore": z
        }
    except Exception as e:
        print(f"Error getting stats for {token_id}: {str(e)}")
        return None

async def get_sector_token_breakdown(session, sector_name, tokens):
    token_stats = []
    tasks = []
    
    for token_id, symbol in tokens.items():
        tasks.append(get_token_stats(session, token_id, symbol))
    
    results = await asyncio.gather(*tasks)
    for result in results:
        if result:
            token_stats.append(result)
    
    return sorted(token_stats, key=lambda x: x['zscore'], reverse=True)

def print_stats(stats, history):
    print("\nSECTOR VOLUME ANALYSIS (Ranked by Z-Score)")
    print("-" * 95)
    print(f"{'SECTOR':<20} {'24H VOLUME':<15} {'AVG VOLUME':<15} {'Z-SCORE':<10}")
    print("-" * 95)
    
    # Sort sectors by z-score in descending order
    sorted_sectors = sorted(stats.items(), key=lambda x: x[1]['zscore_volume'], reverse=True)
    
    for sector, sector_stats in sorted_sectors:
        current_vol = format_number(sector_stats['current_volume'])
        avg_vol = format_number(sector_stats['avg_volume'])
        zscore = f"{sector_stats['zscore_volume']:>6.2f}"
        print(f"{sector:<20} {current_vol:<15} {avg_vol:<15} {zscore:<10}")
    
    print("-" * 95)

async def compute_sector_stats(session, sector_name, tokens):
    print(f"\nFetching {sector_name} token volumes (365 days)...")
    daily_volumes = await process_tokens(session, tokens)  # Pass the tokens for this sector
    
    if not daily_volumes:
        print(f"No valid volume data found for {sector_name}!")
        return None, None
        
    # Convert to sorted list of daily volumes
    sorted_dates = sorted(daily_volumes.keys())
    volumes = [daily_volumes[date] for date in sorted_dates]
    
    # Calculate statistics
    current_volume = volumes[-1]
    previous_volume = volumes[-2]
    mu = mean(volumes)
    sigma = stdev(volumes)
    z = (current_volume - mu) / sigma if sigma != 0 else 0
    pctl = percentileofscore(volumes, current_volume)
    dod_change = (current_volume - previous_volume) / previous_volume * 100 if previous_volume != 0 else 0
    
    # Calculate 7-day and 30-day moving averages
    ma7 = mean(volumes[-7:]) if len(volumes) >= 7 else None
    ma30 = mean(volumes[-30:]) if len(volumes) >= 30 else None
    
    # Create stats dictionary
    stats = {
        "date": sorted_dates[-1],
        "current_volume": current_volume,
        "zscore_volume": z,
        "percentile_volume": pctl,
        "dod_change_pct": dod_change,
        "ma7": ma7,
        "ma30": ma30,
        "avg_volume": mu,
        "std_dev": sigma
    }
    
    # Create history DataFrame
    history = pd.DataFrame({
        'date': sorted_dates,
        'volume': volumes
    })
    
    return stats, history

async def main():
    print("\nStarting multi-sector volume analysis (365 days)...")
    
    async with aiohttp.ClientSession() as session:
        all_stats = {}
        all_history = {}
        
        # Process all sectors in parallel
        tasks = []
        for sector_name, tokens in SECTORS.items():
            tasks.append(compute_sector_stats(session, sector_name, tokens))
        
        results = await asyncio.gather(*tasks)
        
        # Collect results
        for (sector_name, tokens), (stats, history) in zip(SECTORS.items(), results):
            if stats is not None and history is not None and not history.empty:
                all_stats[sector_name] = stats
                all_history[sector_name] = history
        
        if all_stats and all_history:
            # Print sector stats
            print_stats(all_stats, all_history)
            
            # Get token breakdowns for high z-score sectors
            high_z_sectors = [s for s, stats in all_stats.items() if stats['zscore_volume'] > 2]
            if high_z_sectors:
                print("\nTOKEN BREAKDOWN FOR HIGH Z-SCORE SECTORS (Z > 2)")
                print("-" * 95)
                
                for sector in high_z_sectors:
                    print(f"\n{sector.upper()} SECTOR TOKENS:")
                    print("-" * 95)
                    print(f"{'TOKEN':<20} {'24H VOLUME':<15} {'AVG VOLUME':<15} {'Z-SCORE':<10}")
                    print("-" * 95)
                    
                    # Get token breakdown for this sector
                    token_stats = await get_sector_token_breakdown(session, sector, SECTORS[sector])
                    for token in token_stats:
                        vol = format_number(token['current_volume'])
                        avg = format_number(token['avg_volume'])
                        z = f"{token['zscore']:>6.2f}"
                        print(f"{token['symbol']:<20} {vol:<15} {avg:<15} {z:<10}")
                    
                    print("-" * 95)
        else:
            print("No valid data found for any sector!")

if __name__ == "__main__":
    asyncio.run(main())
