import streamlit as st
import asyncio
import aiohttp
import pandas as pd
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
from statistics import mean, stdev
from scipy.stats import percentileofscore
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime
import os
from dotenv import load_dotenv
import contextlib
import requests
import numpy as np
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

nest_asyncio.apply()

# Try to get API keys from Streamlit secrets
try:
    API_KEY = st.secrets["COINGECKO_API_KEY"]
    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", None)
    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", None)
except:
    API_KEY = os.getenv('COINGECKO_API_KEY')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

if not API_KEY:
    st.error("No API key found. Please set the COINGECKO_API_KEY in your environment or Streamlit secrets.")
    st.stop()

HEADERS = {"x-cg-pro-api-key": API_KEY}
MIN_MARKET_CAP = 300_000_000  # $300M
IGNORE_SYMBOLS = {"SUSDS","USR","EETH","BTC.B","BUSD","MSOL","XSOLVBTC","USDT", "FDUSD", "USDC", "WBTC", "WETH", "USDD", "LBTC", "TBTC", "USDT0", "SOLVBTC", "CLBTC"}
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
TIMEOUT = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
ZSCORE_THRESHOLD = 2.0  # Z-score threshold for alerts
RATE_LIMIT_DELAY = 1.0  # Rate limit delay in seconds

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
        "the-graph": "GRT",
        "near": "NEAR",
        "internet-computer": "ICP"
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
        "magic": "MAGIC",
        "apecoin": "APE"
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
        "filecoin": "FIL",
        "zksync": "ZK",
        "scroll": "SCR",
        "layerzero": "LAYER",
        "berachain": "BERA",
        "kaito": "KAITO",
        "layerzero-protocol": "ZRO",
        "starknet": "STRK"
    },
    "Dino": {
        "ripple": "XRP",
        "cardano": "ADA",
        "stellar": "XLM",
        "litecoin": "LTC",
        "ethereum-classic": "ETC",
        "eos": "EOS",
        "iota": "IOTA",
        "tezos": "XTZ",
        "zcash": "ZEC"
    },
    "RWA": {
        "chainlink": "LINK",
        "avalanche-2": "AVAX",
        "hedera-hashgraph": "HBAR",
        "red": "RED",
        "ondo-finance": "ONDO",
        "maker": "MKR",
        "xdce-crowd-sale": "XDC"
    },
    "BNB Ecosystem": {
        "dexe": "DEXE",
        "formation-fi": "FORM",
        "pancakeswap-token": "CAKE",
        "floki": "FLOKI",
        "trust-wallet-token": "TWT",
        "beam": "BEAM",
        "axelar": "AXL",
        "zkj": "ZKJ",
        "cheems": "CHEEMS",
        "safepal": "SFP"
    },
    "AI Agents": {
        "virtual-protocol": "VIRTUAL",
        "origintrail": "TRAC",
        "ai16z": "AI16Z",
        "fairum": "FAI",
        "paal-ai": "PAAL",
        "pha": "PHA",
        "ai-xbt": "AIXBT",
        "chaingpt": "CGPT",
        "clanker": "CLANKER",
        "arc": "ARC",
        "griffin": "GRIFFAIN",
        "tai": "TAI",
        "oraichain-token": "ORAI",
        "zerebro": "ZEREBRO"
    },
    "Pump Fun": {
        "fartcoin": "FARTCOIN",
        "would-you-rather": "WOULD",
        "peanut": "PNUT",
        "alchemy": "ALCH",
        "achain": "ACT",
        "goat-coin": "GOAT",
        "banana": "BAN",
        "fartboy": "FARTBOY",
        "arc": "ARC",
        "fwog": "FWOG",
        "vine": "VINE",
        "moodeng": "MOODENG",
        "avalanche-2": "AVA",
        "chillguy": "CHILLGUY",
        "unfed": "UFD",
        "sigma": "SIGMA",
        "luce": "LUCE",
        "jellyjelly": "JELLYJELLY",
        "michi": "MICHI"
    }
}

def send_telegram_alert(symbol, zscore, current_volume, market_cap, volume_change):
    """Send a Telegram alert for significant volume spikes"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram credentials not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your environment variables.")
        return
    
    # Initialize last_alert_times in session state if it doesn't exist
    if 'last_alert_times' not in st.session_state:
        st.session_state.last_alert_times = {}
    
    # Try to load alert times from file
    try:
        with open('alert_times.json', 'r') as f:
            file_alert_times = json.load(f)
            # Convert string timestamps back to datetime objects
            for sym, time_str in file_alert_times.items():
                file_time = datetime.fromisoformat(time_str)
                # Only update if file has more recent time or symbol doesn't exist in session state
                if sym not in st.session_state.last_alert_times or file_time > st.session_state.last_alert_times[sym]:
                    st.session_state.last_alert_times[sym] = file_time
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    current_time = datetime.now()
    
    # Check if we've sent an alert for this token in the last 24 hours
    if symbol in st.session_state.last_alert_times:
        last_alert_time = st.session_state.last_alert_times[symbol]
        hours_since_last_alert = (current_time - last_alert_time).total_seconds() / 3600
        if hours_since_last_alert < 24:
            # Debug: Show why alert was skipped
            st.sidebar.write(f"Skipping alert for {symbol}: {hours_since_last_alert:.2f} hours since last alert")
            return  # Skip alert if less than 24 hours since last alert
    
    message = (
        f"🚨 Volume Spike Alert!\n\n"
        f"Token: {symbol}\n"
        f"Z-Score: {zscore:.2f}\n"
        f"Current Volume: {format_currency(current_volume)}\n"
        f"Market Cap: {format_currency(market_cap)}\n"
        f"24h Change: {volume_change}%\n\n"
        f"Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            # Update last alert time for this token
            st.session_state.last_alert_times[symbol] = current_time
            
            # Save alert times to file
            file_alert_times = {sym: time.isoformat() for sym, time in st.session_state.last_alert_times.items()}
            with open('alert_times.json', 'w') as f:
                json.dump(file_alert_times, f)
                
            # Debug: Show alert was sent
            st.sidebar.write(f"Alert sent for {symbol} at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        st.warning(f"Error sending Telegram alert: {str(e)}")

@st.cache_resource
def get_session():
    return aiohttp.ClientSession(timeout=TIMEOUT)

async def fetch_tokens(_session):
    tokens = []
    for page in range(1, 4):
        for retry in range(MAX_RETRIES):
            try:
                url = "https://pro-api.coingecko.com/api/v3/coins/markets"
                params = {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 100,
                    "page": page,
                    "sparkline": "false"
                }
                async with _session.get(url, headers=HEADERS, params=params) as resp:
                    if resp.status != 200:
                        if retry < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                        st.warning(f"Failed to fetch page {page} after {MAX_RETRIES} retries")
                        break
                        
                    data = await resp.json()
                    if not isinstance(data, list):
                        st.warning(f"Invalid response format for page {page}")
                        break
                    
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
                        except (ValueError, TypeError):
                            continue
                    
                    await asyncio.sleep(0.25)
                    break
                    
            except Exception as e:
                if retry < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                st.error(f"Error fetching page {page}: {str(e)}")
                break
    return tokens

async def fetch_volume_history(_session, token_id):
    for retry in range(MAX_RETRIES):
        try:
            url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart"
            params = {"vs_currency": "usd", "days": 365}
            async with _session.get(url, headers=HEADERS, params=params) as resp:
                if resp.status != 200:
                    if retry < MAX_RETRIES - 1:
                        await asyncio.sleep(RETRY_DELAY)
                        continue
                    return None, None
                data = await resp.json()
                volumes = data.get("total_volumes", [])
                prices = data.get("prices", [])  # Get price data as well
                if len(volumes) < 2 or len(prices) < 2:
                    return None, None
                return [float(v[1]) for v in volumes], [float(p[1]) for p in prices]
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            st.error(f"Error fetching data for {token_id}: {str(e)}")
            return None, None
    return None, None

def format_currency(value):
    """Format large numbers into millions (M) or billions (B)"""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.2f}"

def calculate_realized_volatility(prices, window):
    """Calculate realized volatility over a given window of days"""
    if len(prices) < window + 1:
        return None
    
    # Calculate daily returns
    returns = [np.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
    
    # Calculate rolling volatility (annualized)
    volatility = np.std(returns[-window:]) * np.sqrt(365) * 100  # Convert to percentage
    return volatility

@st.cache_data(ttl=300)
def process_volume_stats(_tokens, volume_data):
    results = []
    for (token_id, symbol, market_cap, _), (volumes, prices) in zip(_tokens, volume_data):
        if volumes is None or prices is None or len(volumes) < 10 or len(prices) < 30:
            continue
        try:
            current_volume = float(volumes[-1])
            previous_volume = float(volumes[-2])
            hist_volumes = [float(v) for v in volumes[:-1]]
            
            # Calculate volatilities
            vol_7d = calculate_realized_volatility(prices, 7)
            vol_30d = calculate_realized_volatility(prices, 30)
            
            if len(hist_volumes) < 10 or previous_volume == 0 or market_cap == 0:
                continue

            avg_volume_7d = mean(hist_volumes[-7:])
            mu = mean(hist_volumes)
            sigma = stdev(hist_volumes)
            z = (current_volume - mu) / sigma if sigma != 0 else 0
            pctl = percentileofscore(hist_volumes, current_volume)
            dod_change = (current_volume - previous_volume) / previous_volume * 100
            vol_mcap_ratio = current_volume / market_cap * 100  # Convert to percentage
            volume_accel = current_volume / avg_volume_7d if avg_volume_7d != 0 else 0

            # Send Telegram alert for significant volume spikes
            if z > ZSCORE_THRESHOLD:
                send_telegram_alert(symbol, z, current_volume, market_cap, dod_change)

            results.append({
                "symbol": symbol,
                "zscore_volume": round(z, 2),
                "percentile_volume": round(pctl, 2),
                "dod_change_pct": f"{round(dod_change, 2)}%",
                "volume_acceleration": round(volume_accel, 2),
                "volume_acceleration_formatted": f"{round(volume_accel, 2)}x",
                "current_volume": current_volume,
                "current_volume_formatted": format_currency(current_volume),
                "avg_volume": mu,
                "avg_volume_formatted": format_currency(mu),
                "market_cap": market_cap,
                "market_cap_formatted": format_currency(market_cap),
                "volume_to_mcap": vol_mcap_ratio,
                "volume_to_mcap_formatted": f"{round(vol_mcap_ratio, 2)}%",
                "volatility_7d": round(vol_7d, 2) if vol_7d is not None else None,
                "volatility_30d": round(vol_30d, 2) if vol_30d is not None else None
            })
        except Exception as e:
            st.warning(f"Error processing {symbol}: {str(e)}")
            continue

    df = pd.DataFrame(results)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    df_zsorted = df.sort_values(by="zscore_volume", ascending=False)
    df_liquidity = df.sort_values(by="volume_to_mcap", ascending=False)
    df_accel = df.sort_values(by="volume_acceleration", ascending=False)
    df_vol = df.dropna(subset=["volatility_7d", "volatility_30d"]).copy()
    return df_zsorted, df_liquidity, df_accel, df_vol

async def run_analysis():
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            tokens = await fetch_tokens(session)
            if not tokens:
                st.error("No tokens found. Please check your API key and try again.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                
            # Prepare tasks for volume history
            tasks = []
            for token in tokens:
                tasks.append((token["id"], token["symbol"], token["market_cap"], fetch_volume_history(session, token["id"])))

            # Gather all volume histories
            volume_data = await tqdm_asyncio.gather(*(t[3] for t in tasks))
            
            # Process the results
            zscore_df, liquidity_df, accel_df, vol_df = process_volume_stats(tasks, volume_data)
            return zscore_df, liquidity_df, accel_df, vol_df
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

async def get_sector_stats(session, sector_name, tokens):
    """Calculate statistics for a sector"""
    daily_volumes = {}
    
    # Process tokens in chunks to avoid rate limiting
    token_ids = list(tokens.keys())
    for i in range(0, len(token_ids), 5):  # Process 5 tokens at a time
        chunk = token_ids[i:i + 5]
        tasks = []
        for token_id in chunk:
            url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart"
            params = {"vs_currency": "usd", "days": "365"}
            tasks.append(make_request(session, url, params))
        
        chunk_results = await asyncio.gather(*tasks)
        
        for token_id, data in zip(chunk, chunk_results):
            if not data or "total_volumes" not in data:
                continue
                
            volumes = data.get("total_volumes", [])
            if len(volumes) < 2:
                continue
                
            # Convert to daily volumes
            for timestamp, volume in volumes:
                date = datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d')
                daily_volumes[date] = daily_volumes.get(date, 0) + float(volume)
        
        await asyncio.sleep(RATE_LIMIT_DELAY)
    
    if not daily_volumes:
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

async def get_token_stats(session, token_id, symbol):
    """Get statistics for a single token"""
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
    """Get detailed breakdown of tokens in a sector"""
    token_stats = []
    tasks = []
    
    for token_id, symbol in tokens.items():
        tasks.append(get_token_stats(session, token_id, symbol))
    
    results = await asyncio.gather(*tasks)
    for result in results:
        if result:
            token_stats.append(result)
    
    return sorted(token_stats, key=lambda x: x['zscore'], reverse=True)

async def get_token_breakdown_for_sector(sector, tokens):
    """Get token breakdown for a sector"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        token_stats = await get_sector_token_breakdown(session, sector, tokens)
    loop.close()
    return token_stats

async def run_sector_analysis(session):
    """Run sector analysis and return results"""
    all_stats = {}
    all_history = {}
    
    # Process all sectors in parallel
    tasks = []
    for sector_name, tokens in SECTORS.items():
        tasks.append(get_sector_stats(session, sector_name, tokens))
    
    results = await asyncio.gather(*tasks)
    
    # Collect results
    for (sector_name, tokens), (stats, history) in zip(SECTORS.items(), results):
        if stats is not None and history is not None and not history.empty:
            all_stats[sector_name] = stats
            all_history[sector_name] = history
    
    return all_stats, all_history

async def get_sector_token_breakdown_with_session(session, sector, tokens):
    """Get token breakdown for a sector with an existing session"""
    return await get_sector_token_breakdown(session, sector, tokens)

async def make_request(session, url, params=None):
    """Make an API request with retry logic"""
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

async def run_all_analysis():
    """Run both volume and sector analysis"""
    async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
        # Run volume analysis
        zscore_df, liquidity_df, accel_df, vol_df = await run_analysis()
        
        # Run sector analysis
        all_stats, all_history = await run_sector_analysis(session)
        
        return zscore_df, liquidity_df, accel_df, vol_df, all_stats, all_history

def get_perpetual_symbols():
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    response = requests.get(url).json()
    return [
        s["symbol"] for s in response["symbols"]
        if s["contractType"] == "PERPETUAL" and s["quoteAsset"] == "USDT"
    ]

def get_mark_price(symbol):
    url = "https://fapi.binance.com/fapi/v1/premiumIndex"
    response = requests.get(url, params={"symbol": symbol})
    return float(response.json()["markPrice"])

def get_avg_funding_rate(symbol):
    now = int(datetime.now().timestamp() * 1000)
    one_week_ago = now - 7 * 24 * 60 * 60 * 1000
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    params = {"symbol": symbol, "startTime": one_week_ago, "endTime": now, "limit": 1000}
    response = requests.get(url, params=params).json()
    if not response:
        return 0.0
    return sum(float(x["fundingRate"]) for x in response) / len(response)

def estimate_forward_pct(avg_8h_rate, days):
    return math.exp(avg_8h_rate * 3 * days) - 1

def process_symbol(symbol):
    try:
        avg_rate = get_avg_funding_rate(symbol)
        pct_3m = estimate_forward_pct(avg_rate, 90)
        pct_6m = estimate_forward_pct(avg_rate, 180)
        pct_12m = estimate_forward_pct(avg_rate, 365)
        return {
            "Symbol": symbol,
            "3M %": round(pct_3m * 100, 2),
            "6M %": round(pct_6m * 100, 2),
            "12M %": round(pct_12m * 100, 2),
        }
    except Exception as e:
        st.warning(f"⚠️ {symbol} skipped: {str(e)}")
        return None

def generate_derivatives_table():
    symbols = get_perpetual_symbols()
    st.info(f"🔄 Calculating implied forward % for {len(symbols)} symbols...")

    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    df = pd.DataFrame(results)
    df.sort_values("12M %", ascending=False, inplace=True)
    return df

def main():
    st.set_page_config(page_title="Flow Analysis", layout="wide")
    st.title("Flow Analysis Dashboard")
    
    # Initialize session state for active tab if not exists
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Volume Spikes"

    # Initialize session state for volatility timeframe if not exists
    if 'volatility_timeframe' not in st.session_state:
        st.session_state.volatility_timeframe = "7d"
        
    # Initialize session state for alert times if not exists
    if 'last_alert_times' not in st.session_state:
        st.session_state.last_alert_times = {}
        
    # Debug: Show current alert times
    if st.sidebar.checkbox("Show Alert Times"):
        st.sidebar.write("Last Alert Times:")
        for symbol, time in st.session_state.last_alert_times.items():
            st.sidebar.write(f"{symbol}: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
        if st.sidebar.button("Reset Alert Times"):
            st.session_state.last_alert_times = {}
            st.sidebar.success("Alert times have been reset")
    
    # Add key features description
    st.markdown("""
    ### Key Features
    - **Volume Spikes Analysis**: Detect unusual trading activity using statistical analysis
    - **Sector Analysis**: Analyze volume trends across different crypto sectors
    - **Volume Acceleration**: Track emerging trends through volume momentum
    - **Volatility Analysis**: Realised volatility dashboard across altcoins
    - **Liquidity Analysis**: Identify tokens with high trading volume relative to market cap
    """)
    
    # Add filters
    st.sidebar.header("Filters")
    min_market_cap = st.sidebar.number_input("Minimum Market Cap ($M)", 
                                            min_value=10, 
                                            max_value=10000, 
                                            value=300, 
                                            step=10)
    min_volume = st.sidebar.number_input("Minimum Daily Volume ($M)", 
                                        min_value=1, 
                                        max_value=100, 
                                        value=10, 
                                        step=1)
    
    # Add volatility timeframe selector to sidebar
    st.session_state.volatility_timeframe = st.sidebar.radio(
        "Volatility Timeframe",
        ["7d", "30d"],
        index=0 if st.session_state.volatility_timeframe == "7d" else 1
    )
    
    # Add refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
    
    # Run analysis
    with st.spinner("Fetching and analyzing data..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        zscore_df, liquidity_df, accel_df, vol_df, all_stats, all_history = loop.run_until_complete(run_all_analysis())
        loop.close()
    
    if len(zscore_df) == 0:
        st.error("No data available. Please try again later or check your API key.")
        return
    
    # Apply filters
    zscore_df = zscore_df[zscore_df["market_cap"] >= min_market_cap * 1e6]
    zscore_df = zscore_df[zscore_df["current_volume"] >= min_volume * 1e6]
    
    # Apply same filters to other dataframes
    liquidity_df = liquidity_df[liquidity_df["market_cap"] >= min_market_cap * 1e6]
    liquidity_df = liquidity_df[liquidity_df["current_volume"] >= min_volume * 1e6]
    
    accel_df = accel_df[accel_df["market_cap"] >= min_market_cap * 1e6]
    accel_df = accel_df[accel_df["current_volume"] >= min_volume * 1e6]
    
    vol_df = vol_df[vol_df["market_cap"] >= min_market_cap * 1e6]
    vol_df = vol_df[vol_df["current_volume"] >= min_volume * 1e6]
    
    # Apply volatility filter
    vol_df = vol_df.sort_values(by="volatility_7d", ascending=False)
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Volume Spikes", "Sector Analysis", "Acceleration", "Volatility", "Liquidity"])

    # Volume Spikes Tab
    with tab1:
        st.header("Top Volume Spike Candidates")
        
        st.markdown("""
        This section identifies unusual trading activity using statistical analysis. The Z-score measures how many standard deviations the current volume is from the historical average. A high Z-score (>2) indicates significantly higher than normal trading activity.
        """)
        
        if len(zscore_df) > 0:
            # Create a new DataFrame for the plot
            plot_df = zscore_df.head(20).copy()
            plot_df['abs_zscore'] = abs(plot_df['zscore_volume'])
            
            # Create scatter plot
            fig = px.scatter(plot_df,
                            x="market_cap",
                            y="current_volume",
                            size="abs_zscore",
                            color="zscore_volume",
                            color_continuous_scale=["red", "yellow", "green"],
                            hover_name="symbol",
                            log_x=True,
                            log_y=True,
                            title="Volume vs Market Cap (Size by |Z-score|, Color by Z-score)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with formatted values
            display_cols = ["symbol", "zscore_volume", "percentile_volume", "dod_change_pct", 
                          "current_volume_formatted", "market_cap_formatted"]
            st.dataframe(zscore_df.head(20)[display_cols].rename(columns={
                "zscore_volume": "Z-Score",
                "percentile_volume": "Percentile",
                "dod_change_pct": "24h Change",
                "current_volume_formatted": "Current Volume",
                "market_cap_formatted": "Market Cap"
            }), use_container_width=True)
        else:
            st.warning("No tokens found matching the criteria.")

    # Sector Analysis Tab
    with tab2:
        st.header("Sector Volume Analysis")
        
        st.markdown("""
        This section examines trading volume across sectors. The Z-score indicates how many standard deviations the current sector volume deviates from its historical average.
        """)
        
        if all_stats:
            # Create DataFrame for sector stats
            sector_data = []
            for sector, stats in all_stats.items():
                sector_data.append({
                    "Sector": sector,
                    "24H Volume": stats["current_volume"],  # Raw value for plotting
                    "24H Volume Formatted": format_currency(stats["current_volume"]),  # Formatted for display
                    "Avg Volume": format_currency(stats["avg_volume"]),
                    "Z-Score": stats["zscore_volume"],  # Raw value for plotting
                    "Z-Score Formatted": f"{stats['zscore_volume']:.2f}",  # Formatted for display
                    "24h Change": stats["dod_change_pct"],  # Raw value for plotting
                    "24h Change Formatted": f"{stats['dod_change_pct']:.2f}%"  # Formatted for display
                })
            
            sector_df = pd.DataFrame(sector_data)
            sector_df = sector_df.sort_values(by="Z-Score", ascending=False)
            
            # Display sector stats table with formatted values
            display_df = sector_df.copy()
            display_df = display_df[["Sector", "24H Volume Formatted", "Avg Volume", "Z-Score Formatted", "24h Change Formatted"]]
            display_df.columns = ["Sector", "24H Volume", "Avg Volume", "Z-Score", "24h Change"]
            st.dataframe(display_df, use_container_width=True)
            
            # Show token breakdown for all sectors
            st.subheader("Token Breakdown by Sector")
            
            # Create a selectbox to choose which sector to view, ordered by Z-score
            selected_sector = st.selectbox(
                "Select a sector to view its token breakdown:",
                options=sector_df["Sector"].tolist(),  # Use the sorted list of sectors
                index=0
            )
            
            # Get token breakdown for the selected sector
            with st.spinner(f"Loading token data for {selected_sector}..."):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                token_stats = loop.run_until_complete(get_token_breakdown_for_sector(selected_sector, SECTORS[selected_sector]))
                loop.close()
                
                # Create DataFrame for token stats
                token_data = []
                for token in token_stats:
                    token_data.append({
                        "Token": token["symbol"],
                        "24H Volume": format_currency(token["current_volume"]),
                        "Avg Volume": format_currency(token["avg_volume"]),
                        "Z-Score": f"{token['zscore']:.2f}"
                    })
                
                token_df = pd.DataFrame(token_data)
                st.dataframe(token_df, use_container_width=True)
                
                # Add a small note about high Z-score sectors
                if all_stats[selected_sector]['zscore_volume'] > 2:
                    st.info(f"Note: {selected_sector} sector is currently showing high volume activity (Z-score > 2)")
        else:
            st.warning("No sector data available. Please try again later.")

    # Acceleration Tab
    with tab3:
        st.header("Top Volume Accelerators")
        
        st.markdown("""
        This section tracks emerging trends by comparing current volume to the 7-day average. A high acceleration value indicates rapidly increasing trading activity, which can signal growing interest in a token.
        """)
        
        if len(accel_df) > 0:
            # Create a new DataFrame for the plot
            plot_df = accel_df.head(20).copy()
            plot_df['abs_accel'] = abs(plot_df['volume_acceleration'])
            
            # Create scatter plot
            fig = px.scatter(plot_df,
                            x="market_cap",
                            y="current_volume",
                            size="abs_accel",
                            color="volume_acceleration",
                            color_continuous_scale=["red", "yellow", "green"],
                            hover_name="symbol",
                            log_x=True,
                            log_y=True,
                            title="Volume vs Market Cap (Size by |Acceleration|, Color by Acceleration)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with formatted values
            display_cols = ["symbol", "volume_acceleration_formatted", "current_volume_formatted", "avg_volume_formatted"]
            st.dataframe(accel_df.head(20)[display_cols].rename(columns={
                "volume_acceleration_formatted": "Volume Acceleration",
                "current_volume_formatted": "Current Volume",
                "avg_volume_formatted": "Average Volume"
            }), use_container_width=True)
        else:
            st.warning("No tokens found matching the criteria.")

    # Volatility Tab
    with tab4:
        st.header("Volatility Analysis")
        
        st.markdown("""
        This section analyzes price volatility using Parkinson's method, which uses high/low price ranges to provide a more accurate measure of volatility than simple returns.
        """)
        
        st.markdown("""
        The volatility shown is annualized (converted to a yearly rate) and expressed as a percentage. You can switch between 7-day and 30-day calculation windows using the timeframe selector in the sidebar.
        """)
        
        if len(vol_df) > 0:
            # Create a new DataFrame for the plot
            plot_df = vol_df.copy()  # Remove .head(20) to show all tokens
            
            # Create scatter plot
            fig = px.scatter(plot_df,
                            x="market_cap",
                            y="volatility_7d" if st.session_state.volatility_timeframe == "7d" else "volatility_30d",
                            size="current_volume",
                            color="volatility_7d" if st.session_state.volatility_timeframe == "7d" else "volatility_30d",
                            color_continuous_scale="Viridis",
                            hover_name="symbol",
                            log_x=True,
                            title=f"Realized Volatility ({st.session_state.volatility_timeframe})")
            
            fig.update_layout(
                yaxis_title=f"Realized Volatility % ({st.session_state.volatility_timeframe})",
                xaxis_title="Market Cap (USD)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with formatted values
            display_cols = ["symbol", "volatility_7d" if st.session_state.volatility_timeframe == "7d" else "volatility_30d", "current_volume_formatted", "market_cap_formatted"]
            st.dataframe(vol_df[display_cols].rename(columns={  # Remove .head(20) to show all tokens
                "volatility_7d" if st.session_state.volatility_timeframe == "7d" else "volatility_30d": f"{st.session_state.volatility_timeframe} Volatility %",
                "current_volume_formatted": "Volume",
                "market_cap_formatted": "Market Cap"
            }), use_container_width=True)
            
            # Add a count of total tokens shown
            st.write(f"Total tokens shown: {len(vol_df)}")
        else:
            st.warning("No tokens found matching the criteria.")

    # Liquidity Tab
    with tab5:
        st.header("Highest Volume/Market Cap Ratios")
        
        st.markdown("""
        This section identifies tokens with high trading volume relative to their market cap. A high volume/market cap ratio indicates strong liquidity and trading activity. This can be useful for identifying tokens that are actively traded despite their size.
        """)
        
        if len(liquidity_df) > 0:
            # Create a new DataFrame for the plot
            plot_df = liquidity_df.head(20).copy()
            
            # Create scatter plot
            fig = px.scatter(plot_df,
                            x="market_cap",
                            y="current_volume",
                            size="volume_to_mcap",
                            color="volume_to_mcap",
                            color_continuous_scale="Viridis",
                            hover_name="symbol",
                            log_x=True,
                            log_y=True,
                            title="Volume vs Market Cap (Size and Color by Volume/MCap Ratio)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table with formatted values
            display_cols = ["symbol", "volume_to_mcap_formatted", "current_volume_formatted", "market_cap_formatted"]
            st.dataframe(liquidity_df.head(20)[display_cols].rename(columns={
                "volume_to_mcap_formatted": "Volume/MCap",
                "current_volume_formatted": "Current Volume",
                "market_cap_formatted": "Market Cap"
            }), use_container_width=True)
        else:
            st.warning("No tokens found matching the criteria.")

    # Add timestamp
    st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 