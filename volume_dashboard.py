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

# Load environment variables
load_dotenv()

nest_asyncio.apply()

# Try to get API key from different sources
API_KEY = st.secrets.get("COINGECKO_API_KEY") or os.getenv('COINGECKO_API_KEY')

if not API_KEY:
    st.error("No API key found. Please set the COINGECKO_API_KEY in your environment or Streamlit secrets.")
    st.stop()

HEADERS = {"x-cg-pro-api-key": API_KEY}
MIN_MARKET_CAP = 300_000_000  # $300M
IGNORE_SYMBOLS = {"XSOLVBTC","USDT", "FDUSD", "USDC", "WBTC", "WETH", "USDD", "LBTC", "TBTC", "USDT0", "SOLVBTC", "CLBTC"}
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
TIMEOUT = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout

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
                    return None
                data = await resp.json()
                volumes = data.get("total_volumes", [])
                if len(volumes) < 2:
                    return None
                return [float(v[1]) for v in volumes]
        except Exception as e:
            if retry < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            st.error(f"Error fetching volume for {token_id}: {str(e)}")
            return None
    return None

@st.cache_data(ttl=300)
def process_volume_stats(_tokens, volume_data):
    results = []
    for (token_id, symbol, market_cap, _), volumes in zip(_tokens, volume_data):
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
                "current_volume": current_volume,
                "avg_volume": mu,
                "market_cap": market_cap,
                "volume_to_mcap": round(vol_mcap_ratio, 4)
            })
        except Exception as e:
            st.warning(f"Error processing {symbol}: {str(e)}")
            continue

    df = pd.DataFrame(results)
    if len(df) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    df_zsorted = df.sort_values(by="zscore_volume", ascending=False)
    df_liquidity = df.sort_values(by="volume_to_mcap", ascending=False)
    df_accel = df.sort_values(by="volume_acceleration", ascending=False)
    return df_zsorted, df_liquidity, df_accel

async def run_analysis():
    try:
        async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
            tokens = await fetch_tokens(session)
            if not tokens:
                st.error("No tokens found. Please check your API key and try again.")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
                
            st.info(f"Found {len(tokens)} tokens. Computing volume statistics...")
            
            # Prepare tasks for volume history
            tasks = []
            for token in tokens:
                tasks.append((token["id"], token["symbol"], token["market_cap"], fetch_volume_history(session, token["id"])))

            # Gather all volume histories
            volume_data = await tqdm_asyncio.gather(*(t[3] for t in tasks))
            
            # Process the results
            zscore_df, liquidity_df, accel_df = process_volume_stats(tasks, volume_data)
            return zscore_df, liquidity_df, accel_df
    except Exception as e:
        st.error(f"Error running analysis: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def main():
    st.set_page_config(page_title="Crypto Volume Analysis", layout="wide")
    st.title("Crypto Volume Analysis Dashboard")
    
    # Add filters
    st.sidebar.header("Filters")
    min_market_cap = st.sidebar.number_input("Minimum Market Cap ($M)", 
                                            min_value=100, 
                                            max_value=1000, 
                                            value=300, 
                                            step=50)
    min_volume = st.sidebar.number_input("Minimum Daily Volume ($M)", 
                                        min_value=1, 
                                        max_value=100, 
                                        value=10, 
                                        step=1)
    
    # Add refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
    
    # Run analysis
    with st.spinner("Fetching and analyzing data..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        zscore_df, liquidity_df, accel_df = loop.run_until_complete(run_analysis())
        loop.close()
    
    if len(zscore_df) == 0:
        st.error("No data available. Please try again later or check your API key.")
        return
    
    # Apply filters
    zscore_df = zscore_df[zscore_df["market_cap"] >= min_market_cap * 1e6]
    zscore_df = zscore_df[zscore_df["current_volume"] >= min_volume * 1e6]
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Volume Spikes", "Liquidity", "Acceleration"])
    
    with tab1:
        st.header("Top Volume Spike Candidates")
        st.write("Tokens with unusual volume spikes (Z-score > 2)")
        
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
            
            # Display table
            st.dataframe(zscore_df.head(20)[["symbol", "zscore_volume", "percentile_volume", 
                                            "dod_change_pct", "current_volume", "market_cap"]],
                        use_container_width=True)
        else:
            st.warning("No tokens found matching the criteria.")
    
    with tab2:
        st.header("Highest Volume/Market Cap Ratios")
        st.write("Tokens with highest volume relative to market cap")
        
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
                            title="Volume vs Market Cap (Size by Volume/Market Cap Ratio)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.dataframe(liquidity_df.head(20)[["symbol", "volume_to_mcap", "current_volume", "market_cap"]],
                        use_container_width=True)
        else:
            st.warning("No tokens found matching the criteria.")
    
    with tab3:
        st.header("Top Volume Accelerators")
        st.write("Tokens with highest volume acceleration (Current vs 7d Avg)")
        
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
            
            # Display table
            st.dataframe(accel_df.head(20)[["symbol", "volume_acceleration", "current_volume", "avg_volume"]],
                        use_container_width=True)
        else:
            st.warning("No tokens found matching the criteria.")
    
    # Add timestamp
    st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 