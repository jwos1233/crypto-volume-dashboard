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
IGNORE_SYMBOLS = {"EETH","BUSD","MSOL","XSOLVBTC","USDT", "FDUSD", "USDC", "WBTC", "WETH", "USDD", "LBTC", "TBTC", "USDT0", "SOLVBTC", "CLBTC"}
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds
TIMEOUT = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout
ZSCORE_THRESHOLD = 2.0  # Z-score threshold for alerts

def send_telegram_alert(symbol, zscore, current_volume, market_cap, volume_change):
    """Send a Telegram alert for significant volume spikes"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram credentials not configured. Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in your environment variables.")
        return
    
    message = (
        f"🚨 Volume Spike Alert!\n\n"
        f"Token: {symbol}\n"
        f"Z-Score: {zscore:.2f}\n"
        f"Current Volume: {format_currency(current_volume)}\n"
        f"Market Cap: {format_currency(market_cap)}\n"
        f"24h Change: {volume_change}%\n\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    params = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
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
                
            st.info(f"Found {len(tokens)} tokens. Computing volume statistics...")
            
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

def main():
    st.set_page_config(page_title="Crypto Volume Analysis", layout="wide")
    st.title("Crypto Volume Analysis Dashboard")
    
    # Initialize session state for active tab if not exists
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Volume Spikes"

    # Initialize session state for volatility timeframe if not exists
    if 'volatility_timeframe' not in st.session_state:
        st.session_state.volatility_timeframe = "7d"

    # Add key features description
    st.markdown("""
    ### Key Features
    - **Volume Spikes Analysis**: Detect unusual trading activity using statistical analysis
    - **Liquidity Analysis**: Identify tokens with high trading volume relative to market cap
    - **Volume Acceleration**: Track emerging trends through volume momentum
    """)
    
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
    
    # Add volatility filters
    st.sidebar.subheader("Volatility Settings")
    min_volatility = st.sidebar.number_input("Minimum Volatility (%)", 
                                            min_value=0, 
                                            max_value=500, 
                                            value=50, 
                                            step=10)
    
    # Add refresh button
    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
    
    # Run analysis
    with st.spinner("Fetching and analyzing data..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        zscore_df, liquidity_df, accel_df, vol_df = loop.run_until_complete(run_analysis())
        loop.close()
    
    if len(zscore_df) == 0:
        st.error("No data available. Please try again later or check your API key.")
        return
    
    # Apply filters
    zscore_df = zscore_df[zscore_df["market_cap"] >= min_market_cap * 1e6]
    zscore_df = zscore_df[zscore_df["current_volume"] >= min_volume * 1e6]
    
    # Apply volatility filter
    vol_df = vol_df[vol_df["volatility_7d"] >= min_volatility]
    vol_df = vol_df.sort_values(by="volatility_7d", ascending=False)
    
    # Create tabs
    tabs = ["Volume Spikes", "Liquidity", "Acceleration", "Volatility"]

    # Create tab buttons in a row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Volume Spikes", key="tab1"):
            st.session_state.active_tab = "Volume Spikes"
    with col2:
        if st.button("Liquidity", key="tab2"):
            st.session_state.active_tab = "Liquidity"
    with col3:
        if st.button("Acceleration", key="tab3"):
            st.session_state.active_tab = "Acceleration"
    with col4:
        if st.button("Volatility", key="tab4"):
            st.session_state.active_tab = "Volatility"

    # Get current active tab
    active_tab = st.session_state.active_tab

    # Show all tabs content
    with active_tab:
        if active_tab == "Volume Spikes":
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
        
        elif active_tab == "Liquidity":
            st.header("Highest Volume/Market Cap Ratios")
            st.write("Tokens with highest volume relative to market cap (shown as percentage)")
            
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
        
        elif active_tab == "Acceleration":
            st.header("Top Volume Accelerators")
            st.write("Tokens with highest volume acceleration (Current Volume / 7-day Average)")
            
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
        
        elif active_tab == "Volatility":
            st.header("Volatility Analysis")
            
            st.markdown("""
            ### Volatility Analysis
            This section analyzes price volatility using Parkinson's method, which uses high/low price ranges to provide a more accurate measure of volatility than simple returns.
            """)
            
            # Add timeframe selector at the top of the volatility tab
            timeframe = st.radio(
                "Select Timeframe:",
                ["7d", "30d"],
                index=0 if st.session_state.volatility_timeframe == "7d" else 1,
                key="vol_timeframe",
                horizontal=True
            )
            
            # Update session state when timeframe changes
            if timeframe != st.session_state.volatility_timeframe:
                st.session_state.volatility_timeframe = timeframe
                # Force a rerun to update the display
                st.experimental_rerun()
            
            st.markdown("""
            The volatility shown is annualized (converted to a yearly rate) and expressed as a percentage. For example:
            - A 50% volatility means the asset's price could move up or down by 50% over a year
            - You can switch between 7-day and 30-day calculation windows using the timeframe selector above
            """)
            
            st.write(f"Realized volatility over {st.session_state.volatility_timeframe} timeframe")
            
            if len(vol_df) > 0:
                # Create a new DataFrame for the plot
                plot_df = vol_df.head(20).copy()
                
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
                st.dataframe(vol_df.head(20)[display_cols].rename(columns={
                    "volatility_7d" if st.session_state.volatility_timeframe == "7d" else "volatility_30d": f"{st.session_state.volatility_timeframe} Volatility %",
                    "current_volume_formatted": "Volume",
                    "market_cap_formatted": "Market Cap"
                }), use_container_width=True)
            else:
                st.warning("No tokens found matching the criteria.")
    
    # Add timestamp
    st.sidebar.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 