[1mdiff --git a/volume_dashboard.py b/volume_dashboard.py[m
[1mindex 5d7efc5..a96a272 100644[m
[1m--- a/volume_dashboard.py[m
[1m+++ b/volume_dashboard.py[m
[36m@@ -11,22 +11,17 @@[m [mimport plotly.graph_objects as go[m
 import time[m
 from datetime import datetime[m
 import os[m
[31m-from dotenv import load_dotenv[m
 import contextlib[m
 [m
[31m-# Load environment variables[m
[31m-load_dotenv()[m
[31m-[m
 nest_asyncio.apply()[m
 [m
[31m-# Try to get API key from different sources[m
[32m+[m[32m# Try to get API keys from Streamlit secrets[m
 try:[m
     API_KEY = st.secrets["COINGECKO_API_KEY"][m
[32m+[m[32m    TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", None)[m
[32m+[m[32m    TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", None)[m
 except:[m
[31m-    API_KEY = os.getenv('COINGECKO_API_KEY')[m
[31m-[m
[31m-if not API_KEY:[m
[31m-    st.error("No API key found. Please set the COINGECKO_API_KEY in your environment or Streamlit secrets.")[m
[32m+[m[32m    st.error("No API key found. Please set the COINGECKO_API_KEY in your Streamlit secrets.")[m
     st.stop()[m
 [m
 HEADERS = {"x-cg-pro-api-key": API_KEY}[m
[36m@@ -35,6 +30,7 @@[m [mIGNORE_SYMBOLS = {"XSOLVBTC","USDT", "FDUSD", "USDC", "WBTC", "WETH", "USDD", "L[m
 MAX_RETRIES = 3[m
 RETRY_DELAY = 1  # seconds[m
 TIMEOUT = aiohttp.ClientTimeout(total=30)  # 30 seconds timeout[m
[32m+[m[32mZSCORE_THRESHOLD = 2.0  # Z-score threshold for alerts[m
 [m
 @st.cache_resource[m
 def get_session():[m
[36m@@ -118,9 +114,29 @@[m [masync def fetch_volume_history(_session, token_id):[m
             return None[m
     return None[m
 [m
[32m+[m[32masync def send_telegram_alert(session, message):[m
[32m+[m[32m    """Send alert to Telegram."""[m
[32m+[m[32m    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):[m
[32m+[m[32m        return[m
[32m+[m[41m        [m
[32m+[m[32m    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"[m
[32m+[m[32m    data = {[m
[32m+[m[32m        "chat_id": TELEGRAM_CHAT_ID,[m
[32m+[m[32m        "text": message,[m
[32m+[m[32m        "parse_mode": "HTML"[m
[32m+[m[32m    }[m
[32m+[m[41m    [m
[32m+[m[32m    try:[m
[32m+[m[32m        async with session.post(url, json=data) as response:[m
[32m+[m[32m            if response.status != 200:[m
[32m+[m[32m                st.error(f"Failed to send Telegram alert: {await response.text()}")[m
[32m+[m[32m    except Exception as e:[m
[32m+[m[32m        st.error(f"Error sending Telegram alert: {str(e)}")[m
[32m+[m
 @st.cache_data(ttl=300)[m
 def process_volume_stats(_tokens, volume_data):[m
     results = [][m
[32m+[m[32m    alerts = [][m
     for (token_id, symbol, market_cap, _), volumes in zip(_tokens, volume_data):[m
         if volumes is None or len(volumes) < 10:[m
             continue[m
[36m@@ -140,6 +156,19 @@[m [mdef process_volume_stats(_tokens, volume_data):[m
             vol_mcap_ratio = current_volume / market_cap[m
             volume_accel = current_volume / avg_volume_7d if avg_volume_7d != 0 else 0[m
 [m
[32m+[m[32m            # Check for alert condition[m
[32m+[m[32m            if abs(z) >= ZSCORE_THRESHOLD:[m
[32m+[m[32m                alert_msg = ([m
[32m+[m[32m                    f"🚨 <b>High Volume Alert</b> 🚨\n"[m
[32m+[m[32m                    f"Token: {symbol}\n"[m
[32m+[m[32m                    f"Z-score: {z:.2f}\n"[m
[32m+[m[32m                    f"Volume Change: {dod_change:+.2f}%\n"[m
[32m+[m[32m                    f"Current Volume: ${current_volume:,.0f}\n"[m
[32m+[m[32m                    f"Market Cap: ${market_cap:,.0f}\n"[m
[32m+[m[32m                    f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"[m
[32m+[m[32m                )[m
[32m+[m[32m                alerts.append(alert_msg)[m
[32m+[m
             results.append({[m
                 "symbol": symbol,[m
                 "zscore_volume": round(z, 2),[m
[36m@@ -157,12 +186,12 @@[m [mdef process_volume_stats(_tokens, volume_data):[m
 [m
     df = pd.DataFrame(results)[m
     if len(df) == 0:[m
[31m-        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()[m
[32m+[m[32m        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), alerts[m
         [m
     df_zsorted = df.sort_values(by="zscore_volume", ascending=False)[m
     df_liquidity = df.sort_values(by="volume_to_mcap", ascending=False)[m
     df_accel = df.sort_values(by="volume_acceleration", ascending=False)[m
[31m-    return df_zsorted, df_liquidity, df_accel[m
[32m+[m[32m    return df_zsorted, df_liquidity, df_accel, alerts[m
 [m
 async def run_analysis():[m
     try:[m
[36m@@ -183,7 +212,12 @@[m [masync def run_analysis():[m
             volume_data = await tqdm_asyncio.gather(*(t[3] for t in tasks))[m
             [m
             # Process the results[m
[31m-            zscore_df, liquidity_df, accel_df = process_volume_stats(tasks, volume_data)[m
[32m+[m[32m            zscore_df, liquidity_df, accel_df, alerts = process_volume_stats(tasks, volume_data)[m
[32m+[m[41m            [m
[32m+[m[32m            # Send alerts if any[m
[32m+[m[32m            for alert in alerts:[m
[32m+[m[32m                await send_telegram_alert(session, alert)[m
[32m+[m[41m                [m
             return zscore_df, liquidity_df, accel_df[m
     except Exception as e:[m
         st.error(f"Error running analysis: {str(e)}")[m
[36m@@ -191,7 +225,26 @@[m [masync def run_analysis():[m
 [m
 def main():[m
     st.set_page_config(page_title="Crypto Volume Analysis", layout="wide")[m
[32m+[m[41m    [m
[32m+[m[32m    # Title and Description[m
     st.title("Crypto Volume Analysis Dashboard")[m
[32m+[m[32m    st.markdown("""[m
[32m+[m[32m    This dashboard analyzes cryptocurrency trading volumes to identify unusual patterns and potential opportunities.[m
[32m+[m[41m    [m
[32m+[m[32m    ### Features:[m
[32m+[m[32m    - **Volume Spikes**: Identifies tokens with unusual trading volume (Z-score analysis)[m
[32m+[m[32m    - **Liquidity**: Shows tokens with highest volume relative to market cap[m
[32m+[m[32m    - **Volume Acceleration**: Tracks tokens with increasing trading activity[m
[32m+[m[32m    - **Telegram Alerts**: Sends notifications when volume Z-scores exceed 2.0[m
[32m+[m[41m    [m
[32m+[m[32m    Data refreshes every 5 minutes. Use the sidebar to adjust filters and refresh data manually.[m
[32m+[m[32m    """)[m
[32m+[m[41m    [m
[32m+[m[32m    # Add Telegram status[m
[32m+[m[32m    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:[m
[32m+[m[32m        st.sidebar.success("✅ Telegram alerts are enabled")[m
[32m+[m[32m    else:[m
[32m+[m[32m        st.sidebar.warning("⚠️ Telegram alerts are disabled. Add bot token and chat ID to enable.")[m
     [m
     # Add filters[m
     st.sidebar.header("Filters")[m
