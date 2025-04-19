import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# CoinGecko Pro API key
API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": API_KEY}

# ----------------------------
# FETCH STABLECOIN MARKET CAP
# ----------------------------
def fetch_stablecoin_marketcap():
    def fetch_coin_marketcap(coin_id):
        url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": "usd",
            "days": 365,
            "interval": "daily"
        }
        response = requests.get(url, params=params, headers=HEADERS)
        if response.status_code != 200:
            return None
        data = response.json()['market_caps']
        df = pd.DataFrame(data, columns=['timestamp', 'marketcap'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['date', 'marketcap']].set_index('date')

    usdt = fetch_coin_marketcap('tether')
    usdc = fetch_coin_marketcap('usd-coin')

    if usdt is not None and usdc is not None:
        combined = usdt.join(usdc, lsuffix='_usdt', rsuffix='_usdc')
        combined['total_stablecoin_mcap'] = combined['marketcap_usdt'] + combined['marketcap_usdc']
        return combined[['total_stablecoin_mcap']]
    else:
        return None

# ----------------------------
# COINGECKO DATA COLLECTION
# ----------------------------
def get_top_100_tokens():
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 100,
        "page": 1,
        "sparkline": False
    }
    response = requests.get(url, params=params, headers=HEADERS)
    coins = response.json()
    stablecoin_symbols = {"usdt", "usdc", "dai", "tusd", "usdp", "gusd", "usdd", "lusd", "frax", "busd"}
    return {
        coin["id"]: coin["symbol"].upper()
        for coin in coins if coin["symbol"].lower() not in stablecoin_symbols
    }

def get_daily_price_history(coin_id, days):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": "usd",
        "days": days,
        "interval": "daily"
    }
    response = requests.get(url, params=params, headers=HEADERS)
    if response.status_code != 200:
        return None
    prices = response.json()["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
    df = df.drop_duplicates(subset="date", keep="last")
    df.set_index("date", inplace=True)
    return df["price"]

def fetch_prices_threaded(token_id, symbol, days):
    try:
        prices = get_daily_price_history(token_id, days)
        return symbol, prices
    except:
        return symbol, None

def compute_breadth(token_map, days):
    price_df = pd.DataFrame()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(fetch_prices_threaded, token_id, symbol, days): symbol
            for token_id, symbol in token_map.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            symbol, prices = future.result()
            if prices is not None:
                price_df[symbol] = prices

    price_df.dropna(axis=1, inplace=True)
    ma50_df = price_df.rolling(window=50).mean()
    breadth = (price_df > ma50_df).sum(axis=1) / price_df.shape[1] * 100
    return breadth, price_df

def compute_implied_correlation(price_df, window=30):
    log_returns = np.log(price_df / price_df.shift(1))
    implied_corr = []
    for i in range(window, len(log_returns)):
        window_returns = log_returns.iloc[i - window:i]
        corr_matrix = window_returns.corr()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        avg_corr = upper_triangle.stack().mean()
        implied_corr.append(avg_corr)
    full_corr = pd.Series([np.nan] * window + implied_corr, index=log_returns.index)
    return full_corr.to_frame(name="implied_correlation")

def plot_all_charts(breadth, price_df, implied_corr_df, stablecoin_df):
    ma50 = price_df.rolling(window=50).mean().iloc[-1]
    latest_prices = price_df.iloc[-1]
    ma_distance = ((latest_prices - ma50) / ma50 * 100).dropna()
    top5 = ma_distance.sort_values(ascending=False).head(5)
    bottom5 = ma_distance.sort_values().head(5)
    current_breadth = breadth.iloc[-1]
    percentile = (breadth < current_breadth).sum() / len(breadth) * 100

    fig, axs = plt.subplots(3, 2, figsize=(18, 16), gridspec_kw={"height_ratios": [2, 1, 1]})

    # Breadth
    ax1 = axs[0, 0]
    ax1.plot(breadth.index, breadth.values, label="Breadth (% Above MA50)", color="blue")
    ax1.axhline(current_breadth, linestyle="--", color="gray", alpha=0.6)
    ax1.set_title("Crypto Breadth Indicator – % of tokens over the 50-day MA (Top 100 by Mcap)")
    ax1.set_ylabel("Percentage (%)")
    ax1.grid(True)
    ax1.legend()

    # Stablecoin market cap
    ax_stable = axs[0, 1]
    if stablecoin_df is not None:
        stablecoin_df.index = pd.to_datetime(stablecoin_df.index)
        stablecoin_df = stablecoin_df.reindex(pd.to_datetime(breadth.index), method='ffill')
        print("Stablecoin plot data points:", stablecoin_df.dropna().shape)
        ax_stable.plot(stablecoin_df.index, stablecoin_df['total_stablecoin_mcap'] / 1e9, label="USDT + USDC", color="darkorange")
        ax_stable.set_title("Stablecoin Market Cap (USDT + USDC, in $B)")
        ax_stable.set_ylabel("Market Cap (B USD)")
        ax_stable.grid(True)
        ax_stable.legend()
    else:
        ax_stable.text(0.5, 0.5, 'Stablecoin data unavailable', ha='center', va='center', fontsize=12)
        ax_stable.set_title("Stablecoin Market Cap")
        ax_stable.axis('off')

    # MA distance bar chart
    ax2 = axs[1, 0]
    combined = pd.concat([top5, bottom5])
    colors = ["green" if x > 0 else "red" for x in combined]
    combined.plot(kind="bar", ax=ax2, color=colors)
    ax2.set_title("Strongest / Weakest Tokens – Performance vs 50-day MA (Top 100 by Mcap)")
    ax2.set_ylabel("% Distance")
    ax2.grid(True)

    # Implied correlation
    ax3 = axs[1, 1]
    ax3.plot(implied_corr_df.index, implied_corr_df["implied_correlation"], color="purple")
    ax3.set_title("Rolling 30-Day Implied Correlation Across Top 100 tokens (by Mcap)")
    ax3.set_ylabel("Correlation")
    ax3.grid(True)

    # Placeholder row 3
    axs[2, 0].axis('off')
    axs[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    lookback_days = 730
    token_map = get_top_100_tokens()
    breadth, price_df = compute_breadth(token_map, lookback_days)
    implied_corr_df = compute_implied_correlation(price_df, window=30)
    stablecoin_df = fetch_stablecoin_marketcap()

    if stablecoin_df is not None:
        print("\n✅ Stablecoin Market Cap (Aligned):")
        print(stablecoin_df.tail())
    else:
        print("\n❌ Stablecoin market cap data not retrieved.")

    plot_all_charts(breadth, price_df, implied_corr_df, stablecoin_df)

if __name__ == "__main__":
    main()