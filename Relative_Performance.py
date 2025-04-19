import asyncio
import aiohttp
import pandas as pd
from tqdm.asyncio import tqdm_asyncio
from collections import Counter

API_KEY = "CG-CD33ohgugtpwgHfLHhEDT3yD"
HEADERS = {"x-cg-pro-api-key": API_KEY}


async def get_top_100_tokens(session):
    url = "https://pro-api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 100, "page": 1}
    async with session.get(url, headers=HEADERS, params=params) as resp:
        return await resp.json()


async def get_price_change(session, token_id, days):
    url = f"https://pro-api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {"vs_currency": "usd", "days": days}
    async with session.get(url, headers=HEADERS, params=params) as resp:
        if resp.status != 200:
            return None
        data = await resp.json()
        prices = data.get("prices", [])
        if len(prices) < 2:
            return None
        start_price, end_price = prices[0][1], prices[-1][1]
        return (end_price - start_price) / start_price * 100


async def get_relative_performance(session, tokens, days):
    btc_return = await get_price_change(session, "bitcoin", days)
    performance = []

    tasks = []
    for token in tokens:
        tasks.append((token["symbol"].upper(), get_price_change(session, token["id"], days)))

    results = await tqdm_asyncio.gather(*(t[1] for t in tasks))

    for (symbol, _), token_return in zip(tasks, results):
        if token_return is None:
            continue
        relative_return = token_return - btc_return
        performance.append({
            "symbol": symbol,
            "relative_return": round(relative_return, 2)
        })

    df = pd.DataFrame(performance)
    return df.sort_values(by="relative_return", ascending=False)


def print_leaderboard(df_7, df_30, df_90, top_n=10):  # Top/bottom 10
    print("\nRelative Strength")
    print(f"{'7d':<25}{'30d':<30}{'90d'}")
    top_7 = df_7.head(top_n)["symbol"].tolist()
    top_30 = df_30.head(top_n)["symbol"].tolist()
    top_90 = df_90.head(top_n)["symbol"].tolist()

    for i in range(top_n):
        print(f"{top_7[i]:<25}{top_30[i]:<30}{top_90[i]}")

    print("\nRelative Weakness")
    print(f"{'7d':<25}{'30d':<30}{'90d'}")
    bot_7 = df_7.tail(top_n)["symbol"].tolist()
    bot_30 = df_30.tail(top_n)["symbol"].tolist()
    bot_90 = df_90.tail(top_n)["symbol"].tolist()

    for i in range(top_n):
        print(f"{bot_7[i]:<25}{bot_30[i]:<30}{bot_90[i]}")

    # Relative Strength Summary
    top_counter = Counter(top_7 + top_30 + top_90)
    top_3 = [k for k, v in top_counter.items() if v == 3]
    top_2 = [k for k, v in top_counter.items() if v == 2 and k not in top_3]
    combined_top = top_3 + top_2
    summary_top = combined_top[:5]  # ⬅️ Cap at 5

    if summary_top:
        print(f"\n🏆 Relative Strength Across Timeframes: {', '.join(summary_top)}")
    else:
        print("\n🏅 No tokens showed strength in 2 or more timeframes.")

    # Relative Weakness Summary
    bot_counter = Counter(bot_7 + bot_30 + bot_90)
    bot_3 = [k for k, v in bot_counter.items() if v == 3]
    bot_2 = [k for k, v in bot_counter.items() if v == 2 and k not in bot_3]
    combined_bot = bot_3 + bot_2
    summary_bot = combined_bot[:5]  # ⬅️ Cap at 5

    if summary_bot:
        print(f"🔻 Relative Weakness Across Timeframes: {', '.join(summary_bot)}")
    else:
        print("🔸 No tokens showed weakness in 2 or more timeframes.")


async def main():
    async with aiohttp.ClientSession() as session:
        print("Fetching top 100 tokens...")
        tokens = await get_top_100_tokens(session)

        print("Fetching 7d performance...")
        df_7 = await get_relative_performance(session, tokens, 7)

        print("Fetching 30d performance...")
        df_30 = await get_relative_performance(session, tokens, 30)

        print("Fetching 90d performance...")
        df_90 = await get_relative_performance(session, tokens, 90)

        print_leaderboard(df_7, df_30, df_90)


if __name__ == "__main__":
    asyncio.run(main())
