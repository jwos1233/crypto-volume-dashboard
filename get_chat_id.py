import requests

# Your bot token
BOT_TOKEN = "7675238786:AAE27scvUtT7lth02I50J_aIU3wZD-RDGuk"

# Get updates from the bot
url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
response = requests.get(url)

if response.status_code == 200:
    data = response.json()
    if data["ok"] and data["result"]:
        print("\nFound chat IDs:")
        for update in data["result"]:
            if "message" in update:
                chat_id = update["message"]["chat"]["id"]
                print(f"Chat ID: {chat_id}")
    else:
        print("\nNo messages found. Please send a message to your bot first.")
else:
    print(f"\nError getting updates: {response.text}") 