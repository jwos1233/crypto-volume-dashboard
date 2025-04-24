# Crypto Volume Analysis Dashboard

A Streamlit dashboard for Crypto flow analysis.

## Features

- Real-time volume analysis for top cryptocurrencies
- Three main views:
  - Volume Spikes: Identifies unusual volume activity
  - Liquidity: Shows volume/market cap ratios
  - Acceleration: Tracks volume acceleration trends
- Interactive visualizations with Plotly
- Customizable filters for market cap and volume

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the dashboard:
   ```bash
   streamlit run volume_dashboard.py
   ```

## API Key

This dashboard uses the CoinGecko API. You'll need to:
1. Get an API key from [CoinGecko](https://www.coingecko.com/en/api)
2. Replace the `API_KEY` in `volume_dashboard.py` with your key

## Deployment

This dashboard can be deployed on Streamlit Cloud or Hugging Face Spaces for 24/7 access.

## License

MIT License 