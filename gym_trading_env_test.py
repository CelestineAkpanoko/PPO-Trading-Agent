# from gym_trading_env.downloader import download
# import datetime
# import pandas as pd

# # Download BTC/USDT historical data from Binance and stores it to directory ./data/binance-BTCUSDT-1h.pkl

# download(
#     exchange_names = ["binance", "bitfinex2", "huobi"],
#     symbols= ["BTC/USDT", "ETH/USDT"],
#     timeframe= "1h",
#     dir = "data",
#     since= datetime.datetime(year= 2019, month= 1, day=1),
#     until = datetime.datetime(year= 2023, month= 1, day=1),
# )
# # Import your fresh data
# # df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")