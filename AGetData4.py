import pandas as pd
from binance.client import Client
from GetkeyReal import api_key, api_secret  # Replace with your module or keys


from binance.client import Client

api_key = "your_api_key"
api_secret = "your_api_secret"


def get_latest_data():
    """
    Retrieves the latest market data and returns it as a dictionary.
    """
    SYMBOL = 'DOGEUSDT'
    INTERVAL = Client.KLINE_INTERVAL_1MINUTE

    # 初始化客户端并设置代理
    client = Client(api_key, api_secret, requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})

    # 获取 K 线数据
    klines = client.futures_klines(symbol=SYMBOL, interval=INTERVAL, limit=200)

    # 将数据分解为独立列并转换为 Python 原生字典
    data_dict = {
        'open': [float(entry[1]) for entry in klines],
        'high': [float(entry[2]) for entry in klines],
        'low': [float(entry[3]) for entry in klines],
        'close': [float(entry[4]) for entry in klines],
        'volume': [float(entry[5]) for entry in klines]
    }

    return data_dict
