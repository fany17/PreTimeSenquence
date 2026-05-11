from binance import Client
import os
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from binance.client import Client
# from Getkey import api_key, api_secret  # 确保从Getkey模块正确导入API key和secret
from GetkeyReal import api_key, api_secret  # 确保从Getkey模块正确导入API key和secret
import time
# 初始化客户端


def test_time_difference(client):
    local_timestamp = int(time.time() * 1000)
    server_time = client.get_server_time()
    server_timestamp = server_time['serverTime']
    time_difference = server_timestamp - local_timestamp
    print(f"Local timestamp: {local_timestamp} ms")
    print(f"Server timestamp: {server_timestamp} ms")
    print(f"Time difference: {time_difference} ms")


def get_market_data(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, limit=100):
    candlesticks = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(candlesticks, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                             'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['close'] = df['close'].astype(float)
    return df


def get_historical_data(client, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, start_time=None, limit=1000):
    """
    从指定的 start_time 获取历史数据直到最近时间。
    :param client: Binance API 客户端
    :param symbol: 交易对 (默认 'BTCUSDT')
    :param interval: K线的时间间隔 (默认 Client.KLINE_INTERVAL_1HOUR)
    :param start_time: 数据开始时间 (例如 '2024-09-01 00:00:00')
    :param limit: 每次请求的数据量上限 (默认 1000)
    :return: DataFrame, 包含从 start_time 到当前时间的数据
    """
    all_data = []
    end_time = None

    if start_time is not None:
        start_time = int(pd.Timestamp(start_time).timestamp() * 1000)  # 转换为毫秒

    while True:
        # 从 start_time 开始获取数据
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit, startTime=start_time)

        # 如果获取到的数据为空，说明已经获取完所有数据
        if not klines:
            print("所有数据已获取完毕。")
            break

        # 将数据转换为 DataFrame
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
                                           'taker_buy_quote_asset_volume', 'ignore'])

        # 转换时间戳格式
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = df['close'].astype(float)

        # 输出获取到的数据信息
        print(f"获取到 {len(df)} 条数据，时间范围: {df['timestamp'].iloc[0]} - {df['timestamp'].iloc[-1]}")

        # 将数据添加到总数据中
        all_data.append(df)

        # 更新 start_time 为最新的时间戳，以获取下一批数据
        start_time = int(df['timestamp'].iloc[-1].timestamp() * 1000)

        # 如果获取的数据少于 limit，说明已经到达当前时间或最早时间
        if len(df) < limit:
            print("获取到的数据少于最大限制，已到达最早时间或最近时间。")
            break

        # 每次请求之间稍作等待，避免频繁请求触发 API 频率限制
        time.sleep(1)

    # 合并所有数据并返回
    all_data = pd.concat(all_data)
    all_data.sort_values(by='timestamp', inplace=True)  # 确保按时间排序
    return all_data


def save_data_to_csv(df, filename="market_data.csv"):
    df.to_csv(filename, index=False)
    print(f"数据已保存到 {filename}")


def save_data_to_pickle(df, filename="market_data.pkl"):
    df.to_pickle(filename)
    print(f"数据已保存到 {filename}")


def load_data_from_csv(filename="market_data.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # 确保时间格式正确
        print(f"数据从 {filename} 读取")
        return df
    else:
        print(f"{filename} 文件不存在。")
        return None


def load_data_from_pickle(filename="market_data.pkl"):
    if os.path.exists(filename):
        df = pd.read_pickle(filename)
        print(f"数据从 {filename} 读取")
        return df
    else:
        print(f"{filename} 文件不存在。")
        return None

# 选择数据来源：1-API，2-CSV，3-Pickle


def get_data(source=1, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, start_time=None, filename="market_data.csv"):
    if source == 1:
        # 从API获取数据
        df = get_historical_data(symbol=symbol, interval=interval, start_time=start_time)
        return df
    elif source == 2:
        # 从CSV文件读取数据
        return load_data_from_csv(filename=filename)
    elif source == 3:
        # 从Pickle文件读取数据
        return load_data_from_pickle(filename=filename)
    else:
        print("无效的数据源选项")
        return None


def calculate_rsi(df, periods=14):
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=periods, min_periods=1).mean()
    avg_loss = loss.rolling(window=periods, min_periods=1).mean()

    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

# 基于RSI生成买卖信号


def generate_rsi_signals(df):
    df['signal'] = 0
    df['signal'] = np.where(df['rsi'] < 30, 1, 0)  # Buy signal when RSI < 30
    df['signal'] = np.where(df['rsi'] > 70, -1, df['signal'])  # Sell signal when RSI > 70
    df['position'] = df['signal'].diff()  # Capture buy/sell changes
    return df

# 更新的计算收益函数，记录实际买卖操作


# 更新的计算收益函数，记录实际买卖操作
def calculate_rsi_strategy_returns(df, start_time=None, initial_capital=10, trade_size=10, fee_rate=0.0005):
    # 计算 RSI
    df = calculate_rsi(df)
    # 生成交易信号
    df = generate_rsi_signals(df)

    # 确保 'timestamp' 列为 datetime 类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 如果指定了 start_time，则过滤数据
    if start_time is not None:
        if not isinstance(start_time, pd.Timestamp):
            start_time = pd.Timestamp(start_time)
        df = df[df['timestamp'] >= start_time].copy().reset_index(drop=True)
    else:
        df = df.copy().reset_index(drop=True)

    # 初始化变量
    position = 0  # 持仓状态：0 表示空仓，1 表示持仓
    cash = initial_capital  # 现金余额
    holdings = 0.0  # 持有的 BTC 数量
    total_values = []  # 总资产价值列表

    # 初始化买入卖出信号列
    df['buy_signals'] = np.nan
    df['sell_signals'] = np.nan

    for i in range(len(df)):
        signal = df.loc[i, 'signal']
        price = df.loc[i, 'close']

        # 买入条件：信号为 1，且当前为空仓，且现金足够
        if signal == 1 and position == 0 and cash >= trade_size:
            # 执行买入
            amount = (trade_size / price) * (1 - fee_rate)  # 购买的 BTC 数量（扣除手续费）
            holdings += amount
            cash -= trade_size  # 扣除交易金额
            position = 1  # 更新持仓状态为持仓
            df.at[i, 'buy_signals'] = price  # 记录买入信号
            print(f"Buying at {df.loc[i, 'timestamp']}, price: {price}, holdings: {holdings} BTC, cash: {cash}")

        # 卖出条件：信号为 -1，且当前为持仓状态
        elif signal == -1 and position == 1:
            # 执行卖出
            proceeds = holdings * price * (1 - fee_rate)  # 卖出所得（扣除手续费）
            cash += proceeds
            print(f"Selling at {df.loc[i, 'timestamp']}, price: {price}, proceeds: {proceeds}, cash: {cash}")
            holdings = 0.0
            position = 0  # 更新持仓状态为空仓
            df.at[i, 'sell_signals'] = price  # 记录卖出信号

        # 计算当前的总资产价值
        total_value = cash + holdings * price
        total_values.append(total_value)

    # 将总资产价值添加到 DataFrame
    df['strategy_returns'] = total_values

    return df


def plot_trading_strategy_and_returns(df):
    plt.figure(figsize=(14, 8))

    # 确保 'timestamp' 列为 datetime 类型
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 绘制价格和收益曲线
    ax1 = plt.subplot(211)
    ax1.plot(df['timestamp'], df['close'], label='Close Price', color='blue', lw=2)

    # 绘制买入卖出信号
    # 买入信号
    buy_signals = df[df['buy_signals'].notnull()]
    ax1.plot(buy_signals['timestamp'], buy_signals['buy_signals'], '^', markersize=10, color='green', label='Buy Signal')

    # 卖出信号
    sell_signals = df[df['sell_signals'].notnull()]
    ax1.plot(sell_signals['timestamp'], sell_signals['sell_signals'], 'v', markersize=10, color='red', label='Sell Signal')

    ax1.set_title('BTC/USDT Trading Strategy (Executed Trades)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price (USDT)')
    ax1.legend()

    # 绘制累计收益曲线
    ax2 = plt.subplot(212)
    ax2.plot(df['timestamp'], df['strategy_returns'], label='Strategy Returns', color='purple', lw=2)
    ax2.set_title('Strategy Cumulative Returns')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Total Value (USDT)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # client = Client(api_key, api_secret, testnet=True, requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})
    client = Client(api_key, api_secret, requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})
    test_time_difference(client)
    start_time = pd.Timestamp('2024-10-15 00:00:00')
    df = get_historical_data(client, symbol='DOGEUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, start_time=start_time)
    save_data_to_pickle(df, filename="market_data_DOGE_new.pkl")
    # df = get_historical_data(client, symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1MINUTE, start_time=start_time)
    # save_data_to_pickle(df, filename="market_data_BTC.pkl")

# df_with_returns = calculate_rsi_strategy_returns(df, initial_capital=10, trade_size=5, start_time=start_time, fee_rate=0.001)
# plot_trading_strategy_and_returns(df_with_returns)
