from Getkey import *
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from B_ConnectBasic import load_data_from_pickle, get_historical_data, save_data_to_pickle
import pandas as pd
from binance.client import Client
from A_PolicyTest import clean_and_prepare_data, calculate_contract_returns2, plot_candlestick_with_signals_and_returns


def got_real_position_target(df):
    """
    Simulate positions based on leverage, take-profit, and stop-loss.
    For each time point, simulate a long or short position and check if it hits
    take-profit or stop-loss. Return 'long', 'short', or 'none' for each timestamp.
    """
    leverage = 50
    target_return_rate = 0.20  # 50% target profit
    stop_loss_rate = 0.80  # 40% stop loss

    position_signals = ['none'] * len(df)  # Initialize the entire column with 'none'

    for i in range(len(df) - 1):  # Exclude the last row since we need future data to simulate
        open_price = df.loc[i, 'close']

        # Calculate target prices for long and short positions
        long_target_price = open_price * (1 + target_return_rate / leverage)
        long_stop_loss_price = open_price * (1 - stop_loss_rate / leverage)

        short_target_price = open_price * (1 - target_return_rate / leverage)
        short_stop_loss_price = open_price * (1 + stop_loss_rate / leverage)

        # Simulate the movement after opening a position
        for j in range(i + 1, len(df)):
            current_high = df.loc[j, 'high']
            current_low = df.loc[j, 'low']

            # Simulate long position
            if current_low <= long_stop_loss_price:
                position_signals[i] = 'short'  # Stop-loss hit on a long position (so record as short signal)
                break
            elif current_high >= long_target_price:
                position_signals[i] = 'long'  # Take-profit hit on a long position (so record as long signal)
                break

            # Simulate short position
            if current_high >= short_stop_loss_price:
                position_signals[i] = 'long'  # Stop-loss hit on a short position (so record as long signal)
                break
            elif current_low <= short_target_price:
                position_signals[i] = 'short'  # Take-profit hit on a short position (so record as short signal)
                break

    # Add the position signals to the DataFrame
    df['position_signal'] = position_signals

    return df


def generate_open_signals(df, interval='30min', window=20):
    """
    Generates open signals based on the got_real_position_target function.
    """
    open_signals = []
    positions = []

    for i, row in df.iterrows():
        if row['position_signal'] in ['long', 'short']:
            open_signals.append(i)
            positions.append(row['position_signal'])

    return open_signals, positions


# 函数三：生成开仓和平仓信号


def generate_signals(df, interval='30min', window=20, leverage=100, margin=10, contract_size=1,
                     target_return_rate=0.5, stop_loss_rate=0.2, fee_rate=0.0005):
    """
    生成所有的开仓和平仓信号，考虑手续费、目标收益、止损和强制平仓的情况
    """
    open_signals, positions = generate_open_signals(df, interval=interval, window=window)
    close_signals = []
    trade_returns = []
    active_positions = []
    df['median_price'] = (df['open'] + df['close']) / 2
    realsum = margin*2

    for i, open_idx in enumerate(open_signals):
        position = positions[i]
        open_price = df.loc[open_idx, 'median_price']
        margin = realsum / 2
        position_size = margin * leverage  # 头寸规模

        # 计算目标价格和止损价格
        if position == 'long':
            target_price = open_price + (open_price * target_return_rate / leverage)
            stop_loss_price = open_price - (open_price * stop_loss_rate / leverage)
        elif position == 'short':
            target_price = open_price - (open_price * target_return_rate / leverage)
            stop_loss_price = open_price + (open_price * stop_loss_rate / leverage)
        else:
            continue

        fee_open = position_size * fee_rate  # 开仓手续费
        close_idx = None
        net_profit = 0

        for j in range(open_idx + 1, len(df)):
            current_high = df.loc[j, 'high']
            current_low = df.loc[j, 'low']

            if position == 'long':
                if current_low <= stop_loss_price:
                    close_price = stop_loss_price
                    close_idx = j
                    profit = (close_price - open_price) * position_size / open_price
                    fee_close = close_price * position_size / open_price * fee_rate
                    net_profit = profit - fee_open - fee_close
                    net_profit = max(net_profit, -margin)
                    print(f"多仓止损 -  时间: {df.loc[j, 'timestamp']}, 价格: {close_price}, 净收益: {net_profit:.2f}")
                    break
                elif current_high >= target_price:
                    close_price = target_price
                    close_idx = j
                    profit = (close_price - open_price) * position_size / open_price
                    fee_close = close_price * position_size / open_price * fee_rate
                    net_profit = profit - fee_open - fee_close
                    print(f"多仓目标达成 - 时间: {df.loc[j, 'timestamp']}, 价格: {close_price}, 净收益: {net_profit:.2f}")
                    break
            elif position == 'short':
                if current_high >= stop_loss_price:
                    close_price = stop_loss_price
                    close_idx = j
                    profit = (open_price - close_price) * position_size / open_price
                    fee_close = close_price * position_size / open_price * fee_rate
                    net_profit = profit - fee_open - fee_close
                    net_profit = max(net_profit, -margin)
                    print(f"空仓止损 - 时间: {df.loc[j, 'timestamp']}, 价格: {close_price}, 净收益: {net_profit:.2f}")
                    break
                elif current_low <= target_price:
                    close_price = target_price
                    close_idx = j
                    profit = (open_price - close_price) * position_size / open_price
                    fee_close = close_price * position_size / open_price * fee_rate
                    net_profit = profit - fee_open - fee_close
                    print(f"空仓目标达成 - 时间: {df.loc[j, 'timestamp']}, 价格: {close_price}, 净收益: {net_profit:.2f}")
                    break

        if close_idx is None:
            # 平在最后一个价格
            close_idx = len(df) - 1
            close_price = df.loc[close_idx, 'median_price']
            if position == 'long':
                profit = (close_price - open_price) * position_size / open_price
            elif position == 'short':
                profit = (open_price - close_price) * position_size / open_price
            fee_close = close_price * position_size / open_price * fee_rate
            net_profit = profit - fee_open - fee_close
            net_profit = max(net_profit, -margin)
            print(f"未找到平仓信号，平在最后 - 时间: {df.loc[close_idx, 'timestamp']}, 价格: {close_price}, 净收益: {net_profit:.2f}")

        close_signals.append(close_idx)
        trade_returns.append((open_idx, close_idx, net_profit, position))
        active_positions.append(position)
        # realsum += net_profit

    return open_signals, close_signals, active_positions, trade_returns


# 主程序
if __name__ == "__main__":
    # 加载数据并清理
    # SYMBOL = 'BTC'
    SYMBOL = 'DOGE'
    getnewdata = 0
    if getnewdata == 0:
        df = load_data_from_pickle(f'market_data_{SYMBOL}.pkl')
        # df = load_data_from_pickle(f'market_data_{SYMBOL}_with_position.pkl')
    elif getnewdata == 1:
        start_time = pd.Timestamp('2024-09-10 00:00:00')
        client = Client(api_key, api_secret, testnet=True)
        df = get_historical_data(client, symbol=f'{SYMBOL}USTD', interval=Client.KLINE_INTERVAL_1MINUTE, start_time=start_time)

    df = clean_and_prepare_data(df)
    df = got_real_position_target(df)
    save_data_to_pickle(df, filename=f"market_data_{SYMBOL}_with_position.pkl")

    # 计算整个策略的收益和交易信息
    # leverage = 100  # 100倍杠杆
    # target_return_rate = 0.8
    # stop_loss_rate = 0.7
    leverage = 50  # 100倍杠杆
    target_return_rate = 0.2
    stop_loss_rate = 0.8
    interval = '15min'
    window = 20

    margin = 1  # 保证金总量
    contract_size = 1  # 合约开单量（根据需要调整）

    open_signals, close_signals, positions, trade_returns = generate_signals(df, interval, window, leverage, margin, contract_size,
                                                                             target_return_rate, stop_loss_rate)
    returns, trade_returns = calculate_contract_returns2(trade_returns)

    # 使用Plotly绘制K线图，标注信号和收益，并绘制收益率曲线
    plot_candlestick_with_signals_and_returns(df, trade_returns, returns, title=f"{SYMBOL}/USDT Candlestick Chart with Signals and Returns")
    # 计算累积出手次数
    cumulative_trades = len(trade_returns)

    # 计算每次交易的持仓时间，并取平均值
    trade_durations = [(df['timestamp'].iloc[close_idx] - df['timestamp'].iloc[open_idx]).total_seconds() / 60
                       for open_idx, close_idx, _, _ in trade_returns]
    average_trade_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0
