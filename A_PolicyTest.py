from C_XGboost_test import *
from GetkeyReal import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from B_ConnectBasic import load_data_from_pickle, get_historical_data
import pandas as pd
from binance.client import Client
# 数据清理和准备函数


def clean_and_prepare_data(df):
    """
    清理和准备数据，将 'open', 'high', 'low', 'close', 'volume' 列转换为 float 类型，并处理异常值
    :param df: 含有市场数据的 DataFrame
    """
    # 确保这些列都是 float 类型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 将无法转换的值设为 NaN

    # 删除包含 NaN 的行（或者可以选择用前一个值填充：df.fillna(method='ffill', inplace=True)）
    df.dropna(subset=['open', 'high', 'low', 'close', 'volume'], inplace=True)

    # 确保 'timestamp' 列是时间格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 确保数据按时间排序
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# 函数一：根据特定时间点决定是否开多仓或空仓


# def determine_position(current_time, df, window=20):
#     """
#     使用Ichimoku和DMI/ADX指标综合决定是否开多仓、空仓或无操作
#     :param current_time: 当前时间点 (pd.Timestamp)
#     :param df: 市场数据的 DataFrame，已按时间排序
#     :param window: 窗口期，用于计算之前的数据窗口
#     :return: 'long', 'short', 或 'none'
#     """
#     df_window = df[df['timestamp'] < current_time].tail(window).copy()

#     if len(df_window) < window:
#         return 'none'  # 数据不足，无法判断

#     df.loc[:, 'median_price'] = (df['open'] + df['close']) / 2

#     # 使用 .loc[] 避免 SettingWithCopyWarning
#     df.loc[:, 'conversion_line'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
#     df.loc[:, 'base_line'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
#     df.loc[:, 'leading_span_a'] = ((df['conversion_line'] + df['base_line']) / 2).shift(26)
#     df.loc[:, 'leading_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
#     # df.loc[:, 'lagging_span'] = df['median_price'].shift(0)

#     # 计算DMI和ADX指标
#     dmi = ta.adx(df['high'], df['low'], df['median_price'], length=14)
#     df.loc[:, 'adx'] = dmi['ADX_14']
#     df.loc[:, 'dmp'] = dmi['DMP_14']  # 正向方向线
#     df.loc[:, 'dmn'] = dmi['DMN_14']  # 负向方向线

#     # 获取当前指标的数值
#     current_adx = df.loc[df['timestamp'] == current_time, 'adx'].values[0]
#     current_dmp = df.loc[df['timestamp'] == current_time, 'dmp'].values[0]
#     current_dmn = df.loc[df['timestamp'] == current_time, 'dmn'].values[0]
#     current_conversion = df.loc[df['timestamp'] == current_time, 'conversion_line'].values[0]
#     current_base = df.loc[df['timestamp'] == current_time, 'base_line'].values[0]
#     leading_span_a = df.loc[df['timestamp'] == current_time, 'leading_span_a'].values[0]
#     leading_span_b = df.loc[df['timestamp'] == current_time, 'leading_span_b'].values[0]

#     # 判断Ichimoku和DMI/ADX的信号
#     if current_conversion > current_base and current_adx > 25 and current_dmp > current_dmn and leading_span_a > leading_span_b:
#         return 'long'  # 看涨信号，开多仓
#     elif current_conversion < current_base and current_adx > 25 and current_dmp < current_dmn and leading_span_a < leading_span_b:
#         return 'short'  # 看跌信号，开空仓

#     return 'none'  # 无强烈信号
#     # return 'long'  # 无强烈信号

def determine_position(current_time, df, window=20):
    """
    使用Ichimoku和DMI/ADX指标综合决定是否开多仓、空仓或无操作
    :param current_time: 当前时间点 (pd.Timestamp)
    :param df: 市场数据的 DataFrame，已按时间排序
    :param window: 窗口期，用于计算之前的数据窗口
    :return: 'long', 'short', 或 'none'
    """
    df_window = df[df['timestamp'] < current_time].tail(window).copy()

    if len(df_window) < window:
        return 'none'  # 数据不足，无法判断

    y = get_position(df_window)

    if y[-1] == 0:
        return 'short'   # 无强烈信号
    elif y[-1] == 1:
        return 'long'

    return 'none'
# 函数二：生成开仓信号


def generate_open_signals(df, interval='30min', window=20):
    """
    生成开仓信号的时间序列
    """
    open_signals = []
    positions = []

    # 删除重复的时间戳，确保索引唯一
    df = df.drop_duplicates(subset='timestamp')

    # 将timestamp设置为索引
    df.set_index('timestamp', inplace=True, drop=False)

    # 重新采样数据，根据指定的间隔
    df_resampled = df.resample(interval, on='timestamp').last().dropna()

    for time, row in df_resampled.iterrows():
        position = determine_position(time, df, window=window)
        if position in ['long', 'short']:
            idx = df.index.get_indexer([time], method='nearest')[0]
            open_signals.append(idx)
            positions.append(position)

    df.reset_index(drop=True, inplace=True)
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


# def calculate_contract_returns(open_signals, close_signals, df, leverage, margin, contract_size, positions, target_return_rate=0.5, stop_loss_rate=0.2, window=20, interval='30min'):
#     """
#     根据开仓和平仓信号计算整个收益曲线，考虑手续费、目标收益和止损
#     每笔交易都有独立的保证金
#     :param open_signals: 开仓信号，包含开仓的索引
#     :param close_signals: 平仓信号，包含平仓的索引
#     :param df: 市场数据的 DataFrame
#     :param leverage: 杠杆倍数
#     :param margin: 每笔交易的保证金
#     :param contract_size: 合约开单量
#     :param positions: 开仓时的仓位类型列表
#     :return: 收益曲线, 交易收益信息（开仓索引, 平仓索引, 每笔交易收益）
#     """
#     total_value = 0  # 初始总资产
#     returns = []  # 收益列表，跟踪整体账户的累积收益
#     # trade_returns = []  # 保存每次交易的开平仓和收益

#     # # 调用 generate_signals 函数来生成信号并计算每笔交易的收益
#     open_signals, close_signals, positions, trade_returns = generate_signals(
#         df, leverage=leverage, margin=margin, contract_size=contract_size, target_return_rate=target_return_rate, stop_loss_rate=stop_loss_rate, window=window, interval=interval)

#     # 累加每次交易的收益
#     for _, _, net_profit, position in trade_returns:
#         total_value += net_profit
#         returns.append(total_value)
#         # print(f"仓位类型: {position}, 当前账户总收益: {total_value:.2f}")

#     return returns, trade_returns  # 返回收益曲线和每笔交易的收益信息


def calculate_contract_returns2(trade_returns):
    """
    根据开仓和平仓信号计算整个收益曲线，考虑手续费、目标收益和止损
    每笔交易都有独立的保证金
    :param open_signals: 开仓信号，包含开仓的索引
    :param close_signals: 平仓信号，包含平仓的索引
    :param df: 市场数据的 DataFrame
    :param leverage: 杠杆倍数
    :param margin: 每笔交易的保证金
    :param contract_size: 合约开单量
    :param positions: 开仓时的仓位类型列表
    :return: 收益曲线, 交易收益信息（开仓索引, 平仓索引, 每笔交易收益）
    """
    total_value = 0  # 初始总资产
    returns = []  # 收益列表，跟踪整体账户的累积收益
    # trade_returns = []  # 保存每次交易的开平仓和收益

    # # 调用 generate_signals 函数来生成信号并计算每笔交易的收益
    # open_signals, close_signals, positions, trade_returns = generate_signals(
    #     df, leverage=leverage, margin=margin, contract_size=contract_size, target_return_rate=target_return_rate, stop_loss_rate=stop_loss_rate, window=window, interval=interval)

    # 累加每次交易的收益
    for _, _, net_profit, position in trade_returns:
        total_value += net_profit
        returns.append(total_value)
        # print(f"仓位类型: {position}, 当前账户总收益: {total_value:.2f}")

    return returns, trade_returns  # 返回收益曲线和每笔交易的收益信息


def plot_candlestick_with_signals_and_returns(df, trade_returns, returns, title="BTC/USDT Trading Duration and Returns"):
    """
    使用 Plotly 绘制交易时间条形图，并在图中标注每笔交易的持续时间，同时在子图中绘制每笔交易的收益柱状图，
    添加环状图和累积收益图。使用颜色区分收益和亏损，使用透明度区分多仓和空仓，并添加 legend。
    :param df: 市场数据的 DataFrame，包含 'timestamp', 'open', 'high', 'low', 'close' 列
    :param trade_returns: 每笔交易的收益列表，格式为 (open_idx, close_idx, net_profit, position)
    :param returns: 整体收益曲线
    :param title: 图表标题
    """
    # 确保 'timestamp' 是时间格式
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 创建带子图的 Plotly 图表，4行：交易时间条形图，收益柱状图，两个环状图，累积收益图
    fig = make_subplots(rows=4, cols=3, shared_xaxes=False,
                        vertical_spacing=0.05,
                        row_heights=[0.5, 0.2, 0.15, 0.15],
                        specs=[[{"type": "xy", "colspan": 3}, None, None],
                               [{"type": "xy", "colspan": 3}, None, None],
                               [{"type": "domain"}, {"type": "domain"}, {"type": "domain"}],
                               [{"type": "xy", "colspan": 3}, None, None]],
                        subplot_titles=(title, "Per Trade Profit/Loss", "All Position Win/Loss", "Long Position Win/Loss", "Short Position Win/Loss", "Cumulative Profit/Loss"))

    # 准备每次交易的开仓和平仓时间
    trade_times = [df['timestamp'].iloc[open_idx] for open_idx, _, _, _ in trade_returns]
    trade_durations = [(df['timestamp'].iloc[close_idx] - df['timestamp'].iloc[open_idx]).total_seconds() /
                       60 for open_idx, close_idx, _, _ in trade_returns]  # 持仓时间（分钟）

    # 使用不同颜色区分多仓和空仓的持仓时间
    bar_colors = ['blue' if position == 'long' else 'orange' for _, _, _, position in trade_returns]

    # 添加持仓时间条形图到第一个子图
    fig.add_trace(go.Bar(x=trade_times, y=trade_durations,
                         name='Trade Duration (Minutes)',
                         marker=dict(color=bar_colors)),
                  row=1, col=1)

    # 准备绘制每个开仓的收益柱状图
    trade_profits = [net_profit for _, _, net_profit, _ in trade_returns]
    trade_styles = ['long' if position == 'long' else 'short' for _, _, _, position in trade_returns]

    # 使用不同颜色区分盈利和亏损，使用透明度区分多仓和空仓
    bar_colors = ['green' if profit > 0 else 'red' for profit in trade_profits]
    bar_styles = [1.0 if position == 'long' else 0.5 for position in trade_styles]  # 多仓高不透明度，空仓低不透明度

    # 添加收益柱状图到第二个子图，区分多仓和空仓
    fig.add_trace(go.Bar(x=trade_times, y=trade_profits,
                         name='Trade Profit/Loss',
                         marker=dict(color=bar_colors, opacity=bar_styles)),  # 使用透明度区分多仓和空仓
                  row=2, col=1)

    # 计算多仓和空仓的盈利和亏损次数
    long_wins = sum(1 for profit, pos in zip(trade_profits, trade_styles) if profit > 0 and pos == 'long')
    long_losses = sum(1 for profit, pos in zip(trade_profits, trade_styles) if profit <= 0 and pos == 'long')
    short_wins = sum(1 for profit, pos in zip(trade_profits, trade_styles) if profit > 0 and pos == 'short')
    short_losses = sum(1 for profit, pos in zip(trade_profits, trade_styles) if profit <= 0 and pos == 'short')

    long_opacity = 1.0
    short_opacity = 0.5
    # 添加所有的环状图到第三行第一列
    fig.add_trace(go.Pie(labels=['All Wins', 'All Losses'],
                         values=[long_wins+short_wins, long_losses+short_losses],
                         hole=0.5,
                         marker=dict(colors=['green', 'red']),
                         opacity=1.0,
                         textinfo='label+percent',
                         showlegend=True,  # 显示 legend
                         name="All Positions"),
                  row=3, col=1)

    # 添加多仓的环状图到第三行第2列
    fig.add_trace(go.Pie(labels=['Long Wins', 'Long Losses'],
                         values=[long_wins, long_losses],
                         hole=0.5,
                         marker=dict(colors=['green', 'red']),
                         opacity=long_opacity,
                         textinfo='label+percent',
                         showlegend=True,  # 显示 legend
                         name="Long Positions"),
                  row=3, col=2)

    # 添加空仓的环状图到第三行第3列
    fig.add_trace(go.Pie(labels=['Short Wins', 'Short Losses'],
                         values=[short_wins, short_losses],
                         hole=0.5,
                         marker=dict(colors=['green', 'red']),
                         opacity=short_opacity,
                         textinfo='label+percent',
                         showlegend=True,  # 显示 legend
                         name="Short Positions"),
                  row=3, col=3)

    # 计算累积收益
    cumulative_returns = [sum(trade_profits[:i+1]) for i in range(len(trade_profits))]

    # 添加累积收益曲线到第四个子图
    fig.add_trace(go.Scatter(x=trade_times, y=cumulative_returns,
                             mode='lines', name='Cumulative Profit/Loss',
                             line=dict(color='blue', width=2)),
                  row=4, col=1)

    # 更新布局并添加 legend
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=False)),
        yaxis_title='Duration (Minutes)',
        yaxis2_title='Profit/Loss (USDT)',
        height=1200,
        hovermode="x unified",
        showlegend=True,  # 打开 legend
        legend=dict(x=1.05, y=1)  # 设置 legend 位置
    )

    # 显示图表
    fig.show()


# 主程序
if __name__ == "__main__":
    # 加载数据并清理
    # SYMBOL = 'BTC'
    SYMBOL = 'DOGE_new'
    getnewdata = 0
    if getnewdata == 0:
        df = load_data_from_pickle(f'market_data_{SYMBOL}.pkl')
    elif getnewdata == 1:
        start_time = pd.Timestamp('2024-10-10 08:55:00')
        client = Client(api_key, api_secret,  requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})
        df = get_historical_data(client, symbol=f'{SYMBOL}USDT', interval=Client.KLINE_INTERVAL_1MINUTE, start_time=start_time)

    # df = clean_and_prepare_data(df)
    y = get_position(df)
    print(y)

    # 计算整个策略的收益和交易信息
    # leverage = 100  # 100倍杠杆
    # target_return_rate = 0.8
    # stop_loss_rate = 0.7
    leverage = 50  # 100倍杠杆
    target_return_rate = 0.38
    stop_loss_rate = 0.28
    interval = '5min'
    window = 200

    margin = 1  # 保证金总量
    contract_size = 1  # 合约开单量（根据需要调整）

    open_signals, close_signals, positions, trade_returns = generate_signals(df, interval, window, leverage, margin, contract_size,
                                                                             target_return_rate, stop_loss_rate)

    # returns, trade_returns = calculate_contract_returns(open_signals, close_signals, df, leverage,
    #                                                     margin, contract_size, positions, target_return_rate, stop_loss_rate, window, interval='5min')
    returns, trade_returns = calculate_contract_returns2(trade_returns)

    # 使用Plotly绘制K线图，标注信号和收益，并绘制收益率曲线
    plot_candlestick_with_signals_and_returns(df, trade_returns, returns, title=f"{SYMBOL}/USDT Candlestick Chart with Signals and Returns")
    # 计算累积出手次数
    cumulative_trades = len(trade_returns)

    # 计算每次交易的持仓时间，并取平均值
    trade_durations = [(df['timestamp'].iloc[close_idx] - df['timestamp'].iloc[open_idx]).total_seconds() / 60
                       for open_idx, close_idx, _, _ in trade_returns]
    average_trade_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0

    # 输出累积出手次数和平均出手时间
    print(f'累积出手次数：{cumulative_trades} | 平均出手时间：{average_trade_duration:.2f} 分钟')
