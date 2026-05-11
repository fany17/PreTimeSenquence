import time
import pandas as pd
from binance.client import Client
from A_PolicyTest import determine_position  # 复用之前的策略函数
import logging

# Binance 实盘 API Key 和 Secret
# from Getkey import api_key, api_secret  # 确保从Getkey模块正确导入API key和secret
from GetkeyReal import api_key, api_secret  # 确保从Getkey模块正确导入API key和secret

# 实盘测试环境设置
# SYMBOL = 'BTCUSDT'
SYMBOL = 'DOGEUSDT'
TRADE_AMOUNT_USD = 0.12  # 交易金额（美元），根据账户余额调整
leverage = 50  # 设置杠杆倍数
INTERVAL = Client.KLINE_INTERVAL_1MINUTE  # 交易频率
WINDOW = 20  # 窗口大小，用于信号判断
TARGET_RETURN_RATE = 0.38  # 目标收益率 5%
STOP_LOSS_RATE = 0.28  # 止损率 2%
FEE_RATE = 0.0005  # 手续费率

# 日志记录
logging.basicConfig(
    filename='real_time_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


def adjust_trade_amount(client, symbol, trade_amount_usd, current_price):
    """
    根据交易金额（美元）、当前价格调整交易量，确保符合交易对最小交易量和步长要求。
    """
    # 计算交易数量
    position_size = trade_amount_usd / current_price

    # 获取交易对的最小交易量和步长
    exchange_info = client.futures_exchange_info()
    symbol_info = next(item for item in exchange_info['symbols'] if item['symbol'] == symbol)
    filters = symbol_info['filters']
    lot_size = next(f for f in filters if f['filterType'] == 'LOT_SIZE')
    min_qty = float(lot_size['minQty'])
    step_size = float(lot_size['stepSize'])

    print(f'最小交易量 {min_qty} 和 最小步长 {step_size}，计划交易量 {position_size}')

    # 确保交易数量不小于最小交易量，并且是 step_size 的倍数
    if position_size < min_qty:
        position_size = min_qty
    position_size = (position_size // step_size) * step_size
    position_size = round(position_size, 8)  # 根据需要调整小数位数

    print(f'实际交易量 {position_size}')
    return position_size


def log_trade_info(message):
    """
    记录日志信息，同时将信息打印到控制台。
    """
    print(message)
    logging.info(message)


def print_trade_summary(trade_side, symbol, executed_qty, price, take_profit_price, stop_loss_price, total_fees):
    """
    简单打印关键的交易信息到控制台。
    """
    if trade_side == 'buy':
        print(f"买入: {executed_qty} {symbol} @ {price} USDT")
    elif trade_side == 'sell':
        print(f"卖出: {executed_qty} {symbol} @ {price} USDT")

    # 打印止盈和止损信息
    if take_profit_price and stop_loss_price:
        print(f"止盈: {take_profit_price:.2f} USDT | 止损: {stop_loss_price:.2f} USDT")

    # 打印手续费
    if total_fees:
        print(f"手续费: {total_fees:.6f} USDT")


def get_latest_data(client, symbol, interval, limit=100):
    """
    获取最新的市场数据，实盘使用。
    """
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
        'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
        'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df


def execute_trade(client, symbol, trade_side, margin, fee_rate, current_price, take_profit_rate=0.05, stop_loss_rate=0.02):
    """
    执行买入或卖出操作，并设置止盈和止损订单，计算头寸规模和手续费。
    """
    try:
        # 检查账户余额
        check_account_balance(client)

        # 设置交易对的杠杆
        client.futures_change_leverage(symbol=symbol, leverage=leverage, recvWindow=10000)

        # 计算头寸规模
        position_size = adjust_trade_amount(client, symbol, margin * leverage, current_price)

        # 计算开仓手续费
        fee_open = position_size * fee_rate * current_price
        print(f'开仓手续费: {fee_open:.6f} USDT')

        # 计算止盈和止损价格
        if trade_side == 'buy':
            take_profit_price = current_price + (current_price * take_profit_rate / leverage)
            stop_loss_price = current_price - (current_price * stop_loss_rate / leverage)
        elif trade_side == 'sell':
            take_profit_price = current_price - (current_price * take_profit_rate / leverage)
            stop_loss_price = current_price + (current_price * stop_loss_rate / leverage)

        # 输出止盈止损价格
        print(f'当前价格: {current_price}---止盈价格: {take_profit_price:.6f} ---止损价格: {stop_loss_price:.6f}')

        # 执行市价单
        if trade_side == 'buy':
            order = client.futures_create_order(
                symbol=symbol,
                side='BUY',
                positionSide='LONG',  # 对冲模式下明确指定仓位方向
                type='MARKET',
                quantity=position_size,
                recvWindow=10000
            )
            order_id = order['orderId']
            # 查询订单详情，确保获取执行的数量和平均价格
            order_status = client.futures_get_order(symbol=symbol, orderId=order_id,
                                                    recvWindow=10000)
            # 检查订单是否已完全执行
            if order_status['status'] == 'FILLED':
                executed_qty = float(order_status['executedQty'])
                price = float(order_status['avgPrice'])
                log_trade_info(f"多仓订单执行: {executed_qty} {symbol} @ {price} USDT")
            else:
                log_trade_info(f"订单未完全执行: {order_status['status']}")
        elif trade_side == 'sell':
            order = client.futures_create_order(
                symbol=symbol,
                side='SELL',
                positionSide='SHORT',  # 对冲模式下明确指定仓位方向
                type='MARKET',
                quantity=position_size,
                recvWindow=10000
            )  # 获取订单 ID
            order_id = order['orderId']
            # 查询订单详情，确保获取执行的数量和平均价格
            order_status = client.futures_get_order(symbol=symbol, orderId=order_id,
                                                    recvWindow=10000)
            # 检查订单是否已完全执行
            if order_status['status'] == 'FILLED':
                executed_qty = float(order_status['executedQty'])
                price = float(order_status['avgPrice'])
                log_trade_info(f"空仓订单执行: {executed_qty} {symbol} @ {price} USDT")
            else:
                log_trade_info(f"订单未完全执行: {order_status['status']}")

        # # 计算平仓手续费
        # fee_close = position_size * fee_rate * current_price
        # print(f'平仓手续费: {fee_close:.6f} USDT')

        # 设置止盈订单（TAKE_PROFIT_MARKET）
        if trade_side == 'buy':
            take_profit_order = client.futures_create_order(
                symbol=symbol,
                side='SELL',
                positionSide='LONG',  # 在多仓下卖出平仓
                type='TAKE_PROFIT_MARKET',
                quantity=position_size,
                stopPrice=round(take_profit_price, 5),
                recvWindow=10000
            )
            log_trade_info(f"止盈订单设置: TP {take_profit_price:.5f} USDT")
            # print(f'止盈订单设置成功: 止盈价格 {take_profit_price:.2f} USDT')
        elif trade_side == 'sell':
            take_profit_order = client.futures_create_order(
                symbol=symbol,
                side='BUY',
                positionSide='SHORT',  # 在空仓下买入平仓
                type='TAKE_PROFIT_MARKET',
                quantity=position_size,
                stopPrice=round(take_profit_price, 5),
                recvWindow=10000
            )
            log_trade_info(f"止盈订单设置: TP {take_profit_price:.5f} USDT")
            # print(f'止盈订单设置成功: 止盈价格 {take_profit_price:.2f} USDT')

        # 设置止损订单（STOP_MARKET）
        if trade_side == 'buy':
            stop_loss_order = client.futures_create_order(
                symbol=symbol,
                side='SELL',
                positionSide='LONG',  # 在多仓下卖出平仓
                type='STOP_MARKET',
                quantity=position_size,
                stopPrice=round(stop_loss_price, 5),
                recvWindow=10000
            )
            log_trade_info(f"止损订单设置: SL {stop_loss_price:.5f} USDT")
            # print(f'止损订单设置成功: 止损价格 {stop_loss_price:.2f} USDT')
        elif trade_side == 'sell':
            stop_loss_order = client.futures_create_order(
                symbol=symbol,
                side='BUY',
                positionSide='SHORT',  # 在空仓下买入平仓
                type='STOP_MARKET',
                quantity=position_size,
                stopPrice=round(stop_loss_price, 5),
                recvWindow=10000
            )
            log_trade_info(f"止损订单设置: SL {stop_loss_price:.5f} USDT")
            # print(f'止损订单设置成功: 止损价格 {stop_loss_price:.2f} USDT')
        # 监控止盈和止损订单状态，确保互相取消
        # monitor_and_cancel_orders(client, symbol, take_profit_order, stop_loss_order)

        # # 计算总手续费
        # total_fees = fee_open + fee_close
        # log_trade_info(f"交易总费用: {total_fees:.6f} USDT")
        # print(f'交易总费用: {total_fees:.6f} USDT')

    except Exception as e:
        log_trade_info(f"交易执行失败: {str(e)}")
        # print(f"交易执行失败: {str(e)}")


def monitor_and_cancel_orders(client, symbol, take_profit_order, stop_loss_order):
    """
    监控止盈和止损订单状态，一旦其中一个执行，取消另一个
    """
    while True:
        try:
            # 获取止盈订单状态
            take_profit_order_status = client.futures_get_order(symbol=symbol, orderId=take_profit_order['orderId'], recvWindow=10000)
            # 获取止损订单状态
            stop_loss_order_status = client.futures_get_order(symbol=symbol, orderId=stop_loss_order['orderId'], recvWindow=10000)

            # 检查止盈订单是否已执行
            if take_profit_order_status['status'] == 'FILLED':
                print("止盈订单已执行，取消止损订单...")
                client.futures_cancel_order(symbol=symbol, orderId=stop_loss_order['orderId'], recvWindow=10000)
                break  # 终止监控循环

            # 检查止损订单是否已执行
            if stop_loss_order_status['status'] == 'FILLED':
                print("止损订单已执行，取消止盈订单...")
                client.futures_cancel_order(symbol=symbol, orderId=take_profit_order['orderId'], recvWindow=10000)
                break  # 终止监控循环

        except Exception as e:
            print(f"订单监控出错: {str(e)}")

        # 每隔1秒检查一次订单状态，避免过度调用API
        time.sleep(1)


def check_account_balance(client):
    """
    检查合约账户余额是否足够。
    """
    try:
        balance_info = client.futures_account_balance(recvWindow=10000)
        usdt_balance = next((item for item in balance_info if item['asset'] == 'USDT'), None)
        if usdt_balance:
            print(f"USDT余额: {usdt_balance['balance']} USDT")
            log_trade_info(f"USDT余额: {usdt_balance['balance']} USDT")
        else:
            log_trade_info("未找到USDT余额信息。")
            print("未找到USDT余额信息。")
    except Exception as e:
        log_trade_info(f"获取账户余额失败: {str(e)}")
        print(f"获取账户余额失败: {str(e)}")


def check_position_and_trade(df, client, current_time):
    """
    检查是否有交易信号并执行交易，设置止盈和止损。
    """
    # 获取当前价格
    current_price = df.loc[df['timestamp'] == current_time, 'close'].values[0]

    # 检查交易信号
    position = determine_position(current_time, df, window=WINDOW)

    if position == 'long':

        print('############################################################')
        log_trade_info(f"{current_time}: - Price:{current_price} 生成多仓信号")

        # 执行买入并设置止盈止损订单
        execute_trade(
            client=client,
            symbol=SYMBOL,
            trade_side='buy',
            margin=TRADE_AMOUNT_USD,
            fee_rate=FEE_RATE,
            current_price=current_price,
            take_profit_rate=TARGET_RETURN_RATE,
            stop_loss_rate=STOP_LOSS_RATE
        )

    elif position == 'short':

        print('############################################################')
        log_trade_info(f"{current_time}: - Price:{current_price} 生成空仓信号")

        # 执行卖出并设置止盈止损订单
        execute_trade(
            client=client,
            symbol=SYMBOL,
            trade_side='sell',
            margin=TRADE_AMOUNT_USD,
            fee_rate=FEE_RATE,
            current_price=current_price,
            take_profit_rate=TARGET_RETURN_RATE,
            stop_loss_rate=STOP_LOSS_RATE
        )

    else:
        log_trade_info(f"{current_time}: - Price:{current_price} 无信号，保持观望")


def main():
    client = Client(api_key, api_secret,  requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})
    # client = Client(api_key, api_secret, testnet=True, requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})
    log_trade_info("开始实盘测试...")

    while True:
        try:
            # 获取最新的市场数据
            df = get_latest_data(client, SYMBOL, INTERVAL)
            # print(df)

            # 获取当前时间戳
            current_time = df['timestamp'].iloc[-1]

            # 检查并根据信号执行交易
            check_position_and_trade(df, client, current_time)

            # 每分钟检测一次，避免过频繁交易
            time.sleep(600)

        except Exception as e:
            log_trade_info(f"运行错误: {str(e)}")
            print(f"运行错误: {str(e)}")
            time.sleep(60)


if __name__ == "__main__":
    main()
