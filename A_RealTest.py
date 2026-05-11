import time
import pandas as pd
from binance.client import Client
import logging
from A_PolicyTest import determine_position

# Binance API keys
# from Getkey import api_key, api_secret  # Replace with your module or keys
from GetkeyReal import api_key, api_secret  # Replace with your module or keys

# Trading parameters
SYMBOL = 'DOGEUSDT'
# SYMBOL = 'BTCUSDT'
TRADE_MARGIN = 0.12  # The margin amount in USDT for each trade
LEVERAGE = 20
INTERVAL = Client.KLINE_INTERVAL_1MINUTE
FEE_RATE = 0.0004  # Adjust based on your fee rate

# Target profit and stop loss percentages
TAKE_PROFIT_PERCENT = 0.5  # 50% profit
STOP_LOSS_PERCENT = 0.4    # 40% loss

# Initialize logging
logging.basicConfig(
    filename='real_time_trading.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)


def log_trade_info(message):
    """
    Logs trade information to both the console and log file.
    """
    print(message)
    logging.info(message)


def get_latest_data(client, symbol, interval, limit=100):
    """
    Retrieves the latest market data.
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


def adjust_trade_quantity(client, symbol, margin, price):
    """
    Adjusts the trade quantity based on margin, price, and Binance's minimum quantity and step size.
    """
    # Calculate the initial position size
    quantity = (margin * LEVERAGE) / price

    # Get the symbol's minimum quantity and step size
    exchange_info = client.futures_exchange_info()
    symbol_info = next(item for item in exchange_info['symbols'] if item['symbol'] == symbol)
    filters = symbol_info['filters']
    lot_size = next(f for f in filters if f['filterType'] == 'LOT_SIZE')
    min_qty = float(lot_size['minQty'])
    step_size = float(lot_size['stepSize'])

    # Adjust quantity to be within allowed limits
    if quantity < min_qty:
        quantity = min_qty
    else:
        quantity = round(quantity - (quantity % step_size), 8)  # Adjust to step size

    return quantity


def check_positions(client, symbol):
    """
    Checks current positions for the symbol.
    """
    positions = client.futures_position_information(symbol=symbol)
    long_position = next((p for p in positions if p['positionSide'] == 'LONG'), None)
    short_position = next((p for p in positions if p['positionSide'] == 'SHORT'), None)
    return long_position, short_position


def cancel_all_orders(client, symbol, position_side):
    """
    Cancels all open orders for the symbol and position side.
    """
    try:
        orders = client.futures_get_open_orders(symbol=symbol)
        for order in orders:
            if order['positionSide'] == position_side:
                client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
        log_trade_info(f"All open orders for {symbol} {position_side} have been canceled.")
    except Exception as e:
        log_trade_info(f"Error canceling orders: {str(e)}")


def place_order_with_tp_sl(client, symbol, side, position_side, quantity, entry_price, margin):
    """
    Places an order and sets take-profit and stop-loss orders.
    """
    try:
        # Place the market order
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            positionSide=position_side,
            quantity=quantity
        )
        log_trade_info(f"{position_side} position opened: {quantity} {symbol} at {entry_price}.")

        # Calculate take-profit and stop-loss prices
        if position_side == 'LONG':
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT / LEVERAGE)
            stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT / LEVERAGE)
            tp_side = 'SELL'
            sl_side = 'SELL'
        else:
            take_profit_price = entry_price * (1 - TAKE_PROFIT_PERCENT / LEVERAGE)
            stop_loss_price = entry_price * (1 + STOP_LOSS_PERCENT / LEVERAGE)
            tp_side = 'BUY'
            sl_side = 'BUY'

        # Set take-profit order
        client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=round(take_profit_price, 5),
            closePosition=True,
            positionSide=position_side,
            timeInForce='GTE_GTC'
        )
        log_trade_info(f"Take-profit set at {take_profit_price}.")

        # Set stop-loss order
        client.futures_create_order(
            symbol=symbol,
            side=sl_side,
            type='STOP_MARKET',
            stopPrice=round(stop_loss_price, 5),
            closePosition=True,
            positionSide=position_side,
            timeInForce='GTE_GTC'
        )
        log_trade_info(f"Stop-loss set at {stop_loss_price}.")

    except Exception as e:
        log_trade_info(f"Error placing order: {str(e)}")


def execute_trade(client, symbol, signal, margin):
    """
    Executes a trade based on the signal.
    """
    # Get current positions
    long_position, short_position = check_positions(client, symbol)

    # Get current price
    df = get_latest_data(client, symbol, INTERVAL)
    current_price = df['close'].iloc[-1]

    # Adjust quantity
    quantity = adjust_trade_quantity(client, symbol, margin, current_price)

    if signal == 'long':
        if float(long_position['positionAmt']) == 0:
            # Cancel existing orders
            cancel_all_orders(client, symbol, 'LONG')
            # Open long position
            place_order_with_tp_sl(
                client=client,
                symbol=symbol,
                side='BUY',
                position_side='LONG',
                quantity=quantity,
                entry_price=current_price,
                margin=margin
            )
        else:
            log_trade_info("Already in a LONG position. No action taken.")
    elif signal == 'short':
        if float(short_position['positionAmt']) == 0:
            # Cancel existing orders
            cancel_all_orders(client, symbol, 'SHORT')
            # Open short position
            place_order_with_tp_sl(
                client=client,
                symbol=symbol,
                side='SELL',
                position_side='SHORT',
                quantity=quantity,
                entry_price=current_price,
                margin=margin
            )
        else:
            log_trade_info("Already in a SHORT position. No action taken.")
    else:
        log_trade_info("No trading signal.")


def main():
    # client = Client(api_key, api_secret)
    client = Client(api_key, api_secret,  requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})
    # client = Client(api_key, api_secret, testnet=True, requests_params={'proxies': {'https': 'http://127.0.0.1:7897'}})
    client.futures_change_leverage(symbol=SYMBOL, leverage=LEVERAGE)
    log_trade_info("Trading bot started.")

    while True:
        try:
            # Get the latest data
            df = get_latest_data(client, SYMBOL, INTERVAL)
            current_time = df['timestamp'].iloc[-1]
            current_price = df.loc[df['timestamp'] == current_time, 'close'].values[0]
            # Get trading signal
            signal = determine_position(current_time, df)
            log_trade_info(f"{current_time} | Price {current_price:.5f} | Signal: {signal}")

            # Execute trade based on the signal
            execute_trade(client, SYMBOL, signal, TRADE_MARGIN)

            # Wait for the next interval
            time.sleep(5)
        except Exception as e:
            log_trade_info(f"Error in main loop: {str(e)}")
            time.sleep(5)


if __name__ == "__main__":
    main()
