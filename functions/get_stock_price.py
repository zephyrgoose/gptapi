import yfinance as yf # type: ignore
import logging

# Configure logging
logging.basicConfig(filename="../debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_stock_price(symbol):
    """Fetches the current stock price for a given stock symbol."""
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.history(period="1d")

        if stock_info.empty:
            logging.warning(f"No data found for stock symbol: {symbol}")
            return {"error": f"No data found for stock symbol {symbol}"}

        latest_price = stock_info["Close"].iloc[-1]
        logging.info(f"Retrieved stock price for {symbol}: {latest_price}")

        return {
            "symbol": symbol,
            "price": round(latest_price, 2),
            "currency": "USD"
        }
    except Exception as e:
        logging.error(f"Error fetching stock price for {symbol}: {e}")
        return {"error": f"Could not retrieve stock data for {symbol}"}
