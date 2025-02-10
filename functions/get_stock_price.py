import yfinance as yf # type: ignore
import logging

# Configure logging
logging.basicConfig(filename="../debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def get_stock_price(symbol):
    """
    Fetches the latest closing stock price for a given stock symbol using Yahoo Finance.

    This function retrieves the most recent stock price available for the specified 
    ticker symbol. If no data is found, an error message is returned.

    Args:
        symbol (str): The stock ticker symbol (e.g., "AAPL" for Apple, "TSLA" for Tesla).

    Returns:
        dict: A dictionary containing stock price information:
            - "symbol" (str): The stock ticker symbol.
            - "price" (float): The most recent closing stock price, rounded to two decimal places.
            - "currency" (str): The currency in which the stock is priced (default: "USD").
            - "error" (str, optional): An error message if data retrieval fails.

    Raises:
        Exception: If there is an issue retrieving stock data from Yahoo Finance.
    """
    
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
