"""
Data Loader - Fetch OHLCV data from various sources
"""

import pandas as pd
from typing import Optional
from ..utils.logger import get_logger
from ..utils.config import get_config

logger = get_logger(__name__)


class DataLoader:
    """Load OHLCV data from Yahoo Finance or other sources."""
    
    def __init__(self):
        """Initialize DataLoader with configuration."""
        self.config = get_config()
    
    def get_yahoo_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with OHLCV data and DatetimeIndex
        
        Raises:
            Exception: If data cannot be fetched
        """
        try:
            logger.info(f"Fetching {symbol} data for period {period} with interval {interval}")
            
            # Import yfinance lazily to avoid import-time binary dependency issues
            try:
                import yfinance as yf
            except Exception as e:
                logger.error(f"yfinance import failed: {e}")
                raise ImportError("yfinance is required to fetch Yahoo data. Install it with 'pip install yfinance' or avoid using get_yahoo_data() in environments where optional binary deps (gevent/greenlet) are incompatible.") from e

            # Download data
            data = yf.download(
                symbol,
                period=period,
                interval=interval,
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Normalize column names to lowercase (handle MultiIndex from yfinance)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in data.columns]
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            logger.info(f"Successfully loaded {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            raise

    def load_csv_data(self, file_path: str, symbol: Optional[str] = None) -> pd.DataFrame:
        """Load OHLCV data from a CSV file.

        The CSV is expected to contain at least: date, open, high, low, close, volume
        Optionally a `symbol` column can be used to filter rows.
        """
        try:
            df = pd.read_csv(file_path)

            # Normalize columns
            df.columns = [c.lower() for c in df.columns]

            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            elif isinstance(df.index, pd.DatetimeIndex):
                # already has datetime index
                pass
            else:
                raise ValueError('CSV must contain a date column or a DatetimeIndex')

            if symbol and 'symbol' in df.columns:
                df = df[df['symbol'] == symbol]

            # Ensure lowercase ohlcv names
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns in CSV: {missing}")

            # Convert numeric values
            df[required_cols] = df[required_cols].astype(float)

            logger.info(f"Loaded CSV data from {file_path} ({len(df)} rows)")
            return df

        except Exception as e:
            logger.error(f"Failed to load CSV data from {file_path}: {e}")
            raise
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate OHLCV data structure.
        
        Args:
            data: DataFrame to validate
        
        Returns:
            True if valid, False otherwise
        """
        if data is None or data.empty:
            logger.warning("Data is empty")
            return False
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            return False
        
        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex")
            return False
        
        # Check for NaN values
        if data[required_cols].isnull().any().any():
            logger.warning("Data contains NaN values")
            return False
        
        # Check that high >= low
        if not (data['high'] >= data['low']).all():
            logger.warning("Found rows where high < low")
            return False
        
        logger.info("Data validation passed")
        return True
    
    def get_binance_data(
        self,
        symbol: str,
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data from Binance (placeholder for future implementation).
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Kline interval
            start_date: Start date string
            end_date: End date string
        
        Returns:
            DataFrame with OHLCV data
        """
        raise NotImplementedError("Binance data loading not yet implemented")


if __name__ == "__main__":
    # Quick test
    loader = DataLoader()
    
    try:
        data = loader.get_yahoo_data("AAPL", period="1mo")
        print(f"✓ Loaded {len(data)} data points")
        print(f"✓ Columns: {list(data.columns)}")
        print(f"✓ Date range: {data.index[0]} to {data.index[-1]}")
        
        if loader.validate_data(data):
            print("✓ Data validation passed")
        else:
            print("✗ Data validation failed")
            
    except Exception as e:
        print(f"✗ Error: {e}")
