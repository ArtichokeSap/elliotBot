"""
Tests for Data Loader Module
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test DataLoader class."""
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        loader = DataLoader()
        assert loader is not None
    
    @patch('src.data.data_loader.yf.download')
    def test_get_yahoo_data_success(self, mock_download, sample_data):
        """Test successful data retrieval from Yahoo Finance."""
        mock_download.return_value = sample_data
        
        loader = DataLoader()
        result = loader.get_yahoo_data('AAPL', period='1mo')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert all(col in result.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        mock_download.assert_called_once()
    
    def test_get_yahoo_data_invalid_symbol(self):
        """Test handling of invalid symbol."""
        loader = DataLoader()
        with pytest.raises(Exception):
            loader.get_yahoo_data('INVALID_SYMBOL_XYZ123', period='1d')
    
    def test_validate_data_structure(self, sample_data):
        """Test data validation."""
        loader = DataLoader()
        
        # Valid data should pass
        assert loader.validate_data(sample_data) is True
        
        # Missing column should fail
        invalid_data = sample_data.drop(columns=['close'])
        assert loader.validate_data(invalid_data) is False
        
        # Empty data should fail
        empty_data = pd.DataFrame()
        assert loader.validate_data(empty_data) is False
    
    def test_data_has_datetime_index(self, sample_data):
        """Test that data has proper datetime index."""
        assert isinstance(sample_data.index, pd.DatetimeIndex)
    
    def test_data_columns_exist(self, sample_data):
        """Test that required columns exist."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        assert all(col in sample_data.columns for col in required_cols)
    
    def test_high_low_relationships(self, sample_data):
        """Test that high >= low for all rows."""
        assert (sample_data['high'] >= sample_data['low']).all()
