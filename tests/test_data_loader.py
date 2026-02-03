import pandas as pd
import io
from src.data.data_loader import DataLoader


def test_load_csv_data(tmp_path):
    csv_content = """date,open,high,low,close,volume,symbol
2024-01-01,100,105,99,102,1000,AAPL
2024-01-02,102,106,101,104,1200,AAPL
"""
    p = tmp_path / "sample.csv"
    p.write_text(csv_content)

    loader = DataLoader()
    df = loader.load_csv_data(str(p), symbol='AAPL')

    assert not df.empty
    assert 'close' in df.columns
    assert df['close'].iloc[0] == 102
