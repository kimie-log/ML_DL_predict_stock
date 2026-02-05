import os
import sys

import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.generate_ticker_data import generate_ticker_data  # noqa: E402


def test_generate_ticker_data_basic():
    df = generate_ticker_data("0050.TW", "2023-01-01", "2023-03-01")

    # 檢查基本欄位是否存在
    for col in ["Date", "Open", "High", "Low", "Close", "Volume", "Pred_Close", "Pred_UpDown"]:
        assert col in df.columns

    # 至少要有幾筆資料
    assert len(df) > 0

    # 不應該有 NaN（前處理已做 ffill + dropna）
    assert not df.isna().any().any()

