import pandas as pd
import yfinance as yf
from typing_extensions import Annotated


def generate_ticker_data(
    ticker: Annotated[str, "股票代碼"],
    start_date: Annotated[str, "資料開始日期（格式：YYYY-MM-DD）"],
    end_date: Annotated[str, "資料結束日期（格式：YYYY-MM-DD）"],
) -> Annotated[pd.DataFrame, "處理後的股票資料，包含新增的預測收盤價和漲跌方向欄位"]:
    """
    下載股票資料並進行資料預處理。

    步驟:
    1. 下載股票資料，填補遺失值（使用前一日的數值填補）。
    2. 新增欄位 Pred_Close：下一天的收盤價。
    3. 新增欄位 Pred_UpDown：下一天收盤價的上漲或下跌方向（上漲: 1, 下跌: 0）。
    """
    ticker_data = (
        pd.DataFrame(yf.download(ticker, start=start_date, end=end_date))
        .droplevel("Ticker", axis=1)
        .reset_index()
        .ffill()
    )
    ticker_data.columns.name = None

    ticker_data["Pred_Close"] = ticker_data["Close"].shift(-1)
    ticker_data["Pred_PctChange_Close"] = (
        ticker_data["Pred_Close"] - ticker_data["Close"]
    ) / ticker_data["Close"]
    ticker_data["Pred_UpDown"] = ticker_data["Pred_PctChange_Close"].apply(
        lambda x: 1 if x > 0 else 0
    )
    ticker_data = ticker_data.reset_index(drop=True)
    ticker_data = ticker_data.ffill().dropna()
    return ticker_data