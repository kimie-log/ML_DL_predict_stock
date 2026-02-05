import numpy as np

from utils.stock_dataset import StockDataset


def test_stock_dataset_len_and_item():
    # 建立簡單的假資料：10 筆、3 個特徵
    data = np.arange(30, dtype=float).reshape(10, 3)
    seq_length = 4
    x_idx = [0, 1]  # 前兩個特徵當作 X
    y_idx = 2       # 第三個特徵當作 y

    dataset = StockDataset(data=data, seq_length=seq_length, x_idx=x_idx, y_idx=y_idx)

    # len 應該是總長度減去 seq_length
    assert len(dataset) == 10 - seq_length

    # 取第一個樣本檢查 shape 與內容
    x, y = dataset[0]
    # x: (seq_length, len(x_idx))
    assert x.shape == (seq_length, len(x_idx))
    # y: scalar tensor
    assert y.shape == ()

    # x 的第一列應對應 data[0, x_idx]，最後一列對應 data[seq_length-1, x_idx]
    np.testing.assert_allclose(x[0].numpy(), data[0, x_idx])
    np.testing.assert_allclose(x[-1].numpy(), data[seq_length - 1, x_idx])
    # y 應該對應 data[seq_length, y_idx]
    assert float(y.numpy()) == data[seq_length, y_idx]

