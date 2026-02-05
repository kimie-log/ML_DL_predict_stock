import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    """自定義數據集，將資料處理為時間序列模型的輸入格式。"""

    def __init__(self, data, seq_length, x_idx, y_idx):
        self.data = data
        self.seq_length = seq_length
        self.x_idx = x_idx
        self.y_idx = y_idx

    def __len__(self):
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx : (idx + self.seq_length), self.x_idx]  # noqa: E203
        x = torch.tensor(x, dtype=torch.float32)
        y = self.data[idx + self.seq_length, self.y_idx]
        y = torch.tensor(y, dtype=torch.float32)
        return x, y