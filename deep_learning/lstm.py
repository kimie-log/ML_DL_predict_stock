# %%
# LSTM 時間序列預測
# 使用過去一段時間的 OHLCV 序列，透過 LSTM 模型預測下一日收盤價。

import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


# 專案根目錄：假設此檔案放在 deep_learning 資料夾下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.generate_ticker_data import generate_ticker_data  # noqa: E402
from utils.stock_dataset import StockDataset  # noqa: E402
from utils.experiment_logger import log_experiment  # noqa: E402
from utils.lineplot_true_and_predicted_result import (  # noqa: E402
    lineplot_true_and_predicted_result,
)
from utils.report_saver import get_run_id, save_run_outputs  # noqa: E402


# %%
# 載入訓練設定
# 使用專案根目錄下的 train_config.json 來設定標的、時間區間以及超參數。

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

TICKER = config["TICKER"]
START_DATE = config["START_DATE"]
END_DATE = config["END_DATE"]
TRAIN_START_DATE = config["TRAIN_START_DATE"]
TRAIN_END_DATE = config["TRAIN_END_DATE"]
VALID_START_DATE = config["VALID_START_DATE"]
VALID_END_DATE = config["VALID_END_DATE"]
TEST_START_DATE = config["TEST_START_DATE"]
TEST_END_DATE = config["TEST_END_DATE"]
SEQ_LENGTH = config["SEQ_LENGTH"]
BATCH_SIZE = config["BATCH_SIZE"]
LEARNING_RATE = config["LEARNING_RATE"]
EPOCHS = config["EPOCHS"]


# %%
# 設定隨機種子，確保實驗可重現。

seed = 1326
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# %%
# 準備資料並進行 MinMax 縮放。

train_data = generate_ticker_data(
    ticker=TICKER,
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
)[["Open", "High", "Low", "Close", "Volume"]]

valid_data = generate_ticker_data(
    ticker=TICKER,
    start_date=VALID_START_DATE,
    end_date=VALID_END_DATE,
)[["Open", "High", "Low", "Close", "Volume"]]

test_data = generate_ticker_data(
    ticker=TICKER,
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
)[["Open", "High", "Low", "Close", "Volume"]]

print(f"訓練集資料筆數: {train_data.shape[0]}")
print(f"驗證集資料筆數: {valid_data.shape[0]}")
print(f"測試集資料筆數: {test_data.shape[0]}")

features_scaler = MinMaxScaler()
features_scaler.fit(train_data)

train_data = features_scaler.transform(train_data)
valid_data = features_scaler.transform(valid_data)
test_data = features_scaler.transform(test_data)

train_dataset = StockDataset(train_data, SEQ_LENGTH, x_idx=[0, 1, 2, 3, 4], y_idx=3)
valid_dataset = StockDataset(valid_data, SEQ_LENGTH, x_idx=[0, 1, 2, 3, 4], y_idx=3)
test_dataset = StockDataset(test_data, SEQ_LENGTH, x_idx=[0, 1, 2, 3, 4], y_idx=3)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %%
# 定義 LSTM 模型。


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, inputs):
        # (batch_size, seq_length, features_nums) -> (batch_size, seq_length, hidden_size)
        outputs, _ = self.lstm(inputs)
        # 取最後一個時間步的輸出
        outputs = outputs[:, -1, :]
        outputs = self.linear(outputs)
        return outputs


model = Model(input_size=5, hidden_size=64, num_layers=2, output_size=1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# %%
# 建立 TensorBoard 與模型參數輸出路徑：
# 統一放在 outputs/{時間戳}/lstm，與 report/ 共用同一 run id。

RUN_ID = get_run_id()
LSTM_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", RUN_ID, "lstm")
WRITER_PATH = os.path.join(LSTM_OUTPUT_DIR, "writer")
MODELPARAM_PATH = os.path.join(LSTM_OUTPUT_DIR, "model_param")
PLOTS_PATH = os.path.join(LSTM_OUTPUT_DIR, "plots")
os.makedirs(WRITER_PATH, exist_ok=True)
os.makedirs(MODELPARAM_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)

writer = SummaryWriter(WRITER_PATH)


# %%
# 訓練與驗證迴圈。


def train_one_epoch(epoch_idx: int):
    model.train()
    total_loss = 0.0
    for step, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = loss_fn(outputs.squeeze(-1), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch_idx)
    return avg_loss


def evaluate(epoch_idx: int, loader: DataLoader, split_name: str):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            outputs = model(x_batch)
            loss = loss_fn(outputs.squeeze(-1), y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    writer.add_scalar(f"Loss/{split_name}", avg_loss, epoch_idx)
    return avg_loss


best_valid_loss = float("inf")
last_train_loss = None

for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(epoch)
    valid_loss = evaluate(epoch, valid_loader, "valid")
    last_train_loss = train_loss

    print(f"[Epoch {epoch}/{EPOCHS}] train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}")

    # 若驗證集表現變好就儲存模型參數
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        save_path = os.path.join(MODELPARAM_PATH, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  -> 儲存最佳模型參數至: {save_path}")


# %%
# 在測試集上做評估並繪圖。

test_loss = evaluate(EPOCHS, test_loader, "test")
print(f"Test Loss (MSE): {test_loss:.4f}")

# 將實驗結果記錄到 experiments/results.csv
log_experiment(
    model_name="LSTM",
    dataset_split="train",
    metrics={"mse": float(last_train_loss) if last_train_loss is not None else float("nan")},
    config=config,
)
log_experiment(
    model_name="LSTM",
    dataset_split="valid",
    metrics={"mse": float(best_valid_loss)},
    config=config,
)
log_experiment(
    model_name="LSTM",
    dataset_split="test",
    metrics={"mse": float(test_loss)},
    config=config,
)

# 產生測試集預測 vs 真實的折線圖（使用縮放後的數值）
model.eval()
preds = []
trues = []
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        preds.extend(outputs.squeeze(-1).cpu().numpy())
        trues.extend(y_batch.cpu().numpy())

plot_path = os.path.join(PLOTS_PATH, "lstm_test_prediction.png")

lineplot_true_and_predicted_result(
    true_values=trues,
    predicted_values=preds,
    title="LSTM Test Set Prediction",
    save_path=plot_path,
)

# 將本次深度學習實驗輸出到 report/
summary_text = (
    f"Train MSE (last epoch): {float(last_train_loss) if last_train_loss is not None else float('nan'):.4f}\n"
    f"Valid MSE (best):       {float(best_valid_loss):.4f}\n"
    f"Test  MSE:              {float(test_loss):.4f}\n"
)

save_run_outputs(
    model_name="LSTM",
    category="deep_learning",
    text_reports={"mse_summary": summary_text},
    artifact_paths=[plot_path],
    params={
        "TICKER": TICKER,
        "START_DATE": START_DATE,
        "END_DATE": END_DATE,
        "TRAIN_START_DATE": TRAIN_START_DATE,
        "TRAIN_END_DATE": TRAIN_END_DATE,
        "VALID_START_DATE": VALID_START_DATE,
        "VALID_END_DATE": VALID_END_DATE,
        "TEST_START_DATE": TEST_START_DATE,
        "TEST_END_DATE": TEST_END_DATE,
        "SEQ_LENGTH": SEQ_LENGTH,
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "hidden_size": 64,
        "num_layers": 2,
    },
)


if __name__ == "__main__":
    # 直接執行此檔案即可完成 LSTM 訓練與測試流程。
    pass

