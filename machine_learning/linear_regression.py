# %%
# 線性迴歸範例
# 使用開盤價、最高價、最低價、收盤價、交易量來預測隔日收盤價。
# 範例流程：
# 1. 從 yfinance 下載指定股票的歷史資料並前處理。
# 2. 使用線性迴歸模型學習「今日 OHLCV -> 隔日收盤價」的關係。
# 3. 以訓練 / 測試 MSE 評估模型，並繪製真實值與預測值。

import json
import os
import sys

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 專案根目錄：假設此檔案放在 machine_learning 資料夾下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.generate_ticker_data import generate_ticker_data  # noqa: E402
from utils.lineplot_true_and_predicted_result import (  # noqa: E402
    lineplot_true_and_predicted_result,
)
from utils.experiment_logger import log_experiment  # noqa: E402
from utils.report_saver import save_run_outputs  # noqa: E402


# 從 config/config.json 載入股票與日期設定
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)


# %%
# 準備訓練 / 驗證 / 測試資料
# 使用設定檔中的股票代碼與時間區間產生特徵與標籤：
# 特徵：Open, High, Low, Close, Volume
# 標籤：隔日收盤價（Pred_Close），並嚴格依時間切出 TRAIN / VALID / TEST 三段。

ticker = config["TICKER"]

train_data = generate_ticker_data(
    ticker=ticker,
    start_date=config["TRAIN_START_DATE"],
    end_date=config["TRAIN_END_DATE"],
)
valid_data = generate_ticker_data(
    ticker=ticker,
    start_date=config["VALID_START_DATE"],
    end_date=config["VALID_END_DATE"],
)
test_data = generate_ticker_data(
    ticker=ticker,
    start_date=config["TEST_START_DATE"],
    end_date=config["TEST_END_DATE"],
)

X_train = train_data[["Open", "High", "Low", "Close", "Volume"]]
y_train = train_data["Pred_Close"]

X_valid = valid_data[["Open", "High", "Low", "Close", "Volume"]]
y_valid = valid_data["Pred_Close"]

X_test = test_data[["Open", "High", "Low", "Close", "Volume"]]
y_test = test_data["Pred_Close"]

print(f"訓練集資料筆數: {X_train.shape[0]}")
print(f"驗證集資料筆數: {X_valid.shape[0]}")
print(f"測試集資料筆數: {X_test.shape[0]}")


# %%
# 建立並訓練線性迴歸模型，接著在訓練集與測試集上做推論。

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)


# %%
# 使用均方誤差（MSE）評估模型在訓練 / 驗證 / 測試資料上的表現。

mse_train = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
mse_valid = mean_squared_error(y_true=y_valid, y_pred=y_valid_pred)
mse_test = mean_squared_error(y_true=y_test, y_pred=y_test_pred)

print(f"Train Mean Squared Error: {mse_train:.4f}")
print(f"Valid Mean Squared Error: {mse_valid:.4f}")
print(f"Test Mean Squared Error: {mse_test:.4f}")

# 將實驗結果記錄到 experiments/results.csv
log_experiment(
    model_name="LinearRegression",
    dataset_split="train",
    metrics={"mse": mse_train},
    config=config,
)
log_experiment(
    model_name="LinearRegression",
    dataset_split="valid",
    metrics={"mse": mse_valid},
    config=config,
)
log_experiment(
    model_name="LinearRegression",
    dataset_split="test",
    metrics={"mse": mse_test},
    config=config,
)


# %%
# 視覺化真實值與預測值的折線圖，分別針對訓練集 / 驗證集 / 測試集。

plots_dir = os.path.join(PROJECT_ROOT, "outputs","temp","linear_regression")
os.makedirs(plots_dir, exist_ok=True)

train_plot_path = os.path.join(plots_dir, "train.png")
valid_plot_path = os.path.join(plots_dir, "valid.png")
test_plot_path = os.path.join(plots_dir, "test.png")

lineplot_true_and_predicted_result(
    true_values=y_train.values,
    predicted_values=y_train_pred,
    title="Linear Regression Result For TrainSet",
    save_path=train_plot_path,
)

lineplot_true_and_predicted_result(
    true_values=y_valid.values,
    predicted_values=y_valid_pred,
    title="Linear Regression Result For ValidSet",
    save_path=valid_plot_path,
)

lineplot_true_and_predicted_result(
    true_values=y_test.values,
    predicted_values=y_test_pred,
    title="Linear Regression Result For TestSet",
    save_path=test_plot_path,
)

# 將本次實驗的 MSE 與圖檔輸出到 report/
summary_text = (
    f"Train MSE: {mse_train:.4f}\n"
    f"Valid MSE: {mse_valid:.4f}\n"
    f"Test  MSE: {mse_test:.4f}\n"
)

save_run_outputs(
    model_name="LinearRegression",
    category="machine_learning",
    text_reports={"mse_summary": summary_text},
    artifact_paths=[train_plot_path, valid_plot_path, test_plot_path],
    params={
        "TICKER": config.get("TICKER"),
        "START_DATE": config.get("START_DATE"),
        "END_DATE": config.get("END_DATE"),
        "TRAIN_START_DATE": config.get("TRAIN_START_DATE"),
        "TRAIN_END_DATE": config.get("TRAIN_END_DATE"),
        "VALID_START_DATE": config.get("VALID_START_DATE"),
        "VALID_END_DATE": config.get("VALID_END_DATE"),
        "TEST_START_DATE": config.get("TEST_START_DATE"),
        "TEST_END_DATE": config.get("TEST_END_DATE"),
    },
)


if __name__ == "__main__":
    # 直接執行此檔案即可跑完整個流程。
    # 由於所有程式碼都在頂層，這裡不需額外呼叫函式。
    pass
