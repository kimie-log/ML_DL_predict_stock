# %%
# 決策樹範例
# 使用決策樹分類模型預測隔日收盤價漲跌方向，並可搭配樹結構／特徵重要性做解釋。

import json
import os
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree


# 專案根目錄：假設此檔案放在 machine_learning 資料夾下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.generate_ticker_data import generate_ticker_data  # noqa: E402
from utils.experiment_logger import log_experiment  # noqa: E402
from utils.report_saver import save_run_outputs  # noqa: E402


# 從 config/config.json 載入股票與日期設定
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)


# %%
# 準備訓練資料：特徵為 OHLCV，標籤為 Pred_UpDown（漲跌方向），並依時間切成 TRAIN / VALID / TEST。

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
y_train = train_data["Pred_UpDown"]

X_valid = valid_data[["Open", "High", "Low", "Close", "Volume"]]
y_valid = valid_data["Pred_UpDown"]

X_test = test_data[["Open", "High", "Low", "Close", "Volume"]]
y_test = test_data["Pred_UpDown"]

print(f"訓練集資料筆數: {X_train.shape[0]}")
print(f"驗證集資料筆數: {X_valid.shape[0]}")
print(f"測試集資料筆數: {X_test.shape[0]}")


# %%
# 建立與訓練決策樹模型。

model = DecisionTreeClassifier(random_state=1326, max_depth=None)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_valid_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)


# %%
# 輸出分類報告。

print("Train Classification Report:")
report_train_str = classification_report(y_true=y_train, y_pred=y_train_pred)
report_train = classification_report(
    y_true=y_train, y_pred=y_train_pred, output_dict=True
)
print(report_train_str)

print("Valid Classification Report:")
report_valid_str = classification_report(y_true=y_valid, y_pred=y_valid_pred)
report_valid = classification_report(
    y_true=y_valid, y_pred=y_valid_pred, output_dict=True
)
print(report_valid_str)

print("Test Classification Report:")
report_test_str = classification_report(y_true=y_test, y_pred=y_test_pred)
report_test = classification_report(
    y_true=y_test, y_pred=y_test_pred, output_dict=True
)
print(report_test_str)

# 記錄 accuracy / macro f1
log_experiment(
    model_name="DecisionTree",
    dataset_split="train",
    metrics={
        "accuracy": report_train["accuracy"],
        "macro_f1": report_train["macro avg"]["f1-score"],
    },
    config=config,
)
log_experiment(
    model_name="DecisionTree",
    dataset_split="valid",
    metrics={
        "accuracy": report_valid["accuracy"],
        "macro_f1": report_valid["macro avg"]["f1-score"],
    },
    config=config,
)
log_experiment(
    model_name="DecisionTree",
    dataset_split="test",
    metrics={
        "accuracy": report_test["accuracy"],
        "macro_f1": report_test["macro avg"]["f1-score"],
    },
    config=config,
)


# %%
# 視覺化決策樹結構（僅作為示意，深度較大時圖會很複雜）。

plt.figure(figsize=(16, 10))
plot_tree(
    model,
    feature_names=X_train.columns,
    class_names=["Down", "Up"],
    filled=True,
    max_depth=3,  # 僅顯示前幾層，避免過於複雜
)
plt.title("Decision Tree (First 3 Levels)")

# 先將樹圖存成檔案，再顯示
tree_png_path = os.path.join(PROJECT_ROOT, "outputs", "temp","decision_tree", "decision_tree_first3_levels.png")
os.makedirs(os.path.dirname(tree_png_path), exist_ok=True)
plt.tight_layout()
plt.savefig(tree_png_path, bbox_inches="tight")

# 使用通用工具函式儲存本次實驗的報表與圖檔
save_run_outputs(
    model_name="DecisionTree",
    category="machine_learning",
    text_reports={
        "train_classification_report": report_train_str,
        "valid_classification_report": report_valid_str,
        "test_classification_report": report_test_str,
    },
    artifact_paths=[tree_png_path],
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
        "max_depth": None,
    },
)


if __name__ == "__main__":
    # 直接執行此檔案即可跑完決策樹範例。
    pass
