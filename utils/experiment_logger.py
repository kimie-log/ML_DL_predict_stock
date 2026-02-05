import csv
import os
from datetime import datetime, timezone
from typing import Dict, Any


def log_experiment(
    model_name: str,
    dataset_split: str,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    output_dir: str = "experiments",
    filename: str = "results.csv",
) -> None:
    """
    將單次實驗結果以 row 形式附加寫入 CSV。

    - model_name: 模型名稱，例如 "LinearRegression"、"LSTM"。
    - dataset_split: "train" / "valid" / "test"。
    - metrics: 例如 {"mse": 0.1234} 或 {"accuracy": 0.55, "f1": 0.48}。
    - config: 會被簡單扁平化儲存關鍵欄位（如 TICKER、時間區間、SEQ_LENGTH 等）。
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    # 只挑重點欄位，避免把整個 config 巨量塞進一格
    summary_config = {
        "ticker": config.get("TICKER"),
        "start_date": config.get("START_DATE"),
        "end_date": config.get("END_DATE"),
        "train_start": config.get("TRAIN_START_DATE"),
        "train_end": config.get("TRAIN_END_DATE"),
        "valid_start": config.get("VALID_START_DATE"),
        "valid_end": config.get("VALID_END_DATE"),
        "test_start": config.get("TEST_START_DATE"),
        "test_end": config.get("TEST_END_DATE"),
        "seq_length": config.get("SEQ_LENGTH"),
        "batch_size": config.get("BATCH_SIZE"),
        "learning_rate": config.get("LEARNING_RATE"),
        "epochs": config.get("EPOCHS"),
    }

    # 使用 timezone-aware 的 UTC 時間，避免 datetime.utcnow() 的棄用警告
    timestamp = datetime.now(timezone.utc).isoformat()

    row: Dict[str, Any] = {
        "timestamp_utc": timestamp,
        "model_name": model_name,
        "dataset_split": dataset_split,
        **summary_config,
        **metrics,
    }

    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

