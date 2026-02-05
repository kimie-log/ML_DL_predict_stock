import csv
import os
import sys

# 加入專案根目錄
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.experiment_logger import log_experiment


def test_log_experiment_writes_csv(tmp_path):
    output_dir = tmp_path / "experiments"
    config = {
        "TICKER": "0050.TW",
        "START_DATE": "2020-01-01",
        "END_DATE": "2020-12-31",
        "TRAIN_START_DATE": "2020-01-01",
        "TRAIN_END_DATE": "2020-06-30",
        "VALID_START_DATE": "2020-07-01",
        "VALID_END_DATE": "2020-09-30",
        "TEST_START_DATE": "2020-10-01",
        "TEST_END_DATE": "2020-12-31",
        "SEQ_LENGTH": 10,
        "BATCH_SIZE": 32,
        "LEARNING_RATE": 0.001,
        "EPOCHS": 5,
    }

    # 呼叫兩次，確認會附加兩列
    log_experiment(
        model_name="UnitTestModel",
        dataset_split="train",
        metrics={"mse": 0.1},
        config=config,
        output_dir=str(output_dir),
        filename="results.csv",
    )
    log_experiment(
        model_name="UnitTestModel",
        dataset_split="test",
        metrics={"mse": 0.2},
        config=config,
        output_dir=str(output_dir),
        filename="results.csv",
    )

    csv_path = output_dir / "results.csv"
    assert csv_path.exists()

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    # 檢查部分欄位是否正確寫入
    assert rows[0]["model_name"] == "UnitTestModel"
    assert rows[0]["dataset_split"] == "train"
    assert rows[0]["ticker"] == "0050.TW"
    assert rows[0]["seq_length"] == "10"
    assert rows[0]["batch_size"] == "32"
    assert rows[0]["learning_rate"] == "0.001"
    assert rows[0]["epochs"] == "5"
    # metrics 欄位
    assert rows[0]["mse"] == "0.1"
    assert rows[1]["mse"] == "0.2"

