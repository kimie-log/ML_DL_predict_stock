import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, Any, Optional


# 若環境變數中有設定 REPORT_RUN_ID，則所有腳本在同一輪實驗中會共用同一個時間戳。
ENV_RUN_ID_VAR = "REPORT_RUN_ID"


def get_run_id() -> str:
    """回傳本輪實驗的 run id（與 report_saver 共用，可用於 outputs/ 路徑）。"""
    env_run_id = os.getenv(ENV_RUN_ID_VAR)
    return env_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")


def _sanitize_name(name: str) -> str:
    """將模型名稱 / 類別名稱轉成適合當作資料夾名稱的字串。"""
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip())
    return cleaned or "model"


def save_run_outputs(
    model_name: str,
    category: str,
    text_reports: Optional[Mapping[str, str]] = None,
    artifact_paths: Optional[Sequence[str]] = None,
    params: Optional[Mapping[str, Any]] = None,
    base_dir: str = "report",
) -> str:
    """
    儲存單次實驗輸出（文字報告、圖片檔 / 其他檔案、參數）到時間戳目錄。

    目錄結構：
        report/{YYYYMMDD_HHMMSS}/{category}_{model_name}/
            - metadata.json
            - *.txt       （由 text_reports 產生）
            - 原始檔案複製過來（由 artifact_paths 產生）

    參數：
        - model_name:    模型名稱，例如 "LinearRegression"、"LSTM"。
        - category:      類別，例如 "machine_learning"、"deep_learning"。
        - text_reports:  key 為檔名（不含副檔名），value 為要寫入的文字內容。
        - artifact_paths:要一併複製過來的檔案路徑（例如 PNG 圖片、額外輸出）。
        - params:        本次實驗重要參數（會輸出到 metadata.json）。
        - base_dir:      最外層報告目錄名稱，預設為 "report"。

    回傳：
        - 最終輸出目錄的絕對路徑字串。
    """
    timestamp = get_run_id()
    safe_model = _sanitize_name(model_name)
    safe_category = _sanitize_name(category)

    base_path = Path(base_dir)
    run_dir = base_path / timestamp / f"{safe_category}_{safe_model}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # 寫入文字報告
    if text_reports:
        for name, content in text_reports.items():
            safe_name = _sanitize_name(name)
            txt_path = run_dir / f"{safe_name}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)

    # 複製其他輸出檔案（例如圖檔）
    if artifact_paths:
        for src in artifact_paths:
            src_path = Path(src)
            if not src_path.exists():
                continue
            target_path = run_dir / src_path.name
            # 若同名檔案已存在，避免覆蓋可在檔名後面加上序號或時間戳。
            if target_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                target_path = run_dir / f"{stem}_{timestamp}{suffix}"
            shutil.copy2(src_path, target_path)

    # 寫入 metadata.json（包含模型名稱、類別、時間戳與參數）
    metadata = {
        "model_name": model_name,
        "category": category,
        "timestamp": timestamp,
        "params": dict(params) if params is not None else {},
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return os.path.abspath(run_dir)

