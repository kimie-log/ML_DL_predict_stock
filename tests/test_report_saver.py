import os
from pathlib import Path

from utils.report_saver import ENV_RUN_ID_VAR, get_run_id, save_run_outputs


def test_get_run_id_uses_env(monkeypatch):
    monkeypatch.setenv(ENV_RUN_ID_VAR, "TEST_RUN_ID_123")
    assert get_run_id() == "TEST_RUN_ID_123"


def test_save_run_outputs_creates_structure(tmp_path, monkeypatch):
    # 固定 run id，方便檢查路徑
    monkeypatch.setenv(ENV_RUN_ID_VAR, "20260101_000000")

    base_dir = tmp_path / "report"
    text_reports = {
        "summary": "this is a test",
        "metrics": "mse=0.1234",
    }

    artifact_src = tmp_path / "dummy.png"
    artifact_src.write_bytes(b"fake-binary")

    out_dir = save_run_outputs(
        model_name="MyModel",
        category="machine_learning",
        text_reports=text_reports,
        artifact_paths=[str(artifact_src)],
        params={"foo": "bar"},
        base_dir=str(base_dir),
    )

    out_path = Path(out_dir)
    # 路徑應該包含固定的 run id 與類別_模型名稱
    assert "20260101_000000" in str(out_path)
    assert out_path.name == "machine_learning_MyModel"

    # 檢查文字檔
    summary_path = out_path / "summary.txt"
    metrics_path = out_path / "metrics.txt"
    assert summary_path.exists()
    assert metrics_path.exists()
    assert summary_path.read_text(encoding="utf-8") == "this is a test"

    # 檢查複製的 artifact
    copied_artifact = out_path / "dummy.png"
    assert copied_artifact.exists()
    assert copied_artifact.read_bytes() == b"fake-binary"

    # 檢查 metadata.json
    metadata_path = out_path / "metadata.json"
    assert metadata_path.exists()
    content = metadata_path.read_text(encoding="utf-8")
    assert '"model_name": "MyModel"' in content
    assert '"category": "machine_learning"' in content
    assert '"foo": "bar"' in content

