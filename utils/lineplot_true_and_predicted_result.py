import matplotlib.pyplot as plt
import pandas as pd
from typing_extensions import Annotated


def lineplot_true_and_predicted_result(
    true_values: Annotated[pd.Series, "真實的目標值"],
    predicted_values: Annotated[pd.Series, "模型預測的目標值"],
    title: Annotated[str, "圖表標題"],
    save_path: Annotated[
        str | None,
        "選填：若提供路徑則會將圖存成檔案（同時仍會顯示）",
    ] = None,
):
    """繪製真實值和預測值的折線圖（可選擇同時存檔）"""
    plt.figure(figsize=(14, 7))
    plt.plot(true_values, label="True", linewidth=3)
    plt.plot(predicted_values, label="Predicted", linewidth=3)
    plt.title(title, fontsize=30)
    plt.xlabel("Time", fontsize=30)
    plt.ylabel("Close Price", fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fontsize=25)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
