## ML & DL Stock Prediction on Taiwan 0050

### 專案定位

這是一個以 **台股 0050** 為例，系統性比較多種 **機器學習（ML）與深度學習（DL）模型** 在同一筆股價資料上的預測效能的專案
專案強調：

-   **完整 MLOps 起點流程**：從資料下載、特徵工程、資料集切分，到模型訓練、視覺化與結果分析
-   **同一資料、多種模型公平比較**：所有模型共用**相同的 OHLCV 歷史資料與三段時間切分方式（TRAIN / VALID / TEST）**，方便檢視不同模型的優缺點

目前支援兩大任務：

-   **迴歸**：預測隔日收盤價（`Pred_Close`）
-   **二元分類**：預測隔日漲跌方向（`Pred_UpDown`，1 = 上漲、0 = 下跌）

共用的工具模組：

-   `utils/generate_ticker_data.py`：下載並整理股價資料（新增 `Pred_Close`、`Pred_UpDown` 等欄位）
-   `utils/lineplot_true_and_predicted_result.py`：繪製真實值與預測值折線圖（可同時存 PNG 檔，方便作品集使用）
-   `utils/stock_dataset.py`：深度學習用的時間序列資料集
-   `utils/experiment_logger.py`：統一將各模型的 train / valid / test 指標與關鍵設定記錄到 `experiments/results.csv`
-   `utils/report_saver.py`：將單次實驗的文字報告、圖檔與重要超參數集中存到 `report/{時間戳}/{類別_模型名稱}/`；並提供 `get_run_id()` 供深度學習腳本將模型權重與圖檔存到 `outputs/{同一時間戳}/{模型名}/`
-   `config/config.json`：機器學習與深度學習範例的標的、日期與訓練超參數

---

## 專案重點亮點（Highlights）

-   **同一資料多模型比較**：線性 / 邏輯迴歸、決策樹、隨機森林、XGBoost、LSTM、LSTM+CNN、Attention，一次彙整
-   **時間序列正確切分**：嚴格保留時間順序，避免資料洩漏，所有模型（ML + DL）統一使用 `config.json` 中定義的 TRAIN / VALID / TEST 三段連續時間區間，示範實務切法
-   **可重現實驗流程**：所有關鍵設定集中在 `config` ，方便重新實驗與調參
-   **結果視覺化**：支援 TensorBoard 與折線圖，快速檢視訓練曲線與預測 vs 真實走勢
-   **易於擴充**：可以很容易替換標的（不只 0050）、新增技術指標特徵或加入其他模型

---

## 技術棧（Tech Stack）

-   **語言**：Python 3.13
-   **資料處理**：pandas, NumPy, scikit-learn
-   **機器學習**：scikit-learn, XGBoost
-   **深度學習**：PyTorch
-   **視覺化 / 監控**：Matplotlib、TensorBoard
-   **環境管理**：Conda、`requirements.txt`

---

## 快速開始（Quick Start）

### 安裝環境

1. 建議使用 Conda 建立虛擬環境（名稱可自訂，例如 `ML_DL_predict_stock`）：

```bash
conda create -n ML_DL_predict_stock python=3.13
conda activate ML_DL_predict_stock
```

2. 安裝必要套件（可依你實際使用調整）：

```bash
pip install -r requirements.txt
```

在 macOS 若要使用 XGBoost，建議另外安裝 OpenMP：

```bash
brew install libomp
```

3. 在專案根目錄下執行（確保目前路徑是 `/Users/.../ML_DL_predict_stock`）：

```bash
cd /Users/szwei/Desktop/ML_DL_predict_stock
```

### 執行機器學習與深度學習範例

若已安裝 `make`（macOS / Linux 預設都有），可以直接使用 `Makefile` 中的指令：

```bash
# 一次跑完 ML + 全部 DL，report/ 與 outputs/ 都落在同一時間戳下
make run-all-with-report

# 僅 ML 同一時間戳
make run-all-ml-with-report

# 僅 DL（LSTM / LSTM+CNN / Attention）同一時間戳
make run-all-train-lstm-with-report
```

或直接執行各腳本：

```bash
# 機器學習
python machine_learning/linear_regression.py
python machine_learning/logistic_regression.py
python machine_learning/decision_tree.py
python machine_learning/random_forest.py
python machine_learning/xgboost_binary_classification.py

# 深度學習
python deep_learning/lstm.py
python deep_learning/lstm_cnn.py
python deep_learning/attention_model.py
```

### 查看 TensorBoard（深度學習訓練過程）

深度學習的 TensorBoard log、模型參數（`best_model.pth`）與預測圖會依**時間戳**存放：

-   路徑格式：`outputs/{時間戳}/{模型名}/`，例如：
    -   `outputs/20260205_140332/lstm/`（內含 `writer/`、`model_param/`、`plots/`）
    -   `outputs/20260205_140332/lstm_cnn/`
    -   `outputs/20260205_140332/attention/`
-   使用 `make run-all-with-report` 或 `make run-all-train-lstm-with-report` 時，該時間戳與 `report/{時間戳}/` 相同，方便對應同一次實驗

查看訓練曲線：

```bash
tensorboard --logdir outputs
```

---

## 專案結構概覽

（僅列出與模型訓練較相關的主要檔案）

-   `machine_learning/`：傳統機器學習模型（迴歸 / 分類）
-   `deep_learning/`：深度學習時間序列模型
-   `utils/`：
    -   `generate_ticker_data.py`：股價資料下載與前處理
    -   `lineplot_true_and_predicted_result.py`：預測結果視覺化
    -   `stock_dataset.py`：時間序列 Dataset
    -   `experiment_logger.py`：實驗結果統一寫入 `experiments/results.csv`
    -   `report_saver.py`：報告輸出到 `report/{時間戳}/{category}_{model_name}/`，並提供 `get_run_id()` 供 DL 腳本共用同一時間戳
-   `config/config.json`：機器學習與深度學習訓練設定（標的、日期、超參數等）
-   `outputs/`：深度學習輸出依時間戳存放，結構為 `outputs/{時間戳}/{lstm|lstm_cnn|attention}/`（內含 `writer/`、`model_param/`、`plots/`）
-   `experiments/results.csv`：所有模型（ML + DL）的 train / valid / test 指標彙總
-   `compare_models.py`：讀取 `experiments/results.csv`，整理各模型在 test split 的表現並視覺化
-   `rolling_evaluation_random_forest.py`：以隨機森林在多個相鄰年份視窗進行 rolling evaluation
-   `simple_strategy_backtest_logistic.py`：使用邏輯迴歸預測漲跌，做簡單交易策略回測，與 buy-and-hold 比較
-   `tests/`：最小但實用的單元測試（例如 `test_generate_ticker_data.py`）
-   `Makefile`：常用指令（`run-ml-all`、`train-lstm`、`run-all-with-report`、`run-all-ml-with-report`、`run-all-train-lstm-with-report`、`compare-models`、`rolling-eval`、`simple-backtest`、`test`）

## 機器學習範例說明（皆使用同一筆 OHLCV 數據）

機器學習腳本都位於 `machine_learning/` 資料夾，統一從：

-   `config/config.json` 讀取：
    -   `TICKER`：股票代碼（例如 `0050.TW`）
    -   `TRAIN_START_DATE`、`TRAIN_END_DATE`：訓練集時間範圍
    -   `VALID_START_DATE`、`VALID_END_DATE`：驗證集時間範圍
    -   `TEST_START_DATE`、`TEST_END_DATE`：測試集時間範圍
-   `utils/generate_ticker_data.generate_ticker_data` 取得：
    -   特徵：`Open`, `High`, `Low`, `Close`, `Volume`
    -   標籤：
        -   迴歸：`Pred_Close`
        -   分類：`Pred_UpDown`

---

### `machine_learning/linear_regression.py`

-   **任務**：預測隔日收盤價（連續數值迴歸）
-   **輸入特徵**：當日的 `Open, High, Low, Close, Volume`
-   **輸出標籤**：`Pred_Close`（隔日收盤價）
-   **資料切分機制**：嚴格依 `config.json` 中的 `TRAIN_START/END_DATE`、`VALID_START/END_DATE`、`TEST_START/END_DATE` 切出三段連續時間區間，**保留時間順序**，並同時輸出 train / valid / test 的 MSE

**優點（使用同一筆數據時）：**

-   **模型可解釋性高**：線性迴歸係數易於解讀，能大致看出各特徵對隔日收盤價的線性影響
-   **訓練速度快**：對少量特徵與樣本非常輕量
-   **基準模型（Baseline）**：適合作為與其他複雜模型比較的參考基準

**缺點與注意事項：**

-   **只捕捉線性關係**：無法處理非線性或複雜時間依賴，股價通常不是純線性
-   **未考慮序列資訊**：只看「當日」特徵，不看前幾日動態
-   **同一筆數據重複實驗時**：容易高估效果，建議保留獨立測試區間，且不要頻繁調整參數以免「偷看」測試集

---

### `machine_learning/logistic_regression.py`

-   **任務**：預測隔日收盤價**漲跌方向**（1 = 上漲，0 = 下跌）
-   **輸入特徵**：當日 `Open, High, Low, Close, Volume`
-   **輸出標籤**：`Pred_UpDown`
-   **資料切分機制**：嚴格依 `config.json` 中的三段時間區間（TRAIN / VALID / TEST）切分，**保留時間順序**，並同時輸出 train / valid / test 的分類報告（accuracy、precision、recall、F1）

**優點：**

-   **輸出為機率（sigmoid）**：可取得上漲機率，方便做閾值調整與風險控管
-   **解釋力尚可**：權重可看成各特徵對「上漲機率」的線性影響
-   **適合作為分類 Baseline**：比樹模型更簡單、更穩定

**缺點與注意事項：**

-   **對共線性敏感**：特徵高度相關時，權重可能不穩定
-   **未處理 class imbalance**：若「漲」與「跌」分佈不均，指標可能偏斜（例如 precision/recall 不平衡）
-   **使用同一數據時的陷阱**：
    -   若沒有嚴格區分訓練與未來資料（例如誤用 shuffle=True），會產生「未來資訊洩漏」
    -   建議檢查分類報告中 `precision` / `recall` 是否對其中一類非常差，避免只看 accuracy

---

### `machine_learning/decision_tree.py`

-   **任務**：隔日漲跌方向分類 + 觀察決策規則
-   **輸入 / 標籤**：與邏輯迴歸相同
-   **特點**：
    -   使用 `DecisionTreeClassifier` 建模
    -   以 `plot_tree` 顯示前幾層決策規則

**優點：**

-   **規則可視化**：可直觀看到模型如何依據價量條件做分支，易於教學與解釋
-   **可捕捉非線性邊界**：比純線性模型更靈活

**缺點與注意事項：**

-   **容易 overfitting**：特別是深度過大時，訓練集表現可能極佳、測試集卻很差
-   **對同一筆數據重複調整參數時**：若一直以測試集表現調整樹深度 / 最小樣本數，實際上等於把測試集當驗證集使用，會低估真實風險
-   建議配合：
    -   限制 `max_depth` 或 `min_samples_leaf`
    -   或另外分出獨立驗證區間

---

### `machine_learning/random_forest.py`

-   **任務**：隔日漲跌方向分類 + 特徵重要性分析
-   **模型**：`RandomForestClassifier`，使用多棵樹 Bagging 降低變異

**優點：**

-   **穩定度較單棵樹高**：對資料噪音不那麼敏感
-   **內建特徵重要性**：可快速看出哪些價量特徵對漲跌影響較大
-   **適合在同一筆數據上評估多次**：相較單棵樹，隨機森林對個別資料點的微小異動較不敏感

**缺點與注意事項：**

-   **較難解釋到單一路徑層級**：整體雖然穩定，但單棵樹行為不易逐一追蹤
-   **時間序列風險仍存在**：即便模型強大，如果切分方式不保留時間順序，仍可能「偷看未來」
-   **同一筆數據多次實驗**：如果一再調參、選擇最佳結果報告，會產生「選擇性報告偏誤」，建議保留一段從未使用過的時間段作「最終檢驗」

---

### `machine_learning/xgboost_binary_classification.py`

-   **任務**：使用 XGBoost 做隔日漲跌二分類，並視覺化特徵重要性
-   **模型**：XGBoost `binary:logistic`，以梯度提升樹為基礎

**優點：**

-   **擅長處理複雜非線性與交互作用**：對於價格與成交量的複雜關係特別有用
-   **調參空間大**：可透過 `max_depth`, `eta`, `subsample` 等控制模型容量與正則化
-   **內建特徵重要性與樹結構**：可輔助解釋

**缺點與注意事項：**

-   **容易被同一組訓練資料「練到爆」**：
    -   若反覆以同一測試區間調整超參數，最終結果可能對該區間過度優化
    -   建議至少區分：訓練 / 驗證 / 測試 三段連續時間
-   **計算量與依賴較多**：
    -   需安裝 XGBoost 及 OpenMP（macOS）
    -   在資料量變大時，訓練時間與記憶體消耗也會增加

## 深度學習範例說明（皆使用同一筆序列資料）

深度學習腳本位於 `deep_learning/`，共享下列機制：

-   從 `config/config.json` 讀取：
    -   `TICKER`、各資料區間（TRAIN / VALID / TEST）
    -   `SEQ_LENGTH`：每一樣本的序列長度
    -   `BATCH_SIZE`、`LEARNING_RATE`、`EPOCHS`
-   使用 `generate_ticker_data` 取得 OHLCV，並以 `MinMaxScaler` 縮放
-   透過 `StockDataset` 將連續資料切成時間序列樣本
-   `DataLoader` 建立訓練 / 驗證 / 測試迭代器
-   損失函數皆為 **MSE**（預測下一日收盤價）
-   輸出路徑與時間戳：透過 `utils.report_saver.get_run_id()` 取得本輪 run id（可與 `REPORT_RUN_ID` 環境變數一致），模型權重與圖檔存於：
    -   `outputs/{時間戳}/lstm/`（writer、model_param、plots）
    -   `outputs/{時間戳}/lstm_cnn/`
    -   `outputs/{時間戳}/attention/`

### `deep_learning/lstm.py`

-   **模型結構**：
    -   `nn.LSTM(input_size=5, hidden_size=64, num_layers=2)`
    -   取最後一個時間步輸出，接 `nn.Linear` 映射到單一收盤價預測
-   **適用情境**：
    -   想要捕捉**長期時間依賴**（例如數十天的趨勢）

**優點：**

-   **明確處理時間順序**：每一步依賴前序狀態，適合連續時間序列
-   對於單一標的、固定時間窗的實驗，是很好的 baseline

**缺點與注意事項：**

-   **訓練時間較長**：相較純機器學習模型、或較淺的 MLP
-   **對同一筆數據反覆訓練時**：
    -   容易在訓練集 / 驗證集上「記住」特定模式
    -   建議觀察訓練 / 驗證 loss 差距，並保留真正的獨立測試段

---

### `deep_learning/lstm_cnn.py`

-   **模型結構**：
    -   先透過 LSTM 擷取序列特徵，再用 1D CNN 掃描時間維度的局部模式
    -   最後以線性層輸出單一收盤價

**優點：**

-   **結合長期與局部模式**：
    -   LSTM 負責長期依賴
    -   CNN 擷取短期形態（例如近幾日的波動樣態）
-   對於具有明顯「形狀」的價格軌跡，可能比純 LSTM 更有表現空間

**缺點與注意事項：**

-   **模型容量更大**：在同一筆資料上更容易 overfitting
-   **超參數較多**：LSTM 與 CNN 皆有多個維度需調整
-   在使用同一組序列資料時，建議：
    -   嚴格區分 TRAIN / VALID / TEST 時間段
    -   只根據 VALID 表現調整超參數，最後才看 TEST

---

### `deep_learning/attention_model.py`

-   **模型結構**：
    -   先將每個時間步的特徵透過線性層映射到注意力空間（embedding）
    -   使用 `nn.MultiheadAttention` 做 self-attention
    -   取最後時間步的表示，經線性層輸出預測值

**優點：**

-   **能對「重要時間步」給予較高權重**：
    -   例如重大跳空、異常成交量等事件，理論上可被注意力機制捕捉
-   相對於單向 LSTM，更容易擴展到多頭、多層的複雜結構

**缺點與注意事項：**

-   **更容易 overfitting**：
    -   注意力機制在小資料集上可能只是在記憶特定 pattern
-   **需要更謹慎的正則化與驗證**：
    -   建議觀察不同頭數、embedding 維度對驗證集的影響
-   在同一筆數據上反覆調整 attention 超參數，亦可能對該資料「特化」，因此：
    -   保留**完全沒參與任何調參的測試區間**非常重要

---

## 實驗建議

-   **保留時間順序**：避免打亂時間導致未來資訊洩漏，才能真實反映實際交易情境
-   **嚴格區分訓練 / 驗證 / 測試時間段**：
    -   機器學習與深度學習皆適用，並且盡量使用連續時間區間
-   **不要「看著測試結果調參」**：
    -   只用驗證集探索超參數，測試集只保留給最後一次、最保守的評估
-   **比較重點放在泛化而非神奇高分**：
    -   多次實驗時，關注結果的穩定性與在不同時間段的表現，而不是單一 run 的最高指標
-   **學習路徑建議**：
    -   從線性迴歸 / 邏輯迴歸開始 → 決策樹 / 隨機森林 / XGBoost → LSTM → LSTM+CNN → Attention，
        一步步觀察在**同一筆股價資料**下，模型複雜度提升帶來的行為與風險變化

---

## 如何看懂圖表與分析數值

本專案每個模型訓練完成後，會自動產生「真實值 vs 預測值」的折線圖和對應評估指標，協助快速理解模型表現：

-   **折線圖**（report 目錄下的 PNG 檔）：

    -   X 軸為交易日（時間），Y 軸為預測目標（如隔日收盤價、漲跌標籤）
    -   深色實線通常代表真實值（True），虛線或另一顏色代表模型預測值（Predicted）
    -   若預測曲線能緊貼真實走勢，代表模型有良好擬合能力
    -   若預測常延遲、放大或過度平滑，代表模型尚有改進空間

-   **回歸任務數值指標（如線性回歸等）：**

    -   主要看 **均方誤差（MSE）**，愈小代表預測愈準確
    -   通常會列出訓練集、驗證集、測試集的 MSE
    -   留意測試集 MSE 是否低於驗證集（若測試高很多，代表過擬合）

-   **分類任務數值指標（如邏輯迴歸、決策樹等）：**

    -   會輸出 **正確率（accuracy）**、**精確率（precision）**、**召回率（recall）**、**F1 分數** 等報告
    -   注意不要只看 accuracy，需檢查兩類（上漲/下跌）各自的 precision/recall 表現，有無偏斜
    -   以 macro F1-score 綜合評估泛化能力

-   **不同模型比較方式：**

    -   本專案所有模型訓練/驗證/測試區間分割一致，數值可直接跨模型比較
    -   建議「以測試集指標為主」，評估真實未來預測能力，而非僅看訓練與驗證集

-   **多次實驗或不同設定下：**
    -   可利用 `experiments/results.csv` 匯總不同模型/參數/資料分割結果

> **建議：預測股價等金融資料時，不能單只追求單一指標高分，更要看模型是否穩定、結果能否泛化到未來未見資料折線圖與多數值指標搭配才能全面理解模型的行為與局限**
