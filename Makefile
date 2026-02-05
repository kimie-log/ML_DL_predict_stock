.PHONY: run-ml-all train-lstm compare-models rolling-eval simple-backtest test run-all-with-report

run-ml-all:
	python machine_learning/linear_regression.py
	python machine_learning/logistic_regression.py
	python machine_learning/decision_tree.py
	python machine_learning/random_forest.py
	python machine_learning/xgboost_binary_classification.py

train-lstm:
	python deep_learning/lstm.py
	python deep_learning/lstm_cnn.py
	python deep_learning/attention_model.py

compare-models:
	python compare_models.py

rolling-eval:
	python rolling_evaluation_random_forest.py

simple-backtest:
	python simple_strategy_backtest_logistic.py

test:
	pytest -q

# 在同一個 REPORT_RUN_ID 底下依序跑完 ML 與 LSTM，產生單一時間戳的報告目錄
run-all-with-report:
	REPORT_RUN_ID=$$(date +%Y%m%d_%H%M%S); \
	export REPORT_RUN_ID; \
	$(MAKE) run-ml-all; \
	$(MAKE) train-lstm

# 在同一個 REPORT_RUN_ID 底下依序跑完 ML，產生單一時間戳的報告目錄
run-all-ml-with-report:
	REPORT_RUN_ID=$$(date +%Y%m%d_%H%M%S); \
	export REPORT_RUN_ID; \
	$(MAKE) run-ml-all

# 在同一個 REPORT_RUN_ID 底下依序跑完 LSTM，產生單一時間戳的報告目錄
run-all-train-lstm-with-report:
	REPORT_RUN_ID=$$(date +%Y%m%d_%H%M%S); \
	export REPORT_RUN_ID; \
	$(MAKE) train-lstm