.PHONY: setup download-data train backtest dashboard test lint clean deploy-dev

# ============================================================
# Development Environment
# ============================================================

setup:
	pip install -e ".[dev]"
	pre-commit install || true
	mkdir -p data/raw data/processed data/models
	@echo "✅ Setup complete!"

# ============================================================
# Data Pipeline
# ============================================================

download-data:
	python scripts/download_data.py \
		--symbol BTC/USDT \
		--timeframes 1h 4h 1d \
		--since 2024-01-01 \
		--output data/raw/
	@echo "✅ Data downloaded!"

features:
	python -m src.data.features \
		--input data/raw/ \
		--output data/processed/
	@echo "✅ Features generated!"

# ============================================================
# Model Training
# ============================================================

train-xgboost:
	python -m src.models.trainer \
		--model xgboost \
		--config configs/model_config.yaml \
		--data data/processed/btc_usdt_features_1h.parquet
	@echo "✅ XGBoost training complete!"

train-lstm:
	python -m src.models.trainer \
		--model lstm \
		--config configs/model_config.yaml \
		--data data/processed/btc_usdt_features_1h.parquet
	@echo "✅ LSTM training complete!"

train: train-xgboost train-lstm
	python -m src.models.ensemble --train
	@echo "✅ All models trained!"

# ============================================================
# Backtesting
# ============================================================

backtest:
	python -m src.backtest.engine \
		--config configs/trading_config.yaml \
		--data data/processed/btc_usdt_features_1h.parquet \
		--output data/models/backtest_results.json
	@echo "✅ Backtest complete!"

walk-forward:
	python -m src.backtest.engine \
		--mode walk-forward \
		--config configs/trading_config.yaml
	@echo "✅ Walk-forward validation complete!"

# ============================================================
# Dashboard
# ============================================================

dashboard:
	streamlit run src/dashboard/app.py --server.port 8501

# ============================================================
# Testing & Quality
# ============================================================

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

# ============================================================
# AWS Deployment
# ============================================================

deploy-dev:
	cd infra && npx cdk deploy --all \
		--context env=dev \
		--require-approval never

deploy-staging:
	cd infra && npx cdk deploy --all \
		--context env=staging

deploy-prod:
	cd infra && npx cdk deploy --all \
		--context env=prod

destroy:
	cd infra && npx cdk destroy --all --force
	@echo "⚠️  All stacks destroyed!"

# ============================================================
# Utilities
# ============================================================

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned!"

costs:
	@echo "Checking AWS costs..."
	aws ce get-cost-and-usage \
		--time-period Start=$$(date -d '30 days ago' +%Y-%m-%d),End=$$(date +%Y-%m-%d) \
		--granularity MONTHLY \
		--metrics BlendedCost \
		--group-by Type=DIMENSION,Key=SERVICE \
		--output table
