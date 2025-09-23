# FraudGraphX — Presentation Kit (CPU)

**Dataset:** `data/raw/MOCK_DATA.csv`

## Quick Start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data.preprocess --input data/raw/MOCK_DATA.csv --out data/processed/transactions.parquet
python -m src.features.build_all --input data/processed/transactions.parquet --out data/processed/transactions_with_features.parquet
python -m src.models.autoencoder --input data/processed/transactions_with_features.parquet --out data/processed/transactions_with_features.parquet
python -m src.models.train_xgb --input data/processed/transactions_with_features.parquet --model_path models/xgb_final.joblib
python -m src.explain.shap_report --model models/xgb_final.joblib --data data/processed/transactions_with_features.parquet --out reports/
python -m src.visual.heatmap --data data/processed/transactions_with_features.parquet --out reports/
streamlit run src/visual/dashboard_streamlit.py
# optional:
uvicorn src.serve.app_fastapi:app --reload
```
