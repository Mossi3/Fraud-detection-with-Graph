## Graph-based Credit Card Fraud Detection (GNN + Heatmap)

Simple end-to-end school project to detect fraud rings and collusion using a heterogeneous graph of cards, merchants, devices, and IPs. Includes:

- Graph construction from transactions (cards/merchants/devices/IPs)
- Deep learning GNN (GraphSAGE-like) link predictor for fraud risk
- Community detection (Louvain) to find fraud rings
- Metrics (PR-AUC, cluster purity)
- Visualizations (ring graph export, fraud heatmap)
- FastAPI service with cURL examples

### 1) Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Generate mock dataset:

```bash
python scripts/generate_mock_data.py --out data/mock_transactions.csv --n_tx 4000
```

Quick train on mock data and produce artifacts:

```bash
python -m src.fraud_graph.train --data data/mock_transactions.csv --epochs 8 --threshold 0.55
```

Run API server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2) API cURL examples

- Health

```bash
curl -s http://localhost:8000/health | jq
```

- Train (reads `data/mock_transactions.csv` by default)

```bash
curl -s -X POST http://localhost:8000/train -H 'Content-Type: application/json' \
  -d '{"data_path":"data/mock_transactions.csv","epochs":8,"threshold":0.55}' | jq
```

- Predict fraud risk for a new transaction

```bash
curl -s -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "transaction_id": "tx_demo_1",
  "card_id": "card_12",
  "merchant_id": "merchant_7",
  "device_id": "device_9",
  "ip_address": "192.168.1.88",
  "amount": 87.20,
  "timestamp": "2025-01-01T12:34:56"
}' | jq
```

- Metrics (PR-AUC, cluster purity)

```bash
curl -s http://localhost:8000/metrics | jq
```

- Top clusters (detected rings)

```bash
curl -s http://localhost:8000/clusters | jq
```

- Heatmap of a detected ring (PNG)

```bash
curl -s 'http://localhost:8000/heatmap?cluster_id=0' -o heatmap_top_cluster.png
```

- Graph export of a detected ring (JSON for viz tools)

```bash
curl -s 'http://localhost:8000/graph?cluster_id=0' | jq | tee top_cluster_graph.json
```

### 3) Data

- Default: synthetic `data/mock_transactions.csv` created by `scripts/generate_mock_data.py`.
- Optional: You can adapt loaders to any Kaggle dataset with fields resembling: `card_id`, `merchant_id`, `device_id`, `ip_address`, `amount`, `timestamp`, `is_fraud`.

### 4) Project Layout

```
.
├── app/
│   └── main.py                # FastAPI service
├── artifacts/                 # Saved model, mappings, figures
├── data/
│   └── mock_transactions.csv  # Synthetic dataset (generated)
├── scripts/
│   └── generate_mock_data.py  # Creates synthetic fraud rings
├── src/
│   └── fraud_graph/
│       ├── data.py            # Load/split utilities
│       ├── graph.py           # Node indexing and adjacency
│       ├── model.py           # GraphSAGE-like GNN and trainer
│       ├── community.py       # Louvain rings + purity
│       ├── metrics.py         # PR-AUC and helpers
│       ├── viz.py             # Heatmap and graph export
│       └── train.py           # CLI training entrypoint
└── requirements.txt
```

### 5) Notes

- The GNN uses learnable entity embeddings and message passing over the heterogeneous graph to produce node representations. Fraud risk for a transaction is scored on the `(card, merchant)` pair via a dot-product classifier.
- Louvain clustering is run on a subgraph filtered by predicted fraud probabilities to reveal rings/collusive communities.
- Heatmap shows card–merchant risk intensities for a selected ring, which is great for presentations.

