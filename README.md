## Graph-based Credit Card Fraud Detection (School Project)

End-to-end, simple project to simulate and detect organized fraud rings using a heterogeneous graph of cards, merchants, devices, and IPs. Includes:

- Mock data generator with fraud rings
- Graph construction and features
- Simple GNN (GraphSAGE-style) for card fraud scoring
- Community detection (Louvain) to find rings
- Visualizations: PR curve, ring graph, heatmaps
- FastAPI service with cURL examples

### 1) Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run API
uvicorn app.api.main:app --host 0.0.0.0 --port 8000
```

### 2) Train with Mock Data via API

```bash
curl -X POST 'http://localhost:8000/train' \
  -H 'Content-Type: application/json' \
  -d '{"use_mock": true, "num_transactions": 20000, "seed": 42}'
```

### 3) Get Metrics and Rings

```bash
curl 'http://localhost:8000/metrics' | jq
curl 'http://localhost:8000/rings' | jq
```

### 4) Score New Transactions (simple demo)

```bash
curl -X POST 'http://localhost:8000/infer' \
  -H 'Content-Type: application/json' \
  -d '{
    "transactions": [
      {"transaction_id": "t_demo_1", "card_id": "c_100", "merchant_id": "m_5", "device_id": "d_9", "ip": "10.0.0.1", "amount": 129.5, "timestamp": "2025-01-01T00:00:00"},
      {"transaction_id": "t_demo_2", "card_id": "c_101", "merchant_id": "m_5", "device_id": "d_9", "ip": "10.0.0.2", "amount": 9.0, "timestamp": "2025-01-01T00:10:00"}
    ]
  }'
```

### 5) Local CLI (optional)

```bash
source .venv/bin/activate
python app/train.py --use-mock --num-transactions 20000 --seed 42
```

### Dataset Notes

- Kaggle credit card dataset (`creditcard.csv`) does not contain merchants/devices/IPs. This project uses a mock relational generator that mimics such entities and can also merge Kaggle amounts/timestamps if provided.
- To use a custom CSV with columns: `transaction_id,card_id,merchant_id,device_id,ip,amount,timestamp,label`, place it under `data/` and pass its path to the training endpoint or CLI.

### Project Layout

```
app/
  api/
    main.py
  data/
    generate_mock_data.py
    loaders.py
  graph/
    build_graph.py
    community.py
  models/
    gnn.py
  viz/
    plots.py
  train.py
artifacts/           # models, mappings, metrics, rings
data/                # input CSVs and generated mocks
outputs/             # plots and visualizations
```

### cURL Reference

- Train (mock):

```bash
curl -X POST 'http://localhost:8000/train' -H 'Content-Type: application/json' -d '{"use_mock": true}'
```

- Train (custom CSV):

```bash
curl -X POST 'http://localhost:8000/train' -H 'Content-Type: application/json' -d '{"use_mock": false, "csv_path": "data/your_relational.csv"}'
```

- Get metrics:

```bash
curl 'http://localhost:8000/metrics'
```

- Get rings:

```bash
curl 'http://localhost:8000/rings'
```

- Infer (score cards involved in posted transactions):

```bash
curl -X POST 'http://localhost:8000/infer' -H 'Content-Type: application/json' -d '{"transactions": [{"transaction_id": "t1", "card_id": "c_1", "merchant_id": "m_2", "device_id": "d_3", "ip": "10.0.0.9", "amount": 50, "timestamp": "2025-01-01T12:00:00"}]}'
```

### License

Educational use only.

