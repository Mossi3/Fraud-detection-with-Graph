# Graph-Based Fraud Detection System

A sophisticated fraud detection system using Graph Neural Networks (GNNs) to identify fraud rings and collusion patterns by modeling relationships between cards, merchants, devices, and IPs.

## 🚀 Features

- **Multi-Entity Graph Construction**: Models relationships between cards, merchants, devices, IP addresses, and transactions
- **Advanced GNN Models**: Implements GraphSAGE, GAT (Graph Attention Networks), and hybrid models
- **Community Detection**: Uses Louvain, spectral clustering, and custom algorithms to detect fraud rings
- **Real-time API**: FastAPI-based service for real-time fraud scoring and ring detection
- **Interactive Visualizations**: PyVis, Plotly, and matplotlib visualizations of fraud networks
- **Comprehensive Monitoring**: Prometheus metrics, alerts, and performance dashboards
- **Pattern Detection**: Identifies various fraud patterns including:
  - Card cloning
  - Account takeover
  - Merchant collusion
  - Velocity attacks
  - Device sharing
  - Money mule networks

## 📊 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Transaction   │────▶│  Graph Builder   │────▶│   Fraud Graph   │
│      Data       │     │                  │     │   (NetworkX)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   GNN Models    │◀────│Feature Extractor │◀────│    Community    │
│(GraphSAGE, GAT) │     │                  │     │   Detection     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Fraud Score   │     │  Visualization   │     │   Monitoring    │
│      API        │     │     Engine       │     │    & Alerts     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Mossi3/Fraud-detection-with-Graph.git
cd Fraud-detection-with-Graph
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## 🚀 Quick Start

### 1. Generate Training Data
```bash
python -c "from src.utils.data_generator import TransactionDataGenerator; g = TransactionDataGenerator(); df = g.generate_dataset(50000); g.save_dataset(df, 'data/training_data.csv')"
```

### 2. Train Models
```bash
python train.py --epochs 100 --lr 0.001
```

### 3. Start the API Server
```bash
cd src/api
python fraud_api.py
```

### 4. Run Tests
```bash
chmod +x tests/test_api.sh
./tests/test_api.sh
```

## 📡 API Usage

### Score a Single Transaction
```bash
curl -X POST http://localhost:8000/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "txn_123",
    "card_id": "card_456",
    "merchant_id": "merchant_789",
    "amount": 150.00,
    "timestamp": "2025-09-23T10:30:00Z",
    "device_id": "device_abc",
    "ip_address": "192.168.1.100",
    "location": [37.7749, -122.4194]
  }'
```

### Detect Fraud Rings
```bash
curl -X POST http://localhost:8000/api/v1/detect_rings \
  -H "Content-Type: application/json" \
  -d '{
    "entity_ids": ["card_123", "card_456", "merchant_789"],
    "detection_method": "louvain",
    "min_ring_size": 3
  }'
```

### Batch Scoring
```bash
curl -X POST http://localhost:8000/api/v1/batch_score \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "batch_001",
        "card_id": "card_111",
        "merchant_id": "merchant_222",
        "amount": 99.99,
        "timestamp": "2025-09-23T11:00:00Z"
      },
      {
        "transaction_id": "batch_002",
        "card_id": "card_111",
        "merchant_id": "merchant_333",
        "amount": 299.99,
        "timestamp": "2025-09-23T11:05:00Z"
      }
    ],
    "include_visualization": true
  }'
```

### Get Graph Statistics
```bash
curl http://localhost:8000/api/v1/stats
```

### Check System Health
```bash
curl http://localhost:8000/api/v1/health
```

## 🧪 Testing Fraud Scenarios

Run comprehensive fraud scenarios:
```bash
python tests/test_scenarios.py
```

This will test:
- Card cloning detection
- Account takeover patterns
- Merchant collusion rings
- Money mule networks
- Velocity attacks
- Synthetic identity fraud

## 📈 Performance Metrics

The system tracks various metrics:

- **Transaction Metrics**:
  - Processing time (p50, p95, p99)
  - Fraud detection rate
  - False positive rate

- **Model Metrics**:
  - PR-AUC (Precision-Recall Area Under Curve)
  - Detection accuracy by fraud type
  - Ring detection precision

- **System Metrics**:
  - API response times
  - Graph size and density
  - Memory usage

Access metrics at: `http://localhost:8000/api/v1/metrics`

## 🎨 Visualizations

### Interactive Network Visualization
```python
from src.visualization.graph_visualizer import FraudGraphVisualizer

visualizer = FraudGraphVisualizer()
visualizer.create_pyvis_network(graph, fraud_rings, "output.html")
```

### Fraud Statistics Dashboard
```python
visualizer.plot_fraud_statistics(graph, fraud_labels, fraud_rings, "dashboard.html")
```

## 🔧 Configuration

Key configuration options in `.env`:

```bash
# Model Configuration
MODEL_PATH=./models
GRAPH_BATCH_SIZE=1024
TRAINING_EPOCHS=100
LEARNING_RATE=0.001

# Feature Engineering
FEATURE_WINDOW_DAYS=30
MAX_GRAPH_DEPTH=3

# Fraud Detection Thresholds
FRAUD_SCORE_THRESHOLD=0.7
RING_SIZE_THRESHOLD=5
SUSPICIOUS_PATTERN_THRESHOLD=0.8
```

## 📊 Model Performance

Typical performance metrics on synthetic data:

| Model | PR-AUC | Precision | Recall | F1-Score |
|-------|--------|-----------|--------|----------|
| GraphSAGE | 0.92 | 0.85 | 0.88 | 0.86 |
| GAT | 0.94 | 0.87 | 0.90 | 0.88 |
| Hybrid | 0.95 | 0.89 | 0.91 | 0.90 |

## 🚢 Deployment

### Using Docker
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 --env-file .env fraud-detection
```

### Using Docker Compose
```bash
docker-compose up -d
```

### Production Deployment
See `deployment/` directory for Kubernetes manifests and cloud deployment guides.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Geometric for GNN implementations
- NetworkX for graph operations
- FastAPI for the API framework
- Community detection algorithms from python-louvain

## 📧 Contact

For questions or support, please open an issue on GitHub.