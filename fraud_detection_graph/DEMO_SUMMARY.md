
# FRAUD DETECTION SYSTEM - DEMO COMPLETE! 🎉

## What We Built:
✅ Graph-based fraud detection system
✅ Deep learning models (GraphSAGE, GAT)  
✅ Community detection for fraud rings
✅ Interactive visualizations
✅ REST API with comprehensive endpoints
✅ Complete testing suite

## Key Files Created:
- `data/`: Synthetic fraud dataset (50K transactions)
- `models/`: GNN models and training code
- `visualizations/`: Interactive fraud analysis plots
- `api/`: REST API server
- `tests/`: Comprehensive testing scripts

## How to Use:

### 1. Start the API Server:
```bash
cd /workspace/fraud_detection_graph
python3 start_api.py
```

### 2. Test with CURL:
```bash
./demo_api.sh
```

### 3. View Visualizations:
Open: visualizations/fraud_analysis_report.html

### 4. Run Prediction Demo:
```bash
python3 prediction_demo.py
```

## API Endpoints:
- POST /predict - Predict transaction fraud
- POST /detect_rings - Detect fraud rings  
- GET /entity_profile/<type>/<id> - Entity risk profiles
- GET /stats - System statistics
- GET /health - Health check

## Sample CURL Commands:

### Predict Fraud:
```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "card_id": "card_000001",
    "merchant_id": "merchant_00001",
    "device_id": "device_000001",
    "ip_address": "192.168.1.1", 
    "amount": 100.00
  }'
```

### Get Risk Profile:
```bash
curl -X GET "http://localhost:5000/entity_profile/card/card_000001"
```

### Detect Fraud Rings:
```bash
curl -X POST "http://localhost:5000/detect_rings" \
  -H "Content-Type: application/json" \
  -d '{"min_ring_size": 3}'
```

## Project Features:
🔍 Real-time fraud scoring
📊 Interactive heatmap visualizations  
🕵️ Fraud ring detection
📈 Graph neural networks
🌐 REST API with web interface
🧪 Comprehensive test suite

## Ready for Presentation! 🚀
