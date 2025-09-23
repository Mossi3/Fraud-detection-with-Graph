# Credit Card Fraud Detection API - Curl Commands for Testing

## Base URL
```
http://localhost:5000
```

## 1. Health Check
```bash
curl -X GET http://localhost:5000/api/health
```

## 2. Get Statistics
```bash
curl -X GET http://localhost:5000/api/statistics
```

## 3. Single Transaction Fraud Detection

### Test with Fraud Ring Transaction (High Risk)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "card_id": "card_001000",
    "merchant_id": "merchant_0200",
    "device_id": "device_005000",
    "ip_address": "192.168.1.1",
    "amount": 1500.00,
    "category": "electronics",
    "country": "US"
  }'
```

### Test with Normal Transaction (Low Risk)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "card_id": "card_15000",
    "merchant_id": "merchant_1500",
    "device_id": "device_15000",
    "ip_address": "203.1.1.1",
    "amount": 25.50,
    "category": "groceries",
    "country": "US"
  }'
```

### Test with Suspicious Amount (Medium Risk)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "card_id": "card_12000",
    "merchant_id": "merchant_1200",
    "device_id": "device_12000",
    "ip_address": "203.2.2.2",
    "amount": 2500.00,
    "category": "jewelry",
    "country": "CA"
  }'
```

## 4. Batch Transaction Detection
```bash
curl -X POST http://localhost:5000/api/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "card_id": "card_001001",
        "merchant_id": "merchant_0201",
        "device_id": "device_005001",
        "ip_address": "192.168.1.2",
        "amount": 2000.00,
        "category": "electronics",
        "country": "US"
      },
      {
        "card_id": "card_16000",
        "merchant_id": "merchant_1600",
        "device_id": "device_16000",
        "ip_address": "203.3.3.3",
        "amount": 45.75,
        "category": "gas",
        "country": "US"
      },
      {
        "card_id": "card_002000",
        "merchant_id": "merchant_0300",
        "device_id": "device_006000",
        "ip_address": "10.0.1.1",
        "amount": 3000.00,
        "category": "cash_advance",
        "country": "UK"
      }
    ]
  }'
```

## 5. Get Fraud Rings Information
```bash
curl -X GET http://localhost:5000/api/fraud_rings
```

## 6. Get Heatmap Visualizations
```bash
curl -X GET http://localhost:5000/api/heatmaps
```

## 7. Get Graph Statistics
```bash
curl -X GET http://localhost:5000/api/graph_stats
```

## 8. Get Sample Transactions
```bash
curl -X GET http://localhost:5000/api/sample_transactions
```

## 9. Get Entity Information

### Get Card Information
```bash
curl -X GET http://localhost:5000/api/entity/card/card_001000
```

### Get Merchant Information
```bash
curl -X GET http://localhost:5000/api/entity/merchant/merchant_0200
```

### Get Device Information
```bash
curl -X GET http://localhost:5000/api/entity/device/device_005000
```

### Get IP Information
```bash
curl -X GET http://localhost:5000/api/entity/ip/192.168.1.1
```

## 10. Presentation Demo Commands

### Demo 1: Show System Health
```bash
echo "=== Fraud Detection System Health ==="
curl -s http://localhost:5000/api/health | python3 -m json.tool
```

### Demo 2: Show Overall Statistics
```bash
echo "=== System Statistics ==="
curl -s http://localhost:5000/api/statistics | python3 -m json.tool
```

### Demo 3: Detect Fraud Ring Transaction
```bash
echo "=== Detecting Fraud Ring Transaction ==="
curl -s -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "card_id": "card_001000",
    "merchant_id": "merchant_0200",
    "device_id": "device_005000",
    "ip_address": "192.168.1.1",
    "amount": 1500.00,
    "category": "electronics",
    "country": "US"
  }' | python3 -m json.tool
```

### Demo 4: Show Detected Fraud Rings
```bash
echo "=== Detected Fraud Rings ==="
curl -s http://localhost:5000/api/fraud_rings | python3 -m json.tool
```

### Demo 5: Show Graph Statistics
```bash
echo "=== Graph-Based Analysis ==="
curl -s http://localhost:5000/api/graph_stats | python3 -m json.tool
```

## 11. Performance Testing

### Test Multiple Requests
```bash
for i in {1..10}; do
  echo "Request $i:"
  curl -s -X POST http://localhost:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{
      "card_id": "card_'$(shuf -i 10000-20000 -n 1)'",
      "merchant_id": "merchant_'$(shuf -i 1000-2000 -n 1)'",
      "device_id": "device_'$(shuf -i 10000-20000 -n 1)'",
      "ip_address": "203.'$(shuf -i 1-10 -n 1)'.'$(shuf -i 1-10 -n 1)'.'$(shuf -i 1-10 -n 1)'",
      "amount": '$(shuf -i 10-1000 -n 1)'.00,
      "category": "online",
      "country": "US"
    }' | grep -o '"fraud_probability":[0-9.]*'
done
```

## 12. Error Testing

### Test Missing Fields
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "card_id": "card_001000",
    "amount": 1500.00
  }'
```

### Test Invalid Entity Type
```bash
curl -X GET http://localhost:5000/api/entity/invalid/card_001000
```

## 13. Web Interface Access
```bash
# Open the web dashboard in browser
echo "Web Dashboard: http://localhost:5000"
```

## Expected Results

### Fraud Ring Transaction (High Risk)
- Deep Learning Score: ~0.8
- Graph-Based Score: ~0.9
- Combined Score: ~0.85
- Fraud Ring: "ring_1"

### Normal Transaction (Low Risk)
- Deep Learning Score: ~0.1-0.3
- Graph-Based Score: ~0.1-0.2
- Combined Score: ~0.15
- Fraud Ring: null

### Statistics
- Total Transactions: 50,000
- Fraud Rate: ~10%
- Fraud Rings: 3
- Total Amount: ~$2M+

## Notes
- All timestamps are in ISO format
- Fraud probabilities range from 0.0 to 1.0
- Confidence scores range from 0.0 to 1.0
- The system uses both deep learning and graph-based approaches
- Fraud rings are pre-defined patterns of collusion
- Heatmaps show fraud patterns across time, amounts, and categories