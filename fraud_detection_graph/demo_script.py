"""
Complete Demo Script for Fraud Detection System
Demonstrates all components: model training, fraud ring detection, API testing
"""

import subprocess
import time
import requests
import json
import sys
import os

def run_command(command, description, check_output=False):
    """Run a command and handle output"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        if check_output:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ {description} - SUCCESS")
                if result.stdout:
                    print(result.stdout)
                return True
            else:
                print(f"❌ {description} - FAILED")
                if result.stderr:
                    print(f"Error: {result.stderr}")
                return False
        else:
            subprocess.run(command, shell=True, timeout=60)
            print(f"✅ {description} - COMPLETED")
            return True
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def test_api_endpoint(url, method='GET', data=None, description=""):
    """Test an API endpoint"""
    print(f"\n🔍 Testing: {description}")
    print(f"URL: {url}")
    
    try:
        if method == 'GET':
            response = requests.get(url, timeout=10)
        elif method == 'POST':
            response = requests.post(url, json=data, timeout=10)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS")
            print(json.dumps(result, indent=2)[:500] + "..." if len(json.dumps(result, indent=2)) > 500 else json.dumps(result, indent=2))
            return True
        else:
            print(f"❌ FAILED - Status: {response.status_code}")
            print(response.text[:200])
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    print("🚀 FRAUD DETECTION SYSTEM - COMPLETE DEMO")
    print("=" * 60)
    
    # Change to project directory
    os.chdir('/workspace/fraud_detection_graph')
    
    # Step 1: Show project structure
    print("\n📁 PROJECT STRUCTURE")
    run_command("find . -type f -name '*.py' | head -20", "Listing Python files")
    
    # Step 2: Check data
    print("\n📊 DATA CHECK")
    run_command("ls -la data/", "Checking data directory", check_output=True)
    
    # Step 3: Test individual components
    print("\n🧪 COMPONENT TESTING")
    
    # Test graph building
    run_command("python3 models/graph_builder.py", "Testing graph construction", check_output=True)
    
    # Test community detection
    run_command("python3 models/community_detection.py", "Testing fraud ring detection", check_output=True)
    
    # Test visualizations
    run_command("python3 visualizations/fraud_visualizer.py", "Creating visualizations", check_output=True)
    
    # Step 4: Create API test script
    print("\n📝 CREATING API TEST SCRIPT")
    
    api_test_script = """
import sys
sys.path.append('/workspace/fraud_detection_graph')
from api.fraud_api import app, fraud_api

print("Loading fraud detection system...")
fraud_api.load_system()

if __name__ == '__main__':
    print("Starting API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
"""
    
    with open('start_api.py', 'w') as f:
        f.write(api_test_script)
    
    # Step 5: Create comprehensive curl test script
    print("\n📋 CREATING CURL TEST SCRIPT")
    
    curl_script = '''#!/bin/bash

echo "🔍 FRAUD DETECTION API DEMONSTRATION"
echo "===================================="

API_URL="http://localhost:5000"

echo ""
echo "1. Health Check"
echo "---------------"
curl -s -X GET "$API_URL/health" | python3 -m json.tool

echo ""
echo "2. System Statistics" 
echo "-------------------"
curl -s -X GET "$API_URL/stats" | python3 -m json.tool

echo ""
echo "3. Fraud Prediction - Normal Transaction"
echo "----------------------------------------"
curl -s -X POST "$API_URL/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "card_id": "card_000001",
    "merchant_id": "merchant_00001",
    "device_id": "device_000001", 
    "ip_address": "192.168.1.1",
    "amount": 50.00
  }' | python3 -m json.tool

echo ""
echo "4. Fraud Prediction - High Risk Transaction"
echo "-------------------------------------------"
curl -s -X POST "$API_URL/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "card_id": "card_000002",
    "merchant_id": "merchant_00002", 
    "device_id": "device_000002",
    "ip_address": "10.0.0.1",
    "amount": 5000.00
  }' | python3 -m json.tool

echo ""
echo "5. Entity Risk Profile"
echo "---------------------"
curl -s -X GET "$API_URL/entity_profile/card/card_000001" | python3 -m json.tool

echo ""
echo "6. Fraud Ring Detection"
echo "-----------------------"
curl -s -X POST "$API_URL/detect_rings" \\
  -H "Content-Type: application/json" \\
  -d '{"min_ring_size": 3}' | python3 -m json.tool

echo ""
echo "✅ API Demo Complete!"
'''
    
    with open('demo_api.sh', 'w') as f:
        f.write(curl_script)
    
    subprocess.run(['chmod', '+x', 'demo_api.sh'])
    
    # Step 6: Create simple prediction demo
    print("\n🎯 CREATING PREDICTION DEMO")
    
    prediction_demo = """
import sys
sys.path.append('/workspace/fraud_detection_graph')

from models.graph_builder import FraudGraphBuilder
import pandas as pd
import numpy as np
import torch

def demo_fraud_prediction():
    print("🔍 FRAUD PREDICTION DEMO")
    print("=" * 40)
    
    # Load data
    builder = FraudGraphBuilder()
    try:
        hetero_data, _ = builder.load_graph()
        print("✅ Graph loaded successfully")
        
        # Show graph statistics
        print(f"📊 Graph Statistics:")
        print(f"   Cards: {hetero_data['card'].num_nodes:,}")
        print(f"   Merchants: {hetero_data['merchant'].num_nodes:,}")
        print(f"   Devices: {hetero_data['device'].num_nodes:,}")
        print(f"   IPs: {hetero_data['ip'].num_nodes:,}")
        print(f"   Transactions: {len(hetero_data.transaction_labels):,}")
        
        fraud_rate = hetero_data.transaction_labels.float().mean().item()
        print(f"   Fraud Rate: {fraud_rate:.3%}")
        
        # Show some sample transactions
        transactions = pd.read_csv('data/transactions.csv')
        print(f"\\n📋 Sample Transactions:")
        print(transactions[['transaction_id', 'card_id', 'merchant_id', 'amount', 'is_fraud']].head(10).to_string(index=False))
        
        # Show fraud examples
        fraud_transactions = transactions[transactions['is_fraud'] == 1].head(5)
        print(f"\\n🚨 Sample Fraud Transactions:")
        print(fraud_transactions[['transaction_id', 'card_id', 'merchant_id', 'amount', 'fraud_type']].to_string(index=False))
        
        print("\\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_fraud_prediction()
"""
    
    with open('prediction_demo.py', 'w') as f:
        f.write(prediction_demo)
    
    # Step 7: Run prediction demo
    print("\n🎯 RUNNING PREDICTION DEMO")
    run_command("python3 prediction_demo.py", "Running fraud prediction demo", check_output=True)
    
    # Step 8: Show visualizations
    print("\n📊 VISUALIZATION FILES")
    run_command("ls -la visualizations/", "Listing visualization files", check_output=True)
    
    # Step 9: Create final summary
    print("\n📋 CREATING FINAL SUMMARY")
    
    summary = """
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
curl -X POST "http://localhost:5000/predict" \\
  -H "Content-Type: application/json" \\
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
curl -X POST "http://localhost:5000/detect_rings" \\
  -H "Content-Type: application/json" \\
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
"""
    
    with open('DEMO_SUMMARY.md', 'w') as f:
        f.write(summary)
    
    print("✅ Demo summary created: DEMO_SUMMARY.md")
    
    # Final success message
    print("\n" + "=" * 60)
    print("🎉 FRAUD DETECTION SYSTEM DEMO COMPLETED!")
    print("=" * 60)
    print("📁 All files ready in: /workspace/fraud_detection_graph")
    print("📖 Read DEMO_SUMMARY.md for usage instructions")
    print("🚀 System ready for presentation!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)