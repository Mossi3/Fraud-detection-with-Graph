#!/usr/bin/env python3
"""
Test script for Credit Card Fraud Detection API
This script provides comprehensive testing and demonstration capabilities
"""

import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime
import os
import sys

# Add project root to path
sys.path.append('/workspace')

from data_processor import FraudDataProcessor

class FraudDetectionTester:
    """Comprehensive testing class for fraud detection API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.data_processor = FraudDataProcessor()
        
    def check_api_health(self):
        """Check if API is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json()
        except Exception as e:
            return False, str(e)
    
    def generate_test_transactions(self, num_transactions=10, fraud_rate=0.1):
        """Generate test transactions"""
        transactions = []
        
        for i in range(num_transactions):
            # Generate base transaction
            transaction = {
                "Time": np.random.uniform(0, 1000),
                "V1": np.random.normal(0, 1),
                "V2": np.random.normal(0, 1),
                "V3": np.random.normal(0, 1),
                "V4": np.random.normal(0, 1),
                "V5": np.random.normal(0, 1),
                "V6": np.random.normal(0, 1),
                "V7": np.random.normal(0, 1),
                "V8": np.random.normal(0, 1),
                "V9": np.random.normal(0, 1),
                "V10": np.random.normal(0, 1),
                "V11": np.random.normal(0, 1),
                "V12": np.random.normal(0, 1),
                "V13": np.random.normal(0, 1),
                "V14": np.random.normal(0, 1),
                "V15": np.random.normal(0, 1),
                "V16": np.random.normal(0, 1),
                "V17": np.random.normal(0, 1),
                "V18": np.random.normal(0, 1),
                "V19": np.random.normal(0, 1),
                "V20": np.random.normal(0, 1),
                "V21": np.random.normal(0, 1),
                "V22": np.random.normal(0, 1),
                "V23": np.random.normal(0, 1),
                "V24": np.random.normal(0, 1),
                "V25": np.random.normal(0, 1),
                "V26": np.random.normal(0, 1),
                "V27": np.random.normal(0, 1),
                "V28": np.random.normal(0, 1),
                "Amount": np.random.exponential(50),
                "Merchant": np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail', 'pharmacy']),
                "Country": np.random.choice(['US', 'CA', 'UK', 'DE', 'FR', 'AU', 'JP']),
                "Device": np.random.choice(['mobile', 'desktop', 'tablet']),
                "IP_Country": np.random.choice(['US', 'CA', 'UK', 'DE', 'FR', 'AU', 'JP']),
                "Hour": np.random.randint(0, 24),
                "Day": np.random.randint(1, 32),
                "Card_ID": f"CARD_{np.random.randint(10000, 99999):05d}",
                "Merchant_ID": f"M_{np.random.randint(100, 999):03d}",
                "Device_ID": f"D_{np.random.randint(1000, 9999):04d}",
                "IP_Address": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            }
            
            # Make some transactions suspicious
            if np.random.random() < fraud_rate:
                transaction["Amount"] = np.random.exponential(200)  # Higher amounts
                transaction["V1"] = np.random.normal(2, 1)  # Suspicious patterns
                transaction["V2"] = np.random.normal(-2, 1)
                transaction["V3"] = np.random.normal(1.5, 1)
                transaction["V4"] = np.random.normal(-1.5, 1)
                transaction["Hour"] = np.random.choice([0, 1, 2, 3, 4, 5])  # Unusual hours
            
            transactions.append(transaction)
        
        return transactions
    
    def test_single_transaction(self, transaction=None):
        """Test single transaction detection"""
        if transaction is None:
            transaction = self.generate_test_transactions(1)[0]
        
        print(f"🔍 Testing single transaction: {transaction['Card_ID']}")
        
        try:
            response = requests.post(
                f"{self.base_url}/detect",
                json=transaction,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Transaction analyzed successfully")
                print(f"   Fraud Probability: {result['fraud_probability']:.3f}")
                print(f"   Is Fraud: {result['is_fraud']}")
                print(f"   Confidence: {result['confidence']}")
                print(f"   Risk Factors: {', '.join(result['risk_factors'])}")
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_batch_transactions(self, num_transactions=50):
        """Test batch transaction detection"""
        print(f"📦 Testing batch detection with {num_transactions} transactions")
        
        transactions = self.generate_test_transactions(num_transactions)
        
        try:
            response = requests.post(
                f"{self.base_url}/batch_detect",
                json={"transactions": transactions},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch analysis completed")
                print(f"   Total Transactions: {result['summary']['total_transactions']}")
                print(f"   Fraud Detected: {result['summary']['fraud_detected']}")
                print(f"   Fraud Rate: {result['summary']['fraud_rate']:.3f}")
                print(f"   Processing Time: {result['summary']['processing_time']:.2f}s")
                print(f"   Avg Fraud Probability: {result['summary']['average_fraud_probability']:.3f}")
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_graph_analysis(self):
        """Test graph-based fraud analysis"""
        print("🕸️ Testing graph analysis")
        
        try:
            response = requests.post(
                f"{self.base_url}/graph_analysis",
                json={
                    "analysis_type": "fraud_rings",
                    "card_ids": None,
                    "merchant_ids": None,
                    "device_ids": None,
                    "ip_addresses": None
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Graph analysis completed")
                print(f"   Risk Score: {result['risk_score']:.3f}")
                print(f"   Fraud Rings Detected: {len(result['fraud_rings'])}")
                print(f"   Suspicious Patterns: {len(result['suspicious_patterns'])}")
                print(f"   Recommendations: {len(result['recommendations'])}")
                
                if result['fraud_rings']:
                    print("   Top Fraud Rings:")
                    for i, ring in enumerate(result['fraud_rings'][:3]):
                        print(f"     Ring {i+1}: Size={ring['size']}, Fraud Rate={ring['fraud_rate']:.3f}")
                
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_heatmap_generation(self, analysis_type="fraud_patterns"):
        """Test heatmap generation"""
        print(f"📊 Testing heatmap generation: {analysis_type}")
        
        try:
            response = requests.post(
                f"{self.base_url}/heatmap",
                json={
                    "analysis_type": analysis_type,
                    "parameters": {}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Heatmap generated successfully")
                print(f"   Analysis Type: {result['analysis_type']}")
                print(f"   Heatmap URL: {result['heatmap_url']}")
                print(f"   Generated At: {result['generated_at']}")
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_system_stats(self):
        """Test system statistics"""
        print("📈 Testing system statistics")
        
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ System stats retrieved")
                print(f"   Total Transactions: {result['total_transactions_processed']}")
                print(f"   Fraud Detection Rate: {result['fraud_detection_rate']:.3f}")
                print(f"   Active Models: {', '.join(result['active_models'])}")
                print(f"   System Uptime: {result['system_uptime']}")
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def test_model_training(self):
        """Test model training"""
        print("🤖 Testing model training")
        
        try:
            response = requests.post(f"{self.base_url}/train", timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Model training initiated")
                print(f"   Status: {result['status']}")
                print(f"   Message: {result['message']}")
                return result
            else:
                print(f"❌ API Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🚀 Starting Comprehensive Fraud Detection Test Suite")
        print("=" * 60)
        
        # Check API health
        is_healthy, health_info = self.check_api_health()
        if not is_healthy:
            print(f"❌ API is not healthy: {health_info}")
            return False
        
        print(f"✅ API is healthy: {health_info}")
        print()
        
        # Test individual components
        tests = [
            ("Single Transaction", self.test_single_transaction),
            ("Batch Transactions", lambda: self.test_batch_transactions(20)),
            ("Graph Analysis", self.test_graph_analysis),
            ("Heatmap Generation", self.test_heatmap_generation),
            ("System Statistics", self.test_system_stats),
            ("Model Training", self.test_model_training)
        ]
        
        results = {}
        for test_name, test_func in tests:
            print(f"🧪 Running {test_name} test...")
            try:
                result = test_func()
                results[test_name] = result is not None
                print(f"   Result: {'✅ PASSED' if result else '❌ FAILED'}")
            except Exception as e:
                print(f"   Result: ❌ FAILED - {e}")
                results[test_name] = False
            print()
        
        # Summary
        print("📊 Test Summary")
        print("=" * 30)
        passed = sum(results.values())
        total = len(results)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {passed/total*100:.1f}%")
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")
        
        return passed == total

def generate_curl_commands():
    """Generate curl commands for API testing"""
    print("🔧 Generating CURL Commands for API Testing")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Health check
    print("1. Health Check:")
    print(f"curl -X GET {base_url}/health")
    print()
    
    # Single transaction detection
    print("2. Single Transaction Detection:")
    transaction_data = {
        "Time": 100.0,
        "V1": 1.5,
        "V2": -1.2,
        "V3": 0.8,
        "V4": -0.9,
        "V5": 0.3,
        "V6": -0.5,
        "V7": 0.7,
        "V8": -0.4,
        "V9": 0.6,
        "V10": -0.3,
        "V11": 0.4,
        "V12": -0.2,
        "V13": 0.3,
        "V14": -0.1,
        "V15": 0.2,
        "V16": -0.05,
        "V17": 0.1,
        "V18": -0.02,
        "V19": 0.05,
        "V20": -0.01,
        "V21": 0.02,
        "V22": -0.005,
        "V23": 0.01,
        "V24": -0.002,
        "V25": 0.005,
        "V26": -0.001,
        "V27": 0.002,
        "V28": -0.0005,
        "Amount": 150.0,
        "Merchant": "online",
        "Country": "US",
        "Device": "mobile",
        "IP_Country": "US",
        "Hour": 14,
        "Day": 15,
        "Card_ID": "CARD_12345",
        "Merchant_ID": "M_001",
        "Device_ID": "D_1001",
        "IP_Address": "192.168.1.100"
    }
    
    curl_cmd = f'curl -X POST {base_url}/detect -H "Content-Type: application/json" -d \'{json.dumps(transaction_data)}\''
    print(curl_cmd)
    print()
    
    # Batch transaction detection
    print("3. Batch Transaction Detection:")
    batch_data = {"transactions": [transaction_data, transaction_data]}
    curl_cmd = f'curl -X POST {base_url}/batch_detect -H "Content-Type: application/json" -d \'{json.dumps(batch_data)}\''
    print(curl_cmd)
    print()
    
    # Graph analysis
    print("4. Graph Analysis:")
    graph_data = {
        "analysis_type": "fraud_rings",
        "card_ids": None,
        "merchant_ids": None,
        "device_ids": None,
        "ip_addresses": None
    }
    curl_cmd = f'curl -X POST {base_url}/graph_analysis -H "Content-Type: application/json" -d \'{json.dumps(graph_data)}\''
    print(curl_cmd)
    print()
    
    # Heatmap generation
    print("5. Heatmap Generation:")
    heatmap_data = {
        "analysis_type": "fraud_patterns",
        "parameters": {}
    }
    curl_cmd = f'curl -X POST {base_url}/heatmap -H "Content-Type: application/json" -d \'{json.dumps(heatmap_data)}\''
    print(curl_cmd)
    print()
    
    # System statistics
    print("6. System Statistics:")
    print(f"curl -X GET {base_url}/stats")
    print()
    
    # Model training
    print("7. Model Training:")
    print(f"curl -X POST {base_url}/train")
    print()
    
    # Available models
    print("8. Available Models:")
    print(f"curl -X GET {base_url}/models")
    print()

def create_test_data_files():
    """Create test data files for demonstration"""
    print("📁 Creating Test Data Files")
    print("=" * 30)
    
    tester = FraudDetectionTester()
    
    # Generate different types of test data
    test_scenarios = [
        ("normal_transactions", 100, 0.01),  # 1% fraud rate
        ("suspicious_transactions", 50, 0.2),  # 20% fraud rate
        ("high_fraud_transactions", 30, 0.5),  # 50% fraud rate
        ("mixed_transactions", 200, 0.05)  # 5% fraud rate
    ]
    
    for scenario_name, num_transactions, fraud_rate in test_scenarios:
        print(f"Creating {scenario_name}...")
        transactions = tester.generate_test_transactions(num_transactions, fraud_rate)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(transactions)
        df['Class'] = [1 if np.random.random() < fraud_rate else 0 for _ in range(num_transactions)]
        
        filename = f"/workspace/data/{scenario_name}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Saved {filename} ({len(df)} transactions, {df['Class'].mean():.1%} fraud rate)")
    
    print()
    print("📊 Test Data Summary:")
    for scenario_name, _, _ in test_scenarios:
        df = pd.read_csv(f"/workspace/data/{scenario_name}.csv")
        print(f"  {scenario_name}: {len(df)} transactions, {df['Class'].mean():.1%} fraud rate")

def create_presentation_script():
    """Create a presentation script for demonstrations"""
    script_content = """#!/bin/bash
# Credit Card Fraud Detection System - Presentation Script
# This script demonstrates the fraud detection system capabilities

echo "🛡️ Credit Card Fraud Detection System - Live Demo"
echo "=================================================="
echo ""

# Check if API is running
echo "🔍 Checking API status..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is running"
else
    echo "❌ API is not running. Please start it with: python api.py"
    exit 1
fi

echo ""
echo "📊 System Overview:"
curl -s http://localhost:8000/stats | jq '.'

echo ""
echo "🔍 Single Transaction Analysis:"
echo "Testing a suspicious transaction..."

# Create a suspicious transaction
SUSPICIOUS_TX='{
    "Time": 100.0,
    "V1": 2.5,
    "V2": -2.1,
    "V3": 1.8,
    "V4": -1.9,
    "V5": 0.3,
    "V6": -0.5,
    "V7": 0.7,
    "V8": -0.4,
    "V9": 0.6,
    "V10": -0.3,
    "V11": 0.4,
    "V12": -0.2,
    "V13": 0.3,
    "V14": -0.1,
    "V15": 0.2,
    "V16": -0.05,
    "V17": 0.1,
    "V18": -0.02,
    "V19": 0.05,
    "V20": -0.01,
    "V21": 0.02,
    "V22": -0.005,
    "V23": 0.01,
    "V24": -0.002,
    "V25": 0.005,
    "V26": -0.001,
    "V27": 0.002,
    "V28": -0.0005,
    "Amount": 500.0,
    "Merchant": "online",
    "Country": "US",
    "Device": "mobile",
    "IP_Country": "US",
    "Hour": 2,
    "Day": 15,
    "Card_ID": "CARD_SUSPICIOUS",
    "Merchant_ID": "M_001",
    "Device_ID": "D_1001",
    "IP_Address": "192.168.1.100"
}'

curl -s -X POST http://localhost:8000/detect -H "Content-Type: application/json" -d "$SUSPICIOUS_TX" | jq '.'

echo ""
echo "📦 Batch Analysis:"
echo "Analyzing multiple transactions..."

# Create batch data
BATCH_DATA='{
    "transactions": [
        {
            "Time": 100.0,
            "V1": 0.1,
            "V2": -0.1,
            "V3": 0.2,
            "V4": -0.2,
            "V5": 0.1,
            "V6": -0.1,
            "V7": 0.2,
            "V8": -0.2,
            "V9": 0.1,
            "V10": -0.1,
            "V11": 0.2,
            "V12": -0.2,
            "V13": 0.1,
            "V14": -0.1,
            "V15": 0.2,
            "V16": -0.2,
            "V17": 0.1,
            "V18": -0.1,
            "V19": 0.2,
            "V20": -0.2,
            "V21": 0.1,
            "V22": -0.1,
            "V23": 0.2,
            "V24": -0.2,
            "V25": 0.1,
            "V26": -0.1,
            "V27": 0.2,
            "V28": -0.2,
            "Amount": 25.0,
            "Merchant": "grocery",
            "Country": "US",
            "Device": "mobile",
            "IP_Country": "US",
            "Hour": 14,
            "Day": 15,
            "Card_ID": "CARD_NORMAL",
            "Merchant_ID": "M_002",
            "Device_ID": "D_1002",
            "IP_Address": "192.168.1.101"
        },
        {
            "Time": 200.0,
            "V1": 2.0,
            "V2": -2.0,
            "V3": 1.5,
            "V4": -1.5,
            "V5": 0.3,
            "V6": -0.5,
            "V7": 0.7,
            "V8": -0.4,
            "V9": 0.6,
            "V10": -0.3,
            "V11": 0.4,
            "V12": -0.2,
            "V13": 0.3,
            "V14": -0.1,
            "V15": 0.2,
            "V16": -0.05,
            "V17": 0.1,
            "V18": -0.02,
            "V19": 0.05,
            "V20": -0.01,
            "V21": 0.02,
            "V22": -0.005,
            "V23": 0.01,
            "V24": -0.002,
            "V25": 0.005,
            "V26": -0.001,
            "V27": 0.002,
            "V28": -0.0005,
            "Amount": 300.0,
            "Merchant": "online",
            "Country": "US",
            "Device": "mobile",
            "IP_Country": "US",
            "Hour": 3,
            "Day": 15,
            "Card_ID": "CARD_SUSPICIOUS_2",
            "Merchant_ID": "M_003",
            "Device_ID": "D_1003",
            "IP_Address": "192.168.1.102"
        }
    ]
}'

curl -s -X POST http://localhost:8000/batch_detect -H "Content-Type: application/json" -d "$BATCH_DATA" | jq '.'

echo ""
echo "🕸️ Graph Analysis:"
echo "Analyzing fraud rings and patterns..."

GRAPH_DATA='{
    "analysis_type": "fraud_rings",
    "card_ids": null,
    "merchant_ids": null,
    "device_ids": null,
    "ip_addresses": null
}'

curl -s -X POST http://localhost:8000/graph_analysis -H "Content-Type: application/json" -d "$GRAPH_DATA" | jq '.'

echo ""
echo "📈 Heatmap Generation:"
echo "Generating fraud pattern heatmap..."

HEATMAP_DATA='{
    "analysis_type": "fraud_patterns",
    "parameters": {}
}'

curl -s -X POST http://localhost:8000/heatmap -H "Content-Type: application/json" -d "$HEATMAP_DATA" | jq '.'

echo ""
echo "🤖 Available Models:"
curl -s http://localhost:8000/models | jq '.'

echo ""
echo "✅ Demo completed successfully!"
echo "🌐 Web Interface: http://localhost:8501"
echo "📚 API Documentation: http://localhost:8000/docs"
"""
    
    with open("/workspace/presentation_demo.sh", "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod("/workspace/presentation_demo.sh", 0o755)
    
    print("✅ Created presentation_demo.sh")
    print("   Run with: ./presentation_demo.sh")

if __name__ == "__main__":
    print("🚀 Credit Card Fraud Detection - Test Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = FraudDetectionTester()
    
    # Run comprehensive test
    print("Running comprehensive test suite...")
    success = tester.run_comprehensive_test()
    
    print("\n" + "=" * 50)
    
    # Generate curl commands
    generate_curl_commands()
    
    # Create test data files
    create_test_data_files()
    
    # Create presentation script
    create_presentation_script()
    
    print("\n🎯 Test Suite Complete!")
    print("📁 Test data files created in /workspace/data/")
    print("🔧 CURL commands generated above")
    print("🎬 Presentation script: ./presentation_demo.sh")
    
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check API status.")