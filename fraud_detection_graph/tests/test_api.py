"""
API Testing Script with CURL Commands
Comprehensive testing suite for the fraud detection API
"""

import requests
import json
import time
import random
import subprocess
from typing import Dict, List
import pandas as pd

class FraudAPITester:
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.test_results = []
        
    def wait_for_api(self, timeout: int = 60):
        """Wait for API to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("✅ API is ready!")
                    return True
            except:
                pass
            time.sleep(2)
        
        print("❌ API failed to start within timeout")
        return False
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("🔍 Testing health check...")
        
        curl_command = f"""
curl -X GET "{self.base_url}/health" \\
  -H "Content-Type: application/json"
        """
        print(f"CURL Command:\n{curl_command}")
        
        try:
            response = requests.get(f"{self.base_url}/health")
            result = {
                "test": "health_check",
                "status_code": response.status_code,
                "response": response.json(),
                "success": response.status_code == 200
            }
            self.test_results.append(result)
            print(f"✅ Health check: {result['success']}")
            return result
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return {"test": "health_check", "success": False, "error": str(e)}
    
    def test_fraud_prediction(self):
        """Test fraud prediction endpoint"""
        print("🔍 Testing fraud prediction...")
        
        # Test cases with different risk levels
        test_cases = [
            {
                "name": "Normal Transaction",
                "data": {
                    "card_id": "card_000001",
                    "merchant_id": "merchant_00001",
                    "device_id": "device_000001",
                    "ip_address": "192.168.1.1",
                    "amount": 50.00
                }
            },
            {
                "name": "High Amount Transaction",
                "data": {
                    "card_id": "card_000002",
                    "merchant_id": "merchant_00002",
                    "device_id": "device_000002",
                    "ip_address": "10.0.0.1",
                    "amount": 5000.00
                }
            },
            {
                "name": "Suspicious Pattern",
                "data": {
                    "card_id": "card_000003",
                    "merchant_id": "merchant_00003",
                    "device_id": "device_000003",
                    "ip_address": "172.16.0.1",
                    "amount": 1000.00
                }
            }
        ]
        
        results = []
        for test_case in test_cases:
            print(f"  Testing: {test_case['name']}")
            
            curl_command = f"""
curl -X POST "{self.base_url}/predict" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(test_case["data"])}'
            """
            print(f"CURL Command:\n{curl_command}")
            
            try:
                response = requests.post(f"{self.base_url}/predict", json=test_case["data"])
                result = {
                    "test": f"fraud_prediction_{test_case['name'].lower().replace(' ', '_')}",
                    "status_code": response.status_code,
                    "response": response.json(),
                    "success": response.status_code == 200 and "fraud_probability" in response.json()
                }
                results.append(result)
                self.test_results.append(result)
                
                if result["success"]:
                    fraud_prob = result["response"]["fraud_probability"]
                    risk_level = result["response"]["risk_level"]
                    print(f"  ✅ {test_case['name']}: Fraud Prob: {fraud_prob:.4f}, Risk: {risk_level}")
                else:
                    print(f"  ❌ {test_case['name']}: Failed")
                    
            except Exception as e:
                print(f"  ❌ {test_case['name']}: Error - {e}")
                results.append({"test": test_case['name'], "success": False, "error": str(e)})
        
        return results
    
    def test_fraud_ring_detection(self):
        """Test fraud ring detection endpoint"""
        print("🔍 Testing fraud ring detection...")
        
        test_data = {"min_ring_size": 3}
        
        curl_command = f"""
curl -X POST "{self.base_url}/detect_rings" \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(test_data)}'
        """
        print(f"CURL Command:\n{curl_command}")
        
        try:
            response = requests.post(f"{self.base_url}/detect_rings", json=test_data)
            result = {
                "test": "fraud_ring_detection",
                "status_code": response.status_code,
                "response": response.json(),
                "success": response.status_code == 200 and "detected_rings_count" in response.json()
            }
            self.test_results.append(result)
            
            if result["success"]:
                rings_count = result["response"]["detected_rings_count"]
                precision = result["response"]["evaluation_metrics"]["precision"]
                recall = result["response"]["evaluation_metrics"]["recall"]
                print(f"✅ Ring detection: {rings_count} rings, Precision: {precision:.3f}, Recall: {recall:.3f}")
            else:
                print(f"❌ Ring detection failed")
                
            return result
        except Exception as e:
            print(f"❌ Ring detection error: {e}")
            return {"test": "fraud_ring_detection", "success": False, "error": str(e)}
    
    def test_entity_profiles(self):
        """Test entity profile endpoint"""
        print("🔍 Testing entity profiles...")
        
        test_entities = [
            {"type": "card", "id": "card_000001"},
            {"type": "merchant", "id": "merchant_00001"},
            {"type": "device", "id": "device_000001"},
            {"type": "ip", "id": "192.168.1.1"}
        ]
        
        results = []
        for entity in test_entities:
            print(f"  Testing: {entity['type']} - {entity['id']}")
            
            curl_command = f"""
curl -X GET "{self.base_url}/entity_profile/{entity['type']}/{entity['id']}" \\
  -H "Content-Type: application/json"
            """
            print(f"CURL Command:\n{curl_command}")
            
            try:
                response = requests.get(f"{self.base_url}/entity_profile/{entity['type']}/{entity['id']}")
                result = {
                    "test": f"entity_profile_{entity['type']}",
                    "status_code": response.status_code,
                    "response": response.json(),
                    "success": response.status_code == 200 and "risk_score" in response.json()
                }
                results.append(result)
                self.test_results.append(result)
                
                if result["success"]:
                    risk_score = result["response"]["risk_score"]
                    total_txns = result["response"]["transaction_stats"]["total_transactions"]
                    fraud_rate = result["response"]["transaction_stats"]["fraud_rate"]
                    print(f"  ✅ {entity['type']}: Risk: {risk_score:.4f}, Txns: {total_txns}, Fraud Rate: {fraud_rate:.4f}")
                else:
                    print(f"  ❌ {entity['type']}: Failed")
                    
            except Exception as e:
                print(f"  ❌ {entity['type']}: Error - {e}")
                results.append({"test": f"entity_profile_{entity['type']}", "success": False, "error": str(e)})
        
        return results
    
    def test_system_stats(self):
        """Test system statistics endpoint"""
        print("🔍 Testing system statistics...")
        
        curl_command = f"""
curl -X GET "{self.base_url}/stats" \\
  -H "Content-Type: application/json"
        """
        print(f"CURL Command:\n{curl_command}")
        
        try:
            response = requests.get(f"{self.base_url}/stats")
            result = {
                "test": "system_stats",
                "status_code": response.status_code,
                "response": response.json(),
                "success": response.status_code == 200 and "graph_stats" in response.json()
            }
            self.test_results.append(result)
            
            if result["success"]:
                stats = result["response"]["graph_stats"]
                print(f"✅ Stats: {stats['card_nodes']} cards, {stats['merchant_nodes']} merchants, {stats['total_transactions']} transactions")
            else:
                print(f"❌ Stats failed")
                
            return result
        except Exception as e:
            print(f"❌ Stats error: {e}")
            return {"test": "system_stats", "success": False, "error": str(e)}
    
    def run_load_test(self, num_requests: int = 10):
        """Run load test with multiple concurrent requests"""
        print(f"🔍 Running load test with {num_requests} requests...")
        
        # Generate random test data
        test_data = []
        for i in range(num_requests):
            data = {
                "card_id": f"card_{random.randint(0, 9999):06d}",
                "merchant_id": f"merchant_{random.randint(0, 999):05d}",
                "device_id": f"device_{random.randint(0, 4999):06d}",
                "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
                "amount": round(random.uniform(10, 1000), 2)
            }
            test_data.append(data)
        
        # Run requests and measure performance
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        for i, data in enumerate(test_data):
            request_start = time.time()
            try:
                response = requests.post(f"{self.base_url}/predict", json=data, timeout=10)
                request_time = time.time() - request_start
                response_times.append(request_time)
                
                if response.status_code == 200:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    
            except Exception as e:
                failed_requests += 1
                print(f"Request {i+1} failed: {e}")
        
        total_time = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        requests_per_second = num_requests / total_time
        
        load_test_result = {
            "test": "load_test",
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "total_time": round(total_time, 2),
            "avg_response_time": round(avg_response_time, 4),
            "requests_per_second": round(requests_per_second, 2),
            "success": failed_requests == 0
        }
        
        self.test_results.append(load_test_result)
        
        print(f"✅ Load test completed:")
        print(f"  Total requests: {num_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Failed: {failed_requests}")
        print(f"  Avg response time: {avg_response_time:.4f}s")
        print(f"  Requests/second: {requests_per_second:.2f}")
        
        return load_test_result
    
    def generate_curl_script(self):
        """Generate a shell script with all CURL commands"""
        script_content = """#!/bin/bash

# Fraud Detection API Test Script
# Run this script to test all API endpoints

API_URL="http://localhost:5000"

echo "🔍 Testing Fraud Detection API..."
echo "=================================="

echo ""
echo "1. Health Check"
echo "---------------"
curl -X GET "$API_URL/health" \\
  -H "Content-Type: application/json" | jq .

echo ""
echo "2. System Statistics"
echo "-------------------"
curl -X GET "$API_URL/stats" \\
  -H "Content-Type: application/json" | jq .

echo ""
echo "3. Fraud Prediction - Normal Transaction"
echo "----------------------------------------"
curl -X POST "$API_URL/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "card_id": "card_000001",
    "merchant_id": "merchant_00001", 
    "device_id": "device_000001",
    "ip_address": "192.168.1.1",
    "amount": 50.00
  }' | jq .

echo ""
echo "4. Fraud Prediction - High Risk Transaction"
echo "-------------------------------------------"
curl -X POST "$API_URL/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "card_id": "card_000002",
    "merchant_id": "merchant_00002",
    "device_id": "device_000002", 
    "ip_address": "10.0.0.1",
    "amount": 5000.00
  }' | jq .

echo ""
echo "5. Fraud Ring Detection"
echo "-----------------------"
curl -X POST "$API_URL/detect_rings" \\
  -H "Content-Type: application/json" \\
  -d '{"min_ring_size": 3}' | jq .

echo ""
echo "6. Entity Risk Profile - Card"
echo "-----------------------------"
curl -X GET "$API_URL/entity_profile/card/card_000001" \\
  -H "Content-Type: application/json" | jq .

echo ""
echo "7. Entity Risk Profile - Merchant"
echo "---------------------------------"
curl -X GET "$API_URL/entity_profile/merchant/merchant_00001" \\
  -H "Content-Type: application/json" | jq .

echo ""
echo "8. Entity Risk Profile - Device"
echo "-------------------------------"
curl -X GET "$API_URL/entity_profile/device/device_000001" \\
  -H "Content-Type: application/json" | jq .

echo ""
echo "9. Entity Risk Profile - IP"
echo "---------------------------"
curl -X GET "$API_URL/entity_profile/ip/192.168.1.1" \\
  -H "Content-Type: application/json" | jq .

echo ""
echo "✅ API testing completed!"
"""
        
        with open('/workspace/fraud_detection_graph/tests/test_api.sh', 'w') as f:
            f.write(script_content)
        
        # Make script executable
        subprocess.run(['chmod', '+x', '/workspace/fraud_detection_graph/tests/test_api.sh'])
        
        print("✅ CURL test script generated: /workspace/fraud_detection_graph/tests/test_api.sh")
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting comprehensive API testing...")
        print("=" * 50)
        
        # Wait for API to be ready
        if not self.wait_for_api():
            print("❌ API not ready, aborting tests")
            return
        
        # Run all tests
        self.test_health_check()
        print()
        
        self.test_system_stats()
        print()
        
        self.test_fraud_prediction()
        print()
        
        self.test_fraud_ring_detection()
        print()
        
        self.test_entity_profiles()
        print()
        
        self.run_load_test(10)
        print()
        
        # Generate CURL script
        self.generate_curl_script()
        
        # Summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test results summary"""
        print("=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results if result.get('success', False))
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {successful_tests/total_tests*100:.1f}%")
        
        print("\nDetailed Results:")
        for result in self.test_results:
            status = "✅" if result.get('success', False) else "❌"
            test_name = result.get('test', 'unknown')
            print(f"  {status} {test_name}")
        
        # Save results to file
        with open('/workspace/fraud_detection_graph/tests/test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: /workspace/fraud_detection_graph/tests/test_results.json")

if __name__ == "__main__":
    tester = FraudAPITester()
    tester.run_all_tests()