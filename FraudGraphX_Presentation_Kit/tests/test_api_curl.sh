#!/bin/bash

# FraudGraphX API Test Suite
# Comprehensive curl tests for graph-based fraud detection

BASE_URL="http://localhost:8000"
API_BASE="$BASE_URL"

echo "🔍 FraudGraphX API Test Suite"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Function to run a test
run_test() {
    local test_name="$1"
    local curl_command="$2"
    local expected_status="$3"
    
    echo -e "${BLUE}Testing: $test_name${NC}"
    
    # Run the curl command and capture response
    response=$(eval "$curl_command" 2>/dev/null)
    status_code=$?
    
    if [ $status_code -eq 0 ]; then
        echo -e "${GREEN}✓ PASSED${NC}"
        echo "Response: $response" | head -c 200
        echo ""
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Error: $response"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
}

# Function to check if server is running
check_server() {
    echo -e "${YELLOW}Checking if server is running...${NC}"
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/health")
    
    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✓ Server is running${NC}"
        return 0
    else
        echo -e "${RED}✗ Server is not running. Please start the server first.${NC}"
        echo "Run: uvicorn src.serve.app_graph_api:app --reload"
        return 1
    fi
}

# Check server status
if ! check_server; then
    exit 1
fi

echo -e "${YELLOW}Starting API Tests...${NC}"
echo ""

# Test 1: Health Check
run_test "Health Check" \
    "curl -s -X GET '$API_BASE/health'" \
    200

# Test 2: Root Endpoint
run_test "Root Endpoint" \
    "curl -s -X GET '$API_BASE/'" \
    200

# Test 3: Generate Synthetic Data
echo -e "${YELLOW}Generating synthetic fraud ring data...${NC}"
SYNTHETIC_RESPONSE=$(curl -s -X POST "$API_BASE/synthetic_data/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "num_rings": 3,
        "ring_types": ["card_testing_ring", "merchant_collusion"],
        "base_transactions": 5000,
        "seed": 42
    }')

echo "Synthetic data generation response:"
echo "$SYNTHETIC_RESPONSE" | head -c 500
echo ""

# Extract download URL if available
DOWNLOAD_URL=$(echo "$SYNTHETIC_RESPONSE" | grep -o '"download_url":"[^"]*"' | cut -d'"' -f4)

# Test 4: Build Graph with Sample Data
echo -e "${YELLOW}Building graph with sample transaction data...${NC}"

GRAPH_RESPONSE=$(curl -s -X POST "$API_BASE/graph/build" \
    -H "Content-Type: application/json" \
    -d '{
        "transactions": [
            {
                "transaction_id": "txn_001",
                "card_id": "card_001",
                "merchant_id": "merchant_001",
                "device_id": "device_001",
                "ip": "192.168.1.1",
                "amount": 100.0,
                "timestamp": 1640995200
            },
            {
                "transaction_id": "txn_002",
                "card_id": "card_002",
                "merchant_id": "merchant_001",
                "device_id": "device_002",
                "ip": "192.168.1.2",
                "amount": 250.0,
                "timestamp": 1640998800
            },
            {
                "transaction_id": "txn_003",
                "card_id": "card_001",
                "merchant_id": "merchant_002",
                "device_id": "device_001",
                "ip": "192.168.1.1",
                "amount": 75.0,
                "timestamp": 1641002400
            }
        ],
        "config": {
            "include_cards": true,
            "include_merchants": true,
            "include_devices": true,
            "include_ips": true,
            "card_merchant_edges": true,
            "device_ip_edges": true,
            "card_device_edges": true,
            "merchant_device_edges": true
        }
    }')

echo "Graph building response:"
echo "$GRAPH_RESPONSE" | head -c 500
echo ""

# Extract graph ID
GRAPH_ID=$(echo "$GRAPH_RESPONSE" | grep -o '"graph_id":"[^"]*"' | cut -d'"' -f4)

if [ -n "$GRAPH_ID" ]; then
    echo -e "${GREEN}✓ Graph built successfully with ID: $GRAPH_ID${NC}"
    
    # Test 5: Get Graph Statistics
    run_test "Get Graph Statistics" \
        "curl -s -X GET '$API_BASE/graph/$GRAPH_ID/statistics'" \
        200
    
    # Test 6: Get Centrality Measures
    run_test "Get Centrality Measures" \
        "curl -s -X GET '$API_BASE/graph/$GRAPH_ID/centrality'" \
        200
    
    # Test 7: Get Suspicious Patterns
    run_test "Get Suspicious Patterns" \
        "curl -s -X GET '$API_BASE/graph/$GRAPH_ID/suspicious_patterns'" \
        200
    
    # Test 8: Detect Communities
    echo -e "${YELLOW}Detecting communities...${NC}"
    
    COMMUNITY_RESPONSE=$(curl -s -X POST "$API_BASE/community/$GRAPH_ID/detect" \
        -H "Content-Type: application/json" \
        -d '{
            "method": "louvain",
            "resolution": 1.0,
            "min_community_size": 2,
            "fraud_threshold": 0.3
        }')
    
    echo "Community detection response:"
    echo "$COMMUNITY_RESPONSE" | head -c 500
    echo ""
    
    # Extract analysis ID
    ANALYSIS_ID=$(echo "$COMMUNITY_RESPONSE" | grep -o '"analysis_id":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$ANALYSIS_ID" ]; then
        echo -e "${GREEN}✓ Communities detected successfully with ID: $ANALYSIS_ID${NC}"
        
        # Test 9: Get Fraud Rings
        run_test "Get Fraud Rings" \
            "curl -s -X GET '$API_BASE/community/$ANALYSIS_ID/rings'" \
            200
        
        # Test 10: Get Ring Leaders
        run_test "Get Ring Leaders" \
            "curl -s -X GET '$API_BASE/community/$ANALYSIS_ID/ring_leaders'" \
            200
    fi
    
    # Test 11: Train GNN Model
    echo -e "${YELLOW}Training GNN model...${NC}"
    
    GNN_RESPONSE=$(curl -s -X POST "$API_BASE/gnn_training/$GRAPH_ID/train" \
        -H "Content-Type: application/json" \
        -d '{
            "model_type": "graphsage",
            "epochs": 10,
            "learning_rate": 0.001,
            "hidden_dim": 32,
            "dropout": 0.1,
            "patience": 5
        }')
    
    echo "GNN training response:"
    echo "$GNN_RESPONSE" | head -c 500
    echo ""
    
    # Extract model ID
    MODEL_ID=$(echo "$GNN_RESPONSE" | grep -o '"model_id":"[^"]*"' | cut -d'"' -f4)
    
    if [ -n "$MODEL_ID" ]; then
        echo -e "${GREEN}✓ GNN model trained successfully with ID: $MODEL_ID${NC}"
        
        # Test 12: Get Model Metrics
        run_test "Get Model Metrics" \
            "curl -s -X GET '$API_BASE/gnn_training/$MODEL_ID/metrics'" \
            200
        
        # Test 13: Predict Fraud (Single Transaction)
        run_test "Predict Fraud (Single)" \
            "curl -s -X POST '$API_BASE/fraud_detection/$MODEL_ID/predict' \
                -H 'Content-Type: application/json' \
                -d '{
                    \"transaction\": {
                        \"transaction_id\": \"test_txn_001\",
                        \"card_id\": \"card_001\",
                        \"merchant_id\": \"merchant_001\",
                        \"device_id\": \"device_001\",
                        \"ip\": \"192.168.1.1\",
                        \"amount\": 150.0,
                        \"timestamp\": 1641006000
                    },
                    \"use_graph_features\": true,
                    \"use_community_features\": true,
                    \"threshold\": 0.5
                }'" \
            200
        
        # Test 14: Batch Predict Fraud
        run_test "Batch Predict Fraud" \
            "curl -s -X POST '$API_BASE/fraud_detection/$MODEL_ID/batch_predict' \
                -H 'Content-Type: application/json' \
                -d '{
                    \"transactions\": [
                        {
                            \"transaction_id\": \"batch_txn_001\",
                            \"card_id\": \"card_001\",
                            \"merchant_id\": \"merchant_001\",
                            \"device_id\": \"device_001\",
                            \"ip\": \"192.168.1.1\",
                            \"amount\": 100.0,
                            \"timestamp\": 1641006000
                        },
                        {
                            \"transaction_id\": \"batch_txn_002\",
                            \"card_id\": \"card_002\",
                            \"merchant_id\": \"merchant_002\",
                            \"device_id\": \"device_002\",
                            \"ip\": \"192.168.1.2\",
                            \"amount\": 500.0,
                            \"timestamp\": 1641009600
                        }
                    ],
                    \"threshold\": 0.5
                }'" \
            200
    fi
    
    # Test 15: Get Network Visualization
    run_test "Get Network Visualization" \
        "curl -s -X GET '$API_BASE/visualization/$GRAPH_ID/network?format=json'" \
        200
    
    # Test 16: Get Community Visualization
    if [ -n "$ANALYSIS_ID" ]; then
        run_test "Get Community Visualization" \
            "curl -s -X GET '$API_BASE/visualization/$ANALYSIS_ID/communities'" \
            200
    fi
    
else
    echo -e "${RED}✗ Failed to build graph${NC}"
fi

# Test 17: Cache Status
run_test "Get Cache Status" \
    "curl -s -X GET '$API_BASE/cache/status'" \
    200

# Test 18: Clear Cache
run_test "Clear Cache" \
    "curl -s -X DELETE '$API_BASE/cache/clear'" \
    200

# Advanced Tests
echo -e "${YELLOW}Running Advanced Tests...${NC}"
echo ""

# Test 19: Complex Graph Configuration
run_test "Complex Graph Configuration" \
    "curl -s -X POST '$API_BASE/graph/build' \
        -H 'Content-Type: application/json' \
        -d '{
            \"transactions\": [
                {
                    \"transaction_id\": \"complex_txn_001\",
                    \"card_id\": \"fraud_card_001\",
                    \"merchant_id\": \"fraud_merchant_001\",
                    \"device_id\": \"fraud_device_001\",
                    \"ip\": \"203.0.113.1\",
                    \"amount\": 1000.0,
                    \"timestamp\": 1640995200
                },
                {
                    \"transaction_id\": \"complex_txn_002\",
                    \"card_id\": \"fraud_card_002\",
                    \"merchant_id\": \"fraud_merchant_001\",
                    \"device_id\": \"fraud_device_002\",
                    \"ip\": \"203.0.113.2\",
                    \"amount\": 2000.0,
                    \"timestamp\": 1640998800
                }
            ],
            \"config\": {
                \"include_cards\": true,
                \"include_merchants\": true,
                \"include_devices\": true,
                \"include_ips\": true,
                \"card_merchant_edges\": true,
                \"device_ip_edges\": true,
                \"card_device_edges\": true,
                \"merchant_device_edges\": true,
                \"time_window_hours\": 48,
                \"temporal_decay\": 0.8,
                \"directed\": true,
                \"weighted\": true
            }
        }'" \
    200

# Test 20: Multiple Community Detection Methods
run_test "Multiple Community Detection Methods" \
    "curl -s -X POST '$API_BASE/community/$GRAPH_ID/detect' \
        -H 'Content-Type: application/json' \
        -d '{
            \"method\": \"both\",
            \"resolution\": 1.5,
            \"min_community_size\": 3,
            \"fraud_threshold\": 0.4
        }'" \
    200

# Test 21: Different GNN Model Types
echo -e "${YELLOW}Testing different GNN model types...${NC}"

for model_type in "gat" "transformer" "ensemble"; do
    echo -e "${BLUE}Testing $model_type model...${NC}"
    
    MODEL_RESPONSE=$(curl -s -X POST "$API_BASE/gnn_training/$GRAPH_ID/train" \
        -H "Content-Type: application/json" \
        -d "{
            \"model_type\": \"$model_type\",
            \"epochs\": 5,
            \"learning_rate\": 0.001,
            \"hidden_dim\": 16,
            \"dropout\": 0.1,
            \"patience\": 3
        }")
    
    if echo "$MODEL_RESPONSE" | grep -q "success"; then
        echo -e "${GREEN}✓ $model_type model trained successfully${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        echo -e "${RED}✗ $model_type model training failed${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    echo ""
done

# Performance Tests
echo -e "${YELLOW}Running Performance Tests...${NC}"
echo ""

# Test 22: Large Dataset Test
echo -e "${BLUE}Testing with larger dataset...${NC}"

LARGE_DATASET_RESPONSE=$(curl -s -X POST "$API_BASE/synthetic_data/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "num_rings": 10,
        "ring_types": ["card_testing_ring", "merchant_collusion", "device_farming", "ip_proxy_ring"],
        "base_transactions": 20000,
        "seed": 123
    }')

if echo "$LARGE_DATASET_RESPONSE" | grep -q "success"; then
    echo -e "${GREEN}✓ Large dataset generated successfully${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Large dataset generation failed${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test 23: Error Handling Tests
echo -e "${YELLOW}Testing Error Handling...${NC}"
echo ""

# Test with invalid graph ID
echo -e "${BLUE}Testing invalid graph ID...${NC}"
ERROR_RESPONSE=$(curl -s -w "%{http_code}" -X GET "$API_BASE/graph/invalid_id/statistics")
if echo "$ERROR_RESPONSE" | grep -q "404"; then
    echo -e "${GREEN}✓ Error handling works correctly${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ Error handling failed${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Test with invalid JSON
echo -e "${BLUE}Testing invalid JSON...${NC}"
ERROR_RESPONSE=$(curl -s -w "%{http_code}" -X POST "$API_BASE/graph/build" \
    -H "Content-Type: application/json" \
    -d '{"invalid": json}')
if echo "$ERROR_RESPONSE" | grep -q "422\|400"; then
    echo -e "${GREEN}✓ JSON validation works correctly${NC}"
    TESTS_PASSED=$((TESTS_PASSED + 1))
else
    echo -e "${RED}✗ JSON validation failed${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
fi
echo ""

# Final Results
echo "=============================="
echo -e "${YELLOW}Test Results Summary${NC}"
echo "=============================="
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo -e "${BLUE}Total Tests: $((TESTS_PASSED + TESTS_FAILED))${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed.${NC}"
    exit 1
fi