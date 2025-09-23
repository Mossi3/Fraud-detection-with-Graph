#!/bin/bash

# Comprehensive curl tests for FraudGraphX Advanced API
# Run this script to test all API endpoints

set -e  # Exit on error

# Configuration
API_BASE_URL="http://localhost:8000"
TEMP_DIR="/tmp/fraudgraphx_tests"
mkdir -p $TEMP_DIR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_response() {
    local response_file=$1
    local expected_status=$2
    local test_name=$3
    
    if [ -f "$response_file" ]; then
        local status_code=$(grep "HTTP/" "$response_file" | tail -1 | awk '{print $2}')
        if [ "$status_code" = "$expected_status" ]; then
            log_success "$test_name - Status: $status_code"
            return 0
        else
            log_error "$test_name - Expected: $expected_status, Got: $status_code"
            cat "$response_file"
            return 1
        fi
    else
        log_error "$test_name - No response file found"
        return 1
    fi
}

# Start tests
log_info "Starting FraudGraphX API Tests"
log_info "API Base URL: $API_BASE_URL"
echo "=================================================="

# Test 1: Health Check
log_info "Test 1: Health Check"
curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X GET "$API_BASE_URL/health" \
     -H "Content-Type: application/json" \
     > "$TEMP_DIR/health_response.txt" 2>&1

if check_response "$TEMP_DIR/health_response.txt" "200" "Health Check"; then
    echo "Response:"
    grep -v "HTTP/" "$TEMP_DIR/health_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/health_response.txt"
fi
echo ""

# Test 2: Single Transaction Prediction
log_info "Test 2: Single Transaction Prediction"
cat > "$TEMP_DIR/single_transaction.json" << 'EOF'
{
    "transaction_id": "txn_001",
    "card_id": "card_12345",
    "merchant_id": "merchant_67890",
    "device_id": "device_abcde",
    "ip": "192.168.1.100",
    "amount": 1500.50,
    "transaction_type": "purchase",
    "merchant_category": "electronics",
    "hour": 23,
    "day_of_week": 5,
    "velocity_1h": 3,
    "velocity_24h": 8,
    "amount_std_dev": 2.5,
    "location_risk_score": 0.7
}
EOF

curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X POST "$API_BASE_URL/predict/single" \
     -H "Content-Type: application/json" \
     -d @"$TEMP_DIR/single_transaction.json" \
     > "$TEMP_DIR/single_prediction_response.txt" 2>&1

if check_response "$TEMP_DIR/single_prediction_response.txt" "200" "Single Transaction Prediction"; then
    echo "Response:"
    grep -v "HTTP/" "$TEMP_DIR/single_prediction_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/single_prediction_response.txt"
fi
echo ""

# Test 3: Batch Transaction Prediction
log_info "Test 3: Batch Transaction Prediction"
cat > "$TEMP_DIR/batch_transactions.json" << 'EOF'
{
    "transactions": [
        {
            "transaction_id": "txn_002",
            "card_id": "card_11111",
            "merchant_id": "merchant_22222",
            "device_id": "device_33333",
            "ip": "192.168.1.101",
            "amount": 50.00,
            "transaction_type": "purchase",
            "merchant_category": "grocery",
            "hour": 14,
            "day_of_week": 2,
            "velocity_1h": 1,
            "velocity_24h": 3,
            "amount_std_dev": 0.5,
            "location_risk_score": 0.1
        },
        {
            "transaction_id": "txn_003",
            "card_id": "card_44444",
            "merchant_id": "merchant_55555",
            "device_id": "device_66666",
            "ip": "10.0.0.1",
            "amount": 2500.00,
            "transaction_type": "withdrawal",
            "merchant_category": "cash_advance",
            "hour": 2,
            "day_of_week": 0,
            "velocity_1h": 8,
            "velocity_24h": 15,
            "amount_std_dev": 5.0,
            "location_risk_score": 0.9
        },
        {
            "transaction_id": "txn_004",
            "card_id": "card_77777",
            "merchant_id": "merchant_88888",
            "device_id": "device_99999",
            "ip": "172.16.0.1",
            "amount": 75.25,
            "transaction_type": "purchase",
            "merchant_category": "restaurant",
            "hour": 19,
            "day_of_week": 4,
            "velocity_1h": 0,
            "velocity_24h": 2,
            "amount_std_dev": 1.0,
            "location_risk_score": 0.3
        }
    ]
}
EOF

curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X POST "$API_BASE_URL/predict/batch" \
     -H "Content-Type: application/json" \
     -d @"$TEMP_DIR/batch_transactions.json" \
     > "$TEMP_DIR/batch_prediction_response.txt" 2>&1

if check_response "$TEMP_DIR/batch_prediction_response.txt" "200" "Batch Transaction Prediction"; then
    echo "Response:"
    grep -v "HTTP/" "$TEMP_DIR/batch_prediction_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/batch_prediction_response.txt"
fi
echo ""

# Test 4: Upload Training Data
log_info "Test 4: Upload Training Data"
# Create sample CSV data
cat > "$TEMP_DIR/sample_data.csv" << 'EOF'
transaction_id,card_id,merchant_id,device_id,ip,amount,transaction_type,merchant_category,hour,day_of_week,velocity_1h,velocity_24h,amount_std_dev,location_risk_score,fraud
txn_001,card_001,merchant_001,device_001,192.168.1.1,100.50,purchase,grocery,14,1,1,3,0.5,0.2,0
txn_002,card_002,merchant_002,device_002,192.168.1.2,2500.00,withdrawal,cash_advance,2,0,5,12,3.0,0.8,1
txn_003,card_003,merchant_003,device_003,192.168.1.3,75.25,purchase,restaurant,19,4,0,1,0.2,0.1,0
txn_004,card_001,merchant_004,device_001,192.168.1.1,1200.00,purchase,electronics,23,5,3,8,2.0,0.7,1
txn_005,card_004,merchant_001,device_004,192.168.1.4,45.00,purchase,grocery,10,2,0,2,0.3,0.1,0
EOF

curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X POST "$API_BASE_URL/data/upload" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@$TEMP_DIR/sample_data.csv" \
     > "$TEMP_DIR/upload_response.txt" 2>&1

if check_response "$TEMP_DIR/upload_response.txt" "200" "Data Upload"; then
    echo "Response:"
    grep -v "HTTP/" "$TEMP_DIR/upload_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/upload_response.txt"
fi
echo ""

# Test 5: Fraud Ring Detection
log_info "Test 5: Fraud Ring Detection"
cat > "$TEMP_DIR/ring_detection.json" << 'EOF'
{
    "method": "louvain",
    "min_ring_size": 2,
    "max_ring_size": 10,
    "fraud_threshold": 0.3
}
EOF

curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X POST "$API_BASE_URL/rings/detect" \
     -H "Content-Type: application/json" \
     -d @"$TEMP_DIR/ring_detection.json" \
     > "$TEMP_DIR/ring_detection_response.txt" 2>&1

if check_response "$TEMP_DIR/ring_detection_response.txt" "200" "Ring Detection"; then
    echo "Response:"
    grep -v "HTTP/" "$TEMP_DIR/ring_detection_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/ring_detection_response.txt"
    
    # Extract ring ID for visualization test
    RING_ID=$(grep -v "HTTP/" "$TEMP_DIR/ring_detection_response.txt" | jq -r '.detected_rings | keys[0]' 2>/dev/null || echo "")
    if [ "$RING_ID" != "null" ] && [ "$RING_ID" != "" ]; then
        echo "First detected ring ID: $RING_ID"
        echo "$RING_ID" > "$TEMP_DIR/ring_id.txt"
    fi
fi
echo ""

# Test 6: Ring Visualization (if ring was detected)
if [ -f "$TEMP_DIR/ring_id.txt" ]; then
    RING_ID=$(cat "$TEMP_DIR/ring_id.txt")
    log_info "Test 6: Ring Visualization for $RING_ID"
    
    curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
         -X GET "$API_BASE_URL/rings/visualize/$RING_ID" \
         -H "Content-Type: application/json" \
         > "$TEMP_DIR/ring_viz_response.txt" 2>&1
    
    if check_response "$TEMP_DIR/ring_viz_response.txt" "200" "Ring Visualization"; then
        echo "Visualization data generated successfully"
        # Don't print full response as it's very large
        grep -v "HTTP/" "$TEMP_DIR/ring_viz_response.txt" | jq '.ring_info' 2>/dev/null || echo "Ring info extracted"
    fi
else
    log_warning "Test 6: Skipping Ring Visualization - no rings detected"
fi
echo ""

# Test 7: Model Training
log_info "Test 7: Model Training"
cat > "$TEMP_DIR/training_request.json" << 'EOF'
{
    "model_type": "graphsage",
    "hidden_dim": 64,
    "num_layers": 2,
    "learning_rate": 0.001,
    "epochs": 20,
    "early_stopping_patience": 10
}
EOF

curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X POST "$API_BASE_URL/model/train" \
     -H "Content-Type: application/json" \
     -d @"$TEMP_DIR/training_request.json" \
     > "$TEMP_DIR/training_response.txt" 2>&1

if check_response "$TEMP_DIR/training_response.txt" "200" "Model Training"; then
    echo "Response:"
    grep -v "HTTP/" "$TEMP_DIR/training_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/training_response.txt"
    
    # Extract training ID for status check
    TRAINING_ID=$(grep -v "HTTP/" "$TEMP_DIR/training_response.txt" | jq -r '.training_id' 2>/dev/null || echo "")
    if [ "$TRAINING_ID" != "null" ] && [ "$TRAINING_ID" != "" ]; then
        echo "Training ID: $TRAINING_ID"
        echo "$TRAINING_ID" > "$TEMP_DIR/training_id.txt"
    fi
fi
echo ""

# Test 8: Training Status (if training was started)
if [ -f "$TEMP_DIR/training_id.txt" ]; then
    TRAINING_ID=$(cat "$TEMP_DIR/training_id.txt")
    log_info "Test 8: Training Status for $TRAINING_ID"
    
    curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
         -X GET "$API_BASE_URL/model/training_status/$TRAINING_ID" \
         -H "Content-Type: application/json" \
         > "$TEMP_DIR/training_status_response.txt" 2>&1
    
    if check_response "$TEMP_DIR/training_status_response.txt" "200" "Training Status"; then
        echo "Response:"
        grep -v "HTTP/" "$TEMP_DIR/training_status_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/training_status_response.txt"
    fi
else
    log_warning "Test 8: Skipping Training Status - no training started"
fi
echo ""

# Test 9: Analytics Summary
log_info "Test 9: Analytics Summary"
curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X GET "$API_BASE_URL/analytics/summary" \
     -H "Content-Type: application/json" \
     > "$TEMP_DIR/analytics_response.txt" 2>&1

if check_response "$TEMP_DIR/analytics_response.txt" "200" "Analytics Summary"; then
    echo "Response:"
    grep -v "HTTP/" "$TEMP_DIR/analytics_response.txt" | jq '.' 2>/dev/null || cat "$TEMP_DIR/analytics_response.txt"
fi
echo ""

# Test 10: Export Predictions
log_info "Test 10: Export Predictions"
curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X GET "$API_BASE_URL/export/predictions" \
     -H "Content-Type: application/json" \
     -o "$TEMP_DIR/exported_predictions.csv" \
     > "$TEMP_DIR/export_response.txt" 2>&1

if check_response "$TEMP_DIR/export_response.txt" "200" "Export Predictions"; then
    echo "Predictions exported to CSV file"
    if [ -f "$TEMP_DIR/exported_predictions.csv" ]; then
        echo "First few lines of exported file:"
        head -5 "$TEMP_DIR/exported_predictions.csv"
    fi
fi
echo ""

# Test 11: Error Handling - Invalid Transaction
log_info "Test 11: Error Handling - Invalid Transaction"
cat > "$TEMP_DIR/invalid_transaction.json" << 'EOF'
{
    "transaction_id": "txn_invalid",
    "card_id": "",
    "merchant_id": "merchant_test",
    "device_id": "device_test",
    "ip": "invalid_ip",
    "amount": -100,
    "transaction_type": "invalid_type",
    "merchant_category": "test",
    "hour": 25,
    "day_of_week": 8
}
EOF

curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X POST "$API_BASE_URL/predict/single" \
     -H "Content-Type: application/json" \
     -d @"$TEMP_DIR/invalid_transaction.json" \
     > "$TEMP_DIR/invalid_response.txt" 2>&1

if check_response "$TEMP_DIR/invalid_response.txt" "422" "Invalid Transaction Error"; then
    echo "Validation error correctly returned:"
    grep -v "HTTP/" "$TEMP_DIR/invalid_response.txt" | jq '.detail' 2>/dev/null || cat "$TEMP_DIR/invalid_response.txt"
fi
echo ""

# Test 12: Non-existent Endpoint
log_info "Test 12: Non-existent Endpoint"
curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\n" \
     -X GET "$API_BASE_URL/non-existent-endpoint" \
     -H "Content-Type: application/json" \
     > "$TEMP_DIR/not_found_response.txt" 2>&1

if check_response "$TEMP_DIR/not_found_response.txt" "404" "Not Found Error"; then
    echo "404 error correctly returned"
fi
echo ""

# Performance Tests
log_info "Performance Test: Multiple Concurrent Requests"
echo "Sending 5 concurrent single prediction requests..."

for i in {1..5}; do
    (
        curl -s -w "\nHTTP/%{http_version} %{http_code} %{http_reason}\nTime: %{time_total}s\n" \
             -X POST "$API_BASE_URL/predict/single" \
             -H "Content-Type: application/json" \
             -d @"$TEMP_DIR/single_transaction.json" \
             > "$TEMP_DIR/perf_response_$i.txt" 2>&1
    ) &
done

wait  # Wait for all background jobs to complete

echo "Concurrent request results:"
for i in {1..5}; do
    if [ -f "$TEMP_DIR/perf_response_$i.txt" ]; then
        status=$(grep "HTTP/" "$TEMP_DIR/perf_response_$i.txt" | tail -1 | awk '{print $2}')
        time=$(grep "Time:" "$TEMP_DIR/perf_response_$i.txt" | awk '{print $2}')
        echo "Request $i: Status $status, Time $time"
    fi
done
echo ""

# Summary
echo "=================================================="
log_info "Test Summary"
echo ""

# Count successful tests
SUCCESS_COUNT=0
TOTAL_TESTS=12

for i in {1..12}; do
    case $i in
        1) test_file="health_response.txt"; expected="200" ;;
        2) test_file="single_prediction_response.txt"; expected="200" ;;
        3) test_file="batch_prediction_response.txt"; expected="200" ;;
        4) test_file="upload_response.txt"; expected="200" ;;
        5) test_file="ring_detection_response.txt"; expected="200" ;;
        6) test_file="ring_viz_response.txt"; expected="200" ;;
        7) test_file="training_response.txt"; expected="200" ;;
        8) test_file="training_status_response.txt"; expected="200" ;;
        9) test_file="analytics_response.txt"; expected="200" ;;
        10) test_file="export_response.txt"; expected="200" ;;
        11) test_file="invalid_response.txt"; expected="422" ;;
        12) test_file="not_found_response.txt"; expected="404" ;;
    esac
    
    if [ -f "$TEMP_DIR/$test_file" ]; then
        status=$(grep "HTTP/" "$TEMP_DIR/$test_file" | tail -1 | awk '{print $2}' 2>/dev/null || echo "")
        if [ "$status" = "$expected" ]; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
    fi
done

echo "Tests passed: $SUCCESS_COUNT/$TOTAL_TESTS"

if [ $SUCCESS_COUNT -eq $TOTAL_TESTS ]; then
    log_success "All tests passed! 🎉"
else
    log_warning "Some tests failed. Check the detailed output above."
fi

echo ""
log_info "Test files saved in: $TEMP_DIR"
log_info "You can examine individual responses in the temp directory"

# Cleanup option
echo ""
read -p "Do you want to clean up test files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEMP_DIR"
    log_info "Test files cleaned up"
else
    log_info "Test files preserved in $TEMP_DIR"
fi