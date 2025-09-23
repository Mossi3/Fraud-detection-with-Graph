#!/bin/bash

# Fraud Detection Graph API Test Suite
# This script contains comprehensive curl tests for all API endpoints

# Configuration
API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
API_VERSION="v1"
API_URL="${API_BASE_URL}/api/${API_VERSION}"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper functions
print_test_header() {
    echo -e "\n${YELLOW}===== $1 =====${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
}

check_response() {
    if [ $1 -eq 0 ]; then
        print_success "$2"
    else
        print_failure "$2"
    fi
}

# Start tests
echo "Starting Fraud Detection API Tests"
echo "API URL: ${API_URL}"
echo "=================================="

# Test 1: Health Check
print_test_header "Test 1: Health Check"
RESPONSE=$(curl -s -w "\n%{http_code}" ${API_URL}/health)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Health check passed (HTTP $HTTP_CODE)"
    echo "Response: $BODY"
else
    print_failure "Health check failed (HTTP $HTTP_CODE)"
fi

# Test 2: Single Transaction Scoring - Normal Transaction
print_test_header "Test 2: Score Normal Transaction"
NORMAL_TXN=$(cat <<EOF
{
  "transaction_id": "txn_001",
  "card_id": "card_1234567890",
  "merchant_id": "merchant_amazon",
  "amount": 49.99,
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S)Z",
  "device_id": "device_abc123",
  "ip_address": "192.168.1.100",
  "location": [37.7749, -122.4194],
  "additional_features": {
    "merchant_category": "online_retail",
    "payment_method": "credit"
  }
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/score \
  -H "Content-Type: application/json" \
  -d "$NORMAL_TXN")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Normal transaction scored successfully"
    echo "Response: $BODY" | jq '.'
else
    print_failure "Failed to score normal transaction (HTTP $HTTP_CODE)"
fi

# Test 3: Single Transaction Scoring - Suspicious Transaction
print_test_header "Test 3: Score Suspicious Transaction"
SUSPICIOUS_TXN=$(cat <<EOF
{
  "transaction_id": "txn_002",
  "card_id": "card_1234567890",
  "merchant_id": "merchant_suspicious",
  "amount": 5000.00,
  "timestamp": "$(date -u -d '3 hours ago' +%Y-%m-%dT%H:%M:%S)Z",
  "device_id": "device_xyz789",
  "ip_address": "45.67.89.10",
  "location": [40.7128, -74.0060],
  "additional_features": {
    "merchant_category": "high_risk",
    "payment_method": "wire_transfer",
    "unusual_time": true
  }
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/score \
  -H "Content-Type: application/json" \
  -d "$SUSPICIOUS_TXN")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Suspicious transaction scored successfully"
    echo "Response: $BODY" | jq '.'
else
    print_failure "Failed to score suspicious transaction (HTTP $HTTP_CODE)"
fi

# Test 4: Batch Transaction Scoring
print_test_header "Test 4: Batch Transaction Scoring"
BATCH_REQUEST=$(cat <<EOF
{
  "transactions": [
    {
      "transaction_id": "batch_001",
      "card_id": "card_9876543210",
      "merchant_id": "merchant_walmart",
      "amount": 127.45,
      "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S)Z",
      "device_id": "device_mobile1",
      "ip_address": "10.0.0.1"
    },
    {
      "transaction_id": "batch_002",
      "card_id": "card_9876543210",
      "merchant_id": "merchant_gas_station",
      "amount": 45.00,
      "timestamp": "$(date -u -d '5 minutes' +%Y-%m-%dT%H:%M:%S)Z",
      "device_id": "device_mobile1",
      "ip_address": "10.0.0.1"
    },
    {
      "transaction_id": "batch_003",
      "card_id": "card_9876543210",
      "merchant_id": "merchant_atm",
      "amount": 500.00,
      "timestamp": "$(date -u -d '10 minutes' +%Y-%m-%dT%H:%M:%S)Z",
      "device_id": "device_unknown",
      "ip_address": "200.100.50.25"
    }
  ],
  "include_visualization": true
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/batch_score \
  -H "Content-Type: application/json" \
  -d "$BATCH_REQUEST")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Batch scoring completed successfully"
    echo "Summary:"
    echo "$BODY" | jq '{total_transactions, high_risk_count, visualization_url}'
else
    print_failure "Failed to score batch transactions (HTTP $HTTP_CODE)"
fi

# Test 5: Fraud Ring Detection
print_test_header "Test 5: Fraud Ring Detection"
RING_REQUEST=$(cat <<EOF
{
  "entity_ids": [
    "card_1234567890",
    "card_9876543210",
    "merchant_suspicious",
    "device_xyz789",
    "ip_45.67.89.10"
  ],
  "detection_method": "louvain",
  "min_ring_size": 3
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/detect_rings \
  -H "Content-Type: application/json" \
  -d "$RING_REQUEST")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Fraud ring detection completed"
    echo "Response: $BODY" | jq '.'
else
    print_failure "Failed to detect fraud rings (HTTP $HTTP_CODE)"
fi

# Test 6: Graph Statistics
print_test_header "Test 6: Graph Statistics"
RESPONSE=$(curl -s -w "\n%{http_code}" ${API_URL}/stats)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Retrieved graph statistics"
    echo "Response: $BODY" | jq '.'
else
    print_failure "Failed to get graph statistics (HTTP $HTTP_CODE)"
fi

# Test 7: Performance Metrics
print_test_header "Test 7: Performance Metrics"
RESPONSE=$(curl -s -w "\n%{http_code}" ${API_URL}/metrics)
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Retrieved performance metrics"
    echo "Key Metrics:"
    echo "$BODY" | jq '{transactions, performance, alerts}'
else
    print_failure "Failed to get metrics (HTTP $HTTP_CODE)"
fi

# Test 8: Velocity Pattern Detection
print_test_header "Test 8: Velocity Pattern Detection"
echo "Simulating rapid transactions from same card..."

for i in {1..7}; do
    VELOCITY_TXN=$(cat <<EOF
{
  "transaction_id": "velocity_test_$i",
  "card_id": "card_velocity_test",
  "merchant_id": "merchant_test_$i",
  "amount": $((RANDOM % 200 + 50)),
  "timestamp": "$(date -u -d "$i minutes" +%Y-%m-%dT%H:%M:%S)Z",
  "device_id": "device_velocity",
  "ip_address": "172.16.0.$i"
}
EOF
)
    
    curl -s -X POST ${API_URL}/score \
      -H "Content-Type: application/json" \
      -d "$VELOCITY_TXN" > /dev/null
done

# Check the last transaction for velocity pattern
FINAL_VELOCITY_TXN=$(cat <<EOF
{
  "transaction_id": "velocity_test_final",
  "card_id": "card_velocity_test",
  "merchant_id": "merchant_test_final",
  "amount": 999.99,
  "timestamp": "$(date -u -d "8 minutes" +%Y-%m-%dT%H:%M:%S)Z",
  "device_id": "device_velocity",
  "ip_address": "172.16.0.10"
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/score \
  -H "Content-Type: application/json" \
  -d "$FINAL_VELOCITY_TXN")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    PATTERNS=$(echo "$BODY" | jq -r '.fraud_patterns[]' 2>/dev/null)
    if echo "$PATTERNS" | grep -q "high_velocity"; then
        print_success "Velocity pattern detected correctly"
    else
        print_failure "Velocity pattern not detected"
    fi
else
    print_failure "Failed velocity pattern test (HTTP $HTTP_CODE)"
fi

# Test 9: Device Sharing Detection
print_test_header "Test 9: Device Sharing Detection"
echo "Simulating multiple cards using same device..."

SHARED_DEVICE="device_shared_123"
for i in {1..4}; do
    DEVICE_TXN=$(cat <<EOF
{
  "transaction_id": "device_test_$i",
  "card_id": "card_device_user_$i",
  "merchant_id": "merchant_online",
  "amount": $((RANDOM % 300 + 100)),
  "timestamp": "$(date -u -d "$((i*5)) minutes" +%Y-%m-%dT%H:%M:%S)Z",
  "device_id": "$SHARED_DEVICE",
  "ip_address": "192.168.1.1"
}
EOF
)
    
    curl -s -X POST ${API_URL}/score \
      -H "Content-Type: application/json" \
      -d "$DEVICE_TXN" > /dev/null
done

# Check detection on final transaction
FINAL_DEVICE_TXN=$(cat <<EOF
{
  "transaction_id": "device_test_final",
  "card_id": "card_device_user_5",
  "merchant_id": "merchant_suspicious_online",
  "amount": 1500.00,
  "timestamp": "$(date -u -d "25 minutes" +%Y-%m-%dT%H:%M:%S)Z",
  "device_id": "$SHARED_DEVICE",
  "ip_address": "192.168.1.1"
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/score \
  -H "Content-Type: application/json" \
  -d "$FINAL_DEVICE_TXN")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" -eq 200 ]; then
    PATTERNS=$(echo "$BODY" | jq -r '.fraud_patterns[]' 2>/dev/null)
    if echo "$PATTERNS" | grep -q "device_sharing"; then
        print_success "Device sharing pattern detected correctly"
    else
        print_failure "Device sharing pattern not detected"
    fi
else
    print_failure "Failed device sharing test (HTTP $HTTP_CODE)"
fi

# Test 10: Error Handling - Invalid Request
print_test_header "Test 10: Error Handling"
INVALID_REQUEST=$(cat <<EOF
{
  "transaction_id": "invalid_test",
  "amount": -100
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/score \
  -H "Content-Type: application/json" \
  -d "$INVALID_REQUEST")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_CODE" -eq 422 ] || [ "$HTTP_CODE" -eq 400 ]; then
    print_success "Invalid request properly rejected"
else
    print_failure "Invalid request not handled correctly (HTTP $HTTP_CODE)"
fi

# Test 11: Ring Detection with Time Window
print_test_header "Test 11: Time-based Ring Detection"
TIMED_RING_REQUEST=$(cat <<EOF
{
  "entity_ids": [
    "card_velocity_test",
    "card_device_user_1",
    "card_device_user_2"
  ],
  "detection_method": "spectral",
  "min_ring_size": 2,
  "time_window_hours": 1
}
EOF
)

RESPONSE=$(curl -s -w "\n%{http_code}" \
  -X POST ${API_URL}/detect_rings \
  -H "Content-Type: application/json" \
  -d "$TIMED_RING_REQUEST")

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
if [ "$HTTP_CODE" -eq 200 ]; then
    print_success "Time-based ring detection completed"
else
    print_failure "Time-based ring detection failed (HTTP $HTTP_CODE)"
fi

# Test 12: Concurrent Requests
print_test_header "Test 12: Concurrent Request Handling"
echo "Sending 10 concurrent requests..."

for i in {1..10}; do
    (
        CONCURRENT_TXN=$(cat <<EOF
{
  "transaction_id": "concurrent_$i",
  "card_id": "card_concurrent_$i",
  "merchant_id": "merchant_stress_test",
  "amount": $((RANDOM % 1000 + 100)),
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%S)Z"
}
EOF
)
        curl -s -X POST ${API_URL}/score \
          -H "Content-Type: application/json" \
          -d "$CONCURRENT_TXN" > /dev/null
    ) &
done

wait
print_success "Concurrent requests handled"

# Summary
echo -e "\n${YELLOW}===== Test Summary =====${NC}"
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED))
echo -e "Total Tests: $TOTAL_TESTS"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed.${NC}"
    exit 1
fi