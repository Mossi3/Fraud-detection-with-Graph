#!/bin/bash

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
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
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
curl -s -X POST "$API_URL/predict" \
  -H "Content-Type: application/json" \
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
curl -s -X POST "$API_URL/detect_rings" \
  -H "Content-Type: application/json" \
  -d '{"min_ring_size": 3}' | python3 -m json.tool

echo ""
echo "✅ API Demo Complete!"
