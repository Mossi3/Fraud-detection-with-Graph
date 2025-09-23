#!/bin/bash

echo "🚀 Starting Credit Card Fraud Detection System"
echo "=============================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

# Check if required files exist
if [ ! -f "/workspace/data/transactions.csv" ]; then
    echo "📊 Generating mock data..."
    python3 /workspace/data/simple_mock_data.py
fi

if [ ! -f "/workspace/data/heatmap_data.json" ]; then
    echo "🔥 Generating heatmaps..."
    python3 /workspace/utils/heatmap_generator.py
fi

if [ ! -f "/workspace/data/simple_graph_data.json" ]; then
    echo "🕸️ Generating graph data..."
    python3 /workspace/models/simple_graph_detector.py
fi

echo "✅ All data files ready"
echo ""
echo "🌐 Starting Flask API server..."
echo "📱 Web Dashboard: http://localhost:5000"
echo "🔗 API Base URL: http://localhost:5000/api"
echo ""
echo "📋 Available endpoints:"
echo "   GET  /api/health              - Health check"
echo "   GET  /api/statistics          - System statistics"
echo "   POST /api/predict             - Single transaction fraud detection"
echo "   POST /api/batch_predict       - Batch transaction fraud detection"
echo "   GET  /api/fraud_rings         - Detected fraud rings"
echo "   GET  /api/heatmaps            - Fraud pattern heatmaps"
echo "   GET  /api/graph_stats         - Graph-based statistics"
echo "   GET  /api/sample_transactions - Sample transactions for testing"
echo "   GET  /api/entity/{type}/{id}  - Entity information"
echo ""
echo "🧪 Test with curl commands (see curl_examples.md)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the Flask application
python3 /workspace/app.py