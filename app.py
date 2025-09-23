from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import csv
from datetime import datetime
import random
import sys
import os

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import our models
from fraud_detector import fraud_detector
from simple_graph_detector import graph_detector
from heatmap_generator import heatmap_generator

app = Flask(__name__)
CORS(app)

# Global variables to store data
transactions_data = []
fraud_rings_data = {}

def load_data():
    """Load all data on startup"""
    global transactions_data, fraud_rings_data
    
    # Load transactions
    with open('/workspace/data/transactions.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['amount'] = float(row['amount'])
            row['is_fraud'] = int(row['is_fraud'])
            transactions_data.append(row)
    
    # Load fraud rings
    with open('/workspace/data/fraud_rings.json', 'r') as f:
        fraud_rings_data = json.load(f)

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'total_transactions': len(transactions_data)
    })

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud for a single transaction"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['card_id', 'merchant_id', 'device_id', 'ip_address', 'amount']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create transaction object
        transaction = {
            'transaction_id': f'txn_{random.randint(100000, 999999)}',
            'card_id': data['card_id'],
            'merchant_id': data['merchant_id'],
            'device_id': data['device_id'],
            'ip_address': data['ip_address'],
            'amount': float(data['amount']),
            'timestamp': datetime.now().isoformat(),
            'category': data.get('category', 'online'),
            'country': data.get('country', 'US'),
            'is_fraud': 0,  # Unknown initially
            'fraud_ring': None
        }
        
        # Get predictions from both models
        deep_learning_prediction = fraud_detector.predict(transaction)
        graph_prediction = graph_detector.predict_transaction_fraud(transaction)
        
        # Combine predictions
        combined_score = (deep_learning_prediction['fraud_probability'] + 
                         graph_prediction['fraud_probability']) / 2
        
        response = {
            'transaction_id': transaction['transaction_id'],
            'predictions': {
                'deep_learning': deep_learning_prediction,
                'graph_based': graph_prediction,
                'combined': {
                    'fraud_probability': combined_score,
                    'is_fraud': combined_score > 0.5,
                    'confidence': abs(combined_score - 0.5) * 2
                }
            },
            'timestamp': transaction['timestamp']
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Predict fraud for multiple transactions"""
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        if not transactions:
            return jsonify({'error': 'No transactions provided'}), 400
        
        results = []
        for i, txn_data in enumerate(transactions):
            try:
                # Create transaction object
                transaction = {
                    'transaction_id': f'batch_txn_{i}_{random.randint(100000, 999999)}',
                    'card_id': txn_data['card_id'],
                    'merchant_id': txn_data['merchant_id'],
                    'device_id': txn_data['device_id'],
                    'ip_address': txn_data['ip_address'],
                    'amount': float(txn_data['amount']),
                    'timestamp': datetime.now().isoformat(),
                    'category': txn_data.get('category', 'online'),
                    'country': txn_data.get('country', 'US'),
                    'is_fraud': 0,
                    'fraud_ring': None
                }
                
                # Get predictions
                deep_learning_prediction = fraud_detector.predict(transaction)
                graph_prediction = graph_detector.predict_transaction_fraud(transaction)
                
                combined_score = (deep_learning_prediction['fraud_probability'] + 
                                 graph_prediction['fraud_probability']) / 2
                
                results.append({
                    'transaction_id': transaction['transaction_id'],
                    'predictions': {
                        'deep_learning': deep_learning_prediction,
                        'graph_based': graph_prediction,
                        'combined': {
                            'fraud_probability': combined_score,
                            'is_fraud': combined_score > 0.5,
                            'confidence': abs(combined_score - 0.5) * 2
                        }
                    }
                })
                
            except Exception as e:
                results.append({
                    'transaction_id': f'batch_txn_{i}',
                    'error': str(e)
                })
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/fraud_rings')
def get_fraud_rings():
    """Get information about detected fraud rings"""
    try:
        rings = fraud_detector.get_fraud_rings()
        return jsonify({
            'fraud_rings': rings,
            'total_rings': len(rings),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/heatmaps')
def get_heatmaps():
    """Get heatmap visualizations"""
    try:
        heatmaps = heatmap_generator.generate_all_heatmaps()
        return jsonify({
            'heatmaps': heatmaps['data'],
            'html_visualizations': heatmaps['html'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get overall statistics"""
    try:
        # Basic statistics
        total_transactions = len(transactions_data)
        fraud_transactions = sum(1 for t in transactions_data if t['is_fraud'])
        fraud_rate = fraud_transactions / total_transactions if total_transactions > 0 else 0
        
        # Amount statistics
        amounts = [t['amount'] for t in transactions_data]
        fraud_amounts = [t['amount'] for t in transactions_data if t['is_fraud']]
        
        stats = {
            'total_transactions': total_transactions,
            'fraud_transactions': fraud_transactions,
            'fraud_rate': fraud_rate,
            'total_amount': sum(amounts),
            'fraud_amount': sum(fraud_amounts),
            'avg_transaction_amount': sum(amounts) / len(amounts) if amounts else 0,
            'avg_fraud_amount': sum(fraud_amounts) / len(fraud_amounts) if fraud_amounts else 0,
            'max_amount': max(amounts) if amounts else 0,
            'min_amount': min(amounts) if amounts else 0,
            'fraud_rings': len(fraud_rings_data),
            'unique_cards': len(set(t['card_id'] for t in transactions_data)),
            'unique_merchants': len(set(t['merchant_id'] for t in transactions_data)),
            'unique_devices': len(set(t['device_id'] for t in transactions_data)),
            'unique_ips': len(set(t['ip_address'] for t in transactions_data))
        }
        
        return jsonify({
            'statistics': stats,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/entity/<entity_type>/<entity_id>')
def get_entity_info(entity_type, entity_id):
    """Get information about a specific entity"""
    try:
        if entity_type not in ['card', 'merchant', 'device', 'ip']:
            return jsonify({'error': 'Invalid entity type'}), 400
        
        # Get entity features
        features = fraud_detector.get_entity_features(entity_type, entity_id)
        
        if not features:
            return jsonify({'error': 'Entity not found'}), 404
        
        # Get recent transactions for this entity
        recent_transactions = []
        for txn in transactions_data[-100:]:  # Last 100 transactions
            if entity_type == 'card' and txn['card_id'] == entity_id:
                recent_transactions.append(txn)
            elif entity_type == 'merchant' and txn['merchant_id'] == entity_id:
                recent_transactions.append(txn)
            elif entity_type == 'device' and txn['device_id'] == entity_id:
                recent_transactions.append(txn)
            elif entity_type == 'ip' and txn['ip_address'] == entity_id:
                recent_transactions.append(txn)
        
        return jsonify({
            'entity_type': entity_type,
            'entity_id': entity_id,
            'features': features,
            'recent_transactions': recent_transactions[:10],  # Last 10 transactions
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample_transactions')
def get_sample_transactions():
    """Get sample transactions for testing"""
    try:
        # Get a mix of fraud and normal transactions
        fraud_transactions = [t for t in transactions_data if t['is_fraud']][:5]
        normal_transactions = [t for t in transactions_data if not t['is_fraud']][:5]
        
        sample_transactions = fraud_transactions + normal_transactions
        random.shuffle(sample_transactions)
        
        return jsonify({
            'sample_transactions': sample_transactions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/graph_stats')
def get_graph_stats():
    """Get graph-based statistics"""
    try:
        stats = graph_detector.get_graph_statistics()
        rings = graph_detector.detect_fraud_rings()
        
        return jsonify({
            'graph_statistics': stats,
            'detected_fraud_rings': rings,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Loading data...")
    load_data()
    print(f"Loaded {len(transactions_data)} transactions")
    print(f"Loaded {len(fraud_rings_data)} fraud rings")
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)