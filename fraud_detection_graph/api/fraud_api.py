"""
Flask API for Fraud Detection System
Provides REST endpoints for fraud detection, risk scoring, and fraud ring analysis.
"""

from flask import Flask, request, jsonify, render_template_string, send_from_directory
import pandas as pd
import numpy as np
import torch
import json
import pickle
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append('/workspace/fraud_detection_graph')

from models.graph_builder import FraudGraphBuilder
from models.gnn_models import create_model, FraudDetectionTrainer
from models.community_detection import FraudRingDetector
from visualizations.fraud_visualizer import FraudVisualizer
from typing import Dict, List, Any, Optional

app = Flask(__name__)

class FraudDetectionAPI:
    def __init__(self):
        self.builder = None
        self.model = None
        self.hetero_data = None
        self.detector = None
        self.visualizer = None
        self.node_mappings = {}
        self.is_loaded = False
        
    def load_system(self):
        """Load all system components"""
        try:
            print("Loading fraud detection system...")
            
            # Load graph builder and data
            self.builder = FraudGraphBuilder()
            self.hetero_data, _ = self.builder.load_graph()
            
            # Load node mappings
            with open('/workspace/fraud_detection_graph/data/node_mappings.pkl', 'rb') as f:
                self.node_mappings = pickle.load(f)
            
            # Load trained model if available
            try:
                model_path = '/workspace/fraud_detection_graph/models/best_model.pt'
                if os.path.exists(model_path):
                    self.model = create_model(self.hetero_data, 'graphsage', hidden_dim=64)
                    self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
                    self.model.eval()
                    print("Loaded trained model")
                else:
                    # Create and train a simple model for demo
                    self.model = create_model(self.hetero_data, 'graphsage', hidden_dim=64)
                    print("Created new model (not trained)")
            except Exception as e:
                print(f"Model loading error: {e}")
                self.model = None
            
            # Load fraud ring detector
            self.detector = FraudRingDetector()
            self.detector.load_graph_data()
            
            # Load visualizer
            self.visualizer = FraudVisualizer()
            self.visualizer.load_data()
            
            self.is_loaded = True
            print("System loaded successfully!")
            
        except Exception as e:
            print(f"Error loading system: {e}")
            self.is_loaded = False
    
    def predict_transaction_fraud(self, card_id: str, merchant_id: str, 
                                device_id: str, ip_address: str, amount: float) -> Dict:
        """Predict fraud probability for a transaction"""
        if not self.is_loaded or self.model is None:
            return {"error": "System not loaded or model not available"}
        
        try:
            # Map entity IDs to node indices
            if (card_id not in self.node_mappings['card'] or
                merchant_id not in self.node_mappings['merchant'] or
                device_id not in self.node_mappings['device'] or
                ip_address not in self.node_mappings['ip']):
                return {"error": "One or more entities not found in graph"}
            
            card_idx = self.node_mappings['card'][card_id]
            merchant_idx = self.node_mappings['merchant'][merchant_id]
            device_idx = self.node_mappings['device'][device_id]
            ip_idx = self.node_mappings['ip'][ip_address]
            
            # Get model predictions
            with torch.no_grad():
                x_dict = {key: x for key, x in self.hetero_data.x_dict.items()}
                edge_index_dict = {key: edge_index for key, edge_index in self.hetero_data.edge_index_dict.items()}
                
                outputs = self.model(x_dict, edge_index_dict)
                
                # Predict transaction fraud
                transaction_pair = torch.tensor([[card_idx, merchant_idx]], dtype=torch.long)
                fraud_logits = self.model.predict_transaction_fraud(outputs['embeddings'], transaction_pair)
                fraud_prob = torch.softmax(fraud_logits, dim=1)[0, 1].item()
                
                # Get risk scores for entities
                card_risk = outputs['risk_scores']['card_risk_score'][card_idx].item()
                merchant_risk = outputs['risk_scores']['merchant_risk_score'][merchant_idx].item()
                device_risk = outputs['risk_scores']['device_risk_score'][device_idx].item()
                ip_risk = outputs['risk_scores']['ip_risk_score'][ip_idx].item()
                
                # Calculate composite risk score
                composite_risk = (fraud_prob * 0.4 + card_risk * 0.25 + 
                                merchant_risk * 0.15 + device_risk * 0.1 + ip_risk * 0.1)
                
                # Determine risk level
                if composite_risk > 0.8:
                    risk_level = "CRITICAL"
                elif composite_risk > 0.6:
                    risk_level = "HIGH"
                elif composite_risk > 0.4:
                    risk_level = "MEDIUM"
                elif composite_risk > 0.2:
                    risk_level = "LOW"
                else:
                    risk_level = "MINIMAL"
                
                return {
                    "fraud_probability": round(fraud_prob, 4),
                    "composite_risk_score": round(composite_risk, 4),
                    "risk_level": risk_level,
                    "entity_risks": {
                        "card_risk": round(card_risk, 4),
                        "merchant_risk": round(merchant_risk, 4),
                        "device_risk": round(device_risk, 4),
                        "ip_risk": round(ip_risk, 4)
                    },
                    "recommendation": "BLOCK" if composite_risk > 0.7 else "REVIEW" if composite_risk > 0.4 else "APPROVE",
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def detect_fraud_rings(self, min_ring_size: int = 3) -> Dict:
        """Detect fraud rings in the current graph"""
        if not self.is_loaded:
            return {"error": "System not loaded"}
        
        try:
            # Detect fraud rings
            detected_rings = self.detector.identify_fraud_rings(min_ring_size=min_ring_size)
            
            # Evaluate against ground truth
            true_rings = self.detector.data['fraud_rings']
            evaluation = self.detector.evaluate_ring_detection(detected_rings, true_rings)
            
            # Prepare response
            rings_summary = []
            for ring in detected_rings[:10]:  # Return top 10 rings
                rings_summary.append({
                    "ring_id": ring['ring_id'],
                    "size": ring['size'],
                    "fraud_score": round(ring['fraud_score'], 4),
                    "entity_type": ring['entity_type'],
                    "total_transactions": ring['total_transactions'],
                    "fraud_transactions": ring['fraud_transactions'],
                    "estimated_loss": ring.get('estimated_loss', 0)
                })
            
            return {
                "detected_rings_count": len(detected_rings),
                "evaluation_metrics": {
                    "precision": round(evaluation['precision'], 4),
                    "recall": round(evaluation['recall'], 4),
                    "f1_score": round(evaluation['f1'], 4)
                },
                "top_rings": rings_summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Ring detection error: {str(e)}"}
    
    def get_entity_risk_profile(self, entity_type: str, entity_id: str) -> Dict:
        """Get detailed risk profile for an entity"""
        if not self.is_loaded:
            return {"error": "System not loaded"}
        
        try:
            if entity_type not in self.node_mappings or entity_id not in self.node_mappings[entity_type]:
                return {"error": f"Entity {entity_id} of type {entity_type} not found"}
            
            entity_idx = self.node_mappings[entity_type][entity_id]
            
            # Get transactions for this entity
            if entity_type == 'card':
                entity_transactions = self.detector.data['transactions'][
                    self.detector.data['transactions']['card_id'] == entity_id
                ]
            elif entity_type == 'merchant':
                entity_transactions = self.detector.data['transactions'][
                    self.detector.data['transactions']['merchant_id'] == entity_id
                ]
            elif entity_type == 'device':
                entity_transactions = self.detector.data['transactions'][
                    self.detector.data['transactions']['device_id'] == entity_id
                ]
            elif entity_type == 'ip':
                entity_transactions = self.detector.data['transactions'][
                    self.detector.data['transactions']['ip_address'] == entity_id
                ]
            else:
                return {"error": f"Unknown entity type: {entity_type}"}
            
            if len(entity_transactions) == 0:
                return {"error": f"No transactions found for entity {entity_id}"}
            
            # Calculate statistics
            total_transactions = len(entity_transactions)
            fraud_transactions = entity_transactions['is_fraud'].sum()
            fraud_rate = fraud_transactions / total_transactions
            avg_amount = entity_transactions['amount'].mean()
            total_amount = entity_transactions['amount'].sum()
            
            # Get model risk score if available
            model_risk = 0.5  # Default
            if self.model is not None:
                try:
                    with torch.no_grad():
                        x_dict = {key: x for key, x in self.hetero_data.x_dict.items()}
                        edge_index_dict = {key: edge_index for key, edge_index in self.hetero_data.edge_index_dict.items()}
                        outputs = self.model(x_dict, edge_index_dict)
                        model_risk = outputs['risk_scores'][f'{entity_type}_risk_score'][entity_idx].item()
                except:
                    pass
            
            # Recent activity analysis
            entity_transactions['timestamp'] = pd.to_datetime(entity_transactions['timestamp'])
            recent_transactions = entity_transactions[
                entity_transactions['timestamp'] > (entity_transactions['timestamp'].max() - pd.Timedelta(days=7))
            ]
            
            return {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "risk_score": round(model_risk, 4),
                "transaction_stats": {
                    "total_transactions": int(total_transactions),
                    "fraud_transactions": int(fraud_transactions),
                    "fraud_rate": round(fraud_rate, 4),
                    "avg_transaction_amount": round(avg_amount, 2),
                    "total_transaction_amount": round(total_amount, 2)
                },
                "recent_activity": {
                    "transactions_last_7_days": len(recent_transactions),
                    "fraud_last_7_days": int(recent_transactions['is_fraud'].sum()),
                    "avg_amount_last_7_days": round(recent_transactions['amount'].mean(), 2) if len(recent_transactions) > 0 else 0
                },
                "risk_factors": self._analyze_risk_factors(entity_transactions, entity_type),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Risk profile error: {str(e)}"}
    
    def _analyze_risk_factors(self, transactions: pd.DataFrame, entity_type: str) -> List[str]:
        """Analyze risk factors for an entity"""
        risk_factors = []
        
        fraud_rate = transactions['is_fraud'].mean()
        if fraud_rate > 0.1:
            risk_factors.append(f"High fraud rate: {fraud_rate:.2%}")
        
        if entity_type == 'card':
            unique_merchants = transactions['merchant_id'].nunique()
            if unique_merchants > 20:
                risk_factors.append(f"High merchant diversity: {unique_merchants} unique merchants")
            
            avg_amount = transactions['amount'].mean()
            if avg_amount > 1000:
                risk_factors.append(f"High average transaction amount: ${avg_amount:.2f}")
        
        # Time-based patterns
        transactions['hour'] = pd.to_datetime(transactions['timestamp']).dt.hour
        night_transactions = transactions[transactions['hour'].isin([0, 1, 2, 3, 4, 5])]
        if len(night_transactions) > len(transactions) * 0.3:
            risk_factors.append("High proportion of night-time transactions")
        
        return risk_factors

# Initialize the API system
fraud_api = FraudDetectionAPI()

# API Routes
@app.route('/')
def home():
    """API documentation and testing interface"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .header { text-align: center; color: #333; margin-bottom: 30px; }
            .endpoint { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { background: #007bff; color: white; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .test-form { background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }
            input, select, button { margin: 5px; padding: 8px; }
            button { background: #28a745; color: white; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #218838; }
            .result { background: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; border-color: #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Fraud Detection API</h1>
                <p>Graph-based Deep Learning Fraud Detection System</p>
                <p>Status: <span id="status">{{ 'Loaded' if is_loaded else 'Not Loaded' }}</span></p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /predict</h3>
                <p>Predict fraud probability for a transaction</p>
                <div class="test-form">
                    <h4>Test Transaction Prediction</h4>
                    <input type="text" id="card_id" placeholder="Card ID (e.g., card_000001)" style="width: 150px;">
                    <input type="text" id="merchant_id" placeholder="Merchant ID (e.g., merchant_00001)" style="width: 150px;">
                    <input type="text" id="device_id" placeholder="Device ID (e.g., device_000001)" style="width: 150px;">
                    <input type="text" id="ip_address" placeholder="IP Address (e.g., 192.168.1.1)" style="width: 150px;">
                    <input type="number" id="amount" placeholder="Amount" step="0.01" style="width: 100px;">
                    <button onclick="predictFraud()">Predict Fraud</button>
                    <div id="predict-result"></div>
                </div>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /detect_rings</h3>
                <p>Detect fraud rings in the graph</p>
                <div class="test-form">
                    <h4>Detect Fraud Rings</h4>
                    <input type="number" id="min_ring_size" placeholder="Min Ring Size" value="3" min="2" max="10" style="width: 100px;">
                    <button onclick="detectRings()">Detect Rings</button>
                    <div id="rings-result"></div>
                </div>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /entity_profile/{entity_type}/{entity_id}</h3>
                <p>Get detailed risk profile for an entity</p>
                <div class="test-form">
                    <h4>Get Entity Risk Profile</h4>
                    <select id="entity_type">
                        <option value="card">Card</option>
                        <option value="merchant">Merchant</option>
                        <option value="device">Device</option>
                        <option value="ip">IP</option>
                    </select>
                    <input type="text" id="entity_id" placeholder="Entity ID" style="width: 200px;">
                    <button onclick="getEntityProfile()">Get Profile</button>
                    <div id="profile-result"></div>
                </div>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /visualizations</h3>
                <p>Access fraud detection visualizations</p>
                <div class="test-form">
                    <button onclick="window.open('/visualizations/', '_blank')">View Visualizations</button>
                </div>
            </div>
        </div>
        
        <script>
            async function predictFraud() {
                const data = {
                    card_id: document.getElementById('card_id').value,
                    merchant_id: document.getElementById('merchant_id').value,
                    device_id: document.getElementById('device_id').value,
                    ip_address: document.getElementById('ip_address').value,
                    amount: parseFloat(document.getElementById('amount').value)
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    document.getElementById('predict-result').innerHTML = 
                        '<div class="result"><pre>' + JSON.stringify(result, null, 2) + '</pre></div>';
                } catch (error) {
                    document.getElementById('predict-result').innerHTML = 
                        '<div class="result error">Error: ' + error.message + '</div>';
                }
            }
            
            async function detectRings() {
                const data = {
                    min_ring_size: parseInt(document.getElementById('min_ring_size').value)
                };
                
                try {
                    const response = await fetch('/detect_rings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(data)
                    });
                    const result = await response.json();
                    document.getElementById('rings-result').innerHTML = 
                        '<div class="result"><pre>' + JSON.stringify(result, null, 2) + '</pre></div>';
                } catch (error) {
                    document.getElementById('rings-result').innerHTML = 
                        '<div class="result error">Error: ' + error.message + '</div>';
                }
            }
            
            async function getEntityProfile() {
                const entityType = document.getElementById('entity_type').value;
                const entityId = document.getElementById('entity_id').value;
                
                try {
                    const response = await fetch(`/entity_profile/${entityType}/${entityId}`);
                    const result = await response.json();
                    document.getElementById('profile-result').innerHTML = 
                        '<div class="result"><pre>' + JSON.stringify(result, null, 2) + '</pre></div>';
                } catch (error) {
                    document.getElementById('profile-result').innerHTML = 
                        '<div class="result error">Error: ' + error.message + '</div>';
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template, is_loaded=fraud_api.is_loaded)

@app.route('/predict', methods=['POST'])
def predict_fraud():
    """Predict fraud for a transaction"""
    try:
        data = request.json
        result = fraud_api.predict_transaction_fraud(
            card_id=data['card_id'],
            merchant_id=data['merchant_id'],
            device_id=data['device_id'],
            ip_address=data['ip_address'],
            amount=data['amount']
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/detect_rings', methods=['POST'])
def detect_fraud_rings():
    """Detect fraud rings"""
    try:
        data = request.json
        min_ring_size = data.get('min_ring_size', 3)
        result = fraud_api.detect_fraud_rings(min_ring_size=min_ring_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/entity_profile/<entity_type>/<entity_id>', methods=['GET'])
def get_entity_profile(entity_type, entity_id):
    """Get entity risk profile"""
    try:
        result = fraud_api.get_entity_risk_profile(entity_type, entity_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/visualizations/')
def visualizations_index():
    """Serve visualizations directory"""
    return send_from_directory('/workspace/fraud_detection_graph/visualizations/', 'fraud_analysis_report.html')

@app.route('/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization files"""
    return send_from_directory('/workspace/fraud_detection_graph/visualizations/', filename)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "system_loaded": fraud_api.is_loaded,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if not fraud_api.is_loaded:
        return jsonify({"error": "System not loaded"}), 503
    
    try:
        stats = {
            "graph_stats": {
                "card_nodes": fraud_api.hetero_data['card'].num_nodes,
                "merchant_nodes": fraud_api.hetero_data['merchant'].num_nodes,
                "device_nodes": fraud_api.hetero_data['device'].num_nodes,
                "ip_nodes": fraud_api.hetero_data['ip'].num_nodes,
                "total_transactions": len(fraud_api.hetero_data.transaction_labels),
                "fraud_transactions": fraud_api.hetero_data.transaction_labels.sum().item()
            },
            "model_loaded": fraud_api.model is not None,
            "timestamp": datetime.now().isoformat()
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Fraud Detection API...")
    fraud_api.load_system()
    app.run(host='0.0.0.0', port=5000, debug=True)