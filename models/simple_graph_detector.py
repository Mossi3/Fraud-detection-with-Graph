import json
import csv
from collections import defaultdict
from typing import Dict, List, Set

class SimpleGraphDetector:
    """Simplified graph-based fraud detection"""
    
    def __init__(self):
        self.transactions = []
        self.fraud_rings = {}
        self.load_data()
    
    def load_data(self):
        """Load transaction data and fraud rings"""
        # Load transactions
        with open('/workspace/data/transactions.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['amount'] = float(row['amount'])
                row['is_fraud'] = int(row['is_fraud'])
                self.transactions.append(row)
        
        # Load fraud rings
        with open('/workspace/data/fraud_rings.json', 'r') as f:
            self.fraud_rings = json.load(f)
    
    def detect_fraud_rings(self) -> Dict:
        """Detect fraud rings using entity relationships"""
        detected_rings = {}
        
        for ring_name, ring_data in self.fraud_rings.items():
            # Count transactions for this ring
            ring_transactions = [t for t in self.transactions if t['fraud_ring'] == ring_name]
            
            detected_rings[ring_name] = {
                'cards': ring_data['cards'],
                'merchants': ring_data['merchants'],
                'devices': ring_data['devices'],
                'ips': ring_data['ips'],
                'transaction_count': len(ring_transactions),
                'total_amount': sum(t['amount'] for t in ring_transactions),
                'fraud_rate': sum(t['is_fraud'] for t in ring_transactions) / len(ring_transactions) if ring_transactions else 0,
                'avg_amount': sum(t['amount'] for t in ring_transactions) / len(ring_transactions) if ring_transactions else 0
            }
        
        return detected_rings
    
    def predict_transaction_fraud(self, transaction: Dict) -> Dict:
        """Predict fraud for a transaction using graph-based features"""
        fraud_score = 0
        
        # Check if transaction belongs to a known fraud ring
        fraud_ring = None
        for ring_name, ring_data in self.fraud_rings.items():
            if (transaction['card_id'] in ring_data['cards'] and
                transaction['merchant_id'] in ring_data['merchants'] and
                transaction['device_id'] in ring_data['devices'] and
                transaction['ip_address'] in ring_data['ips']):
                fraud_ring = ring_name
                fraud_score += 0.6  # High score for fraud ring membership
                break
        
        # Amount-based scoring
        amount = transaction['amount']
        if amount > 1000:
            fraud_score += 0.2
        elif amount > 500:
            fraud_score += 0.1
        
        # Category-based scoring
        category = transaction['category']
        if category in ['electronics', 'jewelry', 'cash_advance']:
            fraud_score += 0.1
        
        fraud_score = min(fraud_score, 1.0)
        
        return {
            'fraud_probability': fraud_score,
            'is_fraud': fraud_score > 0.5,
            'fraud_ring': fraud_ring,
            'confidence': abs(fraud_score - 0.5) * 2
        }
    
    def get_graph_statistics(self) -> Dict:
        """Get graph statistics"""
        stats = {
            'total_transactions': len(self.transactions),
            'fraud_transactions': sum(1 for t in self.transactions if t['is_fraud']),
            'fraud_rate': sum(1 for t in self.transactions if t['is_fraud']) / len(self.transactions),
            'fraud_rings': len(self.fraud_rings),
            'unique_cards': len(set(t['card_id'] for t in self.transactions)),
            'unique_merchants': len(set(t['merchant_id'] for t in self.transactions)),
            'unique_devices': len(set(t['device_id'] for t in self.transactions)),
            'unique_ips': len(set(t['ip_address'] for t in self.transactions))
        }
        
        return stats

# Initialize graph detector
graph_detector = SimpleGraphDetector()

if __name__ == "__main__":
    print("Simple Graph Fraud Detection System")
    print("=" * 40)
    
    # Get statistics
    stats = graph_detector.get_graph_statistics()
    print(f"Total transactions: {stats['total_transactions']}")
    print(f"Fraud transactions: {stats['fraud_transactions']}")
    print(f"Fraud rate: {stats['fraud_rate']:.2%}")
    print(f"Fraud rings: {stats['fraud_rings']}")
    
    # Test prediction
    test_transaction = {
        'transaction_id': 'test_001',
        'card_id': 'card_001000',
        'merchant_id': 'merchant_0200',
        'device_id': 'device_005000',
        'ip_address': '192.168.1.1',
        'amount': 1500.0,
        'timestamp': '2024-01-15T10:30:00',
        'category': 'electronics',
        'country': 'US',
        'is_fraud': 1,
        'fraud_ring': 'ring_1'
    }
    
    prediction = graph_detector.predict_transaction_fraud(test_transaction)
    print(f"\nTest prediction: {prediction}")
    
    # Get fraud rings
    rings = graph_detector.detect_fraud_rings()
    print(f"\nDetected fraud rings: {len(rings)}")
    
    # Save data
    graph_data = {
        'statistics': stats,
        'fraud_rings': rings
    }
    
    with open('/workspace/data/simple_graph_data.json', 'w') as f:
        json.dump(graph_data, f, indent=2, default=str)
    
    print("Graph data saved successfully!")