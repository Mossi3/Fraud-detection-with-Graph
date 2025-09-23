import json
import csv
import random
import math
from datetime import datetime
from typing import Dict, List, Tuple
import pickle

class SimpleFraudDetector:
    """A simple fraud detection model using rule-based and statistical methods"""
    
    def __init__(self):
        self.fraud_rules = {}
        self.statistical_features = {}
        self.fraud_rings = {}
        self.load_data()
        
    def load_data(self):
        """Load transaction data and fraud rings"""
        # Load transactions
        self.transactions = []
        with open('/workspace/data/transactions.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['amount'] = float(row['amount'])
                row['is_fraud'] = int(row['is_fraud'])
                self.transactions.append(row)
        
        # Load fraud rings
        with open('/workspace/data/fraud_rings.json', 'r') as f:
            self.fraud_rings = json.load(f)
        
        # Calculate statistical features
        self._calculate_features()
        
    def _calculate_features(self):
        """Calculate statistical features for fraud detection"""
        # Card statistics
        card_stats = {}
        for txn in self.transactions:
            card_id = txn['card_id']
            if card_id not in card_stats:
                card_stats[card_id] = {
                    'transactions': [],
                    'amounts': [],
                    'merchants': set(),
                    'devices': set(),
                    'ips': set(),
                    'fraud_count': 0
                }
            
            card_stats[card_id]['transactions'].append(txn)
            card_stats[card_id]['amounts'].append(txn['amount'])
            card_stats[card_id]['merchants'].add(txn['merchant_id'])
            card_stats[card_id]['devices'].add(txn['device_id'])
            card_stats[card_id]['ips'].add(txn['ip_address'])
            if txn['is_fraud']:
                card_stats[card_id]['fraud_count'] += 1
        
        # Calculate derived features
        for card_id, stats in card_stats.items():
            amounts = stats['amounts']
            stats['avg_amount'] = sum(amounts) / len(amounts)
            stats['max_amount'] = max(amounts)
            stats['min_amount'] = min(amounts)
            stats['amount_std'] = math.sqrt(sum((x - stats['avg_amount'])**2 for x in amounts) / len(amounts))
            stats['unique_merchants'] = len(stats['merchants'])
            stats['unique_devices'] = len(stats['devices'])
            stats['unique_ips'] = len(stats['ips'])
            stats['fraud_rate'] = stats['fraud_count'] / len(stats['transactions'])
            
            # Convert sets to lists for JSON serialization
            stats['merchants'] = list(stats['merchants'])
            stats['devices'] = list(stats['devices'])
            stats['ips'] = list(stats['ips'])
        
        self.card_stats = card_stats
        
        # Merchant statistics
        merchant_stats = {}
        for txn in self.transactions:
            merchant_id = txn['merchant_id']
            if merchant_id not in merchant_stats:
                merchant_stats[merchant_id] = {
                    'transactions': [],
                    'amounts': [],
                    'cards': set(),
                    'fraud_count': 0
                }
            
            merchant_stats[merchant_id]['transactions'].append(txn)
            merchant_stats[merchant_id]['amounts'].append(txn['amount'])
            merchant_stats[merchant_id]['cards'].add(txn['card_id'])
            if txn['is_fraud']:
                merchant_stats[merchant_id]['fraud_count'] += 1
        
        for merchant_id, stats in merchant_stats.items():
            amounts = stats['amounts']
            stats['avg_amount'] = sum(amounts) / len(amounts)
            stats['fraud_rate'] = stats['fraud_count'] / len(stats['transactions'])
            stats['unique_cards'] = len(stats['cards'])
            stats['cards'] = list(stats['cards'])
        
        self.merchant_stats = merchant_stats
    
    def detect_fraud_ring(self, card_id: str, merchant_id: str, device_id: str, ip_address: str) -> str:
        """Detect if transaction belongs to a fraud ring"""
        for ring_name, ring_data in self.fraud_rings.items():
            if (card_id in ring_data['cards'] and 
                merchant_id in ring_data['merchants'] and 
                device_id in ring_data['devices'] and 
                ip_address in ring_data['ips']):
                return ring_name
        return None
    
    def calculate_fraud_score(self, transaction: Dict) -> float:
        """Calculate fraud score for a transaction"""
        score = 0.0
        
        # Amount-based scoring
        amount = transaction['amount']
        if amount > 1000:
            score += 0.3
        elif amount > 500:
            score += 0.2
        elif amount > 100:
            score += 0.1
        
        # Card statistics
        card_id = transaction['card_id']
        if card_id in self.card_stats:
            card_stats = self.card_stats[card_id]
            
            # High fraud rate
            if card_stats['fraud_rate'] > 0.5:
                score += 0.4
            elif card_stats['fraud_rate'] > 0.2:
                score += 0.2
            
            # Unusual amount
            if amount > card_stats['avg_amount'] * 3:
                score += 0.3
            elif amount > card_stats['avg_amount'] * 2:
                score += 0.2
            
            # Many unique merchants (potential card testing)
            if card_stats['unique_merchants'] > 50:
                score += 0.2
            elif card_stats['unique_merchants'] > 20:
                score += 0.1
        
        # Merchant statistics
        merchant_id = transaction['merchant_id']
        if merchant_id in self.merchant_stats:
            merchant_stats = self.merchant_stats[merchant_id]
            
            # High fraud rate merchant
            if merchant_stats['fraud_rate'] > 0.3:
                score += 0.3
            elif merchant_stats['fraud_rate'] > 0.1:
                score += 0.1
        
        # Fraud ring detection
        fraud_ring = self.detect_fraud_ring(
            transaction['card_id'],
            transaction['merchant_id'],
            transaction['device_id'],
            transaction['ip_address']
        )
        if fraud_ring:
            score += 0.5
        
        # Category-based scoring
        category = transaction['category']
        if category in ['electronics', 'jewelry', 'cash_advance']:
            score += 0.1
        
        # Time-based scoring (simplified)
        hour = datetime.fromisoformat(transaction['timestamp']).hour
        if hour < 6 or hour > 22:  # Unusual hours
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def predict(self, transaction: Dict) -> Dict:
        """Predict fraud probability and ring membership"""
        fraud_score = self.calculate_fraud_score(transaction)
        fraud_ring = self.detect_fraud_ring(
            transaction['card_id'],
            transaction['merchant_id'],
            transaction['device_id'],
            transaction['ip_address']
        )
        
        return {
            'fraud_probability': fraud_score,
            'is_fraud': fraud_score > 0.5,
            'fraud_ring': fraud_ring,
            'confidence': abs(fraud_score - 0.5) * 2  # Higher confidence when score is far from 0.5
        }
    
    def get_fraud_rings(self) -> Dict:
        """Get information about detected fraud rings"""
        ring_stats = {}
        
        for ring_name, ring_data in self.fraud_rings.items():
            # Count transactions in this ring
            ring_transactions = [t for t in self.transactions if t['fraud_ring'] == ring_name]
            
            ring_stats[ring_name] = {
                'cards': ring_data['cards'],
                'merchants': ring_data['merchants'],
                'devices': ring_data['devices'],
                'ips': ring_data['ips'],
                'transaction_count': len(ring_transactions),
                'total_amount': sum(t['amount'] for t in ring_transactions),
                'avg_amount': sum(t['amount'] for t in ring_transactions) / len(ring_transactions) if ring_transactions else 0
            }
        
        return ring_stats
    
    def get_entity_features(self, entity_type: str, entity_id: str) -> Dict:
        """Get features for a specific entity"""
        if entity_type == 'card' and entity_id in self.card_stats:
            return self.card_stats[entity_id]
        elif entity_type == 'merchant' and entity_id in self.merchant_stats:
            return self.merchant_stats[entity_id]
        else:
            return {}

# Initialize the fraud detector
fraud_detector = SimpleFraudDetector()

if __name__ == "__main__":
    # Test the model
    test_transaction = {
        'transaction_id': 'test_001',
        'card_id': 'card_1000001',  # From fraud ring 1
        'merchant_id': 'merchant_0201',  # From fraud ring 1
        'device_id': 'device_5000001',  # From fraud ring 1
        'ip_address': '192.168.1.1',  # From fraud ring 1
        'amount': 1500.0,
        'timestamp': datetime.now().isoformat(),
        'category': 'electronics',
        'country': 'US',
        'is_fraud': 1,
        'fraud_ring': 'ring_1'
    }
    
    prediction = fraud_detector.predict(test_transaction)
    print("Test prediction:", prediction)
    
    # Get fraud rings info
    rings = fraud_detector.get_fraud_rings()
    print("\nFraud rings:", json.dumps(rings, indent=2))