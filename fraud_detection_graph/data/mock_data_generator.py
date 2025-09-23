"""
Mock Data Generator for Credit Card Fraud Detection
Generates synthetic data for cards, merchants, devices, IPs, and transactions
with realistic fraud patterns and relationships.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

class FraudDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration parameters
        self.num_cards = 10000
        self.num_merchants = 1000
        self.num_devices = 5000
        self.num_ips = 2000
        self.num_transactions = 50000
        self.fraud_rate = 0.02  # 2% fraud rate
        
        # Fraud ring parameters
        self.num_fraud_rings = 15
        self.ring_size_range = (3, 12)
        
    def generate_cards(self) -> pd.DataFrame:
        """Generate synthetic credit card data"""
        cards = []
        
        for i in range(self.num_cards):
            card = {
                'card_id': f'card_{i:06d}',
                'card_number': self._generate_card_number(),
                'card_type': np.random.choice(['Visa', 'MasterCard', 'Amex', 'Discover'], 
                                            p=[0.5, 0.3, 0.15, 0.05]),
                'credit_limit': np.random.lognormal(9, 0.5) * 100,  # Log-normal distribution
                'issue_date': self._random_date_past(365*3),
                'customer_age': np.random.normal(45, 15),
                'customer_income': np.random.lognormal(10.5, 0.8),
                'risk_score': np.random.beta(2, 8),  # Most customers are low risk
                'is_compromised': False
            }
            cards.append(card)
            
        return pd.DataFrame(cards)
    
    def generate_merchants(self) -> pd.DataFrame:
        """Generate synthetic merchant data"""
        merchants = []
        merchant_categories = [
            'Gas Station', 'Grocery Store', 'Restaurant', 'Online Retail',
            'Department Store', 'ATM', 'Hotel', 'Airline', 'Pharmacy',
            'Electronics', 'Clothing', 'Entertainment'
        ]
        
        for i in range(self.num_merchants):
            merchant = {
                'merchant_id': f'merchant_{i:05d}',
                'merchant_name': f'Merchant_{i}',
                'category': np.random.choice(merchant_categories),
                'location_city': f'City_{np.random.randint(1, 100)}',
                'location_state': f'State_{np.random.randint(1, 50)}',
                'avg_transaction_amount': np.random.lognormal(4, 1),
                'fraud_history_count': np.random.poisson(0.5),
                'risk_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.8, 0.15, 0.05])
            }
            merchants.append(merchant)
            
        return pd.DataFrame(merchants)
    
    def generate_devices(self) -> pd.DataFrame:
        """Generate synthetic device data"""
        devices = []
        device_types = ['Mobile', 'Desktop', 'Tablet', 'ATM', 'POS']
        os_types = ['iOS', 'Android', 'Windows', 'MacOS', 'Linux']
        
        for i in range(self.num_devices):
            device = {
                'device_id': f'device_{i:06d}',
                'device_type': np.random.choice(device_types),
                'os_type': np.random.choice(os_types),
                'browser': np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge', 'Other']),
                'first_seen': self._random_date_past(365),
                'last_seen': self._random_date_past(7),
                'transaction_count': np.random.poisson(10),
                'is_suspicious': np.random.choice([True, False], p=[0.05, 0.95])
            }
            devices.append(device)
            
        return pd.DataFrame(devices)
    
    def generate_ips(self) -> pd.DataFrame:
        """Generate synthetic IP address data"""
        ips = []
        
        for i in range(self.num_ips):
            ip = {
                'ip_address': self._generate_ip(),
                'country': f'Country_{np.random.randint(1, 50)}',
                'region': f'Region_{np.random.randint(1, 20)}',
                'isp': f'ISP_{np.random.randint(1, 100)}',
                'is_vpn': np.random.choice([True, False], p=[0.1, 0.9]),
                'is_tor': np.random.choice([True, False], p=[0.02, 0.98]),
                'reputation_score': np.random.beta(8, 2),  # Most IPs have good reputation
                'first_seen': self._random_date_past(365),
                'last_seen': self._random_date_past(1)
            }
            ips.append(ip)
            
        return pd.DataFrame(ips)
    
    def generate_fraud_rings(self, cards_df: pd.DataFrame, merchants_df: pd.DataFrame, 
                           devices_df: pd.DataFrame, ips_df: pd.DataFrame) -> List[Dict]:
        """Generate fraud rings - groups of connected entities involved in fraud"""
        fraud_rings = []
        
        for ring_id in range(self.num_fraud_rings):
            ring_size = np.random.randint(*self.ring_size_range)
            
            # Select entities for this fraud ring
            ring_cards = np.random.choice(cards_df['card_id'], 
                                        size=min(ring_size, len(cards_df)), 
                                        replace=False)
            ring_merchants = np.random.choice(merchants_df['merchant_id'], 
                                            size=min(max(1, ring_size//3), len(merchants_df)), 
                                            replace=False)
            ring_devices = np.random.choice(devices_df['device_id'], 
                                          size=min(max(1, ring_size//2), len(devices_df)), 
                                          replace=False)
            ring_ips = np.random.choice(ips_df['ip_address'], 
                                      size=min(max(1, ring_size//4), len(ips_df)), 
                                      replace=False)
            
            fraud_ring = {
                'ring_id': f'ring_{ring_id:03d}',
                'ring_type': np.random.choice(['Account_Takeover', 'Card_Testing', 'Collusion', 'Money_Laundering']),
                'cards': ring_cards.tolist(),
                'merchants': ring_merchants.tolist(),
                'devices': ring_devices.tolist(),
                'ips': ring_ips.tolist(),
                'activity_start': self._random_date_past(90),
                'activity_end': self._random_date_past(7),
                'estimated_loss': np.random.lognormal(8, 1.5)
            }
            fraud_rings.append(fraud_ring)
            
        return fraud_rings
    
    def generate_transactions(self, cards_df: pd.DataFrame, merchants_df: pd.DataFrame,
                            devices_df: pd.DataFrame, ips_df: pd.DataFrame,
                            fraud_rings: List[Dict]) -> pd.DataFrame:
        """Generate synthetic transaction data with fraud patterns"""
        transactions = []
        
        # Create mapping for fraud ring entities
        fraud_entities = {'cards': set(), 'merchants': set(), 'devices': set(), 'ips': set()}
        for ring in fraud_rings:
            fraud_entities['cards'].update(ring['cards'])
            fraud_entities['merchants'].update(ring['merchants'])
            fraud_entities['devices'].update(ring['devices'])
            fraud_entities['ips'].update(ring['ips'])
        
        for i in range(self.num_transactions):
            # Select entities for transaction
            card_id = np.random.choice(cards_df['card_id'])
            merchant_id = np.random.choice(merchants_df['merchant_id'])
            device_id = np.random.choice(devices_df['device_id'])
            ip_address = np.random.choice(ips_df['ip_address'])
            
            # Determine if transaction is fraudulent
            is_fraud = self._determine_fraud(card_id, merchant_id, device_id, ip_address, fraud_entities)
            
            # Generate transaction details
            base_amount = np.random.lognormal(4, 1.5)
            if is_fraud:
                # Fraudulent transactions tend to be higher amounts
                amount = base_amount * np.random.lognormal(0.5, 0.8)
            else:
                amount = base_amount
                
            transaction = {
                'transaction_id': f'txn_{i:08d}',
                'card_id': card_id,
                'merchant_id': merchant_id,
                'device_id': device_id,
                'ip_address': ip_address,
                'amount': round(amount, 2),
                'timestamp': self._random_timestamp_recent(),
                'is_fraud': is_fraud,
                'fraud_type': self._get_fraud_type(card_id, merchant_id, device_id, ip_address, fraud_rings) if is_fraud else None,
                'response_time_ms': np.random.exponential(200),
                'merchant_category': merchants_df[merchants_df['merchant_id'] == merchant_id]['category'].iloc[0]
            }
            transactions.append(transaction)
            
        return pd.DataFrame(transactions)
    
    def _generate_card_number(self) -> str:
        """Generate a fake credit card number"""
        # Generate 16-digit card number (not real)
        return ''.join([str(np.random.randint(0, 10)) for _ in range(16)])
    
    def _generate_ip(self) -> str:
        """Generate a fake IP address"""
        return f"{np.random.randint(1, 255)}.{np.random.randint(0, 255)}.{np.random.randint(0, 255)}.{np.random.randint(1, 255)}"
    
    def _random_date_past(self, max_days_ago: int) -> str:
        """Generate a random date in the past"""
        days_ago = np.random.randint(0, max_days_ago)
        date = datetime.now() - timedelta(days=days_ago)
        return date.strftime('%Y-%m-%d')
    
    def _random_timestamp_recent(self) -> str:
        """Generate a random timestamp in the last 30 days"""
        days_ago = np.random.randint(0, 30)
        hours = np.random.randint(0, 24)
        minutes = np.random.randint(0, 60)
        seconds = np.random.randint(0, 60)
        
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes, seconds=seconds)
        return timestamp.strftime('%Y-%m-%d %H:%M:%S')
    
    def _determine_fraud(self, card_id: str, merchant_id: str, device_id: str, 
                        ip_address: str, fraud_entities: Dict) -> bool:
        """Determine if a transaction should be fraudulent based on entity involvement in fraud rings"""
        # Base fraud probability
        fraud_prob = self.fraud_rate
        
        # Increase probability if entities are in fraud rings
        if card_id in fraud_entities['cards']:
            fraud_prob *= 10
        if merchant_id in fraud_entities['merchants']:
            fraud_prob *= 5
        if device_id in fraud_entities['devices']:
            fraud_prob *= 8
        if ip_address in fraud_entities['ips']:
            fraud_prob *= 6
            
        return np.random.random() < min(fraud_prob, 0.95)
    
    def _get_fraud_type(self, card_id: str, merchant_id: str, device_id: str,
                       ip_address: str, fraud_rings: List[Dict]) -> str:
        """Get the fraud type based on which ring the entities belong to"""
        for ring in fraud_rings:
            if (card_id in ring['cards'] or merchant_id in ring['merchants'] or
                device_id in ring['devices'] or ip_address in ring['ips']):
                return ring['ring_type']
        return 'Individual_Fraud'
    
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic data"""
        print("Generating cards...")
        cards_df = self.generate_cards()
        
        print("Generating merchants...")
        merchants_df = self.generate_merchants()
        
        print("Generating devices...")
        devices_df = self.generate_devices()
        
        print("Generating IPs...")
        ips_df = self.generate_ips()
        
        print("Generating fraud rings...")
        fraud_rings = self.generate_fraud_rings(cards_df, merchants_df, devices_df, ips_df)
        
        print("Generating transactions...")
        transactions_df = self.generate_transactions(cards_df, merchants_df, devices_df, ips_df, fraud_rings)
        
        # Save fraud rings as JSON
        with open('/workspace/fraud_detection_graph/data/fraud_rings.json', 'w') as f:
            json.dump(fraud_rings, f, indent=2, default=str)
        
        return {
            'cards': cards_df,
            'merchants': merchants_df,
            'devices': devices_df,
            'ips': ips_df,
            'transactions': transactions_df,
            'fraud_rings': fraud_rings
        }

if __name__ == "__main__":
    generator = FraudDataGenerator()
    data = generator.generate_all_data()
    
    # Save all dataframes
    for name, df in data.items():
        if isinstance(df, pd.DataFrame):
            df.to_csv(f'/workspace/fraud_detection_graph/data/{name}.csv', index=False)
            print(f"Saved {name}.csv with {len(df)} records")
    
    print(f"Generated {len(data['fraud_rings'])} fraud rings")
    print(f"Fraud rate in transactions: {data['transactions']['is_fraud'].mean():.3f}")