"""
Advanced fraud data generator with realistic fraud rings and patterns.
Creates transaction data with cards, merchants, devices, IPs, and fraud rings.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse
import os

class FraudDataGenerator:
    def __init__(self, n_transactions: int = 50000, fraud_rate: float = 0.05, n_fraud_rings: int = 10):
        self.n_transactions = n_transactions
        self.fraud_rate = fraud_rate
        self.n_fraud_rings = n_fraud_rings
        
        # Generate entities
        self.n_cards = max(1000, n_transactions // 20)
        self.n_merchants = max(500, n_transactions // 50)
        self.n_devices = max(800, n_transactions // 25)
        self.n_ips = max(600, n_transactions // 30)
        
        # Seed for reproducibility
        np.random.seed(42)
        random.seed(42)
        
    def generate_fraud_rings(self) -> Dict[str, List]:
        """Generate fraud rings with shared entities"""
        fraud_rings = {}
        
        for ring_id in range(self.n_fraud_rings):
            ring_size = np.random.randint(5, 15)  # 5-15 transactions per ring
            
            # Each ring shares some entities
            shared_cards = np.random.choice(self.n_cards, size=np.random.randint(2, 5), replace=False)
            shared_merchants = np.random.choice(self.n_merchants, size=np.random.randint(1, 3), replace=False)
            shared_devices = np.random.choice(self.n_devices, size=np.random.randint(1, 4), replace=False)
            shared_ips = np.random.choice(self.n_ips, size=np.random.randint(1, 3), replace=False)
            
            fraud_rings[f'ring_{ring_id}'] = {
                'size': ring_size,
                'cards': shared_cards.tolist(),
                'merchants': shared_merchants.tolist(),
                'devices': shared_devices.tolist(),
                'ips': shared_ips.tolist()
            }
            
        return fraud_rings
    
    def generate_transactions(self) -> pd.DataFrame:
        """Generate transaction data with fraud patterns"""
        fraud_rings = self.generate_fraud_rings()
        
        transactions = []
        fraud_transactions = int(self.n_transactions * self.fraud_rate)
        
        # Generate fraud transactions from rings
        ring_transactions = 0
        for ring_id, ring_data in fraud_rings.items():
            for _ in range(ring_data['size']):
                if ring_transactions >= fraud_transactions:
                    break
                    
                transaction = self._create_fraud_transaction(ring_data, ring_id)
                transactions.append(transaction)
                ring_transactions += 1
        
        # Generate remaining fraud transactions (not in rings)
        for _ in range(fraud_transactions - ring_transactions):
            transaction = self._create_random_fraud_transaction()
            transactions.append(transaction)
        
        # Generate legitimate transactions
        for _ in range(self.n_transactions - fraud_transactions):
            transaction = self._create_legitimate_transaction()
            transactions.append(transaction)
        
        # Shuffle transactions
        random.shuffle(transactions)
        
        # Add transaction IDs and timestamps
        base_time = datetime.now() - timedelta(days=30)
        for i, txn in enumerate(transactions):
            txn['transaction_id'] = f'txn_{i:06d}'
            txn['timestamp'] = base_time + timedelta(
                seconds=np.random.randint(0, 30 * 24 * 3600)
            )
        
        return pd.DataFrame(transactions)
    
    def _create_fraud_transaction(self, ring_data: Dict, ring_id: str) -> Dict:
        """Create a fraudulent transaction within a ring"""
        return {
            'card_id': f'card_{np.random.choice(ring_data["cards"])}',
            'merchant_id': f'merchant_{np.random.choice(ring_data["merchants"])}',
            'device_id': f'device_{np.random.choice(ring_data["devices"])}',
            'ip': self._generate_ip(np.random.choice(ring_data["ips"])),
            'amount': np.random.exponential(200) + 50,  # Higher amounts for fraud
            'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], p=[0.7, 0.2, 0.1]),
            'merchant_category': self._get_fraud_merchant_category(),
            'hour': np.random.choice(range(24), p=self._get_fraud_hour_distribution()),
            'day_of_week': np.random.randint(0, 7),
            'fraud': 1,
            'fraud_ring': ring_id,
            'velocity_1h': np.random.poisson(3) + 1,  # Higher velocity for fraud
            'velocity_24h': np.random.poisson(8) + 2,
            'amount_std_dev': np.random.exponential(2) + 1,
            'location_risk_score': np.random.beta(2, 1),  # Higher risk for fraud locations
        }
    
    def _create_random_fraud_transaction(self) -> Dict:
        """Create a random fraudulent transaction (not in a ring)"""
        return {
            'card_id': f'card_{np.random.randint(0, self.n_cards)}',
            'merchant_id': f'merchant_{np.random.randint(0, self.n_merchants)}',
            'device_id': f'device_{np.random.randint(0, self.n_devices)}',
            'ip': self._generate_ip(np.random.randint(0, self.n_ips)),
            'amount': np.random.exponential(150) + 30,
            'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], p=[0.6, 0.3, 0.1]),
            'merchant_category': self._get_fraud_merchant_category(),
            'hour': np.random.choice(range(24), p=self._get_fraud_hour_distribution()),
            'day_of_week': np.random.randint(0, 7),
            'fraud': 1,
            'fraud_ring': 'isolated',
            'velocity_1h': np.random.poisson(2) + 1,
            'velocity_24h': np.random.poisson(5) + 1,
            'amount_std_dev': np.random.exponential(1.5) + 0.5,
            'location_risk_score': np.random.beta(1.5, 1),
        }
    
    def _create_legitimate_transaction(self) -> Dict:
        """Create a legitimate transaction"""
        return {
            'card_id': f'card_{np.random.randint(0, self.n_cards)}',
            'merchant_id': f'merchant_{np.random.randint(0, self.n_merchants)}',
            'device_id': f'device_{np.random.randint(0, self.n_devices)}',
            'ip': self._generate_ip(np.random.randint(0, self.n_ips)),
            'amount': np.random.lognormal(3, 1) + 10,  # More typical spending patterns
            'transaction_type': np.random.choice(['purchase', 'withdrawal', 'transfer'], p=[0.8, 0.15, 0.05]),
            'merchant_category': self._get_legitimate_merchant_category(),
            'hour': np.random.choice(range(24), p=self._get_legitimate_hour_distribution()),
            'day_of_week': np.random.randint(0, 7),
            'fraud': 0,
            'fraud_ring': 'none',
            'velocity_1h': np.random.poisson(0.5),
            'velocity_24h': np.random.poisson(2),
            'amount_std_dev': np.random.exponential(0.5),
            'location_risk_score': np.random.beta(1, 2),  # Lower risk for legitimate
        }
    
    def _generate_ip(self, ip_id: int) -> str:
        """Generate realistic IP addresses"""
        if ip_id < 100:  # Some shared IPs for fraud rings
            return f"192.168.1.{ip_id}"
        else:
            return f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    
    def _get_fraud_merchant_category(self) -> str:
        """Get merchant categories more common in fraud"""
        categories = ['online_retail', 'electronics', 'cash_advance', 'gambling', 'adult_services', 'travel']
        return np.random.choice(categories, p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05])
    
    def _get_legitimate_merchant_category(self) -> str:
        """Get merchant categories more common in legitimate transactions"""
        categories = ['grocery', 'gas_station', 'restaurant', 'retail', 'pharmacy', 'online_retail', 'entertainment']
        return np.random.choice(categories, p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05])
    
    def _get_fraud_hour_distribution(self) -> np.ndarray:
        """Hour distribution for fraudulent transactions (more at night)"""
        hours = np.ones(24)
        hours[22:24] *= 3  # Late night
        hours[0:6] *= 2.5  # Early morning
        return hours / hours.sum()
    
    def _get_legitimate_hour_distribution(self) -> np.ndarray:
        """Hour distribution for legitimate transactions (more during day)"""
        hours = np.ones(24)
        hours[8:18] *= 2  # Business hours
        hours[18:22] *= 1.5  # Evening
        return hours / hours.sum()

def main():
    parser = argparse.ArgumentParser(description="Generate fraud detection dataset")
    parser.add_argument('--n_transactions', type=int, default=50000, help='Number of transactions to generate')
    parser.add_argument('--fraud_rate', type=float, default=0.05, help='Fraction of fraudulent transactions')
    parser.add_argument('--n_fraud_rings', type=int, default=10, help='Number of fraud rings to generate')
    parser.add_argument('--output', type=str, default='data/raw/fraud_transactions.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate data
    generator = FraudDataGenerator(
        n_transactions=args.n_transactions,
        fraud_rate=args.fraud_rate,
        n_fraud_rings=args.n_fraud_rings
    )
    
    print(f"Generating {args.n_transactions} transactions with {args.fraud_rate:.1%} fraud rate...")
    df = generator.generate_transactions()
    
    # Save data
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['fraud'].sum()} ({df['fraud'].mean():.1%})")
    print(f"Fraud rings: {len(df[df['fraud_ring'] != 'none']['fraud_ring'].unique())}")
    print(f"Unique cards: {df['card_id'].nunique()}")
    print(f"Unique merchants: {df['merchant_id'].nunique()}")
    print(f"Unique devices: {df['device_id'].nunique()}")
    print(f"Unique IPs: {df['ip'].nunique()}")

if __name__ == "__main__":
    main()