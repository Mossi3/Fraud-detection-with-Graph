"""
Data generator for creating synthetic fraud detection datasets.
Generates realistic transaction patterns with labeled fraud cases.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Tuple, Any
from faker import Faker
import networkx as nx
from loguru import logger

fake = Faker()


class TransactionDataGenerator:
    """Generates synthetic transaction data with fraud patterns"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        self.cards = []
        self.merchants = []
        self.devices = []
        self.ips = []
        
        # Configuration
        self.fraud_rate = 0.02  # 2% fraud rate
        self.start_date = datetime.now() - timedelta(days=90)
        
    def generate_entities(self, num_cards: int = 1000, 
                         num_merchants: int = 500,
                         num_devices: int = 1200,
                         num_ips: int = 2000):
        """Generate entity pools"""
        logger.info(f"Generating entities: {num_cards} cards, {num_merchants} merchants")
        
        # Generate cards
        self.cards = [self._generate_card() for _ in range(num_cards)]
        
        # Generate merchants with categories
        merchant_categories = [
            'grocery', 'gas', 'restaurant', 'retail', 'online', 
            'travel', 'entertainment', 'utilities', 'healthcare', 'other'
        ]
        
        self.merchants = []
        for _ in range(num_merchants):
            merchant = {
                'id': f"merchant_{fake.company().replace(' ', '_').lower()}_{fake.uuid4()[:6]}",
                'category': random.choice(merchant_categories),
                'risk_level': np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.25, 0.05]),
                'location': (fake.latitude(), fake.longitude())
            }
            self.merchants.append(merchant)
        
        # Generate devices
        self.devices = [self._generate_device() for _ in range(num_devices)]
        
        # Generate IP addresses
        self.ips = [self._generate_ip() for _ in range(num_ips)]
        
    def _generate_card(self) -> Dict[str, Any]:
        """Generate card entity"""
        return {
            'id': f"card_{''.join(random.choices('0123456789', k=16))}",
            'creation_date': self.start_date - timedelta(days=random.randint(0, 365)),
            'credit_limit': random.choice([1000, 2500, 5000, 10000, 25000]),
            'card_type': random.choice(['debit', 'credit']),
            'issuer': random.choice(['bank_a', 'bank_b', 'bank_c', 'bank_d'])
        }
    
    def _generate_device(self) -> Dict[str, Any]:
        """Generate device entity"""
        return {
            'id': f"device_{fake.uuid4()[:12]}",
            'type': random.choice(['mobile', 'desktop', 'tablet', 'pos_terminal']),
            'os': random.choice(['ios', 'android', 'windows', 'macos', 'linux', 'other']),
            'trusted': random.random() > 0.1  # 90% trusted devices
        }
    
    def _generate_ip(self) -> Dict[str, Any]:
        """Generate IP entity"""
        ip_type = random.choices(['residential', 'commercial', 'vpn', 'tor'], 
                                weights=[0.7, 0.2, 0.08, 0.02])[0]
        return {
            'address': fake.ipv4(),
            'type': ip_type,
            'country': fake.country_code(),
            'risk_score': 0.1 if ip_type == 'residential' else 0.5 if ip_type == 'vpn' else 0.9
        }
    
    def generate_normal_transactions(self, num_transactions: int = 10000) -> pd.DataFrame:
        """Generate normal transaction patterns"""
        logger.info(f"Generating {num_transactions} normal transactions")
        
        transactions = []
        
        for _ in range(num_transactions):
            # Select entities
            card = random.choice(self.cards)
            merchant = random.choice(self.merchants)
            
            # Card-device association (people usually use same devices)
            if random.random() < 0.8:  # 80% use regular device
                device_pool = random.sample(self.devices, k=min(3, len(self.devices)))
                device = random.choice(device_pool)
            else:
                device = random.choice(self.devices)
            
            # Similar for IPs
            if random.random() < 0.7:  # 70% use regular IPs
                ip_pool = random.sample(self.ips, k=min(5, len(self.ips)))
                ip = random.choice(ip_pool)
            else:
                ip = random.choice(self.ips)
            
            # Generate transaction
            txn = self._create_normal_transaction(card, merchant, device, ip)
            transactions.append(txn)
        
        return pd.DataFrame(transactions)
    
    def _create_normal_transaction(self, card: Dict, merchant: Dict, 
                                 device: Dict, ip: Dict) -> Dict[str, Any]:
        """Create a normal transaction"""
        # Time patterns - more transactions during day
        hour = np.random.choice(range(24), p=self._get_hour_distribution())
        timestamp = self.start_date + timedelta(
            days=random.randint(0, 90),
            hours=hour,
            minutes=random.randint(0, 59)
        )
        
        # Amount based on merchant category
        amount = self._get_normal_amount(merchant['category'])
        
        return {
            'transaction_id': f"txn_{fake.uuid4()}",
            'card_id': card['id'],
            'merchant_id': merchant['id'],
            'amount': amount,
            'timestamp': timestamp,
            'device_id': device['id'],
            'ip_address': ip['address'],
            'merchant_category': merchant['category'],
            'location_lat': merchant['location'][0],
            'location_lon': merchant['location'][1],
            'is_fraud': False,
            'fraud_type': None
        }
    
    def _get_hour_distribution(self) -> np.ndarray:
        """Get probability distribution for transaction hours"""
        # Peak hours: 10-12, 14-16, 19-21
        probs = np.ones(24) * 0.02  # Base probability
        probs[10:13] = 0.08  # Morning peak
        probs[14:17] = 0.06  # Afternoon
        probs[19:22] = 0.07  # Evening peak
        probs[0:6] = 0.01    # Night - low activity
        return probs / probs.sum()
    
    def _get_normal_amount(self, category: str) -> float:
        """Get normal transaction amount for category"""
        amounts = {
            'grocery': (20, 150, 50),
            'gas': (30, 80, 45),
            'restaurant': (15, 100, 35),
            'retail': (25, 300, 75),
            'online': (10, 500, 80),
            'travel': (100, 2000, 400),
            'entertainment': (20, 200, 50),
            'utilities': (50, 300, 120),
            'healthcare': (50, 500, 150),
            'other': (10, 200, 50)
        }
        
        min_amt, max_amt, mean_amt = amounts.get(category, (10, 200, 50))
        amount = np.random.gamma(2, mean_amt/2)
        return round(np.clip(amount, min_amt, max_amt), 2)
    
    def inject_fraud_patterns(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Inject various fraud patterns into transactions"""
        logger.info("Injecting fraud patterns")
        
        df = transactions_df.copy()
        total_frauds = int(len(df) * self.fraud_rate)
        
        # Distribute fraud types
        fraud_distribution = {
            'card_theft': 0.3,
            'account_takeover': 0.2,
            'merchant_fraud': 0.15,
            'velocity_attack': 0.15,
            'device_sharing': 0.1,
            'location_anomaly': 0.1
        }
        
        fraud_counts = {
            fraud_type: int(total_frauds * ratio) 
            for fraud_type, ratio in fraud_distribution.items()
        }
        
        # Apply each fraud pattern
        fraud_indices = set()
        
        for fraud_type, count in fraud_counts.items():
            if fraud_type == 'card_theft':
                indices = self._inject_card_theft(df, count)
            elif fraud_type == 'account_takeover':
                indices = self._inject_account_takeover(df, count)
            elif fraud_type == 'merchant_fraud':
                indices = self._inject_merchant_fraud(df, count)
            elif fraud_type == 'velocity_attack':
                indices = self._inject_velocity_attack(df, count)
            elif fraud_type == 'device_sharing':
                indices = self._inject_device_sharing(df, count)
            elif fraud_type == 'location_anomaly':
                indices = self._inject_location_anomaly(df, count)
            
            fraud_indices.update(indices)
        
        # Mark fraud transactions
        df.loc[list(fraud_indices), 'is_fraud'] = True
        
        logger.info(f"Injected {len(fraud_indices)} fraudulent transactions")
        return df
    
    def _inject_card_theft(self, df: pd.DataFrame, count: int) -> List[int]:
        """Inject card theft pattern"""
        indices = []
        
        for _ in range(count // 5):  # Each theft involves ~5 transactions
            # Select a card
            card_id = random.choice(self.cards)['id']
            card_txns = df[df['card_id'] == card_id].index.tolist()
            
            if len(card_txns) > 5:
                # Pick a point for theft
                theft_point = random.choice(card_txns[:-5])
                theft_time = df.loc[theft_point, 'timestamp']
                
                # Create burst of high-value transactions
                new_device = random.choice(self.devices)['id']
                new_ip = random.choice([ip for ip in self.ips if ip['type'] in ['vpn', 'tor']])['address']
                
                for i in range(5):
                    idx = len(df) + len(indices)
                    fraud_txn = {
                        'transaction_id': f"fraud_txn_{fake.uuid4()}",
                        'card_id': card_id,
                        'merchant_id': random.choice([m['id'] for m in self.merchants if m['risk_level'] == 'high']),
                        'amount': random.uniform(500, 2000),
                        'timestamp': theft_time + timedelta(minutes=i*10),
                        'device_id': new_device,
                        'ip_address': new_ip,
                        'is_fraud': True,
                        'fraud_type': 'card_theft'
                    }
                    indices.append(idx)
                    df = df.append(fraud_txn, ignore_index=True)
        
        return indices
    
    def _inject_account_takeover(self, df: pd.DataFrame, count: int) -> List[int]:
        """Inject account takeover pattern"""
        indices = []
        
        for _ in range(count // 8):
            card_id = random.choice(self.cards)['id']
            card_txns = df[df['card_id'] == card_id]
            
            if len(card_txns) > 0:
                # Get normal spending pattern
                normal_amount = card_txns['amount'].median()
                
                # Takeover point
                takeover_time = df['timestamp'].max() - timedelta(days=random.randint(1, 7))
                
                # Changed behavior
                for i in range(8):
                    idx = len(df) + len(indices)
                    fraud_txn = {
                        'transaction_id': f"fraud_ato_{fake.uuid4()}",
                        'card_id': card_id,
                        'merchant_id': random.choice([m['id'] for m in self.merchants]),
                        'amount': normal_amount * random.uniform(5, 20),  # Much higher than normal
                        'timestamp': takeover_time + timedelta(hours=i),
                        'device_id': random.choice(self.devices)['id'],  # New device
                        'ip_address': random.choice(self.ips)['address'],
                        'is_fraud': True,
                        'fraud_type': 'account_takeover'
                    }
                    indices.append(idx)
                    df = df.append(fraud_txn, ignore_index=True)
        
        return indices
    
    def _inject_merchant_fraud(self, df: pd.DataFrame, count: int) -> List[int]:
        """Inject merchant fraud pattern"""
        indices = []
        
        # Create fraudulent merchants
        fraud_merchants = [f"fraud_merchant_{i}" for i in range(3)]
        
        # Multiple cards transacting with fraud merchants
        cards_involved = random.sample(self.cards, k=min(20, len(self.cards)))
        
        for i in range(count):
            idx = len(df) + len(indices)
            card = random.choice(cards_involved)
            
            fraud_txn = {
                'transaction_id': f"fraud_merchant_{fake.uuid4()}",
                'card_id': card['id'],
                'merchant_id': random.choice(fraud_merchants),
                'amount': random.uniform(100, 1000),
                'timestamp': self.start_date + timedelta(days=random.randint(0, 90)),
                'device_id': random.choice(self.devices)['id'],
                'ip_address': random.choice(self.ips)['address'],
                'is_fraud': True,
                'fraud_type': 'merchant_fraud'
            }
            indices.append(idx)
            df = df.append(fraud_txn, ignore_index=True)
        
        return indices
    
    def _inject_velocity_attack(self, df: pd.DataFrame, count: int) -> List[int]:
        """Inject high-velocity fraud pattern"""
        indices = []
        
        for _ in range(count // 10):
            card_id = random.choice(self.cards)['id']
            attack_time = df['timestamp'].max() - timedelta(hours=random.randint(1, 48))
            
            # Rapid sequence of transactions
            for i in range(10):
                idx = len(df) + len(indices)
                fraud_txn = {
                    'transaction_id': f"fraud_velocity_{fake.uuid4()}",
                    'card_id': card_id,
                    'merchant_id': random.choice(self.merchants)['id'],
                    'amount': random.uniform(100, 500),
                    'timestamp': attack_time + timedelta(minutes=i*2),  # Every 2 minutes
                    'device_id': random.choice(self.devices)['id'],
                    'ip_address': random.choice(self.ips)['address'],
                    'is_fraud': True,
                    'fraud_type': 'velocity_attack'
                }
                indices.append(idx)
                df = df.append(fraud_txn, ignore_index=True)
        
        return indices
    
    def _inject_device_sharing(self, df: pd.DataFrame, count: int) -> List[int]:
        """Inject device sharing fraud pattern"""
        indices = []
        
        # Select a device used by multiple cards
        shared_device = random.choice(self.devices)['id']
        cards_on_device = random.sample(self.cards, k=min(5, len(self.cards)))
        
        for i in range(count):
            idx = len(df) + len(indices)
            card = random.choice(cards_on_device)
            
            fraud_txn = {
                'transaction_id': f"fraud_device_{fake.uuid4()}",
                'card_id': card['id'],
                'merchant_id': random.choice(self.merchants)['id'],
                'amount': random.uniform(50, 500),
                'timestamp': self.start_date + timedelta(days=random.randint(0, 90)),
                'device_id': shared_device,
                'ip_address': random.choice(self.ips)['address'],
                'is_fraud': True,
                'fraud_type': 'device_sharing'
            }
            indices.append(idx)
            df = df.append(fraud_txn, ignore_index=True)
        
        return indices
    
    def _inject_location_anomaly(self, df: pd.DataFrame, count: int) -> List[int]:
        """Inject location anomaly fraud pattern"""
        indices = []
        
        for i in range(count):
            card_id = random.choice(self.cards)['id']
            
            # Transaction from unusual location
            idx = len(df) + len(indices)
            fraud_txn = {
                'transaction_id': f"fraud_location_{fake.uuid4()}",
                'card_id': card_id,
                'merchant_id': random.choice(self.merchants)['id'],
                'amount': random.uniform(100, 1000),
                'timestamp': self.start_date + timedelta(days=random.randint(0, 90)),
                'device_id': random.choice(self.devices)['id'],
                'ip_address': random.choice([ip for ip in self.ips if ip['country'] != 'US'])['address'],
                'location_lat': fake.latitude(),
                'location_lon': fake.longitude(),
                'is_fraud': True,
                'fraud_type': 'location_anomaly'
            }
            indices.append(idx)
            df = df.append(fraud_txn, ignore_index=True)
        
        return indices
    
    def generate_dataset(self, num_transactions: int = 50000) -> pd.DataFrame:
        """Generate complete dataset with normal and fraudulent transactions"""
        # Generate entities
        self.generate_entities()
        
        # Generate normal transactions
        df = self.generate_normal_transactions(int(num_transactions * 0.95))
        
        # Inject fraud patterns
        df = self.inject_fraud_patterns(df)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Add additional features
        df = self._add_derived_features(df)
        
        logger.info(f"Generated dataset with {len(df)} transactions, "
                   f"{df['is_fraud'].sum()} fraudulent ({df['is_fraud'].mean()*100:.2f}%)")
        
        return df
    
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features useful for fraud detection"""
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        
        # Velocity features (transactions per card in last hour)
        df['card_velocity_1h'] = 0
        for card_id in df['card_id'].unique():
            card_mask = df['card_id'] == card_id
            card_txns = df[card_mask].copy()
            
            for idx in card_txns.index:
                timestamp = df.loc[idx, 'timestamp']
                recent_mask = (card_txns['timestamp'] >= timestamp - timedelta(hours=1)) & \
                            (card_txns['timestamp'] < timestamp)
                df.loc[idx, 'card_velocity_1h'] = recent_mask.sum()
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filepath: str):
        """Save dataset to file"""
        df.to_csv(filepath, index=False)
        logger.info(f"Dataset saved to {filepath}")
        
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """Load dataset from file"""
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df


if __name__ == "__main__":
    # Generate sample dataset
    generator = TransactionDataGenerator()
    df = generator.generate_dataset(num_transactions=10000)
    
    # Save dataset
    generator.save_dataset(df, "data/fraud_detection_dataset.csv")
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"\nFraud type distribution:")
    print(df[df['is_fraud']]['fraud_type'].value_counts())
    print(f"\nAmount statistics:")
    print(df.groupby('is_fraud')['amount'].describe())