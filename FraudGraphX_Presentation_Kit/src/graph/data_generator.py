"""
Synthetic Fraud Ring Data Generator
Generates realistic fraud ring patterns for testing and validation
"""

import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import uuid
import ipaddress

logger = logging.getLogger(__name__)

class FraudRingGenerator:
    """
    Generate synthetic fraud ring data with realistic patterns
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Fraud ring configurations
        self.ring_configs = {
            'card_testing_ring': {
                'size_range': (5, 15),
                'fraud_rate': 0.8,
                'merchant_overlap': 0.3,
                'device_overlap': 0.7,
                'ip_overlap': 0.9,
                'temporal_pattern': 'burst',
                'amount_pattern': 'low_high'
            },
            'merchant_collusion': {
                'size_range': (3, 8),
                'fraud_rate': 0.6,
                'merchant_overlap': 0.9,
                'device_overlap': 0.2,
                'ip_overlap': 0.4,
                'temporal_pattern': 'consistent',
                'amount_pattern': 'medium'
            },
            'device_farming': {
                'size_range': (10, 25),
                'fraud_rate': 0.7,
                'merchant_overlap': 0.1,
                'device_overlap': 0.8,
                'ip_overlap': 0.6,
                'temporal_pattern': 'distributed',
                'amount_pattern': 'low'
            },
            'ip_proxy_ring': {
                'size_range': (8, 20),
                'fraud_rate': 0.9,
                'merchant_overlap': 0.2,
                'device_overlap': 0.3,
                'ip_overlap': 0.95,
                'temporal_pattern': 'burst',
                'amount_pattern': 'high'
            }
        }
    
    def generate_fraud_rings(self, num_rings: int = 5, 
                           ring_types: List[str] = None,
                           base_transactions: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic fraud rings with transaction data
        
        Args:
            num_rings: Number of fraud rings to generate
            ring_types: Types of rings to generate
            base_transactions: Base number of legitimate transactions
            
        Returns:
            DataFrame with transaction data including fraud rings
        """
        
        if ring_types is None:
            ring_types = list(self.ring_configs.keys())
        
        logger.info(f"Generating {num_rings} fraud rings of types: {ring_types}")
        
        # Generate legitimate transactions first
        legitimate_df = self._generate_legitimate_transactions(base_transactions)
        
        # Generate fraud rings
        fraud_rings = []
        for i in range(num_rings):
            ring_type = random.choice(ring_types)
            ring_config = self.ring_configs[ring_type]
            
            ring_df = self._generate_single_ring(ring_type, ring_config, i)
            fraud_rings.append(ring_df)
        
        # Combine all data
        all_transactions = [legitimate_df] + fraud_rings
        combined_df = pd.concat(all_transactions, ignore_index=True)
        
        # Shuffle and add final features
        combined_df = combined_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        # Add ring identifiers
        combined_df = self._add_ring_identifiers(combined_df, fraud_rings)
        
        logger.info(f"Generated {len(combined_df)} total transactions with {len(fraud_rings)} fraud rings")
        
        return combined_df
    
    def _generate_legitimate_transactions(self, num_transactions: int) -> pd.DataFrame:
        """Generate legitimate transaction data"""
        
        transactions = []
        
        # Generate base entities
        num_cards = max(100, num_transactions // 50)
        num_merchants = max(50, num_transactions // 100)
        num_devices = max(200, num_transactions // 30)
        num_ips = max(100, num_transactions // 80)
        
        cards = [f"card_{i:06d}" for i in range(num_cards)]
        merchants = [f"merchant_{i:06d}" for i in range(num_merchants)]
        devices = [f"device_{i:06d}" for i in range(num_devices)]
        ips = [self._generate_ip() for _ in range(num_ips)]
        
        # Generate transactions
        for i in range(num_transactions):
            transaction = {
                'transaction_id': f"txn_{i:08d}",
                'card_id': random.choice(cards),
                'merchant_id': random.choice(merchants),
                'device_id': random.choice(devices),
                'ip': random.choice(ips),
                'amount': self._generate_amount('legitimate'),
                'fraud': 0,
                'timestamp': self._generate_timestamp(),
                'ring_id': None,
                'ring_type': None
            }
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def _generate_single_ring(self, ring_type: str, config: Dict, ring_id: int) -> pd.DataFrame:
        """Generate a single fraud ring"""
        
        ring_size = random.randint(config['size_range'][0], config['size_range'][1])
        
        # Generate ring entities
        ring_entities = self._generate_ring_entities(ring_type, config, ring_size)
        
        # Generate transactions within the ring
        transactions = []
        num_transactions = ring_size * random.randint(3, 8)  # 3-8 transactions per entity
        
        for i in range(num_transactions):
            # Select entities based on overlap patterns
            card_id = self._select_entity(ring_entities['cards'], config['merchant_overlap'])
            merchant_id = self._select_entity(ring_entities['merchants'], config['merchant_overlap'])
            device_id = self._select_entity(ring_entities['devices'], config['device_overlap'])
            ip = self._select_entity(ring_entities['ips'], config['ip_overlap'])
            
            # Determine if this transaction is fraudulent
            is_fraud = random.random() < config['fraud_rate']
            
            transaction = {
                'transaction_id': f"ring_{ring_id}_txn_{i:06d}",
                'card_id': card_id,
                'merchant_id': merchant_id,
                'device_id': device_id,
                'ip': ip,
                'amount': self._generate_amount(config['amount_pattern']),
                'fraud': 1 if is_fraud else 0,
                'timestamp': self._generate_timestamp(config['temporal_pattern']),
                'ring_id': ring_id,
                'ring_type': ring_type
            }
            transactions.append(transaction)
        
        return pd.DataFrame(transactions)
    
    def _generate_ring_entities(self, ring_type: str, config: Dict, ring_size: int) -> Dict[str, List]:
        """Generate entities for a fraud ring"""
        
        entities = {
            'cards': [],
            'merchants': [],
            'devices': [],
            'ips': []
        }
        
        # Generate cards
        num_cards = max(ring_size, int(ring_size * (1 + config['merchant_overlap'])))
        entities['cards'] = [f"fraud_card_{ring_type}_{i:04d}" for i in range(num_cards)]
        
        # Generate merchants
        num_merchants = max(1, int(ring_size * config['merchant_overlap']))
        entities['merchants'] = [f"fraud_merchant_{ring_type}_{i:04d}" for i in range(num_merchants)]
        
        # Generate devices
        num_devices = max(ring_size, int(ring_size * (1 + config['device_overlap'])))
        entities['devices'] = [f"fraud_device_{ring_type}_{i:04d}" for i in range(num_devices)]
        
        # Generate IPs
        num_ips = max(1, int(ring_size * config['ip_overlap']))
        entities['ips'] = [self._generate_ip() for _ in range(num_ips)]
        
        return entities
    
    def _select_entity(self, entities: List[str], overlap_prob: float) -> str:
        """Select an entity based on overlap probability"""
        
        if random.random() < overlap_prob and len(entities) > 1:
            # Select from overlapping entities (first few)
            overlap_size = max(1, len(entities) // 3)
            return random.choice(entities[:overlap_size])
        else:
            # Select from all entities
            return random.choice(entities)
    
    def _generate_amount(self, pattern: str) -> float:
        """Generate transaction amount based on pattern"""
        
        if pattern == 'legitimate':
            # Legitimate transactions: mostly small amounts with some large
            if random.random() < 0.8:
                return round(random.uniform(10, 200), 2)
            else:
                return round(random.uniform(200, 2000), 2)
        
        elif pattern == 'low':
            # Low amounts (card testing)
            return round(random.uniform(1, 50), 2)
        
        elif pattern == 'medium':
            # Medium amounts
            return round(random.uniform(50, 500), 2)
        
        elif pattern == 'high':
            # High amounts
            return round(random.uniform(500, 5000), 2)
        
        elif pattern == 'low_high':
            # Mix of low and high (card testing + large fraud)
            if random.random() < 0.7:
                return round(random.uniform(1, 50), 2)
            else:
                return round(random.uniform(1000, 10000), 2)
        
        else:
            return round(random.uniform(10, 500), 2)
    
    def _generate_timestamp(self, pattern: str = 'random') -> int:
        """Generate timestamp based on pattern"""
        
        base_time = datetime(2023, 1, 1)
        
        if pattern == 'random':
            # Random timestamps over a year
            days_offset = random.randint(0, 365)
            hours_offset = random.randint(0, 23)
            minutes_offset = random.randint(0, 59)
            
            timestamp = base_time + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
        
        elif pattern == 'burst':
            # Burst pattern: transactions clustered in short time periods
            burst_days = random.randint(0, 365)
            burst_hours = random.randint(0, 23)
            
            # Within a 2-hour window
            minutes_offset = random.randint(0, 120)
            timestamp = base_time + timedelta(days=burst_days, hours=burst_hours, minutes=minutes_offset)
        
        elif pattern == 'consistent':
            # Consistent pattern: regular intervals
            days_offset = random.randint(0, 365)
            # Regular intervals (every 2-4 hours)
            hour_interval = random.choice([2, 3, 4])
            hours_offset = random.randint(0, 23) // hour_interval * hour_interval
            minutes_offset = random.randint(0, 59)
            
            timestamp = base_time + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
        
        elif pattern == 'distributed':
            # Distributed pattern: spread throughout the day
            days_offset = random.randint(0, 365)
            hours_offset = random.randint(0, 23)
            minutes_offset = random.randint(0, 59)
            
            timestamp = base_time + timedelta(days=days_offset, hours=hours_offset, minutes=minutes_offset)
        
        else:
            timestamp = base_time + timedelta(days=random.randint(0, 365))
        
        return int(timestamp.timestamp())
    
    def _generate_ip(self) -> str:
        """Generate a random IP address"""
        
        # Generate random IP in common ranges
        ip_ranges = [
            (192, 168, 1, 0),    # Private range
            (10, 0, 0, 0),       # Private range
            (172, 16, 0, 0),     # Private range
            (203, 0, 113, 0),    # Test range
        ]
        
        base_range = random.choice(ip_ranges)
        
        if base_range[0] == 192:
            return f"192.168.{random.randint(1, 255)}.{random.randint(1, 254)}"
        elif base_range[0] == 10:
            return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        elif base_range[0] == 172:
            return f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
        else:
            return f"203.0.113.{random.randint(1, 254)}"
    
    def _add_ring_identifiers(self, df: pd.DataFrame, fraud_rings: List[pd.DataFrame]) -> pd.DataFrame:
        """Add ring identifiers to the dataframe"""
        
        # Create ring mapping
        ring_mapping = {}
        
        for i, ring_df in enumerate(fraud_rings):
            ring_type = ring_df['ring_type'].iloc[0]
            ring_id = ring_df['ring_id'].iloc[0]
            
            ring_mapping[f"{ring_type}_{ring_id}"] = {
                'ring_id': ring_id,
                'ring_type': ring_type,
                'size': len(ring_df),
                'fraud_rate': ring_df['fraud'].mean()
            }
        
        # Add ring information to legitimate transactions
        df.loc[df['ring_id'].isna(), 'ring_id'] = -1
        df.loc[df['ring_type'].isna(), 'ring_type'] = 'legitimate'
        
        return df
    
    def generate_evaluation_dataset(self, num_rings: int = 10, 
                                 ring_types: List[str] = None,
                                 base_transactions: int = 50000) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate a comprehensive evaluation dataset
        
        Args:
            num_rings: Number of fraud rings
            ring_types: Types of rings to include
            base_transactions: Number of legitimate transactions
            
        Returns:
            Tuple of (dataframe, metadata)
        """
        
        if ring_types is None:
            ring_types = list(self.ring_configs.keys())
        
        # Generate data
        df = self.generate_fraud_rings(num_rings, ring_types, base_transactions)
        
        # Create metadata
        metadata = {
            'total_transactions': len(df),
            'fraud_transactions': df['fraud'].sum(),
            'fraud_rate': df['fraud'].mean(),
            'num_rings': len(df[df['ring_id'] != -1]['ring_id'].unique()),
            'ring_types': df[df['ring_type'] != 'legitimate']['ring_type'].value_counts().to_dict(),
            'time_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            },
            'amount_stats': {
                'mean': df['amount'].mean(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max()
            }
        }
        
        # Ring-specific metadata
        ring_metadata = {}
        for ring_id in df[df['ring_id'] != -1]['ring_id'].unique():
            ring_data = df[df['ring_id'] == ring_id]
            ring_metadata[ring_id] = {
                'size': len(ring_data),
                'fraud_rate': ring_data['fraud'].mean(),
                'ring_type': ring_data['ring_type'].iloc[0],
                'unique_cards': ring_data['card_id'].nunique(),
                'unique_merchants': ring_data['merchant_id'].nunique(),
                'unique_devices': ring_data['device_id'].nunique(),
                'unique_ips': ring_data['ip'].nunique(),
                'amount_range': (ring_data['amount'].min(), ring_data['amount'].max())
            }
        
        metadata['rings'] = ring_metadata
        
        return df, metadata
    
    def save_dataset(self, df: pd.DataFrame, metadata: Dict, 
                    data_path: str = 'data/synthetic_fraud_rings.csv',
                    metadata_path: str = 'data/synthetic_metadata.json') -> None:
        """
        Save generated dataset and metadata
        
        Args:
            df: Generated dataframe
            metadata: Dataset metadata
            data_path: Path to save CSV data
            metadata_path: Path to save JSON metadata
        """
        
        import json
        import os
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        # Save data
        df.to_csv(data_path, index=False)
        logger.info(f"Dataset saved to {data_path}")
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Metadata saved to {metadata_path}")

def generate_sample_dataset():
    """Generate a sample dataset for testing"""
    
    generator = FraudRingGenerator(seed=42)
    
    # Generate evaluation dataset
    df, metadata = generator.generate_evaluation_dataset(
        num_rings=8,
        ring_types=['card_testing_ring', 'merchant_collusion', 'device_farming', 'ip_proxy_ring'],
        base_transactions=20000
    )
    
    # Save dataset
    generator.save_dataset(df, metadata)
    
    print("Sample dataset generated!")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud rate: {df['fraud'].mean():.2%}")
    print(f"Number of fraud rings: {len(df[df['ring_id'] != -1]['ring_id'].unique())}")
    
    return df, metadata

if __name__ == "__main__":
    generate_sample_dataset()