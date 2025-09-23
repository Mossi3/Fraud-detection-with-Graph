import json
import random
import csv
from datetime import datetime, timedelta

# Set random seed for reproducibility
random.seed(42)

# Define fraud rings and patterns
fraud_rings = {
    'ring_1': {
        'cards': [f'card_{i:06d}' for i in range(1000, 1010)],
        'merchants': [f'merchant_{i:04d}' for i in range(200, 210)],
        'devices': [f'device_{i:06d}' for i in range(5000, 5010)],
        'ips': [f'192.168.{i}.{j}' for i in range(1, 3) for j in range(1, 6)]
    },
    'ring_2': {
        'cards': [f'card_{i:06d}' for i in range(2000, 2015)],
        'merchants': [f'merchant_{i:04d}' for i in range(300, 315)],
        'devices': [f'device_{i:06d}' for i in range(6000, 6015)],
        'ips': [f'10.0.{i}.{j}' for i in range(1, 4) for j in range(1, 6)]
    },
    'ring_3': {
        'cards': [f'card_{i:06d}' for i in range(3000, 3020)],
        'merchants': [f'merchant_{i:04d}' for i in range(400, 420)],
        'devices': [f'device_{i:06d}' for i in range(7000, 7020)],
        'ips': [f'172.16.{i}.{j}' for i in range(1, 5) for j in range(1, 6)]
    }
}

# Normal entities
normal_cards = [f'card_{i:06d}' for i in range(10000, 20000)]
normal_merchants = [f'merchant_{i:04d}' for i in range(1000, 2000)]
normal_devices = [f'device_{i:06d}' for i in range(10000, 20000)]
normal_ips = [f'203.{i}.{j}.{k}' for i in range(1, 10) for j in range(1, 10) for k in range(1, 10)]

def generate_normal_transaction():
    """Generate a normal transaction"""
    card = random.choice(normal_cards)
    merchant = random.choice(normal_merchants)
    device = random.choice(normal_devices)
    ip = random.choice(normal_ips)
    
    # Generate amount using log-normal distribution approximation
    amount = round(random.lognormvariate(3, 1), 2)
    
    return {
        'transaction_id': f'txn_{random.randint(100000, 999999)}',
        'card_id': card,
        'merchant_id': merchant,
        'device_id': device,
        'ip_address': ip,
        'amount': amount,
        'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        'category': random.choice(['groceries', 'gas', 'restaurant', 'retail', 'online']),
        'country': random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
        'is_fraud': 0,
        'fraud_ring': None
    }

def generate_fraud_transaction():
    """Generate a fraud transaction"""
    ring_name = random.choice(list(fraud_rings.keys()))
    ring = fraud_rings[ring_name]
    
    # Use entities from the same fraud ring
    card = random.choice(ring['cards'])
    merchant = random.choice(ring['merchants'])
    device = random.choice(ring['devices'])
    ip = random.choice(ring['ips'])
    
    # Fraud transactions tend to be larger amounts
    amount = round(random.lognormvariate(5, 1.5), 2)
    
    return {
        'transaction_id': f'txn_{random.randint(100000, 999999)}',
        'card_id': card,
        'merchant_id': merchant,
        'device_id': device,
        'ip_address': ip,
        'amount': amount,
        'timestamp': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
        'category': random.choice(['electronics', 'jewelry', 'cash_advance', 'online']),
        'country': random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
        'is_fraud': 1,
        'fraud_ring': ring_name
    }

def create_mock_dataset():
    """Create and save mock dataset"""
    transactions = []
    
    # Generate normal transactions (90%)
    n_normal = 45000
    print(f"Generating {n_normal} normal transactions...")
    for i in range(n_normal):
        transactions.append(generate_normal_transaction())
    
    # Generate fraud transactions (10%)
    n_fraud = 5000
    print(f"Generating {n_fraud} fraud transactions...")
    for i in range(n_fraud):
        transactions.append(generate_fraud_transaction())
    
    # Shuffle transactions
    random.shuffle(transactions)
    
    # Save to CSV
    with open('/workspace/data/transactions.csv', 'w', newline='') as csvfile:
        fieldnames = transactions[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)
    
    # Save fraud rings info
    with open('/workspace/data/fraud_rings.json', 'w') as f:
        json.dump(fraud_rings, f, indent=2)
    
    # Calculate statistics
    fraud_count = sum(1 for t in transactions if t['is_fraud'] == 1)
    total_count = len(transactions)
    
    print(f"Dataset created with {total_count} transactions")
    print(f"Fraud rate: {fraud_count/total_count:.2%}")
    print(f"Fraud rings: {len(fraud_rings)}")
    
    return transactions

if __name__ == "__main__":
    create_mock_dataset()