"""
Test scenarios for fraud detection system.
Generates realistic test data and fraud patterns.
"""

import random
import string
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
import requests
from faker import Faker

fake = Faker()


class FraudScenarioGenerator:
    """Generates various fraud scenarios for testing"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.api_base_url = api_base_url
        self.api_url = f"{api_base_url}/api/v1"
        
        # Track generated entities
        self.cards = []
        self.merchants = []
        self.devices = []
        self.ips = []
        
    def generate_card_id(self) -> str:
        """Generate realistic card ID"""
        card_id = ''.join(random.choices(string.digits, k=16))
        self.cards.append(card_id)
        return card_id
    
    def generate_merchant_id(self, category: str = None) -> str:
        """Generate merchant ID"""
        categories = ['retail', 'restaurant', 'gas', 'grocery', 'online', 'atm']
        if not category:
            category = random.choice(categories)
        
        merchant_id = f"{fake.company().replace(' ', '_').lower()}_{category}"
        self.merchants.append(merchant_id)
        return merchant_id
    
    def generate_device_id(self) -> str:
        """Generate device ID"""
        device_id = f"device_{fake.uuid4()[:8]}"
        self.devices.append(device_id)
        return device_id
    
    def generate_ip_address(self) -> str:
        """Generate IP address"""
        ip = fake.ipv4()
        self.ips.append(ip)
        return ip
    
    def create_transaction(self,
                         card_id: str = None,
                         merchant_id: str = None,
                         amount: float = None,
                         timestamp: datetime = None,
                         device_id: str = None,
                         ip_address: str = None) -> Dict[str, Any]:
        """Create a transaction"""
        if not timestamp:
            timestamp = datetime.now()
            
        return {
            "transaction_id": f"txn_{fake.uuid4()[:12]}",
            "card_id": card_id or self.generate_card_id(),
            "merchant_id": merchant_id or self.generate_merchant_id(),
            "amount": amount or round(random.uniform(10, 500), 2),
            "timestamp": timestamp.isoformat(),
            "device_id": device_id or self.generate_device_id(),
            "ip_address": ip_address or self.generate_ip_address(),
            "location": [fake.latitude(), fake.longitude()]
        }
    
    def scenario_card_cloning(self) -> List[Dict[str, Any]]:
        """
        Scenario: Card cloning fraud
        Same card used from multiple locations/devices simultaneously
        """
        print("\n=== Card Cloning Scenario ===")
        transactions = []
        
        # Original card and owner's device
        card_id = self.generate_card_id()
        owner_device = self.generate_device_id()
        owner_ip = self.generate_ip_address()
        
        # Legitimate transactions
        base_time = datetime.now() - timedelta(hours=2)
        for i in range(3):
            txn = self.create_transaction(
                card_id=card_id,
                device_id=owner_device,
                ip_address=owner_ip,
                timestamp=base_time + timedelta(minutes=i*30),
                amount=random.uniform(20, 100)
            )
            transactions.append(txn)
        
        # Cloned card usage from different location
        clone_device = self.generate_device_id()
        clone_ip = self.generate_ip_address()
        
        # Simultaneous usage
        fraud_time = base_time + timedelta(hours=1)
        
        # Transaction from original location
        txn1 = self.create_transaction(
            card_id=card_id,
            device_id=owner_device,
            ip_address=owner_ip,
            timestamp=fraud_time,
            merchant_id="grocery_store_local",
            amount=45.67
        )
        transactions.append(txn1)
        
        # Transaction from cloned card (different city) within minutes
        txn2 = self.create_transaction(
            card_id=card_id,
            device_id=clone_device,
            ip_address=clone_ip,
            timestamp=fraud_time + timedelta(minutes=5),
            merchant_id="atm_withdrawal_distant",
            amount=500.00
        )
        transactions.append(txn2)
        
        print(f"Generated {len(transactions)} transactions for card cloning scenario")
        return transactions
    
    def scenario_account_takeover(self) -> List[Dict[str, Any]]:
        """
        Scenario: Account takeover
        Sudden change in spending patterns and device/location
        """
        print("\n=== Account Takeover Scenario ===")
        transactions = []
        
        card_id = self.generate_card_id()
        legitimate_device = self.generate_device_id()
        legitimate_ip = self.generate_ip_address()
        
        # Normal spending pattern (small amounts, regular merchants)
        base_time = datetime.now() - timedelta(days=7)
        for day in range(7):
            for _ in range(random.randint(1, 3)):
                txn = self.create_transaction(
                    card_id=card_id,
                    device_id=legitimate_device,
                    ip_address=legitimate_ip,
                    timestamp=base_time + timedelta(days=day, hours=random.randint(8, 20)),
                    merchant_id=random.choice(["grocery_store", "gas_station", "restaurant"]),
                    amount=random.uniform(15, 80)
                )
                transactions.append(txn)
        
        # Account takeover - new device, different pattern
        fraud_device = self.generate_device_id()
        fraud_ip = self.generate_ip_address()
        takeover_time = datetime.now() - timedelta(hours=2)
        
        # Rapid high-value transactions
        fraud_merchants = ["electronics_store", "jewelry_shop", "luxury_goods", "atm_withdrawal"]
        for i, merchant in enumerate(fraud_merchants):
            txn = self.create_transaction(
                card_id=card_id,
                device_id=fraud_device,
                ip_address=fraud_ip,
                timestamp=takeover_time + timedelta(minutes=i*15),
                merchant_id=merchant,
                amount=random.uniform(500, 2000)
            )
            transactions.append(txn)
        
        print(f"Generated {len(transactions)} transactions for account takeover scenario")
        return transactions
    
    def scenario_merchant_collusion(self) -> List[Dict[str, Any]]:
        """
        Scenario: Merchant collusion ring
        Multiple cards making transactions at suspicious merchants
        """
        print("\n=== Merchant Collusion Scenario ===")
        transactions = []
        
        # Create colluding merchants
        fraud_merchants = [
            self.generate_merchant_id("online"),
            self.generate_merchant_id("online"),
            self.generate_merchant_id("services")
        ]
        
        # Create multiple cards involved in the ring
        fraud_cards = [self.generate_card_id() for _ in range(5)]
        
        # Shared devices/IPs (indicating coordination)
        shared_devices = [self.generate_device_id() for _ in range(2)]
        shared_ips = [self.generate_ip_address() for _ in range(2)]
        
        base_time = datetime.now() - timedelta(days=3)
        
        # Generate circular transaction patterns
        for day in range(3):
            daily_time = base_time + timedelta(days=day)
            
            # Each card transacts with multiple fraud merchants
            for card_idx, card_id in enumerate(fraud_cards):
                device = shared_devices[card_idx % 2]
                ip = shared_ips[card_idx % 2]
                
                for merchant_idx, merchant_id in enumerate(fraud_merchants):
                    if (card_idx + merchant_idx) % 2 == 0:  # Create pattern
                        txn = self.create_transaction(
                            card_id=card_id,
                            merchant_id=merchant_id,
                            device_id=device,
                            ip_address=ip,
                            timestamp=daily_time + timedelta(hours=merchant_idx*2),
                            amount=random.uniform(200, 1000)
                        )
                        transactions.append(txn)
        
        print(f"Generated {len(transactions)} transactions for merchant collusion scenario")
        return transactions
    
    def scenario_money_mule_network(self) -> List[Dict[str, Any]]:
        """
        Scenario: Money mule network
        Chain of transactions moving money between accounts
        """
        print("\n=== Money Mule Network Scenario ===")
        transactions = []
        
        # Create mule accounts
        mule_cards = [self.generate_card_id() for _ in range(6)]
        
        # Initial fraudulent deposit
        source_amount = 10000
        current_time = datetime.now() - timedelta(hours=24)
        
        # Layer 1: Initial distribution
        layer1_amounts = [3000, 3500, 3500]
        for i, amount in enumerate(layer1_amounts):
            txn = self.create_transaction(
                card_id=mule_cards[0],
                merchant_id="atm_deposit",
                timestamp=current_time,
                amount=amount
            )
            transactions.append(txn)
            
            # Transfer to next layer
            txn = self.create_transaction(
                card_id=mule_cards[i+1],
                merchant_id="p2p_transfer",
                timestamp=current_time + timedelta(minutes=30),
                amount=amount * 0.95  # Small fee
            )
            transactions.append(txn)
        
        # Layer 2: Further distribution
        current_time += timedelta(hours=2)
        for i in range(1, 4):
            split_amount = layer1_amounts[i-1] * 0.95 / 2
            
            for j in range(2):
                if i+j+2 < len(mule_cards):
                    txn = self.create_transaction(
                        card_id=mule_cards[i],
                        merchant_id="money_transfer_service",
                        timestamp=current_time + timedelta(minutes=j*15),
                        amount=split_amount
                    )
                    transactions.append(txn)
        
        print(f"Generated {len(transactions)} transactions for money mule scenario")
        return transactions
    
    def scenario_velocity_attack(self) -> List[Dict[str, Any]]:
        """
        Scenario: High velocity fraud
        Rapid sequence of transactions to maximize fraud before detection
        """
        print("\n=== Velocity Attack Scenario ===")
        transactions = []
        
        card_id = self.generate_card_id()
        device_id = self.generate_device_id()
        
        attack_start = datetime.now() - timedelta(minutes=30)
        
        # Rapid transactions at different merchants
        merchants = [
            ("atm_withdrawal", 500),
            ("electronics_store", 1200),
            ("gift_card_shop", 1000),
            ("online_marketplace", 800),
            ("gas_station", 100),
            ("convenience_store", 200),
            ("online_gaming", 500),
            ("cryptocurrency_exchange", 2000)
        ]
        
        for i, (merchant, amount) in enumerate(merchants):
            txn = self.create_transaction(
                card_id=card_id,
                merchant_id=merchant,
                device_id=device_id,
                timestamp=attack_start + timedelta(minutes=i*3),
                amount=amount
            )
            transactions.append(txn)
        
        print(f"Generated {len(transactions)} transactions for velocity attack scenario")
        return transactions
    
    def scenario_synthetic_identity(self) -> List[Dict[str, Any]]:
        """
        Scenario: Synthetic identity fraud
        Fake identity with gradually building credit history
        """
        print("\n=== Synthetic Identity Scenario ===")
        transactions = []
        
        # Create synthetic identity
        synthetic_card = self.generate_card_id()
        synthetic_device = self.generate_device_id()
        synthetic_ip = self.generate_ip_address()
        
        # Phase 1: Build history with small transactions (3 months)
        start_date = datetime.now() - timedelta(days=90)
        
        for week in range(12):
            week_start = start_date + timedelta(weeks=week)
            
            # 2-3 small transactions per week
            for _ in range(random.randint(2, 3)):
                txn = self.create_transaction(
                    card_id=synthetic_card,
                    device_id=synthetic_device,
                    ip_address=synthetic_ip,
                    timestamp=week_start + timedelta(days=random.randint(0, 6)),
                    merchant_id=random.choice(["grocery", "gas", "pharmacy"]),
                    amount=random.uniform(20, 100)
                )
                transactions.append(txn)
        
        # Phase 2: Bust out - max out credit
        bust_out_time = datetime.now() - timedelta(days=2)
        
        high_value_merchants = [
            ("electronics_superstore", 3000),
            ("jewelry_boutique", 5000),
            ("luxury_goods", 4000),
            ("cash_advance", 2000)
        ]
        
        for merchant, amount in high_value_merchants:
            txn = self.create_transaction(
                card_id=synthetic_card,
                device_id=synthetic_device,
                ip_address=synthetic_ip,
                timestamp=bust_out_time + timedelta(hours=random.randint(0, 24)),
                merchant_id=merchant,
                amount=amount
            )
            transactions.append(txn)
        
        print(f"Generated {len(transactions)} transactions for synthetic identity scenario")
        return transactions
    
    def run_scenario(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Submit transactions to API and collect results"""
        results = {
            "total_transactions": len(transactions),
            "fraud_scores": [],
            "high_risk_count": 0,
            "detected_patterns": set(),
            "processing_times": []
        }
        
        for txn in transactions:
            try:
                response = requests.post(
                    f"{self.api_url}/score",
                    json=txn,
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results["fraud_scores"].append(data["fraud_score"])
                    results["processing_times"].append(data["processing_time_ms"])
                    
                    if data["risk_level"] in ["HIGH", "CRITICAL"]:
                        results["high_risk_count"] += 1
                    
                    results["detected_patterns"].update(data["fraud_patterns"])
                    
            except Exception as e:
                print(f"Error processing transaction: {e}")
        
        return results
    
    def run_all_scenarios(self):
        """Run all fraud scenarios"""
        scenarios = [
            ("Card Cloning", self.scenario_card_cloning),
            ("Account Takeover", self.scenario_account_takeover),
            ("Merchant Collusion", self.scenario_merchant_collusion),
            ("Money Mule Network", self.scenario_money_mule_network),
            ("Velocity Attack", self.scenario_velocity_attack),
            ("Synthetic Identity", self.scenario_synthetic_identity)
        ]
        
        all_results = {}
        
        for name, scenario_func in scenarios:
            print(f"\n{'='*50}")
            print(f"Running {name} Scenario")
            print('='*50)
            
            transactions = scenario_func()
            results = self.run_scenario(transactions)
            
            # Calculate statistics
            if results["fraud_scores"]:
                avg_score = sum(results["fraud_scores"]) / len(results["fraud_scores"])
                detection_rate = results["high_risk_count"] / results["total_transactions"]
                avg_time = sum(results["processing_times"]) / len(results["processing_times"])
                
                print(f"\nResults:")
                print(f"  Average Fraud Score: {avg_score:.3f}")
                print(f"  Detection Rate: {detection_rate:.1%}")
                print(f"  Average Processing Time: {avg_time:.1f}ms")
                print(f"  Detected Patterns: {', '.join(results['detected_patterns'])}")
                
                all_results[name] = {
                    "avg_fraud_score": avg_score,
                    "detection_rate": detection_rate,
                    "avg_processing_time": avg_time,
                    "patterns": list(results["detected_patterns"])
                }
        
        # Save results
        with open("scenario_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n\nResults saved to scenario_results.json")
        return all_results


if __name__ == "__main__":
    # Run all scenarios
    generator = FraudScenarioGenerator()
    generator.run_all_scenarios()