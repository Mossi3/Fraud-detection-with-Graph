import json
import csv
import math
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import base64
import io

class HeatmapGenerator:
    """Generate heatmap visualizations for fraud patterns"""
    
    def __init__(self):
        self.transactions = []
        self.load_data()
    
    def load_data(self):
        """Load transaction data"""
        with open('/workspace/data/transactions.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['amount'] = float(row['amount'])
                row['is_fraud'] = int(row['is_fraud'])
                row['timestamp'] = datetime.fromisoformat(row['timestamp'])
                self.transactions.append(row)
    
    def generate_time_heatmap(self) -> dict:
        """Generate fraud heatmap by hour and day of week"""
        # Initialize heatmap data
        heatmap_data = {}
        for day in range(7):  # Monday = 0, Sunday = 6
            heatmap_data[day] = {}
            for hour in range(24):
                heatmap_data[day][hour] = {'total': 0, 'fraud': 0, 'fraud_rate': 0}
        
        # Count transactions by time
        for txn in self.transactions:
            day_of_week = txn['timestamp'].weekday()
            hour = txn['timestamp'].hour
            
            heatmap_data[day_of_week][hour]['total'] += 1
            if txn['is_fraud']:
                heatmap_data[day_of_week][hour]['fraud'] += 1
        
        # Calculate fraud rates
        for day in heatmap_data:
            for hour in heatmap_data[day]:
                total = heatmap_data[day][hour]['total']
                fraud = heatmap_data[day][hour]['fraud']
                heatmap_data[day][hour]['fraud_rate'] = fraud / total if total > 0 else 0
        
        return heatmap_data
    
    def generate_amount_heatmap(self) -> dict:
        """Generate fraud heatmap by amount ranges and categories"""
        # Define amount ranges
        amount_ranges = [
            (0, 50, '0-50'),
            (50, 100, '50-100'),
            (100, 500, '100-500'),
            (500, 1000, '500-1000'),
            (1000, 5000, '1000-5000'),
            (5000, float('inf'), '5000+')
        ]
        
        categories = ['groceries', 'gas', 'restaurant', 'retail', 'online', 'electronics', 'jewelry', 'cash_advance']
        
        heatmap_data = {}
        for category in categories:
            heatmap_data[category] = {}
            for min_amt, max_amt, label in amount_ranges:
                heatmap_data[category][label] = {'total': 0, 'fraud': 0, 'fraud_rate': 0}
        
        # Count transactions
        for txn in self.transactions:
            category = txn['category']
            amount = txn['amount']
            
            if category in heatmap_data:
                for min_amt, max_amt, label in amount_ranges:
                    if min_amt <= amount < max_amt:
                        heatmap_data[category][label]['total'] += 1
                        if txn['is_fraud']:
                            heatmap_data[category][label]['fraud'] += 1
        
        # Calculate fraud rates
        for category in heatmap_data:
            for label in heatmap_data[category]:
                total = heatmap_data[category][label]['total']
                fraud = heatmap_data[category][label]['fraud']
                heatmap_data[category][label]['fraud_rate'] = fraud / total if total > 0 else 0
        
        return heatmap_data
    
    def generate_merchant_heatmap(self) -> dict:
        """Generate fraud heatmap by merchant and country"""
        countries = ['US', 'CA', 'UK', 'DE', 'FR']
        
        # Get top merchants by transaction count
        merchant_counts = Counter(txn['merchant_id'] for txn in self.transactions)
        top_merchants = [merchant for merchant, count in merchant_counts.most_common(20)]
        
        heatmap_data = {}
        for merchant in top_merchants:
            heatmap_data[merchant] = {}
            for country in countries:
                heatmap_data[merchant][country] = {'total': 0, 'fraud': 0, 'fraud_rate': 0}
        
        # Count transactions
        for txn in self.transactions:
            merchant = txn['merchant_id']
            country = txn['country']
            
            if merchant in heatmap_data and country in heatmap_data[merchant]:
                heatmap_data[merchant][country]['total'] += 1
                if txn['is_fraud']:
                    heatmap_data[merchant][country]['fraud'] += 1
        
        # Calculate fraud rates
        for merchant in heatmap_data:
            for country in heatmap_data[merchant]:
                total = heatmap_data[merchant][country]['total']
                fraud = heatmap_data[merchant][country]['fraud']
                heatmap_data[merchant][country]['fraud_rate'] = fraud / total if total > 0 else 0
        
        return heatmap_data
    
    def generate_fraud_ring_heatmap(self) -> dict:
        """Generate heatmap showing fraud ring activity"""
        with open('/workspace/data/fraud_rings.json', 'r') as f:
            fraud_rings = json.load(f)
        
        heatmap_data = {}
        
        for ring_name, ring_data in fraud_rings.items():
            heatmap_data[ring_name] = {
                'cards': len(ring_data['cards']),
                'merchants': len(ring_data['merchants']),
                'devices': len(ring_data['devices']),
                'ips': len(ring_data['ips']),
                'transactions': 0,
                'total_amount': 0,
                'fraud_rate': 0
            }
            
            # Count transactions for this ring
            ring_transactions = [txn for txn in self.transactions if txn['fraud_ring'] == ring_name]
            heatmap_data[ring_name]['transactions'] = len(ring_transactions)
            heatmap_data[ring_name]['total_amount'] = sum(txn['amount'] for txn in ring_transactions)
            heatmap_data[ring_name]['fraud_rate'] = sum(txn['is_fraud'] for txn in ring_transactions) / len(ring_transactions) if ring_transactions else 0
        
        return heatmap_data
    
    def generate_network_heatmap(self) -> dict:
        """Generate heatmap showing network connections between entities"""
        # Count connections between cards and merchants
        card_merchant_connections = defaultdict(set)
        card_device_connections = defaultdict(set)
        merchant_device_connections = defaultdict(set)
        
        for txn in self.transactions:
            card_merchant_connections[txn['card_id']].add(txn['merchant_id'])
            card_device_connections[txn['card_id']].add(txn['device_id'])
            merchant_device_connections[txn['merchant_id']].add(txn['device_id'])
        
        # Get top entities by connection count
        top_cards = sorted(card_merchant_connections.keys(), 
                          key=lambda x: len(card_merchant_connections[x]), reverse=True)[:15]
        top_merchants = sorted(merchant_device_connections.keys(),
                              key=lambda x: len(merchant_device_connections[x]), reverse=True)[:15]
        
        heatmap_data = {}
        for card in top_cards:
            heatmap_data[card] = {}
            for merchant in top_merchants:
                if merchant in card_merchant_connections[card]:
                    heatmap_data[card][merchant] = 1
                else:
                    heatmap_data[card][merchant] = 0
        
        return heatmap_data
    
    def create_simple_heatmap_html(self, data: dict, title: str, x_label: str, y_label: str) -> str:
        """Create a simple HTML heatmap using CSS"""
        html = f"""
        <div style="margin: 20px; padding: 20px; border: 1px solid #ccc; border-radius: 8px;">
            <h3 style="text-align: center; color: #333;">{title}</h3>
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-weight: bold;">{y_label}</span>
                <span style="font-weight: bold;">{x_label}</span>
            </div>
            <div style="display: grid; gap: 2px;">
        """
        
        # Create grid
        max_value = 0
        for row_data in data.values():
            if isinstance(row_data, dict):
                for value in row_data.values():
                    if isinstance(value, (int, float)):
                        max_value = max(max_value, value)
                    elif isinstance(value, dict) and 'fraud_rate' in value:
                        max_value = max(max_value, value['fraud_rate'])
            elif isinstance(row_data, (int, float)):
                max_value = max(max_value, row_data)
        
        for row_key, row_data in data.items():
            html += f'<div style="display: flex; gap: 2px;">'
            html += f'<div style="width: 100px; text-align: right; padding: 5px; font-size: 12px;">{row_key}</div>'
            
            if isinstance(row_data, dict):
                for col_key, value in row_data.items():
                    if isinstance(value, dict) and 'fraud_rate' in value:
                        display_value = value['fraud_rate']
                    else:
                        display_value = value
                    
                    intensity = display_value / max_value if max_value > 0 else 0
                    color_intensity = int(255 * intensity)
                    color = f"rgb({255-color_intensity}, {255-color_intensity}, 255)"
                    html += f'<div style="width: 30px; height: 30px; background-color: {color}; border: 1px solid #ddd; display: flex; align-items: center; justify-content: center; font-size: 10px;" title="{row_key} x {col_key}: {display_value:.3f}">{display_value:.2f}</div>'
            else:
                intensity = row_data / max_value if max_value > 0 else 0
                color_intensity = int(255 * intensity)
                color = f"rgb({255-color_intensity}, {255-color_intensity}, 255)"
                html += f'<div style="width: 30px; height: 30px; background-color: {color}; border: 1px solid #ddd; display: flex; align-items: center; justify-content: center; font-size: 10px;" title="{row_key}: {row_data:.3f}">{row_data:.2f}</div>'
            
            html += '</div>'
        
        html += """
            </div>
            <div style="margin-top: 10px; text-align: center;">
                <span style="font-size: 12px; color: #666;">Darker colors indicate higher values</span>
            </div>
        </div>
        """
        
        return html
    
    def generate_all_heatmaps(self) -> dict:
        """Generate all heatmap visualizations"""
        heatmaps = {
            'time_heatmap': self.generate_time_heatmap(),
            'amount_heatmap': self.generate_amount_heatmap(),
            'merchant_heatmap': self.generate_merchant_heatmap(),
            'fraud_ring_heatmap': self.generate_fraud_ring_heatmap(),
            'network_heatmap': self.generate_network_heatmap()
        }
        
        # Create HTML visualizations
        html_heatmaps = {
            'time_html': self.create_simple_heatmap_html(
                heatmaps['time_heatmap'], 
                'Fraud Rate by Time', 
                'Hour of Day', 
                'Day of Week'
            ),
            'amount_html': self.create_simple_heatmap_html(
                heatmaps['amount_heatmap'], 
                'Fraud Rate by Amount and Category', 
                'Amount Range', 
                'Category'
            ),
            'merchant_html': self.create_simple_heatmap_html(
                heatmaps['merchant_heatmap'], 
                'Fraud Rate by Merchant and Country', 
                'Country', 
                'Merchant'
            ),
            'fraud_ring_html': self.create_simple_heatmap_html(
                heatmaps['fraud_ring_heatmap'], 
                'Fraud Ring Statistics', 
                'Metric', 
                'Ring'
            )
        }
        
        return {
            'data': heatmaps,
            'html': html_heatmaps
        }

# Initialize heatmap generator
heatmap_generator = HeatmapGenerator()

if __name__ == "__main__":
    # Generate all heatmaps
    heatmaps = heatmap_generator.generate_all_heatmaps()
    
    # Save heatmap data
    with open('/workspace/data/heatmap_data.json', 'w') as f:
        json.dump(heatmaps['data'], f, indent=2, default=str)
    
    # Save HTML heatmaps
    with open('/workspace/data/heatmap_html.json', 'w') as f:
        json.dump(heatmaps['html'], f, indent=2)
    
    print("Heatmaps generated successfully!")
    print("Available heatmaps:")
    for key in heatmaps['data'].keys():
        print(f"  - {key}")