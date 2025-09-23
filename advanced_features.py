import asyncio
import websockets
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import threading
from collections import deque
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class FraudAlert:
    """Fraud alert data structure"""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    transaction_id: str
    fraud_probability: float
    risk_factors: List[str]
    description: str
    card_id: str
    amount: float
    merchant: str
    country: str

class RealTimeMonitor:
    """Real-time fraud monitoring system"""
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.alerts = deque(maxlen=1000)  # Keep last 1000 alerts
        self.metrics = {
            'total_transactions': 0,
            'fraud_detected': 0,
            'alerts_sent': 0,
            'false_positives': 0,
            'detection_rate': 0.0,
            'avg_processing_time': 0.0
        }
        self.thresholds = {
            'fraud_probability': 0.7,
            'high_amount': 1000.0,
            'unusual_hour': [0, 1, 2, 3, 4, 5],
            'max_transactions_per_minute': 10
        }
        self.connected_clients = set()
        self.running = False
        
    async def start_monitoring(self):
        """Start real-time monitoring"""
        self.running = True
        logger.info("Starting real-time fraud monitoring...")
        
        # Start WebSocket server
        await self.start_websocket_server()
        
    async def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        logger.info("Stopping real-time fraud monitoring...")
        
    async def start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        async def handle_client(websocket, path):
            self.connected_clients.add(websocket)
            logger.info(f"Client connected: {websocket.remote_address}")
            
            try:
                # Send initial metrics
                await websocket.send(json.dumps({
                    'type': 'metrics',
                    'data': self.metrics
                }))
                
                # Keep connection alive
                async for message in websocket:
                    data = json.loads(message)
                    await self.handle_client_message(websocket, data)
                    
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.connected_clients.remove(websocket)
                logger.info(f"Client disconnected: {websocket.remote_address}")
        
        # Start WebSocket server
        start_server = websockets.serve(handle_client, "localhost", 8765)
        await start_server
        logger.info("WebSocket server started on ws://localhost:8765")
    
    async def handle_client_message(self, websocket, data):
        """Handle messages from WebSocket clients"""
        message_type = data.get('type')
        
        if message_type == 'subscribe':
            # Client wants to subscribe to updates
            await websocket.send(json.dumps({
                'type': 'subscribed',
                'message': 'Successfully subscribed to fraud alerts'
            }))
        elif message_type == 'get_alerts':
            # Client wants recent alerts
            recent_alerts = list(self.alerts)[-10:]  # Last 10 alerts
            await websocket.send(json.dumps({
                'type': 'alerts',
                'data': [self._serialize_alert(alert) for alert in recent_alerts]
            }))
    
    async def broadcast_alert(self, alert: FraudAlert):
        """Broadcast alert to all connected clients"""
        if not self.connected_clients:
            return
            
        message = {
            'type': 'alert',
            'data': self._serialize_alert(alert)
        }
        
        # Send to all connected clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected
    
    async def broadcast_metrics(self):
        """Broadcast updated metrics to all connected clients"""
        if not self.connected_clients:
            return
            
        message = {
            'type': 'metrics',
            'data': self.metrics
        }
        
        # Send to all connected clients
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected
    
    def _serialize_alert(self, alert: FraudAlert) -> Dict[str, Any]:
        """Serialize alert for JSON transmission"""
        return {
            'alert_id': alert.alert_id,
            'timestamp': alert.timestamp.isoformat(),
            'level': alert.level.value,
            'transaction_id': alert.transaction_id,
            'fraud_probability': alert.fraud_probability,
            'risk_factors': alert.risk_factors,
            'description': alert.description,
            'card_id': alert.card_id,
            'amount': alert.amount,
            'merchant': alert.merchant,
            'country': alert.country
        }
    
    async def process_transaction(self, transaction_data: Dict[str, Any]):
        """Process a transaction and generate alerts if needed"""
        start_time = time.time()
        
        try:
            # Call fraud detection API
            response = requests.post(
                f"{self.api_base_url}/detect",
                json=transaction_data,
                timeout=5
            )
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code}")
                return
            
            result = response.json()
            processing_time = time.time() - start_time
            
            # Update metrics
            self.metrics['total_transactions'] += 1
            self.metrics['avg_processing_time'] = (
                (self.metrics['avg_processing_time'] * (self.metrics['total_transactions'] - 1) + processing_time) /
                self.metrics['total_transactions']
            )
            
            # Check if fraud detected
            if result['is_fraud']:
                self.metrics['fraud_detected'] += 1
                self.metrics['detection_rate'] = (
                    self.metrics['fraud_detected'] / self.metrics['total_transactions']
                )
                
                # Generate alert
                alert = self._create_fraud_alert(transaction_data, result)
                await self._handle_fraud_alert(alert)
            
            # Broadcast updated metrics
            await self.broadcast_metrics()
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
    
    def _create_fraud_alert(self, transaction_data: Dict[str, Any], result: Dict[str, Any]) -> FraudAlert:
        """Create fraud alert from transaction and result"""
        alert_id = f"ALERT_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Determine alert level
        fraud_prob = result['fraud_probability']
        if fraud_prob > 0.9:
            level = AlertLevel.CRITICAL
        elif fraud_prob > 0.8:
            level = AlertLevel.HIGH
        elif fraud_prob > 0.6:
            level = AlertLevel.MEDIUM
        else:
            level = AlertLevel.LOW
        
        # Create description
        description = f"Fraud detected for card {transaction_data['Card_ID']} at {transaction_data['Merchant']}"
        if transaction_data['Amount'] > self.thresholds['high_amount']:
            description += f" with high amount ${transaction_data['Amount']:.2f}"
        
        return FraudAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            level=level,
            transaction_id=transaction_data['Card_ID'],
            fraud_probability=fraud_prob,
            risk_factors=result['risk_factors'],
            description=description,
            card_id=transaction_data['Card_ID'],
            amount=transaction_data['Amount'],
            merchant=transaction_data['Merchant'],
            country=transaction_data['Country']
        )
    
    async def _handle_fraud_alert(self, alert: FraudAlert):
        """Handle fraud alert - send notifications, log, etc."""
        # Add to alerts queue
        self.alerts.append(alert)
        self.metrics['alerts_sent'] += 1
        
        # Log alert
        logger.warning(f"FRAUD ALERT [{alert.level.value.upper()}]: {alert.description}")
        
        # Send notifications based on level
        if alert.level in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
            await self._send_notifications(alert)
        
        # Broadcast to WebSocket clients
        await self.broadcast_alert(alert)
    
    async def _send_notifications(self, alert: FraudAlert):
        """Send notifications for high-priority alerts"""
        # Email notification (if configured)
        if os.getenv('SMTP_SERVER'):
            await self._send_email_alert(alert)
        
        # Slack notification (if configured)
        if os.getenv('SLACK_WEBHOOK_URL'):
            await self._send_slack_alert(alert)
        
        # SMS notification (if configured)
        if os.getenv('SMS_API_KEY'):
            await self._send_sms_alert(alert)
    
    async def _send_email_alert(self, alert: FraudAlert):
        """Send email alert"""
        try:
            # Email configuration
            smtp_server = os.getenv('SMTP_SERVER')
            smtp_port = int(os.getenv('SMTP_PORT', 587))
            username = os.getenv('SMTP_USERNAME')
            password = os.getenv('SMTP_PASSWORD')
            to_email = os.getenv('ALERT_EMAIL')
            
            if not all([smtp_server, username, password, to_email]):
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = to_email
            msg['Subject'] = f"🚨 Fraud Alert [{alert.level.value.upper()}] - {alert.card_id}"
            
            body = f"""
            Fraud Alert Detected!
            
            Alert ID: {alert.alert_id}
            Timestamp: {alert.timestamp}
            Level: {alert.level.value.upper()}
            
            Transaction Details:
            - Card ID: {alert.card_id}
            - Amount: ${alert.amount:.2f}
            - Merchant: {alert.merchant}
            - Country: {alert.country}
            
            Risk Factors:
            {chr(10).join(f"- {factor}" for factor in alert.risk_factors)}
            
            Fraud Probability: {alert.fraud_probability:.1%}
            
            Please investigate immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(username, password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: FraudAlert):
        """Send Slack alert"""
        try:
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                return
            
            # Create Slack message
            color = {
                AlertLevel.LOW: "#36a64f",
                AlertLevel.MEDIUM: "#ffaa00",
                AlertLevel.HIGH: "#ff6600",
                AlertLevel.CRITICAL: "#ff0000"
            }[alert.level]
            
            message = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"🚨 Fraud Alert [{alert.level.value.upper()}]",
                        "fields": [
                            {"title": "Card ID", "value": alert.card_id, "short": True},
                            {"title": "Amount", "value": f"${alert.amount:.2f}", "short": True},
                            {"title": "Merchant", "value": alert.merchant, "short": True},
                            {"title": "Country", "value": alert.country, "short": True},
                            {"title": "Fraud Probability", "value": f"{alert.fraud_probability:.1%}", "short": True},
                            {"title": "Risk Factors", "value": ", ".join(alert.risk_factors), "short": False}
                        ],
                        "footer": "Fraud Detection System",
                        "ts": int(alert.timestamp.timestamp())
                    }
                ]
            }
            
            # Send to Slack
            response = requests.post(webhook_url, json=message, timeout=10)
            if response.status_code == 200:
                logger.info(f"Slack alert sent for {alert.alert_id}")
            else:
                logger.error(f"Slack API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    async def _send_sms_alert(self, alert: FraudAlert):
        """Send SMS alert"""
        try:
            api_key = os.getenv('SMS_API_KEY')
            phone_number = os.getenv('ALERT_PHONE')
            
            if not all([api_key, phone_number]):
                return
            
            # Create SMS message
            message = f"🚨 FRAUD ALERT [{alert.level.value.upper()}]: Card {alert.card_id} - ${alert.amount:.2f} at {alert.merchant}. Prob: {alert.fraud_probability:.1%}"
            
            # Send SMS (using Twilio or similar service)
            # This is a placeholder - implement based on your SMS provider
            logger.info(f"SMS alert would be sent: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current monitoring metrics"""
        return self.metrics.copy()
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        recent_alerts = list(self.alerts)[-limit:]
        return [self._serialize_alert(alert) for alert in recent_alerts]
    
    def update_thresholds(self, new_thresholds: Dict[str, Any]):
        """Update monitoring thresholds"""
        self.thresholds.update(new_thresholds)
        logger.info(f"Updated thresholds: {new_thresholds}")

class AlertManager:
    """Advanced alert management system"""
    
    def __init__(self):
        self.alert_rules = []
        self.alert_history = []
        self.suppression_rules = []
        
    def add_alert_rule(self, rule: Dict[str, Any]):
        """Add custom alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule['name']}")
    
    def add_suppression_rule(self, rule: Dict[str, Any]):
        """Add alert suppression rule"""
        self.suppression_rules.append(rule)
        logger.info(f"Added suppression rule: {rule['name']}")
    
    def should_suppress_alert(self, alert: FraudAlert) -> bool:
        """Check if alert should be suppressed"""
        for rule in self.suppression_rules:
            if self._evaluate_rule(rule, alert):
                return True
        return False
    
    def _evaluate_rule(self, rule: Dict[str, Any], alert: FraudAlert) -> bool:
        """Evaluate a rule against an alert"""
        # Implement rule evaluation logic
        # This is a simplified version
        return False

class PerformanceMonitor:
    """System performance monitoring"""
    
    def __init__(self):
        self.performance_metrics = {
            'api_response_time': deque(maxlen=100),
            'model_inference_time': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'active_connections': 0
        }
    
    def record_api_response_time(self, response_time: float):
        """Record API response time"""
        self.performance_metrics['api_response_time'].append(response_time)
    
    def record_model_inference_time(self, inference_time: float):
        """Record model inference time"""
        self.performance_metrics['model_inference_time'].append(inference_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        summary = {}
        for metric, values in self.performance_metrics.items():
            if values:
                summary[metric] = {
                    'current': values[-1],
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values)
                }
        return summary

# Global instances
monitor = RealTimeMonitor()
alert_manager = AlertManager()
performance_monitor = PerformanceMonitor()

async def start_real_time_monitoring():
    """Start the real-time monitoring system"""
    await monitor.start_monitoring()
    
    # Add some default alert rules
    monitor.add_alert_rule({
        'name': 'high_amount_fraud',
        'condition': 'amount > 1000 and fraud_probability > 0.8',
        'action': 'send_critical_alert'
    })
    
    monitor.add_alert_rule({
        'name': 'unusual_time_fraud',
        'condition': 'hour in [0,1,2,3,4,5] and fraud_probability > 0.6',
        'action': 'send_high_alert'
    })
    
    # Add suppression rules
    monitor.add_suppression_rule({
        'name': 'suppress_low_amount',
        'condition': 'amount < 10',
        'action': 'suppress'
    })

if __name__ == "__main__":
    # Example usage
    async def main():
        await start_real_time_monitoring()
        
        # Simulate some transactions
        for i in range(5):
            transaction = {
                "Time": 100.0 + i,
                "V1": np.random.normal(0, 1),
                "V2": np.random.normal(0, 1),
                "V3": np.random.normal(0, 1),
                "V4": np.random.normal(0, 1),
                "V5": np.random.normal(0, 1),
                "V6": np.random.normal(0, 1),
                "V7": np.random.normal(0, 1),
                "V8": np.random.normal(0, 1),
                "V9": np.random.normal(0, 1),
                "V10": np.random.normal(0, 1),
                "V11": np.random.normal(0, 1),
                "V12": np.random.normal(0, 1),
                "V13": np.random.normal(0, 1),
                "V14": np.random.normal(0, 1),
                "V15": np.random.normal(0, 1),
                "V16": np.random.normal(0, 1),
                "V17": np.random.normal(0, 1),
                "V18": np.random.normal(0, 1),
                "V19": np.random.normal(0, 1),
                "V20": np.random.normal(0, 1),
                "V21": np.random.normal(0, 1),
                "V22": np.random.normal(0, 1),
                "V23": np.random.normal(0, 1),
                "V24": np.random.normal(0, 1),
                "V25": np.random.normal(0, 1),
                "V26": np.random.normal(0, 1),
                "V27": np.random.normal(0, 1),
                "V28": np.random.normal(0, 1),
                "Amount": np.random.exponential(50),
                "Merchant": np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail']),
                "Country": np.random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
                "Device": np.random.choice(['mobile', 'desktop', 'tablet']),
                "IP_Country": np.random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
                "Hour": np.random.randint(0, 24),
                "Day": np.random.randint(1, 32),
                "Card_ID": f"CARD_{np.random.randint(10000, 99999):05d}",
                "Merchant_ID": f"M_{np.random.randint(100, 999):03d}",
                "Device_ID": f"D_{np.random.randint(1000, 9999):04d}",
                "IP_Address": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            }
            
            await monitor.process_transaction(transaction)
            await asyncio.sleep(1)  # Wait 1 second between transactions
        
        # Keep running
        while True:
            await asyncio.sleep(60)  # Update every minute
    
    asyncio.run(main())