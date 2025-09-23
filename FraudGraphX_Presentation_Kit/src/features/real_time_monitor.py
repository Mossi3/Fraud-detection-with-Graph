"""
Real-time fraud detection monitoring and alerting system.
Includes streaming analytics, anomaly detection, and automated responses.
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import redis
import websockets
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading
from queue import Queue, Empty

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
TRANSACTION_COUNTER = Counter('fraud_transactions_total', 'Total transactions processed', ['fraud_prediction'])
PREDICTION_TIME = Histogram('fraud_prediction_duration_seconds', 'Time spent on predictions')
FRAUD_SCORE_GAUGE = Gauge('fraud_score_current', 'Current fraud score')
RING_ALERTS = Counter('fraud_ring_alerts_total', 'Total fraud ring alerts')
ANOMALY_ALERTS = Counter('fraud_anomaly_alerts_total', 'Total anomaly alerts')

@dataclass
class TransactionEvent:
    """Real-time transaction event."""
    transaction_id: str
    timestamp: datetime
    card_id: str
    merchant_id: str
    device_id: str
    ip: str
    amount: float
    fraud_probability: float
    fraud_prediction: bool
    risk_factors: Dict[str, float]
    processing_time_ms: float

@dataclass
class FraudAlert:
    """Fraud alert event."""
    alert_id: str
    alert_type: str  # 'high_risk_transaction', 'fraud_ring', 'velocity_spike', 'anomaly'
    severity: str    # 'low', 'medium', 'high', 'critical'
    timestamp: datetime
    description: str
    entities: Dict[str, Any]
    recommended_actions: List[str]
    confidence: float

class RealTimeFraudMonitor:
    """Real-time fraud detection monitoring system."""
    
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis_client = None
        self.transaction_queue = Queue(maxsize=10000)
        self.alert_queue = Queue(maxsize=1000)
        self.websocket_clients = set()
        
        # Monitoring windows
        self.transaction_window = deque(maxlen=10000)  # Last 10k transactions
        self.velocity_windows = {
            '1m': deque(maxlen=1000),
            '5m': deque(maxlen=5000),
            '1h': deque(maxlen=60000)
        }
        
        # Anomaly detection models
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.scaler = StandardScaler()
        
        # Alert thresholds
        self.alert_thresholds = {
            'high_fraud_probability': 0.8,
            'velocity_spike_factor': 3.0,
            'anomaly_score_threshold': -0.5,
            'ring_size_threshold': 5
        }
        
        # Entity tracking
        self.entity_stats = defaultdict(lambda: {
            'transaction_count': 0,
            'fraud_count': 0,
            'total_amount': 0.0,
            'last_seen': None,
            'risk_score': 0.0
        })
        
        # Background tasks
        self.monitoring_tasks = []
        self.is_running = False
        
        # Initialize Redis connection
        self._init_redis(redis_host, redis_port)
        
        # Start Prometheus metrics server
        start_http_server(8001)
        logger.info("Prometheus metrics server started on port 8001")
    
    def _init_redis(self, host: str, port: int):
        """Initialize Redis connection."""
        try:
            self.redis_client = redis.Redis(host=host, port=port, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Redis connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def start_monitoring(self):
        """Start the real-time monitoring system."""
        self.is_running = True
        logger.info("Starting real-time fraud monitoring system")
        
        # Start background tasks
        self.monitoring_tasks = [
            asyncio.create_task(self._process_transaction_queue()),
            asyncio.create_task(self._velocity_monitoring()),
            asyncio.create_task(self._anomaly_detection()),
            asyncio.create_task(self._ring_monitoring()),
            asyncio.create_task(self._alert_processor()),
            asyncio.create_task(self._websocket_server()),
            asyncio.create_task(self._periodic_model_update())
        ]
        
        # Wait for all tasks
        await asyncio.gather(*self.monitoring_tasks)
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        self.is_running = False
        logger.info("Stopping real-time fraud monitoring system")
        
        for task in self.monitoring_tasks:
            task.cancel()
        
        await asyncio.gather(*self.monitoring_tasks, return_exceptions=True)
    
    def add_transaction(self, transaction_event: TransactionEvent):
        """Add a new transaction for monitoring."""
        try:
            self.transaction_queue.put_nowait(transaction_event)
            
            # Update Prometheus metrics
            TRANSACTION_COUNTER.labels(
                fraud_prediction=str(transaction_event.fraud_prediction)
            ).inc()
            PREDICTION_TIME.observe(transaction_event.processing_time_ms / 1000)
            FRAUD_SCORE_GAUGE.set(transaction_event.fraud_probability)
            
        except Exception as e:
            logger.error(f"Failed to add transaction to queue: {e}")
    
    async def _process_transaction_queue(self):
        """Process incoming transactions."""
        while self.is_running:
            try:
                # Get transaction from queue (non-blocking)
                try:
                    transaction = self.transaction_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.1)
                    continue
                
                # Add to monitoring windows
                self.transaction_window.append(transaction)
                current_time = transaction.timestamp
                
                # Add to velocity windows
                for window_name, window in self.velocity_windows.items():
                    window.append((current_time, transaction))
                
                # Update entity statistics
                self._update_entity_stats(transaction)
                
                # Check for immediate alerts
                await self._check_transaction_alerts(transaction)
                
                # Store in Redis if available
                if self.redis_client:
                    await self._store_transaction_redis(transaction)
                
                self.transaction_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing transaction: {e}")
                await asyncio.sleep(1)
    
    async def _velocity_monitoring(self):
        """Monitor transaction velocity for spikes."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for window_name, window in self.velocity_windows.items():
                    # Clean old transactions
                    if window_name == '1m':
                        cutoff = current_time - timedelta(minutes=1)
                    elif window_name == '5m':
                        cutoff = current_time - timedelta(minutes=5)
                    else:  # 1h
                        cutoff = current_time - timedelta(hours=1)
                    
                    # Remove old transactions
                    while window and window[0][0] < cutoff:
                        window.popleft()
                    
                    # Check for velocity spikes
                    if len(window) > 0:
                        await self._check_velocity_spikes(window_name, window)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in velocity monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _anomaly_detection(self):
        """Detect anomalies in transaction patterns."""
        while self.is_running:
            try:
                if len(self.transaction_window) < 100:
                    await asyncio.sleep(60)
                    continue
                
                # Prepare feature matrix
                features = []
                transactions = list(self.transaction_window)[-1000:]  # Last 1000 transactions
                
                for txn in transactions:
                    feature_vector = [
                        txn.amount,
                        txn.fraud_probability,
                        len(txn.risk_factors),
                        sum(txn.risk_factors.values()),
                        txn.timestamp.hour,
                        txn.timestamp.weekday(),
                        txn.processing_time_ms
                    ]
                    features.append(feature_vector)
                
                features_array = np.array(features)
                
                # Scale features
                features_scaled = self.scaler.fit_transform(features_array)
                
                # Detect anomalies
                anomaly_scores = self.isolation_forest.fit_predict(features_scaled)
                anomaly_probabilities = self.isolation_forest.score_samples(features_scaled)
                
                # Check for significant anomalies
                for i, (score, prob) in enumerate(zip(anomaly_scores, anomaly_probabilities)):
                    if score == -1 and prob < self.alert_thresholds['anomaly_score_threshold']:
                        await self._create_anomaly_alert(transactions[i], prob)
                
                await asyncio.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in anomaly detection: {e}")
                await asyncio.sleep(300)
    
    async def _ring_monitoring(self):
        """Monitor for potential fraud ring activity."""
        while self.is_running:
            try:
                if len(self.transaction_window) < 50:
                    await asyncio.sleep(120)
                    continue
                
                # Group transactions by entities
                entity_groups = defaultdict(list)
                recent_transactions = list(self.transaction_window)[-500:]
                
                for txn in recent_transactions:
                    # Group by shared entities
                    entity_groups[txn.card_id].append(txn)
                    entity_groups[txn.merchant_id].append(txn)
                    entity_groups[txn.device_id].append(txn)
                    entity_groups[txn.ip].append(txn)
                
                # Look for suspicious patterns
                for entity, transactions in entity_groups.items():
                    if len(transactions) >= self.alert_thresholds['ring_size_threshold']:
                        fraud_rate = sum(1 for t in transactions if t.fraud_prediction) / len(transactions)
                        
                        if fraud_rate > 0.5:  # High fraud rate
                            await self._create_ring_alert(entity, transactions, fraud_rate)
                
                await asyncio.sleep(180)  # Run every 3 minutes
                
            except Exception as e:
                logger.error(f"Error in ring monitoring: {e}")
                await asyncio.sleep(180)
    
    async def _alert_processor(self):
        """Process and distribute alerts."""
        while self.is_running:
            try:
                try:
                    alert = self.alert_queue.get_nowait()
                except Empty:
                    await asyncio.sleep(0.5)
                    continue
                
                # Log alert
                logger.warning(f"FRAUD ALERT: {alert.alert_type} - {alert.description}")
                
                # Store alert
                if self.redis_client:
                    await self._store_alert_redis(alert)
                
                # Send to WebSocket clients
                await self._broadcast_alert(alert)
                
                # Update metrics
                if alert.alert_type == 'fraud_ring':
                    RING_ALERTS.inc()
                elif alert.alert_type == 'anomaly':
                    ANOMALY_ALERTS.inc()
                
                self.alert_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
                await asyncio.sleep(1)
    
    async def _websocket_server(self):
        """WebSocket server for real-time updates."""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            try:
                await websocket.wait_closed()
            finally:
                self.websocket_clients.remove(websocket)
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        
        try:
            server = await websockets.serve(handle_client, "localhost", 8765)
            logger.info("WebSocket server started on ws://localhost:8765")
            await server.wait_closed()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    async def _periodic_model_update(self):
        """Periodically update anomaly detection models."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Update every hour
                
                if len(self.transaction_window) > 1000:
                    logger.info("Updating anomaly detection models...")
                    
                    # Retrain isolation forest with recent data
                    features = []
                    transactions = list(self.transaction_window)[-5000:]  # Last 5000 transactions
                    
                    for txn in transactions:
                        feature_vector = [
                            txn.amount,
                            txn.fraud_probability,
                            len(txn.risk_factors),
                            sum(txn.risk_factors.values()),
                            txn.timestamp.hour,
                            txn.timestamp.weekday(),
                            txn.processing_time_ms
                        ]
                        features.append(feature_vector)
                    
                    features_array = np.array(features)
                    features_scaled = self.scaler.fit_transform(features_array)
                    
                    # Retrain model
                    self.isolation_forest.fit(features_scaled)
                    
                    logger.info("Anomaly detection models updated")
                
            except Exception as e:
                logger.error(f"Error updating models: {e}")
    
    def _update_entity_stats(self, transaction: TransactionEvent):
        """Update statistics for entities involved in transaction."""
        entities = [
            f"card_{transaction.card_id}",
            f"merchant_{transaction.merchant_id}",
            f"device_{transaction.device_id}",
            f"ip_{transaction.ip}"
        ]
        
        for entity in entities:
            stats = self.entity_stats[entity]
            stats['transaction_count'] += 1
            stats['total_amount'] += transaction.amount
            stats['last_seen'] = transaction.timestamp
            
            if transaction.fraud_prediction:
                stats['fraud_count'] += 1
            
            # Update risk score (exponential moving average)
            alpha = 0.1
            stats['risk_score'] = (alpha * transaction.fraud_probability + 
                                 (1 - alpha) * stats['risk_score'])
    
    async def _check_transaction_alerts(self, transaction: TransactionEvent):
        """Check if transaction should trigger immediate alerts."""
        # High fraud probability alert
        if transaction.fraud_probability >= self.alert_thresholds['high_fraud_probability']:
            alert = FraudAlert(
                alert_id=f"high_risk_{transaction.transaction_id}",
                alert_type="high_risk_transaction",
                severity="high",
                timestamp=transaction.timestamp,
                description=f"High fraud probability: {transaction.fraud_probability:.3f}",
                entities={
                    'transaction_id': transaction.transaction_id,
                    'card_id': transaction.card_id,
                    'amount': transaction.amount
                },
                recommended_actions=[
                    "Review transaction manually",
                    "Contact cardholder",
                    "Consider blocking card temporarily"
                ],
                confidence=transaction.fraud_probability
            )
            
            try:
                self.alert_queue.put_nowait(alert)
            except Exception as e:
                logger.error(f"Failed to queue alert: {e}")
    
    async def _check_velocity_spikes(self, window_name: str, window: deque):
        """Check for velocity spikes in transaction patterns."""
        if len(window) < 10:
            return
        
        # Calculate current velocity
        current_velocity = len(window)
        
        # Get historical average (simplified)
        if window_name == '1m':
            expected_velocity = 5  # Expected transactions per minute
        elif window_name == '5m':
            expected_velocity = 25  # Expected transactions per 5 minutes
        else:  # 1h
            expected_velocity = 300  # Expected transactions per hour
        
        # Check for spike
        if current_velocity > expected_velocity * self.alert_thresholds['velocity_spike_factor']:
            alert = FraudAlert(
                alert_id=f"velocity_spike_{window_name}_{int(time.time())}",
                alert_type="velocity_spike",
                severity="medium",
                timestamp=datetime.now(),
                description=f"Transaction velocity spike: {current_velocity} in {window_name} (expected: {expected_velocity})",
                entities={'window': window_name, 'current_velocity': current_velocity},
                recommended_actions=[
                    "Investigate traffic source",
                    "Check for bot activity",
                    "Review recent transactions"
                ],
                confidence=min(current_velocity / expected_velocity / self.alert_thresholds['velocity_spike_factor'], 1.0)
            )
            
            try:
                self.alert_queue.put_nowait(alert)
            except Exception as e:
                logger.error(f"Failed to queue velocity alert: {e}")
    
    async def _create_anomaly_alert(self, transaction: TransactionEvent, anomaly_score: float):
        """Create alert for detected anomaly."""
        alert = FraudAlert(
            alert_id=f"anomaly_{transaction.transaction_id}",
            alert_type="anomaly",
            severity="medium",
            timestamp=transaction.timestamp,
            description=f"Anomalous transaction pattern detected (score: {anomaly_score:.3f})",
            entities={
                'transaction_id': transaction.transaction_id,
                'anomaly_score': anomaly_score,
                'amount': transaction.amount
            },
            recommended_actions=[
                "Review transaction details",
                "Check for unusual patterns",
                "Validate with cardholder"
            ],
            confidence=abs(anomaly_score)
        )
        
        try:
            self.alert_queue.put_nowait(alert)
        except Exception as e:
            logger.error(f"Failed to queue anomaly alert: {e}")
    
    async def _create_ring_alert(self, entity: str, transactions: List[TransactionEvent], fraud_rate: float):
        """Create alert for potential fraud ring."""
        alert = FraudAlert(
            alert_id=f"ring_{entity}_{int(time.time())}",
            alert_type="fraud_ring",
            severity="critical",
            timestamp=datetime.now(),
            description=f"Potential fraud ring detected: {entity} with {len(transactions)} transactions ({fraud_rate:.1%} fraud rate)",
            entities={
                'entity': entity,
                'transaction_count': len(transactions),
                'fraud_rate': fraud_rate,
                'transaction_ids': [t.transaction_id for t in transactions[-10:]]  # Last 10
            },
            recommended_actions=[
                "Block entity immediately",
                "Review all related transactions",
                "Investigate connected entities",
                "Alert fraud investigation team"
            ],
            confidence=fraud_rate
        )
        
        try:
            self.alert_queue.put_nowait(alert)
        except Exception as e:
            logger.error(f"Failed to queue ring alert: {e}")
    
    async def _store_transaction_redis(self, transaction: TransactionEvent):
        """Store transaction in Redis."""
        try:
            key = f"transaction:{transaction.transaction_id}"
            value = json.dumps(asdict(transaction), default=str)
            self.redis_client.setex(key, 3600, value)  # 1 hour TTL
        except Exception as e:
            logger.error(f"Failed to store transaction in Redis: {e}")
    
    async def _store_alert_redis(self, alert: FraudAlert):
        """Store alert in Redis."""
        try:
            key = f"alert:{alert.alert_id}"
            value = json.dumps(asdict(alert), default=str)
            self.redis_client.setex(key, 86400, value)  # 24 hour TTL
            
            # Also add to alerts list
            self.redis_client.lpush("alerts:recent", value)
            self.redis_client.ltrim("alerts:recent", 0, 999)  # Keep last 1000 alerts
        except Exception as e:
            logger.error(f"Failed to store alert in Redis: {e}")
    
    async def _broadcast_alert(self, alert: FraudAlert):
        """Broadcast alert to WebSocket clients."""
        if not self.websocket_clients:
            return
        
        message = json.dumps({
            'type': 'alert',
            'data': asdict(alert)
        }, default=str)
        
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except Exception as e:
                logger.warning(f"Failed to send alert to WebSocket client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return {
            'transactions_processed': len(self.transaction_window),
            'alerts_pending': self.alert_queue.qsize(),
            'websocket_clients': len(self.websocket_clients),
            'entity_count': len(self.entity_stats),
            'velocity_windows': {
                name: len(window) for name, window in self.velocity_windows.items()
            },
            'top_risky_entities': sorted(
                [(entity, stats['risk_score']) for entity, stats in self.entity_stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

def main():
    """Example usage of real-time monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Start real-time fraud monitoring")
    parser.add_argument('--redis-host', default='localhost', help='Redis host')
    parser.add_argument('--redis-port', type=int, default=6379, help='Redis port')
    
    args = parser.parse_args()
    
    # Create and start monitor
    monitor = RealTimeFraudMonitor(args.redis_host, args.redis_port)
    
    async def run_monitor():
        try:
            await monitor.start_monitoring()
        except KeyboardInterrupt:
            logger.info("Shutting down monitor...")
            await monitor.stop_monitoring()
    
    # Run the monitor
    asyncio.run(run_monitor())

if __name__ == "__main__":
    main()