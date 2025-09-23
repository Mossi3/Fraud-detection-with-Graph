"""
Monitoring and metrics collection for fraud detection system.
Tracks performance, accuracy, and system health metrics.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np
from loguru import logger
import json
import asyncio
from dataclasses import dataclass, asdict


# Prometheus metrics
transaction_counter = Counter('fraud_detection_transactions_total', 
                            'Total number of transactions processed',
                            ['risk_level'])

fraud_score_histogram = Histogram('fraud_detection_score_distribution',
                                'Distribution of fraud scores',
                                buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9))

processing_time_histogram = Histogram('fraud_detection_processing_time_ms',
                                    'Transaction processing time in milliseconds',
                                    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500))

active_fraud_rings_gauge = Gauge('fraud_detection_active_rings',
                               'Number of active fraud rings detected')

graph_size_gauge = Gauge('fraud_detection_graph_size',
                        'Size of the fraud detection graph',
                        ['metric'])

model_accuracy_gauge = Gauge('fraud_detection_model_accuracy',
                           'Current model accuracy',
                           ['model'])

api_request_counter = Counter('fraud_detection_api_requests_total',
                            'Total API requests',
                            ['endpoint', 'status'])

alert_counter = Counter('fraud_detection_alerts_total',
                       'Total alerts generated',
                       ['alert_type', 'severity'])


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    alert_type: str
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW
    title: str
    description: str
    entities: List[str]
    metrics: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None


class MetricsCollector:
    """
    Collects and aggregates metrics for the fraud detection system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Time series data storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=1000))
        self.alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Performance tracking
        self.prediction_times = deque(maxlen=1000)
        self.fraud_scores = deque(maxlen=1000)
        self.true_labels = deque(maxlen=1000)
        self.predicted_labels = deque(maxlen=1000)
        
        # Real-time statistics
        self.hourly_stats = defaultdict(lambda: {
            'transactions': 0,
            'fraud_detected': 0,
            'processing_time_sum': 0,
            'unique_cards': set(),
            'unique_merchants': set()
        })
        
        # Alert thresholds
        self.thresholds = {
            'high_fraud_rate': 0.1,
            'processing_time_ms': 1000,
            'velocity_threshold': 10,
            'ring_size_threshold': 10,
            'accuracy_threshold': 0.85
        }
        
    def record_prediction(self, fraud_score: float, processing_time_ms: float,
                         risk_level: str = None):
        """Record a fraud prediction"""
        # Update Prometheus metrics
        fraud_score_histogram.observe(fraud_score)
        processing_time_histogram.observe(processing_time_ms)
        
        if risk_level:
            transaction_counter.labels(risk_level=risk_level).inc()
        
        # Store for analysis
        self.fraud_scores.append(fraud_score)
        self.prediction_times.append(processing_time_ms)
        
        # Update time series
        now = datetime.now()
        self.metrics_history['fraud_scores'].append((now, fraud_score))
        self.metrics_history['processing_times'].append((now, processing_time_ms))
        
        # Update hourly stats
        hour_key = now.strftime('%Y-%m-%d %H:00')
        self.hourly_stats[hour_key]['transactions'] += 1
        self.hourly_stats[hour_key]['processing_time_sum'] += processing_time_ms
        
        if fraud_score > 0.7:
            self.hourly_stats[hour_key]['fraud_detected'] += 1
        
        # Check for anomalies
        self._check_performance_anomalies(processing_time_ms)
    
    def record_graph_update(self, num_nodes: int, num_edges: int,
                          node_type_distribution: Dict[str, int]):
        """Record graph statistics"""
        graph_size_gauge.labels(metric='nodes').set(num_nodes)
        graph_size_gauge.labels(metric='edges').set(num_edges)
        
        now = datetime.now()
        self.metrics_history['graph_nodes'].append((now, num_nodes))
        self.metrics_history['graph_edges'].append((now, num_edges))
        
        for node_type, count in node_type_distribution.items():
            self.metrics_history[f'nodes_{node_type}'].append((now, count))
    
    def record_fraud_ring_detection(self, num_rings: int, 
                                  ring_sizes: List[int],
                                  ring_scores: List[float]):
        """Record fraud ring detection results"""
        active_fraud_rings_gauge.set(num_rings)
        
        now = datetime.now()
        self.metrics_history['fraud_rings'].append((now, num_rings))
        
        if ring_sizes:
            avg_ring_size = np.mean(ring_sizes)
            max_ring_size = max(ring_sizes)
            
            self.metrics_history['avg_ring_size'].append((now, avg_ring_size))
            self.metrics_history['max_ring_size'].append((now, max_ring_size))
            
            # Check for large rings
            if max_ring_size > self.thresholds['ring_size_threshold']:
                self._create_alert(
                    alert_type='large_fraud_ring',
                    severity='HIGH',
                    title=f"Large Fraud Ring Detected",
                    description=f"Detected fraud ring with {max_ring_size} members",
                    metrics={'ring_size': max_ring_size, 'num_rings': num_rings}
                )
    
    def record_api_request(self, endpoint: str, status_code: int,
                         response_time_ms: float):
        """Record API request metrics"""
        status = 'success' if 200 <= status_code < 300 else 'error'
        api_request_counter.labels(endpoint=endpoint, status=status).inc()
        
        now = datetime.now()
        self.metrics_history[f'api_{endpoint}_{status}'].append((now, 1))
        self.metrics_history[f'api_{endpoint}_response_time'].append(
            (now, response_time_ms)
        )
    
    def update_model_accuracy(self, model_name: str, accuracy: float,
                            precision: float = None, recall: float = None):
        """Update model accuracy metrics"""
        model_accuracy_gauge.labels(model=model_name).set(accuracy)
        
        now = datetime.now()
        self.metrics_history[f'{model_name}_accuracy'].append((now, accuracy))
        
        if precision is not None:
            self.metrics_history[f'{model_name}_precision'].append((now, precision))
        
        if recall is not None:
            self.metrics_history[f'{model_name}_recall'].append((now, recall))
        
        # Check accuracy threshold
        if accuracy < self.thresholds['accuracy_threshold']:
            self._create_alert(
                alert_type='low_model_accuracy',
                severity='MEDIUM',
                title=f"Low Model Accuracy: {model_name}",
                description=f"Model accuracy dropped to {accuracy:.2%}",
                metrics={'accuracy': accuracy, 'model': model_name}
            )
    
    def _check_performance_anomalies(self, processing_time_ms: float):
        """Check for performance anomalies"""
        # High processing time
        if processing_time_ms > self.thresholds['processing_time_ms']:
            self._create_alert(
                alert_type='high_processing_time',
                severity='MEDIUM',
                title="High Transaction Processing Time",
                description=f"Processing time: {processing_time_ms:.1f}ms",
                metrics={'processing_time_ms': processing_time_ms}
            )
        
        # Check recent fraud rate
        if len(self.fraud_scores) >= 100:
            recent_fraud_rate = sum(1 for s in list(self.fraud_scores)[-100:]
                                  if s > 0.7) / 100
            
            if recent_fraud_rate > self.thresholds['high_fraud_rate']:
                self._create_alert(
                    alert_type='high_fraud_rate',
                    severity='HIGH',
                    title="High Fraud Rate Detected",
                    description=f"Fraud rate: {recent_fraud_rate:.1%} in last 100 transactions",
                    metrics={'fraud_rate': recent_fraud_rate}
                )
    
    def _create_alert(self, alert_type: str, severity: str,
                     title: str, description: str,
                     entities: List[str] = None,
                     metrics: Dict[str, Any] = None):
        """Create a new alert"""
        alert_id = f"{alert_type}_{datetime.now().timestamp()}"
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            alert_type=alert_type,
            severity=severity,
            title=title,
            description=description,
            entities=entities or [],
            metrics=metrics or {}
        )
        
        self.alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update Prometheus counter
        alert_counter.labels(alert_type=alert_type, severity=severity).inc()
        
        # Log alert
        logger.warning(f"ALERT [{severity}] {title}: {description}")
        
        # Trigger notification (would integrate with external systems)
        asyncio.create_task(self._send_alert_notification(alert))
    
    async def _send_alert_notification(self, alert: Alert):
        """Send alert notification (placeholder for integration)"""
        # In production, would send to:
        # - Slack/Teams
        # - Email
        # - PagerDuty
        # - SMS
        pass
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.alerts:
            alert = self.alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            
            # Remove from active alerts
            del self.alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.title}")
    
    def get_metrics(self, time_range_minutes: int = 60) -> Dict[str, Any]:
        """Get current metrics summary"""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=time_range_minutes)
        
        # Calculate recent metrics
        recent_scores = [score for ts, score in self.metrics_history['fraud_scores']
                        if ts > cutoff_time]
        recent_times = [time for ts, time in self.metrics_history['processing_times']
                       if ts > cutoff_time]
        
        metrics = {
            'timestamp': now.isoformat(),
            'time_range_minutes': time_range_minutes,
            'transactions': {
                'total': len(recent_scores),
                'fraud_detected': sum(1 for s in recent_scores if s > 0.7),
                'fraud_rate': sum(1 for s in recent_scores if s > 0.7) / max(len(recent_scores), 1)
            },
            'performance': {
                'avg_processing_time_ms': np.mean(recent_times) if recent_times else 0,
                'p95_processing_time_ms': np.percentile(recent_times, 95) if recent_times else 0,
                'p99_processing_time_ms': np.percentile(recent_times, 99) if recent_times else 0
            },
            'fraud_scores': {
                'mean': np.mean(recent_scores) if recent_scores else 0,
                'std': np.std(recent_scores) if recent_scores else 0,
                'min': min(recent_scores) if recent_scores else 0,
                'max': max(recent_scores) if recent_scores else 0
            },
            'alerts': {
                'active': len(self.alerts),
                'by_severity': self._count_alerts_by_severity(),
                'recent': [asdict(alert) for alert in list(self.alert_history)[-10:]]
            }
        }
        
        # Add hourly statistics
        recent_hours = []
        for i in range(24):
            hour = now - timedelta(hours=i)
            hour_key = hour.strftime('%Y-%m-%d %H:00')
            if hour_key in self.hourly_stats:
                stats = self.hourly_stats[hour_key]
                recent_hours.append({
                    'hour': hour_key,
                    'transactions': stats['transactions'],
                    'fraud_detected': stats['fraud_detected'],
                    'avg_processing_time': stats['processing_time_sum'] / max(stats['transactions'], 1)
                })
        
        metrics['hourly_stats'] = recent_hours
        
        return metrics
    
    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Count active alerts by severity"""
        counts = defaultdict(int)
        for alert in self.alerts.values():
            counts[alert.severity] += 1
        return dict(counts)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        now = datetime.now()
        
        # Prepare time series data
        time_series = {}
        
        for metric_name, values in self.metrics_history.items():
            if values:
                # Get last 1000 points
                recent_values = list(values)
                time_series[metric_name] = {
                    'timestamps': [ts.isoformat() for ts, _ in recent_values],
                    'values': [val for _, val in recent_values]
                }
        
        # Calculate KPIs
        last_hour_scores = [s for ts, s in self.metrics_history['fraud_scores']
                          if ts > now - timedelta(hours=1)]
        
        kpis = {
            'transactions_per_hour': len(last_hour_scores),
            'current_fraud_rate': sum(1 for s in last_hour_scores if s > 0.7) / max(len(last_hour_scores), 1),
            'active_alerts': len(self.alerts),
            'system_health': self._calculate_system_health()
        }
        
        return {
            'kpis': kpis,
            'time_series': time_series,
            'alerts': [asdict(a) for a in self.alerts.values()],
            'last_updated': now.isoformat()
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score (0-100)"""
        health_score = 100.0
        
        # Deduct for active alerts
        for alert in self.alerts.values():
            if alert.severity == 'CRITICAL':
                health_score -= 20
            elif alert.severity == 'HIGH':
                health_score -= 10
            elif alert.severity == 'MEDIUM':
                health_score -= 5
        
        # Deduct for high processing times
        if self.prediction_times:
            recent_avg = np.mean(list(self.prediction_times)[-100:])
            if recent_avg > 500:
                health_score -= 10
            elif recent_avg > 250:
                health_score -= 5
        
        return max(health_score, 0)
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        data = {
            'metrics': self.get_metrics(),
            'dashboard': self.get_dashboard_data(),
            'export_time': datetime.now().isoformat()
        }
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported metrics to {filepath}")


class AlertManager:
    """
    Manages alerts and notifications for the fraud detection system.
    """
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.notification_channels = []
        
    def add_notification_channel(self, channel):
        """Add a notification channel"""
        self.notification_channels.append(channel)
        
    async def check_alert_conditions(self):
        """Periodically check for alert conditions"""
        while True:
            try:
                # Check various conditions
                await self._check_fraud_spike()
                await self._check_performance_degradation()
                await self._check_ring_formation_rate()
                
                # Sleep for check interval
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in alert checking: {e}")
                await asyncio.sleep(60)
    
    async def _check_fraud_spike(self):
        """Check for sudden spike in fraud"""
        metrics = self.metrics_collector.get_metrics(time_range_minutes=10)
        current_rate = metrics['transactions']['fraud_rate']
        
        # Compare with historical average
        historical_metrics = self.metrics_collector.get_metrics(time_range_minutes=60)
        historical_rate = historical_metrics['transactions']['fraud_rate']
        
        if current_rate > historical_rate * 2 and current_rate > 0.05:
            self.metrics_collector._create_alert(
                alert_type='fraud_spike',
                severity='CRITICAL',
                title="Fraud Spike Detected",
                description=f"Current fraud rate {current_rate:.1%} vs historical {historical_rate:.1%}",
                metrics={'current_rate': current_rate, 'historical_rate': historical_rate}
            )
    
    async def _check_performance_degradation(self):
        """Check for performance degradation"""
        metrics = self.metrics_collector.get_metrics(time_range_minutes=5)
        p95_time = metrics['performance']['p95_processing_time_ms']
        
        if p95_time > 2000:  # 2 seconds
            self.metrics_collector._create_alert(
                alert_type='performance_degradation',
                severity='HIGH',
                title="Performance Degradation",
                description=f"P95 processing time: {p95_time:.0f}ms",
                metrics={'p95_processing_time': p95_time}
            )
    
    async def _check_ring_formation_rate(self):
        """Check for rapid fraud ring formation"""
        # Get recent fraud ring detections
        recent_rings = [
            (ts, count) for ts, count in self.metrics_collector.metrics_history['fraud_rings']
            if ts > datetime.now() - timedelta(hours=1)
        ]
        
        if len(recent_rings) > 5:  # More than 5 new rings in an hour
            self.metrics_collector._create_alert(
                alert_type='rapid_ring_formation',
                severity='HIGH',
                title="Rapid Fraud Ring Formation",
                description=f"Detected {len(recent_rings)} new fraud rings in the last hour",
                metrics={'new_rings': len(recent_rings)}
            )