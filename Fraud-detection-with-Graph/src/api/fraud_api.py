"""
Real-time fraud detection API using FastAPI.
Provides endpoints for transaction scoring, ring detection, and monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import asyncio
import redis
import json
import numpy as np
import torch
import pickle
import os
from loguru import logger
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.graph_builder import FraudGraph, GraphBuilder, Transaction
from features.feature_extractor import GraphFeatureExtractor
from models.gnn_models import GraphSAGE, GAT, FraudGNN
from models.community_detection import CommunityDetector
from visualization.graph_visualizer import FraudGraphVisualizer
from monitoring.metrics_collector import MetricsCollector


# API Models
class TransactionRequest(BaseModel):
    """Transaction scoring request"""
    transaction_id: str
    card_id: str
    merchant_id: str
    amount: float = Field(gt=0)
    timestamp: datetime
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    location: Optional[List[float]] = None
    additional_features: Optional[Dict[str, Any]] = None
    
    @validator('location')
    def validate_location(cls, v):
        if v and len(v) != 2:
            raise ValueError('Location must be [latitude, longitude]')
        return v


class TransactionResponse(BaseModel):
    """Transaction scoring response"""
    transaction_id: str
    fraud_score: float = Field(ge=0, le=1)
    risk_level: str
    fraud_patterns: List[str]
    related_entities: Dict[str, List[str]]
    processing_time_ms: float
    recommendation: str


class RingDetectionRequest(BaseModel):
    """Fraud ring detection request"""
    entity_ids: List[str]
    detection_method: str = "louvain"
    min_ring_size: int = Field(default=3, ge=2)
    time_window_hours: Optional[int] = None


class RingDetectionResponse(BaseModel):
    """Fraud ring detection response"""
    fraud_rings: List[List[str]]
    ring_scores: Dict[str, float]
    ring_characteristics: List[Dict[str, Any]]
    visualization_url: Optional[str] = None


class BatchScoringRequest(BaseModel):
    """Batch transaction scoring request"""
    transactions: List[TransactionRequest]
    include_visualization: bool = False


class GraphStatsResponse(BaseModel):
    """Graph statistics response"""
    total_nodes: int
    total_edges: int
    node_type_distribution: Dict[str, int]
    fraud_rate: float
    active_fraud_rings: int
    graph_metrics: Dict[str, float]


# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection Graph API",
    description="Real-time fraud detection using Graph Neural Networks",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global instances
class FraudDetectionSystem:
    """Main fraud detection system"""
    
    def __init__(self):
        self.graph_builder = GraphBuilder()
        self.fraud_graph = FraudGraph()
        self.feature_extractor = GraphFeatureExtractor()
        self.community_detector = CommunityDetector()
        self.visualizer = FraudGraphVisualizer()
        self.metrics_collector = MetricsCollector()
        
        # Models
        self.models = {}
        self.load_models()
        
        # Redis for caching
        self.redis_client = None
        self.init_redis()
        
        # Transaction buffer for real-time updates
        self.transaction_buffer = []
        self.last_graph_update = datetime.now()
        
    def init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=int(os.getenv('REDIS_DB', 0)),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def load_models(self):
        """Load pre-trained models"""
        model_path = os.getenv('MODEL_PATH', './models')
        
        try:
            # Load GraphSAGE
            if os.path.exists(f"{model_path}/graphsage.pt"):
                self.models['graphsage'] = torch.load(f"{model_path}/graphsage.pt")
                self.models['graphsage'].eval()
                logger.info("GraphSAGE model loaded")
            
            # Load GAT
            if os.path.exists(f"{model_path}/gat.pt"):
                self.models['gat'] = torch.load(f"{model_path}/gat.pt")
                self.models['gat'].eval()
                logger.info("GAT model loaded")
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    async def score_transaction(self, transaction: TransactionRequest) -> TransactionResponse:
        """Score a single transaction for fraud"""
        start_time = datetime.now()
        
        # Convert to Transaction object
        txn = Transaction(
            transaction_id=transaction.transaction_id,
            card_id=transaction.card_id,
            merchant_id=transaction.merchant_id,
            amount=transaction.amount,
            timestamp=transaction.timestamp,
            device_id=transaction.device_id,
            ip_address=transaction.ip_address,
            location=tuple(transaction.location) if transaction.location else None,
            additional_features=transaction.additional_features
        )
        
        # Add to buffer
        self.transaction_buffer.append(txn)
        
        # Update graph if needed
        if len(self.transaction_buffer) >= 100 or \
           (datetime.now() - self.last_graph_update).seconds > 60:
            await self._update_graph()
        
        # Extract features
        node_id = f"txn_{transaction.transaction_id}"
        if node_id not in self.fraud_graph.graph:
            # Process single transaction
            self.graph_builder._process_transaction(txn)
        
        features = self.feature_extractor.extract_node_features(
            self.fraud_graph.graph, node_id
        )
        
        # Score with models
        fraud_score = await self._score_with_models(features)
        
        # Detect patterns
        patterns = self._detect_transaction_patterns(txn)
        
        # Get related entities
        related_entities = self._get_related_entities(node_id)
        
        # Determine risk level and recommendation
        risk_level, recommendation = self._assess_risk(fraud_score, patterns)
        
        # Track metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        self.metrics_collector.record_prediction(fraud_score, processing_time)
        
        return TransactionResponse(
            transaction_id=transaction.transaction_id,
            fraud_score=fraud_score,
            risk_level=risk_level,
            fraud_patterns=patterns,
            related_entities=related_entities,
            processing_time_ms=processing_time,
            recommendation=recommendation
        )
    
    async def _update_graph(self):
        """Update graph with buffered transactions"""
        if self.transaction_buffer:
            self.fraud_graph = self.graph_builder.build_from_transactions(
                self.transaction_buffer
            )
            self.transaction_buffer = []
            self.last_graph_update = datetime.now()
            logger.info("Graph updated with new transactions")
    
    async def _score_with_models(self, features: np.ndarray) -> float:
        """Score transaction using ensemble of models"""
        scores = []
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # GraphSAGE prediction
        if 'graphsage' in self.models:
            with torch.no_grad():
                sage_output = self.models['graphsage'](
                    feature_tensor, 
                    torch.LongTensor([[0, 0]])  # Dummy edge for single node
                )
                sage_score = torch.softmax(sage_output, dim=1)[0, 1].item()
                scores.append(sage_score)
        
        # GAT prediction
        if 'gat' in self.models:
            with torch.no_grad():
                gat_output, _ = self.models['gat'](
                    feature_tensor,
                    torch.LongTensor([[0, 0]])
                )
                gat_score = torch.softmax(gat_output, dim=1)[0, 1].item()
                scores.append(gat_score)
        
        # Rule-based scoring
        rule_score = self._calculate_rule_based_score(features)
        scores.append(rule_score)
        
        # Ensemble averaging
        return np.mean(scores) if scores else 0.5
    
    def _calculate_rule_based_score(self, features: np.ndarray) -> float:
        """Calculate rule-based fraud score"""
        score = 0.0
        
        # High-risk indicators
        if features[0] > 10:  # High in-degree
            score += 0.2
        if features[8] > 2:   # Multiple velocity patterns
            score += 0.3
        if features[14] > 0:  # Fraud connections
            score += 0.3
        if features[15] > 0.5:  # High fraud proximity
            score += 0.2
            
        return min(score, 1.0)
    
    def _detect_transaction_patterns(self, transaction: Transaction) -> List[str]:
        """Detect fraud patterns in transaction"""
        patterns = []
        
        # Velocity check
        recent_txns = self._get_recent_transactions(transaction.card_id, hours=1)
        if len(recent_txns) > 5:
            patterns.append("high_velocity")
        
        # Amount anomaly
        if transaction.amount > 5000:
            patterns.append("high_amount")
        
        # Time anomaly
        if 0 <= transaction.timestamp.hour < 6:
            patterns.append("unusual_time")
        
        # Location anomaly
        if transaction.location:
            # Check against card's typical locations
            if self._is_location_anomaly(transaction.card_id, transaction.location):
                patterns.append("location_anomaly")
        
        # Device/IP sharing
        if transaction.device_id:
            sharing_cards = self._get_device_sharing_cards(transaction.device_id)
            if len(sharing_cards) > 2:
                patterns.append("device_sharing")
        
        return patterns
    
    def _get_recent_transactions(self, card_id: str, hours: int) -> List[Transaction]:
        """Get recent transactions for a card"""
        recent = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for txn in self.transaction_buffer:
            if txn.card_id == card_id and txn.timestamp >= cutoff_time:
                recent.append(txn)
                
        return recent
    
    def _is_location_anomaly(self, card_id: str, location: tuple) -> bool:
        """Check if location is anomalous for card"""
        # Simplified check - in production, would use historical locations
        return False
    
    def _get_device_sharing_cards(self, device_id: str) -> Set[str]:
        """Get cards sharing a device"""
        cards = set()
        
        for txn in self.transaction_buffer:
            if txn.device_id == device_id:
                cards.add(txn.card_id)
                
        return cards
    
    def _get_related_entities(self, node_id: str) -> Dict[str, List[str]]:
        """Get entities related to a transaction"""
        related = {
            'cards': [],
            'merchants': [],
            'devices': [],
            'ips': []
        }
        
        if node_id not in self.fraud_graph.graph:
            return related
        
        # Get neighbors
        for neighbor in self.fraud_graph.graph.neighbors(node_id):
            neighbor_type = self.fraud_graph.graph.nodes[neighbor].get('entity_type')
            
            if neighbor_type == 'card':
                related['cards'].append(neighbor.replace('card_', ''))
            elif neighbor_type == 'merchant':
                related['merchants'].append(neighbor.replace('merchant_', ''))
            elif neighbor_type == 'device':
                related['devices'].append(neighbor.replace('device_', ''))
            elif neighbor_type == 'ip_address':
                related['ips'].append(neighbor.replace('ip_', ''))
                
        return related
    
    def _assess_risk(self, fraud_score: float, patterns: List[str]) -> tuple:
        """Assess risk level and provide recommendation"""
        # Adjust score based on patterns
        pattern_boost = len(patterns) * 0.1
        adjusted_score = min(fraud_score + pattern_boost, 1.0)
        
        if adjusted_score >= 0.8:
            risk_level = "CRITICAL"
            recommendation = "BLOCK - High fraud probability detected"
        elif adjusted_score >= 0.6:
            risk_level = "HIGH"
            recommendation = "REVIEW - Manual verification required"
        elif adjusted_score >= 0.4:
            risk_level = "MEDIUM"
            recommendation = "MONITOR - Additional authentication recommended"
        elif adjusted_score >= 0.2:
            risk_level = "LOW"
            recommendation = "ALLOW - Continue monitoring"
        else:
            risk_level = "MINIMAL"
            recommendation = "ALLOW - Transaction appears legitimate"
            
        return risk_level, recommendation
    
    async def detect_fraud_rings(self, request: RingDetectionRequest) -> RingDetectionResponse:
        """Detect fraud rings from entities"""
        # Get subgraph around entities
        subgraph = self.fraud_graph.get_subgraph(request.entity_ids, max_depth=2)
        
        # Detect communities
        communities = self.community_detector.detect_communities(
            subgraph, method=request.detection_method
        )
        
        # Filter for fraud rings
        fraud_labels = {node: self._is_fraudulent_entity(node) 
                       for node in subgraph.nodes()}
        
        fraud_rings = self.community_detector.detect_fraud_rings(
            subgraph, fraud_labels, 
            min_ring_size=request.min_ring_size
        )
        
        # Analyze rings
        ring_characteristics = []
        ring_scores = {}
        
        for ring in fraud_rings:
            ring_frozen = frozenset(ring)
            characteristics = self.community_detector.analyze_ring_characteristics(
                subgraph, ring
            )
            ring_characteristics.append(characteristics)
            ring_scores[str(list(ring))] = self.community_detector.ring_scores.get(
                ring_frozen, 0
            )
        
        # Create visualization
        viz_url = None
        if fraud_rings:
            viz_path = f"visualizations/rings_{datetime.now().timestamp()}.html"
            self.visualizer.visualize_fraud_ring(
                subgraph, fraud_rings[0], fraud_labels, viz_path
            )
            viz_url = f"/static/{viz_path}"
        
        return RingDetectionResponse(
            fraud_rings=[list(ring) for ring in fraud_rings],
            ring_scores=ring_scores,
            ring_characteristics=ring_characteristics,
            visualization_url=viz_url
        )
    
    def _is_fraudulent_entity(self, node_id: str) -> bool:
        """Check if entity is fraudulent"""
        # In production, would check against known fraud database
        # For demo, use heuristics
        node_data = self.fraud_graph.graph.nodes.get(node_id, {})
        return node_data.get('is_fraud', False)
    
    def get_graph_statistics(self) -> GraphStatsResponse:
        """Get current graph statistics"""
        stats = self.graph_builder.get_graph_statistics()
        
        # Count fraud nodes
        fraud_count = sum(1 for _, data in self.fraud_graph.graph.nodes(data=True)
                         if data.get('is_fraud', False))
        fraud_rate = fraud_count / max(stats['total_nodes'], 1)
        
        # Count active rings
        active_rings = len(self.community_detector.fraud_rings)
        
        return GraphStatsResponse(
            total_nodes=stats['total_nodes'],
            total_edges=stats['total_edges'],
            node_type_distribution=stats['node_types'],
            fraud_rate=fraud_rate,
            active_fraud_rings=active_rings,
            graph_metrics={
                'density': stats['density'],
                'avg_degree': stats['avg_degree'],
                'max_degree': stats['max_degree'],
                'num_components': stats['num_components']
            }
        )


# Initialize system
fraud_system = FraudDetectionSystem()


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """API documentation"""
    return """
    <html>
        <head>
            <title>Fraud Detection Graph API</title>
        </head>
        <body>
            <h1>Fraud Detection Graph API</h1>
            <p>Real-time fraud detection using Graph Neural Networks</p>
            <h2>Available Endpoints:</h2>
            <ul>
                <li><a href="/docs">Interactive API Documentation</a></li>
                <li>POST /api/v1/score - Score a single transaction</li>
                <li>POST /api/v1/batch_score - Score multiple transactions</li>
                <li>POST /api/v1/detect_rings - Detect fraud rings</li>
                <li>GET /api/v1/stats - Get graph statistics</li>
                <li>GET /api/v1/health - Health check</li>
            </ul>
        </body>
    </html>
    """


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "graph_loaded": fraud_system.fraud_graph.graph.number_of_nodes() > 0,
        "models_loaded": len(fraud_system.models) > 0,
        "redis_connected": fraud_system.redis_client is not None
    }


@app.post("/api/v1/score", response_model=TransactionResponse)
async def score_transaction(transaction: TransactionRequest):
    """Score a single transaction for fraud"""
    try:
        response = await fraud_system.score_transaction(transaction)
        return response
    except Exception as e:
        logger.error(f"Error scoring transaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch_score")
async def batch_score_transactions(request: BatchScoringRequest):
    """Score multiple transactions"""
    try:
        results = []
        
        for transaction in request.transactions:
            result = await fraud_system.score_transaction(transaction)
            results.append(result)
        
        # Generate visualization if requested
        viz_url = None
        if request.include_visualization:
            viz_path = f"visualizations/batch_{datetime.now().timestamp()}.html"
            fraud_system.visualizer.plot_fraud_statistics(
                fraud_system.fraud_graph.graph,
                {},  # fraud labels
                output_path=viz_path
            )
            viz_url = f"/static/{viz_path}"
        
        return {
            "results": results,
            "total_transactions": len(results),
            "high_risk_count": sum(1 for r in results if r.risk_level in ["HIGH", "CRITICAL"]),
            "visualization_url": viz_url
        }
    except Exception as e:
        logger.error(f"Error in batch scoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/detect_rings", response_model=RingDetectionResponse)
async def detect_fraud_rings(request: RingDetectionRequest):
    """Detect fraud rings from entities"""
    try:
        response = await fraud_system.detect_fraud_rings(request)
        return response
    except Exception as e:
        logger.error(f"Error detecting fraud rings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", response_model=GraphStatsResponse)
async def get_graph_statistics():
    """Get current graph statistics"""
    try:
        return fraud_system.get_graph_statistics()
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics")
async def get_metrics():
    """Get performance metrics"""
    try:
        metrics = fraud_system.metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    logger.info("Starting Fraud Detection API...")
    
    # Load initial data if available
    data_path = os.getenv("INITIAL_DATA_PATH")
    if data_path and os.path.exists(data_path):
        # Load and process initial transactions
        logger.info(f"Loading initial data from {data_path}")
        # Implementation depends on data format
    
    logger.info("Fraud Detection API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Fraud Detection API...")
    
    # Save current state if needed
    if fraud_system.redis_client:
        fraud_system.redis_client.close()
    
    logger.info("Fraud Detection API shut down")


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(app, host=host, port=port)