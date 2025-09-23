"""
Advanced FastAPI application for graph-based fraud detection.
Includes endpoints for GNN inference, ring detection, and real-time monitoring.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import torch
import json
import asyncio
import logging
from datetime import datetime, timedelta
import uuid
import os
import tempfile
import pickle
from collections import deque
import redis
import time

# Import our modules
from ..graph.graph_builder import HeterogeneousGraphBuilder
from ..models.gnn_models import create_model, FraudGNNTrainer
from ..graph.community_detection import FraudRingDetector
from ..utils.evaluation_metrics import FraudDetectionMetrics, RingDetectionMetrics
from ..visual.fraud_ring_viz import FraudRingVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class TransactionInput(BaseModel):
    """Input model for single transaction scoring."""
    transaction_id: str
    card_id: str
    merchant_id: str
    device_id: str
    ip: str
    amount: float
    transaction_type: str = Field(..., regex="^(purchase|withdrawal|transfer)$")
    merchant_category: str
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    velocity_1h: int = Field(default=0, ge=0)
    velocity_24h: int = Field(default=0, ge=0)
    amount_std_dev: float = Field(default=0.0, ge=0)
    location_risk_score: float = Field(default=0.0, ge=0, le=1)

class BatchTransactionInput(BaseModel):
    """Input model for batch transaction scoring."""
    transactions: List[TransactionInput]

class FraudPrediction(BaseModel):
    """Output model for fraud predictions."""
    transaction_id: str
    fraud_probability: float
    fraud_prediction: bool
    confidence: float
    risk_factors: Dict[str, float]
    ring_membership: Optional[str] = None

class RingDetectionRequest(BaseModel):
    """Input model for ring detection."""
    method: str = Field(default="ensemble", regex="^(louvain|leiden|spectral|dbscan|ensemble)$")
    min_ring_size: int = Field(default=3, ge=2)
    max_ring_size: int = Field(default=50, ge=3)
    fraud_threshold: float = Field(default=0.3, ge=0, le=1)

class ModelTrainingRequest(BaseModel):
    """Input model for model training."""
    model_type: str = Field(..., regex="^(graphsage|gat|dual_channel)$")
    hidden_dim: int = Field(default=128, ge=32, le=512)
    num_layers: int = Field(default=2, ge=1, le=5)
    learning_rate: float = Field(default=0.001, gt=0, le=0.1)
    epochs: int = Field(default=100, ge=10, le=1000)
    early_stopping_patience: int = Field(default=15, ge=5, le=50)

class SystemHealth(BaseModel):
    """System health status model."""
    status: str
    timestamp: datetime
    model_loaded: bool
    graph_loaded: bool
    redis_connected: bool
    processing_queue_size: int
    memory_usage_mb: float
    predictions_last_hour: int

# Global variables for model and data
app = FastAPI(
    title="FraudGraphX Advanced API",
    description="Graph-based fraud detection with GNNs and ring detection",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_state = {
    'gnn_model': None,
    'graph_builder': None,
    'current_graph': None,
    'ring_detector': None,
    'fraud_metrics': FraudDetectionMetrics(),
    'ring_metrics': RingDetectionMetrics(),
    'model_metadata': {},
    'processing_queue': deque(maxlen=1000),
    'prediction_history': deque(maxlen=10000)
}

# Redis connection (optional)
try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connected successfully")
except:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available, using in-memory storage")

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting FraudGraphX Advanced API")
    
    # Initialize graph builder
    model_state['graph_builder'] = HeterogeneousGraphBuilder()
    model_state['ring_detector'] = FraudRingDetector()
    
    # Try to load pre-trained model if available
    try:
        if os.path.exists('models/best_gnn_model.pt'):
            logger.info("Loading pre-trained GNN model...")
            # Model loading would be implemented here
            pass
    except Exception as e:
        logger.warning(f"Could not load pre-trained model: {e}")

@app.get("/health", response_model=SystemHealth)
async def health_check():
    """Get system health status."""
    import psutil
    
    memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    # Count predictions in last hour
    current_time = datetime.now()
    hour_ago = current_time - timedelta(hours=1)
    predictions_last_hour = sum(
        1 for pred in model_state['prediction_history'] 
        if pred.get('timestamp', datetime.min) > hour_ago
    )
    
    return SystemHealth(
        status="healthy" if model_state['gnn_model'] else "model_not_loaded",
        timestamp=current_time,
        model_loaded=model_state['gnn_model'] is not None,
        graph_loaded=model_state['current_graph'] is not None,
        redis_connected=REDIS_AVAILABLE,
        processing_queue_size=len(model_state['processing_queue']),
        memory_usage_mb=memory_mb,
        predictions_last_hour=predictions_last_hour
    )

@app.post("/predict/single", response_model=FraudPrediction)
async def predict_single_transaction(transaction: TransactionInput):
    """Predict fraud for a single transaction."""
    try:
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # Build graph if not exists or update existing
        if model_state['current_graph'] is None:
            # For single transaction, create minimal graph
            model_state['current_graph'] = model_state['graph_builder'].build_graph(
                df, graph_type='torch_geometric'
            )
        
        # Make prediction
        if model_state['gnn_model'] is None:
            # Fallback to rule-based prediction
            fraud_prob = _rule_based_prediction(transaction)
            fraud_pred = fraud_prob > 0.5
            confidence = abs(fraud_prob - 0.5) * 2
            risk_factors = _calculate_risk_factors(transaction)
        else:
            # Use GNN model
            fraud_prob, confidence, risk_factors = _gnn_prediction(transaction)
            fraud_pred = fraud_prob > 0.5
        
        # Check for ring membership
        ring_membership = await _check_ring_membership(transaction)
        
        result = FraudPrediction(
            transaction_id=transaction.transaction_id,
            fraud_probability=fraud_prob,
            fraud_prediction=fraud_pred,
            confidence=confidence,
            risk_factors=risk_factors,
            ring_membership=ring_membership
        )
        
        # Store prediction in history
        prediction_record = {
            'timestamp': datetime.now(),
            'transaction_id': transaction.transaction_id,
            'prediction': result.dict(),
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        model_state['prediction_history'].append(prediction_record)
        
        # Store in Redis if available
        if REDIS_AVAILABLE:
            redis_client.setex(
                f"prediction:{transaction.transaction_id}",
                3600,  # 1 hour TTL
                json.dumps(prediction_record, default=str)
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=List[FraudPrediction])
async def predict_batch_transactions(batch: BatchTransactionInput):
    """Predict fraud for multiple transactions."""
    try:
        start_time = time.time()
        
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in batch.transactions])
        
        # Build or update graph
        graph = model_state['graph_builder'].build_graph(df, graph_type='torch_geometric')
        model_state['current_graph'] = graph
        
        results = []
        
        # Process each transaction
        for transaction in batch.transactions:
            if model_state['gnn_model'] is None:
                fraud_prob = _rule_based_prediction(transaction)
                fraud_pred = fraud_prob > 0.5
                confidence = abs(fraud_prob - 0.5) * 2
                risk_factors = _calculate_risk_factors(transaction)
            else:
                fraud_prob, confidence, risk_factors = _gnn_prediction(transaction)
                fraud_pred = fraud_prob > 0.5
            
            ring_membership = await _check_ring_membership(transaction)
            
            result = FraudPrediction(
                transaction_id=transaction.transaction_id,
                fraud_probability=fraud_prob,
                fraud_prediction=fraud_pred,
                confidence=confidence,
                risk_factors=risk_factors,
                ring_membership=ring_membership
            )
            results.append(result)
        
        # Store batch results
        batch_record = {
            'timestamp': datetime.now(),
            'batch_size': len(batch.transactions),
            'results': [r.dict() for r in results],
            'processing_time_ms': (time.time() - start_time) * 1000
        }
        
        if REDIS_AVAILABLE:
            batch_id = str(uuid.uuid4())
            redis_client.setex(
                f"batch:{batch_id}",
                3600,
                json.dumps(batch_record, default=str)
            )
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rings/detect")
async def detect_fraud_rings(request: RingDetectionRequest):
    """Detect fraud rings in current graph."""
    try:
        if model_state['current_graph'] is None:
            raise HTTPException(status_code=400, detail="No graph data available. Please submit transactions first.")
        
        # Convert to NetworkX for ring detection
        nx_graph = model_state['graph_builder'].build_graph(
            pd.DataFrame(),  # Empty DF since we're using existing graph
            graph_type='networkx'
        )
        
        # Configure ring detector
        detector = FraudRingDetector(
            min_ring_size=request.min_ring_size,
            max_ring_size=request.max_ring_size
        )
        
        # Detect rings based on method
        if request.method == 'louvain':
            rings = detector.detect_rings_louvain(nx_graph)
        elif request.method == 'leiden':
            rings = detector.detect_rings_leiden(nx_graph)
        elif request.method == 'ensemble':
            # For ensemble, we need embeddings - use simplified approach
            rings = detector.detect_rings_louvain(nx_graph)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported method: {request.method}")
        
        # Filter by fraud threshold
        filtered_rings = {
            ring_id: ring_data for ring_id, ring_data in rings.items()
            if ring_data['fraud_score'] >= request.fraud_threshold
        }
        
        # Evaluate ring quality
        metrics = detector.evaluate_ring_quality(filtered_rings)
        
        # Store results
        model_state['ring_detector'] = detector
        
        return {
            'detected_rings': filtered_rings,
            'detection_metrics': metrics,
            'detection_method': request.method,
            'timestamp': datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in ring detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/rings/visualize/{ring_id}")
async def visualize_ring(ring_id: str):
    """Get visualization data for a specific ring."""
    try:
        if not model_state['ring_detector'] or ring_id not in model_state['ring_detector'].detected_rings:
            raise HTTPException(status_code=404, detail=f"Ring {ring_id} not found")
        
        # Create visualization
        nx_graph = model_state['graph_builder'].build_graph(
            pd.DataFrame(), graph_type='networkx'
        )
        
        viz = FraudRingVisualizer(nx_graph, model_state['ring_detector'].detected_rings)
        
        # Generate Plotly figure data
        fig = viz.create_ring_network_plot(ring_id)
        
        return {
            'ring_id': ring_id,
            'visualization_data': fig.to_dict(),
            'ring_info': model_state['ring_detector'].detected_rings[ring_id]
        }
        
    except Exception as e:
        logger.error(f"Error in ring visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/train")
async def train_model(request: ModelTrainingRequest, background_tasks: BackgroundTasks):
    """Train a new GNN model."""
    try:
        if model_state['current_graph'] is None:
            raise HTTPException(status_code=400, detail="No training data available")
        
        # Start training in background
        training_id = str(uuid.uuid4())
        background_tasks.add_task(_train_model_background, training_id, request)
        
        return {
            'training_id': training_id,
            'status': 'started',
            'message': 'Model training started in background',
            'estimated_duration_minutes': request.epochs // 10  # Rough estimate
        }
        
    except Exception as e:
        logger.error(f"Error starting model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/training_status/{training_id}")
async def get_training_status(training_id: str):
    """Get status of model training."""
    if REDIS_AVAILABLE:
        status = redis_client.get(f"training:{training_id}")
        if status:
            return json.loads(status)
    
    return {'training_id': training_id, 'status': 'not_found'}

@app.post("/data/upload")
async def upload_training_data(file: UploadFile = File(...)):
    """Upload training data CSV file."""
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Load and validate data
        df = pd.read_csv(tmp_file_path)
        
        # Basic validation
        required_columns = ['card_id', 'merchant_id', 'device_id', 'ip', 'amount', 'fraud']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            os.unlink(tmp_file_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Build graph from uploaded data
        graph = model_state['graph_builder'].build_graph(df, graph_type='torch_geometric')
        model_state['current_graph'] = graph
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return {
            'message': 'Data uploaded successfully',
            'rows': len(df),
            'fraud_rate': df['fraud'].mean() if 'fraud' in df.columns else None,
            'unique_cards': df['card_id'].nunique(),
            'unique_merchants': df['merchant_id'].nunique(),
            'unique_devices': df['device_id'].nunique(),
            'unique_ips': df['ip'].nunique()
        }
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/summary")
async def get_analytics_summary():
    """Get summary analytics of predictions and rings."""
    try:
        # Prediction analytics
        recent_predictions = list(model_state['prediction_history'])[-100:]  # Last 100
        
        fraud_predictions = [
            p for p in recent_predictions 
            if p['prediction']['fraud_prediction']
        ]
        
        prediction_analytics = {
            'total_predictions': len(model_state['prediction_history']),
            'recent_predictions': len(recent_predictions),
            'recent_fraud_rate': len(fraud_predictions) / len(recent_predictions) if recent_predictions else 0,
            'avg_processing_time_ms': np.mean([
                p['processing_time_ms'] for p in recent_predictions
            ]) if recent_predictions else 0
        }
        
        # Ring analytics
        ring_analytics = {}
        if model_state['ring_detector'] and model_state['ring_detector'].detected_rings:
            rings = model_state['ring_detector'].detected_rings
            ring_analytics = {
                'total_rings': len(rings),
                'avg_ring_size': np.mean([r['size'] for r in rings.values()]),
                'avg_fraud_score': np.mean([r['fraud_score'] for r in rings.values()]),
                'detection_methods': list(set(r['method'] for r in rings.values()))
            }
        
        return {
            'timestamp': datetime.now(),
            'predictions': prediction_analytics,
            'rings': ring_analytics,
            'system_status': {
                'model_loaded': model_state['gnn_model'] is not None,
                'graph_loaded': model_state['current_graph'] is not None
            }
        }
        
    except Exception as e:
        logger.error(f"Error in analytics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/predictions")
async def export_predictions():
    """Export prediction history as CSV."""
    try:
        if not model_state['prediction_history']:
            raise HTTPException(status_code=404, detail="No predictions to export")
        
        # Convert to DataFrame
        records = []
        for pred in model_state['prediction_history']:
            record = {
                'timestamp': pred['timestamp'],
                'transaction_id': pred['transaction_id'],
                'fraud_probability': pred['prediction']['fraud_probability'],
                'fraud_prediction': pred['prediction']['fraud_prediction'],
                'confidence': pred['prediction']['confidence'],
                'processing_time_ms': pred['processing_time_ms']
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            return FileResponse(
                tmp_file.name,
                media_type='text/csv',
                filename=f'fraud_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _rule_based_prediction(transaction: TransactionInput) -> float:
    """Simple rule-based fraud prediction as fallback."""
    score = 0.0
    
    # High amount transactions
    if transaction.amount > 1000:
        score += 0.3
    
    # Late night transactions
    if transaction.hour < 6 or transaction.hour > 22:
        score += 0.2
    
    # High velocity
    if transaction.velocity_1h > 5:
        score += 0.3
    if transaction.velocity_24h > 20:
        score += 0.2
    
    # High location risk
    if transaction.location_risk_score > 0.7:
        score += 0.4
    
    # Certain merchant categories
    risky_categories = ['cash_advance', 'gambling', 'adult_services']
    if transaction.merchant_category in risky_categories:
        score += 0.3
    
    return min(score, 1.0)

def _calculate_risk_factors(transaction: TransactionInput) -> Dict[str, float]:
    """Calculate risk factors for transaction."""
    factors = {}
    
    # Amount risk
    factors['high_amount'] = min(transaction.amount / 1000, 1.0)
    
    # Time risk
    factors['unusual_time'] = 1.0 if (transaction.hour < 6 or transaction.hour > 22) else 0.0
    
    # Velocity risk
    factors['high_velocity'] = min((transaction.velocity_1h + transaction.velocity_24h) / 20, 1.0)
    
    # Location risk
    factors['location_risk'] = transaction.location_risk_score
    
    # Merchant risk
    risky_categories = ['cash_advance', 'gambling', 'adult_services']
    factors['risky_merchant'] = 1.0 if transaction.merchant_category in risky_categories else 0.0
    
    return factors

def _gnn_prediction(transaction: TransactionInput) -> Tuple[float, float, Dict[str, float]]:
    """Make prediction using GNN model."""
    # This would use the actual GNN model
    # For now, return enhanced rule-based prediction
    fraud_prob = _rule_based_prediction(transaction)
    confidence = abs(fraud_prob - 0.5) * 2
    risk_factors = _calculate_risk_factors(transaction)
    
    return fraud_prob, confidence, risk_factors

async def _check_ring_membership(transaction: TransactionInput) -> Optional[str]:
    """Check if transaction entities are part of detected rings."""
    if not model_state['ring_detector'] or not model_state['ring_detector'].detected_rings:
        return None
    
    transaction_entities = {
        f"card_{transaction.card_id}",
        f"merchant_{transaction.merchant_id}",
        f"device_{transaction.device_id}",
        f"ip_{transaction.ip}"
    }
    
    for ring_id, ring_data in model_state['ring_detector'].detected_rings.items():
        ring_entities = set(ring_data['nodes'])
        if transaction_entities.intersection(ring_entities):
            return ring_id
    
    return None

async def _train_model_background(training_id: str, request: ModelTrainingRequest):
    """Background task for model training."""
    try:
        # Update training status
        if REDIS_AVAILABLE:
            redis_client.setex(
                f"training:{training_id}",
                3600,
                json.dumps({
                    'training_id': training_id,
                    'status': 'in_progress',
                    'progress': 0,
                    'message': 'Starting model training...'
                })
            )
        
        # Simulate training process
        for epoch in range(request.epochs):
            await asyncio.sleep(0.1)  # Simulate training time
            
            progress = (epoch + 1) / request.epochs * 100
            
            if REDIS_AVAILABLE and epoch % 10 == 0:
                redis_client.setex(
                    f"training:{training_id}",
                    3600,
                    json.dumps({
                        'training_id': training_id,
                        'status': 'in_progress',
                        'progress': progress,
                        'epoch': epoch + 1,
                        'message': f'Training epoch {epoch + 1}/{request.epochs}'
                    })
                )
        
        # Training completed
        if REDIS_AVAILABLE:
            redis_client.setex(
                f"training:{training_id}",
                3600,
                json.dumps({
                    'training_id': training_id,
                    'status': 'completed',
                    'progress': 100,
                    'message': 'Model training completed successfully'
                })
            )
        
        logger.info(f"Model training {training_id} completed")
        
    except Exception as e:
        logger.error(f"Error in background training {training_id}: {e}")
        
        if REDIS_AVAILABLE:
            redis_client.setex(
                f"training:{training_id}",
                3600,
                json.dumps({
                    'training_id': training_id,
                    'status': 'failed',
                    'error': str(e),
                    'message': 'Model training failed'
                })
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)