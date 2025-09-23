"""
Advanced REST API for Graph-based Fraud Detection
Comprehensive endpoints for fraud detection, community analysis, and visualization
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import json
import logging
import asyncio
from datetime import datetime
import os
import tempfile
import uuid

# Import our graph modules
from src.graph.construction import HeterogeneousGraphBuilder, GraphConfig
from src.graph.community_detection import CommunityDetector, FraudRingAnalyzer
from src.graph.models import create_gnn_model
from src.graph.training import GNNTrainer, GraphDataProcessor
from src.graph.visualization import GraphVisualizer
from src.graph.data_generator import FraudRingGenerator

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FraudGraphX API",
    description="Advanced Graph-based Fraud Detection API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for caching
_graph_cache = {}
_model_cache = {}
_analysis_cache = {}

# Pydantic models
class TransactionData(BaseModel):
    transaction_id: str
    card_id: str
    merchant_id: str
    device_id: str
    ip: str
    amount: float
    timestamp: Optional[int] = None

class BatchTransactionData(BaseModel):
    transactions: List[TransactionData]

class GraphConfigRequest(BaseModel):
    include_cards: bool = True
    include_merchants: bool = True
    include_devices: bool = True
    include_ips: bool = True
    include_accounts: bool = False
    card_merchant_edges: bool = True
    device_ip_edges: bool = True
    card_device_edges: bool = True
    merchant_device_edges: bool = True
    account_card_edges: bool = False
    time_window_hours: int = 24
    temporal_decay: float = 0.9
    directed: bool = True
    weighted: bool = True
    self_loops: bool = False

class CommunityDetectionRequest(BaseModel):
    method: str = Field("louvain", description="Community detection method: louvain, leiden, or both")
    resolution: float = Field(1.0, description="Resolution parameter for modularity optimization")
    min_community_size: int = Field(3, description="Minimum community size")
    fraud_threshold: float = Field(0.3, description="Minimum fraud rate to consider community suspicious")

class GNNTrainingRequest(BaseModel):
    model_type: str = Field("graphsage", description="GNN model type: graphsage, gat, transformer, ensemble")
    epochs: int = Field(100, description="Number of training epochs")
    learning_rate: float = Field(0.001, description="Learning rate")
    hidden_dim: int = Field(64, description="Hidden dimension size")
    dropout: float = Field(0.1, description="Dropout rate")
    patience: int = Field(10, description="Early stopping patience")

class FraudDetectionRequest(BaseModel):
    transaction: TransactionData
    use_graph_features: bool = True
    use_community_features: bool = True
    threshold: float = 0.5

class SyntheticDataRequest(BaseModel):
    num_rings: int = Field(5, description="Number of fraud rings to generate")
    ring_types: List[str] = Field(["card_testing_ring", "merchant_collusion"], description="Types of rings to generate")
    base_transactions: int = Field(10000, description="Number of legitimate transactions")
    seed: int = Field(42, description="Random seed")

# Utility functions
def get_graph_from_cache(graph_id: str) -> Optional[HeterogeneousGraphBuilder]:
    """Get graph from cache"""
    return _graph_cache.get(graph_id)

def cache_graph(graph_id: str, graph: HeterogeneousGraphBuilder):
    """Cache graph"""
    _graph_cache[graph_id] = graph

def get_model_from_cache(model_id: str) -> Optional[GNNTrainer]:
    """Get model from cache"""
    return _model_cache.get(model_id)

def cache_model(model_id: str, model: GNNTrainer):
    """Cache model"""
    _model_cache[model_id] = model

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FraudGraphX API - Graph-based Fraud Detection",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "graph": "/graph",
            "community": "/community",
            "fraud_detection": "/fraud_detection",
            "gnn_training": "/gnn_training",
            "synthetic_data": "/synthetic_data",
            "visualization": "/visualization"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cached_graphs": len(_graph_cache),
        "cached_models": len(_model_cache)
    }

# Graph Construction Endpoints

@app.post("/graph/build")
async def build_graph(
    transactions: BatchTransactionData,
    config: GraphConfigRequest = GraphConfigRequest(),
    graph_id: Optional[str] = None
):
    """Build a heterogeneous graph from transaction data"""
    
    try:
        # Generate graph ID if not provided
        if graph_id is None:
            graph_id = str(uuid.uuid4())
        
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in transactions.transactions])
        
        # Create graph configuration
        graph_config = GraphConfig(
            include_cards=config.include_cards,
            include_merchants=config.include_merchants,
            include_devices=config.include_devices,
            include_ips=config.include_ips,
            include_accounts=config.include_accounts,
            card_merchant_edges=config.card_merchant_edges,
            device_ip_edges=config.device_ip_edges,
            card_device_edges=config.card_device_edges,
            merchant_device_edges=config.merchant_device_edges,
            account_card_edges=config.account_card_edges,
            time_window_hours=config.time_window_hours,
            temporal_decay=config.temporal_decay,
            directed=config.directed,
            weighted=config.weighted,
            self_loops=config.self_loops
        )
        
        # Build graph
        builder = HeterogeneousGraphBuilder(graph_config)
        builder.add_transaction_data(df)
        
        # Cache graph
        cache_graph(graph_id, builder)
        
        # Get statistics
        stats = builder.get_graph_statistics()
        
        return {
            "graph_id": graph_id,
            "status": "success",
            "statistics": stats,
            "message": f"Graph built successfully with {stats['num_nodes']} nodes and {stats['num_edges']} edges"
        }
        
    except Exception as e:
        logger.error(f"Error building graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error building graph: {str(e)}")

@app.get("/graph/{graph_id}/statistics")
async def get_graph_statistics(graph_id: str):
    """Get graph statistics"""
    
    graph = get_graph_from_cache(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    stats = graph.get_graph_statistics()
    return {"graph_id": graph_id, "statistics": stats}

@app.get("/graph/{graph_id}/centrality")
async def get_centrality_measures(graph_id: str):
    """Get centrality measures for the graph"""
    
    graph = get_graph_from_cache(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    centrality_measures = graph.calculate_centrality_measures()
    return {"graph_id": graph_id, "centrality_measures": centrality_measures}

@app.get("/graph/{graph_id}/suspicious_patterns")
async def get_suspicious_patterns(graph_id: str):
    """Get suspicious patterns in the graph"""
    
    graph = get_graph_from_cache(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    suspicious_patterns = graph.detect_suspicious_patterns()
    return {"graph_id": graph_id, "suspicious_patterns": suspicious_patterns}

# Community Detection Endpoints

@app.post("/community/{graph_id}/detect")
async def detect_communities(
    graph_id: str,
    request: CommunityDetectionRequest = CommunityDetectionRequest()
):
    """Detect communities in the graph"""
    
    graph = get_graph_from_cache(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        # Get NetworkX graph
        nx_graph = graph.to_networkx()
        
        # Create community detector
        detector = CommunityDetector(nx_graph)
        
        # Detect communities based on method
        communities = {}
        
        if request.method in ["louvain", "both"]:
            louvain_communities = detector.detect_louvain_communities(
                resolution=request.resolution
            )
            communities["louvain"] = louvain_communities
        
        if request.method in ["leiden", "both"]:
            leiden_communities = detector.detect_leiden_communities(
                resolution=request.resolution
            )
            communities["leiden"] = leiden_communities
        
        # Detect fraud rings
        fraud_labels = {}  # This would come from transaction data
        fraud_rings = detector.detect_fraud_rings(
            fraud_labels=fraud_labels,
            min_community_size=request.min_community_size,
            fraud_threshold=request.fraud_threshold
        )
        
        # Cache analysis
        analysis_id = f"{graph_id}_community_{request.method}"
        _analysis_cache[analysis_id] = {
            "communities": communities,
            "fraud_rings": fraud_rings,
            "detector": detector
        }
        
        return {
            "analysis_id": analysis_id,
            "communities": communities,
            "fraud_rings": fraud_rings,
            "statistics": detector.community_stats
        }
        
    except Exception as e:
        logger.error(f"Error detecting communities: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error detecting communities: {str(e)}")

@app.get("/community/{analysis_id}/rings")
async def get_fraud_rings(analysis_id: str):
    """Get detected fraud rings"""
    
    analysis = _analysis_cache.get(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return {
        "analysis_id": analysis_id,
        "fraud_rings": analysis["fraud_rings"]
    }

@app.get("/community/{analysis_id}/ring_leaders")
async def get_ring_leaders(analysis_id: str):
    """Get potential ring leaders"""
    
    analysis = _analysis_cache.get(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    detector = analysis["detector"]
    communities = analysis["communities"]
    
    # Get fraud labels (this would come from actual data)
    fraud_labels = {}
    
    ring_leaders = {}
    for method, comms in communities.items():
        analyzer = FraudRingAnalyzer(comms, detector.graph)
        leaders = analyzer.detect_ring_leaders(fraud_labels)
        ring_leaders[method] = leaders
    
    return {
        "analysis_id": analysis_id,
        "ring_leaders": ring_leaders
    }

# GNN Training Endpoints

@app.post("/gnn_training/{graph_id}/train")
async def train_gnn_model(
    graph_id: str,
    request: GNNTrainingRequest = GNNTrainingRequest(),
    model_id: Optional[str] = None
):
    """Train a GNN model for fraud detection"""
    
    graph = get_graph_from_cache(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        # Generate model ID if not provided
        if model_id is None:
            model_id = f"{graph_id}_{request.model_type}_{uuid.uuid4().hex[:8]}"
        
        # Convert graph to PyTorch Geometric format
        nx_graph = graph.to_networkx()
        processor = GraphDataProcessor()
        graph_data = processor.networkx_to_pytorch_geometric(nx_graph)
        
        # Create labels (this would come from actual fraud data)
        labels = torch.zeros(len(graph_data['node_mapping']))
        
        # Create model
        input_dim = graph_data['x'].shape[1]
        model = create_gnn_model(
            request.model_type,
            input_dim,
            hidden_dim=request.hidden_dim,
            dropout=request.dropout
        )
        
        # Create trainer
        trainer = GNNTrainer(model)
        
        # Prepare data
        train_data, val_data, test_data = trainer.prepare_data(graph_data, labels)
        
        # Train model
        training_history = trainer.train(
            train_data, val_data,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            patience=request.patience
        )
        
        # Evaluate model
        metrics = trainer.evaluate(test_data)
        
        # Cache model
        cache_model(model_id, trainer)
        
        return {
            "model_id": model_id,
            "status": "success",
            "metrics": metrics,
            "training_history": training_history,
            "message": f"Model trained successfully with validation AUC: {trainer.best_val_auc:.4f}"
        }
        
    except Exception as e:
        logger.error(f"Error training GNN model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error training GNN model: {str(e)}")

@app.get("/gnn_training/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get trained model metrics"""
    
    model = get_model_from_cache(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "model_id": model_id,
        "metrics": model.metrics,
        "best_val_auc": model.best_val_auc
    }

# Fraud Detection Endpoints

@app.post("/fraud_detection/{model_id}/predict")
async def predict_fraud(
    model_id: str,
    request: FraudDetectionRequest
):
    """Predict fraud for a single transaction"""
    
    model = get_model_from_cache(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        # This is a simplified prediction - in practice, you'd need to:
        # 1. Add the transaction to the graph
        # 2. Extract features
        # 3. Run prediction
        
        # For now, return a mock prediction
        prediction = {
            "transaction_id": request.transaction.transaction_id,
            "fraud_probability": 0.25,  # Mock value
            "prediction": "legitimate" if 0.25 < request.threshold else "fraud",
            "confidence": 0.75,
            "features_used": {
                "graph_features": request.use_graph_features,
                "community_features": request.use_community_features
            }
        }
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error predicting fraud: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting fraud: {str(e)}")

@app.post("/fraud_detection/{model_id}/batch_predict")
async def batch_predict_fraud(
    model_id: str,
    transactions: BatchTransactionData,
    threshold: float = 0.5
):
    """Predict fraud for multiple transactions"""
    
    model = get_model_from_cache(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    
    try:
        predictions = []
        
        for transaction in transactions.transactions:
            # Mock prediction for each transaction
            fraud_prob = np.random.random()  # Replace with actual prediction
            
            prediction = {
                "transaction_id": transaction.transaction_id,
                "fraud_probability": fraud_prob,
                "prediction": "legitimate" if fraud_prob < threshold else "fraud",
                "confidence": max(fraud_prob, 1 - fraud_prob)
            }
            predictions.append(prediction)
        
        return {
            "model_id": model_id,
            "predictions": predictions,
            "summary": {
                "total_transactions": len(predictions),
                "fraud_predictions": sum(1 for p in predictions if p["prediction"] == "fraud"),
                "legitimate_predictions": sum(1 for p in predictions if p["prediction"] == "legitimate")
            }
        }
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in batch prediction: {str(e)}")

# Synthetic Data Generation Endpoints

@app.post("/synthetic_data/generate")
async def generate_synthetic_data(
    request: SyntheticDataRequest = SyntheticDataRequest()
):
    """Generate synthetic fraud ring data"""
    
    try:
        # Create generator
        generator = FraudRingGenerator(seed=request.seed)
        
        # Generate dataset
        df, metadata = generator.generate_evaluation_dataset(
            num_rings=request.num_rings,
            ring_types=request.ring_types,
            base_transactions=request.base_transactions
        )
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()
        
        return {
            "status": "success",
            "metadata": metadata,
            "download_url": f"/synthetic_data/download/{os.path.basename(temp_file.name)}",
            "message": f"Generated {len(df)} transactions with {request.num_rings} fraud rings"
        }
        
    except Exception as e:
        logger.error(f"Error generating synthetic data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating synthetic data: {str(e)}")

@app.get("/synthetic_data/download/{filename}")
async def download_synthetic_data(filename: str):
    """Download generated synthetic data"""
    
    # In a real implementation, you'd have proper file management
    # For now, this is a placeholder
    raise HTTPException(status_code=404, detail="File not found")

# Visualization Endpoints

@app.get("/visualization/{graph_id}/network")
async def get_network_visualization(
    graph_id: str,
    format: str = "json",
    include_communities: bool = False
):
    """Get network visualization data"""
    
    graph = get_graph_from_cache(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    
    try:
        nx_graph = graph.to_networkx()
        
        if format == "json":
            # Convert to JSON format for frontend visualization
            nodes = []
            edges = []
            
            for node, data in nx_graph.nodes(data=True):
                nodes.append({
                    "id": node,
                    "type": data.get("node_type", "unknown"),
                    "degree": nx_graph.degree(node)
                })
            
            for u, v, data in nx_graph.edges(data=True):
                edges.append({
                    "source": u,
                    "target": v,
                    "weight": data.get("weight", 1.0),
                    "type": data.get("edge_type", "unknown")
                })
            
            return {
                "graph_id": graph_id,
                "nodes": nodes,
                "edges": edges,
                "statistics": graph.get_graph_statistics()
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
    except Exception as e:
        logger.error(f"Error creating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating visualization: {str(e)}")

@app.get("/visualization/{analysis_id}/communities")
async def get_community_visualization(analysis_id: str):
    """Get community visualization data"""
    
    analysis = _analysis_cache.get(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    communities = analysis["communities"]
    fraud_rings = analysis["fraud_rings"]
    
    # Convert to visualization format
    community_data = {}
    
    for method, comms in communities.items():
        community_data[method] = []
        
        for comm_id, nodes in comms.items():
            community_data[method].append({
                "id": comm_id,
                "nodes": nodes,
                "size": len(nodes),
                "fraud_ring": comm_id in fraud_rings.get(method, {})
            })
    
    return {
        "analysis_id": analysis_id,
        "communities": community_data,
        "fraud_rings": fraud_rings
    }

# Utility Endpoints

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cached data"""
    
    global _graph_cache, _model_cache, _analysis_cache
    
    _graph_cache.clear()
    _model_cache.clear()
    _analysis_cache.clear()
    
    return {
        "status": "success",
        "message": "All cache cleared"
    }

@app.get("/cache/status")
async def get_cache_status():
    """Get cache status"""
    
    return {
        "graphs": len(_graph_cache),
        "models": len(_model_cache),
        "analyses": len(_analysis_cache),
        "total_memory_usage": "N/A"  # Would implement actual memory tracking
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)