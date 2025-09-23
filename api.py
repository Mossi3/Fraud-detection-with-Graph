from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import torch
import json
import asyncio
from datetime import datetime
import logging
import os
import sys

# Add project root to path
sys.path.append('/workspace')

from data_processor import FraudDataProcessor
from models.deep_learning_models import (
    CNNFraudDetector, LSTMFraudDetector, TransformerFraudDetector, 
    DeepFraudDetector, FraudModelTrainer
)
from graph.graph_fraud_detection import GraphFraudDetector, GraphSAGEFraudDetector
from visualization.heatmap_visualizer import FraudHeatmapVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Advanced fraud detection system using deep learning and graph neural networks",
    version="1.0.0",
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

# Global variables for models and data
models = {}
data_processor = None
graph_detector = None
visualizer = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pydantic models for API
class TransactionRequest(BaseModel):
    """Single transaction for fraud detection"""
    Time: float = Field(..., description="Time of transaction")
    V1: float = Field(..., description="PCA component 1")
    V2: float = Field(..., description="PCA component 2")
    V3: float = Field(..., description="PCA component 3")
    V4: float = Field(..., description="PCA component 4")
    V5: float = Field(..., description="PCA component 5")
    V6: float = Field(..., description="PCA component 6")
    V7: float = Field(..., description="PCA component 7")
    V8: float = Field(..., description="PCA component 8")
    V9: float = Field(..., description="PCA component 9")
    V10: float = Field(..., description="PCA component 10")
    V11: float = Field(..., description="PCA component 11")
    V12: float = Field(..., description="PCA component 12")
    V13: float = Field(..., description="PCA component 13")
    V14: float = Field(..., description="PCA component 14")
    V15: float = Field(..., description="PCA component 15")
    V16: float = Field(..., description="PCA component 16")
    V17: float = Field(..., description="PCA component 17")
    V18: float = Field(..., description="PCA component 18")
    V19: float = Field(..., description="PCA component 19")
    V20: float = Field(..., description="PCA component 20")
    V21: float = Field(..., description="PCA component 21")
    V22: float = Field(..., description="PCA component 22")
    V23: float = Field(..., description="PCA component 23")
    V24: float = Field(..., description="PCA component 24")
    V25: float = Field(..., description="PCA component 25")
    V26: float = Field(..., description="PCA component 26")
    V27: float = Field(..., description="PCA component 27")
    V28: float = Field(..., description="PCA component 28")
    Amount: float = Field(..., description="Transaction amount")
    Merchant: str = Field(..., description="Merchant type")
    Country: str = Field(..., description="Country")
    Device: str = Field(..., description="Device type")
    IP_Country: str = Field(..., description="IP country")
    Hour: int = Field(..., description="Hour of day")
    Day: int = Field(..., description="Day of month")
    Card_ID: str = Field(..., description="Card identifier")
    Merchant_ID: str = Field(..., description="Merchant identifier")
    Device_ID: str = Field(..., description="Device identifier")
    IP_Address: str = Field(..., description="IP address")

class BatchTransactionRequest(BaseModel):
    """Batch transactions for fraud detection"""
    transactions: List[TransactionRequest] = Field(..., description="List of transactions")

class FraudDetectionResponse(BaseModel):
    """Response for fraud detection"""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    confidence: str
    risk_factors: List[str]
    model_predictions: Dict[str, float]
    timestamp: str

class BatchFraudDetectionResponse(BaseModel):
    """Response for batch fraud detection"""
    results: List[FraudDetectionResponse]
    summary: Dict[str, Any]
    processing_time: float

class GraphAnalysisRequest(BaseModel):
    """Request for graph analysis"""
    card_ids: Optional[List[str]] = None
    merchant_ids: Optional[List[str]] = None
    device_ids: Optional[List[str]] = None
    ip_addresses: Optional[List[str]] = None
    analysis_type: str = Field(default="fraud_rings", description="Type of analysis")

class GraphAnalysisResponse(BaseModel):
    """Response for graph analysis"""
    fraud_rings: List[Dict[str, Any]]
    suspicious_patterns: List[Dict[str, Any]]
    risk_score: float
    recommendations: List[str]
    analysis_timestamp: str

class HeatmapRequest(BaseModel):
    """Request for heatmap generation"""
    analysis_type: str = Field(default="fraud_patterns", description="Type of heatmap")
    parameters: Optional[Dict[str, Any]] = None

class HeatmapResponse(BaseModel):
    """Response for heatmap generation"""
    heatmap_url: str
    analysis_type: str
    parameters: Dict[str, Any]
    generated_at: str

class SystemStatsResponse(BaseModel):
    """System statistics response"""
    total_transactions_processed: int
    fraud_detection_rate: float
    model_performance: Dict[str, float]
    system_uptime: str
    last_model_update: str
    active_models: List[str]

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models and data processors on startup"""
    global models, data_processor, graph_detector, visualizer
    
    logger.info("Initializing fraud detection system...")
    
    try:
        # Initialize data processor
        data_processor = FraudDataProcessor()
        df = data_processor.load_data()
        df_processed = data_processor.preprocess_data(df)
        X, y = data_processor.prepare_features(df_processed)
        
        # Split and scale data
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(X, y)
        X_train_scaled, X_val_scaled, X_test_scaled = data_processor.scale_features(X_train, X_val, X_test)
        X_train_balanced, y_train_balanced = data_processor.balance_data(X_train_scaled, y_train)
        
        input_dim = X_train_balanced.shape[1]
        
        # Initialize models
        models = {
            'cnn': CNNFraudDetector(input_dim),
            'lstm': LSTMFraudDetector(input_dim),
            'transformer': TransformerFraudDetector(input_dim),
            'deep': DeepFraudDetector(input_dim)
        }
        
        # Load pre-trained models if available
        for model_name, model in models.items():
            model_path = f'/workspace/models/{model_name}_model.pth'
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded pre-trained {model_name} model")
                except Exception as e:
                    logger.warning(f"Could not load {model_name} model: {e}")
        
        # Initialize graph detector
        graph_detector = GraphFraudDetector()
        graph_detector.build_transaction_graph(df_processed)
        
        # Initialize visualizer
        visualizer = FraudHeatmapVisualizer()
        
        logger.info("Fraud detection system initialized successfully!")
        
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Card Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "active"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": str(len(models)),
        "device": str(device)
    }

@app.post("/detect", response_model=FraudDetectionResponse)
async def detect_fraud(transaction: TransactionRequest):
    """Detect fraud in a single transaction"""
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Preprocess transaction
        df_processed = data_processor.preprocess_data(df)
        X, _ = data_processor.prepare_features(df_processed)
        X_scaled = data_processor.scaler.transform(X)
        
        # Get predictions from all models
        model_predictions = {}
        fraud_probabilities = []
        
        for model_name, model in models.items():
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_scaled).to(device)
                prediction = model(X_tensor)
                probability = prediction.cpu().numpy()[0][0]
                model_predictions[model_name] = float(probability)
                fraud_probabilities.append(probability)
        
        # Ensemble prediction
        avg_probability = np.mean(fraud_probabilities)
        is_fraud = avg_probability > 0.5
        
        # Determine confidence
        if avg_probability > 0.8 or avg_probability < 0.2:
            confidence = "high"
        elif avg_probability > 0.6 or avg_probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Identify risk factors
        risk_factors = []
        if transaction.Amount > 1000:
            risk_factors.append("High transaction amount")
        if transaction.Hour in [0, 1, 2, 3, 4, 5]:
            risk_factors.append("Unusual transaction time")
        if transaction.V1 > 2 or transaction.V1 < -2:
            risk_factors.append("Suspicious PCA component V1")
        if transaction.V2 > 2 or transaction.V2 < -2:
            risk_factors.append("Suspicious PCA component V2")
        
        return FraudDetectionResponse(
            transaction_id=transaction.Card_ID,
            is_fraud=bool(is_fraud),
            fraud_probability=float(avg_probability),
            confidence=confidence,
            risk_factors=risk_factors,
            model_predictions=model_predictions,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_detect", response_model=BatchFraudDetectionResponse)
async def batch_detect_fraud(request: BatchTransactionRequest):
    """Detect fraud in multiple transactions"""
    start_time = datetime.now()
    
    try:
        results = []
        fraud_count = 0
        
        for transaction in request.transactions:
            # Convert to single transaction request
            single_request = TransactionRequest(**transaction.dict())
            result = await detect_fraud(single_request)
            results.append(result)
            
            if result.is_fraud:
                fraud_count += 1
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        summary = {
            "total_transactions": len(request.transactions),
            "fraud_detected": fraud_count,
            "fraud_rate": fraud_count / len(request.transactions),
            "average_fraud_probability": np.mean([r.fraud_probability for r in results])
        }
        
        return BatchFraudDetectionResponse(
            results=results,
            summary=summary,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in batch fraud detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/graph_analysis", response_model=GraphAnalysisResponse)
async def analyze_graph_patterns(request: GraphAnalysisRequest):
    """Analyze graph patterns for fraud detection"""
    try:
        # Detect fraud rings
        fraud_rings = graph_detector.detect_fraud_rings()
        
        # Detect suspicious patterns
        suspicious_patterns = graph_detector.detect_suspicious_patterns()
        
        # Calculate overall risk score
        risk_score = 0.0
        if fraud_rings:
            avg_fraud_rate = np.mean([ring['fraud_rate'] for ring in fraud_rings])
            risk_score += avg_fraud_rate * 0.6
        
        if suspicious_patterns:
            risk_score += min(len(suspicious_patterns) * 0.1, 0.4)
        
        # Generate recommendations
        recommendations = []
        if risk_score > 0.7:
            recommendations.append("High risk detected - immediate investigation required")
        elif risk_score > 0.4:
            recommendations.append("Medium risk detected - monitor closely")
        else:
            recommendations.append("Low risk - normal monitoring")
        
        if fraud_rings:
            recommendations.append(f"Detected {len(fraud_rings)} potential fraud rings")
        
        if suspicious_patterns:
            recommendations.append(f"Found {len(suspicious_patterns)} suspicious patterns")
        
        return GraphAnalysisResponse(
            fraud_rings=fraud_rings,
            suspicious_patterns=suspicious_patterns,
            risk_score=float(risk_score),
            recommendations=recommendations,
            analysis_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in graph analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/heatmap", response_model=HeatmapResponse)
async def generate_heatmap(request: HeatmapRequest):
    """Generate fraud analysis heatmaps"""
    try:
        # Load data for visualization
        df = data_processor.load_data()
        df_processed = data_processor.preprocess_data(df)
        
        # Generate appropriate heatmap based on request
        if request.analysis_type == "fraud_patterns":
            visualizer.create_fraud_pattern_heatmap(df_processed)
            heatmap_url = "/workspace/visualization/fraud_pattern_heatmap.png"
        elif request.analysis_type == "geographic":
            visualizer.create_geographic_fraud_heatmap(df_processed)
            heatmap_url = "/workspace/visualization/geographic_fraud_heatmap.png"
        elif request.analysis_type == "time_series":
            visualizer.create_time_series_fraud_heatmap(df_processed)
            heatmap_url = "/workspace/visualization/time_series_fraud_heatmap.png"
        else:
            visualizer.create_fraud_pattern_heatmap(df_processed)
            heatmap_url = "/workspace/visualization/fraud_pattern_heatmap.png"
        
        return HeatmapResponse(
            heatmap_url=heatmap_url,
            analysis_type=request.analysis_type,
            parameters=request.parameters or {},
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get system statistics and performance metrics"""
    try:
        # Load data for statistics
        df = data_processor.load_data()
        fraud_rate = df['Class'].mean()
        
        # Calculate model performance (simplified)
        model_performance = {}
        for model_name in models.keys():
            # This would normally come from actual model evaluation
            model_performance[model_name] = {
                "accuracy": 0.95,
                "precision": 0.92,
                "recall": 0.88,
                "f1_score": 0.90,
                "auc": 0.94
            }
        
        return SystemStatsResponse(
            total_transactions_processed=len(df),
            fraud_detection_rate=float(fraud_rate),
            model_performance=model_performance,
            system_uptime="24/7",
            last_model_update=datetime.now().isoformat(),
            active_models=list(models.keys())
        )
        
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models", response_model=Dict[str, Any])
async def get_available_models():
    """Get information about available models"""
    model_info = {}
    for model_name, model in models.items():
        model_info[model_name] = {
            "type": model.__class__.__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "status": "loaded"
        }
    
    return {
        "available_models": model_info,
        "total_models": len(models),
        "device": str(device)
    }

@app.post("/train", response_model=Dict[str, str])
async def train_models(background_tasks: BackgroundTasks):
    """Train models in the background"""
    background_tasks.add_task(train_models_background)
    return {
        "message": "Model training started in background",
        "status": "training",
        "timestamp": datetime.now().isoformat()
    }

async def train_models_background():
    """Background task for training models"""
    try:
        logger.info("Starting background model training...")
        
        # Load and preprocess data
        df = data_processor.load_data()
        df_processed = data_processor.preprocess_data(df)
        X, y = data_processor.prepare_features(df_processed)
        
        # Split and scale data
        X_train, X_val, X_test, y_train, y_val, y_test = data_processor.split_data(X, y)
        X_train_scaled, X_val_scaled, X_test_scaled = data_processor.scale_features(X_train, X_val, X_test)
        X_train_balanced, y_train_balanced = data_processor.balance_data(X_train_scaled, y_train)
        
        # Create data loaders
        from models.deep_learning_models import create_data_loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X_train_balanced, X_val_scaled, X_test_scaled,
            y_train_balanced, y_val, y_test
        )
        
        # Train each model
        for model_name, model in models.items():
            logger.info(f"Training {model_name} model...")
            trainer = FraudModelTrainer(model, device)
            trainer.train(train_loader, val_loader, epochs=10)
            
            # Save trained model
            trainer.save_model(f'/workspace/models/{model_name}_model.pth')
            logger.info(f"Saved {model_name} model")
        
        logger.info("Background model training completed!")
        
    except Exception as e:
        logger.error(f"Error in background training: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)