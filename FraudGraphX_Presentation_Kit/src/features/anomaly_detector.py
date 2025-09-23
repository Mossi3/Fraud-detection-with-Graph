"""
Advanced anomaly detection for fraud detection.
Implements multiple algorithms including autoencoders, isolation forest, and graph-based anomalies.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any, Union
import networkx as nx
from collections import defaultdict
import logging
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepAutoencoder(nn.Module):
    """Deep autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers[:-1])  # Remove last dropout
        
        # Decoder
        decoder_layers = []
        hidden_dims_reversed = hidden_dims[::-1][1:] + [input_dim]
        
        for hidden_dim in hidden_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU() if hidden_dim != input_dim else nn.Sigmoid(),
                nn.Dropout(0.1) if hidden_dim != input_dim else nn.Identity()
            ])
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers[:-1])  # Remove last dropout
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)

class VariationalAutoencoder(nn.Module):
    """Variational autoencoder for probabilistic anomaly detection."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dims: List[int] = [128, 64]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims[::-1]:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.extend([
            nn.Linear(prev_dim, input_dim),
            nn.Sigmoid()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

class GraphAnomalyDetector:
    """Graph-based anomaly detection for fraud rings and suspicious patterns."""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.node_features = {}
        self.edge_features = {}
        self.anomaly_scores = {}
    
    def calculate_node_features(self) -> Dict[str, np.ndarray]:
        """Calculate various node-level features for anomaly detection."""
        features = {}
        
        for node in self.graph.nodes():
            node_features = []
            
            # Basic graph features
            degree = self.graph.degree(node)
            clustering_coeff = nx.clustering(self.graph, node)
            
            # Centrality measures
            try:
                betweenness = nx.betweenness_centrality(self.graph)[node]
                closeness = nx.closeness_centrality(self.graph)[node]
                eigenvector = nx.eigenvector_centrality(self.graph, max_iter=1000)[node]
            except:
                betweenness = closeness = eigenvector = 0.0
            
            # Local neighborhood features
            neighbors = list(self.graph.neighbors(node))
            neighbor_degrees = [self.graph.degree(n) for n in neighbors]
            avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
            
            # Fraud-specific features (if available)
            node_data = self.graph.nodes[node]
            fraud_indicator = node_data.get('fraud', 0)
            node_type = node_data.get('node_type', 'unknown')
            
            # Combine features
            node_features.extend([
                degree,
                clustering_coeff,
                betweenness,
                closeness,
                eigenvector,
                avg_neighbor_degree,
                len(neighbors),
                fraud_indicator
            ])
            
            # Add type-specific features
            if 'transaction' in node:
                node_features.extend([1, 0, 0, 0, 0])  # One-hot for transaction
            elif 'card' in node:
                node_features.extend([0, 1, 0, 0, 0])  # One-hot for card
            elif 'merchant' in node:
                node_features.extend([0, 0, 1, 0, 0])  # One-hot for merchant
            elif 'device' in node:
                node_features.extend([0, 0, 0, 1, 0])  # One-hot for device
            elif 'ip' in node:
                node_features.extend([0, 0, 0, 0, 1])  # One-hot for IP
            else:
                node_features.extend([0, 0, 0, 0, 0])
            
            features[node] = np.array(node_features)
        
        self.node_features = features
        return features
    
    def detect_structural_anomalies(self) -> Dict[str, float]:
        """Detect structural anomalies in the graph."""
        if not self.node_features:
            self.calculate_node_features()
        
        # Convert to matrix
        nodes = list(self.node_features.keys())
        feature_matrix = np.array([self.node_features[node] for node in nodes])
        
        # Normalize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Apply isolation forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(feature_matrix_scaled)
        anomaly_scores = iso_forest.score_samples(feature_matrix_scaled)
        
        # Convert to dictionary
        structural_anomalies = {}
        for i, node in enumerate(nodes):
            structural_anomalies[node] = {
                'anomaly_score': anomaly_scores[i],
                'is_anomaly': anomaly_labels[i] == -1,
                'features': self.node_features[node]
            }
        
        return structural_anomalies
    
    def detect_community_anomalies(self) -> Dict[str, float]:
        """Detect nodes that don't fit well into communities."""
        try:
            # Detect communities
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph)
            
            # Calculate modularity contribution of each node
            modularity_contributions = {}
            total_edges = self.graph.number_of_edges()
            
            for node in self.graph.nodes():
                node_community = partition[node]
                
                # Count internal and external connections
                internal_connections = 0
                external_connections = 0
                
                for neighbor in self.graph.neighbors(node):
                    if partition[neighbor] == node_community:
                        internal_connections += 1
                    else:
                        external_connections += 1
                
                # Calculate anomaly score (higher external connections = more anomalous)
                total_connections = internal_connections + external_connections
                if total_connections > 0:
                    external_ratio = external_connections / total_connections
                    modularity_contributions[node] = external_ratio
                else:
                    modularity_contributions[node] = 0.0
            
            return modularity_contributions
            
        except Exception as e:
            logger.warning(f"Community anomaly detection failed: {e}")
            return {}
    
    def detect_temporal_anomalies(self, time_windows: List[Tuple[str, str]]) -> Dict[str, float]:
        """Detect temporal anomalies in transaction patterns."""
        # This would analyze temporal patterns in the graph
        # For now, return empty dict as placeholder
        return {}

class MultiModalAnomalyDetector:
    """Multi-modal anomaly detector combining multiple approaches."""
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Initialize models
        self.models['isolation_forest'] = IsolationForest(
            contamination=contamination, 
            random_state=42, 
            n_jobs=-1
        )
        self.models['one_class_svm'] = OneClassSVM(nu=contamination)
        self.models['autoencoder'] = None  # Will be initialized based on data
        self.models['vae'] = None  # Will be initialized based on data
        
        # Model weights for ensemble
        self.model_weights = {
            'isolation_forest': 0.3,
            'one_class_svm': 0.2,
            'autoencoder': 0.3,
            'vae': 0.2
        }
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for anomaly detection."""
        # Select relevant columns
        feature_columns = [
            'amount', 'hour', 'day_of_week', 'velocity_1h', 'velocity_24h',
            'amount_std_dev', 'location_risk_score'
        ]
        
        # Add categorical features (one-hot encoded)
        categorical_features = pd.get_dummies(df[['transaction_type', 'merchant_category']])
        
        # Combine numerical and categorical features
        numerical_features = df[feature_columns].fillna(0)
        all_features = pd.concat([numerical_features, categorical_features], axis=1)
        
        return all_features.values
    
    def train(self, df: pd.DataFrame, graph: Optional[nx.Graph] = None):
        """Train all anomaly detection models."""
        logger.info("Training multi-modal anomaly detector...")
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        features_standard = self.scalers['standard'].fit_transform(features)
        features_minmax = self.scalers['minmax'].fit_transform(features)
        
        # Train traditional ML models
        logger.info("Training Isolation Forest...")
        self.models['isolation_forest'].fit(features_standard)
        
        logger.info("Training One-Class SVM...")
        self.models['one_class_svm'].fit(features_standard)
        
        # Train deep learning models
        logger.info("Training Autoencoder...")
        self._train_autoencoder(features_minmax)
        
        logger.info("Training VAE...")
        self._train_vae(features_minmax)
        
        # Train graph-based detector if graph is provided
        if graph is not None:
            logger.info("Training Graph Anomaly Detector...")
            self.graph_detector = GraphAnomalyDetector(graph)
            self.graph_detector.calculate_node_features()
        
        self.is_trained = True
        logger.info("Multi-modal anomaly detector training completed")
    
    def _train_autoencoder(self, features: np.ndarray):
        """Train deep autoencoder."""
        input_dim = features.shape[1]
        hidden_dims = [min(128, input_dim), min(64, input_dim//2), min(32, input_dim//4)]
        hidden_dims = [dim for dim in hidden_dims if dim > 0]
        
        self.models['autoencoder'] = DeepAutoencoder(input_dim, hidden_dims)
        
        # Prepare data
        tensor_data = torch.FloatTensor(features)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.models['autoencoder'].parameters(), lr=0.001)
        
        self.models['autoencoder'].train()
        for epoch in range(50):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                reconstructed = self.models['autoencoder'](batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Autoencoder Epoch {epoch}: Loss = {total_loss/len(dataloader):.6f}")
        
        self.models['autoencoder'].eval()
    
    def _train_vae(self, features: np.ndarray):
        """Train Variational Autoencoder."""
        input_dim = features.shape[1]
        latent_dim = min(32, input_dim//4)
        hidden_dims = [min(128, input_dim), min(64, input_dim//2)]
        hidden_dims = [dim for dim in hidden_dims if dim > 0]
        
        self.models['vae'] = VariationalAutoencoder(input_dim, latent_dim, hidden_dims)
        
        # Prepare data
        tensor_data = torch.FloatTensor(features)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        # Training
        optimizer = optim.Adam(self.models['vae'].parameters(), lr=0.001)
        
        def vae_loss(recon_x, x, mu, logvar):
            BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return BCE + KLD
        
        self.models['vae'].train()
        for epoch in range(50):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                recon_batch, mu, logvar = self.models['vae'](batch_x)
                loss = vae_loss(recon_batch, batch_x, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"VAE Epoch {epoch}: Loss = {total_loss/len(dataloader):.6f}")
        
        self.models['vae'].eval()
    
    def predict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict anomalies using all models."""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Prepare features
        features = self.prepare_features(df)
        features_standard = self.scalers['standard'].transform(features)
        features_minmax = self.scalers['minmax'].transform(features)
        
        predictions = {}
        
        # Traditional ML models
        predictions['isolation_forest'] = self.models['isolation_forest'].decision_function(features_standard)
        predictions['one_class_svm'] = self.models['one_class_svm'].decision_function(features_standard)
        
        # Deep learning models
        with torch.no_grad():
            tensor_data = torch.FloatTensor(features_minmax)
            
            # Autoencoder
            reconstructed = self.models['autoencoder'](tensor_data)
            reconstruction_errors = torch.mean((tensor_data - reconstructed) ** 2, dim=1)
            predictions['autoencoder'] = -reconstruction_errors.numpy()  # Negative for consistency
            
            # VAE
            recon_batch, mu, logvar = self.models['vae'](tensor_data)
            vae_errors = torch.mean((tensor_data - recon_batch) ** 2, dim=1)
            predictions['vae'] = -vae_errors.numpy()  # Negative for consistency
        
        return predictions
    
    def predict_ensemble(self, df: pd.DataFrame) -> np.ndarray:
        """Predict anomalies using ensemble of all models."""
        predictions = self.predict(df)
        
        # Normalize predictions to [0, 1]
        normalized_predictions = {}
        for model_name, preds in predictions.items():
            min_val, max_val = preds.min(), preds.max()
            if max_val > min_val:
                normalized_predictions[model_name] = (preds - min_val) / (max_val - min_val)
            else:
                normalized_predictions[model_name] = np.zeros_like(preds)
        
        # Weighted ensemble
        ensemble_scores = np.zeros(len(df))
        for model_name, weight in self.model_weights.items():
            if model_name in normalized_predictions:
                ensemble_scores += weight * normalized_predictions[model_name]
        
        return ensemble_scores
    
    def explain_anomaly(self, df: pd.DataFrame, index: int) -> Dict[str, Any]:
        """Explain why a specific transaction is considered anomalous."""
        if index >= len(df):
            raise ValueError("Index out of range")
        
        transaction = df.iloc[index]
        
        # Get individual model predictions
        predictions = self.predict(df.iloc[[index]])
        
        # Feature importance (simplified)
        features = self.prepare_features(df.iloc[[index]])[0]
        feature_names = list(df.columns)
        
        # Calculate feature deviations from normal
        explanation = {
            'transaction_id': transaction.get('transaction_id', f'idx_{index}'),
            'anomaly_scores': {model: float(preds[0]) for model, preds in predictions.items()},
            'feature_analysis': {},
            'risk_factors': []
        }
        
        # Analyze key features
        if transaction.get('amount', 0) > df['amount'].quantile(0.95):
            explanation['risk_factors'].append('High transaction amount')
        
        if transaction.get('velocity_1h', 0) > df['velocity_1h'].quantile(0.95):
            explanation['risk_factors'].append('High transaction velocity')
        
        if transaction.get('location_risk_score', 0) > 0.7:
            explanation['risk_factors'].append('High location risk')
        
        unusual_time = transaction.get('hour', 12)
        if unusual_time < 6 or unusual_time > 22:
            explanation['risk_factors'].append('Unusual transaction time')
        
        return explanation
    
    def save_models(self, directory: str):
        """Save trained models to directory."""
        os.makedirs(directory, exist_ok=True)
        
        # Save traditional ML models
        joblib.dump(self.models['isolation_forest'], f"{directory}/isolation_forest.joblib")
        joblib.dump(self.models['one_class_svm'], f"{directory}/one_class_svm.joblib")
        joblib.dump(self.scalers, f"{directory}/scalers.joblib")
        
        # Save deep learning models
        if self.models['autoencoder']:
            torch.save(self.models['autoencoder'].state_dict(), f"{directory}/autoencoder.pt")
        if self.models['vae']:
            torch.save(self.models['vae'].state_dict(), f"{directory}/vae.pt")
        
        logger.info(f"Models saved to {directory}")
    
    def load_models(self, directory: str):
        """Load trained models from directory."""
        # Load traditional ML models
        self.models['isolation_forest'] = joblib.load(f"{directory}/isolation_forest.joblib")
        self.models['one_class_svm'] = joblib.load(f"{directory}/one_class_svm.joblib")
        self.scalers = joblib.load(f"{directory}/scalers.joblib")
        
        # Deep learning models would need architecture info to load
        # This is a simplified version
        
        self.is_trained = True
        logger.info(f"Models loaded from {directory}")

def main():
    """Example usage of anomaly detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train anomaly detection models")
    parser.add_argument('--data', type=str, required=True, help='Training data CSV')
    parser.add_argument('--output', type=str, default='models/anomaly', help='Output directory for models')
    parser.add_argument('--contamination', type=float, default=0.1, help='Expected contamination rate')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} transactions")
    
    # Train detector
    detector = MultiModalAnomalyDetector(contamination=args.contamination)
    detector.train(df)
    
    # Make predictions
    anomaly_scores = detector.predict_ensemble(df)
    
    # Find top anomalies
    top_anomalies_idx = np.argsort(anomaly_scores)[-10:]  # Top 10 anomalies
    
    logger.info("Top 10 anomalies:")
    for i, idx in enumerate(top_anomalies_idx):
        score = anomaly_scores[idx]
        explanation = detector.explain_anomaly(df, idx)
        logger.info(f"{i+1}. Index {idx}: Score {score:.3f}, Factors: {explanation['risk_factors']}")
    
    # Save models
    detector.save_models(args.output)
    
    # Save results
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = anomaly_scores > np.percentile(anomaly_scores, 90)  # Top 10%
    df.to_csv(f"{args.output}/results.csv", index=False)
    
    logger.info(f"Results saved to {args.output}/results.csv")

if __name__ == "__main__":
    main()