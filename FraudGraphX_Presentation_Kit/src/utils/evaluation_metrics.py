"""
Advanced evaluation metrics for fraud detection and ring detection.
Includes PR-AUC, cluster purity, and specialized fraud detection metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, auc, roc_curve, roc_auc_score,
    confusion_matrix, classification_report, average_precision_score
)
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from typing import Dict, List, Tuple, Optional, Any
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionMetrics:
    """Comprehensive evaluation metrics for fraud detection systems."""
    
    def __init__(self):
        self.metrics_history = []
        self.threshold_metrics = {}
    
    def calculate_pr_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate Precision-Recall AUC and related metrics."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Average Precision Score (alternative calculation)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Find optimal threshold based on F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return {
            'pr_auc': pr_auc,
            'average_precision': avg_precision,
            'optimal_threshold': optimal_threshold,
            'optimal_precision': precision[optimal_idx],
            'optimal_recall': recall[optimal_idx],
            'optimal_f1': f1_scores[optimal_idx],
            'precision_curve': precision.tolist(),
            'recall_curve': recall.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def calculate_roc_auc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate ROC AUC and related metrics."""
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = roc_auc_score(y_true, y_scores)
        
        # Find optimal threshold using Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'roc_auc': roc_auc,
            'optimal_threshold_youden': optimal_threshold,
            'optimal_tpr': tpr[optimal_idx],
            'optimal_fpr': fpr[optimal_idx],
            'fpr_curve': fpr.tolist(),
            'tpr_curve': tpr.tolist(),
            'thresholds': thresholds.tolist()
        }
    
    def calculate_threshold_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, 
                                  thresholds: Optional[List[float]] = None) -> Dict[float, Dict]:
        """Calculate metrics at different thresholds."""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 9)
        
        threshold_results = {}
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # False positive rate and false negative rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            # Lift and other business metrics
            fraud_rate = np.mean(y_true)
            predicted_fraud_rate = np.mean(y_pred)
            lift = (precision / fraud_rate) if fraud_rate > 0 else 0
            
            threshold_results[threshold] = {
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'accuracy': accuracy,
                'fpr': fpr,
                'fnr': fnr,
                'lift': lift,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'predicted_fraud_rate': predicted_fraud_rate
            }
        
        self.threshold_metrics = threshold_results
        return threshold_results
    
    def calculate_cost_based_metrics(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   cost_fp: float = 1.0, cost_fn: float = 10.0,
                                   cost_investigation: float = 5.0) -> Dict[str, Any]:
        """Calculate cost-based evaluation metrics for fraud detection."""
        thresholds = np.linspace(0.01, 0.99, 100)
        costs = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Calculate total cost
            investigation_cost = (tp + fp) * cost_investigation  # Cost to investigate flagged cases
            false_positive_cost = fp * cost_fp  # Cost of false alarms
            false_negative_cost = fn * cost_fn  # Cost of missed fraud
            
            total_cost = investigation_cost + false_positive_cost + false_negative_cost
            costs.append(total_cost)
        
        # Find optimal threshold that minimizes cost
        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds[optimal_idx]
        min_cost = costs[optimal_idx]
        
        # Calculate savings compared to no model (all fraud goes undetected)
        baseline_cost = len(y_true) * np.mean(y_true) * cost_fn
        cost_savings = baseline_cost - min_cost
        
        return {
            'optimal_threshold_cost': optimal_threshold,
            'min_cost': min_cost,
            'baseline_cost': baseline_cost,
            'cost_savings': cost_savings,
            'cost_reduction_pct': (cost_savings / baseline_cost) * 100 if baseline_cost > 0 else 0,
            'costs_by_threshold': list(zip(thresholds.tolist(), costs))
        }
    
    def evaluate_model_performance(self, y_true: np.ndarray, y_scores: np.ndarray,
                                 y_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive model performance evaluation."""
        
        # If predictions not provided, use 0.5 threshold
        if y_pred is None:
            y_pred = (y_scores >= 0.5).astype(int)
        
        # Basic metrics
        pr_metrics = self.calculate_pr_auc(y_true, y_scores)
        roc_metrics = self.calculate_roc_auc(y_true, y_scores)
        threshold_metrics = self.calculate_threshold_metrics(y_true, y_scores)
        cost_metrics = self.calculate_cost_based_metrics(y_true, y_scores)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional fraud-specific metrics
        fraud_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Same as recall
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        fraud_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Volume metrics
        total_flagged = tp + fp
        total_fraud = tp + fn
        total_legitimate = tn + fp
        
        results = {
            'pr_auc': pr_metrics['pr_auc'],
            'roc_auc': roc_metrics['roc_auc'],
            'fraud_detection_rate': fraud_detection_rate,
            'false_positive_rate': false_positive_rate,
            'fraud_precision': fraud_precision,
            'f1_score': class_report['1']['f1-score'],
            'accuracy': class_report['accuracy'],
            'total_transactions': len(y_true),
            'total_fraud': int(total_fraud),
            'total_legitimate': int(total_legitimate),
            'total_flagged': int(total_flagged),
            'confusion_matrix': cm.tolist(),
            'detailed_pr_metrics': pr_metrics,
            'detailed_roc_metrics': roc_metrics,
            'threshold_analysis': threshold_metrics,
            'cost_analysis': cost_metrics,
            'classification_report': class_report
        }
        
        return results

class RingDetectionMetrics:
    """Metrics for evaluating fraud ring detection quality."""
    
    def __init__(self):
        self.ring_quality_scores = {}
    
    def calculate_cluster_purity(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate cluster purity score."""
        if len(cluster_labels) != len(true_labels):
            raise ValueError("Cluster labels and true labels must have same length")
        
        total_samples = len(cluster_labels)
        if total_samples == 0:
            return 0.0
        
        # Calculate purity for each cluster
        unique_clusters = np.unique(cluster_labels)
        weighted_purity = 0.0
        
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size == 0:
                continue
            
            # Find most common true label in this cluster
            cluster_true_labels = true_labels[cluster_mask]
            most_common_label = Counter(cluster_true_labels).most_common(1)[0][1]
            
            cluster_purity = most_common_label / cluster_size
            weighted_purity += (cluster_size / total_samples) * cluster_purity
        
        return weighted_purity
    
    def calculate_silhouette_score(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality."""
        from sklearn.metrics import silhouette_score
        
        if len(np.unique(cluster_labels)) < 2:
            return 0.0
        
        return silhouette_score(embeddings, cluster_labels)
    
    def calculate_ring_homogeneity(self, detected_rings: Dict[str, Dict], 
                                  ground_truth_labels: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """Calculate homogeneity scores for detected rings."""
        ring_homogeneity = {}
        
        for ring_id, ring_data in detected_rings.items():
            nodes = ring_data['nodes']
            
            if ground_truth_labels:
                # Use ground truth labels if available
                ring_labels = [ground_truth_labels.get(node, 0) for node in nodes]
                fraud_count = sum(ring_labels)
            else:
                # Use heuristic based on node names
                fraud_count = sum(1 for node in nodes if 'fraud' in node.lower())
            
            homogeneity = fraud_count / len(nodes) if len(nodes) > 0 else 0
            ring_homogeneity[ring_id] = homogeneity
        
        return ring_homogeneity
    
    def calculate_ring_connectivity(self, graph, detected_rings: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate internal connectivity of detected rings."""
        import networkx as nx
        
        ring_connectivity = {}
        
        for ring_id, ring_data in detected_rings.items():
            nodes = ring_data['nodes']
            
            if len(nodes) < 2:
                ring_connectivity[ring_id] = 0.0
                continue
            
            # Create subgraph for the ring
            subgraph = graph.subgraph(nodes)
            
            # Calculate density (actual edges / possible edges)
            n_nodes = len(nodes)
            possible_edges = n_nodes * (n_nodes - 1) / 2
            actual_edges = subgraph.number_of_edges()
            
            density = actual_edges / possible_edges if possible_edges > 0 else 0
            ring_connectivity[ring_id] = density
        
        return ring_connectivity
    
    def evaluate_ring_detection(self, detected_rings: Dict[str, Dict], 
                               graph, embeddings: Optional[np.ndarray] = None,
                               ground_truth_rings: Optional[Dict] = None) -> Dict[str, Any]:
        """Comprehensive evaluation of ring detection results."""
        
        results = {
            'num_rings_detected': len(detected_rings),
            'ring_sizes': [ring['size'] for ring in detected_rings.values()],
            'avg_ring_size': np.mean([ring['size'] for ring in detected_rings.values()]) if detected_rings else 0,
            'fraud_scores': [ring['fraud_score'] for ring in detected_rings.values()],
            'avg_fraud_score': np.mean([ring['fraud_score'] for ring in detected_rings.values()]) if detected_rings else 0
        }
        
        # Calculate homogeneity
        homogeneity_scores = self.calculate_ring_homogeneity(detected_rings)
        results['ring_homogeneity'] = homogeneity_scores
        results['avg_homogeneity'] = np.mean(list(homogeneity_scores.values())) if homogeneity_scores else 0
        
        # Calculate connectivity
        if graph:
            connectivity_scores = self.calculate_ring_connectivity(graph, detected_rings)
            results['ring_connectivity'] = connectivity_scores
            results['avg_connectivity'] = np.mean(list(connectivity_scores.values())) if connectivity_scores else 0
        
        # Calculate clustering metrics if embeddings are provided
        if embeddings is not None and detected_rings:
            # Create cluster labels for all nodes
            all_nodes = []
            cluster_labels = []
            
            for i, (ring_id, ring_data) in enumerate(detected_rings.items()):
                for node in ring_data['nodes']:
                    all_nodes.append(node)
                    cluster_labels.append(i)
            
            if len(cluster_labels) > 1:
                # Get embeddings for ring nodes (simplified)
                node_embeddings = embeddings[:len(all_nodes)]  # Simplified mapping
                
                if len(np.unique(cluster_labels)) > 1:
                    silhouette = self.calculate_silhouette_score(node_embeddings, np.array(cluster_labels))
                    results['silhouette_score'] = silhouette
        
        # External evaluation against ground truth
        if ground_truth_rings:
            external_metrics = self._calculate_external_ring_metrics(detected_rings, ground_truth_rings)
            results.update(external_metrics)
        
        return results
    
    def _calculate_external_ring_metrics(self, detected_rings: Dict, ground_truth_rings: Dict) -> Dict:
        """Calculate precision, recall, F1 for ring detection against ground truth."""
        
        # Convert rings to sets of nodes for comparison
        detected_node_sets = [set(ring['nodes']) for ring in detected_rings.values()]
        ground_truth_node_sets = [set(ring['nodes']) for ring in ground_truth_rings.values()]
        
        # Calculate overlap-based precision and recall
        true_positives = 0
        for gt_ring in ground_truth_node_sets:
            # Check if any detected ring has significant overlap with ground truth ring
            best_overlap = 0
            for det_ring in detected_node_sets:
                overlap = len(gt_ring.intersection(det_ring))
                union = len(gt_ring.union(det_ring))
                jaccard = overlap / union if union > 0 else 0
                best_overlap = max(best_overlap, jaccard)
            
            if best_overlap > 0.5:  # Threshold for considering a match
                true_positives += 1
        
        precision = true_positives / len(detected_rings) if detected_rings else 0
        recall = true_positives / len(ground_truth_rings) if ground_truth_rings else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'ring_detection_precision': precision,
            'ring_detection_recall': recall,
            'ring_detection_f1': f1,
            'ground_truth_rings': len(ground_truth_rings),
            'detected_rings_matched': true_positives
        }

class MetricsVisualizer:
    """Create visualizations for evaluation metrics."""
    
    @staticmethod
    def plot_pr_curve(pr_metrics: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision = pr_metrics['precision_curve']
        recall = pr_metrics['recall_curve']
        pr_auc = pr_metrics['pr_auc']
        
        ax.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax.fill_between(recall, precision, alpha=0.3)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_roc_curve(roc_metrics: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr = roc_metrics['fpr_curve']
        tpr = roc_metrics['tpr_curve']
        roc_auc = roc_metrics['roc_auc']
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.3)
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_threshold_analysis(threshold_metrics: Dict, save_path: Optional[str] = None) -> plt.Figure:
        """Plot metrics vs threshold."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        thresholds = list(threshold_metrics.keys())
        precisions = [m['precision'] for m in threshold_metrics.values()]
        recalls = [m['recall'] for m in threshold_metrics.values()]
        f1_scores = [m['f1_score'] for m in threshold_metrics.values()]
        fprs = [m['fpr'] for m in threshold_metrics.values()]
        
        # Precision vs Threshold
        ax1.plot(thresholds, precisions, 'b-', linewidth=2)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Threshold')
        ax1.grid(True, alpha=0.3)
        
        # Recall vs Threshold
        ax2.plot(thresholds, recalls, 'r-', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Recall')
        ax2.set_title('Recall vs Threshold')
        ax2.grid(True, alpha=0.3)
        
        # F1 Score vs Threshold
        ax3.plot(thresholds, f1_scores, 'g-', linewidth=2)
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Score vs Threshold')
        ax3.grid(True, alpha=0.3)
        
        # FPR vs Threshold
        ax4.plot(thresholds, fprs, 'm-', linewidth=2)
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('False Positive Rate')
        ax4.set_title('FPR vs Threshold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

def main():
    """Example usage of evaluation metrics."""
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    y_true = np.random.binomial(1, 0.1, n_samples)  # 10% fraud rate
    y_scores = np.random.beta(2, 8, n_samples)  # Skewed scores
    y_scores[y_true == 1] += np.random.beta(8, 2, np.sum(y_true))  # Higher scores for fraud
    
    # Evaluate fraud detection
    fraud_metrics = FraudDetectionMetrics()
    results = fraud_metrics.evaluate_model_performance(y_true, y_scores)
    
    print("Fraud Detection Results:")
    print(f"PR-AUC: {results['pr_auc']:.3f}")
    print(f"ROC-AUC: {results['roc_auc']:.3f}")
    print(f"F1 Score: {results['f1_score']:.3f}")
    print(f"Fraud Detection Rate: {results['fraud_detection_rate']:.3f}")
    print(f"False Positive Rate: {results['false_positive_rate']:.3f}")
    
    # Create visualizations
    viz = MetricsVisualizer()
    viz.plot_pr_curve(results['detailed_pr_metrics'], 'pr_curve.png')
    viz.plot_roc_curve(results['detailed_roc_metrics'], 'roc_curve.png')
    viz.plot_threshold_analysis(results['threshold_analysis'], 'threshold_analysis.png')

if __name__ == "__main__":
    main()