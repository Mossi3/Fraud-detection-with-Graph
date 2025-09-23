"""
Comprehensive Evaluation Metrics for Graph-based Fraud Detection
Implements PR-AUC, cluster purity, and other advanced metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class FraudDetectionEvaluator:
    """
    Comprehensive evaluator for fraud detection models
    """
    
    def __init__(self):
        self.metrics = {}
        self.predictions = None
        self.labels = None
        self.thresholds = None
        
    def evaluate_binary_classification(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate binary classification performance
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        metrics = {}
        
        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['tn'] = int(cm[0, 0])
        metrics['fp'] = int(cm[0, 1])
        metrics['fn'] = int(cm[1, 0])
        metrics['tp'] = int(cm[1, 1])
        
        # Additional metrics
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # F-beta scores
        metrics['f2_score'] = f1_score(y_true, y_pred, beta=2, zero_division=0)
        metrics['f0_5_score'] = f1_score(y_true, y_pred, beta=0.5, zero_division=0)
        
        # Probability-based metrics
        if y_prob is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            
            # Precision-Recall curve metrics
            precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
            
            # Find optimal threshold (maximizing F1)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            metrics['optimal_threshold'] = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            metrics['optimal_f1'] = f1_scores[optimal_idx]
            
            # Store for plotting
            self.predictions = y_pred
            self.labels = y_true
            self.thresholds = thresholds
        
        return metrics
    
    def evaluate_graph_based_detection(self, graph: Any, communities: Dict[int, List],
                                     fraud_labels: Dict[str, int],
                                     predictions: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Evaluate graph-based fraud detection performance
        
        Args:
            graph: NetworkX graph
            communities: Detected communities
            fraud_labels: Ground truth fraud labels
            predictions: Model predictions (optional)
            
        Returns:
            Dictionary of graph-based evaluation metrics
        """
        
        metrics = {}
        
        # Community-level metrics
        community_metrics = self._evaluate_communities(communities, fraud_labels)
        metrics.update(community_metrics)
        
        # Graph structure metrics
        graph_metrics = self._evaluate_graph_structure(graph, fraud_labels)
        metrics.update(graph_metrics)
        
        # Node-level metrics
        if predictions:
            node_metrics = self._evaluate_node_predictions(predictions, fraud_labels)
            metrics.update(node_metrics)
        
        return metrics
    
    def _evaluate_communities(self, communities: Dict[int, List],
                            fraud_labels: Dict[str, int]) -> Dict[str, float]:
        """Evaluate community detection quality"""
        
        metrics = {}
        
        # Community purity
        purities = []
        fraud_rates = []
        sizes = []
        
        for comm_id, nodes in communities.items():
            if len(nodes) < 2:  # Skip single-node communities
                continue
            
            # Calculate fraud rate in community
            fraud_count = sum(fraud_labels.get(node, 0) for node in nodes)
            fraud_rate = fraud_count / len(nodes)
            fraud_rates.append(fraud_rate)
            sizes.append(len(nodes))
            
            # Calculate purity (majority class ratio)
            fraud_nodes = [node for node in nodes if fraud_labels.get(node, 0) == 1]
            legitimate_nodes = [node for node in nodes if fraud_labels.get(node, 0) == 0]
            
            if len(fraud_nodes) > len(legitimate_nodes):
                purity = len(fraud_nodes) / len(nodes)
            else:
                purity = len(legitimate_nodes) / len(nodes)
            
            purities.append(purity)
        
        metrics['avg_community_purity'] = np.mean(purities) if purities else 0
        metrics['avg_fraud_rate'] = np.mean(fraud_rates) if fraud_rates else 0
        metrics['avg_community_size'] = np.mean(sizes) if sizes else 0
        
        # Community coverage
        all_nodes = set()
        for nodes in communities.values():
            all_nodes.update(nodes)
        
        total_fraud_nodes = sum(1 for label in fraud_labels.values() if label == 1)
        covered_fraud_nodes = sum(1 for node in all_nodes if fraud_labels.get(node, 0) == 1)
        
        metrics['fraud_coverage'] = covered_fraud_nodes / total_fraud_nodes if total_fraud_nodes > 0 else 0
        metrics['total_coverage'] = len(all_nodes) / len(fraud_labels) if fraud_labels else 0
        
        return metrics
    
    def _evaluate_graph_structure(self, graph: Any, fraud_labels: Dict[str, int]) -> Dict[str, float]:
        """Evaluate graph structure metrics"""
        
        metrics = {}
        
        # Calculate centrality measures for fraud vs legitimate nodes
        fraud_nodes = [node for node, label in fraud_labels.items() if label == 1]
        legitimate_nodes = [node for node, label in fraud_labels.items() if label == 0]
        
        if fraud_nodes and legitimate_nodes:
            # Degree centrality
            fraud_degrees = [graph.degree(node) for node in fraud_nodes if node in graph]
            legitimate_degrees = [graph.degree(node) for node in legitimate_nodes if node in graph]
            
            if fraud_degrees and legitimate_degrees:
                metrics['avg_fraud_degree'] = np.mean(fraud_degrees)
                metrics['avg_legitimate_degree'] = np.mean(legitimate_degrees)
                metrics['degree_separation'] = abs(metrics['avg_fraud_degree'] - metrics['avg_legitimate_degree'])
            
            # Clustering coefficient
            fraud_clustering = []
            legitimate_clustering = []
            
            for node in fraud_nodes:
                if node in graph:
                    clustering = nx.clustering(graph, node)
                    fraud_clustering.append(clustering)
            
            for node in legitimate_nodes:
                if node in graph:
                    clustering = nx.clustering(graph, node)
                    legitimate_clustering.append(clustering)
            
            if fraud_clustering and legitimate_clustering:
                metrics['avg_fraud_clustering'] = np.mean(fraud_clustering)
                metrics['avg_legitimate_clustering'] = np.mean(legitimate_clustering)
                metrics['clustering_separation'] = abs(metrics['avg_fraud_clustering'] - metrics['avg_legitimate_clustering'])
        
        return metrics
    
    def _evaluate_node_predictions(self, predictions: Dict[str, float],
                                fraud_labels: Dict[str, int]) -> Dict[str, float]:
        """Evaluate node-level predictions"""
        
        # Convert to arrays
        y_true = []
        y_pred = []
        
        for node in fraud_labels:
            if node in predictions:
                y_true.append(fraud_labels[node])
                y_pred.append(predictions[node])
        
        if not y_true:
            return {}
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Binary predictions using 0.5 threshold
        y_pred_binary = (y_pred >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = self.evaluate_binary_classification(y_true, y_pred_binary, y_pred)
        
        return metrics
    
    def evaluate_fraud_rings(self, detected_rings: Dict[str, Dict],
                           ground_truth_rings: Dict[str, List]) -> Dict[str, float]:
        """
        Evaluate fraud ring detection performance
        
        Args:
            detected_rings: Detected fraud rings
            ground_truth_rings: Ground truth fraud rings
            
        Returns:
            Dictionary of ring detection metrics
        """
        
        metrics = {}
        
        # Convert ground truth to sets for easier comparison
        gt_ring_sets = {}
        for ring_id, nodes in ground_truth_rings.items():
            gt_ring_sets[ring_id] = set(nodes)
        
        # Evaluate each detection method
        for method, rings in detected_rings.items():
            method_metrics = self._evaluate_ring_method(rings, gt_ring_sets)
            for key, value in method_metrics.items():
                metrics[f"{method}_{key}"] = value
        
        return metrics
    
    def _evaluate_ring_method(self, detected_rings: Dict[int, Dict],
                            gt_ring_sets: Dict[str, set]) -> Dict[str, float]:
        """Evaluate a single ring detection method"""
        
        metrics = {}
        
        # Precision: How many detected rings are correct?
        correct_rings = 0
        total_detected = len(detected_rings)
        
        for ring_id, ring_data in detected_rings.items():
            detected_nodes = set(ring_data['nodes'])
            
            # Check if this ring matches any ground truth ring
            best_overlap = 0
            for gt_nodes in gt_ring_sets.values():
                overlap = len(detected_nodes & gt_nodes) / len(detected_nodes | gt_nodes)
                best_overlap = max(best_overlap, overlap)
            
            if best_overlap > 0.5:  # Threshold for considering a ring "correct"
                correct_rings += 1
        
        metrics['ring_precision'] = correct_rings / total_detected if total_detected > 0 else 0
        
        # Recall: How many ground truth rings were detected?
        detected_gt_rings = 0
        total_gt_rings = len(gt_ring_sets)
        
        for gt_nodes in gt_ring_sets.values():
            best_overlap = 0
            for ring_data in detected_rings.values():
                detected_nodes = set(ring_data['nodes'])
                overlap = len(detected_nodes & gt_nodes) / len(detected_nodes | gt_nodes)
                best_overlap = max(best_overlap, overlap)
            
            if best_overlap > 0.5:
                detected_gt_rings += 1
        
        metrics['ring_recall'] = detected_gt_rings / total_gt_rings if total_gt_rings > 0 else 0
        
        # F1 score
        precision = metrics['ring_precision']
        recall = metrics['ring_recall']
        metrics['ring_f1'] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return metrics
    
    def calculate_cluster_purity(self, true_labels: np.ndarray,
                               cluster_labels: np.ndarray) -> float:
        """
        Calculate cluster purity
        
        Args:
            true_labels: True class labels
            cluster_labels: Cluster assignments
            
        Returns:
            Cluster purity score
        """
        
        total_purity = 0
        total_points = len(true_labels)
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                # Find most common class in cluster
                most_common_count = Counter(cluster_true_labels).most_common(1)[0][1]
                cluster_purity = most_common_count / len(cluster_true_labels)
                total_purity += cluster_purity * len(cluster_true_labels)
        
        return total_purity / total_points
    
    def plot_evaluation_results(self, save_path: str = None) -> None:
        """Plot comprehensive evaluation results"""
        
        if self.predictions is None or self.labels is None:
            logger.warning("No evaluation data available for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(self.labels, self.predictions)
        axes[0, 0].plot(recall, precision, linewidth=2)
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision-Recall Curve')
        axes[0, 0].grid(True)
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.labels, self.predictions)
        axes[0, 1].plot(fpr, tpr, linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].grid(True)
        
        # Confusion Matrix
        y_pred_binary = (self.predictions >= 0.5).astype(int)
        cm = confusion_matrix(self.labels, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
        
        # Score Distribution
        axes[1, 1].hist(self.predictions[self.labels == 0], bins=50, alpha=0.7, label='Legitimate', density=True)
        axes[1, 1].hist(self.predictions[self.labels == 1], bins=50, alpha=0.7, label='Fraud', density=True)
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[1, 1].set_xlabel('Prediction Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Score Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = []
        report.append("=" * 60)
        report.append("FRAUD DETECTION EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Classification Metrics
        if any(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1_score']):
            report.append("CLASSIFICATION METRICS:")
            report.append("-" * 30)
            report.append(f"Accuracy:        {metrics.get('accuracy', 0):.4f}")
            report.append(f"Precision:       {metrics.get('precision', 0):.4f}")
            report.append(f"Recall:          {metrics.get('recall', 0):.4f}")
            report.append(f"F1-Score:        {metrics.get('f1_score', 0):.4f}")
            report.append(f"ROC-AUC:         {metrics.get('roc_auc', 0):.4f}")
            report.append(f"PR-AUC:          {metrics.get('pr_auc', 0):.4f}")
            report.append("")
        
        # Community Metrics
        if any(key in metrics for key in ['avg_community_purity', 'fraud_coverage']):
            report.append("COMMUNITY DETECTION METRICS:")
            report.append("-" * 30)
            report.append(f"Avg Community Purity: {metrics.get('avg_community_purity', 0):.4f}")
            report.append(f"Fraud Coverage:      {metrics.get('fraud_coverage', 0):.4f}")
            report.append(f"Total Coverage:       {metrics.get('total_coverage', 0):.4f}")
            report.append(f"Avg Community Size:   {metrics.get('avg_community_size', 0):.2f}")
            report.append("")
        
        # Ring Detection Metrics
        ring_methods = [key for key in metrics.keys() if key.endswith('_ring_precision')]
        if ring_methods:
            report.append("FRAUD RING DETECTION METRICS:")
            report.append("-" * 30)
            for method in ring_methods:
                method_name = method.replace('_ring_precision', '')
                precision = metrics.get(f"{method_name}_ring_precision", 0)
                recall = metrics.get(f"{method_name}_ring_recall", 0)
                f1 = metrics.get(f"{method_name}_ring_f1", 0)
                report.append(f"{method_name.title()} Method:")
                report.append(f"  Precision: {precision:.4f}")
                report.append(f"  Recall:    {recall:.4f}")
                report.append(f"  F1-Score:  {f1:.4f}")
            report.append("")
        
        # Graph Structure Metrics
        if any(key in metrics for key in ['degree_separation', 'clustering_separation']):
            report.append("GRAPH STRUCTURE METRICS:")
            report.append("-" * 30)
            report.append(f"Degree Separation:     {metrics.get('degree_separation', 0):.4f}")
            report.append(f"Clustering Separation: {metrics.get('clustering_separation', 0):.4f}")
            report.append(f"Avg Fraud Degree:      {metrics.get('avg_fraud_degree', 0):.2f}")
            report.append(f"Avg Legitimate Degree: {metrics.get('avg_legitimate_degree', 0):.2f}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 30)
        
        # Overall performance score
        performance_metrics = ['accuracy', 'f1_score', 'roc_auc', 'pr_auc']
        available_metrics = [metrics.get(metric, 0) for metric in performance_metrics if metric in metrics]
        
        if available_metrics:
            overall_score = np.mean(available_metrics)
            report.append(f"Overall Performance Score: {overall_score:.4f}")
        
        # Recommendations
        report.append("")
        report.append("RECOMMENDATIONS:")
        report.append("-" * 30)
        
        if metrics.get('precision', 0) < 0.7:
            report.append("- Consider adjusting threshold to improve precision")
        
        if metrics.get('recall', 0) < 0.7:
            report.append("- Consider feature engineering to improve recall")
        
        if metrics.get('avg_community_purity', 0) < 0.8:
            report.append("- Community detection may need parameter tuning")
        
        if metrics.get('fraud_coverage', 0) < 0.8:
            report.append("- Consider increasing community detection sensitivity")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)

def evaluate_fraud_detection_pipeline(df: pd.DataFrame, 
                                   graph: Any,
                                   communities: Dict[int, List],
                                   predictions: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Complete evaluation pipeline for fraud detection
    
    Args:
        df: Transaction dataframe
        graph: NetworkX graph
        communities: Detected communities
        predictions: Model predictions
        
    Returns:
        Comprehensive evaluation metrics
    """
    
    evaluator = FraudDetectionEvaluator()
    
    # Create fraud labels
    fraud_labels = {}
    if 'fraud' in df.columns and 'card_id' in df.columns:
        fraud_labels = dict(zip(df['card_id'], df['fraud']))
    
    # Evaluate graph-based detection
    metrics = evaluator.evaluate_graph_based_detection(
        graph, communities, fraud_labels, predictions
    )
    
    # Evaluate node-level predictions if available
    if predictions:
        node_metrics = evaluator._evaluate_node_predictions(predictions, fraud_labels)
        metrics.update(node_metrics)
    
    return metrics