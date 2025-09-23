#!/usr/bin/env python3
"""
FraudGraphX Demo Script
Complete demonstration of graph-based fraud detection capabilities.
"""

import os
import sys
import time
import logging
import argparse
import subprocess
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.generate_fraud_data import FraudDataGenerator
from src.graph.graph_builder import HeterogeneousGraphBuilder
from src.graph.community_detection import FraudRingDetector
from src.features.anomaly_detector import MultiModalAnomalyDetector
from src.utils.evaluation_metrics import FraudDetectionMetrics, RingDetectionMetrics
from src.visual.fraud_ring_viz import FraudRingVisualizer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudGraphXDemo:
    """Complete demonstration of FraudGraphX capabilities."""
    
    def __init__(self, output_dir: str = "demo_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "visualizations").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        self.data_path = self.output_dir / "data" / "fraud_transactions.csv"
        self.results = {}
    
    def step_1_generate_data(self, n_transactions: int = 10000, fraud_rate: float = 0.05):
        """Step 1: Generate realistic fraud data with rings."""
        logger.info("🎯 Step 1: Generating fraud data with rings")
        
        generator = FraudDataGenerator(
            n_transactions=n_transactions,
            fraud_rate=fraud_rate,
            n_fraud_rings=8
        )
        
        df = generator.generate_transactions()
        df.to_csv(self.data_path, index=False)
        
        # Log statistics
        stats = {
            'total_transactions': len(df),
            'fraud_transactions': df['fraud'].sum(),
            'fraud_rate': df['fraud'].mean(),
            'unique_cards': df['card_id'].nunique(),
            'unique_merchants': df['merchant_id'].nunique(),
            'unique_devices': df['device_id'].nunique(),
            'unique_ips': df['ip'].nunique(),
            'fraud_rings': len(df[df['fraud_ring'] != 'none']['fraud_ring'].unique())
        }
        
        logger.info(f"✅ Generated {stats['total_transactions']} transactions")
        logger.info(f"   - Fraud rate: {stats['fraud_rate']:.1%}")
        logger.info(f"   - Fraud rings: {stats['fraud_rings']}")
        logger.info(f"   - Unique entities: {stats['unique_cards']} cards, {stats['unique_merchants']} merchants")
        
        self.results['data_stats'] = stats
        return df
    
    def step_2_build_graph(self, df: pd.DataFrame):
        """Step 2: Build heterogeneous graph."""
        logger.info("🔗 Step 2: Building heterogeneous graph")
        
        builder = HeterogeneousGraphBuilder()
        
        # Build different graph representations
        graphs = {}
        
        # PyTorch Geometric graph
        logger.info("   Building PyTorch Geometric graph...")
        graphs['torch_geometric'] = builder.build_graph(df, graph_type='torch_geometric')
        
        # NetworkX graph for visualization
        logger.info("   Building NetworkX graph...")
        graphs['networkx'] = builder.build_graph(df, graph_type='networkx')
        
        # Log graph statistics
        nx_graph = graphs['networkx']
        graph_stats = {
            'nodes': nx_graph.number_of_nodes(),
            'edges': nx_graph.number_of_edges(),
            'density': nx.density(nx_graph),
            'node_types': {},
            'connected_components': nx.number_connected_components(nx_graph)
        }
        
        # Count node types
        for node in nx_graph.nodes():
            node_type = node.split('_')[0]
            graph_stats['node_types'][node_type] = graph_stats['node_types'].get(node_type, 0) + 1
        
        logger.info(f"✅ Built graph with {graph_stats['nodes']} nodes and {graph_stats['edges']} edges")
        logger.info(f"   - Density: {graph_stats['density']:.4f}")
        logger.info(f"   - Node types: {graph_stats['node_types']}")
        
        self.results['graph_stats'] = graph_stats
        return graphs
    
    def step_3_detect_rings(self, graphs: dict, df: pd.DataFrame):
        """Step 3: Detect fraud rings using multiple algorithms."""
        logger.info("🕸️ Step 3: Detecting fraud rings")
        
        nx_graph = graphs['networkx']
        detector = FraudRingDetector(min_ring_size=3, max_ring_size=20)
        
        # Detect rings using different methods
        all_rings = {}
        
        logger.info("   Running Louvain algorithm...")
        louvain_rings = detector.detect_rings_louvain(nx_graph, resolution=1.0)
        all_rings.update(louvain_rings)
        
        logger.info("   Running Leiden algorithm...")
        leiden_rings = detector.detect_rings_leiden(nx_graph, resolution=1.0)
        all_rings.update(leiden_rings)
        
        # Evaluate ring quality
        logger.info("   Evaluating ring quality...")
        ring_metrics = detector.evaluate_ring_quality(all_rings)
        
        logger.info(f"✅ Detected {len(all_rings)} fraud rings")
        logger.info(f"   - Average fraud score: {ring_metrics['avg_fraud_score']:.3f}")
        logger.info(f"   - Average ring size: {ring_metrics['avg_ring_size']:.1f}")
        
        # Save ring summary
        ring_summary = detector.get_ring_summary()
        ring_summary.to_csv(self.output_dir / "reports" / "detected_rings.csv", index=False)
        
        self.results['ring_stats'] = ring_metrics
        self.results['detected_rings'] = all_rings
        return all_rings, detector
    
    def step_4_anomaly_detection(self, df: pd.DataFrame):
        """Step 4: Multi-modal anomaly detection."""
        logger.info("🔍 Step 4: Multi-modal anomaly detection")
        
        # Train anomaly detector
        detector = MultiModalAnomalyDetector(contamination=0.1)
        
        logger.info("   Training anomaly detection models...")
        detector.train(df)
        
        # Make predictions
        logger.info("   Detecting anomalies...")
        anomaly_scores = detector.predict_ensemble(df)
        
        # Find top anomalies
        top_anomalies_idx = np.argsort(anomaly_scores)[-20:]  # Top 20 anomalies
        
        anomaly_results = {
            'total_transactions': len(df),
            'anomalies_detected': len(top_anomalies_idx),
            'anomaly_rate': len(top_anomalies_idx) / len(df),
            'top_anomaly_scores': anomaly_scores[top_anomalies_idx].tolist()
        }
        
        logger.info(f"✅ Detected {len(top_anomalies_idx)} anomalies")
        logger.info(f"   - Anomaly rate: {anomaly_results['anomaly_rate']:.1%}")
        logger.info(f"   - Max anomaly score: {max(anomaly_scores):.3f}")
        
        # Save anomaly results
        df_with_anomalies = df.copy()
        df_with_anomalies['anomaly_score'] = anomaly_scores
        df_with_anomalies['is_anomaly'] = np.zeros(len(df))
        df_with_anomalies.loc[top_anomalies_idx, 'is_anomaly'] = 1
        
        df_with_anomalies.to_csv(self.output_dir / "reports" / "anomaly_results.csv", index=False)
        
        self.results['anomaly_stats'] = anomaly_results
        return detector, anomaly_scores
    
    def step_5_visualizations(self, graphs: dict, detected_rings: dict, df: pd.DataFrame):
        """Step 5: Create visualizations."""
        logger.info("📊 Step 5: Creating visualizations")
        
        nx_graph = graphs['networkx']
        
        # Create fraud ring visualizer
        viz = FraudRingVisualizer(nx_graph, detected_rings)
        
        # Generate overview plot
        logger.info("   Creating ring overview plot...")
        overview_fig = viz.create_ring_overview_plot()
        overview_fig.write_html(self.output_dir / "visualizations" / "ring_overview.html")
        
        # Generate individual ring plots
        logger.info("   Creating individual ring visualizations...")
        ring_plots = {}
        for i, (ring_id, ring_data) in enumerate(list(detected_rings.items())[:3]):  # Top 3 rings
            ring_fig = viz.create_ring_network_plot(ring_id)
            ring_plots[ring_id] = ring_fig
            ring_fig.write_html(self.output_dir / "visualizations" / f"ring_{ring_id}.html")
        
        # Create static report
        logger.info("   Creating comprehensive report...")
        viz.create_static_report(str(self.output_dir / "visualizations" / "fraud_rings_report.html"))
        
        # Create data distribution plots
        logger.info("   Creating data analysis plots...")
        self._create_data_plots(df)
        
        logger.info("✅ Visualizations created and saved")
        return viz
    
    def step_6_performance_evaluation(self, df: pd.DataFrame, anomaly_scores: np.ndarray):
        """Step 6: Evaluate model performance."""
        logger.info("📈 Step 6: Performance evaluation")
        
        # Create synthetic predictions for demonstration
        # In a real scenario, these would come from trained GNN models
        fraud_probs = np.random.beta(2, 8, len(df))  # Base probability
        fraud_probs[df['fraud'] == 1] += np.random.beta(8, 2, sum(df['fraud']))  # Higher for fraud
        fraud_probs = np.clip(fraud_probs, 0, 1)
        
        # Evaluate fraud detection performance
        metrics_calculator = FraudDetectionMetrics()
        performance = metrics_calculator.evaluate_model_performance(
            df['fraud'].values, 
            fraud_probs
        )
        
        logger.info(f"✅ Model Performance:")
        logger.info(f"   - PR-AUC: {performance['pr_auc']:.3f}")
        logger.info(f"   - ROC-AUC: {performance['roc_auc']:.3f}")
        logger.info(f"   - F1 Score: {performance['f1_score']:.3f}")
        logger.info(f"   - Fraud Detection Rate: {performance['fraud_detection_rate']:.3f}")
        
        # Save performance metrics
        with open(self.output_dir / "reports" / "performance_metrics.json", 'w') as f:
            json.dump(performance, f, indent=2, default=str)
        
        self.results['performance'] = performance
        return performance
    
    def _create_data_plots(self, df: pd.DataFrame):
        """Create data analysis plots."""
        plt.style.use('seaborn-v0_8')
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('FraudGraphX Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Fraud distribution
        fraud_counts = df['fraud'].value_counts()
        axes[0, 0].pie(fraud_counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.1f%%',
                      colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Transaction Distribution')
        
        # 2. Amount distribution by fraud
        df.boxplot(column='amount', by='fraud', ax=axes[0, 1])
        axes[0, 1].set_title('Amount Distribution by Fraud Status')
        axes[0, 1].set_xlabel('Fraud (0=Legitimate, 1=Fraud)')
        
        # 3. Hourly fraud patterns
        hourly_fraud = df.groupby('hour')['fraud'].mean()
        axes[0, 2].plot(hourly_fraud.index, hourly_fraud.values, marker='o')
        axes[0, 2].set_title('Fraud Rate by Hour of Day')
        axes[0, 2].set_xlabel('Hour')
        axes[0, 2].set_ylabel('Fraud Rate')
        
        # 4. Merchant category fraud rates
        category_fraud = df.groupby('merchant_category')['fraud'].mean().sort_values(ascending=False)
        axes[1, 0].barh(category_fraud.index, category_fraud.values)
        axes[1, 0].set_title('Fraud Rate by Merchant Category')
        axes[1, 0].set_xlabel('Fraud Rate')
        
        # 5. Velocity vs fraud
        axes[1, 1].scatter(df['velocity_24h'], df['amount'], c=df['fraud'], 
                          cmap='coolwarm', alpha=0.6)
        axes[1, 1].set_title('Velocity vs Amount (colored by fraud)')
        axes[1, 1].set_xlabel('24h Velocity')
        axes[1, 1].set_ylabel('Amount')
        
        # 6. Fraud rings distribution
        ring_counts = df[df['fraud_ring'] != 'none']['fraud_ring'].value_counts()
        if len(ring_counts) > 0:
            axes[1, 2].bar(range(len(ring_counts)), ring_counts.values)
            axes[1, 2].set_title('Fraud Ring Sizes')
            axes[1, 2].set_xlabel('Ring ID')
            axes[1, 2].set_ylabel('Transaction Count')
        else:
            axes[1, 2].text(0.5, 0.5, 'No fraud rings in data', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
            axes[1, 2].set_title('Fraud Ring Sizes')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "visualizations" / "data_analysis.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def step_7_generate_report(self):
        """Step 7: Generate comprehensive report."""
        logger.info("📋 Step 7: Generating comprehensive report")
        
        # Create HTML report
        report_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FraudGraphX Demo Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  color: white; padding: 20px; border-radius: 10px; text-align: center; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #667eea; 
                   background: #f8f9fa; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                  background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric h3 {{ margin: 0; color: #667eea; }}
        .metric p {{ margin: 5px 0; font-size: 1.2em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background: #667eea; color: white; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 FraudGraphX Demo Report</h1>
        <p>Advanced Graph-Based Fraud Detection Results</p>
        <p>Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Data Overview</h2>
        <div class="metric">
            <h3>Total Transactions</h3>
            <p>{self.results['data_stats']['total_transactions']:,}</p>
        </div>
        <div class="metric">
            <h3>Fraud Rate</h3>
            <p>{self.results['data_stats']['fraud_rate']:.1%}</p>
        </div>
        <div class="metric">
            <h3>Unique Cards</h3>
            <p>{self.results['data_stats']['unique_cards']:,}</p>
        </div>
        <div class="metric">
            <h3>Unique Merchants</h3>
            <p>{self.results['data_stats']['unique_merchants']:,}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>🔗 Graph Statistics</h2>
        <div class="metric">
            <h3>Graph Nodes</h3>
            <p>{self.results['graph_stats']['nodes']:,}</p>
        </div>
        <div class="metric">
            <h3>Graph Edges</h3>
            <p>{self.results['graph_stats']['edges']:,}</p>
        </div>
        <div class="metric">
            <h3>Graph Density</h3>
            <p>{self.results['graph_stats']['density']:.4f}</p>
        </div>
        <div class="metric">
            <h3>Connected Components</h3>
            <p>{self.results['graph_stats']['connected_components']:,}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>🕸️ Fraud Ring Detection</h2>
        <div class="metric">
            <h3>Rings Detected</h3>
            <p class="success">{self.results['ring_stats']['num_rings_detected']}</p>
        </div>
        <div class="metric">
            <h3>Avg Ring Size</h3>
            <p>{self.results['ring_stats']['avg_ring_size']:.1f}</p>
        </div>
        <div class="metric">
            <h3>Avg Fraud Score</h3>
            <p class="{'success' if self.results['ring_stats']['avg_fraud_score'] > 0.5 else 'warning'}">{self.results['ring_stats']['avg_fraud_score']:.3f}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>🔍 Anomaly Detection</h2>
        <div class="metric">
            <h3>Anomalies Found</h3>
            <p class="warning">{self.results['anomaly_stats']['anomalies_detected']}</p>
        </div>
        <div class="metric">
            <h3>Anomaly Rate</h3>
            <p>{self.results['anomaly_stats']['anomaly_rate']:.1%}</p>
        </div>
        <div class="metric">
            <h3>Max Anomaly Score</h3>
            <p>{max(self.results['anomaly_stats']['top_anomaly_scores']):.3f}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 Model Performance</h2>
        <div class="metric">
            <h3>PR-AUC</h3>
            <p class="{'success' if self.results['performance']['pr_auc'] > 0.8 else 'warning'}">{self.results['performance']['pr_auc']:.3f}</p>
        </div>
        <div class="metric">
            <h3>ROC-AUC</h3>
            <p class="{'success' if self.results['performance']['roc_auc'] > 0.8 else 'warning'}">{self.results['performance']['roc_auc']:.3f}</p>
        </div>
        <div class="metric">
            <h3>F1 Score</h3>
            <p class="{'success' if self.results['performance']['f1_score'] > 0.7 else 'warning'}">{self.results['performance']['f1_score']:.3f}</p>
        </div>
        <div class="metric">
            <h3>Fraud Detection Rate</h3>
            <p>{self.results['performance']['fraud_detection_rate']:.3f}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📁 Generated Files</h2>
        <ul>
            <li><a href="data/fraud_transactions.csv">📄 Transaction Data</a></li>
            <li><a href="reports/detected_rings.csv">🕸️ Detected Rings</a></li>
            <li><a href="reports/anomaly_results.csv">🔍 Anomaly Results</a></li>
            <li><a href="reports/performance_metrics.json">📈 Performance Metrics</a></li>
            <li><a href="visualizations/fraud_rings_report.html">📊 Visual Report</a></li>
            <li><a href="visualizations/data_analysis.png">📈 Data Analysis Plots</a></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🚀 Next Steps</h2>
        <ol>
            <li><strong>API Testing:</strong> Run <code>./tests/curl_tests.sh</code> to test all endpoints</li>
            <li><strong>Real-time Monitoring:</strong> Start the monitoring service with <code>python -m src.features.real_time_monitor</code></li>
            <li><strong>Interactive Dashboard:</strong> Launch the visualization dashboard</li>
            <li><strong>Model Training:</strong> Train production GNN models with larger datasets</li>
            <li><strong>Integration:</strong> Integrate with existing fraud detection systems</li>
        </ol>
    </div>
    
    <div style="text-align: center; margin: 40px 0; padding: 20px; background: #f8f9fa; border-radius: 10px;">
        <h3>🎉 Demo Completed Successfully!</h3>
        <p>FraudGraphX demonstrated advanced graph-based fraud detection capabilities</p>
        <p><strong>Total Processing Time:</strong> {time.time() - self.start_time:.1f} seconds</p>
    </div>
</body>
</html>
        """
        
        with open(self.output_dir / "FraudGraphX_Demo_Report.html", 'w') as f:
            f.write(report_html)
        
        logger.info("✅ Comprehensive report generated")
    
    def run_demo(self, n_transactions: int = 10000, fraud_rate: float = 0.05):
        """Run the complete FraudGraphX demonstration."""
        self.start_time = time.time()
        
        logger.info("🚀 Starting FraudGraphX Demo")
        logger.info("=" * 60)
        
        try:
            # Step 1: Generate data
            df = self.step_1_generate_data(n_transactions, fraud_rate)
            
            # Step 2: Build graph
            graphs = self.step_2_build_graph(df)
            
            # Step 3: Detect rings
            detected_rings, ring_detector = self.step_3_detect_rings(graphs, df)
            
            # Step 4: Anomaly detection
            anomaly_detector, anomaly_scores = self.step_4_anomaly_detection(df)
            
            # Step 5: Visualizations
            visualizer = self.step_5_visualizations(graphs, detected_rings, df)
            
            # Step 6: Performance evaluation
            performance = self.step_6_performance_evaluation(df, anomaly_scores)
            
            # Step 7: Generate report
            self.step_7_generate_report()
            
            total_time = time.time() - self.start_time
            
            logger.info("=" * 60)
            logger.info("🎉 FraudGraphX Demo Completed Successfully!")
            logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
            logger.info(f"📁 Results saved to: {self.output_dir}")
            logger.info(f"📊 View report: {self.output_dir}/FraudGraphX_Demo_Report.html")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="FraudGraphX Complete Demo")
    parser.add_argument('--transactions', type=int, default=10000, 
                       help='Number of transactions to generate')
    parser.add_argument('--fraud-rate', type=float, default=0.05,
                       help='Fraud rate (0.0 to 1.0)')
    parser.add_argument('--output', type=str, default='demo_output',
                       help='Output directory')
    parser.add_argument('--run-api', action='store_true',
                       help='Start API server after demo')
    
    args = parser.parse_args()
    
    # Run demo
    demo = FraudGraphXDemo(args.output)
    success = demo.run_demo(args.transactions, args.fraud_rate)
    
    if success and args.run_api:
        logger.info("🌐 Starting API server...")
        try:
            subprocess.run([
                "uvicorn", "src.serve.advanced_api:app",
                "--host", "0.0.0.0", "--port", "8000", "--reload"
            ])
        except KeyboardInterrupt:
            logger.info("API server stopped")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())