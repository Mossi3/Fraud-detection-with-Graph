"""
Fraud Detection Visualization Module
Creates heatmaps, network visualizations, and fraud ring discovery plots.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
import warnings
warnings.filterwarnings('ignore')

class FraudVisualizer:
    """Comprehensive visualization toolkit for fraud detection analysis"""
    
    def __init__(self, output_dir: str = '/workspace/fraud_detection_graph/visualizations/'):
        self.output_dir = output_dir
        self.color_palette = px.colors.qualitative.Set3
        
        # Ensure output directory exists
        import os
        os.makedirs(output_dir, exist_ok=True)
        
    def load_data(self, data_path: str = '/workspace/fraud_detection_graph/data/'):
        """Load all necessary data for visualization"""
        # Load transaction data
        self.transactions = pd.read_csv(f'{data_path}transactions.csv')
        self.cards = pd.read_csv(f'{data_path}cards.csv')
        self.merchants = pd.read_csv(f'{data_path}merchants.csv')
        self.devices = pd.read_csv(f'{data_path}devices.csv')
        self.ips = pd.read_csv(f'{data_path}ips.csv')
        
        # Load fraud rings
        with open(f'{data_path}fraud_rings.json', 'r') as f:
            self.true_fraud_rings = json.load(f)
            
        # Load detected fraud rings if available
        try:
            with open(f'{data_path}detected_fraud_rings.json', 'r') as f:
                detected_data = json.load(f)
                self.detected_fraud_rings = detected_data['fraud_rings']
        except FileNotFoundError:
            self.detected_fraud_rings = []
            
        print(f"Loaded data: {len(self.transactions)} transactions, {len(self.true_fraud_rings)} true rings, {len(self.detected_fraud_rings)} detected rings")
    
    def create_transaction_heatmap(self, time_window: str = 'hour') -> go.Figure:
        """Create heatmap of transaction patterns over time"""
        df = self.transactions.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['weekday'] = df['timestamp'].dt.day_name()
        
        if time_window == 'hour':
            # Hourly heatmap
            heatmap_data = df.groupby(['day', 'hour']).agg({
                'is_fraud': ['count', 'sum']
            }).reset_index()
            heatmap_data.columns = ['day', 'hour', 'total_transactions', 'fraud_transactions']
            heatmap_data['fraud_rate'] = heatmap_data['fraud_transactions'] / heatmap_data['total_transactions']
            
            # Pivot for heatmap
            pivot_data = heatmap_data.pivot(index='day', columns='hour', values='fraud_rate').fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Reds',
                colorbar=dict(title='Fraud Rate'),
                hoverongaps=False
            ))
            
            fig.update_layout(
                title='Fraud Rate Heatmap by Day and Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Day of Month',
                width=800,
                height=600
            )
            
        elif time_window == 'weekday':
            # Weekday heatmap
            heatmap_data = df.groupby(['weekday', 'hour']).agg({
                'is_fraud': ['count', 'sum']
            }).reset_index()
            heatmap_data.columns = ['weekday', 'hour', 'total_transactions', 'fraud_transactions']
            heatmap_data['fraud_rate'] = heatmap_data['fraud_transactions'] / heatmap_data['total_transactions']
            
            # Order weekdays
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data['weekday'] = pd.Categorical(heatmap_data['weekday'], categories=weekday_order, ordered=True)
            
            pivot_data = heatmap_data.pivot(index='weekday', columns='hour', values='fraud_rate').fillna(0)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Reds',
                colorbar=dict(title='Fraud Rate')
            ))
            
            fig.update_layout(
                title='Fraud Rate Heatmap by Weekday and Hour',
                xaxis_title='Hour of Day',
                yaxis_title='Day of Week',
                width=800,
                height=400
            )
        
        return fig
    
    def create_entity_relationship_heatmap(self) -> go.Figure:
        """Create heatmap showing relationships between different entity types"""
        # Calculate relationship strengths
        relationships = {}
        
        # Card-Merchant relationships
        card_merchant = self.transactions.groupby(['card_id', 'merchant_id']).agg({
            'is_fraud': ['count', 'sum'],
            'amount': 'sum'
        }).reset_index()
        card_merchant.columns = ['card_id', 'merchant_id', 'transaction_count', 'fraud_count', 'total_amount']
        card_merchant['fraud_rate'] = card_merchant['fraud_count'] / card_merchant['transaction_count']
        
        # Create matrices for top entities
        top_cards = card_merchant.groupby('card_id')['fraud_count'].sum().nlargest(20).index
        top_merchants = card_merchant.groupby('merchant_id')['fraud_count'].sum().nlargest(15).index
        
        subset = card_merchant[
            (card_merchant['card_id'].isin(top_cards)) & 
            (card_merchant['merchant_id'].isin(top_merchants))
        ]
        
        heatmap_matrix = subset.pivot(index='card_id', columns='merchant_id', values='fraud_rate').fillna(0)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_matrix.values,
            x=[f'M_{i}' for i in range(len(heatmap_matrix.columns))],
            y=[f'C_{i}' for i in range(len(heatmap_matrix.index))],
            colorscale='Viridis',
            colorbar=dict(title='Fraud Rate')
        ))
        
        fig.update_layout(
            title='Card-Merchant Fraud Rate Heatmap (Top Entities)',
            xaxis_title='Merchants',
            yaxis_title='Cards',
            width=800,
            height=600
        )
        
        return fig
    
    def visualize_fraud_rings(self, ring_type: str = 'detected') -> go.Figure:
        """Visualize fraud rings as network graphs"""
        rings = self.detected_fraud_rings if ring_type == 'detected' else self.true_fraud_rings
        
        if not rings:
            return go.Figure().add_annotation(text="No fraud rings to visualize", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges for each ring
        ring_colors = {}
        node_ring_mapping = {}
        
        for ring_idx, ring in enumerate(rings[:10]):  # Limit to first 10 rings for clarity
            color = self.color_palette[ring_idx % len(self.color_palette)]
            ring_id = ring.get('ring_id', f'ring_{ring_idx}')
            
            # Handle different ring formats
            if 'nodes' in ring:  # Detected rings format
                nodes = ring['nodes']
                for node in nodes:
                    G.add_node(node, ring=ring_id, color=color)
                    node_ring_mapping[node] = ring_id
                    ring_colors[ring_id] = color
                
                # Add edges within the ring
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        G.add_edge(nodes[i], nodes[j], ring=ring_id)
                        
            else:  # True rings format
                all_entities = []
                for entity_type in ['cards', 'merchants', 'devices', 'ips']:
                    if entity_type in ring:
                        for entity in ring[entity_type]:
                            node_name = f"{entity_type[:-1]}_{entity}"
                            all_entities.append(node_name)
                            G.add_node(node_name, ring=ring_id, color=color, type=entity_type[:-1])
                            node_ring_mapping[node_name] = ring_id
                            ring_colors[ring_id] = color
                
                # Add edges within the ring
                for i in range(len(all_entities)):
                    for j in range(i+1, len(all_entities)):
                        G.add_edge(all_entities[i], all_entities[j], ring=ring_id)
        
        if G.number_of_nodes() == 0:
            return go.Figure().add_annotation(text="No nodes to visualize", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Prepare node traces
        node_traces = {}
        for ring_id, color in ring_colors.items():
            node_traces[ring_id] = {
                'x': [],
                'y': [],
                'text': [],
                'color': color
            }
        
        for node in G.nodes():
            x, y = pos[node]
            ring_id = node_ring_mapping.get(node, 'unknown')
            if ring_id in node_traces:
                node_traces[ring_id]['x'].append(x)
                node_traces[ring_id]['y'].append(y)
                node_traces[ring_id]['text'].append(node)
        
        # Prepare edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes by ring
        for ring_id, trace_data in node_traces.items():
            fig.add_trace(go.Scatter(
                x=trace_data['x'], y=trace_data['y'],
                mode='markers',
                hoverinfo='text',
                text=trace_data['text'],
                marker=dict(
                    size=10,
                    color=trace_data['color'],
                    line=dict(width=2, color='white')
                ),
                name=ring_id
            ))
        
        fig.update_layout(
            title=f'{ring_type.title()} Fraud Rings Network Visualization',
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[dict(
                text=f"Showing {len(rings)} fraud rings",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='#888', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=1000,
            height=800
        )
        
        return fig
    
    def create_risk_score_heatmap(self, model_results: Optional[Dict] = None) -> go.Figure:
        """Create heatmap of risk scores across different dimensions"""
        if model_results is None:
            # Create synthetic risk scores based on transaction patterns
            risk_data = self.transactions.groupby(['card_id']).agg({
                'is_fraud': ['count', 'sum', 'mean'],
                'amount': ['mean', 'std', 'sum']
            }).reset_index()
            
            risk_data.columns = ['card_id', 'transaction_count', 'fraud_count', 'fraud_rate', 
                               'avg_amount', 'amount_std', 'total_amount']
            risk_data['amount_std'] = risk_data['amount_std'].fillna(0)
            
            # Calculate composite risk score
            risk_data['risk_score'] = (
                risk_data['fraud_rate'] * 0.4 +
                (risk_data['fraud_count'] / risk_data['fraud_count'].max()) * 0.3 +
                (risk_data['avg_amount'] / risk_data['avg_amount'].max()) * 0.2 +
                (risk_data['amount_std'] / risk_data['amount_std'].max()) * 0.1
            )
        else:
            # Use actual model risk scores
            risk_data = model_results.get('risk_scores', {})
        
        # Create risk score distribution heatmap
        top_risky_cards = risk_data.nlargest(100, 'risk_score')
        
        # Group into risk buckets
        top_risky_cards['risk_bucket'] = pd.cut(top_risky_cards['risk_score'], 
                                               bins=10, labels=[f'Bucket_{i}' for i in range(10)])
        
        # Create heatmap data
        heatmap_data = top_risky_cards.groupby(['risk_bucket']).agg({
            'fraud_rate': 'mean',
            'avg_amount': 'mean',
            'transaction_count': 'mean'
        }).reset_index()
        
        # Normalize for heatmap
        for col in ['fraud_rate', 'avg_amount', 'transaction_count']:
            heatmap_data[f'{col}_norm'] = heatmap_data[col] / heatmap_data[col].max()
        
        # Create matrix
        matrix_data = heatmap_data[['fraud_rate_norm', 'avg_amount_norm', 'transaction_count_norm']].T
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix_data.values,
            x=[f'Risk Bucket {i}' for i in range(len(matrix_data.columns))],
            y=['Fraud Rate', 'Avg Amount', 'Transaction Count'],
            colorscale='RdYlBu_r',
            colorbar=dict(title='Normalized Score')
        ))
        
        fig.update_layout(
            title='Risk Score Heatmap by Risk Buckets',
            xaxis_title='Risk Buckets (Low to High)',
            yaxis_title='Risk Factors',
            width=800,
            height=400
        )
        
        return fig
    
    def create_embedding_visualization(self, embeddings: torch.Tensor, 
                                     labels: List[str], fraud_labels: List[int],
                                     method: str = 'tsne') -> go.Figure:
        """Visualize node embeddings in 2D space"""
        embeddings_np = embeddings.detach().cpu().numpy()
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np)//4))
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embedding_2d = reducer.fit_transform(embeddings_np)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Separate fraud and non-fraud points
        fraud_mask = np.array(fraud_labels) == 1
        
        # Non-fraud points
        fig.add_trace(go.Scatter(
            x=embedding_2d[~fraud_mask, 0],
            y=embedding_2d[~fraud_mask, 1],
            mode='markers',
            marker=dict(
                size=6,
                color='lightblue',
                opacity=0.6
            ),
            name='Normal',
            text=[labels[i] for i in range(len(labels)) if not fraud_mask[i]],
            hovertemplate='%{text}<br>X: %{x}<br>Y: %{y}<extra></extra>'
        ))
        
        # Fraud points
        if fraud_mask.any():
            fig.add_trace(go.Scatter(
                x=embedding_2d[fraud_mask, 0],
                y=embedding_2d[fraud_mask, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    opacity=0.8,
                    symbol='diamond'
                ),
                name='Fraud',
                text=[labels[i] for i in range(len(labels)) if fraud_mask[i]],
                hovertemplate='%{text}<br>X: %{x}<br>Y: %{y}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'Node Embeddings Visualization ({method.upper()})',
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2',
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_performance_dashboard(self, model_results: Dict) -> go.Figure:
        """Create comprehensive performance dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'ROC Curves', 'Precision-Recall', 'Confusion Matrix'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        if 'train_losses' in model_results:
            fig.add_trace(
                go.Scatter(x=list(range(len(model_results['train_losses']))), 
                          y=model_results['train_losses'],
                          name='Training Loss',
                          line=dict(color='blue')),
                row=1, col=1
            )
        
        # Add more plots based on available data
        fig.update_layout(
            title='Model Performance Dashboard',
            height=800,
            showlegend=True
        )
        
        return fig
    
    def save_all_visualizations(self):
        """Generate and save all visualizations"""
        print("Generating visualizations...")
        
        # Transaction heatmaps
        print("Creating transaction heatmaps...")
        hourly_heatmap = self.create_transaction_heatmap('hour')
        hourly_heatmap.write_html(f'{self.output_dir}transaction_heatmap_hourly.html')
        
        weekday_heatmap = self.create_transaction_heatmap('weekday')
        weekday_heatmap.write_html(f'{self.output_dir}transaction_heatmap_weekday.html')
        
        # Entity relationship heatmap
        print("Creating entity relationship heatmap...")
        relationship_heatmap = self.create_entity_relationship_heatmap()
        relationship_heatmap.write_html(f'{self.output_dir}entity_relationship_heatmap.html')
        
        # Fraud ring visualizations
        print("Creating fraud ring visualizations...")
        if self.detected_fraud_rings:
            detected_rings_viz = self.visualize_fraud_rings('detected')
            detected_rings_viz.write_html(f'{self.output_dir}detected_fraud_rings.html')
        
        if self.true_fraud_rings:
            true_rings_viz = self.visualize_fraud_rings('true')
            true_rings_viz.write_html(f'{self.output_dir}true_fraud_rings.html')
        
        # Risk score heatmap
        print("Creating risk score heatmap...")
        risk_heatmap = self.create_risk_score_heatmap()
        risk_heatmap.write_html(f'{self.output_dir}risk_score_heatmap.html')
        
        print(f"All visualizations saved to {self.output_dir}")
        
        # Create summary report
        self._create_summary_report()
    
    def _create_summary_report(self):
        """Create HTML summary report with all visualizations"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Detection Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ text-align: center; color: #333; }}
                .section {{ margin: 30px 0; }}
                .viz-container {{ text-align: center; margin: 20px 0; }}
                .stats {{ background: #f5f5f5; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Credit Card Fraud Detection Analysis</h1>
                <p>Graph-based Deep Learning Approach</p>
            </div>
            
            <div class="section">
                <h2>Dataset Overview</h2>
                <div class="stats">
                    <p><strong>Total Transactions:</strong> {len(self.transactions):,}</p>
                    <p><strong>Fraudulent Transactions:</strong> {self.transactions['is_fraud'].sum():,} ({self.transactions['is_fraud'].mean():.2%})</p>
                    <p><strong>Unique Cards:</strong> {self.transactions['card_id'].nunique():,}</p>
                    <p><strong>Unique Merchants:</strong> {self.transactions['merchant_id'].nunique():,}</p>
                    <p><strong>True Fraud Rings:</strong> {len(self.true_fraud_rings)}</p>
                    <p><strong>Detected Fraud Rings:</strong> {len(self.detected_fraud_rings)}</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="viz-container">
                    <h3>Transaction Pattern Analysis</h3>
                    <p><a href="transaction_heatmap_hourly.html" target="_blank">Hourly Transaction Heatmap</a></p>
                    <p><a href="transaction_heatmap_weekday.html" target="_blank">Weekly Transaction Heatmap</a></p>
                </div>
                
                <div class="viz-container">
                    <h3>Entity Relationship Analysis</h3>
                    <p><a href="entity_relationship_heatmap.html" target="_blank">Card-Merchant Relationship Heatmap</a></p>
                    <p><a href="risk_score_heatmap.html" target="_blank">Risk Score Analysis</a></p>
                </div>
                
                <div class="viz-container">
                    <h3>Fraud Ring Detection</h3>
                    <p><a href="true_fraud_rings.html" target="_blank">Ground Truth Fraud Rings</a></p>
                    <p><a href="detected_fraud_rings.html" target="_blank">Detected Fraud Rings</a></p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(f'{self.output_dir}fraud_analysis_report.html', 'w') as f:
            f.write(html_content)
        
        print(f"Summary report saved to {self.output_dir}fraud_analysis_report.html")

if __name__ == "__main__":
    # Initialize visualizer
    visualizer = FraudVisualizer()
    
    # Load data
    print("Loading data...")
    visualizer.load_data()
    
    # Generate all visualizations
    visualizer.save_all_visualizations()
    
    print("Visualization complete! Check the visualizations directory for outputs.")