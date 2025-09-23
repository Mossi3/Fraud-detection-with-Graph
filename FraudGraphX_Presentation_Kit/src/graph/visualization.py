"""
Interactive Graph Visualization Dashboard
Advanced visualization for fraud detection graphs
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """
    Advanced graph visualization for fraud detection
    """
    
    def __init__(self, graph: nx.Graph, communities: Dict[int, List] = None):
        self.graph = graph
        self.communities = communities or {}
        self.layout = None
        
    def create_interactive_network(self, fraud_labels: Dict[str, int] = None,
                                 node_size_factor: float = 10,
                                 edge_width_factor: float = 1) -> go.Figure:
        """
        Create interactive network visualization
        """
        
        # Calculate layout if not already done
        if self.layout is None:
            self.layout = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in self.graph.nodes():
            x, y = self.layout[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node information
            node_info = f"Node: {node}<br>"
            node_info += f"Type: {self.graph.nodes[node].get('node_type', 'unknown')}<br>"
            node_info += f"Degree: {self.graph.degree(node)}<br>"
            
            if fraud_labels and node in fraud_labels:
                node_info += f"Fraud: {'Yes' if fraud_labels[node] == 1 else 'No'}<br>"
            
            node_text.append(node_info)
            
            # Node color based on fraud status
            if fraud_labels and node in fraud_labels:
                node_colors.append('red' if fraud_labels[node] == 1 else 'blue')
            else:
                node_colors.append('lightblue')
            
            # Node size based on degree
            degree = self.graph.degree(node)
            node_sizes.append(max(5, degree * node_size_factor))
        
        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in self.graph.edges():
            x0, y0 = self.layout[edge[0]]
            x1, y1 = self.layout[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            # Edge information
            edge_data = self.graph.get_edge_data(edge[0], edge[1], default={})
            edge_info.append(f"Edge: {edge[0]} -> {edge[1]}<br>Type: {edge_data.get('edge_type', 'unknown')}<br>Weight: {edge_data.get('weight', 1.0):.2f}")
        
        # Create network trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                colorbar=dict(
                    thickness=15,
                    xanchor="left",
                    titleside="right"
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Fraud Detection Graph',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Interactive fraud detection graph",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='#888', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_community_visualization(self, fraud_labels: Dict[str, int] = None) -> go.Figure:
        """
        Create community visualization with different colors for each community
        """
        
        if not self.communities:
            return self.create_interactive_network(fraud_labels)
        
        # Calculate layout
        if self.layout is None:
            self.layout = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Prepare data for each community
        traces = []
        colors = px.colors.qualitative.Set3
        
        for comm_id, nodes in self.communities.items():
            if len(nodes) < 2:  # Skip single-node communities
                continue
            
            # Get subgraph for this community
            subgraph = self.graph.subgraph(nodes)
            
            # Node positions
            node_x = []
            node_y = []
            node_text = []
            node_colors = []
            node_sizes = []
            
            for node in nodes:
                x, y = self.layout[node]
                node_x.append(x)
                node_y.append(y)
                
                # Node information
                node_info = f"Node: {node}<br>"
                node_info += f"Community: {comm_id}<br>"
                node_info += f"Type: {self.graph.nodes[node].get('node_type', 'unknown')}<br>"
                node_info += f"Degree: {self.graph.degree(node)}<br>"
                
                if fraud_labels and node in fraud_labels:
                    node_info += f"Fraud: {'Yes' if fraud_labels[node] == 1 else 'No'}<br>"
                
                node_text.append(node_info)
                
                # Node color based on fraud status within community
                if fraud_labels and node in fraud_labels:
                    node_colors.append('red' if fraud_labels[node] == 1 else 'orange')
                else:
                    node_colors.append(colors[comm_id % len(colors)])
                
                # Node size based on degree
                degree = self.graph.degree(node)
                node_sizes.append(max(5, degree * 8))
            
            # Edge positions for this community
            edge_x = []
            edge_y = []
            
            for edge in subgraph.edges():
                x0, y0 = self.layout[edge[0]]
                x1, y1 = self.layout[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Add edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color=colors[comm_id % len(colors)]),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            traces.append(edge_trace)
            
            # Add node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                hoverinfo='text',
                text=node_text,
                marker=dict(
                    color=node_colors,
                    size=node_sizes,
                    line=dict(width=2, color='black')
                ),
                name=f'Community {comm_id}',
                showlegend=True
            )
            traces.append(node_trace)
        
        # Create figure
        fig = go.Figure(data=traces,
                       layout=go.Layout(
                           title='Community Detection Results',
                           titlefont_size=16,
                           showlegend=True,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def create_centrality_heatmap(self, centrality_measures: Dict[str, Dict]) -> go.Figure:
        """
        Create heatmap of centrality measures
        """
        
        # Prepare data
        nodes = list(centrality_measures['degree'].keys())
        measures = list(centrality_measures.keys())
        
        # Create matrix
        matrix = []
        for measure in measures:
            row = [centrality_measures[measure].get(node, 0) for node in nodes]
            matrix.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=nodes,
            y=measures,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Centrality Measures Heatmap',
            xaxis_title='Nodes',
            yaxis_title='Centrality Measures'
        )
        
        return fig
    
    def create_fraud_ring_analysis(self, fraud_rings: Dict[str, Dict]) -> go.Figure:
        """
        Create analysis visualization for detected fraud rings
        """
        
        # Prepare data for bar chart
        ring_ids = []
        fraud_rates = []
        ring_sizes = []
        suspicious_scores = []
        
        for method, rings in fraud_rings.items():
            for ring_id, ring_data in rings.items():
                ring_ids.append(f"{method}_{ring_id}")
                fraud_rates.append(ring_data['fraud_rate'])
                ring_sizes.append(ring_data['size'])
                suspicious_scores.append(ring_data['suspicious_score'])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fraud Rate by Ring', 'Ring Size Distribution', 
                          'Suspicious Score Distribution', 'Fraud Rate vs Ring Size'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Fraud rate bar chart
        fig.add_trace(
            go.Bar(x=ring_ids, y=fraud_rates, name='Fraud Rate'),
            row=1, col=1
        )
        
        # Ring size histogram
        fig.add_trace(
            go.Histogram(x=ring_sizes, name='Ring Size', nbinsx=10),
            row=1, col=2
        )
        
        # Suspicious score histogram
        fig.add_trace(
            go.Histogram(x=suspicious_scores, name='Suspicious Score', nbinsx=10),
            row=2, col=1
        )
        
        # Scatter plot: fraud rate vs ring size
        fig.add_trace(
            go.Scatter(x=ring_sizes, y=fraud_rates, mode='markers', 
                      name='Fraud Rate vs Size', text=ring_ids),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Fraud Ring Analysis")
        
        return fig
    
    def create_temporal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Create temporal analysis of fraud patterns
        """
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')
        
        # Group by time periods
        df['hour'] = df['datetime'].dt.hour
        df['day'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Fraud Rate by Hour', 'Fraud Rate by Day',
                          'Fraud Rate by Month', 'Transaction Volume Over Time'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Hourly fraud rate
        hourly_fraud = df.groupby('hour')['fraud'].mean()
        fig.add_trace(
            go.Bar(x=hourly_fraud.index, y=hourly_fraud.values, name='Hourly Fraud Rate'),
            row=1, col=1
        )
        
        # Daily fraud rate
        daily_fraud = df.groupby('day')['fraud'].mean()
        fig.add_trace(
            go.Bar(x=daily_fraud.index, y=daily_fraud.values, name='Daily Fraud Rate'),
            row=1, col=2
        )
        
        # Monthly fraud rate
        monthly_fraud = df.groupby('month')['fraud'].mean()
        fig.add_trace(
            go.Bar(x=monthly_fraud.index, y=monthly_fraud.values, name='Monthly Fraud Rate'),
            row=2, col=1
        )
        
        # Transaction volume over time
        daily_volume = df.groupby(df['datetime'].dt.date).size()
        fig.add_trace(
            go.Scatter(x=daily_volume.index, y=daily_volume.values, 
                      mode='lines', name='Daily Volume'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Temporal Fraud Analysis")
        
        return fig

def create_streamlit_dashboard():
    """
    Create Streamlit dashboard for graph-based fraud detection
    """
    
    st.set_page_config(
        page_title="FraudGraphX - Graph-based Fraud Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 FraudGraphX - Graph-based Fraud Detection Dashboard")
    st.markdown("Advanced graph-based fraud detection with community analysis and visualization")
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload transaction data (CSV)",
        type=['csv'],
        help="Upload a CSV file with transaction data"
    )
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} transactions")
        
        # Configuration options
        st.sidebar.subheader("Graph Configuration")
        
        include_cards = st.sidebar.checkbox("Include Cards", value=True)
        include_merchants = st.sidebar.checkbox("Include Merchants", value=True)
        include_devices = st.sidebar.checkbox("Include Devices", value=True)
        include_ips = st.sidebar.checkbox("Include IPs", value=True)
        
        st.sidebar.subheader("Community Detection")
        
        community_method = st.sidebar.selectbox(
            "Community Detection Method",
            ["Louvain", "Leiden", "Both"]
        )
        
        resolution = st.sidebar.slider("Resolution Parameter", 0.1, 2.0, 1.0, 0.1)
        
        st.sidebar.subheader("Visualization")
        
        node_size_factor = st.sidebar.slider("Node Size Factor", 1, 20, 10)
        edge_width_factor = st.sidebar.slider("Edge Width Factor", 0.1, 2.0, 1.0, 0.1)
        
        # Main content
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "🕸️ Graph Visualization", "🏘️ Community Analysis", 
            "🔍 Fraud Rings", "📈 Temporal Analysis"
        ])
        
        with tab1:
            st.header("Data Overview")
            
            # Basic statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Transactions", len(df))
            
            with col2:
                fraud_count = df['fraud'].sum() if 'fraud' in df.columns else 0
                st.metric("Fraud Transactions", fraud_count)
            
            with col3:
                fraud_rate = fraud_count / len(df) if len(df) > 0 else 0
                st.metric("Fraud Rate", f"{fraud_rate:.2%}")
            
            with col4:
                unique_cards = df['card_id'].nunique() if 'card_id' in df.columns else 0
                st.metric("Unique Cards", unique_cards)
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Column information
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-null Count': df.count(),
                'Null Count': df.isnull().sum()
            })
            st.dataframe(col_info)
        
        with tab2:
            st.header("Graph Visualization")
            
            if st.button("Generate Graph"):
                with st.spinner("Building graph..."):
                    # Import graph construction
                    from src.graph.construction import HeterogeneousGraphBuilder, GraphConfig
                    
                    # Configure graph
                    config = GraphConfig(
                        include_cards=include_cards,
                        include_merchants=include_merchants,
                        include_devices=include_devices,
                        include_ips=include_ips
                    )
                    
                    # Build graph
                    builder = HeterogeneousGraphBuilder(config)
                    builder.add_transaction_data(df)
                    graph = builder.to_networkx()
                    
                    # Get fraud labels
                    fraud_labels = {}
                    if 'fraud' in df.columns and 'card_id' in df.columns:
                        fraud_labels = dict(zip(df['card_id'], df['fraud']))
                    
                    # Create visualizer
                    visualizer = GraphVisualizer(graph)
                    
                    # Create visualization
                    fig = visualizer.create_interactive_network(fraud_labels, node_size_factor, edge_width_factor)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Graph statistics
                    stats = builder.get_graph_statistics()
                    st.subheader("Graph Statistics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Number of Nodes", stats['num_nodes'])
                        st.metric("Number of Edges", stats['num_edges'])
                        st.metric("Graph Density", f"{stats['density']:.4f}")
                    
                    with col2:
                        st.metric("Is Connected", "Yes" if stats['is_connected'] else "No")
                        st.metric("Number of Components", stats['num_components'])
                        st.metric("Average Clustering", f"{stats['average_clustering']:.4f}")
        
        with tab3:
            st.header("Community Analysis")
            
            if st.button("Detect Communities"):
                with st.spinner("Detecting communities..."):
                    # Import community detection
                    from src.graph.community_detection import CommunityDetector
                    
                    # Build graph
                    from src.graph.construction import HeterogeneousGraphBuilder, GraphConfig
                    config = GraphConfig()
                    builder = HeterogeneousGraphBuilder(config)
                    builder.add_transaction_data(df)
                    graph = builder.to_networkx()
                    
                    # Detect communities
                    detector = CommunityDetector(graph)
                    
                    communities = {}
                    if community_method in ["Louvain", "Both"]:
                        louvain_communities = detector.detect_louvain_communities(resolution)
                        communities['louvain'] = louvain_communities
                    
                    if community_method in ["Leiden", "Both"]:
                        leiden_communities = detector.detect_leiden_communities(resolution)
                        communities['leiden'] = leiden_communities
                    
                    # Get fraud labels
                    fraud_labels = {}
                    if 'fraud' in df.columns and 'card_id' in df.columns:
                        fraud_labels = dict(zip(df['card_id'], df['fraud']))
                    
                    # Create visualizations
                    visualizer = GraphVisualizer(graph, communities.get('louvain', {}))
                    fig = visualizer.create_community_visualization(fraud_labels)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Community statistics
                    st.subheader("Community Statistics")
                    
                    for method, comms in communities.items():
                        st.write(f"**{method.title()} Communities:**")
                        
                        stats = detector.community_stats.get(method, {})
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Number of Communities", stats.get('num_communities', 0))
                            st.metric("Average Size", f"{stats.get('avg_community_size', 0):.1f}")
                        
                        with col2:
                            st.metric("Max Size", stats.get('max_community_size', 0))
                            st.metric("Min Size", stats.get('min_community_size', 0))
                        
                        with col3:
                            st.metric("Modularity", f"{stats.get('modularity', 0):.4f}")
        
        with tab4:
            st.header("Fraud Ring Detection")
            
            if st.button("Detect Fraud Rings"):
                with st.spinner("Detecting fraud rings..."):
                    # Build graph and detect communities
                    from src.graph.construction import HeterogeneousGraphBuilder, GraphConfig
                    from src.graph.community_detection import CommunityDetector
                    
                    config = GraphConfig()
                    builder = HeterogeneousGraphBuilder(config)
                    builder.add_transaction_data(df)
                    graph = builder.to_networkx()
                    
                    detector = CommunityDetector(graph)
                    communities = detector.detect_louvain_communities(resolution)
                    
                    # Get fraud labels
                    fraud_labels = {}
                    if 'fraud' in df.columns and 'card_id' in df.columns:
                        fraud_labels = dict(zip(df['card_id'], df['fraud']))
                    
                    # Detect fraud rings
                    fraud_rings = detector.detect_fraud_rings(fraud_labels)
                    
                    # Create visualization
                    visualizer = GraphVisualizer(graph, communities)
                    fig = visualizer.create_fraud_ring_analysis(fraud_rings)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fraud ring details
                    st.subheader("Detected Fraud Rings")
                    
                    for method, rings in fraud_rings.items():
                        st.write(f"**{method.title()} Method:**")
                        
                        if rings:
                            ring_data = []
                            for ring_id, ring_info in rings.items():
                                ring_data.append({
                                    'Ring ID': ring_id,
                                    'Size': ring_info['size'],
                                    'Fraud Rate': f"{ring_info['fraud_rate']:.2%}",
                                    'Fraud Count': ring_info['fraud_count'],
                                    'Suspicious Score': f"{ring_info['suspicious_score']:.3f}"
                                })
                            
                            ring_df = pd.DataFrame(ring_data)
                            st.dataframe(ring_df)
                        else:
                            st.info("No fraud rings detected with current parameters.")
        
        with tab5:
            st.header("Temporal Analysis")
            
            if st.button("Analyze Temporal Patterns"):
                with st.spinner("Analyzing temporal patterns..."):
                    visualizer = GraphVisualizer(nx.Graph())
                    fig = visualizer.create_temporal_analysis(df)
                    st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Show sample data structure
        st.subheader("Expected Data Format")
        sample_data = {
            'transaction_id': [1, 2, 3, 4, 5],
            'card_id': ['card_001', 'card_002', 'card_001', 'card_003', 'card_002'],
            'merchant_id': ['merchant_001', 'merchant_002', 'merchant_001', 'merchant_003', 'merchant_002'],
            'device_id': ['device_001', 'device_002', 'device_001', 'device_003', 'device_002'],
            'ip': ['192.168.1.1', '192.168.1.2', '192.168.1.1', '192.168.1.3', '192.168.1.2'],
            'amount': [100.0, 250.0, 75.0, 500.0, 150.0],
            'fraud': [0, 1, 0, 1, 0],
            'timestamp': [1640995200, 1640998800, 1641002400, 1641006000, 1641009600]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

if __name__ == "__main__":
    create_streamlit_dashboard()