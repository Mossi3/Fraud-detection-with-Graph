"""
Interactive visualization module for fraud detection graphs.
Creates compelling visualizations of fraud rings and transaction patterns.
"""

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from pyvis.network import Network
import seaborn as sns
from datetime import datetime
import json
import os
from loguru import logger


class FraudGraphVisualizer:
    """
    Creates interactive and static visualizations of fraud detection graphs.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.color_scheme = {
            'transaction': '#3498db',
            'card': '#e74c3c',
            'merchant': '#2ecc71',
            'device': '#f39c12',
            'ip_address': '#9b59b6',
            'velocity_pattern': '#e67e22',
            'fraud': '#c0392b',
            'normal': '#95a5a6',
            'suspicious': '#f39c12'
        }
        
    def visualize_fraud_ring(self, graph: nx.Graph, 
                           ring_nodes: Set[str],
                           fraud_labels: Dict[str, bool] = None,
                           output_path: str = None,
                           interactive: bool = True) -> Optional[go.Figure]:
        """
        Create visualization focusing on a specific fraud ring.
        
        Args:
            graph: NetworkX graph
            ring_nodes: Set of nodes in the fraud ring
            fraud_labels: Dictionary of fraud labels
            output_path: Path to save visualization
            interactive: Whether to create interactive plot
            
        Returns:
            Plotly figure if interactive, else None
        """
        fraud_labels = fraud_labels or {}
        
        # Extract subgraph
        extended_nodes = set(ring_nodes)
        
        # Include immediate neighbors for context
        for node in ring_nodes:
            extended_nodes.update(graph.neighbors(node))
            
        subgraph = graph.subgraph(extended_nodes)
        
        if interactive:
            return self._create_interactive_ring_viz(
                subgraph, ring_nodes, fraud_labels, output_path
            )
        else:
            self._create_static_ring_viz(
                subgraph, ring_nodes, fraud_labels, output_path
            )
            
    def _create_interactive_ring_viz(self, graph: nx.Graph,
                                   ring_nodes: Set[str],
                                   fraud_labels: Dict[str, bool],
                                   output_path: Optional[str]) -> go.Figure:
        """Create interactive Plotly visualization"""
        # Calculate layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Prepare edge trace
        edge_trace = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            # Highlight edges within the ring
            if edge[0] in ring_nodes and edge[1] in ring_nodes:
                color = 'red'
                width = 3
            else:
                color = '#888'
                width = 1
                
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Prepare node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='RdYlBu',
                size=[],
                color=[],
                colorbar=dict(
                    thickness=15,
                    title='Node Risk Score',
                    xanchor='left',
                    titleside='right'
                ),
            )
        )
        
        # Add node data
        for node in graph.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            # Node styling
            if node in ring_nodes:
                size = 30
                if fraud_labels.get(node, False):
                    color = 10  # High risk
                else:
                    color = 7   # Medium risk
            else:
                size = 15
                color = 3 if not fraud_labels.get(node, False) else 8
                
            node_trace['marker']['size'] += tuple([size])
            node_trace['marker']['color'] += tuple([color])
            
            # Node info
            node_data = graph.nodes[node]
            node_type = node_data.get('entity_type', 'unknown')
            
            text = f"ID: {node}<br>Type: {node_type}"
            if node_type == 'transaction':
                text += f"<br>Amount: ${node_data.get('amount', 0):.2f}"
            
            node_trace['text'] += tuple([text])
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        # Update layout
        fig.update_layout(
            title=f"Fraud Ring Visualization ({len(ring_nodes)} nodes)",
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Red edges indicate connections within the fraud ring",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Saved interactive visualization to {output_path}")
            
        return fig
    
    def _create_static_ring_viz(self, graph: nx.Graph,
                              ring_nodes: Set[str],
                              fraud_labels: Dict[str, bool],
                              output_path: Optional[str]):
        """Create static matplotlib visualization"""
        plt.figure(figsize=(12, 8))
        
        # Calculate layout
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw nodes
        node_colors = []
        node_sizes = []
        
        for node in graph.nodes():
            node_data = graph.nodes[node]
            entity_type = node_data.get('entity_type', 'unknown')
            
            if node in ring_nodes:
                if fraud_labels.get(node, False):
                    color = self.color_scheme['fraud']
                else:
                    color = self.color_scheme['suspicious']
                size = 500
            else:
                color = self.color_scheme.get(entity_type, self.color_scheme['normal'])
                size = 200
                
            node_colors.append(color)
            node_sizes.append(size)
        
        # Draw edges
        edge_colors = []
        edge_widths = []
        
        for u, v in graph.edges():
            if u in ring_nodes and v in ring_nodes:
                edge_colors.append('red')
                edge_widths.append(2)
            else:
                edge_colors.append('gray')
                edge_widths.append(0.5)
        
        # Draw graph
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors,
                              width=edge_widths, alpha=0.6)
        
        # Add labels for ring nodes
        ring_labels = {node: node.split('_')[1][:8] for node in ring_nodes}
        nx.draw_networkx_labels(graph, pos, ring_labels, font_size=8)
        
        plt.title(f"Fraud Ring Visualization ({len(ring_nodes)} nodes)")
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved static visualization to {output_path}")
        
        plt.close()
    
    def create_pyvis_network(self, graph: nx.Graph,
                           fraud_rings: List[Set[str]] = None,
                           output_path: str = "fraud_network.html"):
        """Create interactive PyVis network visualization"""
        net = Network(height="800px", width="100%", bgcolor="#ffffff", 
                     font_color="black", notebook=False)
        
        # Physics options for better layout
        net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {"iterations": 150}
            }
        }
        """)
        
        # Identify nodes in fraud rings
        fraud_ring_nodes = set()
        if fraud_rings:
            for ring in fraud_rings:
                fraud_ring_nodes.update(ring)
        
        # Add nodes
        for node in graph.nodes():
            node_data = graph.nodes[node]
            entity_type = node_data.get('entity_type', 'unknown')
            
            # Node styling
            if node in fraud_ring_nodes:
                color = self.color_scheme['fraud']
                size = 25
                title = f"FRAUD RING NODE\n{node}\nType: {entity_type}"
            else:
                color = self.color_scheme.get(entity_type, self.color_scheme['normal'])
                size = 15
                title = f"{node}\nType: {entity_type}"
                
            # Add additional info to title
            if entity_type == 'transaction':
                title += f"\nAmount: ${node_data.get('amount', 0):.2f}"
            
            net.add_node(node, label=node.split('_')[0], title=title,
                        color=color, size=size)
        
        # Add edges
        for u, v, data in graph.edges(data=True):
            edge_type = data.get('relationship_type', 'connected')
            
            # Edge styling
            if u in fraud_ring_nodes and v in fraud_ring_nodes:
                color = 'red'
                width = 3
            else:
                color = 'gray'
                width = 1
                
            net.add_edge(u, v, color=color, width=width, title=edge_type)
        
        # Save network
        net.save_graph(output_path)
        logger.info(f"Saved PyVis network to {output_path}")
    
    def plot_fraud_statistics(self, graph: nx.Graph,
                            fraud_labels: Dict[str, bool],
                            fraud_rings: List[Set[str]] = None,
                            output_path: str = None):
        """Create comprehensive fraud statistics dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Entity Type Distribution',
                'Fraud Rate by Entity Type',
                'Transaction Amount Distribution',
                'Fraud Ring Sizes',
                'Network Metrics',
                'Temporal Patterns'
            ),
            specs=[
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'bar'}, {'type': 'scatter'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. Entity Type Distribution
        entity_types = [data.get('entity_type', 'unknown') 
                       for _, data in graph.nodes(data=True)]
        type_counts = pd.Series(entity_types).value_counts()
        
        fig.add_trace(
            go.Pie(labels=type_counts.index, values=type_counts.values,
                  marker_colors=[self.color_scheme.get(t, '#95a5a6') 
                               for t in type_counts.index]),
            row=1, col=1
        )
        
        # 2. Fraud Rate by Entity Type
        fraud_by_type = {}
        for node, data in graph.nodes(data=True):
            entity_type = data.get('entity_type', 'unknown')
            is_fraud = fraud_labels.get(node, False)
            
            if entity_type not in fraud_by_type:
                fraud_by_type[entity_type] = {'total': 0, 'fraud': 0}
            
            fraud_by_type[entity_type]['total'] += 1
            if is_fraud:
                fraud_by_type[entity_type]['fraud'] += 1
        
        types = list(fraud_by_type.keys())
        fraud_rates = [fraud_by_type[t]['fraud'] / fraud_by_type[t]['total'] 
                      for t in types]
        
        fig.add_trace(
            go.Bar(x=types, y=fraud_rates, 
                  marker_color=[self.color_scheme.get(t, '#95a5a6') for t in types]),
            row=1, col=2
        )
        
        # 3. Transaction Amount Distribution
        amounts = []
        fraud_amounts = []
        
        for node, data in graph.nodes(data=True):
            if data.get('entity_type') == 'transaction':
                amount = data.get('amount', 0)
                amounts.append(amount)
                if fraud_labels.get(node, False):
                    fraud_amounts.append(amount)
        
        fig.add_trace(
            go.Histogram(x=amounts, name='All Transactions', opacity=0.7),
            row=1, col=3
        )
        fig.add_trace(
            go.Histogram(x=fraud_amounts, name='Fraud Transactions', opacity=0.7),
            row=1, col=3
        )
        
        # 4. Fraud Ring Sizes
        if fraud_rings:
            ring_sizes = [len(ring) for ring in fraud_rings]
            fig.add_trace(
                go.Bar(y=ring_sizes, x=list(range(len(ring_sizes))),
                      marker_color='red'),
                row=2, col=1
            )
        
        # 5. Network Metrics
        degrees = dict(graph.degree())
        fraud_degrees = [deg for node, deg in degrees.items() 
                        if fraud_labels.get(node, False)]
        normal_degrees = [deg for node, deg in degrees.items() 
                         if not fraud_labels.get(node, False)]
        
        fig.add_trace(
            go.Scatter(y=sorted(normal_degrees, reverse=True),
                      mode='lines', name='Normal Nodes'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(y=sorted(fraud_degrees, reverse=True),
                      mode='lines', name='Fraud Nodes'),
            row=2, col=2
        )
        
        # 6. Temporal Patterns
        timestamps = []
        fraud_timestamps = []
        
        for node, data in graph.nodes(data=True):
            if 'timestamp' in data:
                try:
                    ts = datetime.fromisoformat(data['timestamp'])
                    hour = ts.hour
                    timestamps.append(hour)
                    if fraud_labels.get(node, False):
                        fraud_timestamps.append(hour)
                except:
                    pass
        
        if timestamps:
            hours = list(range(24))
            normal_dist = [timestamps.count(h) for h in hours]
            fraud_dist = [fraud_timestamps.count(h) for h in hours]
            
            fig.add_trace(
                go.Scatter(x=hours, y=normal_dist, mode='lines+markers',
                          name='Normal Activity'),
                row=2, col=3
            )
            fig.add_trace(
                go.Scatter(x=hours, y=fraud_dist, mode='lines+markers',
                          name='Fraud Activity'),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title="Fraud Detection Dashboard",
            height=800,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Entity Type", row=1, col=2)
        fig.update_yaxes(title_text="Fraud Rate", row=1, col=2)
        fig.update_xaxes(title_text="Amount ($)", row=1, col=3)
        fig.update_xaxes(title_text="Ring Index", row=2, col=1)
        fig.update_yaxes(title_text="Ring Size", row=2, col=1)
        fig.update_xaxes(title_text="Node Rank", row=2, col=2)
        fig.update_yaxes(title_text="Degree", row=2, col=2)
        fig.update_xaxes(title_text="Hour of Day", row=2, col=3)
        fig.update_yaxes(title_text="Activity Count", row=2, col=3)
        
        if output_path:
            fig.write_html(output_path)
            logger.info(f"Saved fraud statistics dashboard to {output_path}")
            
        return fig
    
    def create_ring_heatmap(self, graph: nx.Graph,
                          fraud_rings: List[Set[str]],
                          output_path: str = None):
        """Create heatmap showing connections between fraud rings"""
        n_rings = len(fraud_rings)
        connection_matrix = np.zeros((n_rings, n_rings))
        
        # Calculate connections between rings
        for i in range(n_rings):
            for j in range(i+1, n_rings):
                ring1 = fraud_rings[i]
                ring2 = fraud_rings[j]
                
                # Count edges between rings
                connections = 0
                for node1 in ring1:
                    for node2 in ring2:
                        if graph.has_edge(node1, node2):
                            connections += 1
                
                connection_matrix[i, j] = connections
                connection_matrix[j, i] = connections
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(connection_matrix, annot=True, fmt='g', cmap='YlOrRd',
                   xticklabels=[f"Ring {i+1}" for i in range(n_rings)],
                   yticklabels=[f"Ring {i+1}" for i in range(n_rings)])
        
        plt.title("Connections Between Fraud Rings")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Saved ring heatmap to {output_path}")
        
        plt.close()
    
    def animate_fraud_propagation(self, graph: nx.Graph,
                                fraud_sequence: List[str],
                                output_path: str = "fraud_propagation.gif"):
        """Create animation showing fraud propagation over time"""
        import matplotlib.animation as animation
        
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(graph)
        
        def update(frame):
            ax.clear()
            
            # Determine which nodes are fraudulent up to this frame
            current_fraud = set(fraud_sequence[:frame+1])
            
            # Node colors
            node_colors = []
            for node in graph.nodes():
                if node in current_fraud:
                    node_colors.append(self.color_scheme['fraud'])
                else:
                    entity_type = graph.nodes[node].get('entity_type', 'unknown')
                    node_colors.append(self.color_scheme.get(entity_type, '#95a5a6'))
            
            # Draw graph
            nx.draw(graph, pos, node_color=node_colors, with_labels=False,
                   node_size=100, ax=ax)
            
            ax.set_title(f"Fraud Propagation - Step {frame+1}/{len(fraud_sequence)}")
            ax.axis('off')
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(fraud_sequence),
                                     interval=500, repeat=True)
        
        # Save animation
        anim.save(output_path, writer='pillow', fps=2)
        logger.info(f"Saved fraud propagation animation to {output_path}")
        
        plt.close()
    
    def export_for_gephi(self, graph: nx.Graph,
                       fraud_labels: Dict[str, bool],
                       fraud_rings: List[Set[str]],
                       output_path: str = "fraud_graph.gexf"):
        """Export graph in GEXF format for Gephi visualization"""
        # Create a copy to add attributes
        export_graph = graph.copy()
        
        # Add fraud labels
        nx.set_node_attributes(export_graph, fraud_labels, 'is_fraud')
        
        # Add ring membership
        ring_membership = {}
        for idx, ring in enumerate(fraud_rings):
            for node in ring:
                if node not in ring_membership:
                    ring_membership[node] = []
                ring_membership[node].append(idx)
        
        # Convert to string for GEXF
        ring_membership_str = {
            node: ','.join(map(str, rings)) 
            for node, rings in ring_membership.items()
        }
        nx.set_node_attributes(export_graph, ring_membership_str, 'fraud_rings')
        
        # Write GEXF file
        nx.write_gexf(export_graph, output_path)
        logger.info(f"Exported graph to {output_path} for Gephi")