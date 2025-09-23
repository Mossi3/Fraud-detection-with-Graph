"""
Interactive visualization of fraud rings and graph structures.
Supports multiple visualization backends: Plotly, Dash, NetworkX, and Cytoscape.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from typing import Dict, List, Tuple, Optional, Any
import json
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudRingVisualizer:
    """Advanced visualization for fraud rings and graph structures."""
    
    def __init__(self, graph: nx.Graph, detected_rings: Dict[str, Dict]):
        self.graph = graph
        self.detected_rings = detected_rings
        self.node_positions = None
        self.color_palette = px.colors.qualitative.Set3
    
    def create_ring_overview_plot(self) -> go.Figure:
        """Create overview plot of all detected fraud rings."""
        if not self.detected_rings:
            return go.Figure().add_annotation(text="No fraud rings detected", 
                                            xref="paper", yref="paper", x=0.5, y=0.5)
        
        # Prepare data
        ring_data = []
        for ring_id, ring_info in self.detected_rings.items():
            ring_data.append({
                'ring_id': ring_id,
                'size': ring_info['size'],
                'fraud_score': ring_info['fraud_score'],
                'method': ring_info['method'],
                'confidence': ring_info.get('confidence', 1.0)
            })
        
        df = pd.DataFrame(ring_data)
        
        # Create subplot with multiple views
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ring Size Distribution', 'Fraud Score vs Size', 
                          'Detection Method Distribution', 'Ring Confidence'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "bar"}]]
        )
        
        # Ring size distribution
        fig.add_trace(
            go.Histogram(x=df['size'], name='Size Distribution', nbinsx=20),
            row=1, col=1
        )
        
        # Fraud score vs size scatter
        fig.add_trace(
            go.Scatter(
                x=df['size'], y=df['fraud_score'],
                mode='markers+text',
                text=df['ring_id'],
                textposition="top center",
                marker=dict(size=df['confidence']*20, opacity=0.7),
                name='Rings'
            ),
            row=1, col=2
        )
        
        # Detection method pie chart
        method_counts = df['method'].value_counts()
        fig.add_trace(
            go.Pie(labels=method_counts.index, values=method_counts.values, name='Methods'),
            row=2, col=1
        )
        
        # Ring confidence bar chart
        fig.add_trace(
            go.Bar(x=df['ring_id'], y=df['confidence'], name='Confidence'),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Fraud Ring Detection Overview",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def create_ring_network_plot(self, ring_id: str, layout: str = 'spring') -> go.Figure:
        """Create detailed network plot for a specific fraud ring."""
        if ring_id not in self.detected_rings:
            return go.Figure().add_annotation(text=f"Ring {ring_id} not found")
        
        ring_nodes = self.detected_rings[ring_id]['nodes']
        
        # Extract subgraph for the ring
        subgraph = self.graph.subgraph(ring_nodes).copy()
        
        # Add neighboring nodes for context
        extended_nodes = set(ring_nodes)
        for node in ring_nodes:
            if node in self.graph:
                neighbors = list(self.graph.neighbors(node))[:3]  # Limit neighbors
                extended_nodes.update(neighbors)
        
        extended_subgraph = self.graph.subgraph(extended_nodes)
        
        # Calculate layout
        if layout == 'spring':
            pos = nx.spring_layout(extended_subgraph, k=1, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(extended_subgraph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(extended_subgraph)
        else:
            pos = nx.random_layout(extended_subgraph)
        
        # Create edge traces
        edge_x, edge_y = [], []
        edge_info = []
        
        for edge in extended_subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = extended_subgraph.edges[edge]
            edge_info.append(f"{edge[0]} -> {edge[1]}: {edge_data.get('edge_type', 'unknown')}")
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node traces
        node_x = [pos[node][0] for node in extended_subgraph.nodes()]
        node_y = [pos[node][1] for node in extended_subgraph.nodes()]
        
        # Color nodes based on type and fraud status
        node_colors = []
        node_text = []
        node_sizes = []
        
        for node in extended_subgraph.nodes():
            node_data = extended_subgraph.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            
            # Determine color based on type and ring membership
            if node in ring_nodes:
                if 'transaction' in node:
                    color = 'red'  # Fraud transaction
                elif 'card' in node:
                    color = 'orange'  # Compromised card
                elif 'merchant' in node:
                    color = 'purple'  # Suspicious merchant
                elif 'device' in node:
                    color = 'brown'  # Compromised device
                elif 'ip' in node:
                    color = 'pink'  # Suspicious IP
                else:
                    color = 'red'
            else:
                color = 'lightblue'  # Context nodes
            
            node_colors.append(color)
            node_text.append(f"{node}<br>Type: {node_type}")
            node_sizes.append(15 if node in ring_nodes else 8)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node.split('_')[-1] for node in extended_subgraph.nodes()],
            textposition="middle center",
            hovertext=node_text,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Fraud Ring: {ring_id}',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text=f"Ring Size: {len(ring_nodes)}, Fraud Score: {self.detected_rings[ring_id]['fraud_score']:.2f}",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color='black', size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                       )
        
        return fig
    
    def create_cytoscape_elements(self, ring_id: Optional[str] = None) -> List[Dict]:
        """Create Cytoscape elements for interactive graph visualization."""
        elements = []
        
        if ring_id and ring_id in self.detected_rings:
            # Focus on specific ring
            ring_nodes = self.detected_rings[ring_id]['nodes']
            subgraph = self.graph.subgraph(ring_nodes)
        else:
            # Show entire graph (limited for performance)
            subgraph = self.graph
            if len(subgraph.nodes()) > 500:  # Limit for performance
                subgraph = self.graph.subgraph(list(self.graph.nodes())[:500])
        
        # Add nodes
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            
            # Determine if node is in any fraud ring
            in_ring = any(node in ring_data['nodes'] for ring_data in self.detected_rings.values())
            
            elements.append({
                'data': {
                    'id': node,
                    'label': node.split('_')[-1],
                    'node_type': node_type,
                    'in_ring': in_ring
                },
                'classes': f"{node_type} {'fraud-ring' if in_ring else 'normal'}"
            })
        
        # Add edges
        for edge in subgraph.edges():
            edge_data = subgraph.edges[edge]
            elements.append({
                'data': {
                    'source': edge[0],
                    'target': edge[1],
                    'edge_type': edge_data.get('edge_type', 'unknown')
                }
            })
        
        return elements
    
    def create_dash_app(self, port: int = 8050) -> dash.Dash:
        """Create interactive Dash application for fraud ring exploration."""
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Prepare ring options for dropdown
        ring_options = [{'label': 'All Rings', 'value': 'all'}]
        ring_options.extend([
            {'label': f"{ring_id} (Size: {data['size']}, Score: {data['fraud_score']:.2f})", 
             'value': ring_id}
            for ring_id, data in self.detected_rings.items()
        ])
        
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Fraud Ring Detection Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Ring Selection"),
                    dcc.Dropdown(
                        id='ring-selector',
                        options=ring_options,
                        value='all',
                        clearable=False
                    ),
                    html.Br(),
                    
                    html.H3("Layout Options"),
                    dcc.Dropdown(
                        id='layout-selector',
                        options=[
                            {'label': 'Spring', 'value': 'spring'},
                            {'label': 'Circular', 'value': 'circular'},
                            {'label': 'Kamada-Kawai', 'value': 'kamada_kawai'},
                            {'label': 'Random', 'value': 'random'}
                        ],
                        value='spring',
                        clearable=False
                    )
                ], width=3),
                
                dbc.Col([
                    dcc.Graph(id='ring-network-plot', style={'height': '600px'})
                ], width=9)
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Ring Overview"),
                    dcc.Graph(id='ring-overview-plot', figure=self.create_ring_overview_plot())
                ])
            ]),
            
            html.Hr(),
            
            dbc.Row([
                dbc.Col([
                    html.H3("Interactive Network (Cytoscape)"),
                    cyto.Cytoscape(
                        id='cytoscape-graph',
                        elements=self.create_cytoscape_elements(),
                        style={'width': '100%', 'height': '500px'},
                        layout={'name': 'cose'},
                        stylesheet=[
                            {
                                'selector': 'node',
                                'style': {
                                    'content': 'data(label)',
                                    'text-valign': 'center',
                                    'color': 'white',
                                    'text-outline-width': 2,
                                    'text-outline-color': 'black',
                                    'font-size': '10px'
                                }
                            },
                            {
                                'selector': '.transaction',
                                'style': {
                                    'background-color': 'blue',
                                    'shape': 'rectangle'
                                }
                            },
                            {
                                'selector': '.card',
                                'style': {
                                    'background-color': 'green',
                                    'shape': 'ellipse'
                                }
                            },
                            {
                                'selector': '.merchant',
                                'style': {
                                    'background-color': 'orange',
                                    'shape': 'diamond'
                                }
                            },
                            {
                                'selector': '.device',
                                'style': {
                                    'background-color': 'purple',
                                    'shape': 'triangle'
                                }
                            },
                            {
                                'selector': '.ip',
                                'style': {
                                    'background-color': 'brown',
                                    'shape': 'pentagon'
                                }
                            },
                            {
                                'selector': '.fraud-ring',
                                'style': {
                                    'border-width': 3,
                                    'border-color': 'red'
                                }
                            },
                            {
                                'selector': 'edge',
                                'style': {
                                    'curve-style': 'bezier',
                                    'target-arrow-shape': 'triangle',
                                    'line-color': '#ccc',
                                    'target-arrow-color': '#ccc'
                                }
                            }
                        ]
                    )
                ])
            ])
        ], fluid=True)
        
        # Callbacks
        @app.callback(
            Output('ring-network-plot', 'figure'),
            [Input('ring-selector', 'value'),
             Input('layout-selector', 'value')]
        )
        def update_network_plot(selected_ring, layout):
            if selected_ring == 'all':
                return self.create_ring_overview_plot()
            else:
                return self.create_ring_network_plot(selected_ring, layout)
        
        @app.callback(
            Output('cytoscape-graph', 'elements'),
            [Input('ring-selector', 'value')]
        )
        def update_cytoscape(selected_ring):
            if selected_ring == 'all':
                return self.create_cytoscape_elements()
            else:
                return self.create_cytoscape_elements(selected_ring)
        
        return app
    
    def create_static_report(self, output_path: str = 'fraud_rings_report.html'):
        """Create static HTML report of all fraud rings."""
        from plotly.offline import plot
        
        # Create all plots
        overview_fig = self.create_ring_overview_plot()
        
        ring_plots = {}
        for ring_id in self.detected_rings.keys():
            ring_plots[ring_id] = self.create_ring_network_plot(ring_id)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fraud Ring Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .ring-section {{ margin: 30px 0; border: 1px solid #ccc; padding: 20px; }}
                .ring-header {{ background-color: #f0f0f0; padding: 10px; margin: -20px -20px 20px -20px; }}
                .stats {{ display: flex; justify-content: space-around; margin: 10px 0; }}
                .stat {{ text-align: center; }}
            </style>
        </head>
        <body>
            <h1>Fraud Ring Detection Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Overview</h2>
            {plot(overview_fig, output_type='div', include_plotlyjs=True)}
            
            <h2>Individual Fraud Rings</h2>
        """
        
        for ring_id, ring_data in self.detected_rings.items():
            html_content += f"""
            <div class="ring-section">
                <div class="ring-header">
                    <h3>{ring_id}</h3>
                    <div class="stats">
                        <div class="stat">
                            <strong>Size</strong><br>{ring_data['size']} nodes
                        </div>
                        <div class="stat">
                            <strong>Fraud Score</strong><br>{ring_data['fraud_score']:.2f}
                        </div>
                        <div class="stat">
                            <strong>Method</strong><br>{ring_data['method']}
                        </div>
                        <div class="stat">
                            <strong>Confidence</strong><br>{ring_data.get('confidence', 1.0):.2f}
                        </div>
                    </div>
                </div>
                {plot(ring_plots[ring_id], output_type='div', include_plotlyjs=False)}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Static report saved to {output_path}")

def main():
    """Example usage of fraud ring visualizer."""
    import argparse
    from ..graph.graph_builder import HeterogeneousGraphBuilder
    from ..graph.community_detection import FraudRingDetector
    
    parser = argparse.ArgumentParser(description="Visualize fraud rings")
    parser.add_argument('--data', type=str, required=True, help='Transaction data CSV')
    parser.add_argument('--rings', type=str, help='Detected rings JSON file')
    parser.add_argument('--mode', type=str, default='dash', 
                       choices=['dash', 'static', 'both'])
    parser.add_argument('--port', type=int, default=8050, help='Dash app port')
    parser.add_argument('--output', type=str, default='fraud_rings_report.html')
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.data)
    
    # Build graph
    builder = HeterogeneousGraphBuilder()
    nx_graph = builder.build_graph(df, graph_type='networkx')
    
    # Load or detect rings
    if args.rings:
        with open(args.rings, 'r') as f:
            detected_rings = json.load(f)
    else:
        # Detect rings
        detector = FraudRingDetector()
        detected_rings = detector.detect_rings_louvain(nx_graph)
    
    # Create visualizer
    viz = FraudRingVisualizer(nx_graph, detected_rings)
    
    if args.mode in ['dash', 'both']:
        # Create and run Dash app
        app = viz.create_dash_app(port=args.port)
        logger.info(f"Starting Dash app on port {args.port}")
        app.run_server(debug=True, port=args.port)
    
    if args.mode in ['static', 'both']:
        # Create static report
        viz.create_static_report(args.output)

if __name__ == "__main__":
    main()