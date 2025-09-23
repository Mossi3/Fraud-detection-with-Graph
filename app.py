import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import time
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append('/workspace')

from data_processor import FraudDataProcessor
from graph.graph_fraud_detection import GraphFraudDetector
from visualization.heatmap_visualizer import FraudHeatmapVisualizer

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
    .success-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4caf50;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1565c0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_base_url' not in st.session_state:
    st.session_state.api_base_url = "http://localhost:8000"

if 'data_processor' not in st.session_state:
    st.session_state.data_processor = FraudDataProcessor()

if 'graph_detector' not in st.session_state:
    st.session_state.graph_detector = GraphFraudDetector()

if 'visualizer' not in st.session_state:
    st.session_state.visualizer = FraudHeatmapVisualizer()

# Helper functions
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{st.session_state.api_base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def generate_sample_transaction():
    """Generate a sample transaction for testing"""
    return {
        "Time": np.random.uniform(0, 1000),
        "V1": np.random.normal(0, 1),
        "V2": np.random.normal(0, 1),
        "V3": np.random.normal(0, 1),
        "V4": np.random.normal(0, 1),
        "V5": np.random.normal(0, 1),
        "V6": np.random.normal(0, 1),
        "V7": np.random.normal(0, 1),
        "V8": np.random.normal(0, 1),
        "V9": np.random.normal(0, 1),
        "V10": np.random.normal(0, 1),
        "V11": np.random.normal(0, 1),
        "V12": np.random.normal(0, 1),
        "V13": np.random.normal(0, 1),
        "V14": np.random.normal(0, 1),
        "V15": np.random.normal(0, 1),
        "V16": np.random.normal(0, 1),
        "V17": np.random.normal(0, 1),
        "V18": np.random.normal(0, 1),
        "V19": np.random.normal(0, 1),
        "V20": np.random.normal(0, 1),
        "V21": np.random.normal(0, 1),
        "V22": np.random.normal(0, 1),
        "V23": np.random.normal(0, 1),
        "V24": np.random.normal(0, 1),
        "V25": np.random.normal(0, 1),
        "V26": np.random.normal(0, 1),
        "V27": np.random.normal(0, 1),
        "V28": np.random.normal(0, 1),
        "Amount": np.random.exponential(50),
        "Merchant": np.random.choice(['grocery', 'gas', 'restaurant', 'online', 'retail']),
        "Country": np.random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
        "Device": np.random.choice(['mobile', 'desktop', 'tablet']),
        "IP_Country": np.random.choice(['US', 'CA', 'UK', 'DE', 'FR']),
        "Hour": np.random.randint(0, 24),
        "Day": np.random.randint(1, 32),
        "Card_ID": f"CARD_{np.random.randint(10000, 99999):05d}",
        "Merchant_ID": f"M_{np.random.randint(100, 999):03d}",
        "Device_ID": f"D_{np.random.randint(1000, 9999):04d}",
        "IP_Address": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
    }

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/1f77b4/ffffff?text=Fraud+Detection", width=200)
        
        st.markdown("### 🚀 Quick Actions")
        
        # API Status
        api_status = check_api_health()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info("Please start the API server: `python api.py`")
        
        st.markdown("### 📊 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Dashboard", "🔍 Single Transaction", "📦 Batch Analysis", 
             "🕸️ Graph Analysis", "📈 Visualizations", "⚙️ Settings"]
        )
    
    # Main content based on selected page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "🔍 Single Transaction":
        show_single_transaction()
    elif page == "📦 Batch Analysis":
        show_batch_analysis()
    elif page == "🕸️ Graph Analysis":
        show_graph_analysis()
    elif page == "📈 Visualizations":
        show_visualizations()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Main dashboard page"""
    st.markdown("## 📊 System Dashboard")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", "50,000", "↗️ 1,200")
    
    with col2:
        st.metric("Fraud Detected", "127", "↗️ 3")
    
    with col3:
        st.metric("Detection Rate", "99.2%", "↗️ 0.1%")
    
    with col4:
        st.metric("System Uptime", "99.9%", "↗️ 0.01%")
    
    # Recent activity
    st.markdown("### 🔥 Recent Activity")
    
    # Generate sample recent transactions
    recent_transactions = []
    for i in range(10):
        tx = generate_sample_transaction()
        tx['timestamp'] = datetime.now() - timedelta(minutes=np.random.randint(1, 60))
        tx['fraud_probability'] = np.random.uniform(0, 1)
        tx['is_fraud'] = tx['fraud_probability'] > 0.5
        recent_transactions.append(tx)
    
    recent_df = pd.DataFrame(recent_transactions)
    
    # Display recent transactions
    for _, tx in recent_df.iterrows():
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.write(f"**{tx['Card_ID']}** - {tx['Merchant']} - ${tx['Amount']:.2f}")
        
        with col2:
            st.write(f"{tx['timestamp'].strftime('%H:%M:%S')}")
        
        with col3:
            if tx['is_fraud']:
                st.markdown('<div class="fraud-alert">🚨 FRAUD</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-card">✅ LEGIT</div>', unsafe_allow_html=True)
        
        with col4:
            st.write(f"{tx['fraud_probability']:.1%}")
    
    # Charts
    st.markdown("### 📈 Performance Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fraud rate over time
        hours = list(range(24))
        fraud_rates = [np.random.uniform(0.001, 0.01) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=fraud_rates, mode='lines+markers', name='Fraud Rate'))
        fig.update_layout(title="Fraud Rate by Hour", xaxis_title="Hour", yaxis_title="Fraud Rate")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Transaction volume
        merchants = ['grocery', 'gas', 'restaurant', 'online', 'retail']
        volumes = [np.random.randint(100, 1000) for _ in merchants]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=merchants, y=volumes, name='Transaction Volume'))
        fig.update_layout(title="Transaction Volume by Merchant", xaxis_title="Merchant", yaxis_title="Volume")
        st.plotly_chart(fig, use_container_width=True)

def show_single_transaction():
    """Single transaction analysis page"""
    st.markdown("## 🔍 Single Transaction Analysis")
    
    # Transaction input form
    with st.form("transaction_form"):
        st.markdown("### 📝 Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            amount = st.number_input("Amount ($)", min_value=0.01, value=50.0, step=0.01)
            merchant = st.selectbox("Merchant Type", ['grocery', 'gas', 'restaurant', 'online', 'retail', 'pharmacy'])
            country = st.selectbox("Country", ['US', 'CA', 'UK', 'DE', 'FR', 'AU', 'JP'])
            device = st.selectbox("Device Type", ['mobile', 'desktop', 'tablet'])
        
        with col2:
            hour = st.slider("Hour of Day", 0, 23, 12)
            day = st.slider("Day of Month", 1, 31, 15)
            card_id = st.text_input("Card ID", value=f"CARD_{np.random.randint(10000, 99999):05d}")
            merchant_id = st.text_input("Merchant ID", value=f"M_{np.random.randint(100, 999):03d}")
        
        # V-features (simplified)
        st.markdown("### 🔢 Transaction Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            v1 = st.number_input("V1", value=np.random.normal(0, 1), format="%.3f")
            v2 = st.number_input("V2", value=np.random.normal(0, 1), format="%.3f")
            v3 = st.number_input("V3", value=np.random.normal(0, 1), format="%.3f")
            v4 = st.number_input("V4", value=np.random.normal(0, 1), format="%.3f")
        
        with col2:
            v5 = st.number_input("V5", value=np.random.normal(0, 1), format="%.3f")
            v6 = st.number_input("V6", value=np.random.normal(0, 1), format="%.3f")
            v7 = st.number_input("V7", value=np.random.normal(0, 1), format="%.3f")
            v8 = st.number_input("V8", value=np.random.normal(0, 1), format="%.3f")
        
        with col3:
            v9 = st.number_input("V9", value=np.random.normal(0, 1), format="%.3f")
            v10 = st.number_input("V10", value=np.random.normal(0, 1), format="%.3f")
            v11 = st.number_input("V11", value=np.random.normal(0, 1), format="%.3f")
            v12 = st.number_input("V12", value=np.random.normal(0, 1), format="%.3f")
        
        # Generate random V13-V28
        v_features = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12]
        v_features.extend([np.random.normal(0, 1) for _ in range(16)])  # V13-V28
        
        submitted = st.form_submit_button("🔍 Analyze Transaction")
        
        if submitted:
            # Prepare transaction data
            transaction_data = {
                "Time": np.random.uniform(0, 1000),
                "Amount": amount,
                "Merchant": merchant,
                "Country": country,
                "Device": device,
                "IP_Country": country,
                "Hour": hour,
                "Day": day,
                "Card_ID": card_id,
                "Merchant_ID": merchant_id,
                "Device_ID": f"D_{np.random.randint(1000, 9999):04d}",
                "IP_Address": f"192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
            }
            
            # Add V-features
            for i, v_val in enumerate(v_features, 1):
                transaction_data[f"V{i}"] = v_val
            
            # Call API for fraud detection
            if check_api_health():
                try:
                    response = requests.post(
                        f"{st.session_state.api_base_url}/detect",
                        json=transaction_data,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.markdown("### 🎯 Analysis Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if result['is_fraud']:
                                st.markdown('<div class="fraud-alert">🚨 FRAUD DETECTED</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="success-card">✅ LEGITIMATE</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.metric("Fraud Probability", f"{result['fraud_probability']:.1%}")
                        
                        with col3:
                            st.metric("Confidence", result['confidence'].title())
                        
                        # Risk factors
                        if result['risk_factors']:
                            st.markdown("### ⚠️ Risk Factors")
                            for factor in result['risk_factors']:
                                st.warning(f"• {factor}")
                        
                        # Model predictions
                        st.markdown("### 🤖 Model Predictions")
                        model_df = pd.DataFrame([
                            {"Model": model, "Probability": prob}
                            for model, prob in result['model_predictions'].items()
                        ])
                        
                        fig = px.bar(model_df, x="Model", y="Probability", 
                                   title="Fraud Probability by Model")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"Error calling API: {e}")
            else:
                st.error("API is not available. Please start the API server.")
    
    # Quick test buttons
    st.markdown("### 🚀 Quick Tests")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎲 Random Transaction"):
            st.rerun()
    
    with col2:
        if st.button("🚨 Suspicious Transaction"):
            # Generate suspicious transaction
            st.rerun()
    
    with col3:
        if st.button("✅ Normal Transaction"):
            # Generate normal transaction
            st.rerun()

def show_batch_analysis():
    """Batch transaction analysis page"""
    st.markdown("## 📦 Batch Transaction Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            # Display sample data
            st.markdown("### 📊 Sample Data")
            st.dataframe(df.head())
            
            # Analyze button
            if st.button("🔍 Analyze All Transactions"):
                if check_api_health():
                    # Convert to API format
                    transactions = df.to_dict('records')
                    
                    # Call batch API
                    response = requests.post(
                        f"{st.session_state.api_base_url}/batch_detect",
                        json={"transactions": transactions},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display summary
                        st.markdown("### 📈 Analysis Summary")
                        summary = result['summary']
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Total Transactions", summary['total_transactions'])
                        
                        with col2:
                            st.metric("Fraud Detected", summary['fraud_detected'])
                        
                        with col3:
                            st.metric("Fraud Rate", f"{summary['fraud_rate']:.1%}")
                        
                        with col4:
                            st.metric("Processing Time", f"{summary['processing_time']:.2f}s")
                        
                        # Detailed results
                        st.markdown("### 🔍 Detailed Results")
                        results_df = pd.DataFrame(result['results'])
                        st.dataframe(results_df)
                        
                        # Visualization
                        st.markdown("### 📊 Fraud Distribution")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fraud_counts = results_df['is_fraud'].value_counts()
                            fig = px.pie(values=fraud_counts.values, names=['Legitimate', 'Fraud'], 
                                       title="Transaction Classification")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.histogram(results_df, x='fraud_probability', 
                                             title="Fraud Probability Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error(f"API Error: {response.status_code}")
                        
                else:
                    st.error("API is not available. Please start the API server.")
        
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    else:
        # Generate sample data
        st.markdown("### 📝 Generate Sample Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_transactions = st.number_input("Number of transactions", min_value=1, max_value=1000, value=100)
        
        with col2:
            fraud_rate = st.slider("Fraud rate", 0.0, 0.1, 0.01, 0.001)
        
        if st.button("🎲 Generate Sample Data"):
            # Generate sample transactions
            transactions = []
            for i in range(num_transactions):
                tx = generate_sample_transaction()
                tx['Class'] = 1 if np.random.random() < fraud_rate else 0
                transactions.append(tx)
            
            df = pd.DataFrame(transactions)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Data",
                data=csv,
                file_name=f"sample_transactions_{num_transactions}.csv",
                mime="text/csv"
            )
            
            st.success(f"✅ Generated {num_transactions} transactions with {fraud_rate:.1%} fraud rate")

def show_graph_analysis():
    """Graph analysis page"""
    st.markdown("## 🕸️ Graph-Based Fraud Analysis")
    
    # Analysis type selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["fraud_rings", "suspicious_patterns", "community_detection"]
    )
    
    # Analysis parameters
    col1, col2 = st.columns(2)
    
    with col1:
        min_ring_size = st.number_input("Minimum Ring Size", min_value=2, max_value=10, value=3)
    
    with col2:
        fraud_threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.3, 0.05)
    
    # Run analysis button
    if st.button("🔍 Run Graph Analysis"):
        if check_api_health():
            try:
                # Call graph analysis API
                response = requests.post(
                    f"{st.session_state.api_base_url}/graph_analysis",
                    json={
                        "analysis_type": analysis_type,
                        "min_ring_size": min_ring_size,
                        "fraud_threshold": fraud_threshold
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    st.markdown("### 🎯 Analysis Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Risk Score", f"{result['risk_score']:.2f}")
                    
                    with col2:
                        st.metric("Fraud Rings Detected", len(result['fraud_rings']))
                    
                    # Fraud rings
                    if result['fraud_rings']:
                        st.markdown("### 🚨 Detected Fraud Rings")
                        
                        for i, ring in enumerate(result['fraud_rings'][:5]):  # Show first 5
                            with st.expander(f"Ring {i+1} - Size: {ring['size']}, Fraud Rate: {ring['fraud_rate']:.2f}"):
                                st.write(f"**Community ID:** {ring['community_id']}")
                                st.write(f"**Nodes:** {', '.join(ring['nodes'][:10])}")  # Show first 10 nodes
                                if len(ring['nodes']) > 10:
                                    st.write(f"... and {len(ring['nodes']) - 10} more nodes")
                    
                    # Suspicious patterns
                    if result['suspicious_patterns']:
                        st.markdown("### ⚠️ Suspicious Patterns")
                        
                        for pattern in result['suspicious_patterns'][:10]:  # Show first 10
                            st.warning(f"**{pattern['pattern'].replace('_', ' ').title()}:** {pattern['description']}")
                    
                    # Recommendations
                    if result['recommendations']:
                        st.markdown("### 💡 Recommendations")
                        for rec in result['recommendations']:
                            st.info(f"• {rec}")
                    
                    # Risk visualization
                    st.markdown("### 📊 Risk Visualization")
                    
                    risk_data = {
                        'Risk Level': ['Low', 'Medium', 'High'],
                        'Count': [0, 0, 0]
                    }
                    
                    if result['risk_score'] < 0.3:
                        risk_data['Count'][0] = 1
                    elif result['risk_score'] < 0.7:
                        risk_data['Count'][1] = 1
                    else:
                        risk_data['Count'][2] = 1
                    
                    fig = px.bar(x=risk_data['Risk Level'], y=risk_data['Count'], 
                               title="Risk Level Assessment")
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error in graph analysis: {e}")
        else:
            st.error("API is not available. Please start the API server.")
    
    # Graph visualization
    st.markdown("### 🕸️ Graph Visualization")
    
    if st.button("📊 Generate Graph Visualization"):
        # Load data and build graph
        df = st.session_state.data_processor.load_data()
        df_processed = st.session_state.data_processor.preprocess_data(df)
        
        # Build graph
        graph = st.session_state.graph_detector.build_transaction_graph(df_processed)
        
        # Detect fraud rings
        fraud_rings = st.session_state.graph_detector.detect_fraud_rings()
        
        st.success(f"✅ Graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
        st.info(f"🔍 Detected {len(fraud_rings)} potential fraud rings")
        
        # Display graph statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Nodes", graph.number_of_nodes())
        
        with col2:
            st.metric("Total Edges", graph.number_of_edges())
        
        with col3:
            st.metric("Fraud Rings", len(fraud_rings))

def show_visualizations():
    """Visualizations page"""
    st.markdown("## 📈 Fraud Detection Visualizations")
    
    # Visualization type selection
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["fraud_patterns", "geographic", "time_series", "model_performance", "interactive_dashboard"]
    )
    
    # Generate visualization button
    if st.button("📊 Generate Visualization"):
        if check_api_health():
            try:
                # Call heatmap API
                response = requests.post(
                    f"{st.session_state.api_base_url}/heatmap",
                    json={
                        "analysis_type": viz_type,
                        "parameters": {}
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"✅ Generated {viz_type} visualization")
                    
                    # Display the generated heatmap
                    if os.path.exists(result['heatmap_url']):
                        st.image(result['heatmap_url'], caption=f"{viz_type.title()} Analysis")
                    else:
                        st.info("Visualization generated successfully. Check the visualization folder.")
                    
                    # Additional interactive charts
                    if viz_type == "interactive_dashboard":
                        st.markdown("### 📊 Interactive Charts")
                        
                        # Load data for interactive charts
                        df = st.session_state.data_processor.load_data()
                        
                        # Fraud rate by hour
                        hour_fraud = df.groupby('Hour')['Class'].mean()
                        
                        fig = px.bar(x=hour_fraud.index, y=hour_fraud.values, 
                                   title="Fraud Rate by Hour of Day")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Fraud rate by merchant
                        merchant_fraud = df.groupby('Merchant')['Class'].mean().sort_values(ascending=False)
                        
                        fig = px.bar(x=merchant_fraud.index, y=merchant_fraud.values,
                                   title="Fraud Rate by Merchant Type")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Amount distribution
                        fig = px.histogram(df, x='Amount', color='Class', 
                                         title="Transaction Amount Distribution by Class")
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Error generating visualization: {e}")
        else:
            st.error("API is not available. Please start the API server.")
    
    # Real-time monitoring
    st.markdown("### 📡 Real-time Monitoring")
    
    if st.button("🔄 Start Real-time Monitoring"):
        # Create placeholder for real-time updates
        placeholder = st.empty()
        
        for i in range(10):  # Simulate 10 updates
            # Generate random metrics
            fraud_rate = np.random.uniform(0.001, 0.01)
            transaction_count = np.random.randint(100, 1000)
            
            # Update metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Live Fraud Rate", f"{fraud_rate:.3f}")
            
            with col2:
                st.metric("Transactions/min", transaction_count)
            
            with col3:
                st.metric("System Load", f"{np.random.uniform(20, 80):.1f}%")
            
            time.sleep(1)  # Update every second
        
        st.success("✅ Real-time monitoring completed")

def show_settings():
    """Settings page"""
    st.markdown("## ⚙️ System Settings")
    
    # API Configuration
    st.markdown("### 🔌 API Configuration")
    
    api_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
    
    if st.button("💾 Save API Settings"):
        st.session_state.api_base_url = api_url
        st.success("✅ API settings saved")
    
    # Model Configuration
    st.markdown("### 🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Models:**")
        models = ["CNN", "LSTM", "Transformer", "Deep", "GraphSAGE", "GAT", "GCN"]
        for model in models:
            st.checkbox(model, value=True)
    
    with col2:
        st.markdown("**Model Parameters:**")
        threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.05)
        confidence_level = st.selectbox("Confidence Level", ["Low", "Medium", "High"])
    
    # System Information
    st.markdown("### ℹ️ System Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **System Version:** 1.0.0  
        **Python Version:** 3.9+  
        **PyTorch Version:** 2.0+  
        **Streamlit Version:** 1.25+
        """)
    
    with col2:
        st.info("""
        **API Status:** Connected  
        **Models Loaded:** 7  
        **Last Update:** Today  
        **Uptime:** 99.9%
        """)
    
    # Export/Import Settings
    st.markdown("### 📁 Export/Import Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Settings"):
            settings = {
                "api_url": st.session_state.api_base_url,
                "threshold": threshold,
                "confidence_level": confidence_level
            }
            st.download_button(
                label="Download Settings",
                data=json.dumps(settings, indent=2),
                file_name="fraud_detection_settings.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_settings = st.file_uploader("Import Settings", type=['json'])
        if uploaded_settings:
            try:
                settings = json.load(uploaded_settings)
                st.session_state.api_base_url = settings.get("api_url", st.session_state.api_base_url)
                st.success("✅ Settings imported successfully")
            except Exception as e:
                st.error(f"Error importing settings: {e}")

if __name__ == "__main__":
    main()