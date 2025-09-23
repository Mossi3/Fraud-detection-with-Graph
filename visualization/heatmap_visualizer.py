import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import networkx as nx
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FraudHeatmapVisualizer:
    """Advanced heatmap visualization for fraud detection"""
    
    def __init__(self):
        self.colors = {
            'fraud': '#FF4444',
            'legitimate': '#44AA44',
            'suspicious': '#FFAA44',
            'background': '#F0F0F0'
        }
    
    def create_fraud_pattern_heatmap(self, df, title="Fraud Pattern Analysis"):
        """Create comprehensive fraud pattern heatmap"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Hour vs Day heatmap
        pivot_hour_day = df.groupby(['Hour', 'Day'])['Class'].mean().unstack(fill_value=0)
        sns.heatmap(pivot_hour_day, ax=axes[0,0], cmap='Reds', cbar=True)
        axes[0,0].set_title('Fraud Rate by Hour and Day')
        axes[0,0].set_xlabel('Day of Month')
        axes[0,0].set_ylabel('Hour of Day')
        
        # 2. Amount vs Merchant heatmap
        merchant_fraud = df.groupby('Merchant')['Class'].agg(['mean', 'count']).reset_index()
        merchant_fraud = merchant_fraud[merchant_fraud['count'] >= 10]  # Filter for meaningful data
        
        amount_bins = pd.cut(df['Amount'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        merchant_amount_fraud = df.groupby(['Merchant', amount_bins])['Class'].mean().unstack(fill_value=0)
        
        sns.heatmap(merchant_amount_fraud, ax=axes[0,1], cmap='Reds', cbar=True)
        axes[0,1].set_title('Fraud Rate by Merchant and Amount')
        axes[0,1].set_xlabel('Amount Range')
        axes[0,1].set_ylabel('Merchant Type')
        
        # 3. Country vs Device heatmap
        country_device_fraud = df.groupby(['Country', 'Device'])['Class'].mean().unstack(fill_value=0)
        sns.heatmap(country_device_fraud, ax=axes[0,2], cmap='Reds', cbar=True)
        axes[0,2].set_title('Fraud Rate by Country and Device')
        axes[0,2].set_xlabel('Device Type')
        axes[0,2].set_ylabel('Country')
        
        # 4. V-features correlation heatmap
        v_features = [f'V{i}' for i in range(1, 15)]  # First 14 V features
        v_corr = df[v_features + ['Class']].corr()
        sns.heatmap(v_corr, ax=axes[1,0], cmap='RdBu_r', center=0, cbar=True)
        axes[1,0].set_title('V-Features Correlation with Fraud')
        
        # 5. Transaction frequency heatmap
        df['Transaction_Count'] = df.groupby('Card_ID')['Card_ID'].transform('count')
        df['Amount_Category'] = pd.cut(df['Amount'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        freq_fraud = df.groupby(['Transaction_Count', 'Amount_Category'])['Class'].mean().unstack(fill_value=0)
        sns.heatmap(freq_fraud, ax=axes[1,1], cmap='Reds', cbar=True)
        axes[1,1].set_title('Fraud Rate by Transaction Frequency and Amount')
        axes[1,1].set_xlabel('Amount Category')
        axes[1,1].set_ylabel('Transaction Count')
        
        # 6. Time-based fraud pattern
        df['Time_Hour'] = df['Time'] % 24
        df['Time_Day'] = df['Time'] // 24
        time_fraud = df.groupby(['Time_Hour', 'Time_Day'])['Class'].mean().unstack(fill_value=0)
        sns.heatmap(time_fraud, ax=axes[1,2], cmap='Reds', cbar=True)
        axes[1,2].set_title('Fraud Rate Over Time')
        axes[1,2].set_xlabel('Day')
        axes[1,2].set_ylabel('Hour')
        
        plt.tight_layout()
        plt.savefig('/workspace/visualization/fraud_pattern_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_performance_heatmap(self, model_results, title="Model Performance Comparison"):
        """Create heatmap comparing model performance"""
        # Prepare data for heatmap
        metrics = ['AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy']
        models = list(model_results.keys())
        
        performance_matrix = np.zeros((len(models), len(metrics)))
        
        for i, model in enumerate(models):
            result = model_results[model]
            performance_matrix[i, 0] = result.get('auc', 0)
            performance_matrix[i, 1] = result.get('precision', 0)
            performance_matrix[i, 2] = result.get('recall', 0)
            performance_matrix[i, 3] = result.get('f1', 0)
            performance_matrix[i, 4] = result.get('accuracy', 0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(performance_matrix, 
                   xticklabels=metrics, 
                   yticklabels=models,
                   annot=True, 
                   fmt='.3f',
                   cmap='RdYlGn',
                   vmin=0, vmax=1)
        plt.title(title)
        plt.xlabel('Metrics')
        plt.ylabel('Models')
        plt.tight_layout()
        plt.savefig('/workspace/visualization/model_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_fraud_ring_heatmap(self, fraud_rings, graph, title="Fraud Ring Analysis"):
        """Create heatmap showing fraud ring characteristics"""
        if not fraud_rings:
            print("No fraud rings detected")
            return
        
        # Analyze fraud ring characteristics
        ring_data = []
        for ring in fraud_rings:
            ring_info = {
                'Ring_ID': ring['community_id'],
                'Size': ring['size'],
                'Fraud_Rate': ring['fraud_rate'],
                'Card_Count': len([n for n in ring['nodes'] if graph.nodes[n].get('node_type') == 'card']),
                'Merchant_Count': len([n for n in ring['nodes'] if graph.nodes[n].get('node_type') == 'merchant']),
                'Device_Count': len([n for n in ring['nodes'] if graph.nodes[n].get('node_type') == 'device']),
                'IP_Count': len([n for n in ring['nodes'] if graph.nodes[n].get('node_type') == 'ip'])
            }
            ring_data.append(ring_info)
        
        ring_df = pd.DataFrame(ring_data)
        
        # Create heatmap
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Ring size vs fraud rate
        sns.scatterplot(data=ring_df, x='Size', y='Fraud_Rate', 
                       size='Card_Count', hue='Merchant_Count', ax=axes[0,0])
        axes[0,0].set_title('Ring Size vs Fraud Rate')
        axes[0,0].set_xlabel('Ring Size')
        axes[0,0].set_ylabel('Fraud Rate')
        
        # Node type distribution
        node_type_counts = ring_df[['Card_Count', 'Merchant_Count', 'Device_Count', 'IP_Count']].T
        sns.heatmap(node_type_counts, ax=axes[0,1], cmap='Blues', cbar=True)
        axes[0,1].set_title('Node Type Distribution in Rings')
        axes[0,1].set_xlabel('Ring ID')
        axes[0,1].set_ylabel('Node Type')
        
        # Fraud rate distribution
        fraud_rate_bins = pd.cut(ring_df['Fraud_Rate'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        fraud_dist = fraud_rate_bins.value_counts()
        axes[1,0].pie(fraud_dist.values, labels=fraud_dist.index, autopct='%1.1f%%')
        axes[1,0].set_title('Fraud Rate Distribution')
        
        # Ring size distribution
        size_bins = pd.cut(ring_df['Size'], bins=5, labels=['Small', 'Medium', 'Large', 'Very Large', 'Huge'])
        size_dist = size_bins.value_counts()
        axes[1,1].bar(size_dist.index, size_dist.values)
        axes[1,1].set_title('Ring Size Distribution')
        axes[1,1].set_xlabel('Size Category')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('/workspace/visualization/fraud_ring_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_fraud_dashboard(self, df, fraud_rings=None):
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Fraud Rate by Hour', 'Fraud Rate by Amount',
                           'Fraud Rate by Merchant', 'Fraud Rate by Country',
                           'Fraud Rate by Device', 'Transaction Volume'),
            specs=[[{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Fraud rate by hour
        hour_fraud = df.groupby('Hour')['Class'].mean()
        fig.add_trace(
            go.Bar(x=hour_fraud.index, y=hour_fraud.values, name='Fraud Rate by Hour'),
            row=1, col=1
        )
        
        # 2. Fraud rate by amount (histogram)
        fig.add_trace(
            go.Histogram(x=df[df['Class']==1]['Amount'], name='Fraud Amount Distribution'),
            row=1, col=2
        )
        
        # 3. Fraud rate by merchant
        merchant_fraud = df.groupby('Merchant')['Class'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=merchant_fraud.index, y=merchant_fraud.values, name='Fraud Rate by Merchant'),
            row=2, col=1
        )
        
        # 4. Fraud rate by country
        country_fraud = df.groupby('Country')['Class'].mean().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=country_fraud.index, y=country_fraud.values, name='Fraud Rate by Country'),
            row=2, col=2
        )
        
        # 5. Fraud rate by device
        device_fraud = df.groupby('Device')['Class'].mean()
        fig.add_trace(
            go.Bar(x=device_fraud.index, y=device_fraud.values, name='Fraud Rate by Device'),
            row=3, col=1
        )
        
        # 6. Transaction volume over time
        df['Time_Hour'] = df['Time'] % 24
        time_volume = df.groupby('Time_Hour').size()
        fig.add_trace(
            go.Scatter(x=time_volume.index, y=time_volume.values, 
                      mode='lines+markers', name='Transaction Volume'),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=False, title_text="Interactive Fraud Detection Dashboard")
        fig.write_html('/workspace/visualization/interactive_dashboard.html')
        fig.show()
    
    def create_confusion_matrix_heatmap(self, y_true, y_pred, model_name="Model"):
        """Create confusion matrix heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'/workspace/visualization/confusion_matrix_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_roc_curve_heatmap(self, model_results, title="ROC Curves Comparison"):
        """Create ROC curves comparison"""
        plt.figure(figsize=(10, 8))
        
        for model_name, results in model_results.items():
            if 'roc_curve' in results:
                fpr, tpr, _ = results['roc_curve']
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {results.get("auc", 0):.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('/workspace/visualization/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_feature_importance_heatmap(self, feature_importance, title="Feature Importance"):
        """Create feature importance heatmap"""
        if isinstance(feature_importance, dict):
            features = list(feature_importance.keys())
            importance = list(feature_importance.values())
        else:
            features = [f'Feature_{i}' for i in range(len(feature_importance))]
            importance = feature_importance
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importance, y=features, palette='viridis')
        plt.title(title)
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig('/workspace/visualization/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_geographic_fraud_heatmap(self, df, title="Geographic Fraud Distribution"):
        """Create geographic fraud heatmap"""
        country_fraud = df.groupby('Country').agg({
            'Class': ['mean', 'count'],
            'Amount': 'mean'
        }).round(3)
        
        country_fraud.columns = ['Fraud_Rate', 'Transaction_Count', 'Avg_Amount']
        country_fraud = country_fraud[country_fraud['Transaction_Count'] >= 10]  # Filter for meaningful data
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Fraud rate by country
        country_fraud_sorted = country_fraud.sort_values('Fraud_Rate', ascending=False)
        sns.barplot(data=country_fraud_sorted, x='Fraud_Rate', y=country_fraud_sorted.index, ax=axes[0])
        axes[0].set_title('Fraud Rate by Country')
        axes[0].set_xlabel('Fraud Rate')
        
        # Transaction count by country
        country_fraud_sorted_count = country_fraud.sort_values('Transaction_Count', ascending=False)
        sns.barplot(data=country_fraud_sorted_count, x='Transaction_Count', y=country_fraud_sorted_count.index, ax=axes[1])
        axes[1].set_title('Transaction Count by Country')
        axes[1].set_xlabel('Transaction Count')
        
        # Average amount by country
        country_fraud_sorted_amount = country_fraud.sort_values('Avg_Amount', ascending=False)
        sns.barplot(data=country_fraud_sorted_amount, x='Avg_Amount', y=country_fraud_sorted_amount.index, ax=axes[2])
        axes[2].set_title('Average Amount by Country')
        axes[2].set_xlabel('Average Amount')
        
        plt.tight_layout()
        plt.savefig('/workspace/visualization/geographic_fraud_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_time_series_fraud_heatmap(self, df, title="Time Series Fraud Analysis"):
        """Create time series fraud analysis"""
        # Create time bins
        df['Time_Bin'] = pd.cut(df['Time'], bins=50, labels=False)
        
        # Calculate fraud rate over time
        time_fraud = df.groupby('Time_Bin').agg({
            'Class': ['mean', 'count'],
            'Amount': 'mean'
        }).round(3)
        
        time_fraud.columns = ['Fraud_Rate', 'Transaction_Count', 'Avg_Amount']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Fraud rate over time
        axes[0,0].plot(time_fraud.index, time_fraud['Fraud_Rate'], marker='o')
        axes[0,0].set_title('Fraud Rate Over Time')
        axes[0,0].set_xlabel('Time Bin')
        axes[0,0].set_ylabel('Fraud Rate')
        axes[0,0].grid(True)
        
        # Transaction count over time
        axes[0,1].plot(time_fraud.index, time_fraud['Transaction_Count'], marker='o', color='green')
        axes[0,1].set_title('Transaction Count Over Time')
        axes[0,1].set_xlabel('Time Bin')
        axes[0,1].set_ylabel('Transaction Count')
        axes[0,1].grid(True)
        
        # Average amount over time
        axes[1,0].plot(time_fraud.index, time_fraud['Avg_Amount'], marker='o', color='orange')
        axes[1,0].set_title('Average Amount Over Time')
        axes[1,0].set_xlabel('Time Bin')
        axes[1,0].set_ylabel('Average Amount')
        axes[1,0].grid(True)
        
        # Fraud rate vs transaction count scatter
        axes[1,1].scatter(time_fraud['Transaction_Count'], time_fraud['Fraud_Rate'], alpha=0.7)
        axes[1,1].set_title('Fraud Rate vs Transaction Count')
        axes[1,1].set_xlabel('Transaction Count')
        axes[1,1].set_ylabel('Fraud Rate')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('/workspace/visualization/time_series_fraud_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

def generate_comprehensive_visualizations(df, model_results=None, fraud_rings=None):
    """Generate all comprehensive visualizations"""
    visualizer = FraudHeatmapVisualizer()
    
    print("Generating comprehensive fraud detection visualizations...")
    
    # Basic fraud pattern heatmap
    visualizer.create_fraud_pattern_heatmap(df)
    
    # Model performance comparison
    if model_results:
        visualizer.create_model_performance_heatmap(model_results)
    
    # Fraud ring analysis
    if fraud_rings:
        visualizer.create_fraud_ring_heatmap(fraud_rings, None)  # Pass graph if available
    
    # Interactive dashboard
    visualizer.create_interactive_fraud_dashboard(df, fraud_rings)
    
    # Geographic analysis
    visualizer.create_geographic_fraud_heatmap(df)
    
    # Time series analysis
    visualizer.create_time_series_fraud_heatmap(df)
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    # Test the visualizer
    from data_processor import FraudDataProcessor
    
    # Load data
    processor = FraudDataProcessor()
    df = processor.load_data()
    df_processed = processor.preprocess_data(df)
    
    # Generate visualizations
    generate_comprehensive_visualizations(df_processed)