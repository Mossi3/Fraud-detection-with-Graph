# Credit Card Fraud Detection with Deep Learning and Graph Analysis

## Overview
This project implements a comprehensive fraud detection system using:
- Deep Learning models (Neural Networks)
- Heatmap visualizations for fraud patterns
- Graph-based fraud detection using Graph Neural Networks (GNNs)
- REST API for real-time fraud detection
- Interactive web interface for presentations

## Features
- **Deep Learning**: Neural network models for transaction classification
- **Heatmaps**: Visual analysis of fraud patterns across time, merchants, and locations
- **Graph Analysis**: Detection of fraud rings using GraphSAGE and GAT models
- **Real-time API**: REST endpoints for fraud detection
- **Interactive Dashboard**: Fancy web interface for presentations

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Start the Flask API server:
```bash
python app.py
```

2. Start the dashboard:
```bash
python dashboard.py
```

3. Test with curl commands (see curl_examples.md)

## Project Structure
- `app.py` - Flask API server
- `dashboard.py` - Interactive web dashboard
- `models/` - Deep learning and graph models
- `data/` - Mock datasets and preprocessing
- `utils/` - Utility functions
- `static/` - CSS and JavaScript files
- `templates/` - HTML templates