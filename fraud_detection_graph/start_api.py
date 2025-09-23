
import sys
sys.path.append('/workspace/fraud_detection_graph')
from api.fraud_api import app, fraud_api

print("Loading fraud detection system...")
fraud_api.load_system()

if __name__ == '__main__':
    print("Starting API server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
