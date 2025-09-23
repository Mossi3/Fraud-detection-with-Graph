-- Initialize fraud detection database schema

-- Create schema
CREATE SCHEMA IF NOT EXISTS fraud_detection;

-- Transaction table
CREATE TABLE IF NOT EXISTS fraud_detection.transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    card_id VARCHAR(50) NOT NULL,
    merchant_id VARCHAR(100) NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    device_id VARCHAR(50),
    ip_address VARCHAR(45),
    location_lat DECIMAL(10, 8),
    location_lon DECIMAL(11, 8),
    is_fraud BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5, 4),
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_card_id (card_id),
    INDEX idx_merchant_id (merchant_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_fraud (is_fraud)
);

-- Fraud rings table
CREATE TABLE IF NOT EXISTS fraud_detection.fraud_rings (
    ring_id VARCHAR(50) PRIMARY KEY,
    detection_timestamp TIMESTAMP NOT NULL,
    ring_score DECIMAL(5, 4) NOT NULL,
    entity_count INTEGER NOT NULL,
    ring_type VARCHAR(50),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Ring members table
CREATE TABLE IF NOT EXISTS fraud_detection.ring_members (
    ring_id VARCHAR(50) NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    entity_type VARCHAR(20) NOT NULL,
    role VARCHAR(50),
    PRIMARY KEY (ring_id, entity_id),
    FOREIGN KEY (ring_id) REFERENCES fraud_detection.fraud_rings(ring_id)
);

-- Alerts table
CREATE TABLE IF NOT EXISTS fraud_detection.alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    title VARCHAR(200) NOT NULL,
    description TEXT,
    entity_ids JSON,
    metrics JSON,
    resolved BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    INDEX idx_severity (severity),
    INDEX idx_created (created_at)
);

-- Model performance table
CREATE TABLE IF NOT EXISTS fraud_detection.model_performance (
    model_name VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    accuracy DECIMAL(5, 4),
    precision_score DECIMAL(5, 4),
    recall_score DECIMAL(5, 4),
    f1_score DECIMAL(5, 4),
    auc_score DECIMAL(5, 4),
    PRIMARY KEY (model_name, timestamp)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_transactions_card_timestamp 
    ON fraud_detection.transactions(card_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_transactions_merchant_timestamp 
    ON fraud_detection.transactions(merchant_id, timestamp);

CREATE INDEX IF NOT EXISTS idx_fraud_score 
    ON fraud_detection.transactions(fraud_score) 
    WHERE fraud_score IS NOT NULL;

-- Create views for analytics
CREATE OR REPLACE VIEW fraud_detection.fraud_statistics AS
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as total_transactions,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END) as fraud_rate,
    SUM(amount) as total_amount,
    SUM(CASE WHEN is_fraud THEN amount ELSE 0 END) as fraud_amount
FROM fraud_detection.transactions
GROUP BY DATE(timestamp);

CREATE OR REPLACE VIEW fraud_detection.merchant_risk_scores AS
SELECT 
    merchant_id,
    COUNT(*) as transaction_count,
    SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END) as fraud_count,
    AVG(CASE WHEN is_fraud THEN 1.0 ELSE 0.0 END) as fraud_rate,
    AVG(fraud_score) as avg_fraud_score
FROM fraud_detection.transactions
GROUP BY merchant_id
HAVING COUNT(*) >= 10;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA fraud_detection TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA fraud_detection TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA fraud_detection TO postgres;