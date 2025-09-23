import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pickle
import os
from datetime import datetime, timedelta
import random

class FraudDataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = 'Class'
        
    def generate_mock_data(self, n_samples=100000, fraud_ratio=0.001):
        """Generate synthetic credit card fraud data for testing"""
        np.random.seed(42)
        random.seed(42)
        
        # Generate base transaction data
        data = []
        
        # Time features
        start_date = datetime.now() - timedelta(days=365)
        
        for i in range(n_samples):
            # Basic transaction features
            amount = np.random.exponential(50)  # Most transactions are small
            time_hour = np.random.randint(0, 24)
            time_day = np.random.randint(1, 32)
            
            # Generate merchant categories
            merchant_categories = ['grocery', 'gas', 'restaurant', 'online', 'retail', 'pharmacy', 'entertainment']
            merchant = random.choice(merchant_categories)
            
            # Generate location data
            countries = ['US', 'CA', 'UK', 'DE', 'FR', 'AU', 'JP']
            country = random.choice(countries)
            
            # Generate device/IP data
            device_type = random.choice(['mobile', 'desktop', 'tablet'])
            ip_country = random.choice(countries)
            
            # Create transaction record
            transaction = {
                'Time': i,
                'V1': np.random.normal(0, 1),
                'V2': np.random.normal(0, 1),
                'V3': np.random.normal(0, 1),
                'V4': np.random.normal(0, 1),
                'V5': np.random.normal(0, 1),
                'V6': np.random.normal(0, 1),
                'V7': np.random.normal(0, 1),
                'V8': np.random.normal(0, 1),
                'V9': np.random.normal(0, 1),
                'V10': np.random.normal(0, 1),
                'V11': np.random.normal(0, 1),
                'V12': np.random.normal(0, 1),
                'V13': np.random.normal(0, 1),
                'V14': np.random.normal(0, 1),
                'V15': np.random.normal(0, 1),
                'V16': np.random.normal(0, 1),
                'V17': np.random.normal(0, 1),
                'V18': np.random.normal(0, 1),
                'V19': np.random.normal(0, 1),
                'V20': np.random.normal(0, 1),
                'V21': np.random.normal(0, 1),
                'V22': np.random.normal(0, 1),
                'V23': np.random.normal(0, 1),
                'V24': np.random.normal(0, 1),
                'V25': np.random.normal(0, 1),
                'V26': np.random.normal(0, 1),
                'V27': np.random.normal(0, 1),
                'V28': np.random.normal(0, 1),
                'Amount': amount,
                'Class': 0,  # Default to non-fraud
                'Merchant': merchant,
                'Country': country,
                'Device': device_type,
                'IP_Country': ip_country,
                'Hour': time_hour,
                'Day': time_day,
                'Card_ID': f"CARD_{i % 10000:05d}",  # Simulate card reuse
                'Merchant_ID': f"M_{merchant}_{i % 1000:03d}",
                'Device_ID': f"D_{device_type}_{i % 5000:04d}",
                'IP_Address': f"192.168.{i % 255}.{i % 255}",
                'Transaction_ID': f"TXN_{i:08d}",
                'Timestamp': start_date + timedelta(hours=i)
            }
            
            data.append(transaction)
        
        df = pd.DataFrame(data)
        
        # Introduce fraud patterns
        fraud_indices = np.random.choice(df.index, size=int(n_samples * fraud_ratio), replace=False)
        
        for idx in fraud_indices:
            # Make fraud transactions more suspicious
            df.loc[idx, 'Class'] = 1
            df.loc[idx, 'Amount'] = np.random.exponential(200)  # Higher amounts
            df.loc[idx, 'V1'] = np.random.normal(2, 1)  # Suspicious patterns
            df.loc[idx, 'V2'] = np.random.normal(-2, 1)
            df.loc[idx, 'V3'] = np.random.normal(1.5, 1)
            df.loc[idx, 'V4'] = np.random.normal(-1.5, 1)
            
            # Introduce fraud rings (same card, multiple merchants)
            if random.random() < 0.3:  # 30% of frauds are part of rings
                ring_size = random.randint(2, 5)
                ring_cards = [f"CARD_{random.randint(0, 9999):05d}" for _ in range(ring_size)]
                ring_merchants = random.sample(merchant_categories, ring_size)
                
                for i, (card, merchant) in enumerate(zip(ring_cards, ring_merchants)):
                    ring_idx = (idx + i + 1) % len(df)
                    df.loc[ring_idx, 'Class'] = 1
                    df.loc[ring_idx, 'Card_ID'] = card
                    df.loc[ring_idx, 'Merchant'] = merchant
                    df.loc[ring_idx, 'Amount'] = np.random.exponential(150)
        
        return df
    
    def load_data(self, file_path=None):
        """Load credit card fraud dataset"""
        if file_path is None:
            file_path = self.data_path
            
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            print("Generating mock data...")
            df = self.generate_mock_data()
            
        return df
    
    def preprocess_data(self, df):
        """Preprocess the dataset for machine learning"""
        # Handle missing values
        df = df.fillna(df.median())
        
        # Encode categorical variables
        categorical_columns = ['Merchant', 'Country', 'Device', 'IP_Country']
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Create additional features
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        
        # Feature engineering for graph analysis
        df['Card_Merchant_Combo'] = df['Card_ID'] + '_' + df['Merchant_ID']
        df['Device_Merchant_Combo'] = df['Device_ID'] + '_' + df['Merchant_ID']
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        # Select numerical features
        numerical_features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Amount_log']
        numerical_features.extend(['Hour_sin', 'Hour_cos', 'Day_sin', 'Day_cos'])
        
        # Add encoded categorical features
        categorical_features = [col for col in df.columns if col.endswith('_encoded')]
        
        self.feature_columns = numerical_features + categorical_features
        
        # Filter existing columns
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, val_size=0.1):
        """Split data into train, validation, and test sets"""
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train, X_val, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def balance_data(self, X, y, method='undersample'):
        """Balance the dataset"""
        if method == 'undersample':
            from imblearn.under_sampling import RandomUnderSampler
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X, y)
        elif method == 'oversample':
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
        else:
            X_balanced, y_balanced = X, y
            
        return X_balanced, y_balanced
    
    def save_preprocessor(self, filepath):
        """Save the preprocessor for later use"""
        preprocessor_data = {
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessor_data, f)
    
    def load_preprocessor(self, filepath):
        """Load the preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor_data = pickle.load(f)
            
        self.scaler = preprocessor_data['scaler']
        self.label_encoders = preprocessor_data['label_encoders']
        self.feature_columns = preprocessor_data['feature_columns']

if __name__ == "__main__":
    # Test the data processor
    processor = FraudDataProcessor()
    
    # Generate and save mock data
    df = processor.generate_mock_data(n_samples=50000)
    df.to_csv('/workspace/data/creditcard_fraud.csv', index=False)
    
    print(f"Generated dataset with {len(df)} samples")
    print(f"Fraud rate: {df['Class'].mean():.4f}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Preprocess the data
    df_processed = processor.preprocess_data(df)
    X, y = processor.prepare_features(df_processed)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {len(processor.feature_columns)}")