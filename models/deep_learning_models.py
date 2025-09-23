import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

class CNNFraudDetector(nn.Module):
    """Convolutional Neural Network for fraud detection"""
    
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(CNNFraudDetector, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        # Calculate the size after convolutions
        conv_output_size = 128 * (input_dim // 8)  # After 3 pooling operations
        
        self.fc1 = nn.Linear(conv_output_size, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], 1)
        
    def forward(self, x):
        # Reshape for 1D convolution
        x = x.unsqueeze(1)  # Add channel dimension
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x

class LSTMFraudDetector(nn.Module):
    """LSTM-based fraud detector"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(LSTMFraudDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Reshape for LSTM (batch_size, sequence_length, input_dim)
        x = x.unsqueeze(1)  # Add sequence dimension
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output
        x = lstm_out[:, -1, :]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

class TransformerFraudDetector(nn.Module):
    """Transformer-based fraud detector"""
    
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=2, dropout=0.3):
        super(TransformerFraudDetector, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(d_model, 32)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # Reshape for transformer (batch_size, sequence_length, d_model)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.input_projection(x)
        
        # Create attention mask (optional)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Use the last output
        x = x[:, -1, :]
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x

class DeepFraudDetector(nn.Module):
    """Deep feedforward network for fraud detection"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(DeepFraudDetector, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class FraudModelTrainer:
    """Trainer class for fraud detection models"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            target = target.float().unsqueeze(1)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (output > 0.5).float()
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.float().unsqueeze(1)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Train the model"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print('-' * 50)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
    
    def evaluate(self, test_loader):
        """Evaluate the model on test set"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                all_probabilities.extend(output.cpu().numpy())
                all_predictions.extend((output > 0.5).cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        auc_score = roc_auc_score(all_targets, all_probabilities)
        
        print("Classification Report:")
        print(classification_report(all_targets, all_predictions))
        print(f"AUC Score: {auc_score:.4f}")
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets,
            'auc': auc_score
        }
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('/workspace/visualization/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_class': self.model.__class__.__name__,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }, filepath)
    
    def load_model(self, filepath):
        """Load the model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """Create PyTorch data loaders"""
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    y_train_tensor = torch.LongTensor(y_train)
    y_val_tensor = torch.LongTensor(y_val)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the models
    from data_processor import FraudDataProcessor
    
    # Load and preprocess data
    processor = FraudDataProcessor()
    df = processor.load_data()
    df_processed = processor.preprocess_data(df)
    X, y = processor.prepare_features(df_processed)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_features(X_train, X_val, X_test)
    
    # Balance data
    X_train_balanced, y_train_balanced = processor.balance_data(X_train_scaled, y_train)
    
    print(f"Training set shape: {X_train_balanced.shape}")
    print(f"Validation set shape: {X_val_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train_balanced, X_val_scaled, X_test_scaled,
        y_train_balanced, y_val, y_test
    )
    
    # Test different models
    input_dim = X_train_balanced.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {
        'CNN': CNNFraudDetector(input_dim),
        'LSTM': LSTMFraudDetector(input_dim),
        'Transformer': TransformerFraudDetector(input_dim),
        'Deep': DeepFraudDetector(input_dim)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model...")
        trainer = FraudModelTrainer(model, device)
        trainer.train(train_loader, val_loader, epochs=10)
        
        # Evaluate
        results[name] = trainer.evaluate(test_loader)
        trainer.plot_training_history()
        
        # Save model
        trainer.save_model(f'/workspace/models/{name.lower()}_model.pth')
    
    # Print results summary
    print("\nModel Performance Summary:")
    for name, result in results.items():
        print(f"{name}: AUC = {result['auc']:.4f}")