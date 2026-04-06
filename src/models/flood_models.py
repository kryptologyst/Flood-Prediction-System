"""
Flood Prediction System - Models Module

This module implements various machine learning models for flood prediction,
including baseline models, neural networks, and spatial ML approaches.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, Any, Optional
import logging
import joblib
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manage device selection for PyTorch models."""
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device (CUDA > MPS > CPU)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        return device


class FloodNeuralNetwork(nn.Module):
    """Neural network for flood prediction."""
    
    def __init__(self, input_size: int, hidden_sizes: list = [64, 32], dropout_rate: float = 0.2):
        """Initialize the neural network.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
        """
        super(FloodNeuralNetwork, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


class FloodPredictionModels:
    """Collection of flood prediction models."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize models with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = DeviceManager.get_device()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def prepare_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Dict:
        """Prepare data for different model types.
        
        Args:
            X_train, X_val, X_test: Feature matrices
            y_train, y_val, y_test: Target vectors
            
        Returns:
            Dictionary with prepared data for different model types
        """
        # Scale features for neural network
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(self.device)
        
        return {
            'sklearn': {
                'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                'y_train': y_train, 'y_val': y_val, 'y_test': y_test
            },
            'pytorch': {
                'X_train': X_train_tensor, 'X_val': X_val_tensor, 'X_test': X_test_tensor,
                'y_train': y_train_tensor, 'y_val': y_val_tensor, 'y_test': y_test_tensor
            }
        }
    
    def train_logistic_regression(self, data: Dict) -> Dict:
        """Train logistic regression baseline."""
        logger.info("Training Logistic Regression...")
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        train_pred = model.predict(data['X_train'])
        val_pred = model.predict(data['X_val'])
        test_pred = model.predict(data['X_test'])
        
        train_proba = model.predict_proba(data['X_train'])[:, 1]
        val_proba = model.predict_proba(data['X_val'])[:, 1]
        test_proba = model.predict_proba(data['X_test'])[:, 1]
        
        results = self._evaluate_model(
            data['y_train'], train_pred, train_proba,
            data['y_val'], val_pred, val_proba,
            data['y_test'], test_pred, test_proba
        )
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        
        return results
    
    def train_random_forest(self, data: Dict) -> Dict:
        """Train random forest model."""
        logger.info("Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(data['X_train'], data['y_train'])
        
        # Predictions
        train_pred = model.predict(data['X_train'])
        val_pred = model.predict(data['X_val'])
        test_pred = model.predict(data['X_test'])
        
        train_proba = model.predict_proba(data['X_train'])[:, 1]
        val_proba = model.predict_proba(data['X_val'])[:, 1]
        test_proba = model.predict_proba(data['X_test'])[:, 1]
        
        results = self._evaluate_model(
            data['y_train'], train_pred, train_proba,
            data['y_val'], val_pred, val_proba,
            data['y_test'], test_pred, test_proba
        )
        
        self.models['random_forest'] = model
        self.results['random_forest'] = results
        
        return results
    
    def train_xgboost(self, data: Dict) -> Dict:
        """Train XGBoost model."""
        logger.info("Training XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(
            data['X_train'], data['y_train'],
            eval_set=[(data['X_val'], data['y_val'])],
            verbose=False
        )
        
        # Predictions
        train_pred = model.predict(data['X_train'])
        val_pred = model.predict(data['X_val'])
        test_pred = model.predict(data['X_test'])
        
        train_proba = model.predict_proba(data['X_train'])[:, 1]
        val_proba = model.predict_proba(data['X_val'])[:, 1]
        test_proba = model.predict_proba(data['X_test'])[:, 1]
        
        results = self._evaluate_model(
            data['y_train'], train_pred, train_proba,
            data['y_val'], val_pred, val_proba,
            data['y_test'], test_pred, test_proba
        )
        
        self.models['xgboost'] = model
        self.results['xgboost'] = results
        
        return results
    
    def train_lightgbm(self, data: Dict) -> Dict:
        """Train LightGBM model."""
        logger.info("Training LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        )
        model.fit(
            data['X_train'], data['y_train'],
            eval_set=[(data['X_val'], data['y_val'])],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Predictions
        train_pred = model.predict(data['X_train'])
        val_pred = model.predict(data['X_val'])
        test_pred = model.predict(data['X_test'])
        
        train_proba = model.predict_proba(data['X_train'])[:, 1]
        val_proba = model.predict_proba(data['X_val'])[:, 1]
        test_proba = model.predict_proba(data['X_test'])[:, 1]
        
        results = self._evaluate_model(
            data['y_train'], train_pred, train_proba,
            data['y_val'], val_pred, val_proba,
            data['y_test'], test_pred, test_proba
        )
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = results
        
        return results
    
    def train_neural_network(self, data: Dict) -> Dict:
        """Train neural network model."""
        logger.info("Training Neural Network...")
        
        input_size = data['X_train'].shape[1]
        hidden_sizes = self.config['model']['neural_network']['hidden_layers']
        dropout_rate = self.config['model']['neural_network']['dropout_rate']
        
        model = FloodNeuralNetwork(input_size, hidden_sizes, dropout_rate).to(self.device)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['model']['neural_network']['learning_rate']
        )
        
        # Data loaders
        train_dataset = TensorDataset(data['X_train'], data['y_train'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['model']['neural_network']['batch_size'],
            shuffle=True
        )
        
        # Training loop
        epochs = self.config['model']['neural_network']['epochs']
        patience = self.config['model']['neural_network']['early_stopping_patience']
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(data['X_val']).squeeze()
                val_loss = criterion(val_outputs, data['y_val']).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load('best_model.pth'))
        
        # Predictions
        model.eval()
        with torch.no_grad():
            train_proba = model(data['X_train']).cpu().numpy().squeeze()
            val_proba = model(data['X_val']).cpu().numpy().squeeze()
            test_proba = model(data['X_test']).cpu().numpy().squeeze()
        
        train_pred = (train_proba > 0.5).astype(int)
        val_pred = (val_proba > 0.5).astype(int)
        test_pred = (test_proba > 0.5).astype(int)
        
        results = self._evaluate_model(
            data['y_train'].cpu().numpy(), train_pred, train_proba,
            data['y_val'].cpu().numpy(), val_pred, val_proba,
            data['y_test'].cpu().numpy(), test_pred, test_proba
        )
        
        self.models['neural_network'] = model
        self.results['neural_network'] = results
        
        return results
    
    def _evaluate_model(self, y_train_true, y_train_pred, y_train_proba,
                       y_val_true, y_val_pred, y_val_proba,
                       y_test_true, y_test_pred, y_test_proba) -> Dict:
        """Evaluate model performance across train/val/test sets."""
        
        def calculate_metrics(y_true, y_pred, y_proba):
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'auc': roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
            }
        
        return {
            'train': calculate_metrics(y_train_true, y_train_pred, y_train_proba),
            'val': calculate_metrics(y_val_true, y_val_pred, y_val_proba),
            'test': calculate_metrics(y_test_true, y_test_pred, y_test_proba)
        }
    
    def train_all_models(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                        y_train: pd.Series, y_val: pd.Series, y_test: pd.Series) -> Dict:
        """Train all configured models."""
        logger.info("Preparing data for all models...")
        data = self.prepare_data(X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Train baseline models
        baseline_models = self.config['model']['baseline_models']
        
        if 'logistic_regression' in baseline_models:
            self.train_logistic_regression(data['sklearn'])
        
        if 'random_forest' in baseline_models:
            self.train_random_forest(data['sklearn'])
        
        if 'xgboost' in baseline_models:
            self.train_xgboost(data['sklearn'])
        
        if 'lightgbm' in baseline_models:
            self.train_lightgbm(data['sklearn'])
        
        # Train neural network
        self.train_neural_network(data['pytorch'])
        
        logger.info("All models trained successfully!")
        return self.results
    
    def save_models(self, save_dir: str = "assets/models"):
        """Save trained models."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            if name == 'neural_network':
                torch.save(model.state_dict(), save_path / f"{name}.pth")
            else:
                joblib.dump(model, save_path / f"{name}.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, save_path / "scaler.pkl")
        
        logger.info(f"Models saved to {save_path}")
    
    def get_model_leaderboard(self) -> pd.DataFrame:
        """Get model performance leaderboard."""
        leaderboard_data = []
        
        for model_name, results in self.results.items():
            leaderboard_data.append({
                'Model': model_name,
                'Test Accuracy': results['test']['accuracy'],
                'Test Precision': results['test']['precision'],
                'Test Recall': results['test']['recall'],
                'Test F1': results['test']['f1'],
                'Test AUC': results['test']['auc'],
                'Val Accuracy': results['val']['accuracy'],
                'Val F1': results['val']['f1']
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('Test F1', ascending=False)
        
        return leaderboard


if __name__ == "__main__":
    # Example usage
    from src.data.pipeline import FloodDataGenerator
    
    # Generate data
    generator = FloodDataGenerator()
    dataset = generator.generate_dataset()
    
    # Train models
    models = FloodPredictionModels()
    results = models.train_all_models(
        dataset['X_train'], dataset['X_val'], dataset['X_test'],
        dataset['y_train'], dataset['y_val'], dataset['y_test']
    )
    
    # Print leaderboard
    leaderboard = models.get_model_leaderboard()
    print("\nModel Leaderboard:")
    print(leaderboard.round(4))
