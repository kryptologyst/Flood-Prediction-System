#!/usr/bin/env python3
"""
Flood Prediction System - Test Script

This script tests the basic functionality of the flood prediction system
by generating data, training models, and creating visualizations.
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_data_generation():
    """Test data generation functionality."""
    logger.info("Testing data generation...")
    
    try:
        from src.data.pipeline import FloodDataGenerator
        
        generator = FloodDataGenerator()
        dataset = generator.generate_dataset()
        
        logger.info(f"✓ Generated {len(dataset['processed_data'])} samples")
        logger.info(f"✓ Created {len(dataset['spatial_data'])} spatial regions")
        logger.info(f"✓ Engineered {len(dataset['feature_names'])} features")
        logger.info(f"✓ Flood risk rate: {dataset['y_train'].mean():.3f}")
        
        return True, dataset
        
    except Exception as e:
        logger.error(f"✗ Data generation failed: {e}")
        return False, None

def test_model_training(dataset):
    """Test model training functionality."""
    logger.info("Testing model training...")
    
    try:
        from src.models.flood_models import FloodPredictionModels
        
        models = FloodPredictionModels()
        results = models.train_all_models(
            dataset['X_train'], dataset['X_val'], dataset['X_test'],
            dataset['y_train'], dataset['y_val'], dataset['y_test']
        )
        
        leaderboard = models.get_model_leaderboard()
        logger.info("✓ Model training completed successfully")
        logger.info(f"✓ Trained {len(results)} models")
        logger.info(f"✓ Best model: {leaderboard.iloc[0]['Model']} (F1: {leaderboard.iloc[0]['Test F1']:.3f})")
        
        return True, models
        
    except Exception as e:
        logger.error(f"✗ Model training failed: {e}")
        return False, None

def test_visualization(dataset, models):
    """Test visualization functionality."""
    logger.info("Testing visualization...")
    
    try:
        from src.viz.visualization import FloodVisualizer
        
        visualizer = FloodVisualizer()
        
        # Generate sample predictions
        import numpy as np
        n_test = len(dataset['X_test'])
        probabilities = np.random.beta(2, 5, n_test)
        predictions = (probabilities > 0.5).astype(int)
        
        # Test map creation
        risk_map = visualizer.create_flood_risk_map(
            dataset['spatial_data'], predictions, probabilities
        )
        
        # Test time series plot
        time_series_fig = visualizer.plot_time_series(
            dataset['processed_data'], region_id=0
        )
        
        logger.info("✓ Visualization components created successfully")
        logger.info("✓ Risk map generated")
        logger.info("✓ Time series plot created")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Visualization failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting Flood Prediction System tests...")
    
    # Test data generation
    success, dataset = test_data_generation()
    if not success:
        logger.error("Data generation test failed. Exiting.")
        return False
    
    # Test model training
    success, models = test_model_training(dataset)
    if not success:
        logger.error("Model training test failed. Exiting.")
        return False
    
    # Test visualization
    success = test_visualization(dataset, models)
    if not success:
        logger.error("Visualization test failed.")
        return False
    
    logger.info("✓ All tests passed successfully!")
    logger.info("✓ System is ready for use")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
