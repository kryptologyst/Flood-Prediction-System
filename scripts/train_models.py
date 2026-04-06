#!/usr/bin/env python3
"""
Flood Prediction System - Training Script

This script generates data, trains all models, and saves the results.
"""

import sys
from pathlib import Path
import logging
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Flood Prediction Models')
    parser.add_argument('--config', default='configs/config.yaml', help='Configuration file path')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Flood Prediction System training...")
    
    try:
        # Generate data
        logger.info("Generating synthetic flood data...")
        from src.data.pipeline import FloodDataGenerator
        
        generator = FloodDataGenerator(args.config)
        dataset = generator.generate_dataset()
        
        logger.info(f"Generated dataset with {len(dataset['processed_data'])} samples")
        logger.info(f"Features: {len(dataset['feature_names'])}")
        logger.info(f"Flood risk rate: {dataset['y_train'].mean():.3f}")
        
        # Train models
        logger.info("Training machine learning models...")
        from src.models.flood_models import FloodPredictionModels
        
        models = FloodPredictionModels(args.config)
        results = models.train_all_models(
            dataset['X_train'], dataset['X_val'], dataset['X_test'],
            dataset['y_train'], dataset['y_val'], dataset['y_test']
        )
        
        # Display results
        leaderboard = models.get_model_leaderboard()
        logger.info("Model Performance Leaderboard:")
        logger.info(leaderboard.round(4).to_string())
        
        # Save models if requested
        if args.save_models:
            logger.info("Saving trained models...")
            models.save_models()
            logger.info("Models saved successfully")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
