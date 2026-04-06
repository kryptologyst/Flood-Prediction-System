"""
Flood Prediction System - Data Pipeline Module

This module handles data generation, preprocessing, and feature engineering
for flood prediction using synthetic hydrological and meteorological data.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Tuple, Dict, List, Optional
import logging
from pathlib import Path
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class FloodDataGenerator:
    """Generate synthetic flood prediction data with spatial and temporal features."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the data generator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        
        # Data parameters
        self.n_samples = self.config['data']['n_samples']
        self.n_regions = self.config['data']['n_regions']
        self.time_steps = self.config['data']['time_steps']
        
        # Geographic bounds
        self.bounds = self.config['geo']['bounds']
        
    def generate_spatial_locations(self) -> gpd.GeoDataFrame:
        """Generate random spatial locations within the defined bounds.
        
        Returns:
            GeoDataFrame with random points and region IDs
        """
        # Generate random coordinates
        lons = np.random.uniform(
            self.bounds['min_lon'], 
            self.bounds['max_lon'], 
            self.n_regions
        )
        lats = np.random.uniform(
            self.bounds['min_lat'], 
            self.bounds['max_lat'], 
            self.n_regions
        )
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'region_id': range(self.n_regions),
                'longitude': lons,
                'latitude': lats,
                'elevation': np.random.normal(500, 200, self.n_regions),  # meters
                'slope': np.random.normal(10, 5, self.n_regions),  # degrees
                'drainage_area': np.random.lognormal(8, 1, self.n_regions),  # km²
            },
            geometry=gpd.points_from_xy(lons, lats),
            crs=self.config['geo']['crs']
        )
        
        return gdf
    
    def generate_hydrological_data(self, spatial_data: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate synthetic hydrological and meteorological time series data.
        
        Args:
            spatial_data: GeoDataFrame with spatial locations
            
        Returns:
            DataFrame with time series hydrological data
        """
        data_list = []
        
        for _, region in spatial_data.iterrows():
            region_id = region['region_id']
            elevation = region['elevation']
            slope = region['slope']
            drainage_area = region['drainage_area']
            
            # Generate time series for this region
            for t in range(self.time_steps):
                # Seasonal patterns
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * t / 365)
                
                # Rainfall (mm) - higher in winter, influenced by elevation
                rainfall_base = 50 + elevation * 0.01
                rainfall = np.random.exponential(rainfall_base * seasonal_factor)
                
                # River level (meters) - correlated with rainfall and drainage
                river_level_base = 2.0 + rainfall * 0.01 + drainage_area * 0.001
                river_level = np.random.normal(river_level_base, 0.5)
                
                # Soil moisture (0-1) - influenced by rainfall and slope
                soil_moisture_base = 0.3 + rainfall * 0.002 - slope * 0.005
                soil_moisture = np.clip(np.random.normal(soil_moisture_base, 0.1), 0, 1)
                
                # Runoff rate (mm/hr) - influenced by rainfall, slope, and soil moisture
                runoff_base = rainfall * 0.1 + slope * 0.5 + soil_moisture * 20
                runoff_rate = np.random.exponential(runoff_base)
                
                # Temperature (°C) - seasonal pattern
                temperature = 15 + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 3)
                
                # Wind speed (m/s)
                wind_speed = np.random.exponential(5)
                
                # Flood risk label based on multiple factors
                flood_risk = (
                    (rainfall > 100) & (river_level > 3.5) & (soil_moisture > 0.7) |
                    (runoff_rate > 50) & (rainfall > 80) |
                    (river_level > 4.5) |
                    (rainfall > 150)
                ).astype(int)
                
                data_list.append({
                    'region_id': region_id,
                    'time_step': t,
                    'rainfall': rainfall,
                    'river_level': river_level,
                    'soil_moisture': soil_moisture,
                    'runoff_rate': runoff_rate,
                    'temperature': temperature,
                    'wind_speed': wind_speed,
                    'elevation': elevation,
                    'slope': slope,
                    'drainage_area': drainage_area,
                    'flood_risk': flood_risk
                })
        
        return pd.DataFrame(data_list)
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional temporal and spatial features.
        
        Args:
            data: Raw hydrological data
            
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Temporal features
        df['day_of_year'] = df['time_step'] % 365
        df['month'] = (df['day_of_year'] / 30.44).astype(int) + 1
        df['season'] = ((df['month'] - 1) // 3) % 4
        
        # Temporal lags
        temporal_lags = self.config['data']['temporal_lags']
        for lag in temporal_lags:
            df[f'rainfall_lag_{lag}'] = df.groupby('region_id')['rainfall'].shift(lag)
            df[f'river_level_lag_{lag}'] = df.groupby('region_id')['river_level'].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14]:
            df[f'rainfall_rolling_mean_{window}'] = (
                df.groupby('region_id')['rainfall'].rolling(window=window).mean().reset_index(0, drop=True)
            )
            df[f'rainfall_rolling_std_{window}'] = (
                df.groupby('region_id')['rainfall'].rolling(window=window).std().reset_index(0, drop=True)
            )
        
        # Derived features
        df['rainfall_intensity'] = df['rainfall'] * df['runoff_rate']
        df['flood_potential'] = (
            df['rainfall'] * df['soil_moisture'] * df['slope'] / 100
        )
        df['drainage_efficiency'] = df['runoff_rate'] / (df['drainage_area'] + 1)
        
        # Fill NaN values from lagged features
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        return df
    
    def create_train_test_split(
        self, 
        data: pd.DataFrame, 
        spatial_data: gpd.GeoDataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Create temporal and spatial train/test splits.
        
        Args:
            data: Processed hydrological data
            spatial_data: Spatial location data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Feature columns (exclude target and metadata)
        feature_cols = [col for col in data.columns 
                       if col not in ['region_id', 'time_step', 'flood_risk']]
        
        X = data[feature_cols]
        y = data['flood_risk']
        
        # Temporal split (use last portion for testing)
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        test_ratio = self.config['data']['test_ratio']
        
        # Split by time steps
        max_time = data['time_step'].max()
        train_end = int(max_time * train_ratio)
        val_end = int(max_time * (train_ratio + val_ratio))
        
        train_mask = data['time_step'] <= train_end
        val_mask = (data['time_step'] > train_end) & (data['time_step'] <= val_end)
        test_mask = data['time_step'] > val_end
        
        X_train = X[train_mask]
        X_val = X[val_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_val = y[val_mask]
        y_test = y[test_mask]
        
        logger.info(f"Train samples: {len(X_train)}")
        logger.info(f"Validation samples: {len(X_val)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def generate_dataset(self) -> Dict:
        """Generate complete flood prediction dataset.
        
        Returns:
            Dictionary containing all generated data components
        """
        logger.info("Generating spatial locations...")
        spatial_data = self.generate_spatial_locations()
        
        logger.info("Generating hydrological time series...")
        hydrological_data = self.generate_hydrological_data(spatial_data)
        
        logger.info("Engineering features...")
        processed_data = self.engineer_features(hydrological_data)
        
        logger.info("Creating train/test splits...")
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_test_split(
            processed_data, spatial_data
        )
        
        # Save data
        data_dir = Path(self.config['data']['processed_dir'])
        data_dir.mkdir(parents=True, exist_ok=True)
        
        spatial_data.to_file(data_dir / "spatial_locations.geojson", driver="GeoJSON")
        processed_data.to_csv(data_dir / "processed_data.csv", index=False)
        
        return {
            'spatial_data': spatial_data,
            'processed_data': processed_data,
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': X_train.columns.tolist()
        }


if __name__ == "__main__":
    # Generate dataset
    generator = FloodDataGenerator()
    dataset = generator.generate_dataset()
    
    print("Dataset generated successfully!")
    print(f"Total samples: {len(dataset['processed_data'])}")
    print(f"Features: {len(dataset['feature_names'])}")
    print(f"Flood risk rate: {dataset['y_train'].mean():.3f}")
