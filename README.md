# Flood Prediction System

A comprehensive machine learning system for flood risk assessment using synthetic hydrological and meteorological data. This project demonstrates advanced techniques in environmental data science, spatial machine learning, and interactive visualization.

## Overview

This flood prediction system combines multiple machine learning approaches to assess flood risk based on hydrological variables, meteorological conditions, and spatial features. The system generates synthetic data that mimics real-world flood scenarios and provides interactive tools for risk assessment and visualization.

## Features

- **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost, LightGBM, and Neural Networks
- **Spatiotemporal Data**: Time series hydrological data with spatial features
- **Interactive Mapping**: Real-time flood risk visualization with Folium
- **Time Series Analysis**: Comprehensive hydrological variable tracking
- **Model Comparison**: Performance benchmarking and feature importance analysis
- **Web Dashboard**: Streamlit-based interactive interface

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Flood-Prediction-System.git
cd Flood-Prediction-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the interactive demo:
```bash
streamlit run demo/app.py
```

### Basic Usage

```python
from src.data.pipeline import FloodDataGenerator
from src.models.flood_models import FloodPredictionModels

# Generate synthetic flood data
generator = FloodDataGenerator()
dataset = generator.generate_dataset()

# Train multiple models
models = FloodPredictionModels()
results = models.train_all_models(
    dataset['X_train'], dataset['X_val'], dataset['X_test'],
    dataset['y_train'], dataset['y_val'], dataset['y_test']
)

# View model performance
leaderboard = models.get_model_leaderboard()
print(leaderboard)
```

## Project Structure

```
flood-prediction-system/
├── src/                          # Source code
│   ├── data/                     # Data pipeline
│   │   └── pipeline.py           # Data generation and preprocessing
│   ├── models/                   # ML models
│   │   └── flood_models.py       # Model implementations
│   ├── eval/                     # Evaluation metrics
│   └── viz/                      # Visualization tools
│       └── visualization.py      # Plotting and mapping functions
├── configs/                       # Configuration files
│   └── config.yaml              # Main configuration
├── data/                         # Data storage
│   ├── raw/                      # Raw data
│   ├── processed/                # Processed data
│   └── external/                 # External datasets
├── assets/                       # Generated outputs
│   ├── models/                   # Trained models
│   └── plots/                    # Generated plots
├── demo/                         # Interactive demo
│   └── app.py                    # Streamlit application
├── tests/                        # Unit tests
├── scripts/                      # Utility scripts
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Data Schema

### Hydrological Variables

- **Rainfall**: Daily precipitation in mm
- **River Level**: Water level in meters
- **Soil Moisture**: Moisture content (0-1 scale)
- **Runoff Rate**: Surface runoff in mm/hr
- **Temperature**: Air temperature in °C
- **Wind Speed**: Wind velocity in m/s

### Spatial Features

- **Elevation**: Terrain elevation in meters
- **Slope**: Terrain slope in degrees
- **Drainage Area**: Catchment area in km²
- **Coordinates**: Longitude and latitude

### Temporal Features

- **Time Steps**: Daily time series
- **Seasonal Patterns**: Monthly and seasonal indicators
- **Lagged Variables**: Previous day/week values
- **Rolling Statistics**: Moving averages and standard deviations

## Model Performance

The system implements multiple machine learning approaches:

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | Baseline | - | - | - | - |
| Random Forest | Ensemble | - | - | - | - |
| XGBoost | Gradient Boosting | - | - | - | - |
| LightGBM | Gradient Boosting | - | - | - | - |
| Neural Network | Deep Learning | - | - | - | - |

*Performance metrics are generated during training and displayed in the interactive dashboard.*

## Configuration

The system uses YAML configuration files for easy customization:

### Data Configuration
- Number of samples and regions
- Temporal and spatial parameters
- Train/validation/test splits

### Model Configuration
- Baseline model selection
- Neural network architecture
- Training hyperparameters

### Geographic Configuration
- Coordinate reference systems
- Region boundaries
- Grid resolution

### Visualization Configuration
- Color schemes
- Map settings
- Plot parameters

## Interactive Demo

The Streamlit demo provides:

1. **Data Generation**: Create synthetic flood datasets
2. **Model Training**: Train all ML models
3. **Risk Mapping**: Interactive flood risk visualization
4. **Time Series Analysis**: Detailed hydrological variable tracking
5. **Performance Comparison**: Model benchmarking
6. **Analysis Tools**: Feature correlation and risk distribution

### Demo Features

- **Real-time Risk Assessment**: Adjustable risk thresholds
- **Spatial Visualization**: Interactive maps with risk categories
- **Temporal Analysis**: Time series plots for selected regions
- **Model Comparison**: Performance metrics and leaderboards
- **Feature Analysis**: Correlation matrices and importance plots

## Training and Evaluation

### Training Commands

```bash
# Generate data and train models
python -m src.data.pipeline
python -m src.models.flood_models

# Run interactive demo
streamlit run demo/app.py
```

### Evaluation Metrics

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Spatial Metrics**: Regional performance analysis
- **Temporal Metrics**: Time-based validation
- **Risk Assessment**: Threshold-based evaluation

## Data Sources and Generation

This system uses **synthetic data** generated to simulate real-world flood scenarios:

- **Hydrological Patterns**: Seasonal rainfall and river level variations
- **Spatial Correlation**: Geographic proximity effects
- **Temporal Dependencies**: Time series relationships
- **Risk Factors**: Multi-factor flood risk determination

### Data Generation Process

1. **Spatial Locations**: Random points within defined geographic bounds
2. **Hydrological Time Series**: Seasonal patterns with realistic correlations
3. **Feature Engineering**: Temporal lags, rolling statistics, derived features
4. **Risk Labeling**: Multi-factor flood risk determination

## Known Limitations

- **Synthetic Data**: Uses generated data, not real-world measurements
- **Simplified Hydrology**: Simplified hydrological processes
- **Limited Spatial Resolution**: Coarse spatial grid
- **No Real-time Integration**: No live data feeds

## Disclaimer

**IMPORTANT**: This system is designed for **research and educational purposes only**. It should not be used for:

- Operational flood prediction
- Emergency response decisions
- Real-world risk assessment
- Public safety applications

For operational flood prediction, consult official meteorological and hydrological services.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**kryptologyst**  
GitHub: [https://github.com/kryptologyst](https://github.com/kryptologyst)

## Issues and Support

For questions, issues, or contributions, please visit:
- GitHub Issues: [https://github.com/kryptologyst](https://github.com/kryptologyst)
- Project Repository: [Repository URL]

## Acknowledgments

- Synthetic data generation techniques
- Machine learning model implementations
- Geospatial visualization libraries
- Streamlit framework for interactive demos
# Flood-Prediction-System
