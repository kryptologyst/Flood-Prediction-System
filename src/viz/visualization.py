"""
Flood Prediction System - Visualization Module

This module provides comprehensive visualization capabilities for flood prediction
results, including interactive maps, time series plots, and risk dashboards.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
import streamlit as st
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


class FloodVisualizer:
    """Comprehensive visualization tools for flood prediction results."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize visualizer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Color scheme
        self.colors = self.config['viz']['flood_colors']
        
    def create_flood_risk_map(self, spatial_data: gpd.GeoDataFrame, 
                            predictions: np.ndarray, 
                            probabilities: np.ndarray,
                            save_path: Optional[str] = None) -> folium.Map:
        """Create interactive flood risk map.
        
        Args:
            spatial_data: GeoDataFrame with spatial locations
            predictions: Binary flood predictions
            probabilities: Flood risk probabilities
            save_path: Optional path to save the map
            
        Returns:
            Folium map object
        """
        # Create base map
        map_center = self.config['viz']['map_center']
        map_zoom = self.config['viz']['map_zoom']
        
        m = folium.Map(
            location=map_center,
            zoom_start=map_zoom,
            tiles='OpenStreetMap'
        )
        
        # Add satellite tiles
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        
        # Create risk categories
        def get_risk_category(prob):
            if prob < 0.3:
                return 'Low Risk'
            elif prob < 0.6:
                return 'Medium Risk'
            elif prob < 0.8:
                return 'High Risk'
            else:
                return 'Extreme Risk'
        
        def get_risk_color(prob):
            if prob < 0.3:
                return self.colors['low_risk']
            elif prob < 0.6:
                return self.colors['medium_risk']
            elif prob < 0.8:
                return self.colors['high_risk']
            else:
                return self.colors['extreme_risk']
        
        # Add markers for each location
        for idx, row in spatial_data.iterrows():
            if idx < len(predictions):
                prob = probabilities[idx]
                pred = predictions[idx]
                risk_cat = get_risk_category(prob)
                color = get_risk_color(prob)
                
                # Create popup text
                popup_text = f"""
                <b>Region {row['region_id']}</b><br>
                Risk Level: {risk_cat}<br>
                Probability: {prob:.3f}<br>
                Prediction: {'Flood Risk' if pred else 'Safe'}<br>
                Elevation: {row['elevation']:.1f}m<br>
                Slope: {row['slope']:.1f}°
                """
                
                # Add marker
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=200),
                    color='black',
                    weight=1,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Flood Risk Legend</b></p>
        <p><i class="fa fa-circle" style="color:''' + self.colors['low_risk'] + '''"></i> Low Risk (< 30%)</p>
        <p><i class="fa fa-circle" style="color:''' + self.colors['medium_risk'] + '''"></i> Medium Risk (30-60%)</p>
        <p><i class="fa fa-circle" style="color:''' + self.colors['high_risk'] + '''"></i> High Risk (60-80%)</p>
        <p><i class="fa fa-circle" style="color:''' + self.colors['extreme_risk'] + '''"></i> Extreme Risk (> 80%)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
            logger.info(f"Map saved to {save_path}")
        
        return m
    
    def plot_time_series(self, data: pd.DataFrame, region_id: int = 0,
                        save_path: Optional[str] = None) -> go.Figure:
        """Create time series plot for a specific region.
        
        Args:
            data: DataFrame with time series data
            region_id: ID of the region to plot
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        region_data = data[data['region_id'] == region_id].copy()
        region_data = region_data.sort_values('time_step')
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Rainfall and River Level', 'Soil Moisture and Runoff', 'Flood Risk'),
            vertical_spacing=0.1
        )
        
        # Rainfall and River Level
        fig.add_trace(
            go.Scatter(
                x=region_data['time_step'],
                y=region_data['rainfall'],
                name='Rainfall (mm)',
                line=dict(color='blue'),
                yaxis='y1'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=region_data['time_step'],
                y=region_data['river_level'],
                name='River Level (m)',
                line=dict(color='cyan'),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Soil Moisture and Runoff
        fig.add_trace(
            go.Scatter(
                x=region_data['time_step'],
                y=region_data['soil_moisture'],
                name='Soil Moisture',
                line=dict(color='brown'),
                yaxis='y3'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=region_data['time_step'],
                y=region_data['runoff_rate'],
                name='Runoff Rate (mm/hr)',
                line=dict(color='orange'),
                yaxis='y4'
            ),
            row=2, col=1
        )
        
        # Flood Risk
        fig.add_trace(
            go.Scatter(
                x=region_data['time_step'],
                y=region_data['flood_risk'],
                name='Flood Risk',
                line=dict(color='red'),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'Flood Risk Time Series - Region {region_id}',
            height=800,
            showlegend=True
        )
        
        # Update y-axes
        fig.update_yaxes(title_text="Rainfall (mm)", row=1, col=1)
        fig.update_yaxes(title_text="River Level (m)", row=1, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Soil Moisture", row=2, col=1)
        fig.update_yaxes(title_text="Runoff Rate (mm/hr)", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Flood Risk", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Time series plot saved to {save_path}")
        
        return fig
    
    def plot_model_performance(self, results: Dict, save_path: Optional[str] = None) -> go.Figure:
        """Create model performance comparison plot.
        
        Args:
            results: Dictionary with model results
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        # Prepare data
        data = []
        for model in models:
            for metric in metrics:
                data.append({
                    'Model': model,
                    'Metric': metric.upper(),
                    'Train': results[model]['train'][metric],
                    'Validation': results[model]['val'][metric],
                    'Test': results[model]['test'][metric]
                })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        for split in ['Train', 'Validation', 'Test']:
            fig.add_trace(go.Bar(
                name=split,
                x=df['Metric'],
                y=df[split],
                text=df[split].round(3),
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group',
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Performance plot saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self, model, feature_names: List[str], 
                              save_path: Optional[str] = None) -> go.Figure:
        """Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            save_path: Optional path to save the plot
            
        Returns:
            Plotly figure object
        """
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not have feature importance or coefficients")
            return None
        
        # Sort features by importance
        sorted_idx = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]
        
        fig = go.Figure(go.Bar(
            x=sorted_importance,
            y=sorted_features,
            orientation='h',
            text=sorted_importance.round(4),
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Feature Importance',
            xaxis_title='Importance',
            yaxis_title='Features',
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig
    
    def create_risk_dashboard(self, spatial_data: gpd.GeoDataFrame,
                            predictions: np.ndarray,
                            probabilities: np.ndarray,
                            data: pd.DataFrame) -> None:
        """Create comprehensive risk dashboard using Streamlit.
        
        Args:
            spatial_data: GeoDataFrame with spatial locations
            predictions: Binary flood predictions
            probabilities: Flood risk probabilities
            data: DataFrame with time series data
        """
        st.set_page_config(
            page_title="Flood Risk Dashboard",
            page_icon="🌊",
            layout="wide"
        )
        
        st.title("🌊 Flood Prediction System Dashboard")
        st.markdown("---")
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Risk threshold slider
        threshold = st.sidebar.slider(
            "Risk Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum probability threshold for flood risk alert"
        )
        
        # Region selector
        regions = spatial_data['region_id'].tolist()
        selected_region = st.sidebar.selectbox(
            "Select Region",
            options=regions,
            help="Choose a region to view detailed time series"
        )
        
        # Model selector
        model_options = ['All Models', 'Logistic Regression', 'Random Forest', 
                        'XGBoost', 'LightGBM', 'Neural Network']
        selected_model = st.sidebar.selectbox(
            "Select Model",
            options=model_options,
            help="Choose a model to view performance"
        )
        
        # Main dashboard
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Flood Risk Map")
            
            # Filter by threshold
            high_risk_mask = probabilities >= threshold
            high_risk_count = np.sum(high_risk_mask)
            
            st.info(f"Regions with risk ≥ {threshold:.1%}: {high_risk_count}")
            
            # Create and display map
            risk_map = self.create_flood_risk_map(
                spatial_data, predictions, probabilities
            )
            
            # Display map in Streamlit
            st.components.v1.html(risk_map._repr_html_(), height=500)
        
        with col2:
            st.subheader("Risk Statistics")
            
            # Risk distribution
            risk_counts = pd.Series([
                np.sum(probabilities < 0.3),
                np.sum((probabilities >= 0.3) & (probabilities < 0.6)),
                np.sum((probabilities >= 0.6) & (probabilities < 0.8)),
                np.sum(probabilities >= 0.8)
            ], index=['Low', 'Medium', 'High', 'Extreme'])
            
            st.bar_chart(risk_counts)
            
            # Summary statistics
            st.metric("Total Regions", len(spatial_data))
            st.metric("High Risk Regions", high_risk_count)
            st.metric("Average Risk", f"{probabilities.mean():.1%}")
            st.metric("Max Risk", f"{probabilities.max():.1%}")
        
        # Time series section
        st.subheader("Time Series Analysis")
        
        if selected_region is not None:
            time_series_fig = self.plot_time_series(data, selected_region)
            st.plotly_chart(time_series_fig, use_container_width=True)
        
        # Model performance section
        st.subheader("Model Performance")
        
        # This would be populated with actual model results
        st.info("Model performance metrics would be displayed here based on the selected model.")
        
        # Footer
        st.markdown("---")
        st.markdown(
            "**Disclaimer**: This is a research demonstration system. "
            "Not intended for operational flood prediction. "
            "Author: [kryptologyst](https://github.com/kryptologyst)"
        )


if __name__ == "__main__":
    # Example usage
    from src.data.pipeline import FloodDataGenerator
    
    # Generate sample data
    generator = FloodDataGenerator()
    dataset = generator.generate_dataset()
    
    # Create visualizer
    visualizer = FloodVisualizer()
    
    # Create sample predictions
    n_samples = len(dataset['spatial_data'])
    sample_predictions = np.random.randint(0, 2, n_samples)
    sample_probabilities = np.random.uniform(0, 1, n_samples)
    
    # Create map
    risk_map = visualizer.create_flood_risk_map(
        dataset['spatial_data'], 
        sample_predictions, 
        sample_probabilities,
        save_path="assets/flood_risk_map.html"
    )
    
    print("Visualization components created successfully!")
