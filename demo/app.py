"""
Flood Prediction System - Streamlit Demo Application

Interactive web application for flood risk assessment and visualization.
"""

import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data.pipeline import FloodDataGenerator
from src.models.flood_models import FloodPredictionModels
from src.viz.visualization import FloodVisualizer

# Page configuration
st.set_page_config(
    page_title="Flood Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-low { color: #2E8B57; }
    .risk-medium { color: #FFD700; }
    .risk-high { color: #FF4500; }
    .risk-extreme { color: #8B0000; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'models' not in st.session_state:
    st.session_state.models = None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🌊 Flood Prediction System</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Flood Prediction System! This interactive dashboard provides comprehensive 
    flood risk assessment using machine learning models trained on synthetic hydrological data.
    
    **Features:**
    - Real-time flood risk mapping
    - Time series analysis of hydrological variables
    - Multiple ML model comparison
    - Interactive risk threshold adjustment
    """)
    
    # Sidebar
    st.sidebar.header("🎛️ Controls")
    
    # Data generation section
    st.sidebar.subheader("Data & Models")
    
    if st.sidebar.button("🔄 Generate New Dataset", help="Generate fresh synthetic flood data"):
        with st.spinner("Generating dataset..."):
            generator = FloodDataGenerator()
            st.session_state.dataset = generator.generate_dataset()
            st.session_state.data_generated = True
            st.session_state.models_trained = False
        st.success("Dataset generated successfully!")
    
    if st.sidebar.button("🤖 Train All Models", help="Train all ML models"):
        if st.session_state.data_generated:
            with st.spinner("Training models..."):
                models = FloodPredictionModels()
                st.session_state.models = models
                st.session_state.models.train_all_models(
                    st.session_state.dataset['X_train'],
                    st.session_state.dataset['X_val'],
                    st.session_state.dataset['X_test'],
                    st.session_state.dataset['y_train'],
                    st.session_state.dataset['y_val'],
                    st.session_state.dataset['y_test']
                )
                st.session_state.models_trained = True
            st.success("Models trained successfully!")
        else:
            st.error("Please generate dataset first!")
    
    # Risk threshold
    st.sidebar.subheader("Risk Assessment")
    threshold = st.sidebar.slider(
        "🚨 Risk Alert Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum probability for flood risk alert"
    )
    
    # Region selection
    if st.session_state.data_generated:
        regions = st.session_state.dataset['spatial_data']['region_id'].tolist()
        selected_region = st.sidebar.selectbox(
            "📍 Select Region",
            options=regions,
            help="Choose a region for detailed analysis"
        )
    else:
        selected_region = None
    
    # Main content
    if not st.session_state.data_generated:
        show_welcome_screen()
    else:
        show_main_dashboard(threshold, selected_region)

def show_welcome_screen():
    """Show welcome screen with instructions."""
    
    st.info("👆 **Get Started**: Click 'Generate New Dataset' in the sidebar to begin!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 System Overview")
        st.markdown("""
        This flood prediction system uses machine learning to assess flood risk based on:
        
        **Hydrological Variables:**
        - Rainfall intensity and patterns
        - River water levels
        - Soil moisture content
        - Surface runoff rates
        - Temperature and wind conditions
        
        **Spatial Features:**
        - Elevation and terrain slope
        - Drainage basin characteristics
        - Geographic location
        """)
    
    with col2:
        st.subheader("🔬 Technical Approach")
        st.markdown("""
        **Models Implemented:**
        - Logistic Regression (baseline)
        - Random Forest (ensemble)
        - XGBoost (gradient boosting)
        - LightGBM (gradient boosting)
        - Neural Network (deep learning)
        
        **Evaluation Metrics:**
        - Accuracy, Precision, Recall
        - F1-Score, AUC-ROC
        - Spatial cross-validation
        - Temporal validation
        """)
    
    st.subheader("⚠️ Important Disclaimer")
    st.warning("""
    **Research Demonstration Only**: This system is designed for educational and research purposes. 
    It uses synthetic data and should not be used for operational flood prediction or emergency response decisions.
    
    For real-world flood prediction, consult official meteorological and hydrological services.
    """)

def show_main_dashboard(threshold, selected_region):
    """Show main dashboard with data and results."""
    
    dataset = st.session_state.dataset
    spatial_data = dataset['spatial_data']
    processed_data = dataset['processed_data']
    
    # Generate sample predictions if models not trained
    if st.session_state.models_trained:
        # Use actual model predictions
        models = st.session_state.models
        # For demo purposes, use the best model's predictions
        best_model_name = models.get_model_leaderboard().iloc[0]['Model']
        best_model = models.models[best_model_name]
        
        if best_model_name == 'neural_network':
            # Neural network predictions
            X_test_scaled = models.scaler.transform(dataset['X_test'])
            X_test_tensor = torch.FloatTensor(X_test_scaled).to(models.device)
            best_model.eval()
            with torch.no_grad():
                probabilities = best_model(X_test_tensor).cpu().numpy().squeeze()
            predictions = (probabilities > threshold).astype(int)
        else:
            # Tree-based model predictions
            probabilities = best_model.predict_proba(dataset['X_test'])[:, 1]
            predictions = (probabilities > threshold).astype(int)
    else:
        # Generate sample predictions for demo
        n_test = len(dataset['X_test'])
        probabilities = np.random.beta(2, 5, n_test)  # Skewed towards lower values
        predictions = (probabilities > threshold).astype(int)
    
    # Dashboard metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Regions", len(spatial_data))
    
    with col2:
        high_risk_count = np.sum(probabilities >= threshold)
        st.metric("High Risk Regions", high_risk_count)
    
    with col3:
        avg_risk = np.mean(probabilities)
        st.metric("Average Risk", f"{avg_risk:.1%}")
    
    with col4:
        max_risk = np.max(probabilities)
        st.metric("Max Risk", f"{max_risk:.1%}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Risk Map", "📊 Time Series", "📈 Model Performance", "🔍 Analysis"])
    
    with tab1:
        show_risk_map(spatial_data, probabilities, predictions, threshold)
    
    with tab2:
        if selected_region is not None:
            show_time_series(processed_data, selected_region)
        else:
            st.info("Select a region from the sidebar to view time series analysis.")
    
    with tab3:
        if st.session_state.models_trained:
            show_model_performance()
        else:
            st.info("Train models using the sidebar to view performance metrics.")
    
    with tab4:
        show_analysis_tools(processed_data, probabilities, threshold)

def show_risk_map(spatial_data, probabilities, predictions, threshold):
    """Display flood risk map."""
    
    st.subheader("🌍 Flood Risk Assessment Map")
    
    # Risk statistics
    risk_counts = {
        'Low Risk (< 30%)': np.sum(probabilities < 0.3),
        'Medium Risk (30-60%)': np.sum((probabilities >= 0.3) & (probabilities < 0.6)),
        'High Risk (60-80%)': np.sum((probabilities >= 0.6) & (probabilities < 0.8)),
        'Extreme Risk (> 80%)': np.sum(probabilities >= 0.8)
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create map
        visualizer = FloodVisualizer()
        risk_map = visualizer.create_flood_risk_map(
            spatial_data, predictions, probabilities
        )
        
        # Display map
        st.components.v1.html(risk_map._repr_html_(), height=600)
    
    with col2:
        st.subheader("Risk Distribution")
        
        # Risk distribution chart
        risk_df = pd.DataFrame(list(risk_counts.items()), columns=['Risk Level', 'Count'])
        fig = px.pie(risk_df, values='Count', names='Risk Level', 
                    color_discrete_map={
                        'Low Risk (< 30%)': '#2E8B57',
                        'Medium Risk (30-60%)': '#FFD700',
                        'High Risk (60-80%)': '#FF4500',
                        'Extreme Risk (> 80%)': '#8B0000'
                    })
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk statistics
        st.subheader("Risk Statistics")
        for risk_level, count in risk_counts.items():
            percentage = count / len(probabilities) * 100
            st.write(f"**{risk_level}**: {count} regions ({percentage:.1f}%)")

def show_time_series(processed_data, region_id):
    """Display time series analysis for selected region."""
    
    st.subheader(f"📈 Time Series Analysis - Region {region_id}")
    
    region_data = processed_data[processed_data['region_id'] == region_id].copy()
    region_data = region_data.sort_values('time_step')
    
    # Create time series plot
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Rainfall and River Level', 'Soil Moisture and Runoff', 
                       'Temperature and Wind', 'Flood Risk'),
        vertical_spacing=0.08
    )
    
    # Rainfall and River Level
    fig.add_trace(
        go.Scatter(x=region_data['time_step'], y=region_data['rainfall'],
                  name='Rainfall (mm)', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=region_data['time_step'], y=region_data['river_level'],
                  name='River Level (m)', line=dict(color='cyan')),
        row=1, col=1
    )
    
    # Soil Moisture and Runoff
    fig.add_trace(
        go.Scatter(x=region_data['time_step'], y=region_data['soil_moisture'],
                  name='Soil Moisture', line=dict(color='brown')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=region_data['time_step'], y=region_data['runoff_rate'],
                  name='Runoff Rate (mm/hr)', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Temperature and Wind
    fig.add_trace(
        go.Scatter(x=region_data['time_step'], y=region_data['temperature'],
                  name='Temperature (°C)', line=dict(color='red')),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=region_data['time_step'], y=region_data['wind_speed'],
                  name='Wind Speed (m/s)', line=dict(color='green')),
        row=3, col=1
    )
    
    # Flood Risk
    fig.add_trace(
        go.Scatter(x=region_data['time_step'], y=region_data['flood_risk'],
                  name='Flood Risk', line=dict(color='darkred'), fill='tonexty'),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title=f"Region {region_id} - Hydrological Time Series")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Max Rainfall", f"{region_data['rainfall'].max():.1f} mm")
        st.metric("Max River Level", f"{region_data['river_level'].max():.2f} m")
    
    with col2:
        st.metric("Avg Soil Moisture", f"{region_data['soil_moisture'].mean():.2f}")
        st.metric("Max Runoff Rate", f"{region_data['runoff_rate'].max():.1f} mm/hr")
    
    with col3:
        st.metric("Flood Risk Days", f"{region_data['flood_risk'].sum()}")
        st.metric("Risk Rate", f"{region_data['flood_risk'].mean():.1%}")

def show_model_performance():
    """Display model performance metrics."""
    
    st.subheader("🤖 Model Performance Comparison")
    
    models = st.session_state.models
    leaderboard = models.get_model_leaderboard()
    
    # Performance table
    st.dataframe(leaderboard.round(4), use_container_width=True)
    
    # Performance visualization
    metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC']
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            name=metric,
            x=leaderboard['Model'],
            y=leaderboard[metric],
            text=leaderboard[metric].round(3),
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model info
    best_model = leaderboard.iloc[0]
    st.success(f"🏆 **Best Model**: {best_model['Model']} (F1-Score: {best_model['Test F1']:.3f})")

def show_analysis_tools(processed_data, probabilities, threshold):
    """Show analysis tools and insights."""
    
    st.subheader("🔍 Analysis Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Correlation Analysis")
        
        # Select numerical features for correlation
        numerical_features = ['rainfall', 'river_level', 'soil_moisture', 
                             'runoff_rate', 'temperature', 'wind_speed', 
                             'elevation', 'slope', 'drainage_area']
        
        corr_data = processed_data[numerical_features + ['flood_risk']].corr()
        
        fig = px.imshow(corr_data, 
                       text_auto=True, 
                       aspect="auto",
                       color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution Analysis")
        
        # Risk distribution histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=probabilities,
            nbinsx=20,
            name='Risk Distribution',
            marker_color='lightblue'
        ))
        
        # Add threshold line
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"Threshold: {threshold:.2f}")
        
        fig.update_layout(
            title='Flood Risk Probability Distribution',
            xaxis_title='Risk Probability',
            yaxis_title='Frequency',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights
    st.subheader("💡 Key Insights")
    
    insights = [
        f"**Risk Threshold**: {threshold:.1%} of regions exceed the current threshold",
        f"**Risk Distribution**: {np.sum(probabilities < 0.3)} regions have low risk, {np.sum(probabilities >= 0.8)} have extreme risk",
        f"**Data Coverage**: {len(processed_data)} time steps across {processed_data['region_id'].nunique()} regions",
        f"**Model Status**: {'Trained' if st.session_state.models_trained else 'Not trained'} - {'Ready for prediction' if st.session_state.models_trained else 'Generate data and train models first'}"
    ]
    
    for insight in insights:
        st.write(insight)

if __name__ == "__main__":
    main()
