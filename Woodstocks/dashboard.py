import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
st.set_page_config(
    page_title="Woodstocks TSLA AI Trader",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
STOCK_SYMBOL = "TSLA"
MODEL_PATH = "models/best_model.pkl"
METADATA_PATH = "models/model_metadata.json"
DAYS_TO_FETCH = 90  # Fixed analysis period

# Title
st.title("🚀 Woodstocks TSLA AI Trading Dashboard")

@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_data
def get_tsla_data():
    end_date = datetime.today()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    try:
        return yf.download(STOCK_SYMBOL, start=start_date, end=end_date).reset_index()
    except Exception as e:
        st.error(f"Error fetching TSLA data: {e}")
        return None

def generate_features(data):
    if data is None:
        return None
    
    df = data.copy()
    
    # Ensure basic columns exist
    for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            return None
    
    # Generate features sequentially
    try:
        # Basic technical features
        df['daily_return'] = df['Close'].pct_change()
        df['5_day_ma'] = df['Close'].rolling(5).mean()
        df['10_day_volatility'] = df['Close'].pct_change().rolling(10).std()
        
        # Sentiment features (replace with real data)
        df['avg_sentiment'] = np.random.uniform(-0.2, 0.2, len(df))
        df['tweet_count'] = np.random.randint(50, 150, len(df))
        
        # Features that depend on sentiment
        df['sentiment_lag1'] = df['avg_sentiment'].shift(1)
        df['sentiment_ma3'] = df['avg_sentiment'].rolling(3).mean()
        
        # Target variable (not used for prediction)
        df['next_day_change'] = df['Close'].shift(-1) - df['Close']
        
        return df.dropna()
    except Exception as e:
        st.error(f"Error generating features: {e}")
        return None

def make_prediction(model, features):
    try:
        # Remove next_day_change for prediction
        pred_features = {k: v for k, v in features.items() if k != 'next_day_change'}
        pred = model.predict(pd.DataFrame([pred_features]))[0]
        return {
            'value': float(pred),
            'direction': 'UP' if pred > 0 else 'DOWN',
            'confidence': min(99, round(abs(pred)*100, 1))
        }
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main dashboard
model, metadata = load_model()
tsla_data = get_tsla_data()

if tsla_data is not None and model is not None:
    features_df = generate_features(tsla_data)
    
    if features_df is not None:
        # Make predictions
        predictions = []
        for _, row in features_df.iterrows():
            pred = make_prediction(model, row.to_dict())
            predictions.append(pred if pred else {'value': 0, 'direction': 'HOLD', 'confidence': 0})
        
        # Merge predictions with main data
        tsla_data = tsla_data.iloc[-len(predictions):].copy()
        tsla_data['Prediction'] = [p['value'] for p in predictions]
        tsla_data['Direction'] = [p['direction'] for p in predictions]
        tsla_data['Confidence'] = [p['confidence'] for p in predictions]

        # Market Overview
        st.header("📈 Market Overview")
        latest = tsla_data.iloc[-1]
        prev_close = tsla_data.iloc[-2]['Close'] if len(tsla_data) > 1 else latest['Close']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${latest['Close']:.2f}", 
                   f"{(latest['Close'] - prev_close):.2f} ({(latest['Close'] - prev_close)/prev_close*100:.2f}%)")
        col2.metric("Prediction", 
                   f"{'↑' if latest['Direction'] == 'UP' else '↓'} {abs(latest['Prediction']):.2f}%",
                   f"Confidence: {latest['Confidence']}%")
        col3.metric("Volume", f"{latest['Volume']/1e6:.2f}M")
        
        # Price Chart with Predictions
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(
            x=tsla_data['Date'],
            open=tsla_data['Open'],
            high=tsla_data['High'],
            low=tsla_data['Low'],
            close=tsla_data['Close'],
            name='Price'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=tsla_data['Date'],
            y=tsla_data['Close'],
            mode='markers',
            marker=dict(
                color=['green' if p['direction'] == 'UP' else 'red' for p in predictions],
                size=10,
                line=dict(width=1, color='black')
            ),
            name='Predictions'
        ), row=1, col=1)
        
        fig.add_trace(go.Bar(
            x=tsla_data['Date'],
            y=tsla_data['Volume'],
            name='Volume',
            marker_color='rgba(100, 150, 200, 0.6)'
        ), row=2, col=1)
        
        fig.update_layout(height=600, title_text="TSLA Price with Predictions")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment Analysis
        st.header("📊 Sentiment Analysis")
        if 'avg_sentiment' in features_df.columns:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(
                x=tsla_data['Date'],
                y=tsla_data['Close'],
                name='Price'
            ), secondary_y=False)
            
            fig.add_trace(go.Bar(
                x=tsla_data['Date'],
                y=features_df['avg_sentiment'],
                name='Sentiment',
                marker_color=['green' if x > 0 else 'red' for x in features_df['avg_sentiment']]
            ), secondary_y=True)
            
            fig.update_layout(height=400, title_text="Price vs Sentiment")
            st.plotly_chart(fig, use_container_width=True)
        
        # Model Insights
        st.header("🤖 Model Insights")
        if hasattr(model, 'feature_importances_') and metadata:
            features = [f for f in metadata['features'] if f != 'next_day_change']
            importance = model.feature_importances_[:len(features)]  # Ensure same length
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=importance, y=features, ax=ax)
            ax.set_title('Feature Importance')
            st.pyplot(fig)
        
        # Data Download
        st.sidebar.download_button(
            "Download Data",
            tsla_data.to_csv(index=False),
            "tsla_analysis.csv",
            "text/csv"
        )
    else:
        st.error("Failed to generate features for analysis")
else:
    st.error("Failed to load required data or model")