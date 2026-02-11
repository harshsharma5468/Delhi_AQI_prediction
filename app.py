import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
import joblib
from pathlib import Path

# ─── THEME CONFIGURATION ──────────────────────────────────────────────────
# Force Plotly to use dark mode internal logic
pio.templates.default = "plotly_dark"

st.set_page_config(
    page_title="Delhi AQI Predictor",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── PREMIUM DARK CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    /* Global Overrides */
    .stApp {
        background-color: #0F172A;
    }
    
    h1, h2, h3, h4, p, span {
        color: #F8FAFC !important;
        font-family: 'Inter', sans-serif;
    }

    /* Custom Masthead */
    .masthead {
        background: linear-gradient(90deg, #1E3A8A, #3B82F6);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }

    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: 0.3s;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
    }

    /* Prediction Result Box */
    .prediction-box {
        background: #1E293B;
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px solid #3B82F6;
        text-align: center;
        margin-top: 1rem;
    }

    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── DATA & MODEL ENGINE ─────────────────────────────────────────────────
class DelhiAQIPredictor:
    def __init__(self):
        self.load_data()
        self.load_models()
        
    def load_data(self):
        try:
            self.df = pd.read_csv('aqi_data.csv')
            self.df['datetime_utc'] = pd.to_datetime(self.df['datetime_utc'], utc=True)
            # Add time features for model
            self.df['hour'] = self.df['datetime_utc'].dt.hour
            self.df['day'] = self.df['datetime_utc'].dt.day
            self.df['month'] = self.df['datetime_utc'].dt.month
            st.session_state.df = self.df
        except:
            st.error("Missing aqi_data.csv. Please run 'fetch_api.py' first.")
            st.stop()
    
    def load_models(self):
        try:
            self.model = joblib.load('models/trained_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
        except:
            self.model = None
            self.scaler = None
    
    def predict_aqi(self, input_features):
        if self.model is None or self.scaler is None: return None
        scaled = self.scaler.transform([input_features])
        return self.model.predict(scaled)[0]
    
    def get_aqi_info(self, pm25):
        if pm25 <= 30: return "Good", "#10B981", "😊"
        elif pm25 <= 60: return "Satisfactory", "#FBBF24", "🙂"
        elif pm25 <= 90: return "Moderate", "#F59E0B", "😐"
        elif pm25 <= 120: return "Poor", "#EF4444", "😷"
        elif pm25 <= 250: return "Very Poor", "#8B5CF6", "🤢"
        else: return "Severe", "#DC2626", "🚨"

# ─── PLOTLY THEME HELPER ──────────────────────────────────────────────────
def apply_custom_chart_theme(fig):
    """Ensures visibility on dark background"""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F8FAFC"),
        xaxis=dict(gridcolor="#1E293B", linecolor="#334155"),
        yaxis=dict(gridcolor="#1E293B", linecolor="#334155"),
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig

# ─── PAGE RENDERERS ───────────────────────────────────────────────────────
def show_dashboard(predictor):
    latest = predictor.df.iloc[-1]
    pm25_val = latest.get('pm25', 0)
    cat, color, emoji = predictor.get_aqi_info(pm25_val)
    
    # KPI Rows
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">AVG PM2.5</div><div class="metric-value">{predictor.df["pm25"].mean():.1f}</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">CURRENT PM2.5</div><div class="metric-value" style="color:{color}">{pm25_val:.1f}</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">HEALTH STATUS</div><div class="metric-value" style="color:{color}">{cat} {emoji}</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">DATA POINTS</div><div class="metric-value">{len(predictor.df)}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📈 Recent PM2.5 History")
        fig = px.area(predictor.df.tail(100), x='datetime_utc', y='pm25', markers=True)
        fig.update_traces(line_color='#3B82F6', fillcolor='rgba(59, 130, 246, 0.2)')
        # IMPORTANT: theme=None prevents Streamlit from breaking the chart visibility
        st.plotly_chart(apply_custom_chart_theme(fig), use_container_width=True, theme=None)
    
    with col2:
        st.markdown("### 📊 Pollutant Split")
        pollutants = ['pm25', 'pm10', 'no2', 'o3', 'co']
        avg_vals = [predictor.df[p].mean() for p in pollutants if p in predictor.df.columns]
        fig_pie = px.pie(values=avg_vals, names=pollutants, hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(apply_custom_chart_theme(fig_pie), use_container_width=True, theme=None)

def show_prediction(predictor):
    st.markdown("## 🤖 AI Prediction Engine")
    c1, c2 = st.columns([2, 1])
    with c1:
        with st.container(border=True):
            co = st.slider("CO (μg/m³)", 0.0, 10000.0, 2500.0)
            no2 = st.slider("NO₂ (μg/m³)", 0.0, 500.0, 60.0)
            pm10 = st.slider("PM10 (μg/m³)", 0.0, 1000.0, 350.0)
            hour = st.slider("Hour of Day", 0, 23, 12)
            month = st.selectbox("Select Month", range(1, 13), index=1)
        
        btn = st.button("🚀 Predict AQI", type="primary", use_container_width=True)

    with c2:
        if btn:
            # Feature order: [co, no, no2, o3, so2, pm10, nh3, hour, day, month]
            features = [co, 2.0, no2, 15.0, 10.0, pm10, 20.0, hour, 15, month]
            pred = predictor.predict_aqi(features)
            if pred:
                cat, color, emoji = predictor.get_aqi_info(pred)
                st.markdown(f"""
                <div class="prediction-box">
                    <h4 style="color:#94A3B8; margin-bottom:0.5rem">PREDICTED PM2.5</h4>
                    <h1 style="font-size:4.5rem; color:{color}; margin:0">{pred:.1f}</h1>
                    <h2 style="color:{color}">{cat} {emoji}</h2>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Model files missing in /models folder.")

# ─── MAIN APP ROUTER ─────────────────────────────────────────────────────
def main():
    predictor = DelhiAQIPredictor()
    
    st.markdown('<div class="masthead"><h1>🌫️ Delhi AQI Intelligence</h1><p>Full City Average • Real-time Insights</p></div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### 🛠️ NAVIGATION")
        page = st.radio("", ["🏠 Dashboard", "🤖 Predict AQI", "📊 Raw Data Analysis"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### 📡 SYSTEM STATUS")
        st.success("● API: OpenWeather Active")
        if st.button("⟳ Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    if page == "🏠 Dashboard": show_dashboard(predictor)
    elif page == "🤖 Predict AQI": show_prediction(predictor)
    elif page == "📊 Raw Data Analysis": st.dataframe(predictor.df, use_container_width=True)

if __name__ == "__main__":
    main()