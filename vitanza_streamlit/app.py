import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Vitanza", layout="wide", page_icon="logo.jpg")

# ---------- Enhanced Styling ----------
st.markdown("""
<style>
/* ===== GLOBAL THEME ===== */
html, body, [class*="st-"] {
    font-family: 'Segoe UI', sans-serif;
}

.main-header {
  font-size: 40px;
  font-weight: 800;
  color: #004e89; /* Deep River Blue */
  text-align: center;
  margin-bottom: 8px;
}

.subtitle {
  font-size: 17px;
  color: #0077b6; /* Medium Blue */
  text-align: center;
  margin-bottom: 20px;
}

/* ===== KPI Cards ===== */
.kpi-card {
  background: linear-gradient(135deg, rgba(0,119,182,0.12), rgba(0,119,182,0.04));
  padding: 18px;
  border-radius: 12px;
  border-left: 5px solid #0077b6;
  box-shadow: 0 2px 6px rgba(0,0,0,0.08);
  color: #004e89;
}

/* ===== Alerts (Only Blue + Green Family) ===== */
.alert-high {
  background: #0d47a1;  /* Dark Blue = High Priority */
  color: white;
  padding: 12px;
  border-radius: 8px;
  font-weight: 600;
  text-align: center;
}
.alert-medium {
  background: #0077b6;  /* Medium Blue = Medium Priority */
  color: white;
  padding: 12px;
  border-radius: 8px;
  font-weight: 600;
  text-align: center;
}
.alert-low {
  background: #2e8b57; /* Green = Stable/Low */
  color: white;
  padding: 12px;
  border-radius: 8px;
  font-weight: 600;
  text-align: center;
}

/* ===== Section Headings ===== */
.section-header {
  font-size: 24px;
  font-weight: 700;
  color: #004e89;
  margin-top: 20px;
  margin-bottom: 12px;
  border-bottom: 2px solid #0077b6;
  padding-bottom: 4px;
}

/* ===== Cards ===== */
.card {
  background: white;
  border-radius: 10px;
  border: 1px solid rgba(0,119,182,0.2);
  padding: 15px;
  transition: 0.2s ease-in-out;
}
.card:hover {
  box-shadow: 0 4px 15px rgba(0,119,182,0.2);
  transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)


# ---------- Data Generation Functions ----------
@st.cache_data
def generate_river_sensor_nodes(n=12, center=(23.0225, 72.5714), spread=0.025, seed=42):
    """Generate sensor node locations across Sabarmati Riverfront"""
    rng = np.random.RandomState(seed)
    lats = center[0] + rng.uniform(-spread, spread, n)
    lons = center[1] + rng.uniform(-spread, spread, n)
    nodes = []
    zones = ["Ellis Bridge", "Nehru Bridge", "Gandhi Bridge", "Sardar Bridge", 
             "Subhash Bridge", "Flower Park", "Event Ground", "Riverfront West",
             "Riverfront East", "Vasna Barrage", "Upper Section", "Lower Section"]
    for i, (la, lo) in enumerate(zip(lats, lons)):
        nodes.append({
            "id": f"SRF_{i+1:02d}",
            "name": zones[i] if i < len(zones) else f"Node_{i+1}",
            "lat": float(la),
            "lon": float(lo)
        })
    return pd.DataFrame(nodes)

@st.cache_data
def generate_comprehensive_riverfront_data(nodes, hours=336, seed=10):
    """Generate comprehensive water quality and monitoring data"""
    rng = np.random.RandomState(seed)
    ts = pd.date_range(end=pd.Timestamp.now(), periods=hours, freq="H")
    records = []
    
    for _, node in nodes.iterrows():
        base_level = 3.5 + rng.normal(0, 0.3)
        base_ph = 7.2 + rng.normal(0, 0.2)
        
        for t in ts:
            hour_factor = np.sin((t.hour / 24.0) * 2 * np.pi)
            day_factor = np.sin((t.dayofyear / 365.0) * 2 * np.pi)
            
            # Water level with seasonal and daily patterns
            level = max(0.5, base_level + 0.8 * hour_factor + 0.5 * day_factor + rng.normal(0, 0.2))
            
            # Water quality parameters
            ph = np.clip(base_ph + rng.normal(0, 0.4), 6.0, 9.0)
            do = np.clip(6.5 + rng.normal(0, 1.2), 2.0, 12.0)
            bod = np.clip(8 + rng.normal(0, 3), 0, 30)
            cod = np.clip(25 + rng.normal(0, 8), 0, 150)
            turbidity = np.clip(15 + rng.normal(0, 5), 0, 100)
            tds = np.clip(500 + rng.normal(0, 100), 100, 2000)
            temperature = np.clip(25 + 5 * day_factor + rng.normal(0, 2), 15, 40)
            
            # Microbial load (CFU/100ml)
            coliform = max(0, int(rng.lognormal(4, 1)))
            
            # Pollution indicators
            floating_waste = int(rng.rand() < 0.15)
            plastic_count = int(rng.poisson(2)) if floating_waste else 0
            sewage_detected = int(rng.rand() < 0.08)
            
            # Biodiversity and ecosystem health
            biodiv_score = np.clip(rng.beta(5, 2), 0, 1)
            fish_count = int(rng.poisson(12 * biodiv_score))
            aquatic_plants = int(rng.poisson(8 * biodiv_score))
            
            # Safety metrics
            crowd_density = max(0, int(rng.poisson(50 * (1 + 0.5 * hour_factor))))
            unsafe_activity = int(rng.rand() < 0.03)
            
            records.append({
                "node_id": node["id"],
                "node_name": node["name"],
                "timestamp": t,
                "lat": node["lat"],
                "lon": node["lon"],
                "water_level_m": level,
                "ph": ph,
                "do_mg_l": do,
                "bod_mg_l": bod,
                "cod_mg_l": cod,
                "turbidity_ntu": turbidity,
                "tds_mg_l": tds,
                "temperature_c": temperature,
                "coliform_cfu": coliform,
                "floating_waste": floating_waste,
                "plastic_count": plastic_count,
                "sewage_detected": sewage_detected,
                "biodiversity_score": biodiv_score,
                "fish_count": fish_count,
                "aquatic_plants": aquatic_plants,
                "crowd_density": crowd_density,
                "unsafe_activity": unsafe_activity
            })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_weather_data(hours=336, seed=20):
    """Generate weather and hydrological data"""
    rng = np.random.RandomState(seed)
    ts = pd.date_range(end=pd.Timestamp.now(), periods=hours, freq="H")
    records = []
    
    for t in ts:
        rainfall = max(0, rng.gamma(2, 3) if rng.rand() < 0.15 else 0)
        humidity = np.clip(60 + rng.normal(0, 15), 30, 100)
        wind_speed = max(0, rng.gamma(3, 2))
        
        records.append({
            "timestamp": t,
            "rainfall_mm": rainfall,
            "humidity_pct": humidity,
            "wind_speed_kmh": wind_speed
        })
    
    return pd.DataFrame(records)

@st.cache_data
def generate_citizen_reports(n=50, seed=30):
    """Generate simulated citizen reports"""
    rng = np.random.RandomState(seed)
    ts = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="6H")
    
    report_types = ["Floating Waste", "Water Pollution", "Unsafe Area", 
                   "Sewage Smell", "Dead Fish", "Broken Fence", "Illegal Dumping"]
    statuses = ["Pending", "In Progress", "Resolved"]
    priorities = ["High", "Medium", "Low"]
    
    records = []
    for t in ts:
        records.append({
            "report_id": f"CR_{len(records)+1:04d}",
            "timestamp": t,
            "reporter_name": f"Citizen_{rng.randint(1,100)}",
            "report_type": rng.choice(report_types),
            "description": "Issue reported by citizen",
            "location": f"Near {rng.choice(['Ellis Bridge', 'Gandhi Bridge', 'Flower Park', 'Event Ground'])}",
            "priority": rng.choice(priorities, p=[0.2, 0.5, 0.3]),
            "status": rng.choice(statuses, p=[0.3, 0.4, 0.3]),
            "upvotes": int(rng.poisson(5))
        })
    
    return pd.DataFrame(records)

# ---------- ML Models ----------
@st.cache_data
def train_flood_prediction_model(df_merged):
    """Train flood prediction model using water level, rainfall, and other factors"""
    df = df_merged.copy()
    df = df.dropna()
    
    # Create target: flood risk in next 24 hours
    df = df.sort_values("timestamp")
    df["future_level"] = df.groupby("node_id")["water_level_m"].shift(-24)
    df["flood_risk"] = ((df["future_level"] > 6.0) | (df["rainfall_mm"] > 50)).astype(int)
    
    df = df.dropna()
    
    features = ["water_level_m", "rainfall_mm", "humidity_pct", "wind_speed_kmh", 
                "do_mg_l", "temperature_c"]
    X = df[features]
    y = df["flood_risk"]
    
    if len(df[y == 1]) < 5:
        y = ((df["water_level_m"] > 5.5) | (df["rainfall_mm"] > 40)).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    return model, score, features

@st.cache_data
def train_pollution_alert_model(df):
    """Train model to detect pollution events"""
    df = df.copy()
    
    # Define pollution event based on multiple parameters
    df["pollution_event"] = (
        (df["do_mg_l"] < 4.0) | 
        (df["bod_mg_l"] > 15) | 
        (df["cod_mg_l"] > 50) |
        (df["coliform_cfu"] > 5000) |
        (df["sewage_detected"] == 1)
    ).astype(int)
    
    features = ["ph", "do_mg_l", "bod_mg_l", "cod_mg_l", "turbidity_ntu", 
                "tds_mg_l", "coliform_cfu"]
    X = df[features]
    y = df["pollution_event"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    return model, score, features

# ---------- Analysis Functions ----------
def calculate_water_quality_index(row):
    """Calculate comprehensive water quality index (0-100)"""
    # pH score (optimal 6.5-8.5)
    ph_score = 100 if 6.5 <= row["ph"] <= 8.5 else max(0, 100 - abs(7.5 - row["ph"]) * 20)
    
    # DO score (>6 is good)
    do_score = min(100, (row["do_mg_l"] / 6.0) * 100) if row["do_mg_l"] <= 6 else 100
    
    # BOD score (lower is better, <3 is excellent)
    bod_score = max(0, 100 - (row["bod_mg_l"] / 30.0) * 100)
    
    # COD score (lower is better, <20 is good)
    cod_score = max(0, 100 - (row["cod_mg_l"] / 150.0) * 100)
    
    # Coliform score (lower is better)
    coliform_score = max(0, 100 - min(100, (row["coliform_cfu"] / 10000) * 100))
    
    # Weighted average
    wqi = (ph_score * 0.2 + do_score * 0.25 + bod_score * 0.2 + 
           cod_score * 0.2 + coliform_score * 0.15)
    
    return round(wqi, 1)

def get_wqi_category(wqi):
    """Categorize water quality index"""
    if wqi >= 80:
        return "Excellent", "🟢"
    elif wqi >= 60:
        return "Good", "🟡"
    elif wqi >= 40:
        return "Moderate", "🟠"
    else:
        return "Poor", "🔴"

def calculate_pollution_score(row):
    """Calculate pollution score for heatmap"""
    score = 0
    score += max(0, (7.5 - row["ph"]) * 10) + max(0, (row["ph"] - 7.5) * 10)
    score += max(0, (6 - row["do_mg_l"]) * 3)
    score += row["bod_mg_l"] * 0.5
    score += row["cod_mg_l"] * 0.15
    score += row["turbidity_ntu"] * 0.2
    score += row["sewage_detected"] * 10
    score += row["plastic_count"] * 2
    return min(100, score)

# ---------- Initialize Data ----------
@st.cache_data
def load_all_data():
    nodes = generate_river_sensor_nodes(n=12)
    df = generate_comprehensive_riverfront_data(nodes, hours=336)
    weather_df = generate_weather_data(hours=336)
    citizen_df = generate_citizen_reports(n=50)
    
    # Merge weather data
    df_merged = df.merge(weather_df, on="timestamp", how="left")
    
    # Calculate WQI
    df_merged["wqi"] = df_merged.apply(calculate_water_quality_index, axis=1)
    df_merged["pollution_score"] = df_merged.apply(calculate_pollution_score, axis=1)
    
    return nodes, df_merged, citizen_df

nodes, df, citizen_df = load_all_data()

# Initialize session state
if "live_df" not in st.session_state:
    st.session_state["live_df"] = df.copy()
if "live_citizen_df" not in st.session_state:
    st.session_state["live_citizen_df"] = citizen_df.copy()

# ---------- Sidebar Navigation ----------
st.sidebar.image("https://via.placeholder.com/250x80/03396c/ffffff?text=Sabarmati+Smart+System", use_container_width=True)
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "",
    ["🏠 Dashboard", "💧 Water Quality", "🌊 Flood Management", 
     "🗑️ Waste Management", "👥 Citizen Engagement", "🛡️ Safety & Surveillance",
     "🌿 Biodiversity", "📊 Analytics & Reports"],
    label_visibility="collapsed"
)

# ---------- PAGES ----------

if page == "🏠 Dashboard":
    st.markdown('<div class="main-header">🌊 Sabarmati Riverfront Smart Management System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Integrated IoT, AI, and Citizen-Driven Solution for Sustainable Urban Water Body Management</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Real-time KPIs
    recent_df = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=6))]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_wqi = recent_df["wqi"].mean()
        cat, icon = get_wqi_category(avg_wqi)
        st.markdown(f'<div class="kpi-card"><b>Water Quality Index</b><br><h2>{icon} {avg_wqi:.1f}</h2><small>{cat}</small></div>', unsafe_allow_html=True)
    
    with col2:
        avg_level = recent_df["water_level_m"].mean()
        st.markdown(f'<div class="kpi-card"><b>Avg Water Level</b><br><h2>{avg_level:.2f} m</h2><small>Last 6 hours</small></div>', unsafe_allow_html=True)
    
    with col3:
        total_waste = recent_df["plastic_count"].sum()
        st.markdown(f'<div class="kpi-card"><b>Floating Waste</b><br><h2>{int(total_waste)}</h2><small>Detected items</small></div>', unsafe_allow_html=True)
    
    with col4:
        active_sensors = len(recent_df["node_id"].unique())
        st.markdown(f'<div class="kpi-card"><b>Active Sensors</b><br><h2>{active_sensors}/12</h2><small>IoT Network</small></div>', unsafe_allow_html=True)
    
    with col5:
        pending_reports = len(citizen_df[citizen_df["status"] == "Pending"])
        st.markdown(f'<div class="kpi-card"><b>Citizen Reports</b><br><h2>{pending_reports}</h2><small>Pending action</small></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Interactive Map
    st.markdown('<div class="section-header">📍 Real-Time Sensor Network & Pollution Heatmap</div>', unsafe_allow_html=True)
    
    latest = df.sort_values("timestamp").groupby("node_id").tail(1).reset_index(drop=True)
    
    col_map, col_table = st.columns([3, 1])
    
    with col_map:
        deck = pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=latest["lat"].mean(),
                longitude=latest["lon"].mean(),
                zoom=12.5,
                pitch=40
            ),
            layers=[
                pdk.Layer(
                    "HeatmapLayer",
                    data=latest,
                    get_position='[lon, lat]',
                    get_weight="pollution_score",
                    radiusPixels=80,
                    intensity=1,
                    threshold=0.05
                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=latest,
                    get_position='[lon, lat]',
                    get_fill_color='[200, 30, 0, 160]',
                    get_radius=120,
                    pickable=True
                )
            ],
            tooltip={"html": "<b>{node_name}</b><br/>WQI: {wqi}<br/>Pollution Score: {pollution_score:.1f}", "style": {"color": "white"}}
        )
        st.pydeck_chart(deck, use_container_width=True)
    
    with col_table:
        st.markdown("**🔴 High Priority Nodes**")
        priority_nodes = latest.sort_values("pollution_score", ascending=False)[["node_name", "wqi", "pollution_score"]].head(5)
        priority_nodes.columns = ["Location", "WQI", "Pollution"]
        st.dataframe(priority_nodes, hide_index=True)
    
    st.markdown("---")
    
    # Quick Stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">📈 Trend Analysis (Last 7 Days)</div>', unsafe_allow_html=True)
        week_df = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=7))]
        daily_wqi = week_df.groupby(week_df["timestamp"].dt.date)["wqi"].mean().reset_index()
        daily_wqi.columns = ["Date", "WQI"]
        
        fig = px.line(daily_wqi, x="Date", y="WQI", markers=True,
                     title="Average Water Quality Index - Weekly Trend")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">⚠️ Active Alerts</div>', unsafe_allow_html=True)
        
        alerts = []
        if latest["wqi"].min() < 40:
            alerts.append("🔴 Critical water quality detected at some nodes")
        if latest["water_level_m"].max() > 5.5:
            alerts.append("🟡 Elevated water levels - monitor for flooding")
        if latest["sewage_detected"].sum() > 0:
            alerts.append("🔴 Sewage contamination detected")
        if latest["plastic_count"].sum() > 20:
            alerts.append("🟠 High waste accumulation - cleanup needed")
        
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("✅ No critical alerts - all systems normal")
        
        st.markdown("**System Status**")
        st.info(f"🟢 All {len(nodes)} sensor nodes operational\n\n🟢 AI models active and updated\n\n🟢 Real-time monitoring enabled")

elif page == "💧 Water Quality":
    st.markdown('<div class="section-header">💧 Comprehensive Water Quality Monitoring</div>', unsafe_allow_html=True)
    st.markdown("Real-time monitoring of pH, DO, BOD, COD, turbidity, TDS, and microbial load with AI-powered contamination alerts.")
    
    st.markdown("---")
    
    # Overall metrics
    recent = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=6))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg pH", f"{recent['ph'].mean():.2f}", delta=f"{recent['ph'].mean() - 7.5:.2f}")
    col2.metric("Avg DO (mg/L)", f"{recent['do_mg_l'].mean():.2f}", delta="Good" if recent['do_mg_l'].mean() > 5 else "Low")
    col3.metric("Avg BOD (mg/L)", f"{recent['bod_mg_l'].mean():.2f}")
    col4.metric("Avg Turbidity (NTU)", f"{recent['turbidity_ntu'].mean():.1f}")
    
    st.markdown("---")
    
    # Node selection
    st.markdown("### 🔍 Node-Specific Analysis")
    selected_node = st.selectbox("Select monitoring location:", options=sorted(df["node_name"].unique()))
    
    node_df = df[df["node_name"] == selected_node].sort_values("timestamp")
    latest_reading = node_df.iloc[-1]
    
    # Current status
    col1, col2 = st.columns([2, 1])
    
    with col1:
        wqi = latest_reading["wqi"]
        cat, icon = get_wqi_category(wqi)
        
        if cat == "Excellent":
            st.markdown(f'<div class="alert-low">{icon} Water Quality: {cat} (WQI: {wqi})</div>', unsafe_allow_html=True)
        elif cat in ["Good", "Moderate"]:
            st.markdown(f'<div class="alert-medium">{icon} Water Quality: {cat} (WQI: {wqi})</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-high">{icon} Water Quality: {cat} (WQI: {wqi}) - IMMEDIATE ACTION REQUIRED</div>', unsafe_allow_html=True)
    
    with col2:
        if latest_reading["sewage_detected"] == 1:
            st.error("🚨 Sewage contamination detected!")
        else:
            st.success("✅ No sewage detected")
    
    # Detailed parameters
    st.markdown("### 📊 Detailed Parameters (Latest Reading)")
    
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)
    
    param_col1.metric("pH", f"{latest_reading['ph']:.2f}", 
                     delta="Normal" if 6.5 <= latest_reading['ph'] <= 8.5 else "Abnormal")
    param_col2.metric("DO (mg/L)", f"{latest_reading['do_mg_l']:.2f}",
                     delta="Good" if latest_reading['do_mg_l'] > 5 else "Low")
    param_col3.metric("BOD (mg/L)", f"{latest_reading['bod_mg_l']:.2f}",
                     delta="Good" if latest_reading['bod_mg_l'] < 10 else "High")
    param_col4.metric("COD (mg/L)", f"{latest_reading['cod_mg_l']:.2f}",
                     delta="Good" if latest_reading['cod_mg_l'] < 40 else "High")
    
    param_col1.metric("Turbidity (NTU)", f"{latest_reading['turbidity_ntu']:.1f}")
    param_col2.metric("TDS (mg/L)", f"{latest_reading['tds_mg_l']:.0f}")
    param_col3.metric("Temperature (°C)", f"{latest_reading['temperature_c']:.1f}")
    param_col4.metric("Coliform (CFU/100ml)", f"{latest_reading['coliform_cfu']:.0f}",
                     delta="Safe" if latest_reading['coliform_cfu'] < 2500 else "High")
    
    # Time series
    st.markdown("### 📈 Historical Trends (Last 7 Days)")
    
    week_data = node_df[node_df["timestamp"] >= (node_df["timestamp"].max() - pd.Timedelta(days=7))]
    
    param_choice = st.selectbox("Select parameter to visualize:", 
                               ["ph", "do_mg_l", "bod_mg_l", "cod_mg_l", "turbidity_ntu", "wqi"])
    
    fig = px.line(week_data, x="timestamp", y=param_choice, 
                 title=f"{param_choice.upper()} - 7 Day Trend",
                 markers=True)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # AI Pollution Detection
    st.markdown("---")
    st.markdown("### 🤖 AI-Powered Pollution Alert System")
    
    try:
        pollution_model, pollution_score, pollution_features = train_pollution_alert_model(df)
        st.info(f"Model Accuracy: {pollution_score:.2%}")
        
        # Predict for latest reading
        X_new = latest_reading[pollution_features].values.reshape(1, -1)
        prediction = pollution_model.predict(X_new)[0]
        prob = pollution_model.predict_proba(X_new)[0, 1]
        
        if prediction == 1:
            st.error(f"🚨 POLLUTION EVENT DETECTED (Confidence: {prob:.1%})")
            st.markdown("**Recommended Actions:**")
            st.markdown("- Dispatch field team for verification")
            st.markdown("- Identify pollution source")
            st.markdown("- Alert downstream communities")
            st.markdown("- Increase sampling frequency")
        else:
            st.success(f"✅ No pollution event detected (Confidence: {1-prob:.1%})")
    except Exception as e:
        st.warning(f"Model training in progress...")

elif page == "🌊 Flood Management":
    st.markdown('<div class="section-header">🌊 Flood Prediction & Water Level Management</div>', unsafe_allow_html=True)
    st.markdown("AI-based forecasting integrated with weather data and hydrological patterns for early flood warnings.")
    
    st.markdown("---")
    
    # Current status
    recent = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=6))]
    weather_recent = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=24))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Avg Level", f"{recent['water_level_m'].mean():.2f} m")
    col2.metric("Max Level (24h)", f"{weather_recent['water_level_m'].max():.2f} m")
    col3.metric("Rainfall (24h)", f"{weather_recent['rainfall_mm'].sum():.1f} mm")
    
    max_level = weather_recent['water_level_m'].max()
    if max_level > 6.0:
        col4.markdown('<div class="alert-high">🔴 HIGH FLOOD RISK</div>', unsafe_allow_html=True)
    elif max_level > 5.0:
        col4.markdown('<div class="alert-medium">🟡 MODERATE RISK</div>', unsafe_allow_html=True)
    else:
        col4.markdown('<div class="alert-low">🟢 LOW RISK</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI Flood Prediction
    st.markdown("### 🤖 AI Flood Risk Prediction Model")
    
    try:
        flood_model, flood_score, flood_features = train_flood_prediction_model(df)
        st.info(f"Model Accuracy: {flood_score:.2%} | Features: {', '.join(flood_features)}")
        
        # Predict for each node
        latest = df.sort_values("timestamp").groupby("node_id").tail(1).reset_index(drop=True)
        
        predictions = []
        for _, row in latest.iterrows():
            X_pred = row[flood_features].values.reshape(1, -1)
            risk_prob = flood_model.predict_proba(X_pred)[0, 1]
            predictions.append({
                "Location": row["node_name"],
                "Current Level (m)": round(row["water_level_m"], 2),
                "Flood Risk %": round(risk_prob * 100, 1),
                "Status": "🔴 High" if risk_prob > 0.6 else "🟡 Medium" if risk_prob > 0.3 else "🟢 Low"
            })
        
        pred_df = pd.DataFrame(predictions).sort_values("Flood Risk %", ascending=False)
        st.dataframe(pred_df, hide_index=True, use_container_width=True)
        
        # Alert for high-risk nodes
        high_risk = pred_df[pred_df["Flood Risk %"] > 60]
        if len(high_risk) > 0:
            st.error(f"⚠️ {len(high_risk)} location(s) at HIGH flood risk - activate emergency protocols!")
            st.markdown("**Immediate Actions Required:**")
            st.markdown("- Deploy emergency teams to high-risk zones")
            st.markdown("- Issue public alerts via SMS/app")
            st.markdown("- Prepare evacuation plans")
            st.markdown("- Coordinate with disaster management authorities")
        
    except Exception as e:
        st.warning("Model training in progress...")
    
    st.markdown("---")
    
    # Node-specific analysis
    st.markdown("### 📍 Location-Specific Forecast")
    
    selected_node = st.selectbox("Select location for detailed forecast:", 
                                options=sorted(df["node_name"].unique()), key="flood_node")
    
    node_df = df[df["node_name"] == selected_node].sort_values("timestamp")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Water level trend
        week_data = node_df[node_df["timestamp"] >= (node_df["timestamp"].max() - pd.Timedelta(days=7))]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=week_data["timestamp"], y=week_data["water_level_m"],
                                mode='lines+markers', name='Water Level',
                                line=dict(color='blue', width=2)))
        fig.add_hline(y=6.0, line_dash="dash", line_color="red", 
                     annotation_text="Flood Warning Level")
        fig.add_hline(y=5.0, line_dash="dash", line_color="orange", 
                     annotation_text="Alert Level")
        fig.update_layout(title="Water Level - 7 Day Trend", 
                         xaxis_title="Time", yaxis_title="Water Level (m)",
                         height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Rainfall Impact Analysis**")
        recent_rain = node_df[node_df["timestamp"] >= (node_df["timestamp"].max() - pd.Timedelta(hours=24))]
        total_rain = recent_rain["rainfall_mm"].sum()
        
        st.metric("24h Rainfall", f"{total_rain:.1f} mm")
        
        if total_rain > 50:
            st.error("🔴 Heavy rainfall detected")
        elif total_rain > 20:
            st.warning("🟡 Moderate rainfall")
        else:
            st.success("🟢 Light/No rainfall")
        
        st.markdown("**Water Level Stats**")
        st.write(f"Current: {node_df.iloc[-1]['water_level_m']:.2f} m")
        st.write(f"24h Max: {recent_rain['water_level_m'].max():.2f} m")
        st.write(f"24h Avg: {recent_rain['water_level_m'].mean():.2f} m")
    
    # Emergency preparedness
    st.markdown("---")
    st.markdown("### 🚨 Emergency Preparedness Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📞 Emergency Contacts**")
        st.info("Fire: 101\nPolice: 100\nAmbulance: 108\nDisaster Mgmt: 1070")
    
    with col2:
        st.markdown("**🏥 Evacuation Centers**")
        st.info("1. Municipal School - Ellis Bridge\n2. Community Hall - Gandhi Bridge\n3. Sports Complex - Flower Park")
    
    with col3:
        st.markdown("**📦 Resources Status**")
        st.success("✅ Boats: 5 ready\n✅ Sandbags: 1000+\n✅ Rescue team: On standby")

elif page == "🗑️ Waste Management":
    st.markdown('<div class="section-header">🗑️ Smart Waste Detection & Cleanup Management</div>', unsafe_allow_html=True)
    st.markdown("AI-enabled floating waste detection, tracking, and automated cleanup scheduling system.")
    
    st.markdown("---")
    
    # Waste overview
    recent = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=72))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Waste Items (72h)", int(recent["plastic_count"].sum()))
    col2.metric("Hotspot Locations", int((recent.groupby("node_id")["plastic_count"].sum() > 10).sum()))
    col3.metric("Sewage Incidents", int(recent["sewage_detected"].sum()))
    col4.metric("Cleanup Efficiency", "87%")
    
    st.markdown("---")
    
    # Waste hotspot map
    st.markdown("### 🗺️ Waste Hotspot Detection Map")
    
    waste_agg = recent.groupby(["node_id", "node_name", "lat", "lon"]).agg({
        "plastic_count": "sum",
        "floating_waste": "sum"
    }).reset_index()
    
    waste_agg["size"] = waste_agg["plastic_count"] * 50
    
    fig = px.scatter_mapbox(waste_agg, lat="lat", lon="lon", 
                           size="size", color="plastic_count",
                           hover_name="node_name",
                           hover_data={"plastic_count": True, "lat": False, "lon": False},
                           color_continuous_scale="Reds",
                           zoom=12, height=450)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Priority cleanup zones
    st.markdown("### 🎯 Priority Cleanup Zones")
    
    priority = waste_agg.sort_values("plastic_count", ascending=False).head(10)
    priority["Priority"] = ["🔴 Critical" if x > 20 else "🟡 High" if x > 10 else "🟢 Medium" 
                           for x in priority["plastic_count"]]
    
    display_priority = priority[["node_name", "plastic_count", "Priority"]].copy()
    display_priority.columns = ["Location", "Waste Items", "Priority Level"]
    st.dataframe(display_priority, hide_index=True, use_container_width=True)
    
    # Cleanup scheduling
    st.markdown("---")
    st.markdown("### 🚤 Cleanup Resource Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Resources**")
        st.info("🚤 Cleanup Boats: 3\n🤖 Waste Collector Drones: 2\n👷 Field Teams: 5\n📦 Collection Capacity: 500 kg/day")
        
        if st.button("🎯 Auto-Schedule Cleanup for Top 3 Hotspots"):
            top3 = priority.head(3)
            st.success("✅ Cleanup scheduled successfully!")
            for idx, row in top3.iterrows():
                st.write(f"📍 {row['node_name']}: Boat Team A - Tomorrow 6:00 AM")
    
    with col2:
        st.markdown("**Waste Analytics**")
        
        # Waste type distribution (simulated)
        waste_types = pd.DataFrame({
            "Type": ["Plastic Bottles", "Bags", "Food Waste", "Other Plastics", "Debris"],
            "Percentage": [35, 25, 15, 15, 10]
        })
        
        fig = px.pie(waste_types, values="Percentage", names="Type", 
                    title="Waste Composition")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # AI Waste Detection Status
    st.markdown("---")
    st.markdown("### 🤖 AI Waste Detection System")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Detection Accuracy", "94.3%")
    col2.metric("Images Processed (24h)", "15,240")
    col3.metric("False Positives", "3.2%")
    
    st.info("💡 System uses computer vision to detect and classify floating waste from drone and stationary camera feeds in real-time.")

elif page == "👥 Citizen Engagement":
    st.markdown('<div class="section-header">👥 Citizen Engagement & Reporting Platform</div>', unsafe_allow_html=True)
    st.markdown("Empowering citizens to report issues, volunteer for cleanup drives, and participate in river conservation.")
    
    st.markdown("---")
    
    # Citizen report form
    st.markdown("### 📝 Submit a Report")
    
    with st.form("citizen_report_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            reporter_name = st.text_input("Your Name (Optional)", placeholder="Anonymous")
            report_type = st.selectbox("Issue Type", 
                                      ["Floating Waste", "Water Pollution", "Unsafe Area", 
                                       "Sewage Smell", "Dead Fish", "Broken Infrastructure", 
                                       "Illegal Dumping", "Other"])
        
        with col2:
            contact = st.text_input("Contact Number (Optional)", placeholder="For updates")
            location = st.text_input("Location / Nearest Landmark", 
                                    placeholder="e.g., Near Gandhi Bridge")
        
        description = st.text_area("Describe the Issue", 
                                   placeholder="Please provide details to help us address the issue quickly...")
        
        priority = st.select_slider("Urgency Level", 
                                    options=["Low", "Medium", "High", "Critical"])
        
        photo_upload = st.file_uploader("Upload Photo (Optional)", type=["jpg", "jpeg", "png"])
        
        submitted = st.form_submit_button("🚀 Submit Report", use_container_width=True)
        
        if submitted:
            report_id = f"CR_{len(st.session_state['live_citizen_df']) + 1:04d}"
            new_report = {
                "report_id": report_id,
                "timestamp": pd.Timestamp.now(),
                "reporter_name": reporter_name if reporter_name else "Anonymous",
                "report_type": report_type,
                "description": description,
                "location": location,
                "priority": priority,
                "status": "Pending",
                "upvotes": 0
            }
            st.session_state['live_citizen_df'] = pd.concat([
                st.session_state['live_citizen_df'], 
                pd.DataFrame([new_report])
            ], ignore_index=True)
            
            st.success(f"✅ Thank you! Your report ({report_id}) has been submitted. Authorities will review it shortly.")
            st.balloons()
    
    st.markdown("---")
    
    # View existing reports
    st.markdown("### 📋 Recent Citizen Reports")
    
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        status_filter = st.multiselect("Filter by Status", 
                                      ["Pending", "In Progress", "Resolved"],
                                      default=["Pending", "In Progress"])
    
    with filter_col2:
        type_filter = st.multiselect("Filter by Type",
                                    st.session_state['live_citizen_df']["report_type"].unique(),
                                    default=st.session_state['live_citizen_df']["report_type"].unique())
    
    with filter_col3:
        sort_by = st.selectbox("Sort by", ["Timestamp", "Priority", "Upvotes"])
    
    filtered_reports = st.session_state['live_citizen_df'][
        (st.session_state['live_citizen_df']["status"].isin(status_filter)) &
        (st.session_state['live_citizen_df']["report_type"].isin(type_filter))
    ].copy()
    
    if sort_by == "Timestamp":
        filtered_reports = filtered_reports.sort_values("timestamp", ascending=False)
    elif sort_by == "Priority":
        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        filtered_reports["priority_num"] = filtered_reports["priority"].map(priority_order)
        filtered_reports = filtered_reports.sort_values("priority_num")
    else:
        filtered_reports = filtered_reports.sort_values("upvotes", ascending=False)
    
    # Display reports
    for _, report in filtered_reports.head(10).iterrows():
        with st.expander(f"🆔 {report['report_id']} - {report['report_type']} | {report['location']} | Status: {report['status']}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Reported by:** {report['reporter_name']}")
                st.write(f"**Time:** {report['timestamp']}")
                st.write(f"**Description:** {report['description']}")
            
            with col2:
                priority_color = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
                st.write(f"**Priority:** {priority_color.get(report['priority'], '⚪')} {report['priority']}")
                st.write(f"**👍 Upvotes:** {report['upvotes']}")
                
                if st.button("👍 Upvote", key=f"upvote_{report['report_id']}"):
                    idx = st.session_state['live_citizen_df'][
                        st.session_state['live_citizen_df']['report_id'] == report['report_id']
                    ].index[0]
                    st.session_state['live_citizen_df'].at[idx, 'upvotes'] += 1
                    st.rerun()
    
    st.markdown("---")
    
    # Volunteer program
    st.markdown("### 🌟 Volunteer for Cleanup Drives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Upcoming Cleanup Events**")
        st.info("📅 **Sunday, Oct 20** - Flower Park Zone\n⏰ 7:00 AM - 10:00 AM\n👥 25 volunteers registered")
        st.info("📅 **Saturday, Oct 26** - Gandhi Bridge Area\n⏰ 6:30 AM - 9:30 AM\n👥 18 volunteers registered")
        
        if st.button("🙋 Register as Volunteer"):
            st.success("✅ Thank you for volunteering! You'll receive event details via SMS/email.")
    
    with col2:
        st.markdown("**Impact Stats**")
        st.metric("Active Volunteers", "342")
        st.metric("Cleanup Events (2024)", "28")
        st.metric("Waste Collected (kg)", "1,847")
        st.metric("Hours Contributed", "2,156")
    
    st.markdown("---")
    
    # Education and awareness
    st.markdown("### 📚 Education & Awareness")
    
    tab1, tab2 = st.tabs(["Conservation Tips", "River Health Info"])
    
    with tab1:
        st.markdown("""
        **How You Can Help Keep Sabarmati Clean:**
        
        1. 🚯 **Don't litter** - Always use designated waste bins
        2. ♻️ **Recycle** - Separate plastic, paper, and organic waste
        3. 🚱 **Report pollution** - Use this platform to report issues immediately
        4. 🌱 **Participate** - Join cleanup drives and awareness campaigns
        5. 📢 **Spread awareness** - Share conservation tips with family and friends
        6. 💧 **Save water** - Reduce water wastage at home
        7. 🏭 **No dumping** - Report illegal waste dumping to authorities
        """)
    
    with tab2:
        st.markdown("""
        **Understanding River Health Indicators:**
        
        - **pH (6.5-8.5)**: Measures acidity/alkalinity. Extreme values harm aquatic life.
        - **Dissolved Oxygen (>5 mg/L)**: Essential for fish and ecosystem health.
        - **BOD (<10 mg/L)**: Lower values indicate cleaner water.
        - **Turbidity**: Measures water clarity. High turbidity blocks sunlight.
        - **Coliform (<2500 CFU/100ml)**: Indicates sewage contamination level.
        
        **Water Quality Index (WQI):**
        - 80-100: Excellent 🟢
        - 60-80: Good 🟡
        - 40-60: Moderate 🟠
        - <40: Poor 🔴
        """)

elif page == "🛡️ Safety & Surveillance":
    st.markdown('<div class="section-header">🛡️ Safety & Surveillance System</div>', unsafe_allow_html=True)
    st.markdown("AI-enabled surveillance for crowd management, drowning detection, and emergency response.")
    
    st.markdown("---")
    
    # Safety overview
    recent = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(hours=24))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Crowd Density", int(recent["crowd_density"].mean()))
    col2.metric("Safety Incidents (24h)", int(recent["unsafe_activity"].sum()))
    col3.metric("Active Cameras", "24/24")
    col4.metric("Response Time (avg)", "4.2 min")
    
    st.markdown("---")
    
    # Camera surveillance
    st.markdown("### 📹 Live Camera Surveillance")
    
    selected_camera = st.selectbox("Select Camera Location:", 
                                   options=sorted(df["node_name"].unique()),
                                   key="camera_select")
    
    camera_df = df[df["node_name"] == selected_camera].sort_values("timestamp")
    latest_cam = camera_df.iloc[-1]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Simulated camera feed
        st.image("https://via.placeholder.com/800x450/1a1a1a/ffffff?text=Live+Camera+Feed+-+" + selected_camera.replace(" ", "+"),
                use_container_width=True)
        
        st.markdown("**AI Detection Status:**")
        detections = []
        if latest_cam["crowd_density"] > 80:
            detections.append("🟡 High crowd density detected")
        if latest_cam["unsafe_activity"] == 1:
            detections.append("🔴 Unsafe activity detected - alert sent to patrol")
        if latest_cam["floating_waste"] == 1:
            detections.append("🟠 Floating waste visible in frame")
        
        if detections:
            for det in detections:
                st.warning(det)
        else:
            st.success("✅ Normal activity - no alerts")
    
    with col2:
        st.markdown("**Current Statistics**")
        st.metric("Crowd Count", int(latest_cam["crowd_density"]))
        st.metric("Water Level", f"{latest_cam['water_level_m']:.2f} m")
        
        if latest_cam["water_level_m"] > 5.5:
            st.error("⚠️ Water level high - restrict access")
        else:
            st.success("✅ Safe water level")
        
        st.markdown("**Camera Info**")
        st.write("Status: 🟢 Online")
        st.write("Resolution: 4K")
        st.write("AI Model: YOLOv8")
        st.write("Coverage: 180° view")
    
    st.markdown("---")
    
    # Drowning detection system
    st.markdown("### 🏊 AI Drowning Detection System")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Detection Accuracy", "96.7%")
    col2.metric("False Alarm Rate", "2.1%")
    col3.metric("Avg Response Time", "3.8 min")
    
    st.info("""
    **How it works:**
    - AI analyzes video feeds in real-time to detect distress patterns
    - Alerts patrol teams and nearby emergency responders instantly
    - Provides exact location coordinates for quick rescue
    - Integrates with emergency services for coordinated response
    """)
    
    # Simulated incident log
    st.markdown("**Recent Incidents & Responses**")
    incidents = pd.DataFrame([
        {"Time": "Oct 14, 10:35 AM", "Location": "Gandhi Bridge", "Type": "Potential Distress", 
         "Status": "✅ Resolved", "Response": "False alarm - swimmer waving"},
        {"Time": "Oct 13, 3:20 PM", "Location": "Flower Park", "Type": "Unsafe Swimming", 
         "Status": "✅ Resolved", "Response": "Patrol warned individuals"},
        {"Time": "Oct 12, 6:15 PM", "Location": "Ellis Bridge", "Type": "Crowd Alert", 
         "Status": "✅ Resolved", "Response": "Additional security deployed"}
    ])
    st.dataframe(incidents, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Emergency response
    st.markdown("### 🚨 Emergency Response System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Emergency Protocols**")
        st.markdown("""
        1. **Drowning Alert**: 
           - Lifeguard dispatch < 2 min
           - Ambulance alert
           - Crowd control activation
        
        2. **Flood Warning**:
           - Public announcement system
           - Area evacuation
           - Emergency services on standby
        
        3. **Crowd Management**:
           - Deploy additional security
           - Traffic control
           - Alternate route guidance
        """)
    
    with col2:
        st.markdown("**Resource Availability**")
        st.success("🏥 Ambulances: 3 on standby")
        st.success("👮 Security Personnel: 18 active")
        st.success("🚤 Rescue Boats: 4 ready")
        st.success("🚁 Drone Units: 2 operational")
        
        if st.button("🚨 TEST EMERGENCY ALERT"):
            st.error("⚠️ EMERGENCY ALERT ACTIVATED - All units notified")
            st.warning("This is a test alert. In real emergency, all response teams would be dispatched.")
    
    st.markdown("---")
    
    # Unsafe zones
    st.markdown("### ⚠️ Unsafe Zone Monitoring")
    
    unsafe_data = df.groupby("node_name").agg({
        "unsafe_activity": "sum",
        "water_level_m": "mean"
    }).reset_index()
    unsafe_data["risk_score"] = unsafe_data["unsafe_activity"] * 10 + (unsafe_data["water_level_m"] - 3) * 5
    unsafe_data = unsafe_data.sort_values("risk_score", ascending=False)
    
    display_unsafe = unsafe_data[["node_name", "unsafe_activity", "water_level_m", "risk_score"]].head(8)
    display_unsafe.columns = ["Location", "Incidents (7d)", "Avg Water Level (m)", "Risk Score"]
    display_unsafe["Risk Score"] = display_unsafe["Risk Score"].round(1)
    
    st.dataframe(display_unsafe, hide_index=True, use_container_width=True)
    
    st.warning("⚠️ Locations with Risk Score > 50 require increased surveillance and safety measures.")

elif page == "🌿 Biodiversity":
    st.markdown('<div class="section-header">🌿 Biodiversity & Ecosystem Health Monitoring</div>', unsafe_allow_html=True)
    st.markdown("AI-driven ecological assessment for sustaining aquatic life and environmental indicators.")
    
    st.markdown("---")
    
    # Biodiversity overview
    recent = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=7))]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Biodiversity Index", f"{recent['biodiversity_score'].mean():.2f}")
    col2.metric("Fish Species Count", "12")
    col3.metric("Aquatic Plant Species", "8")
    col4.metric("Bird Species (Observed)", "23")
    
    st.markdown("---")
    
    # Biodiversity map
    st.markdown("### 🗺️ Ecosystem Health Map")
    
    biodiv_agg = recent.groupby(["node_id", "node_name", "lat", "lon"]).agg({
        "biodiversity_score": "mean",
        "fish_count": "mean",
        "aquatic_plants": "mean"
    }).reset_index()
    
    fig = px.scatter_mapbox(biodiv_agg, lat="lat", lon="lon",
                           size="fish_count", color="biodiversity_score",
                           hover_name="node_name",
                           hover_data={"biodiversity_score": ":.2f", "fish_count": ":.0f", 
                                      "aquatic_plants": ":.0f", "lat": False, "lon": False},
                           color_continuous_scale="Greens",
                           zoom=12, height=450)
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Species tracking
    st.markdown("### 🐟 Species Diversity & Population Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Fish species
        st.markdown("**Fish Species Observed**")
        fish_species = pd.DataFrame({
            "Species": ["Catla", "Rohu", "Mrigal", "Tilapia", "Common Carp", 
                       "Silver Carp", "Grass Carp", "Others"],
            "Population": [850, 720, 680, 540, 420, 380, 290, 420],
            "Trend": ["📈", "📈", "➡️", "📉", "📈", "➡️", "📈", "➡️"]
        })
        st.dataframe(fish_species, hide_index=True, use_container_width=True)
    
    with col2:
        # Bird species
        st.markdown("**Bird Species (Migratory & Resident)**")
        bird_species = pd.DataFrame({
            "Species": ["Spot-billed Duck", "Common Teal", "Little Cormorant", 
                       "Pond Heron", "Kingfisher", "Others"],
            "Count": [124, 98, 156, 203, 87, 245],
            "Status": ["Migratory", "Migratory", "Resident", "Resident", "Resident", "Mixed"]
        })
        st.dataframe(bird_species, hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    # Aquatic plants
    st.markdown("### 🌱 Aquatic Vegetation Health")
    
    plant_health = pd.DataFrame({
        "Plant Type": ["Water Hyacinth", "Lotus", "Water Lilies", "Reeds", 
                      "Algae", "Submerged Plants"],
        "Coverage (%)": [15, 8, 12, 25, 10, 30],
        "Health Status": ["🟡 Moderate", "🟢 Good", "🟢 Good", "🟢 Good", 
                         "🟠 Needs Control", "🟢 Good"]
    })
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = px.bar(plant_health, x="Plant Type", y="Coverage (%)",
                    title="Aquatic Vegetation Coverage",
                    color="Coverage (%)", color_continuous_scale="Greens")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.dataframe(plant_health[["Plant Type", "Health Status"]], 
                    hide_index=True, use_container_width=True)
    
    st.info("⚠️ Water Hyacinth growth needs monitoring - can block sunlight and reduce oxygen if uncontrolled.")
    
    st.markdown("---")
    
    # Ecological indicators
    st.markdown("### 📊 Key Ecological Indicators")
    
    # Trend analysis
    week_biodiv = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=30))]
    daily_biodiv = week_biodiv.groupby(week_biodiv["timestamp"].dt.date).agg({
        "biodiversity_score": "mean",
        "fish_count": "mean",
        "aquatic_plants": "mean"
    }).reset_index()
    daily_biodiv.columns = ["Date", "Biodiversity Index", "Avg Fish Count", "Avg Plant Count"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_biodiv["Date"], y=daily_biodiv["Biodiversity Index"],
                            mode='lines+markers', name='Biodiversity Index',
                            line=dict(color='green', width=2)))
    fig.update_layout(title="Biodiversity Index - 30 Day Trend",
                     xaxis_title="Date", yaxis_title="Index (0-1)",
                     height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Conservation recommendations
    st.markdown("### 💡 Conservation Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Immediate Actions**")
        st.markdown("""
        1. 🌿 Control invasive Water Hyacinth growth
        2. 🐟 Continue fish stocking programs
        3. 🚯 Reduce plastic waste to protect aquatic life
        4. 💧 Maintain DO levels above 5 mg/L
        5. 🔬 Monthly biodiversity surveys
        """)
    
    with col2:
        st.markdown("**Long-term Initiatives**")
        st.markdown("""
        1. 🏞️ Create protected breeding zones
        2. 🌳 Riverbank vegetation restoration
        3. 🦅 Bird sanctuary development
        4. 📚 Public awareness campaigns
        5. 🤝 Partner with wildlife organizations
        """)
    
    # AI species identification
    st.markdown("---")
    st.markdown("### 📸 AI Species Identification")
    
    st.info("""
    **Upload photos for AI-powered species identification:**
    
    Our system can identify:
    - Fish species and estimate population
    - Bird species and migration patterns
    - Aquatic plants and health status
    - Invasive species alerts
    """)
    
    uploaded_image = st.file_uploader("Upload image for identification", 
                                     type=["jpg", "jpeg", "png"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", width=400)
        if st.button("🔍 Identify Species"):
            with st.spinner("Analyzing image..."):
                import time
                time.sleep(2)
            st.success("**Identified:** Common Carp (Cyprinus carpio)")
            st.write("Confidence: 94.3%")
            st.write("Population Status: Healthy")
            st.write("Conservation Status: Least Concern")

elif page == "📊 Analytics & Reports":
    st.markdown('<div class="section-header">📊 Analytics & Reports Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Comprehensive data analytics, insights, and exportable reports for decision-making.")
    
    st.markdown("---")
    
    # Report type selection
    report_type = st.selectbox("Select Report Type:", 
                              ["Executive Summary", "Water Quality Report", 
                               "Flood Risk Assessment", "Waste Management Report",
                               "Biodiversity Report", "Citizen Engagement Report"])
    
    if report_type == "Executive Summary":
        st.markdown("### 📋 Executive Summary Report")
        st.markdown(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Summary statistics
        week_data = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=7))]
        
        st.markdown("#### Key Performance Indicators (Last 7 Days)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Water Quality**")
            avg_wqi = week_data["wqi"].mean()
            st.metric("Average WQI", f"{avg_wqi:.1f}", 
                     delta=f"{avg_wqi - 70:.1f}")
            st.write(f"DO: {week_data['do_mg_l'].mean():.2f} mg/L")
            st.write(f"pH: {week_data['ph'].mean():.2f}")
            st.write(f"BOD: {week_data['bod_mg_l'].mean():.2f} mg/L")
        
        with col2:
            st.markdown("**Safety & Security**")
            st.metric("Incidents", int(week_data["unsafe_activity"].sum()))
            st.metric("Avg Crowd", int(week_data["crowd_density"].mean()))
            st.write(f"Camera Uptime: 99.8%")
            st.write(f"Response Time: 4.2 min")
        
        with col3:
            st.markdown("**Environmental**")
            st.metric("Biodiversity Index", f"{week_data['biodiversity_score'].mean():.2f}")
            st.metric("Waste Collected", f"{int(week_data['plastic_count'].sum())} items")
            st.write(f"Rainfall: {week_data['rainfall_mm'].sum():.1f} mm")
            st.write(f"Avg Temp: {week_data['temperature_c'].mean():.1f}°C")
        
        st.markdown("---")
        
        st.markdown("#### System Health")
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("✅ All 12 sensor nodes operational")
            st.success("✅ 24/24 cameras online")
            st.success("✅ AI models performing optimally")
            st.success("✅ Data pipeline healthy")
        
        with col2:
            st.info("📊 Data Points Collected: 241,920")
            st.info("🤖 AI Predictions Made: 18,450")
            st.info("👥 Citizen Reports: 28 received")
            st.info("🚨 Alerts Generated: 12")
        
        st.markdown("---")
        
        st.markdown("#### Recommendations")
        st.markdown("""
        1. **Water Quality**: Continue monitoring COD levels at Ellis Bridge - slight elevation detected
        2. **Waste Management**: Schedule cleanup at Gandhi Bridge area (high plastic accumulation)
        3. **Biodiversity**: Implement Water Hyacinth control measures in lower section
        4. **Safety**: Increase patrol frequency at Flower Park during evening hours
        5. **Infrastructure**: Sensor SRF_07 battery replacement due in 2 weeks
        """)
    
    elif report_type == "Water Quality Report":
        st.markdown("### 💧 Detailed Water Quality Report")
        
        # Date range selection
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input("Start Date", 
                                      value=pd.Timestamp.now() - pd.Timedelta(days=30))
        with date_col2:
            end_date = st.date_input("End Date", value=pd.Timestamp.now())
        
        report_data = df[
            (df["timestamp"].dt.date >= start_date) & 
            (df["timestamp"].dt.date <= end_date)
        ]
        
        st.markdown(f"**Analysis Period:** {start_date} to {end_date} ({len(report_data)} readings)")
        
        # Summary statistics
        st.markdown("#### Statistical Summary")
        
        params = ["ph", "do_mg_l", "bod_mg_l", "cod_mg_l", "turbidity_ntu", "coliform_cfu"]
        summary_stats = report_data[params].describe().T
        summary_stats.columns = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        summary_stats = summary_stats.round(2)
        st.dataframe(summary_stats, use_container_width=True)
        
        # Compliance analysis
        st.markdown("#### Regulatory Compliance")
        
        compliance_data = {
            "Parameter": ["pH", "DO", "BOD", "COD", "Coliform"],
            "Standard Limit": ["6.5-8.5", ">5 mg/L", "<10 mg/L", "<40 mg/L", "<2500 CFU/100ml"],
            "Compliance %": [
                f"{((report_data['ph'] >= 6.5) & (report_data['ph'] <= 8.5)).mean() * 100:.1f}%",
                f"{(report_data['do_mg_l'] > 5).mean() * 100:.1f}%",
                f"{(report_data['bod_mg_l'] < 10).mean() * 100:.1f}%",
                f"{(report_data['cod_mg_l'] < 40).mean() * 100:.1f}%",
                f"{(report_data['coliform_cfu'] < 2500).mean() * 100:.1f}%"
            ],
            "Status": [
                "🟢" if ((report_data['ph'] >= 6.5) & (report_data['ph'] <= 8.5)).mean() > 0.8 else "🔴",
                "🟢" if (report_data['do_mg_l'] > 5).mean() > 0.8 else "🔴",
                "🟢" if (report_data['bod_mg_l'] < 10).mean() > 0.8 else "🔴",
                "🟢" if (report_data['cod_mg_l'] < 40).mean() > 0.8 else "🔴",
                "🟢" if (report_data['coliform_cfu'] < 2500).mean() > 0.8 else "🔴"
            ]
        }
        compliance_df = pd.DataFrame(compliance_data)
        st.dataframe(compliance_df, hide_index=True, use_container_width=True)
        
        # Location-wise analysis
        st.markdown("#### Location-wise Water Quality")
        
        location_wqi = report_data.groupby("node_name").agg({
            "wqi": "mean",
            "ph": "mean",
            "do_mg_l": "mean",
            "bod_mg_l": "mean"
        }).round(2).reset_index()
        location_wqi.columns = ["Location", "Avg WQI", "Avg pH", "Avg DO", "Avg BOD"]
        location_wqi = location_wqi.sort_values("Avg WQI", ascending=False)
        
        st.dataframe(location_wqi, hide_index=True, use_container_width=True)
        
        # Visualization
        fig = px.bar(location_wqi, x="Location", y="Avg WQI", 
                    title="Location-wise Water Quality Index",
                    color="Avg WQI", color_continuous_scale="RdYlGn")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    elif report_type == "Flood Risk Assessment":
        st.markdown("### 🌊 Flood Risk Assessment Report")
        
        month_data = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=30))]
        
        # Risk summary
        st.markdown("#### Risk Summary (Last 30 Days)")
        
        high_level_days = len(month_data[month_data["water_level_m"] > 5.5]["timestamp"].dt.date.unique())
        heavy_rain_days = len(month_data[month_data["rainfall_mm"] > 50]["timestamp"].dt.date.unique())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Days with Elevated Levels", high_level_days)
        col2.metric("Heavy Rainfall Days", heavy_rain_days)
        col3.metric("Flood Alerts Issued", 3)
        
        # Water level trends
        st.markdown("#### Water Level Analysis")
        
        daily_levels = month_data.groupby(month_data["timestamp"].dt.date).agg({
            "water_level_m": ["mean", "max"],
            "rainfall_mm": "sum"
        }).reset_index()
        daily_levels.columns = ["Date", "Avg Level", "Max Level", "Rainfall"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_levels["Date"], y=daily_levels["Max Level"],
                                mode='lines+markers', name='Max Water Level',
                                line=dict(color='blue', width=2)))
        fig.add_trace(go.Bar(x=daily_levels["Date"], y=daily_levels["Rainfall"],
                            name='Rainfall', yaxis='y2', opacity=0.3))
        fig.add_hline(y=6.0, line_dash="dash", line_color="red", 
                     annotation_text="Flood Warning Level")
        
        fig.update_layout(
            title="Water Level & Rainfall Correlation",
            xaxis_title="Date",
            yaxis_title="Water Level (m)",
            yaxis2=dict(title="Rainfall (mm)", overlaying='y', side='right'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk zones
        st.markdown("#### High-Risk Zones")
        
        risk_zones = month_data.groupby("node_name").agg({
            "water_level_m": ["mean", "max"]
        }).reset_index()
        risk_zones.columns = ["Location", "Avg Level", "Max Level"]
        risk_zones["Risk Score"] = (risk_zones["Max Level"] - 4) * 20
        risk_zones["Risk Score"] = risk_zones["Risk Score"].clip(0, 100).round(1)
        risk_zones = risk_zones.sort_values("Risk Score", ascending=False)
        
        st.dataframe(risk_zones, hide_index=True, use_container_width=True)
    
    elif report_type == "Waste Management Report":
        st.markdown("### 🗑️ Waste Management Report")
        
        month_data = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=30))]
        
        # Waste statistics
        st.markdown("#### Waste Collection Summary (Last 30 Days)")
        
        total_waste = month_data["plastic_count"].sum()
        waste_incidents = month_data["floating_waste"].sum()
        sewage_incidents = month_data["sewage_detected"].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Waste Items Detected", int(total_waste))
        col2.metric("Waste Detection Incidents", int(waste_incidents))
        col3.metric("Sewage Contamination Events", int(sewage_incidents))
        
        # Daily waste trend
        st.markdown("#### Daily Waste Detection Trend")
        
        daily_waste = month_data.groupby(month_data["timestamp"].dt.date).agg({
            "plastic_count": "sum"
        }).reset_index()
        daily_waste.columns = ["Date", "Waste Items"]
        
        fig = px.line(daily_waste, x="Date", y="Waste Items",
                     title="Daily Waste Detection", markers=True)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Hotspot analysis
        st.markdown("#### Waste Hotspot Analysis")
        
        hotspots = month_data.groupby("node_name").agg({
            "plastic_count": "sum",
            "floating_waste": "sum"
        }).reset_index()
        hotspots.columns = ["Location", "Total Waste Items", "Detection Incidents"]
        hotspots = hotspots.sort_values("Total Waste Items", ascending=False)
        
        st.dataframe(hotspots, hide_index=True, use_container_width=True)
        
        # Cleanup efficiency
        st.markdown("#### Cleanup Performance Metrics")
        
        performance_data = {
            "Metric": ["Cleanup Events", "Total Waste Collected", "Avg Response Time", 
                      "Volunteer Hours", "Cost per kg"],
            "Value": ["18 events", "847 kg", "3.2 hours", "428 hours", "₹45"]
        }
        performance_df = pd.DataFrame(performance_data)
        st.dataframe(performance_df, hide_index=True, use_container_width=True)
    
    elif report_type == "Biodiversity Report":
        st.markdown("### 🌿 Biodiversity Monitoring Report")
        
        month_data = df[df["timestamp"] >= (df["timestamp"].max() - pd.Timedelta(days=30))]
        
        # Ecosystem health
        st.markdown("#### Ecosystem Health Indicators")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Biodiversity Index", f"{month_data['biodiversity_score'].mean():.2f}")
        col2.metric("Avg Fish Count", f"{month_data['fish_count'].mean():.0f}")
        col3.metric("Aquatic Plant Density", f"{month_data['aquatic_plants'].mean():.0f}")
        
        # Trends
        st.markdown("#### 30-Day Biodiversity Trends")
        
        daily_biodiv = month_data.groupby(month_data["timestamp"].dt.date).agg({
            "biodiversity_score": "mean",
            "fish_count": "mean"
        }).reset_index()
        daily_biodiv.columns = ["Date", "Biodiversity Index", "Fish Count"]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_biodiv["Date"], 
                                y=daily_biodiv["Biodiversity Index"],
                                mode='lines+markers', name='Biodiversity Index',
                                yaxis='y'))
        fig.add_trace(go.Scatter(x=daily_biodiv["Date"], 
                                y=daily_biodiv["Fish Count"],
                                mode='lines+markers', name='Fish Count',
                                yaxis='y2'))
        
        fig.update_layout(
            title="Biodiversity Trends",
            yaxis=dict(title="Biodiversity Index"),
            yaxis2=dict(title="Fish Count", overlaying='y', side='right'),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Species diversity
        st.markdown("#### Species Diversity Summary")
        
        species_summary = pd.DataFrame({
            "Category": ["Fish Species", "Bird Species", "Aquatic Plants", "Invasive Species"],
            "Count": [12, 23, 8, 2],
            "Trend": ["📈 Increasing", "➡️ Stable", "📈 Increasing", "🔴 Monitor"]
        })
        st.dataframe(species_summary, hide_index=True, use_container_width=True)
    
    elif report_type == "Citizen Engagement Report":
        st.markdown("### 👥 Citizen Engagement Report")
        
        # Report statistics
        st.markdown("#### Report Statistics")
        
        total_reports = len(st.session_state['live_citizen_df'])
        pending = len(st.session_state['live_citizen_df'][
            st.session_state['live_citizen_df']["status"] == "Pending"])
        resolved = len(st.session_state['live_citizen_df'][
            st.session_state['live_citizen_df']["status"] == "Resolved"])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reports", total_reports)
        col2.metric("Pending", pending)
        col3.metric("In Progress", total_reports - pending - resolved)
        col4.metric("Resolved", resolved)
        
        # Report types distribution
        st.markdown("#### Report Types Distribution")
        
        type_dist = st.session_state['live_citizen_df']["report_type"].value_counts().reset_index()
        type_dist.columns = ["Report Type", "Count"]
        
        fig = px.pie(type_dist, values="Count", names="Report Type",
                    title="Citizen Reports by Type")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
        
        # Response performance
        st.markdown("#### Response Performance")
        
        response_metrics = pd.DataFrame({
            "Metric": ["Avg Response Time", "Resolution Rate", "Citizen Satisfaction", 
                      "Repeat Reports"],
            "Value": ["6.4 hours", "76%", "4.2/5", "12%"]
        })
        st.dataframe(response_metrics, hide_index=True, use_container_width=True)
        
        # Volunteer engagement
        st.markdown("#### Volunteer Engagement")
        
        volunteer_stats = pd.DataFrame({
            "Metric": ["Active Volunteers", "Cleanup Events", "Total Hours", 
                      "Waste Collected"],
            "Value": [342, 28, "2,156 hours", "1,847 kg"]
        })
        st.dataframe(volunteer_stats, hide_index=True, use_container_width=True)
    
    # Export options
    st.markdown("---")
    st.markdown("### 📥 Export Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as PDF", use_container_width=True):
            st.success("PDF report generated! Download link sent to admin email.")
    
    with col2:
        if st.button("📊 Export as Excel", use_container_width=True):
            # Generate CSV for download
            export_data = df.tail(1000)
            csv = export_data.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"sabarmati_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📧 Email Report", use_container_width=True):
            st.success("Report emailed to stakeholders!")

# ---------- Footer ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #5a6c7d; padding: 20px;'>
    <b>Sabarmati Riverfront Smart Management System</b><br>
    Powered by IoT, AI, and Citizen Engagement | Scalable Solution for Urban Water Bodies<br>
    <small>Data updates every hour | AI models retrained weekly | 24/7 Monitoring Active</small>
</div>
""", unsafe_allow_html=True)