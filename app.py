# app.py (optimized with MySQL database connection)
import streamlit as st
import extra_streamlit_components as stx
from streamlit_modal import Modal
import time
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import re
import io
import numpy as np
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import plotly.graph_objects as go
from auth import authenticate_user, check_admin_credentials
from db import create_user, get_user
import hashlib

# --------------------------- CONSTANTS ---------------------------
DATA_SOURCES = {
    "River": "river_data",
    "Dam": "dam_data",
    "EPAN": "epan_data",
    "AWS": "aws_data",
    "ARS": "ars_data",
    "Gate": "gate_data"
}

MAHARASHTRA_LOCATIONS = {
    # Major cities
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Nashik": (20.0059, 73.7910),
    "Aurangabad": (19.8762, 75.3433),
    "Solapur": (17.6599, 75.9064),
    "Amravati": (20.9374, 77.7796),
    "Kolhapur": (16.7050, 74.2433),
    # River basins
    "Godavari Basin": (19.9249, 74.3785),
    "Krishna Basin": (17.0000, 74.0000),
    "Tapi Basin": (21.0000, 75.0000),
    # Major dams
    "Koyna Dam": (17.4000, 73.7500),
    "Jayakwadi Dam": (19.4950, 75.3767),
    "Ujani Dam": (18.0833, 75.1167),
    "Bhandardara Dam": (19.5400, 73.7500),
    # Other important locations
    "Konkan Region": (17.0000, 73.0000),
    "Marathwada": (18.5000, 76.5000),
    "Vidarbha": (21.0000, 78.0000),
}

# --------------------------- PAGE CONFIG ---------------------------
st.set_page_config(
    page_title="HydroAnalytics Pro",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- DATABASE FUNCTIONS ---------------------------
def create_db_connection():
    try:
        password = "mariobot@123"
        encoded_password = quote_plus(password)
        connection_string = f"mysql+mysqlconnector://mariobot:{encoded_password}@103.224.243.31:3307/may_2025_data"
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        st.error(f"Error connecting to MySQL database: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_data(table_name, start_date=None, end_date=None, date_column=None):
    """Fetch data from MySQL database using SQLAlchemy with caching"""
    engine = create_db_connection()
    if engine is None:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {table_name}"
        if start_date and end_date and date_column:
            query += f" WHERE {date_column} BETWEEN '{start_date}' AND '{end_date}'"
        
        df = pd.read_sql(query, engine)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Error as e:
        st.error(f"Error fetching data from {table_name}: {e}")
        return pd.DataFrame()
    finally:
        if engine:
            engine.dispose()

@st.cache_data(ttl=3600)
def load_station_data(station_type):
    """Load and cache data for a specific station type"""
    try:
        table_name = DATA_SOURCES[station_type]
        df = fetch_data(table_name)
        return df
    except Exception as e:
        st.error(f"Error loading {station_type} data: {e}")
        return pd.DataFrame()

# --------------------------- AUTHENTICATION ---------------------------
def login_page():
    """Render the login page and handle authentication"""
    with st.container():
        st.title("Dashboard")
        
        # Initialize session state
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'is_admin' not in st.session_state:
            st.session_state.is_admin = False
        if 'username' not in st.session_state:
            st.session_state.username = None
        
        if st.session_state.authenticated:
            return True
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                user = authenticate_user(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.is_admin = user["is_admin"]
                    st.session_state.username = username
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        
        if not st.session_state.authenticated and st.button("Admin Login"):
            st.session_state.admin_login = True
        
        if st.session_state.get('admin_login', False):
            with st.form("admin_login_form"):
                st.subheader("Admin Authentication")
                admin_username = st.text_input("Admin Username")
                admin_password = st.text_input("Admin Password", type="password")
                admin_submit = st.form_submit_button("Authenticate as Admin")
                
                if admin_submit:
                    if check_admin_credentials(admin_username, admin_password):
                        st.session_state.admin_authenticated = True
                        st.session_state.admin_username = admin_username
                        st.success("Admin authentication successful!")
                    else:
                        st.error("Invalid admin credentials")
            
        if st.session_state.get('admin_authenticated', False):
            st.subheader("User Management")
            
            with st.expander("Create New User", expanded=True):
                with st.form("create_user_form"):
                    new_username = st.text_input("New Username")
                    new_password = st.text_input("New Password", type="password")
                    is_admin = st.checkbox("Is Admin?")
                    create_button = st.form_submit_button("Create User")
                    
                    if create_button:
                        if create_user(new_username, new_password, is_admin):
                            st.success(f"User {new_username} created successfully!")
                        else:
                            st.error("Failed to create user (username may already exist)")
            
            if st.button("Back to Login"):
                st.session_state.admin_login = False
                st.session_state.admin_authenticated = False
                st.rerun()
        
        return st.session_state.authenticated

# --------------------------- UI COMPONENTS ---------------------------
def render_sidebar():
    with st.sidebar:
        logo_path = Path(r"C:\Users\dell6\OneDrive\Desktop\conn\logo.png")
        st.image(str(logo_path), use_container_width=True)
        st.markdown("---")
        
        if st.button("🚪 Logout", key="logout_button", use_container_width=True):
            st.session_state['authenticated'] = False
            st.session_state.pop('username', None)
            st.success("Logged out successfully!")
            time.sleep(1)
            st.rerun()
    
        st.markdown("### 🛠 SYSTEM STATUS")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); 
                        padding: 20px;
                        border-radius: 12px;
                        margin: 16px 0;
                        border: 1px solid #3d484d'>
                <div style='color: #b2bec3; font-size: 0.9em'>LAST UPDATED</div>
                <div style='color: white; font-size: 1.1em; margin: 8px 0'>{current_time}</div>
                <div style='margin: 16px 0'>
                    <div>
                        <div style='color: #b2bec3; font-size: 0.9em'>ACTIVE Locations</div>
                        <div style='color: white; font-size: 1.4em'>692</div>
                    </div>
                </div>
                <div style='height: 6px; background: #3d484d; border-radius: 3px'>
                    <div style='width: 85%; height: 100%; background: #0984e3; border-radius: 3px'></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Load all station data into a dictionary

def render_top_metrics():
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-icon">
                    <span>🌍</span>
                </div>
                <div class="metric-card-value">692</div>
                <div class="metric-card-label">Active Locations</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-icon">
                    <span>📊</span>
                </div>
                <div class="metric-card-value">1,024</div>
                <div class="metric-card-label">Total Data Entries</div>
            </div>
        """, unsafe_allow_html=True)


    with col3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-icon">
                    <span>⚡</span>
                </div>
                <div class="metric-card-value">{datetime.now().strftime('%H:%M')}</div>
                <div class="metric-card-label">Last Updated</div>
            </div>
        """, unsafe_allow_html=True)

# --------------------------- TAB CONTENT ---------------------------
def show_overview_tab():
    st.markdown("""
        <div style='padding: 16px 0 24px 0'>
            <h2 style='color: #2d3436; margin:0; font-size:2.1em'>
                📊 Project Data Distribution Analysis
            </h2>
            <p style='color: #636e72; margin:0; font-size:1.1em'>
                Distribution of monitoring data across projects for each station
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load data for stations with project info
    stations_with_data = []
    for station_name in DATA_SOURCES:
        with st.spinner(f"Loading {station_name} data..."):
            df = load_station_data(station_name)
            if not df.empty and 'project_name' in df.columns:
                stations_with_data.append((station_name, df))

    if stations_with_data:
        cols = st.columns(2)
        for idx, (station_name, df) in enumerate(stations_with_data):
            try:
                project_counts = df['project_name'].value_counts().reset_index()
                project_counts.columns = ['Project', 'Count']
                total_records = project_counts['Count'].sum()
                
                # Create custom labels with both count and percentage
                labels = []
                for _, row in project_counts.iterrows():
                    percent = row['Count'] / total_records * 100
                    labels.append(f"{row['Project']}<br>{row['Count']}")
                
                fig = px.pie(
                    project_counts,
                    names='Project',
                    values='Count',
                    title=f'{station_name} Station<br>Project Distribution',
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    hole=0.35,
                    height=400
                )
                
                # Add center annotation with total records
                fig.add_annotation(
                    text=f"Total:<br>{total_records}",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )
                
                fig.update_traces(
                    text=labels,
                    textposition='inside',
                    hovertemplate="<b>%{label}</b><br>Records: %{value}",
                    pull=[0.05 if i == project_counts['Count'].idxmax() else 0 for i in range(len(project_counts))],
                    marker=dict(line=dict(color='#ffffff', width=2))
                )
                
                fig.update_layout(
                    margin=dict(t=60, b=20, l=20, r=20),
                    title_x=0.1,
                    title_font_size=16,
                    showlegend=False,
                    uniformtext_minsize=10,
                    uniformtext_mode='hide'
                )

                with cols[idx % 2]:
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                continue
    else:
        st.warning("No project data available for any station")

    with st.expander("📊 Chart Interpretation Guide"):
        st.markdown("""
            How to read these charts:
            - Each pie chart shows how data is distributed across projects for a specific station
            - The largest segment is slightly pulled out for emphasis
            - The center shows total records for that station
            - Each segment shows project name, record count, and percentage
            - Hover over segments for additional details
        """)

    st.markdown("---")
    st.markdown("## 🗺 Maharashtra Water Monitoring Network")
    st.markdown("### Station Locations Across Maharashtra")

    location_data = []
    for station_type in DATA_SOURCES:
        with st.spinner(f"Processing {station_type} locations..."):
            df = load_station_data(station_type)
            if not df.empty:
                loc_col = 'location_name' if 'location_name' in df.columns else 'location_id' if 'location_id' in df.columns else None
                if loc_col:
                    unique_locations = df[loc_col].dropna().unique()
                    for loc in unique_locations:
                        if isinstance(loc, (int, float)):
                            continue
                            
                        project_name = "Unknown"
                        if 'project_name' in df.columns:
                            project_counts = df[df[loc_col] == loc]['project_name'].value_counts()
                            if not project_counts.empty:
                                project_name = project_counts.idxmax()
                        
                        best_match = None
                        max_similarity = 0
                        for mah_location in MAHARASHTRA_LOCATIONS:
                            similarity = sum(1 for word in loc.split() if word in mah_location)
                            if similarity > max_similarity:
                                best_match = mah_location
                                max_similarity = similarity
                        
                        if best_match:
                            location_data.append({
                                "Station Type": station_type,
                                "Location": loc,
                                "Maharashtra Location": best_match,
                                "Project": project_name,
                                "Latitude": MAHARASHTRA_LOCATIONS[best_match][0],
                                "Longitude": MAHARASHTRA_LOCATIONS[best_match][1]
                            })

    if location_data:
        map_df = pd.DataFrame(location_data)
        color_discrete_map = {
            "River": "#1f77b4",
            "Dam": "#ff7f0e",
            "AWS": "#2ca02c",
            "EPAN": "#d62728",
            "ARS": "#9467bd",
            "Gate": "#8c564b"
        }
        
        fig = px.scatter_mapbox(
            map_df,
            lat="Latitude",
            lon="Longitude",
            color="Station Type",
            hover_name="Location",
            hover_data=["Project", "Station Type", "Maharashtra Location"],
            zoom=5,
            height=600,
            color_discrete_map=color_discrete_map,
            title="Water Monitoring Stations Across Maharashtra",
            size_max=15
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=5.5,
            mapbox_center={"lat": 19.7515, "lon": 75.7139},
            margin={"r":0,"t":40,"l":0,"b":0},
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        fig.add_trace(go.Scattermapbox(
            lat=[16.0000, 16.0000, 22.0000, 22.0000, 16.0000],
            lon=[72.0000, 80.0000, 80.0000, 72.0000, 72.0000],
            mode='lines',
            line=dict(width=2, color='blue'),
            name='Maharashtra',
            hoverinfo='none'
        ))
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No location data available. Showing Maharashtra reference map.")
        maharashtra_coords = (19.7515, 75.7139)
        fig = px.scatter_mapbox(
            lat=[maharashtra_coords[0]],
            lon=[maharashtra_coords[1]],
            zoom=5,
            height=500
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0},
            mapbox_center={"lat": 19.7515, "lon": 75.7139},
            annotations=[
                dict(
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    text="Water Monitoring Stations in Maharashtra",
                    showarrow=False,
                    font=dict(size=20))
            ]
        )
        st.plotly_chart(fig, use_container_width=True)


def show_categories_tab():
    selected_date = st.date_input(
        "Select Date", 
        value=datetime.now().date(),
        key="station_date_selector"
    )
    selected_date_str = selected_date.strftime("%d/%m/%Y")
    
    button_container = st.container()
    station_tabs = st.tabs(list(DATA_SOURCES.keys()))
    
    all_station_alerts = {}
    
    for idx, (station_name, table_name) in enumerate(DATA_SOURCES.items()):
        with station_tabs[idx]:
            with st.spinner(f"Loading {station_name} data..."):
                df = load_station_data(station_name)
                
                if not df.empty:
                    st.subheader(f"{station_name} Station")
                    
                    if 'project_name' in df.columns:
                        projects = df['project_name'].unique()
                        selected_project = st.selectbox(
                            "Select Project",
                            options=["All Projects"] + list(projects),
                            key=f"proj_{station_name}_{idx}"
                        )
                    
                    if 'data_time' in df.columns and 'last_updated' in df.columns:
                        df['last_updated_dt'] = pd.to_datetime(df['last_updated'], format='%d/%m/%Y %H:%M', errors='coerce')
                        df = df[df['last_updated_dt'].notna()]
                        
                        df['last_updated_date'] = df['last_updated_dt'].dt.strftime('%d/%m/%Y')
                        time_groups = df.groupby('data_time')['last_updated_date'].agg(
                            lambda x: x.mode()[0] if not x.mode().empty else None
                        ).reset_index()
                        time_groups.rename(columns={'last_updated_date': 'majority_date'}, inplace=True)
                        df = df.merge(time_groups, on='data_time')
                        
                        daily_df = df[df['majority_date'] == selected_date_str]
                        if selected_project != "All Projects":
                            daily_df = daily_df[daily_df['project_name'] == selected_project]
                        
                        st.info(f"Showing {len(daily_df)} rows having last_updated date as {selected_date_str}")
                        
                        if not daily_df.empty:
                            # Data Display
                            st.markdown("### 📋 Current Readings")
                            
                            # Initialize alerts for this station and date
                            alert_rows = []
                            constant_value_rows = []  # For EPAN constant value detection
                            
                            # Create a copy of the dataframe for display (excluding specific columns)
                            columns_to_exclude = ['data_date', 'data_time', 'last_updated_date', 'majority_date', 'last_updated_dt']
                            display_df = daily_df.drop(columns=[col for col in columns_to_exclude if col in daily_df.columns])
                            
                            # Reset index to ensure proper alignment
                            display_df = display_df.reset_index(drop=True)
                            daily_df = daily_df.reset_index(drop=True)
                            
                            # Custom Highlighting Function for all station types
                            def highlight_alerts(row):
                                # Get the original row using the current row's position
                                row_position = row.name
                                if row_position >= len(daily_df):
                                    return [''] * len(row)
                                    
                                original_row = daily_df.iloc[row_position]
                                styles = [''] * len(row)
                                alert_detected = False
                                constant_value_detected = False
                                
                                # Common checks for all stations - battery voltage
                                if 'batt_volt' in original_row and pd.notnull(original_row['batt_volt']):
                                    try:
                                        batt_volt = float(original_row['batt_volt'])
                                        if batt_volt < 10.5:
                                            # Highlight the entire row for main display
                                            styles = ['background-color: #ffcccc'] * len(row)
                                            alert_detected = True
                                            original_row['alert_type'] = 'Low Battery (<10.5V)'
                                    except:
                                        pass
                                
                                # Station-specific checks
                                if station_name == 'Gate':
                                    gate_cols = [col for col in daily_df.columns if re.match(r'^g\d+$', col)]
                                    for col in gate_cols:
                                        if col in original_row and pd.notnull(original_row[col]):
                                            try:
                                                if float(original_row[col]) > 0.00:
                                                    styles = ['background-color: #ffcccc'] * len(row)
                                                    alert_detected = True
                                                    break
                                            except:
                                                continue
                                
                                elif station_name == 'EPAN' and 'epan_water_depth' in original_row:
                                    try:
                                        current_depth = float(original_row['epan_water_depth'])
                                        location_id = original_row['location_id'] if 'location_id' in original_row else None
                                        
                                        # First check for constant value (highest priority)
                                        if location_id:
                                            # Get dates to check (previous 3 days + today)
                                            dates_to_check = []
                                            days_back = 0
                                            while len(dates_to_check) < 4:  # We need 4 days total (today + 3 previous)
                                                check_date = selected_date - timedelta(days=days_back)
                                                check_date_str = check_date.strftime('%d/%m/%Y')
                                                
                                                # Filter for this location and date
                                                prev_day_df = df[
                                                    (df['majority_date'] == check_date_str) & 
                                                    (df['location_id'] == location_id)
                                                ]
                                                
                                                if not prev_day_df.empty and 'epan_water_depth' in prev_day_df.columns:
                                                    # Take the most recent reading from that day
                                                    prev_depth = float(prev_day_df['epan_water_depth'].iloc[0])
                                                    dates_to_check.append((check_date_str, prev_depth))
                                                
                                                days_back += 1
                                                if days_back > 10:  # Safety limit
                                                    break
                                            
                                            # If we have 4 days of data, check if all values are equal
                                            if len(dates_to_check) == 4:
                                                all_equal = all(d[1] == current_depth for d in dates_to_check)
                                                if all_equal:
                                                    constant_value_detected = True
                                                    original_row['alert_type'] = 'Constant Water Depth (4 days)'
                                                    original_row['constant_value_days'] = [d[0] for d in dates_to_check]
                                                    if 'epan_water_depth' in row.index:
                                                        depth_index = row.index.get_loc('epan_water_depth')
                                                        styles[depth_index] = 'background-color: #add8e6; font-weight: bold'  # Light blue
                                                    
                                        # Only check other constraints if not a constant value
                                        if not constant_value_detected:
                                            # Check for water depth ≤50 or ≥200
                                            if current_depth <= 50 or current_depth >= 200:
                                                styles = ['background-color: #ffcccc'] * len(row)
                                                alert_detected = True
                                                original_row['alert_type'] = f'Water Depth {"≤50" if current_depth <=50 else "≥200"}'
                                            
                                            # Previous day difference check (go back up to 10 days if needed)
                                            if location_id:
                                                prev_depth = None
                                                days_back = 1
                                                comparison_date = None
                                                
                                                # Check up to 10 previous days for data
                                                while days_back <= 10 and prev_depth is None:
                                                    check_date = selected_date - timedelta(days=days_back)
                                                    check_date_str = check_date.strftime('%d/%m/%Y')
                                                    
                                                    # Filter for this location and date
                                                    prev_day_df = df[
                                                        (df['majority_date'] == check_date_str) & 
                                                        (df['location_id'] == location_id)
                                                    ]
                                                    
                                                    if not prev_day_df.empty and 'epan_water_depth' in prev_day_df.columns:
                                                        # Take the most recent reading from that day
                                                        prev_depth = float(prev_day_df['epan_water_depth'].iloc[0])
                                                        comparison_date = check_date_str
                                                    
                                                    days_back += 1
                                                
                                                # If we found previous data, check the difference
                                                if prev_depth is not None:
                                                    if abs(current_depth - prev_depth) > 15:
                                                        styles = ['background-color: #ffcccc'] * len(row)
                                                        if 'epan_water_depth' in row.index:
                                                            depth_index = row.index.get_loc('epan_water_depth')
                                                            styles[depth_index] = 'background-color: #ff9999; font-weight: bold'
                                                        alert_detected = True
                                                        original_row['alert_type'] = f'Depth Change >15 (vs {comparison_date})'
                                                        original_row['previous_depth'] = prev_depth
                                                        original_row['depth_difference'] = abs(current_depth - prev_depth)
                                    except Exception as e:
                                        st.error(f"Error processing EPAN data: {str(e)}")
                                
                                elif station_name == 'AWS':
                                    # Initialize alert type list if it doesn't exist
                                    if 'alert_type' not in original_row:
                                        original_row['alert_type'] = []
                                    elif isinstance(original_row['alert_type'], str):
                                        original_row['alert_type'] = [original_row['alert_type']]
                                    
                                    # 1. Check for zero values in specified columns
                                    zero_value_columns = ['atmospheric_pressure', 'temperature', 'humidity', 'solar_radiation', 'wind_speed']
                                    for col in zero_value_columns:
                                        if col in original_row and pd.notnull(original_row[col]):
                                            try:
                                                if float(original_row[col]) == 0:
                                                    styles = ['background-color: #ffcccc'] * len(row)
                                                    alert_detected = True
                                                    original_row['alert_type'].append(f'{col.capitalize().replace("_", " ")} is 0')
                                                    # Highlight the specific zero value column
                                                    if col in row.index:
                                                        col_index = row.index.get_loc(col)
                                                        styles[col_index] = 'background-color: #ff9999; font-weight: bold'
                                            except:
                                                pass
                                    
                                    # 2. Check for rain values > 100 (updated constraint)
                                    rain_columns = ['hourly_rain', 'daily_rain']
                                    rain_alert = False
                                    rain_alert_cols = []
                                    for col in rain_columns:
                                        if col in original_row and pd.notnull(original_row[col]):
                                            try:
                                                rain_value = float(original_row[col])
                                                if rain_value > 100:
                                                    styles = ['background-color: #ffcccc'] * len(row)
                                                    alert_detected = True
                                                    rain_alert = True
                                                    rain_alert_cols.append(col)
                                                    # Highlight the specific rain column
                                                    if col in row.index:
                                                        col_index = row.index.get_loc(col)
                                                        styles[col_index] = 'background-color: #ff9999; font-weight: bold'
                                            except:
                                                pass
                                    
                                    if rain_alert:
                                        original_row['alert_type'].append('Rainfall > 100mm')
                                        original_row['rain_alert_columns'] = rain_alert_cols
                                    
                                    # 3. Check for wind speed > 30
                                    if 'wind_speed' in original_row and pd.notnull(original_row['wind_speed']):
                                        try:
                                            wind_speed = float(original_row['wind_speed'])
                                            if wind_speed > 30:
                                                styles = ['background-color: #ffcccc'] * len(row)
                                                alert_detected = True
                                                original_row['alert_type'].append('High Wind Speed (>30)')
                                                # Highlight the wind speed column
                                                if 'wind_speed' in row.index:
                                                    ws_index = row.index.get_loc('wind_speed')
                                                    styles[ws_index] = 'background-color: #ff9999; font-weight: bold'
                                        except:
                                            pass
                                    
                                    # 4. Existing AWS checks
                                    if 'rainfall' in original_row and pd.notnull(original_row['rainfall']):
                                        try:
                                            if float(original_row['rainfall']) > 50:
                                                styles = ['background-color: #ffcccc'] * len(row)
                                                alert_detected = True
                                                original_row['alert_type'].append('High Rainfall (>50mm)')
                                        except:
                                            pass
                                    
                                    if 'temperature' in original_row and pd.notnull(original_row['temperature']):
                                        try:
                                            if float(original_row['temperature']) > 40:
                                                styles = ['background-color: #ffcccc'] * len(row)
                                                alert_detected = True
                                                original_row['alert_type'].append('High Temperature (>40)')
                                        except:
                                            pass
                                    
                                    # Convert alert_type back to string if it was modified
                                    if isinstance(original_row['alert_type'], list):
                                        original_row['alert_type'] = ', '.join(original_row['alert_type'])
                                
                                # River/Dam station level difference check with 10-day lookback
                                elif (station_name in ['River', 'Dam'] and 
                                    'level_mtr' in original_row and 
                                    'location_id' in original_row):
                                    try:
                                        current_level = float(original_row['level_mtr'])
                                        location_id = original_row['location_id']
                                        
                                        # Initialize variables
                                        prev_level = None
                                        days_checked = 0
                                        comparison_date = None
                                        
                                        # Check up to 10 previous days for data
                                        while days_checked < 10 and prev_level is None:
                                            check_date = selected_date - timedelta(days=days_checked + 1)
                                            check_date_str = check_date.strftime('%d/%m/%Y')
                                            
                                            # Filter for this location and date
                                            prev_day_df = df[
                                                (df['majority_date'] == check_date_str) & 
                                                (df['location_id'] == location_id)
                                            ]
                                            
                                            if not prev_day_df.empty and 'level_mtr' in prev_day_df.columns:
                                                # Take the most recent reading from that day
                                                prev_level = float(prev_day_df['level_mtr'].iloc[0])
                                                comparison_date = check_date_str
                                                break
                                                
                                            days_checked += 1
                                        
                                        # If we found previous data, check the difference
                                        if prev_level is not None:
                                            level_diff = abs(current_level - prev_level)
                                            if level_diff > 1:
                                                styles = ['background-color: #ffcccc'] * len(row)
                                                if 'level_mtr' in row.index:
                                                    level_mtr_index = row.index.get_loc('level_mtr')
                                                    styles[level_mtr_index] = 'background-color: #ff9999; font-weight: bold'
                                                alert_detected = True
                                                original_row['alert_type'] = f'Level Change >1m (vs {comparison_date})'
                                                original_row['previous_level'] = prev_level
                                                original_row['level_difference'] = level_diff
                                    except:
                                        pass
                                
                                if alert_detected or constant_value_detected:
                                    alert_rows.append(original_row)
                                
                                return styles
                            
                            # Apply highlighting
                            styled_df = display_df.style.apply(highlight_alerts, axis=1)
                            st.dataframe(
                                styled_df,
                                use_container_width=True,
                                height=min(400, len(display_df) * 35 + 50))
                            
                            # Add download button for current readings
                            csv = display_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Current Readings",
                                data=csv,
                                file_name=f"{station_name}current_readings{selected_date_str.replace('/', '-')}.csv",
                                mime='text/csv',
                                key=f"download_current_{station_name}_{idx}",
                                type="primary",
                                help=f"Download current readings for {station_name} on {selected_date_str}"
                            )
                            
                            # Show alerts for this specific date and station
                            if alert_rows:
                                st.markdown("---")
                                st.subheader(f"⚠ Alerts for {selected_date_str}")
                                
                                # Create columns for alert count and details
                                col1, col2 = st.columns([1, 3])
                                
                                with col1:
                                    st.metric(
                                        label="Total Alerts",
                                        value=len(alert_rows),
                                        help=f"Number of alert records found for {station_name} on {selected_date_str}"
                                    )
                                
                                with col2:
                                    # For River/Dam stations, add level difference explanation
                                    if station_name in ['River', 'Dam'] and any('level_mtr' in row for row in alert_rows):
                                        st.info("""
                                            ℹ Level alerts are triggered when current level_mtr differs by 
                                            more than ±1 meter from the same location's value on any of the 
                                            previous 10 days (checks each day until data is found).
                                        """)
                                    # For EPAN stations, add water depth difference explanation
                                    elif station_name == 'EPAN' and any('epan_water_depth' in row for row in alert_rows):
                                        st.info("""
                                            ℹ EPAN alerts are triggered when:
                                            - Low Battery (<10.5V)
                                            - Water depth ≤50 or ≥200
                                            - Depth differs by more than ±15 from previous available day
                                            - Constant water depth for 4 consecutive days (highest priority)
                                            System checks up to 10 previous days if needed
                                        """)
                                    # For AWS stations
                                    elif station_name == 'AWS':
                                        st.info("""
                                            ℹ AWS alerts are triggered when:
                                            - Atmospheric pressure, temperature, humidity, solar radiation or wind speed is 0
                                            - Hourly or daily rain > 100mm
                                            - Rainfall > 50mm
                                            - Wind speed > 30
                                            - Temperature > 40
                                        """)
                                    # For battery voltage alerts
                                    elif any('Low Battery' in str(row.get('alert_type', '')) for row in alert_rows):
                                        st.info("""
                                            ℹ Battery alerts are triggered when voltage <10.5V
                                        """)
                                
                                alert_df = pd.DataFrame(alert_rows)
                                
                                # Define the desired column order for alerts
                                base_columns = [
                                    'project_name', 'sr_no', 'location_name', 'location_id',
                                    'last_updated', 'batt_volt', 'level_mtr', 'previous_level',
                                    'level_difference', 'alert_type'
                                ]
                                
                                # Get all columns that exist in the dataframe
                                existing_columns = [col for col in base_columns if col in alert_df.columns]
                                
                                # Get remaining columns not in our base list
                                other_columns = [col for col in alert_df.columns if col not in base_columns and col not in columns_to_exclude]
                                
                                # Create the final column order
                                final_columns = existing_columns + other_columns
                                
                                # Reorder the alert dataframe
                                alert_display_df = alert_df[final_columns]
                                
                                # Remove the last_updated_dt column if it exists
                                if 'last_updated_dt' in alert_display_df.columns:
                                    alert_display_df = alert_display_df.drop(columns=['last_updated_dt'])
                                
                                # Store alerts data for this station
                                all_station_alerts[station_name] = alert_display_df
                                
                                # Custom highlighting for alert dataframe
                                def highlight_alert_rows(row):
                                    styles = ['background-color: #ffebee'] * len(row)
                                    try:
                                        # Safely check alert_type
                                        alert_type = str(row.get('alert_type', ''))
                                        
                                        # Highlight battery voltage in light green for low battery alerts
                                        if 'Low Battery' in alert_type and 'batt_volt' in row.index:
                                            batt_index = row.index.get_loc('batt_volt')
                                            styles[batt_index] = 'background-color: #90ee90; font-weight: bold'
                                        
                                        # Highlight EPAN water depth
                                        if station_name == 'EPAN' and 'epan_water_depth' in row:
                                            depth_index = row.index.get_loc('epan_water_depth')
                                            if 'Constant Water Depth' in alert_type:
                                                styles[depth_index] = 'background-color: #add8e6; font-weight: bold'
                                            elif 'Depth Change' in alert_type:
                                                styles[depth_index] = 'background-color: #ff9999; font-weight: bold'
                                            elif 'Water Depth' in alert_type:
                                                styles[depth_index] = 'background-color: #ff9999; font-weight: bold'
                                        
                                        # River/Dam level changes
                                        if (station_name in ['River', 'Dam'] and 
                                            'level_mtr' in row and 
                                            'Level Change' in alert_type):
                                            level_mtr_index = row.index.get_loc('level_mtr')
                                            styles[level_mtr_index] = 'background-color: #ff9999; font-weight: bold'
                                        
                                        # Highlight AWS rain columns in pink when >100mm
                                        if station_name == 'AWS' and 'Rainfall > 100mm' in alert_type:
                                            rain_cols = ['hourly_rain', 'daily_rain']
                                            for rain_col in rain_cols:
                                                if rain_col in row.index and pd.notnull(row[rain_col]):
                                                    try:
                                                        if float(row[rain_col]) > 100:
                                                            col_index = row.index.get_loc(rain_col)
                                                            styles[col_index] = 'background-color: #ffc0cb; font-weight: bold'
                                                    except:
                                                        pass
                                        
                                        # Highlight zero value columns in AWS
                                        if station_name == 'AWS':
                                            zero_cols = ['atmospheric_pressure', 'temperature', 'humidity', 
                                                        'solar_radiation', 'wind_speed']
                                            for col in zero_cols:
                                                if col in row.index and str(row[col]) == '0':
                                                    col_index = row.index.get_loc(col)
                                                    styles[col_index] = 'background-color: #ff9999; font-weight: bold'
                                            
                                    except:
                                        pass
                                    return styles
                                
                                st.dataframe(
                                    alert_display_df.style.apply(highlight_alert_rows, axis=1),
                                    use_container_width=True,
                                    height=min(400, len(alert_rows) * 35 + 50))
                                
                                # Add download button for alerts
                                alert_csv = alert_display_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Alerts Data",
                                    data=alert_csv,
                                    file_name=f"{station_name}alerts{selected_date_str.replace('/', '-')}.csv",
                                    mime='text/csv',
                                    key=f"download_alerts_{station_name}_{idx}",
                                    type="primary",
                                    help=f"Download alert data for {station_name} on {selected_date_str}"
                                )
                            else:
                                st.success(f"✅ No alerts detected for {selected_date_str}")
                        
                        else:
                            st.warning(f"No data available where majority last_updated date is {selected_date_str}")
                    else:
                        st.warning("Required columns (data_time or last_updated) not found in data")
                else:
                    st.warning(f"No data available for {station_name} station")
        
    # Add "Download All Alerts" button at the top right
    if all_station_alerts:
        with button_container:
            # Create a function to generate the combined CSV
            def generate_combined_alerts_csv():
                output = io.StringIO()
                
                for station_name, alert_df in all_station_alerts.items():
                    # Write station name as header
                    output.write(f"{station_name} Station Alerts\n")
                    
                    # Write the dataframe
                    alert_df.to_csv(output, index=False)
                    
                    # Add two empty rows between stations
                    output.write("\n\n")
                
                return output.getvalue().encode('utf-8')
            
            # Place the button at the top right
            st.download_button(
                label="📥 Download All Alerts",
                data=generate_combined_alerts_csv(),
                file_name=f"all_stations_alerts_{selected_date_str.replace('/', '-')}.csv",
                mime='text/csv',
                key="download_all_alerts",
                type="primary",
                help="Download alerts data for all stations in a single CSV file"
            )

def show_history_tab():
    st.subheader("Historical Data Explorer")
    
    # Main filter section in columns
    filter_col1, filter_col2 = st.columns([1, 1])
    
    with filter_col1:
        # Station selection
        station = st.selectbox(
            "Select Station", 
            list(DATA_SOURCES.keys()), 
            key="hist_station"
        )
        
        # Project selection
        project_options = ["All Projects"]
        station_df = load_station_data(station)  # Changed from all_data to load_station_data
        
        if not station_df.empty and 'project_name' in station_df.columns:
            project_options += station_df['project_name'].astype(str).unique().tolist()
        
        project = st.selectbox(
            "Select Project",
            project_options,
            key="hist_project"
        )
    
    with filter_col2:
        # Date range selection
        date_range_option = st.radio(
            "Date Range",
            options=["Custom Date Range", "Last 7 Days", "Last 15 Days", "Last 30 Days"],
            key="hist_date_range",
            horizontal=True
        )
        
        if date_range_option == "Custom Date Range":
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                hist_start = st.date_input(
                    "Start Date", 
                    value=datetime.now() - timedelta(days=7), 
                    key="hist_start"
                )
            with date_col2:
                hist_end = st.date_input(
                    "End Date", 
                    value=datetime.now(), 
                    key="hist_end"
                )
        else:
            days = 7 if date_range_option == "Last 7 Days" else (15 if date_range_option == "Last 15 Days" else 30)
            hist_end = datetime.now().date()
            hist_start = hist_end - timedelta(days=days-1)
            st.info(f"Showing data for: {hist_start.strftime('%d/%m/%Y')} to {hist_end.strftime('%d/%m/%Y')}")

    # Location selection below the first row of filters
    filtered_df = load_station_data(station)  # Changed from all_data to load_station_data
    if not filtered_df.empty:
        if project != "All Projects" and 'project_name' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['project_name'] == project]
        
        # Create combined location options with "All" as first option
        location_options = [("All Locations", "All")]  # Add "All" option first
        
        if 'location_id' in filtered_df.columns and 'location_name' in filtered_df.columns:
            unique_locations = filtered_df[['location_id', 'location_name']].drop_duplicates()
            unique_locations['display'] = unique_locations['location_id'] + " (" + unique_locations['location_name'] + ")"
            unique_locations = unique_locations.sort_values('display')
            
            # Add actual locations after the "All" option
            location_options.extend(zip(unique_locations['display'], unique_locations['location_id']))
            
            location_display_map = dict(location_options)
        else:
            location_display_map = {"All Locations": "All"}
    else:
        location_display_map = {"All Locations": "All"}

    selected_location_display = st.selectbox(
        "Select Location (ID - Name)",
        options=list(location_display_map.keys()),
        index=0,
        key="hist_location_select"
    )

    selected_location_id = location_display_map[selected_location_display]
    
    # Parameter selection and action button
    st.subheader("Data Parameters", divider="gray")
    param_col1, param_col2, button_col = st.columns([2, 2, 1])
    
    with param_col1:
        # Customized parameter selection based on categories tab
        available_params = {
            'River': ['level_mtr', 'batt_volt'],
            'Dam': ['level_mtr', 'batt_volt'],
            'EPAN': ['epan_water_depth', 'batt_volt'],
            'AWS': [
                'wind_speed', 'wind_direction', 'atmospheric_pressure',
                'temperature', 'humidity', 'solar_radiation',
                'hourly_rain', 'daily_rain', 'batt_volt'
            ],
            'ARS': ['hour_rain', 'daily_rain', 'batt_volt'],
            'Gate': ['total_gates', 'batt_volt', 'All Gates']
        }
        
        # Add "All Parameters" option
        param_options = ["All Parameters"] + available_params.get(station, [])
    
    with param_col2:
        selected_params = st.multiselect(
            "Select parameters to display",
            options=param_options,
            default=param_options[0] if len(param_options) > 0 else None,
            help="Choose which parameters to include in the historical view",
            label_visibility="hidden"
        )
    
    with button_col:
        st.write("")  # Spacer for vertical alignment
        load_button = st.button("Load Data", 
                            key="hist_load",
                            use_container_width=True,
                            type="primary")

    # Data loading and filtering
    if load_button:
        with st.spinner("Loading data..."):
            try:
                # Fetch ALL data without date filtering initially
                hist_df = fetch_data(
                    table_name=DATA_SOURCES[station],
                    date_column='data_date'
                )
                
                if not hist_df.empty:
                    # Apply project filter if needed
                    if project != "All Projects" and 'project_name' in hist_df.columns:
                        hist_df = hist_df[hist_df['project_name'] == project]
                    
                    # Apply location filter only if a specific location is selected
                    if selected_location_id != "All" and 'location_id' in hist_df.columns:
                        hist_df = hist_df[hist_df['location_id'] == selected_location_id]
                    
                    # Convert last_updated to datetime
                    hist_df['last_updated_dt'] = pd.to_datetime(
                        hist_df['last_updated'], 
                        format='%d/%m/%Y %H:%M', 
                        errors='coerce'
                    )
                    
                    # Drop rows with invalid dates
                    hist_df = hist_df[hist_df['last_updated_dt'].notna()]
                    
                    # Convert filter dates to datetime for comparison
                    hist_start_dt = pd.to_datetime(hist_start)
                    hist_end_dt = pd.to_datetime(hist_end) + pd.Timedelta(days=1)  # Include entire end day
                    
                    # Filter data based on date range
                    if 'last_updated_dt' in hist_df.columns:
                        hist_df = hist_df[
                            (hist_df['last_updated_dt'] >= hist_start_dt) & 
                            (hist_df['last_updated_dt'] <= hist_end_dt)
                        ]
                    
                    # Handle parameter selection
                    if "All Parameters" in selected_params:
                        valid_columns = hist_df.columns.tolist()
                    else:
                        valid_columns = []
                        for param in selected_params:
                            if param == "All Gates" and station == "Gate":
                                # Add all gate columns + required parameters
                                gate_cols = [f'g{i}' for i in range(1,9) if f'g{i}' in hist_df.columns]
                                valid_columns.extend(['total_gates', 'batt_volt'] + gate_cols)
                            else:
                                if param in hist_df.columns:
                                    valid_columns.append(param)
                    
                    # Always include location info (excluding our temporary columns)
                    location_cols = []
                    for col in ['location_id', 'location_name', 'last_updated']:
                        if col in hist_df.columns:
                            location_cols.append(col)
                    
                    # Combine location columns with selected parameters
                    final_columns = location_cols + [col for col in valid_columns if col not in location_cols]
                    
                    # Remove our temporary columns and duplicates
                    final_columns = [col for col in final_columns if col not in ['last_updated_dt']]
                    final_columns = list(dict.fromkeys(final_columns))
                    
                    if not final_columns:
                        st.error("No valid parameters selected that exist in the dataset")
                        st.stop()
                    
                    filtered_df = hist_df[final_columns]

                    # Display filtered data with explanation
                    st.success(f"""
                        *Showing data for:*  
                        - Location: {selected_location_display}  
                        - Date range: {hist_start.strftime('%d/%m/%Y')} to {hist_end.strftime('%d/%m/%Y')}  
                        - Parameters: {', '.join(selected_params) if selected_params else 'All Parameters'}  
                    """)
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", len(filtered_df))
                    with col2:
                        st.metric("Unique Timestamps", filtered_df['last_updated'].nunique())
                    
                    # Use the same ALERT_COLORS from categories tab for consistency
                    ALERT_COLORS = {
                        'battery': '#ff5252',  # Red
                        'epan_range': '#ff9800',  # Orange
                        'epan_change': '#ff5722',  # Deep orange
                        'epan_consistent': '#9c27b0',  # Purple
                        'epan_negative': '#f44336',  # Red
                        'gate_open': '#4caf50',  # Green
                        'rain_heavy': '#2196f3',  # Blue
                        'aws_zero': '#607d8b',  # Blue grey
                        'high_wind': '#673ab7',  # Deep purple
                        'high_temp': '#e91e63',  # Pink
                        'level_change': '#795548',  # Brown
                        'default': '#ffcccc'  # Light red
                    }
                    
                    # Updated detect_alerts function to match categories tab logic
                    def detect_alerts(df):
                        alert_rows = []
                        
                        for _, row in df.iterrows():
                            styles = [''] * len(row)
                            alert_types = []
                            
                            # Common check for all stations - Battery Voltage
                            if 'batt_volt' in row and pd.notnull(row['batt_volt']):
                                try:
                                    if float(row['batt_volt']) < 10.5:
                                        styles = [f'background-color: {ALERT_COLORS["battery"]}'] * len(row)
                                        alert_types.append("Low Battery (<10.5V)")
                                except:
                                    pass
                            
                            # Station-specific checks
                            if station == 'Gate':
                                # Check if any gate is open (>0)
                                gate_cols = [col for col in filtered_df.columns if re.match(r'^g\d+$', col)]
                                for col in gate_cols:
                                    if col in row and pd.notnull(row[col]):
                                        try:
                                            if float(row[col]) > 0.00:
                                                styles = [f'background-color: {ALERT_COLORS["gate_open"]}'] * len(row)
                                                alert_types.append(f"Gate {col} Open")
                                                break
                                        except:
                                            continue
                            
                            elif station == 'EPAN':
                                # EPAN specific checks
                                if 'epan_water_depth' in row:
                                    try:
                                        current_depth = float(row['epan_water_depth'])
                                        location_id = row['location_id'] if 'location_id' in row else None
                                        
                                        # Check for negative values
                                        if current_depth < 0:
                                            styles = [f'background-color: {ALERT_COLORS["epan_negative"]}'] * len(row)
                                            alert_types.append("Negative Depth")
                                        
                                        # Check for range alerts (0-50 or 200+)
                                        elif current_depth <= 50 or current_depth >= 200:
                                            styles = [f'background-color: {ALERT_COLORS["epan_range"]}'] * len(row)
                                            alert_types.append(f"Water Depth {'≤50' if current_depth <=50 else '≥200'}")
                                        
                                        # Check for day-to-day change >15mm
                                        if location_id:
                                            prev_day = pd.to_datetime(row['last_updated']) - timedelta(days=1)
                                            prev_data = df[
                                                (df['location_id'] == location_id) & 
                                                (pd.to_datetime(df['last_updated']) == prev_day)
                                            ]
                                            
                                            if not prev_data.empty and 'epan_water_depth' in prev_data.columns:
                                                prev_depth = float(prev_data['epan_water_depth'].iloc[0])
                                                if abs(current_depth - prev_depth) > 15:
                                                    styles = [f'background-color: {ALERT_COLORS["epan_change"]}'] * len(row)
                                                    alert_types.append("Large Daily Change (>15mm)")
                                        
                                        # Check if last 4 days have same value (consistency)
                                        if 'location_id' in row:
                                            # Get last 4 days data for this location
                                            last_4_days = df[
                                                (df['location_id'] == row['location_id']) & 
                                                (df['last_updated_dt'] <= pd.to_datetime(row['last_updated']))
                                            ].sort_values('last_updated_dt', ascending=False).head(4)
                                            
                                            if len(last_4_days) >= 4:
                                                last_4_values = last_4_days['epan_water_depth'].astype(float)
                                                if len(set(last_4_values.round(1))) == 1:  # All values same
                                                    styles = [f'background-color: {ALERT_COLORS["epan_consistent"]}'] * len(row)
                                                    alert_types.append("Same Value for 4 Days")
                                    except:
                                        pass
                            
                            elif station == 'AWS':
                                # AWS specific checks
                                zero_value_alerts = []
                                
                                # Check for zero values in critical parameters
                                zero_params = [
                                    'wind_speed', 'wind_direction', 'atmospheric_pressure',
                                    'temperature', 'humidity', 'solar_radiation'
                                ]
                                
                                for param in zero_params:
                                    if param in row and pd.notnull(row[param]):
                                        try:
                                            if float(row[param]) == 0:
                                                zero_value_alerts.append(param)
                                        except:
                                            pass
                                
                                if zero_value_alerts:
                                    styles = [f'background-color: {ALERT_COLORS["aws_zero"]}'] * len(row)
                                    alert_types.append(f"Zero Values: {', '.join(zero_value_alerts)}")
                                
                                # Check for heavy rain
                                rain_alerts = []
                                if 'hourly_rain' in row and pd.notnull(row['hourly_rain']):
                                    try:
                                        if float(row['hourly_rain']) > 100:
                                            rain_alerts.append("Hourly Rain >100mm")
                                    except:
                                        pass
                                
                                if 'daily_rain' in row and pd.notnull(row['daily_rain']):
                                    try:
                                        if float(row['daily_rain']) > 100:
                                            rain_alerts.append("Daily Rain >100mm")
                                    except:
                                        pass
                                
                                if rain_alerts:
                                    styles = [f'background-color: {ALERT_COLORS["rain_heavy"]}'] * len(row)
                                    alert_types.append(", ".join(rain_alerts))
                                
                                # Check for high wind speed (>30)
                                if 'wind_speed' in row and pd.notnull(row['wind_speed']):
                                    try:
                                        if float(row['wind_speed']) > 30:
                                            styles = [f'background-color: {ALERT_COLORS["high_wind"]}'] * len(row)
                                            alert_types.append("High Wind Speed (>30)")
                                    except:
                                        pass
                                
                                # Check for high temperature (>40)
                                if 'temperature' in row and pd.notnull(row['temperature']):
                                    try:
                                        if float(row['temperature']) > 40:
                                            styles = [f'background-color: {ALERT_COLORS["high_temp"]}'] * len(row)
                                            alert_types.append("High Temperature (>40)")
                                    except:
                                        pass
                            
                            elif station == 'ARS':
                                # ARS specific checks - heavy rain
                                rain_alerts = []
                                if 'hour_rain' in row and pd.notnull(row['hour_rain']):
                                    try:
                                        if float(row['hour_rain']) > 100:
                                            rain_alerts.append("Hourly Rain >100mm")
                                    except:
                                        pass
                                
                                if 'daily_rain' in row and pd.notnull(row['daily_rain']):
                                    try:
                                        if float(row['daily_rain']) > 100:
                                            rain_alerts.append("Daily Rain >100mm")
                                    except:
                                        pass
                                
                                if rain_alerts:
                                    styles = [f'background-color: {ALERT_COLORS["rain_heavy"]}'] * len(row)
                                    alert_types.append(", ".join(rain_alerts))
                            
                            elif station in ['River', 'Dam']:
                                # River/Dam level checks
                                if 'level_mtr' in row and 'location_id' in row:
                                    try:
                                        current_level = float(row['level_mtr'])
                                        location_id = row['location_id']
                                        prev_day = pd.to_datetime(row['last_updated']) - timedelta(days=1)
                                        prev_data = df[
                                            (df['location_id'] == location_id) & 
                                            (pd.to_datetime(df['last_updated']) == prev_day)
                                        ]
                                        
                                        if not prev_data.empty and 'level_mtr' in prev_data.columns:
                                            prev_level = float(prev_data['level_mtr'].iloc[0])
                                            if abs(current_level - prev_level) > 1:
                                                styles = [f'background-color: {ALERT_COLORS["level_change"]}'] * len(row)
                                                alert_types.append("Level Change >1m")
                                    except:
                                        pass
                            
                            if alert_types:
                                alert_row = row.to_dict()
                                alert_row['alert_types'] = ", ".join(alert_types)
                                alert_rows.append(alert_row)
                        
                        return pd.DataFrame(alert_rows)
                    
                    # Detect alerts in the filtered data
                    alert_df = detect_alerts(filtered_df)
                    
                    # Highlight function matching categories tab
                    def highlight_alerts(row):
                        if not alert_df.empty and row['last_updated'] in alert_df['last_updated'].values:
                            matching_alert = alert_df[alert_df['last_updated'] == row['last_updated']].iloc[0]
                            alert_types = matching_alert['alert_types'].split(', ')
                            
                            # Default to light red background
                            styles = ['background-color: #ffebee'] * len(row)
                            
                            # Apply specific colors based on alert type
                            if any('Low Battery' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["battery"]}'] * len(row)
                            elif any('Gate Open' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["gate_open"]}'] * len(row)
                            elif any('Negative Depth' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["epan_negative"]}'] * len(row)
                            elif any('Water Depth' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["epan_range"]}'] * len(row)
                            elif any('Large Daily Change' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["epan_change"]}'] * len(row)
                            elif any('Same Value for 4 Days' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["epan_consistent"]}'] * len(row)
                            elif any('Zero Values' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["aws_zero"]}'] * len(row)
                            elif any('Rain >100mm' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["rain_heavy"]}'] * len(row)
                            elif any('High Wind Speed' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["high_wind"]}'] * len(row)
                            elif any('High Temperature' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["high_temp"]}'] * len(row)
                            elif any('Level Change' in atype for atype in alert_types):
                                styles = [f'background-color: {ALERT_COLORS["level_change"]}'] * len(row)
                            
                            return styles
                        return [''] * len(row)
                    
                    # Display data with alerts highlighted
                    st.dataframe(
                        filtered_df.style.apply(highlight_alerts, axis=1),
                        use_container_width=True,
                        height=600,
                        column_config={
                            "location_id": "Location ID",
                            "location_name": "Location Name",
                            "batt_volt": "Battery Voltage",
                            "level_mtr": "Level (meters)",
                            "epan_water_depth": "Water Depth (mm)",
                            "total_gates": "Total Gates",
                            "last_updated": "Last Updated"
                        }
                    )
                    
                    # Show alerts section if there are any
                    if not alert_df.empty:
                        st.subheader("⚠ Detected Alerts", divider="red")
                        
                        # Alert summary statistics
                        alert_counts = pd.DataFrame({
                            'Alert Type': [atype for alert in alert_df['alert_types'] for atype in alert.split(', ')],
                        }).value_counts().reset_index(name='Count')
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Alert Instances", len(alert_df))
                        with col2:
                            st.dataframe(
                                alert_counts,
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        # Create expandable sections for each alert type grouped by severity
                        severity_order = {
                            'Low Battery': 3,
                            'Negative Depth': 3,
                            '>100mm': 3,
                            'High Wind Speed': 2,
                            'High Temperature': 2,
                            'Water Depth': 2,
                            'Large Daily Change': 2,
                            'Level Change': 2,
                            'Zero Values': 1,
                            'Gate Open': 1,
                            'Same Value for 4 Days': 1
                        }
                        
                        # Group alerts by severity
                        critical_alerts = []
                        warning_alerts = []
                        info_alerts = []
                        
                        for _, alert_row in alert_df.iterrows():
                            alert_types = alert_row['alert_types'].split(', ')
                            max_severity = 0
                            for atype in alert_types:
                                for key, val in severity_order.items():
                                    if key in atype and val > max_severity:
                                        max_severity = val
                            
                            if max_severity >= 3:
                                critical_alerts.append(alert_row)
                            elif max_severity == 2:
                                warning_alerts.append(alert_row)
                            else:
                                info_alerts.append(alert_row)
                        
                        # Display alerts by severity with appropriate colors
                        if critical_alerts:
                            with st.expander(f"🔴 Critical Alerts ({len(critical_alerts)})", expanded=True):
                                crit_df = pd.DataFrame(critical_alerts)
                                st.dataframe(
                                    crit_df.style.apply(lambda x: [f'background-color: {ALERT_COLORS["battery"]}']*len(x)),
                                    use_container_width=True,
                                    height=min(400, len(critical_alerts) * 35 + 50))
                        
                        if warning_alerts:
                            with st.expander(f"🟠 Warning Alerts ({len(warning_alerts)})", expanded=False):
                                warn_df = pd.DataFrame(warning_alerts)
                                st.dataframe(
                                    warn_df.style.apply(lambda x: [f'background-color: {ALERT_COLORS["epan_range"]}']*len(x)),
                                    use_container_width=True,
                                    height=min(400, len(warning_alerts) * 35 + 50))
                        
                        if info_alerts:
                            with st.expander(f"🔵 Info Alerts ({len(info_alerts)})", expanded=False):
                                info_df = pd.DataFrame(info_alerts)
                                st.dataframe(
                                    info_df.style.apply(lambda x: [f'background-color: {ALERT_COLORS["aws_zero"]}']*len(x)),
                                    use_container_width=True,
                                    height=min(400, len(info_alerts) * 35 + 50))
                        
                        # Download buttons
                        download_col1, download_col2 = st.columns(2)
                        with download_col1:
                            st.download_button(
                                label="Download Full Data as CSV",
                                data=filtered_df.to_csv(index=False).encode('utf-8'),
                                file_name=f"{station}_{selected_location_id}_history_{hist_start.strftime('%Y-%m-%d')}_{hist_end.strftime('%Y-%m-%d')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                type="secondary"
                            )
                        
                        with download_col2:
                            st.download_button(
                                label="Download Alerts Only",
                                data=alert_df.to_csv(index=False).encode('utf-8'),
                                file_name=f"{station}_{selected_location_id}_alerts_{hist_start.strftime('%Y-%m-%d')}_{hist_end.strftime('%Y-%m-%d')}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                type="primary"
                            )
                    else:
                        st.success("✅ No alerts detected in the selected data range")
                        # Download button for full data when no alerts
                        st.download_button(
                            label="Download Data as CSV",
                            data=filtered_df.to_csv(index=False).encode('utf-8'),
                            file_name=f"{station}_{selected_location_id}_history_{hist_start.strftime('%Y-%m-%d')}_{hist_end.strftime('%Y-%m-%d')}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            type="primary"
                        )
                
                else:
                    st.warning("🚫 No historical data found for selected parameters")
            
            except Exception as e:
                st.error(f"❌ Error loading historical data: {str(e)}")
                st.error("Please check your filters and try again")

def show_custom_tab():
    st.subheader("🔍 Advanced Data Explorer")
    st.markdown("---")

    # --------------------------- FILTERS SECTION ---------------------------
    with st.container(border=True):
        st.markdown("### 🔎 Filter Parameters")
        
        # Date Range Selection Options
        date_range_option = st.radio(
            "Select Date Range",
            options=["Custom Date Range", "Last 7 Days", "Last 15 Days"],
            horizontal=True,
            key="date_range_option"
        )
        
        # Date Range - Show different inputs based on selection
        if date_range_option == "Custom Date Range":
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date", 
                    value=datetime.now() - timedelta(days=30),
                    help="Select start date for data retrieval",
                    key="start_date_filter"
                )
            with col2:
                end_date = st.date_input(
                    "End Date", 
                    value=datetime.now(),
                    help="Select end date for data retrieval",
                    key="end_date_filter"
                )
        else:
            # For "Last X Days" options, calculate the date range
            days = 7 if date_range_option == "Last 7 Days" else 15
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days-1)
            
            # Display the calculated date range
            st.info(f"Showing data for {date_range_option}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Station Type Selection
        station_options = ["All Stations"] + list(DATA_SOURCES.keys())
        selected_station = st.selectbox(
            "Station Type",
            options=station_options,
            index=0,
            help="Select station type to filter",
            key="station_type_select"
        )

        # Project Selection
        project_options = ["All Projects"]
        filtered_df = pd.DataFrame()
        
        if selected_station != "All Stations":
            filtered_df = load_station_data(selected_station)
            if 'project_name' in filtered_df.columns:
                project_options += filtered_df['project_name'].astype(str).unique().tolist()
        
        selected_project = st.selectbox(
            "Project Name",
            options=project_options,
            index=0,
            help="Select project to filter",
            key=f"project_select_{selected_station}"
        )
        
        # Location Selection - Combined ID and Name
        location_options = []

        if selected_station == "All Stations":
            # Combine locations from all stations (with project filter)
            for station in DATA_SOURCES.keys():
                data = load_station_data(station)
                if selected_project != "All Projects":
                    data = data[data['project_name'] == selected_project]
                
                # Check if both ID and name columns exist
                if 'location_id' in data.columns and 'location_name' in data.columns:
                    # Create combined display and use ID as value
                    for _, row in data.drop_duplicates(['location_id', 'location_name']).iterrows():
                        display_text = f"{row['location_id']} ({row['location_name']})"
                        location_options.append((display_text, row['location_id']))
        else:
            # Get locations from filtered data
            if not filtered_df.empty and 'location_id' in filtered_df.columns and 'location_name' in filtered_df.columns:
                if selected_project != "All Projects":
                    filtered_df = filtered_df[filtered_df['project_name'] == selected_project]
                
                # Create combined display and use ID as value
                for _, row in filtered_df.drop_duplicates(['location_id', 'location_name']).iterrows():
                    display_text = f"{row['location_id']} ({row['location_name']})"
                    location_options.append((display_text, row['location_id']))
        
        # Remove duplicates and sort by location ID
        location_options = sorted(list(set(location_options)), key=lambda x: x[1])
        
        # Create selectbox with display text but store location_id as value
        selected_location = None
        if location_options:
            selected_location_display = st.selectbox(
                "Select Location",
                options=[opt[0] for opt in location_options],
                index=0,
                help="Select location to analyze (shows as ID with Name)",
                key=f"loc_sel_{selected_station[:3]}_{selected_project[:4]}"
            )
            # Get the actual location_id from the selected display text
            selected_location = next((opt[1] for opt in location_options if opt[0] == selected_location_display), None)
        else:
            st.warning("No locations found for selected filters")

    # --------------------------- DATA FETCHING AND ALERTS ---------------------------
    if st.button("🚀 Execute Search", type="primary") and selected_location:
        results = {}
        total_records = 0
        all_alerts = []
        all_alert_data = []  # To store all alert data for CSV download
        
        with st.status("🔍 Scanning data sources...", expanded=True) as status:
            try:
                stations_to_search = []
                if selected_station == "All Stations":
                    stations_to_search = list(DATA_SOURCES.items())
                else:
                    stations_to_search = [(selected_station, DATA_SOURCES[selected_station])]
                
                progress_bar = st.progress(0, text="Initializing search...")
                
                for idx, (display_name, table_name) in enumerate(stations_to_search):
                    try:
                        progress_bar.progress(
                            (idx+1)/len(stations_to_search), 
                            text=f"Searching {display_name} station..."
                        )
                        
                        # Fetch ALL data without date filtering initially
                        df = fetch_data(
                            table_name=table_name,
                            date_column='data_date'
                        )
                        
                        if not df.empty:
                            # Convert last_updated to datetime
                            if 'last_updated' in df.columns:
                                df['last_updated_dt'] = pd.to_datetime(
                                    df['last_updated'], 
                                    format='%d/%m/%Y %H:%M', 
                                    errors='coerce'
                                )
                                
                                # Drop rows with invalid dates
                                df = df[df['last_updated_dt'].notna()]
                                
                                # Convert filter dates to datetime for comparison
                                start_dt = pd.to_datetime(start_date)
                                end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                                
                                # Function to check if majority of group dates fall in range
                                def is_majority_in_range(group):
                                    in_range = group.between(start_dt, end_dt)
                                    return in_range.mean() > 0.5
                                
                                # Group by data_time and check majority condition
                                if 'data_time' in df.columns:
                                    time_groups = df.groupby('data_time')['last_updated_dt'].transform(is_majority_in_range)
                                    df = df[time_groups]
                            
                            # Apply project filter
                            if selected_project != "All Projects" and 'project_name' in df.columns:
                                df = df[df['project_name'] == selected_project]
                            
                            # Apply location filter using location_id
                            if 'location_id' in df.columns:
                                df = df[df['location_id'] == selected_location]
                            
                            if not df.empty:
                                # Remove temporary columns before displaying
                                df = df.drop(columns=['last_updated_dt'], errors='ignore')
                                
                                # Get majority date for each data_time group for plotting
                                if 'data_time' in df.columns:
                                    majority_dates = df.groupby('data_time')['last_updated'].agg(
                                        lambda x: x.mode()[0] if not x.mode().empty else None
                                    )
                                    df['plot_date'] = df['data_time'].map(majority_dates)
                                
                                # Initialize alerts list for this station
                                station_alerts = []
                                
                                # Check for alerts in the data
                                for _, row in df.iterrows():
                                    alert_detected = False
                                    alert_info = {
                                        'station': display_name,
                                        'location': selected_location,
                                        'project': selected_project if selected_project != "All Projects" else "All",
                                        'timestamp': row.get('last_updated', ''),
                                        'alert_type': '',
                                        'alert_details': {}
                                    }
                                    
                                    # Common checks for all stations - battery voltage
                                    if 'batt_volt' in row and pd.notnull(row['batt_volt']):
                                        try:
                                            batt_volt = float(row['batt_volt'])
                                            if batt_volt < 10.5:
                                                alert_detected = True
                                                alert_info['alert_type'] = 'Low Battery (<10.5V)'
                                                alert_info['alert_details']['battery_voltage'] = batt_volt
                                        except:
                                            pass
                                    
                                    # Station-specific checks
                                    if display_name == 'Gate':
                                        gate_cols = [col for col in df.columns if re.match(r'^g\d+$', col)]
                                        for col in gate_cols:
                                            if col in row and pd.notnull(row[col]):
                                                try:
                                                    if float(row[col]) > 0.00:
                                                        alert_detected = True
                                                        alert_info['alert_type'] = 'Gate Opening Detected'
                                                        alert_info['alert_details']['gate_column'] = col
                                                        alert_info['alert_details']['gate_value'] = float(row[col])
                                                        break
                                                except:
                                                    continue
                                    
                                    elif display_name == 'EPAN' and 'epan_water_depth' in row:
                                        try:
                                            current_depth = float(row['epan_water_depth'])
                                            location_id = row['location_id'] if 'location_id' in row else None
                                            
                                            # First check for constant value (highest priority)
                                            if location_id:
                                                # Get dates to check (previous 3 days + today)
                                                dates_to_check = []
                                                days_back = 0
                                                while len(dates_to_check) < 4:  # We need 4 days total (today + 3 previous)
                                                    check_date = pd.to_datetime(row['last_updated'], format='%d/%m/%Y %H:%M', errors='coerce') - timedelta(days=days_back)
                                                    if pd.isna(check_date):
                                                        break
                                                    check_date_str = check_date.strftime('%d/%m/%Y')
                                                    
                                                    # Filter for this location and date
                                                    prev_day_df = df[
                                                        (df['last_updated'].str.startswith(check_date_str)) & 
                                                        (df['location_id'] == location_id)
                                                    ]
                                                    
                                                    if not prev_day_df.empty and 'epan_water_depth' in prev_day_df.columns:
                                                        # Take the most recent reading from that day
                                                        prev_depth = float(prev_day_df['epan_water_depth'].iloc[0])
                                                        dates_to_check.append((check_date_str, prev_depth))
                                                    
                                                    days_back += 1
                                                    if days_back > 10:  # Safety limit
                                                        break
                                                
                                                # If we have 4 days of data, check if all values are equal
                                                if len(dates_to_check) == 4:
                                                    all_equal = all(d[1] == current_depth for d in dates_to_check)
                                                    if all_equal:
                                                        alert_detected = True
                                                        alert_info['alert_type'] = 'Constant Water Depth (4 days)'
                                                        alert_info['alert_details']['constant_value'] = current_depth
                                                        alert_info['alert_details']['constant_days'] = [d[0] for d in dates_to_check]
                                            
                                            # Only check other constraints if not a constant value
                                            if not alert_detected:
                                                # Check for water depth ≤50 or ≥200
                                                if current_depth <= 50 or current_depth >= 200:
                                                    alert_detected = True
                                                    alert_info['alert_type'] = f'Water Depth {"≤50" if current_depth <=50 else "≥200"}'
                                                    alert_info['alert_details']['water_depth'] = current_depth
                                                
                                                # Previous day difference check (go back up to 10 days if needed)
                                                if location_id:
                                                    prev_depth = None
                                                    days_back = 1
                                                    comparison_date = None
                                                    
                                                    # Check up to 10 previous days for data
                                                    while days_back <= 10 and prev_depth is None:
                                                        check_date = pd.to_datetime(row['last_updated'], format='%d/%m/%Y %H:%M', errors='coerce') - timedelta(days=days_back)
                                                        if pd.isna(check_date):
                                                            break
                                                        check_date_str = check_date.strftime('%d/%m/%Y')
                                                        
                                                        # Filter for this location and date
                                                        prev_day_df = df[
                                                            (df['last_updated'].str.startswith(check_date_str)) & 
                                                            (df['location_id'] == location_id)
                                                        ]
                                                        
                                                        if not prev_day_df.empty and 'epan_water_depth' in prev_day_df.columns:
                                                            # Take the most recent reading from that day
                                                            prev_depth = float(prev_day_df['epan_water_depth'].iloc[0])
                                                            comparison_date = check_date_str
                                                        
                                                        days_back += 1
                                                    
                                                    # If we found previous data, check the difference
                                                    if prev_depth is not None:
                                                        if abs(current_depth - prev_depth) > 15:
                                                            alert_detected = True
                                                            alert_info['alert_type'] = f'Depth Change >15 (vs {comparison_date})'
                                                            alert_info['alert_details']['current_depth'] = current_depth
                                                            alert_info['alert_details']['previous_depth'] = prev_depth
                                                            alert_info['alert_details']['depth_difference'] = abs(current_depth - prev_depth)
                                        except Exception as e:
                                            st.error(f"Error processing EPAN data: {str(e)}")
                                    
                                    elif display_name == 'AWS':
                                        # Initialize alert type list
                                        alert_types = []
                                        
                                        # 1. Check for zero values in specified columns
                                        zero_value_columns = ['atmospheric_pressure', 'temperature', 'humidity', 'solar_radiation', 'wind_speed']
                                        for col in zero_value_columns:
                                            if col in row and pd.notnull(row[col]):
                                                try:
                                                    if float(row[col]) == 0:
                                                        alert_types.append(f'{col.capitalize().replace("_", " ")} is 0')
                                                        alert_info['alert_details'][col] = 0
                                                except:
                                                    pass
                                        
                                        # 2. Check for rain values > 100 (updated constraint)
                                        rain_columns = ['hourly_rain', 'daily_rain']
                                        rain_alert_cols = []
                                        for col in rain_columns:
                                            if col in row and pd.notnull(row[col]):
                                                try:
                                                    rain_value = float(row[col])
                                                    if rain_value > 100:
                                                        rain_alert_cols.append(col)
                                                        alert_info['alert_details'][col] = rain_value
                                                except:
                                                    pass
                                        
                                        if rain_alert_cols:
                                            alert_types.append('Rainfall > 100mm')
                                        
                                        # 3. Check for wind speed > 30
                                        if 'wind_speed' in row and pd.notnull(row['wind_speed']):
                                            try:
                                                wind_speed = float(row['wind_speed'])
                                                if wind_speed > 30:
                                                    alert_types.append('High Wind Speed (>30)')
                                                    alert_info['alert_details']['wind_speed'] = wind_speed
                                            except:
                                                pass
                                        
                                        # 4. Existing AWS checks
                                        if 'rainfall' in row and pd.notnull(row['rainfall']):
                                            try:
                                                if float(row['rainfall']) > 50:
                                                    alert_types.append('High Rainfall (>50mm)')
                                                    alert_info['alert_details']['rainfall'] = float(row['rainfall'])
                                            except:
                                                pass
                                        
                                        if 'temperature' in row and pd.notnull(row['temperature']):
                                            try:
                                                if float(row['temperature']) > 40:
                                                    alert_types.append('High Temperature (>40)')
                                                    alert_info['alert_details']['temperature'] = float(row['temperature'])
                                            except:
                                                pass
                                        
                                        if alert_types:
                                            alert_detected = True
                                            alert_info['alert_type'] = ', '.join(alert_types)
                                    
                                    # River/Dam station level difference check with 10-day lookback
                                    elif (display_name in ['River', 'Dam'] and 
                                        'level_mtr' in row and 
                                        'location_id' in row):
                                        try:
                                            current_level = float(row['level_mtr'])
                                            location_id = row['location_id']
                                            
                                            # Initialize variables
                                            prev_level = None
                                            days_checked = 0
                                            comparison_date = None
                                            
                                            # Check up to 10 previous days for data
                                            while days_checked < 10 and prev_level is None:
                                                check_date = pd.to_datetime(row['last_updated'], format='%d/%m/%Y %H:%M', errors='coerce') - timedelta(days=days_checked + 1)
                                                if pd.isna(check_date):
                                                    break
                                                check_date_str = check_date.strftime('%d/%m/%Y')
                                                
                                                # Filter for this location and date
                                                prev_day_df = df[
                                                    (df['last_updated'].str.startswith(check_date_str)) & 
                                                    (df['location_id'] == location_id)
                                                ]
                                                
                                                if not prev_day_df.empty and 'level_mtr' in prev_day_df.columns:
                                                    # Take the most recent reading from that day
                                                    prev_level = float(prev_day_df['level_mtr'].iloc[0])
                                                    comparison_date = check_date_str
                                                    break
                                                    
                                                days_checked += 1
                                            
                                            # If we found previous data, check the difference
                                            if prev_level is not None:
                                                level_diff = abs(current_level - prev_level)
                                                if level_diff > 1:
                                                    alert_detected = True
                                                    alert_info['alert_type'] = f'Level Change >1m (vs {comparison_date})'
                                                    alert_info['alert_details']['current_level'] = current_level
                                                    alert_info['alert_details']['previous_level'] = prev_level
                                                    alert_info['alert_details']['level_difference'] = level_diff
                                        except:
                                            pass
                                    
                                    if alert_detected:
                                        station_alerts.append(alert_info)
                                        all_alert_data.append(alert_info)
                                
                                results[display_name] = {
                                    'data': df,
                                    'alerts': station_alerts
                                }
                                total_records += len(df)
                                
                    except Exception as e:
                        st.error(f"Error processing {display_name}: {str(e)}")
                
                status.update(label="Search complete!", state="complete", expanded=False)
                
            finally:
                progress_bar.empty()

        # --------------------------- RESULTS DISPLAY ---------------------------
        if not results:
            st.info(f"🚨 No matching records found for selected filters")
        else:
            # Get the location name for display
            location_name = "Unknown"
            for station_data in results.values():
                if not station_data['data'].empty and 'location_name' in station_data['data'].columns:
                    location_name = station_data['data'].iloc[0]['location_name']
                    break
            
            st.success(f"✅ Found {total_records} records across {len(results)} stations")
            
            # Explanation of filtering logic
            st.info(f"""
                Showing all data where groups with the same data_time have majority 
                of their last_updated dates between {start_date.strftime('%d/%m/%Y')} 
                and {end_date.strftime('%d/%m/%Y')}. Each group is either fully 
                included or excluded based on the majority date analysis.
            """)
            
            # Summary Metrics
            with st.container():
                cols = st.columns(4)
                cols[0].metric("Total Stations", len(results))
                cols[1].metric("Total Records", total_records)
                cols[2].metric("Date Range", f"{start_date} to {end_date}")
                cols[3].metric("Selected Location", f"{selected_location} ({location_name})")
            
            # --------------------------- BATTERY VOLTAGE GRAPHS ---------------------------
            st.markdown("---")
            st.subheader("🔋 Battery Voltage Monitoring")
            
            for display_name, result in results.items():
                if not result['data'].empty and 'batt_volt' in result['data'].columns:
                    try:
                        df = result['data'].copy()
                        
                        # Use plot_date if available, otherwise fall back to other datetime columns
                        datetime_col = 'plot_date' if 'plot_date' in df.columns else None
                        if datetime_col is None:
                            for col in ['timestamp', 'data_date', 'last_updated']:
                                if col in df.columns:
                                    df['plot_date'] = pd.to_datetime(df[col], errors='coerce')
                                    df = df[df['plot_date'].notna()]
                                    datetime_col = 'plot_date'
                                    break
                        
                        if datetime_col is None:
                            st.warning(f"No valid datetime column found for {display_name}")
                            continue
                        
                        # Convert voltage to numeric
                        df['batt_volt'] = pd.to_numeric(df['batt_volt'], errors='coerce')
                        df = df[df['batt_volt'].notna()]
                        
                        if df.empty:
                            st.warning(f"No valid battery voltage data for {display_name}")
                            continue
                        
                        # Create time-based line graph
                        fig = px.line(
                            df,
                            x=datetime_col,
                            y='batt_volt',
                            title=(
                                f'{display_name} Station - {selected_location} ({location_name}) Battery Voltage\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})\n'
                                f'Project: {selected_project}\n'
                                f'Total Readings: {len(df)}'
                            ),
                            labels={'batt_volt': 'Voltage (V)', datetime_col: 'Date/Time'},
                            template='plotly_white',
                            line_shape='spline'
                        )

                        # Add alert threshold line
                        fig.add_hline(
                            y=10.5,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Alert Threshold (10.5V)",
                            annotation_position="bottom right"
                        )

                        # Highlight alert points
                        alerts = df[df['batt_volt'] < 10.5]
                        if not alerts.empty:
                            fig.add_trace(px.scatter(
                                alerts,
                                x=datetime_col,
                                y='batt_volt',
                                color_discrete_sequence=['red'],
                                hover_data={
                                    'batt_volt': ":.2f",
                                    datetime_col: True,
                                    'location_id': True
                                }
                            ).update_traces(
                                name='Alerts',
                                marker=dict(size=8, symbol='x')
                            ).data[0])

                        # Customize layout with explicit min and max for y-axis
                        fig.update_layout(
                            hovermode='x unified',
                            height=500,
                            xaxis=dict(
                                title='Date/Time',
                                tickformat='%d-%b-%Y %H:%M',
                                rangeslider=dict(visible=True)
                            ),
                            yaxis=dict(
                                title='Battery Voltage (V)',
                                range=[max(df['batt_volt'].min() - 0.5, 0), 14]  # Explicit min and max
                            ),
                            showlegend=False
                        )

                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)
                            
                    except Exception as e:
                        st.error(f"Error creating voltage graph for {display_name}: {str(e)}")
            
            # --------------------------- EPAN WATER DEPTH GRAPHS ---------------------------
            if 'EPAN' in results and not results['EPAN']['data'].empty and 'epan_water_depth' in results['EPAN']['data'].columns:
                st.markdown("---")
                st.subheader("💧 EPAN Water Depth Monitoring")
                
                try:
                    epan_df = results['EPAN']['data'].copy()
                    
                    # Use plot_date if available, otherwise fall back to other datetime columns
                    datetime_col = 'plot_date' if 'plot_date' in epan_df.columns else None
                    if datetime_col is None:
                        for col in ['timestamp', 'data_date', 'last_updated']:
                            if col in epan_df.columns:
                                epan_df['plot_date'] = pd.to_datetime(epan_df[col], errors='coerce')
                                epan_df = epan_df[epan_df['plot_date'].notna()]
                                datetime_col = 'plot_date'
                                break
                    
                    if datetime_col is None:
                        st.warning("No valid datetime column found for EPAN data")
                    else:
                        # Convert depth to numeric
                        epan_df['epan_water_depth'] = pd.to_numeric(epan_df['epan_water_depth'], errors='coerce')
                        epan_df = epan_df[epan_df['epan_water_depth'].notna()]
                        
                        if not epan_df.empty:
                            # Create time-based line graph
                            fig = px.line(
                                epan_df,
                                x=datetime_col,
                                y='epan_water_depth',
                                title=(
                                    f'EPAN Station - {selected_location} ({location_name}) Water Depth\n'
                                    f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})\n'
                                    f'Project: {selected_project}\n'
                                    f'Total Readings: {len(epan_df)}'
                                ),
                                labels={'epan_water_depth': 'Depth (mm)', datetime_col: 'Date/Time'},
                                template='plotly_white',
                                line_shape='spline',
                                color_discrete_sequence=['#1a73e8']
                            )

                            # Add threshold lines
                            fig.add_hline(
                                y=15,
                                line_dash="dot",
                                line_color="#ff6b35",
                                annotation_text="CRITICAL LEVEL (15mm)",
                                annotation_font_color="#ff6b35"
                            )
                            fig.add_hline(
                                y=50,
                                line_dash="dash",
                                line_color="orange",
                                annotation_text="LOW LEVEL (50mm)",
                                annotation_font_color="orange"
                            )
                            fig.add_hline(
                                y=200,
                                line_dash="dash",
                                line_color="orange",
                                annotation_text="HIGH LEVEL (200mm)",
                                annotation_font_color="orange"
                            )

                            # Highlight alert points
                            alerts = epan_df[(epan_df['epan_water_depth'] <= 50) | (epan_df['epan_water_depth'] >= 200)]
                            if not alerts.empty:
                                fig.add_trace(px.scatter(
                                    alerts,
                                    x=datetime_col,
                                    y='epan_water_depth',
                                    color_discrete_sequence=['#ff6b35'],
                                    hover_data={
                                        'epan_water_depth': ":.2f",
                                        datetime_col: True,
                                        'location_id': True
                                    }
                                ).update_traces(
                                    name='Alerts',
                                    marker=dict(size=10, symbol='hexagon', line=dict(width=2, color='DarkSlateGrey'))
                                ).data[0])

                            # Customize layout
                            fig.update_layout(
                                hovermode='x unified',
                                height=500,
                                xaxis=dict(
                                    title='Date/Time',
                                    tickformat='%d-%b-%Y %H:%M',
                                    rangeslider=dict(visible=True)
                                ),
                                yaxis=dict(
                                    title='Water Depth (mm)',
                                    range=[0, epan_df['epan_water_depth'].max() + 5]
                                ),
                                showlegend=False
                            )

                            # Display plot
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("No valid EPAN water depth data available")
                
                except Exception as e:
                    st.error(f"Error creating EPAN water depth graph: {str(e)}")

            # --------------------------- DATA TABLE DISPLAY ---------------------------
            st.markdown("---")
            st.subheader("📊 Filtered Data Table")
            
            # Combine all station data into one dataframe for display
            all_data_combined = pd.concat([result['data'] for result in results.values()])
            
            # Display the data table
            st.dataframe(
                all_data_combined,
                use_container_width=True,
                height=400,
                column_config={
                    "location_id": "Location ID",
                    "location_name": "Location Name",
                    "project_name": "Project",
                    "data_date": "Date",
                    "timestamp": "Timestamp",
                    "data_time": "Data Time",
                    "batt_volt": st.column_config.NumberColumn("Battery Voltage", format="%.2f V"),
                    "epan_water_depth": st.column_config.NumberColumn("Water Depth", format="%.1f mm")
                }
            )

def show_trends_tab():
    st.subheader("Advanced Graphical Analysis")
    
    # --------------------------- COMMON FILTERS ---------------------------
    with st.container():
        st.markdown("""
        <div style='background-color: #fff3e6; padding: 15px; border-radius: 5px; margin: 10px 0;'>
            <h2 style='color: #ff6b35; margin:0;'>🔍 Filter Parameters</h2>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date", 
                value=datetime.now() - timedelta(days=7),
                key="common_start"
            )
        with col2:
            end_date = st.date_input(
                "End Date", 
                value=datetime.now(),
                key="common_end"
            )

        station_type = st.selectbox(
            "Station Type",
            list(DATA_SOURCES.keys()),
            index=0,
            help="Select station type to analyze"
        )

        # Load station data using the new database function
        station_data = load_station_data(station_type)
        
        projects = ["All Projects"] + (station_data['project_name'].unique().tolist() 
                if 'project_name' in station_data.columns and not station_data.empty else [])
        selected_project = st.selectbox(
            "Project Name",
            options=projects,
            index=0,
            help="Select project to analyze"
        )
        
        # Filter locations based on selected project
        if selected_project != "All Projects":
            project_data = station_data[station_data['project_name'] == selected_project]
        else:
            project_data = station_data

        locations = []
        if not project_data.empty and 'location_id' in project_data.columns and 'location_name' in project_data.columns:
            locations = project_data.apply(
                lambda row: f"{row['location_id']} ({row['location_name']})", 
                axis=1
            ).unique().tolist()
            locations.sort()
        
        selected_location_display = st.selectbox(
            "Select Location",
            options=locations,
            help="Select location to analyze"
        )
        
        selected_location_id = None
        if selected_location_display and locations:
            selected_location_id = selected_location_display.split(' ')[0]

        location_details = None
        if selected_location_id and not station_data.empty:
            location_record = station_data[
                station_data['location_id'].astype(str) == selected_location_id
            ].iloc[0] if not station_data.empty else None
            
            if location_record is not None:
                location_details = {
                    "ID": location_record.get('location_id', 'N/A'),
                    "Name": location_record.get('location_name', 'N/A'),
                    "Latitude": location_record.get('latitude', 'N/A'),
                    "Longitude": location_record.get('longitude', 'N/A'),
                    "Project": selected_project
                }

    # Initialize session state for graphs and alerts
    if 'batt_fig' not in st.session_state:
        st.session_state.batt_fig = None
    if 'epan_fig' not in st.session_state:
        st.session_state.epan_fig = None
    if 'epan_diff_fig' not in st.session_state:
        st.session_state.epan_diff_fig = None
    if 'gate_fig' not in st.session_state:
        st.session_state.gate_fig = None
    if 'rain_fig' not in st.session_state:
        st.session_state.rain_fig = None
    if 'ars_rain_fig' not in st.session_state:
        st.session_state.ars_rain_fig = None
    if 'aws_params_fig' not in st.session_state:
        st.session_state.aws_params_fig = None
    if 'river_level_fig' not in st.session_state:
        st.session_state.river_level_fig = None
    if 'dam_level_fig' not in st.session_state:
        st.session_state.dam_level_fig = None
    
    # Initialize alert DataFrames
    if 'batt_alerts' not in st.session_state:
        st.session_state.batt_alerts = pd.DataFrame()
    if 'epan_low_alerts' not in st.session_state:
        st.session_state.epan_low_alerts = pd.DataFrame()
    if 'epan_high_alerts' not in st.session_state:
        st.session_state.epan_high_alerts = pd.DataFrame()
    if 'epan_diff_alerts' not in st.session_state:
        st.session_state.epan_diff_alerts = pd.DataFrame()
    if 'epan_constant_alert' not in st.session_state:
        st.session_state.epan_constant_alert = None
    if 'gate_alerts' not in st.session_state:
        st.session_state.gate_alerts = pd.DataFrame()
    if 'rain_alerts' not in st.session_state:
        st.session_state.rain_alerts = pd.DataFrame()
    if 'ars_rain_alerts' not in st.session_state:
        st.session_state.ars_rain_alerts = pd.DataFrame()
    if 'aws_zero_alerts' not in st.session_state:
        st.session_state.aws_zero_alerts = pd.DataFrame()
    if 'river_alerts' not in st.session_state:
        st.session_state.river_alerts = pd.DataFrame()
    if 'dam_alerts' not in st.session_state:
        st.session_state.dam_alerts = pd.DataFrame()
    
    # Initialize graph visibility states with defaults
    if 'show_batt' not in st.session_state:
        st.session_state.show_batt = True
    if 'show_epan' not in st.session_state:
        st.session_state.show_epan = True
    if 'show_epan_diff' not in st.session_state:
        st.session_state.show_epan_diff = True
    if 'show_gate' not in st.session_state:
        st.session_state.show_gate = True
    if 'show_rain' not in st.session_state:
        st.session_state.show_rain = True
    if 'show_ars_rain' not in st.session_state:
        st.session_state.show_ars_rain = True
    if 'show_aws_params' not in st.session_state:
        st.session_state.show_aws_params = True
    if 'show_river_level' not in st.session_state:
        st.session_state.show_river_level = True
    if 'show_dam_level' not in st.session_state:
        st.session_state.show_dam_level = True

    # --------------------------- ANALYSIS EXECUTION ---------------------------
    if st.button("Generate Analysis", type="primary", key="common_generate"):
        if not locations:
            st.warning("No locations available for selected filters")
        elif not selected_location_id:
            st.warning("Please select a location")
        else:
            try:
                # Clear previous state
                st.session_state.batt_fig = None
                st.session_state.epan_fig = None
                st.session_state.epan_diff_fig = None
                st.session_state.gate_fig = None
                st.session_state.rain_fig = None
                st.session_state.ars_rain_fig = None
                st.session_state.aws_params_fig = None
                st.session_state.river_level_fig = None
                st.session_state.dam_level_fig = None
                st.session_state.batt_alerts = pd.DataFrame()
                st.session_state.epan_low_alerts = pd.DataFrame()
                st.session_state.epan_high_alerts = pd.DataFrame()
                st.session_state.epan_diff_alerts = pd.DataFrame()
                st.session_state.epan_constant_alert = None
                st.session_state.gate_alerts = pd.DataFrame()
                st.session_state.rain_alerts = pd.DataFrame()
                st.session_state.ars_rain_alerts = pd.DataFrame()
                st.session_state.aws_zero_alerts = pd.DataFrame()
                st.session_state.river_alerts = pd.DataFrame()
                st.session_state.dam_alerts = pd.DataFrame()
                
                # Filter data
                df = station_data.copy()
                if selected_project != "All Projects":
                    df = df[df['project_name'] == selected_project]
                
                filtered_df = df[df['location_id'].astype(str) == selected_location_id].copy()
                
                if filtered_df.empty:
                    st.warning("No data found for selected location")
                    st.stop()

                # Process dates
                filtered_df['last_updated_dt'] = pd.to_datetime(
                    filtered_df['last_updated'], 
                    format='%d/%m/%Y %H:%M', 
                    errors='coerce'
                )
                filtered_df = filtered_df[filtered_df['last_updated_dt'].notna()]
                
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)
                
                filtered_df = filtered_df[
                    (filtered_df['last_updated_dt'] >= start_dt) & 
                    (filtered_df['last_updated_dt'] <= end_dt)
                ]
                
                if filtered_df.empty:
                    st.warning(f"""No data available for {selected_location_display} between
                            {start_date.strftime('%d-%b-%Y')} and {end_date.strftime('%d-%b-%Y')}""")
                    st.stop()

                datetime_col = 'last_updated_dt'

                # --------------------------- BATTERY VOLTAGE GRAPH ---------------------------
                if 'batt_volt' in filtered_df.columns:
                    filtered_df['batt_volt'] = pd.to_numeric(
                        filtered_df['batt_volt'], 
                        errors='coerce'
                    )
                    batt_df = filtered_df[filtered_df['batt_volt'].notna()].copy()
                    
                    if not batt_df.empty:
                        batt_fig = px.line(
                            batt_df,
                            x=datetime_col,
                            y='batt_volt',
                            title=(
                                f'🔋 {station_type} Station - {selected_location_display} Battery Voltage\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            labels={'batt_volt': 'Voltage (V)'},
                            template='plotly_white',
                            line_shape='spline'
                        )

                        batt_fig.add_hline(
                            y=10.5,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Alert Threshold (10.5V)",
                            annotation_position="bottom right"
                        )

                        batt_alerts = batt_df[batt_df['batt_volt'] < 10.5]
                        st.session_state.batt_alerts = batt_alerts
                        
                        if not batt_alerts.empty:
                            batt_fig.add_trace(px.scatter(
                                batt_alerts,
                                x=datetime_col,
                                y='batt_volt',
                                color_discrete_sequence=['red'],
                                hover_data={
                                    'batt_volt': ":.2f",
                                    datetime_col: True,
                                    'location_id': True
                                }
                            ).update_traces(
                                name='Alerts',
                                marker=dict(size=8, symbol='x')
                            ).data[0])

                        batt_fig.update_layout(
                            hovermode='x unified',
                            height=400,
                            xaxis=dict(
                                title='Date/Time',
                                tickformat='%d-%b-%Y %H:%M',
                                rangeslider=dict(visible=True)
                            ),
                            yaxis=dict(
                                title='Battery Voltage (V)',
                                range=[max(batt_df['batt_volt'].min() - 0.5, 0), 14]
                            ),
                            showlegend=False
                        )
                        
                        st.session_state.batt_fig = batt_fig

                # --------------------------- EPAN WATER DEPTH GRAPH ---------------------------
                if station_type == "EPAN" and 'epan_water_depth' in filtered_df.columns:
                    filtered_df['epan_water_depth'] = pd.to_numeric(
                        filtered_df['epan_water_depth'], 
                        errors='coerce'
                    )
                    epan_df = filtered_df[filtered_df['epan_water_depth'].notna()].copy()
                    
                    if not epan_df.empty:
                        # Create EPAN Water Depth graph
                        epan_fig = px.line(
                            epan_df,
                            x=datetime_col,
                            y='epan_water_depth',
                            title=(
                                f'💧 EPAN Station - {selected_location_display} Water Depth\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            labels={'epan_water_depth': 'Depth (mm)'},
                            template='plotly_white',
                            line_shape='spline',
                            color_discrete_sequence=['#1a73e8']
                        )

                        # Threshold lines
                        epan_fig.add_hline(
                            y=15,
                            line_dash="dot",
                            line_color="#ff6b35",
                            annotation_text="CRITICAL LEVEL (15mm)",
                            annotation_font_color="#ff6b35"
                        )
                        
                        epan_fig.add_hline(
                            y=200,
                            line_dash="dash",
                            line_color="#ff0000",
                            annotation_text="ALERT LEVEL (200mm)",
                            annotation_position="top right",
                            annotation_font_color="#ff0000"
                        )

                        # Constant value detection (only if date range >= 4 days)
                        time_range_days = (end_dt - start_dt).days
                        if time_range_days >= 4:
                            # Get last 4 days of data (even if not consecutive)
                            last_4_days = epan_df.sort_values('last_updated_dt', ascending=False).head(4)
                            
                            # Check if all values are the same
                            if last_4_days['epan_water_depth'].nunique() == 1:
                                constant_value = last_4_days['epan_water_depth'].iloc[0]
                                
                                st.session_state.epan_constant_alert = {
                                    'value': constant_value,
                                    'start': last_4_days['last_updated_dt'].min(),
                                    'end': last_4_days['last_updated_dt'].max(),
                                    'dates': last_4_days['last_updated_dt'].dt.strftime('%Y-%m-%d').unique()
                                }
                                
                                epan_fig.add_trace(px.line(
                                    last_4_days,
                                    x=datetime_col,
                                    y='epan_water_depth',
                                    color_discrete_sequence=['red']
                                ).update_traces(
                                    line=dict(width=4),
                                    name='Constant Value Alert'
                                ).data[0])

                        # Alerts
                        st.session_state.epan_low_alerts = epan_df[epan_df['epan_water_depth'] < 15]
                        st.session_state.epan_high_alerts = epan_df[epan_df['epan_water_depth'] == 200]
                        
                        # Low alerts
                        if not st.session_state.epan_low_alerts.empty:
                            epan_fig.add_trace(px.scatter(
                                st.session_state.epan_low_alerts,
                                x=datetime_col,
                                y='epan_water_depth',
                                color_discrete_sequence=['#ff6b35'],
                                hover_data={
                                    'epan_water_depth': ":.2f",
                                    datetime_col: True,
                                    'location_id': True
                                }
                            ).update_traces(
                                name='Low Alerts',
                                marker=dict(size=10, symbol='hexagon', line=dict(width=2, color='DarkSlateGrey'))
                            ).data[0])
                        
                        # High alerts
                        if not st.session_state.epan_high_alerts.empty:
                            epan_fig.add_trace(px.scatter(
                                st.session_state.epan_high_alerts,
                                x=datetime_col,
                                y='epan_water_depth',
                                color_discrete_sequence=['red'],
                                hover_data={
                                    'epan_water_depth': ":.2f",
                                    datetime_col: True,
                                    'location_id': True
                                }
                            ).update_traces(
                                name='High Alerts (200mm)',
                                marker=dict(size=10, symbol='diamond', line=dict(width=2, color='black'))
                            ).data[0])

                        epan_fig.update_layout(
                            hovermode='x unified',
                            height=400,
                            xaxis=dict(
                                title='Date/Time',
                                tickformat='%d-%b-%Y %H:%M',
                                rangeslider=dict(visible=True)
                            ),
                            yaxis=dict(
                                title='Water Depth (mm)',
                                range=[0, max(epan_df['epan_water_depth'].max() + 5, 200)]
                            ),
                            showlegend=True
                        )
                        
                        st.session_state.epan_fig = epan_fig
                        
                        # Create EPAN Daily Difference graph
                        # Create daily levels (last reading of the day)
                        daily_epan = epan_df.resample('D', on='last_updated_dt').agg(
                            {'epan_water_depth': 'last'}
                        ).reset_index()
                        
                        # Calculate daily differences
                        daily_epan['prev_depth'] = daily_epan['epan_water_depth'].shift(1)
                        daily_epan['depth_diff'] = daily_epan['epan_water_depth'] - daily_epan['prev_depth']
                        
                        # Fill gaps by propagating last valid observation
                        daily_epan['prev_depth_filled'] = daily_epan['prev_depth'].ffill()
                        daily_epan['depth_diff_filled'] = daily_epan['epan_water_depth'] - daily_epan['prev_depth_filled']
                        
                        # Create alerts for differences > 15mm
                        epan_diff_alerts = daily_epan[daily_epan['depth_diff_filled'].abs() > 15]
                        st.session_state.epan_diff_alerts = epan_diff_alerts
                        
                        # Create plot
                        epan_diff_fig = go.Figure()
                        
                        # Add water depth trace
                        epan_diff_fig.add_trace(go.Scatter(
                            x=daily_epan['last_updated_dt'],
                            y=daily_epan['epan_water_depth'],
                            mode='lines+markers',
                            name='Water Depth',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add difference trace
                        epan_diff_fig.add_trace(go.Bar(
                            x=daily_epan['last_updated_dt'],
                            y=daily_epan['depth_diff_filled'],
                            name='Daily Difference',
                            marker_color='orange',
                            opacity=0.7
                        ))
                        
                        # Add alert markers
                        if not epan_diff_alerts.empty:
                            epan_diff_fig.add_trace(go.Scatter(
                                x=epan_diff_alerts['last_updated_dt'],
                                y=epan_diff_alerts['epan_water_depth'],
                                mode='markers',
                                name='Change Alert',
                                marker=dict(color='red', size=10, symbol='triangle-up')
                            ))
                        
                        epan_diff_fig.update_layout(
                            title=(
                                f'📈 EPAN Daily Water Depth Change - {selected_location_display}\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            yaxis_title='Water Depth (mm)',
                            height=400,
                            barmode='overlay'
                        )
                        
                        # Add threshold lines
                        epan_diff_fig.add_hline(
                            y=15,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Upper Threshold"
                        )
                        
                        epan_diff_fig.add_hline(
                            y=-15,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Lower Threshold"
                        )
                        
                        st.session_state.epan_diff_fig = epan_diff_fig
                
                # --------------------------- GATE ANALYSIS ---------------------------
                if station_type == "Gate":
                    # Create list of gate columns
                    gate_cols = [col for col in filtered_df.columns if col.startswith('g') and col[1:].isdigit()]
                    
                    if gate_cols:
                        # Convert gate columns to numeric
                        for col in gate_cols:
                            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce').fillna(0)
                        
                        # Create daily aggregates
                        filtered_df['date'] = filtered_df['last_updated_dt'].dt.date
                        daily_gate = filtered_df.groupby('date')[gate_cols].max().reset_index()
                        
                        # Find days with gate activity
                        gate_alerts = daily_gate.copy()
                        gate_alerts['active_gates'] = gate_alerts.apply(
                            lambda row: [col for col in gate_cols if row[col] > 0], 
                            axis=1
                        )
                        gate_alerts = gate_alerts[gate_alerts['active_gates'].apply(len) > 0]
                        st.session_state.gate_alerts = gate_alerts
                        
                        # Create plot with all gates
                        gate_fig = go.Figure()
                        for col in gate_cols:
                            gate_fig.add_trace(go.Bar(
                                x=daily_gate['date'],
                                y=daily_gate[col],
                                name=f'Gate {col[1:]}',
                                hovertemplate='%{y}',
                                visible='legendonly'  # Start with gates hidden
                            ))
                        
                        # Add trace for total open gates
                        daily_gate['total_open'] = daily_gate[gate_cols].gt(0).sum(axis=1)
                        gate_fig.add_trace(go.Bar(
                            x=daily_gate['date'],
                            y=daily_gate['total_open'],
                            name='Total Open Gates',
                            marker_color='#1f77b4',
                            hovertemplate='Total: %{y}'
                        ))
                        
                        gate_fig.update_layout(
                            title=(
                                f'🚪 Gate Activity - {selected_location_display}\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            barmode='stack',
                            yaxis=dict(title='Gate Value / Count'),
                            height=400
                        )
                        
                        st.session_state.gate_fig = gate_fig

                # --------------------------- AWS RAIN ANALYSIS ---------------------------
                if station_type == "AWS":
                    # Identify rain columns
                    rain_cols = [col for col in ['daily_rain', 'hourly_rain'] if col in filtered_df.columns]
                    
                    if rain_cols:
                        # Convert rain columns to numeric
                        for col in rain_cols:
                            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                        
                        # Create alert column (1 if heavy rain, 0 otherwise)
                        filtered_df['heavy_rain'] = 0
                        for col in rain_cols:
                            filtered_df['heavy_rain'] = filtered_df['heavy_rain'] | (filtered_df[col] > 100)
                        
                        # Create plot
                        rain_fig = go.Figure()
                        
                        # Add rain data
                        for col in rain_cols:
                            rain_fig.add_trace(go.Scatter(
                                x=filtered_df['last_updated_dt'],
                                y=filtered_df[col],
                                mode='lines',
                                name=col.replace('_', ' ').title(),
                                line=dict(width=2)
                            ))
                        
                        # Add alert markers
                        rain_alerts = filtered_df[filtered_df['heavy_rain'] == 1]
                        if not rain_alerts.empty:
                            rain_fig.add_trace(go.Scatter(
                                x=rain_alerts['last_updated_dt'],
                                y=[105] * len(rain_alerts),
                                mode='markers',
                                name='Heavy Rain Alert',
                                marker=dict(color='red', size=8, symbol='x')
                            ))
                        
                        rain_fig.update_layout(
                            title=(
                                f'🌧 AWS Rain Analysis - {selected_location_display}\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            yaxis_title='Rainfall (mm)',
                            height=400,
                            hovermode='x unified'
                        )
                        
                        rain_fig.add_hline(
                            y=100,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Alert Threshold (100mm)"
                        )
                        
                        st.session_state.rain_fig = rain_fig
                        st.session_state.rain_alerts = rain_alerts

                # --------------------------- ARS RAIN ANALYSIS ---------------------------
                if station_type == "ARS":
                    # Identify rain columns
                    rain_cols = [col for col in ['daily_rain', 'hourly_rain'] if col in filtered_df.columns]
                    
                    if rain_cols:
                        # Convert rain columns to numeric
                        for col in rain_cols:
                            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                        
                        # Create alert column (1 if heavy rain, 0 otherwise)
                        filtered_df['heavy_rain'] = 0
                        for col in rain_cols:
                            filtered_df['heavy_rain'] = filtered_df['heavy_rain'] | (filtered_df[col] > 100)
                        
                        # Create plot
                        ars_rain_fig = go.Figure()
                        
                        # Add rain data
                        for col in rain_cols:
                            ars_rain_fig.add_trace(go.Scatter(
                                x=filtered_df['last_updated_dt'],
                                y=filtered_df[col],
                                mode='lines',
                                name=col.replace('_', ' ').title(),
                                line=dict(width=2)
                            ))
                        
                        # Add alert markers
                        ars_rain_alerts = filtered_df[filtered_df['heavy_rain'] == 1]
                        if not ars_rain_alerts.empty:
                            ars_rain_fig.add_trace(go.Scatter(
                                x=ars_rain_alerts['last_updated_dt'],
                                y=[105] * len(ars_rain_alerts),
                                mode='markers',
                                name='Heavy Rain Alert',
                                marker=dict(color='red', size=8, symbol='x')
                            ))
                        
                        ars_rain_fig.update_layout(
                            title=(
                                f'🌧 ARS Rain Analysis - {selected_location_display}\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            yaxis_title='Rainfall (mm)',
                            height=400,
                            hovermode='x unified'
                        )
                        
                        ars_rain_fig.add_hline(
                            y=100,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Alert Threshold (100mm)"
                        )
                        
                        st.session_state.ars_rain_fig = ars_rain_fig
                        st.session_state.ars_rain_alerts = ars_rain_alerts

                # --------------------------- AWS PARAMETERS ANALYSIS ---------------------------
                if station_type == "AWS":
                    # Define sensor columns
                    sensor_cols = ['wind_speed', 'wind_direction', 'atm_pressure', 
                                'temperature', 'humidity', 'solar_radiation']
                    
                    # Convert sensor columns to numeric
                    for col in sensor_cols:
                        if col in filtered_df.columns:
                            filtered_df[col] = pd.to_numeric(filtered_df[col], errors='coerce')
                    
                    # Create plot
                    aws_params_fig = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    
                    # Add sensor data
                    for i, col in enumerate(sensor_cols):
                        if col in filtered_df.columns:
                            aws_params_fig.add_trace(go.Scatter(
                                x=filtered_df['last_updated_dt'],
                                y=filtered_df[col],
                                mode='lines',
                                name=col.replace('_', ' ').title(),
                                line=dict(color=colors[i % len(colors)], width=2),
                                yaxis=f'y{i+1}' if i > 0 else 'y'
                            ))
                    
                    # Create zero alerts
                    zero_alerts = []
                    for col in sensor_cols:
                        if col in filtered_df.columns:
                            zero_mask = filtered_df[col] == 0
                            if zero_mask.any():
                                zero_df = filtered_df[zero_mask]
                                for _, row in zero_df.iterrows():
                                    zero_alerts.append({
                                        'Timestamp': row['last_updated_dt'],
                                        'Parameter': col,
                                        'Value': 0
                                    })
                    
                    st.session_state.aws_zero_alerts = pd.DataFrame(zero_alerts)
                    
                    # Add alert markers
                    if zero_alerts:
                        for col in sensor_cols:
                            if col in filtered_df.columns:
                                zero_points = filtered_df[filtered_df[col] == 0]
                                if not zero_points.empty:
                                    aws_params_fig.add_trace(go.Scatter(
                                        x=zero_points['last_updated_dt'],
                                        y=zero_points[col],
                                        mode='markers',
                                        name=f'{col} Zero Alert',
                                        marker=dict(color='red', size=8, symbol='x')
                                    ))
                    
                    # Create axis layout
                    layout = dict(
                        title=(
                            f'🌬 AWS Parameters - {selected_location_display}\n'
                            f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                        ),
                        height=500,
                        hovermode='x unified'
                    )
                    
                    # Add multiple y-axes if needed
                    for i, col in enumerate(sensor_cols):
                        if col in filtered_df.columns:
                            if i == 0:
                                layout['yaxis'] = dict(title=f'{col}'.title())
                            else:
                                layout[f'yaxis{i+1}'] = dict(
                                    title=f'{col}'.title(),
                                    overlaying='y',
                                    side='right',
                                    position=1 - (0.1 * i)
                                )
                    
                    aws_params_fig.update_layout(layout)
                    st.session_state.aws_params_fig = aws_params_fig

                # --------------------------- RIVER LEVEL ANALYSIS ---------------------------
                if station_type == "River" and 'water_level' in filtered_df.columns:
                    # Convert water level to numeric
                    filtered_df['water_level'] = pd.to_numeric(filtered_df['water_level'], errors='coerce')
                    river_df = filtered_df[filtered_df['water_level'].notna()].copy()
                    
                    if not river_df.empty:
                        # Create daily levels (last reading of the day)
                        daily_river = river_df.resample('D', on='last_updated_dt').agg(
                            {'water_level': 'last'}
                        ).reset_index()
                        
                        # Calculate daily differences using the previous available day
                        daily_river['prev_level'] = daily_river['water_level'].shift(1)
                        
                        # Forward fill missing previous days
                        daily_river['prev_level_filled'] = daily_river['prev_level'].ffill()
                        
                        # Calculate differences using filled values
                        daily_river['level_diff'] = daily_river['water_level'] - daily_river['prev_level_filled']
                        
                        # Create alerts for differences > 1m
                        river_alerts = daily_river[
                            (daily_river['level_diff'].abs() > 1) & 
                            (daily_river['prev_level_filled'].notna())  # Ensure we have a valid comparison
                        ]
                        st.session_state.river_alerts = river_alerts
                        
                        # Create plot
                        river_level_fig = go.Figure()
                        
                        # Add water level trace
                        river_level_fig.add_trace(go.Scatter(
                            x=daily_river['last_updated_dt'],
                            y=daily_river['water_level'],
                            mode='lines+markers',
                            name='Water Level',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Add difference trace
                        river_level_fig.add_trace(go.Bar(
                            x=daily_river['last_updated_dt'],
                            y=daily_river['level_diff'],
                            name='Daily Difference',
                            marker_color='orange',
                            opacity=0.7
                        ))
                        
                        # Add alert markers
                        if not river_alerts.empty:
                            river_level_fig.add_trace(go.Scatter(
                                x=river_alerts['last_updated_dt'],
                                y=river_alerts['water_level'],
                                mode='markers',
                                name='Level Change Alert',
                                marker=dict(color='red', size=10, symbol='triangle-up')
                            ))
                        
                        river_level_fig.update_layout(
                            title=(
                                f'🌊 River Level Analysis - {selected_location_display}\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            yaxis_title='Water Level (m)',
                            height=400,
                            barmode='overlay'
                        )
                        
                        # Add threshold lines
                        river_level_fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Upper Threshold (1m)"
                        )
                        
                        river_level_fig.add_hline(
                            y=-1,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Lower Threshold (-1m)"
                        )
                        
                        st.session_state.river_level_fig = river_level_fig

                # --------------------------- DAM LEVEL ANALYSIS ---------------------------
                if station_type == "Dam" and 'water_level' in filtered_df.columns:
                    # Convert water level to numeric
                    filtered_df['water_level'] = pd.to_numeric(filtered_df['water_level'], errors='coerce')
                    dam_df = filtered_df[filtered_df['water_level'].notna()].copy()
                    
                    if not dam_df.empty:
                        # Create daily levels (last reading of the day)
                        daily_dam = dam_df.resample('D', on='last_updated_dt').agg(
                            {'water_level': 'last'}
                        ).reset_index()
                        
                        # Calculate daily differences using the previous available day
                        daily_dam['prev_level'] = daily_dam['water_level'].shift(1)
                        
                        # Forward fill missing previous days
                        daily_dam['prev_level_filled'] = daily_dam['prev_level'].ffill()
                        
                        # Calculate differences using filled values
                        daily_dam['level_diff'] = daily_dam['water_level'] - daily_dam['prev_level_filled']
                        
                        # Create alerts for differences > 1m
                        dam_alerts = daily_dam[
                            (daily_dam['level_diff'].abs() > 1) & 
                            (daily_dam['prev_level_filled'].notna())  # Ensure we have a valid comparison
                        ]
                        st.session_state.dam_alerts = dam_alerts
                        
                        # Create plot
                        dam_level_fig = go.Figure()
                        
                        # Add water level trace
                        dam_level_fig.add_trace(go.Scatter(
                            x=daily_dam['last_updated_dt'],
                            y=daily_dam['water_level'],
                            mode='lines+markers',
                            name='Water Level',
                            line=dict(color='green', width=2)
                        ))
                        
                        # Add difference trace
                        dam_level_fig.add_trace(go.Bar(
                            x=daily_dam['last_updated_dt'],
                            y=daily_dam['level_diff'],
                            name='Daily Difference',
                            marker_color='purple',
                            opacity=0.7
                        ))
                        
                        # Add alert markers
                        if not dam_alerts.empty:
                            dam_level_fig.add_trace(go.Scatter(
                                x=dam_alerts['last_updated_dt'],
                                y=dam_alerts['water_level'],
                                mode='markers',
                                name='Level Change Alert',
                                marker=dict(color='red', size=10, symbol='triangle-up')
                            ))
                        
                        dam_level_fig.update_layout(
                            title=(
                                f'💧 Dam Level Analysis - {selected_location_display}\n'
                                f'({start_date.strftime("%d-%b-%Y")} to {end_date.strftime("%d-%b-%Y")})'
                            ),
                            yaxis_title='Water Level (m)',
                            height=400,
                            barmode='overlay'
                        )
                        
                        # Add threshold lines
                        dam_level_fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Upper Threshold (1m)"
                        )
                        
                        dam_level_fig.add_hline(
                            y=-1,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="Lower Threshold (-1m)"
                        )
                        
                        st.session_state.dam_level_fig = dam_level_fig

            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                st.stop()

    # --------------------------- DISPLAY LOCATION DETAILS ---------------------------
    if location_details:
        st.subheader("📍 Location Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Location ID", location_details["ID"])
            st.metric("Latitude", location_details["Latitude"])
        with col2:
            st.metric("Location Name", location_details["Name"])
            st.metric("Longitude", location_details["Longitude"])
        with col3:
            st.metric("Project", location_details["Project"])
            st.metric("Station Type", station_type)
    
    # --------------------------- GRAPH VISIBILITY CONTROLS ---------------------------
    # Only show if at least one graph is available
    graphs = [
        st.session_state.batt_fig, st.session_state.epan_fig, st.session_state.epan_diff_fig,
        st.session_state.gate_fig, st.session_state.rain_fig, st.session_state.ars_rain_fig,
        st.session_state.aws_params_fig, st.session_state.river_level_fig, st.session_state.dam_level_fig
    ]
    if any(graphs):
        st.subheader("📊 Graph Visibility Options")
        
        if station_type in ["River", "Dam"]:
            # Special layout for River/Dam stations
            col1, col2 = st.columns(2)
            
            if st.session_state.batt_fig:
                st.session_state.show_batt = col1.checkbox(
                    "Battery Voltage", 
                    value=st.session_state.show_batt,
                    key="vis_batt_river_dam"
                )
            
            if station_type == "River" and st.session_state.river_level_fig:
                st.session_state.show_river_level = col2.checkbox(
                    "River Level", 
                    value=st.session_state.show_river_level,
                    key="vis_river_level"
                )
            elif station_type == "Dam" and st.session_state.dam_level_fig:
                st.session_state.show_dam_level = col2.checkbox(
                    "Dam Level", 
                    value=st.session_state.show_dam_level,
                    key="vis_dam_level"
                )
        else:
            # Standard layout for other station types
            cols = st.columns(9)
            col_index = 0
            
            # Battery Voltage
            if st.session_state.batt_fig:
                st.session_state.show_batt = cols[col_index].checkbox(
                    "Battery Voltage", 
                    value=st.session_state.show_batt,
                    key="vis_batt"
                )
                col_index += 1
            
            # EPAN Water Depth
            if st.session_state.epan_fig:
                st.session_state.show_epan = cols[col_index].checkbox(
                    "EPAN Water Depth", 
                    value=st.session_state.show_epan,
                    key="vis_epan"
                )
                col_index += 1
            
            # EPAN Daily Difference
            if st.session_state.epan_diff_fig:
                st.session_state.show_epan_diff = cols[col_index].checkbox(
                    "EPAN Daily Change", 
                    value=st.session_state.show_epan_diff,
                    key="vis_epan_diff"
                )
                col_index += 1
            
            # Gate Activity
            if st.session_state.gate_fig:
                st.session_state.show_gate = cols[col_index].checkbox(
                    "Gate Activity", 
                    value=st.session_state.show_gate,
                    key="vis_gate"
                )
                col_index += 1
            
            # AWS Rain Analysis
            if st.session_state.rain_fig:
                st.session_state.show_rain = cols[col_index].checkbox(
                    "AWS Rain Analysis", 
                    value=st.session_state.show_rain,
                    key="vis_rain"
                )
                col_index += 1
            
            # ARS Rain Analysis
            if st.session_state.ars_rain_fig:
                st.session_state.show_ars_rain = cols[col_index].checkbox(
                    "ARS Rain Analysis", 
                    value=st.session_state.show_ars_rain,
                    key="vis_ars_rain"
                )
                col_index += 1
            
            # AWS Parameters
            if st.session_state.aws_params_fig:
                st.session_state.show_aws_params = cols[col_index].checkbox(
                    "AWS Parameters", 
                    value=st.session_state.show_aws_params,
                    key="vis_aws_params"
                )
                col_index += 1
    
    # --------------------------- DISPLAY GRAPHS ---------------------------
    # Battery Voltage (for all station types)
    if st.session_state.batt_fig and st.session_state.show_batt:
        st.subheader("🔋 Battery Voltage Analysis")
        st.plotly_chart(st.session_state.batt_fig, use_container_width=True)
        
        if not st.session_state.batt_alerts.empty:
            alert_count = len(st.session_state.batt_alerts)
            with st.expander(f"🔴 Alerts Detected: {alert_count} instances below 10.5V", expanded=False):
                st.dataframe(st.session_state.batt_alerts[['last_updated_dt', 'batt_volt']].rename(
                    columns={'last_updated_dt': 'Timestamp', 'batt_volt': 'Voltage (V)'}
                ))
        else:
            st.success("✅ No voltage alerts detected in selected period")
    
    # River Level Analysis
    if station_type == "River" and st.session_state.river_level_fig and st.session_state.show_river_level:
        st.subheader("🌊 River Level Analysis")
        st.plotly_chart(st.session_state.river_level_fig, use_container_width=True)
        
        if not st.session_state.river_alerts.empty:
            alert_count = len(st.session_state.river_alerts)
            with st.expander(f"🔴 Level Change Alerts: {alert_count} days with >1m difference", expanded=False):
                display_df = st.session_state.river_alerts.copy()
                display_df['Difference'] = display_df['level_diff'].apply(
                    lambda x: f"{x:.2f} m"
                )
                display_df['Previous Level'] = display_df['prev_level_filled'].apply(
                    lambda x: f"{x:.2f} m"
                )
                st.dataframe(display_df[[
                    'last_updated_dt', 
                    'water_level', 
                    'Previous Level',
                    'Difference'
                ]].rename(
                    columns={
                        'last_updated_dt': 'Date', 
                        'water_level': 'Current Level (m)'
                    }
                ))
        else:
            st.success("✅ No significant river level changes detected")
    
    # Dam Level Analysis
    if station_type == "Dam" and st.session_state.dam_level_fig and st.session_state.show_dam_level:
        st.subheader("💧 Dam Level Analysis")
        st.plotly_chart(st.session_state.dam_level_fig, use_container_width=True)
        
        if not st.session_state.dam_alerts.empty:
            alert_count = len(st.session_state.dam_alerts)
            with st.expander(f"🔴 Level Change Alerts: {alert_count} days with >1m difference", expanded=False):
                display_df = st.session_state.dam_alerts.copy()
                display_df['Difference'] = display_df['level_diff'].apply(
                    lambda x: f"{x:.2f} m"
                )
                display_df['Previous Level'] = display_df['prev_level_filled'].apply(
                    lambda x: f"{x:.2f} m"
                )
                st.dataframe(display_df[[
                    'last_updated_dt', 
                    'water_level', 
                    'Previous Level',
                    'Difference'
                ]].rename(
                    columns={
                        'last_updated_dt': 'Date', 
                        'water_level': 'Current Level (m)'
                    }
                ))
        else:
            st.success("✅ No significant dam level changes detected")
    
    # EPAN Water Depth
    if st.session_state.epan_fig and st.session_state.show_epan:
        st.subheader("💧 EPAN Water Depth Analysis")
        st.plotly_chart(st.session_state.epan_fig, use_container_width=True)
    
        # EPAN alerts section
        any_epan_alerts = False
        
        # Low alerts
        if not st.session_state.epan_low_alerts.empty:
            any_epan_alerts = True
            alert_count = len(st.session_state.epan_low_alerts)
            with st.expander(f"🔴 Low Depth Alerts: {alert_count} instances below 15mm", expanded=False):
                st.dataframe(st.session_state.epan_low_alerts[['last_updated_dt', 'epan_water_depth']].rename(
                    columns={'last_updated_dt': 'Timestamp', 'epan_water_depth': 'Depth (mm)'}
                ))
        
        # High alerts
        if not st.session_state.epan_high_alerts.empty:
            any_epan_alerts = True
            alert_count = len(st.session_state.epan_high_alerts)
            with st.expander(f"🔴 High Depth Alerts: {alert_count} instances at exactly 200mm", expanded=False):
                st.dataframe(st.session_state.epan_high_alerts[['last_updated_dt', 'epan_water_depth']].rename(
                    columns={'last_updated_dt': 'Timestamp', 'epan_water_depth': 'Depth (mm)'}
                ))
        
        # Constant value alert
        if st.session_state.epan_constant_alert:
            any_epan_alerts = True
            with st.expander(f"🔴 Constant Value Alert (Last 4 Days)", expanded=False):
                st.write(f"Constant value of {st.session_state.epan_constant_alert['value']} mm detected")
                st.write(f"*Period:* {st.session_state.epan_constant_alert['start'].strftime('%Y-%m-%d %H:%M')} to "
                        f"{st.session_state.epan_constant_alert['end'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"*Dates:* {', '.join(st.session_state.epan_constant_alert['dates'])}")
        
        if station_type == "EPAN" and not any_epan_alerts:
            st.success("✅ No EPAN depth alerts detected in selected period")

    # EPAN Daily Difference
    if st.session_state.epan_diff_fig and st.session_state.show_epan_diff:
        st.subheader("📈 EPAN Daily Water Depth Change")
        st.plotly_chart(st.session_state.epan_diff_fig, use_container_width=True)
        
        if not st.session_state.epan_diff_alerts.empty:
            alert_count = len(st.session_state.epan_diff_alerts)
            with st.expander(f"🔴 Change Alerts: {alert_count} days with >15mm difference", expanded=False):
                # Create display dataframe with all needed columns
                display_df = st.session_state.epan_diff_alerts.copy()
                
                # Add formatted columns for display
                display_df['Previous Day'] = display_df['prev_depth'].apply(
                    lambda x: f"{x:.2f} mm" if not pd.isna(x) else "N/A"
                )
                display_df['Current Day'] = display_df['epan_water_depth'].apply(
                    lambda x: f"{x:.2f} mm"
                )
                display_df['Difference'] = display_df['depth_diff_filled'].apply(
                    lambda x: f"{x:.2f} mm"
                )
                
                # Add arrow indicator showing change direction
                def get_change_direction(row):
                    if pd.isna(row['prev_depth']) or pd.isna(row['depth_diff_filled']):
                        return ""
                    if row['depth_diff_filled'] > 0:
                        return "⬆ Increase"
                    elif row['depth_diff_filled'] < 0:
                        return "⬇ Decrease"
                    return "↔ No Change"
                
                display_df['Change'] = display_df.apply(get_change_direction, axis=1)
                
                # Create the display dataframe with renamed columns
                st.dataframe(
                    display_df[[
                        'last_updated_dt', 
                        'Previous Day', 
                        'Current Day',
                        'Difference',
                        'Change'
                    ]].rename(columns={
                        'last_updated_dt': 'Date',
                    }),
                    use_container_width=True
                )
                
                # Add explanation of the change direction
                st.caption("""
                    Change Direction Indicators  
                    ⬆ Increase: Water depth increased compared to previous day  
                    ⬇ Decrease: Water depth decreased compared to previous day  
                    ↔ No Change: Depth remained the same (difference = 0)
                """)
        else:
            st.success("✅ No significant water depth changes detected")

    # Gate Activity
    if st.session_state.gate_fig and st.session_state.show_gate:
        st.subheader("🚪 Gate Activity Analysis")
        st.plotly_chart(st.session_state.gate_fig, use_container_width=True)
        
        if not st.session_state.gate_alerts.empty:
            alert_count = len(st.session_state.gate_alerts)
            with st.expander(f"🔴 Gate Activity Detected: {alert_count} days with open gates", expanded=False):
                # Format gate information for display
                display_df = st.session_state.gate_alerts.copy()
                display_df['Active Gates'] = display_df['active_gates'].apply(
                    lambda gates: ', '.join([g.replace('g', 'Gate ') for g in gates]) if gates else 'None'
                )
                st.dataframe(display_df[['date', 'Active Gates']].rename(columns={'date': 'Date'}))
        else:
            st.success("✅ No gate activity detected in selected period")

    # AWS Rain Analysis
    if st.session_state.rain_fig and st.session_state.show_rain:
        st.subheader("🌧 AWS Rain Analysis")
        st.plotly_chart(st.session_state.rain_fig, use_container_width=True)
        
        if not st.session_state.rain_alerts.empty:
            alert_count = len(st.session_state.rain_alerts)
            with st.expander(f"🔴 Heavy Rain Alerts: {alert_count} instances above 100mm", expanded=False):
                st.dataframe(st.session_state.rain_alerts[['last_updated_dt', 'daily_rain', 'hourly_rain']].rename(
                    columns={'last_updated_dt': 'Timestamp', 'daily_rain': 'Daily Rain (mm)', 'hourly_rain': 'Hourly Rain (mm)'}
                ))
        else:
            st.success("✅ No heavy rain alerts detected in selected period")

    # ARS Rain Analysis
    if st.session_state.ars_rain_fig and st.session_state.show_ars_rain:
        st.subheader("🌧 ARS Rain Analysis")
        st.plotly_chart(st.session_state.ars_rain_fig, use_container_width=True)
        
        if not st.session_state.ars_rain_alerts.empty:
            alert_count = len(st.session_state.ars_rain_alerts)
            with st.expander(f"🔴 Heavy Rain Alerts: {alert_count} instances above 100mm", expanded=False):
                st.dataframe(st.session_state.ars_rain_alerts[['last_updated_dt', 'daily_rain', 'hourly_rain']].rename(
                    columns={'last_updated_dt': 'Timestamp', 'daily_rain': 'Daily Rain (mm)', 'hourly_rain': 'Hourly Rain (mm)'}
                ))
        else:
            st.success("✅ No heavy rain alerts detected in selected period")

    # AWS Parameters
    if st.session_state.aws_params_fig and st.session_state.show_aws_params:
        st.subheader("🌬 AWS Parameters Analysis")
        st.plotly_chart(st.session_state.aws_params_fig, use_container_width=True)
        
        if not st.session_state.aws_zero_alerts.empty:
            alert_count = len(st.session_state.aws_zero_alerts)
            with st.expander(f"🔴 Zero Value Alerts: {alert_count} instances with sensor readings at zero", expanded=False):
                st.dataframe(st.session_state.aws_zero_alerts)
        else:
            st.success("✅ All AWS sensors reported non-zero values")

# --------------------------- MAIN APP ---------------------------
def main_app():
    # Initialize tab state
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "🌐 Overview"
    
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.markdown("""
        <div class="dashboard-header">
            <h1 class="dashboard-title">HydroAnalytics Pro</h1>
            <p class="dashboard-subtitle">Advanced Water Management Intelligence Platform</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Top metrics
    render_top_metrics()
    
    # Create tabs
    tabs = st.tabs(["🌐 Overview", "📡 Categories", "📜 History", "🔍 Custom Query", "📈 Trends"])
    
    # Update current tab based on selection
    with tabs[0]:
        st.session_state.current_tab = "🌐 Overview"
        show_overview_tab()
    
    with tabs[1]:
        st.session_state.current_tab = "📡 Categories"
        show_categories_tab()
    
    with tabs[2]:
        st.session_state.current_tab = "📜 History"
        show_history_tab()

    with tabs[3]:
        st.session_state.current_tab = "🔍 Custom Query"
        show_custom_tab()

    with tabs[4]:
        st.session_state.current_tab = "📈 Trends"
        show_trends_tab()

# --------------------------- CUSTOM CSS ---------------------------
st.markdown(r"""
        <style>
            /* ===== GLOBAL THEME ===== */
            :root {
                --primary: #333333;    /* Vibrant blue */
                --primary-dark: #2667cc;  /* Darker blue */
                --secondary: #6c757d;     /* Cool gray */
                --accent: #20c997;        /* Teal */
                --background: #f8f9fa;    /* Light gray */
                --card-bg: #f0f0f0;       /* Pure white */
                --text-primary: #212529;  /* Dark gray */
                --text-secondary: #495057;/* Medium gray */
                --success: #28a745;      /* Green */
                --warning: #ffc107;      /* Yellow */
                --danger: #dc3545;       /* Red */
                --dark: #343a40;         /* Dark */
            }

            /* ===== BASE STYLES ===== */
            html, body, .main {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                background-color: var(--background);
                color: var(--text-primary);
                line-height: 1.6;
                margin: 0;
                padding: 0;
            }

            /* ===== MAIN CONTAINER ===== */
          

            /* ===== HEADER ===== */

            .dashboard-title {
                font-weight: 800;
                color: var(--primary);
                margin: 0;
                line-height: 1.2;
                letter-spacing: -0.5px;
            }

            .dashboard-subtitle {
                color: var(--secondary);
                font-weight: 400;
                opacity: 0.9;
            }

            /* ===== METRIC CARDS ===== */
            .metric-card {
                background: var(--card-bg);
                border-radius: 12px;
                padding: 1.75rem;
                margin: 1rem 0;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
                border: 1px solid rgba(0, 0, 0, 0.03);
                transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
                position: relative;
                overflow: hidden;
                height: 100%;
            }

            .metric-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                background: rgba(160, 164, 184, 1);
                border-color: rgba(0, 123, 255, 0.5); /* Bootstrap blue with 50% opacity */
            }

            .metric-card-icon {
                background: linear-gradient(135deg, var(--primary), var(--accent));
                width: 56px;
                height: 56px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.25rem;
            }

            .metric-card-icon span {
                color: white;
                font-size: 1.75rem;
            }

            .metric-card-value {
                font-size: 2.25rem;
                font-weight: 700;
                color: var(--text-primary);
                margin: 0.25rem 0;
                line-height: 1.2;
            }

            .metric-card-label {
                font-size: 0.95rem;
                color: var(--text-secondary);
                opacity: 0.9;
            }

            /* ===== SIDEBAR ===== */
            [data-testid="stSidebar"] {
                background: linear-gradient(195deg, #1e293b 0%, #0f172a 100%);
                box-shadow: 5px 0 15px rgba(0, 0, 0, 0.1);
                padding: 1.5rem;
            }

            [data-testid="stSidebar"] .stButton button {
                background-color: rgba(255, 255, 255, 0.08);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.12);
                width: 100%;
                transition: all 0.2s;
                border-radius: 8px;
                padding: 0.75rem;
                font-weight: 500;
                margin-bottom: 0.75rem;
            }

            [data-testid="stSidebar"] .stButton button:hover {
                background-color: rgba(255, 255, 255, 0.15);
                transform: translateY(-1px);
                border-color: rgba(255, 255, 255, 0.2);
            }

            /* ===== TABS ===== */
            .stTabs [role="tablist"] {
                gap: 0.5rem;
                padding: 0.5rem;
                background: rgba(203, 213, 225, 0.1);
                border-radius: 12px;
                border: none;
            }

            .stTabs [role="tab"] {
                border-radius: 10px !important;
                padding: 0.75rem 1.5rem !important;
                background: rgba(203, 213, 225, 0.1) !important;
                border: none !important;
                color: var(--text-secondary) !important;
                transition: all 0.3s ease;
                font-weight: 500;
                margin: 0 !important;
            }

            .stTabs [role="tab"][aria-selected="true"] {
                background: var(--primary) !important;
                color: white !important;
                box-shadow: 0 2px 8px rgba(58, 134, 255, 0.2);
            }

            /* ===== BUTTONS ===== */
            .stButton > button {
                background-color: var(--primary);
                color: white;
                border: none;
                border-radius: 10px;
                padding: 0.75rem 1.75rem;
                font-weight: 500;
                transition: all 0.2s;
                box-shadow: 0 2px 5px rgba(58, 134, 255, 0.15);
            }

            .stButton > button:hover {
                background-color: var(--primary-dark);
                transform: translateY(-2px);
                box-shadow: 0 5px 12px rgba(58, 134, 255, 0.25);
            }

            /* ===== DATAFRAMES & TABLES ===== */
            .stDataFrame {
                border-radius: 12px !important;
                border: 1px solid rgba(0, 0, 0, 0.05) !important;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.03) !important;
            }

            /* ===== RESPONSIVE DESIGN ===== */
            @media (max-width: 768px) {
                .dashboard-title {
                    font-size: 2.25rem;
                }
                
                .dashboard-subtitle {
                    font-size: 1rem;
                }
                
                .metric-card {
                    padding: 1.5rem !important;
                }
                
                .metric-card-icon {
                    width: 48px;
                    height: 48px;
                }
            }

            /* ===== ANIMATIONS ===== */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }

            .stApp > div {
                animation: fadeIn 0.4s ease-out;
            }
        </style>
    """, unsafe_allow_html=True)
            
    # --------------------------- AUTHENTICATION ---------------------------
    # --------------------------- AUTHENTICATION ---------------------------
# --------------------------- APP FLOW ---------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_page()
    st.stop()  # This will stop execution if not authenticated

# Only runs if authenticated
main_app()
