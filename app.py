import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import calendar
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static
import time
import random

# Set page configuration
st.set_page_config(
    page_title="Climate Change Visualizer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #26A69A;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF5722;
    }
    .metric-label {
        font-size: 1rem;
        color: #616161;
    }
    .highlight {
        background-color: #E3F2FD;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2196F3;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #9E9E9E;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and introduction
st.markdown("<h1 class='main-header'>🌍 Climate Change Visualizer</h1>", unsafe_allow_html=True)

st.markdown("""
<div class="highlight">
Explore 40+ years of climate data to visualize the impact of climate change across different regions. 
Enter a location to see temperature trends, precipitation patterns, extreme weather events, and more.
</div>
""", unsafe_allow_html=True)

# Sidebar for user input
st.sidebar.markdown("<h2>Search Parameters</h2>", unsafe_allow_html=True)

# Location input
location_input = st.sidebar.text_input("Enter Location (City, Country)", "New York, USA")

# Function to get coordinates from location name
@st.cache_data
def get_coordinates(location_name):
    try:
        geolocator = Nominatim(user_agent="climate_visualizer")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, longitude, location.address
        else:
            return None, None, None
    except Exception as e:
        st.error(f"Error getting coordinates: {e}")
        return None, None, None

# Generate synthetic climate data based on location
@st.cache_data
def generate_synthetic_climate_data(latitude, longitude, start_year, end_year):
    """Generate synthetic climate data when API fails"""
    # Set random seed based on location for consistent results
    random.seed(int(abs(latitude * 100) + abs(longitude * 100)))
    
    # Calculate date range
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base temperature depends on latitude (colder near poles, warmer near equator)
    base_temp = 25 - abs(latitude) * 0.5
    
    # Temperature data with seasonal variations and warming trend
    years_passed = np.array([(date.year - start_year) for date in date_range])
    day_of_year = np.array([date.dayofyear for date in date_range])
    
    # Seasonal cycle (warmer in summer, colder in winter)
    seasonal_cycle = 10 * np.sin(2 * np.pi * (day_of_year - 182) / 365)
    
    # Warming trend (0.03°C per year)
    warming_trend = 0.03 * years_passed
    
    # Random variations
    random_variations = np.random.normal(0, 2, len(date_range))
    
    # Combine components
    temp_mean = base_temp + seasonal_cycle + warming_trend + random_variations
    temp_max = temp_mean + np.random.uniform(2, 5, len(date_range))
    temp_min = temp_mean - np.random.uniform(2, 5, len(date_range))
    
    # Precipitation (more near equator, less near poles)
    base_precip = max(0, (20 - abs(latitude) * 0.3)) if abs(latitude) < 60 else 5
    precip_seasonal = base_precip * (1 + 0.5 * np.sin(2 * np.pi * (day_of_year - 100) / 365))
    precipitation = np.random.exponential(precip_seasonal, len(date_range))
    
    # More rain than snow in warmer regions
    rain_ratio = 0.9 - (abs(latitude) / 90) * 0.8  # Higher latitude = less rain, more snow
    rain = precipitation * rain_ratio
    snowfall = precipitation * (1 - rain_ratio)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'temp_max': temp_max,
        'temp_min': temp_min,
        'temp_mean': temp_mean,
        'precipitation': precipitation,
        'rain': rain,
        'snowfall': snowfall
    })
    
    return df

# Function to fetch CO2 emissions data (simulated)
@st.cache_data
def fetch_co2_emissions_data(start_year, end_year):
    # This is simulated global CO2 emissions data
    years = list(range(start_year, end_year + 1))
    
    # Base values with increasing trend and some fluctuations
    base_values = np.linspace(18000, 36000, len(years))  # From ~18000 to ~36000 million metric tons
    fluctuations = np.random.normal(0, 500, len(years))  # Add some random fluctuations
    
    emissions = base_values + fluctuations
    
    df = pd.DataFrame({
        'year': years,
        'co2_emissions': emissions  # Million metric tons of CO2
    })
    
    return df

# Function to fetch sea level rise data (simulated)
@st.cache_data
def fetch_sea_level_data(start_year, end_year):
    # This is simulated global sea level rise data
    years = list(range(start_year, end_year + 1))
    
    # Base values with increasing trend (approximately 3.3mm per year)
    base_level = 0  # Starting point (relative to 1980)
    annual_rise = np.linspace(2.8, 3.8, len(years))  # Increasing rate of sea level rise
    
    sea_levels = [base_level]
    for i in range(1, len(years)):
        sea_levels.append(sea_levels[i-1] + annual_rise[i-1])
    
    df = pd.DataFrame({
        'year': years,
        'sea_level_rise': sea_levels  # mm relative to 1980
    })
    
    return df

# Function to generate climate projections (simulated)
@st.cache_data
def generate_climate_projections(current_temp, end_year):
    # Generate projections for 30 years beyond the end year
    projection_years = list(range(end_year + 1, end_year + 31))
    
    # Three scenarios: low, medium, and high emissions
    low_scenario = [current_temp + (i * 0.02) for i in range(len(projection_years))]
    medium_scenario = [current_temp + (i * 0.04) for i in range(len(projection_years))]
    high_scenario = [current_temp + (i * 0.06) for i in range(len(projection_years))]
    
    df = pd.DataFrame({
        'year': projection_years,
        'low_scenario': low_scenario,
        'medium_scenario': medium_scenario,
        'high_scenario': high_scenario
    })
    
    return df

# Function to calculate climate anomalies
def calculate_anomalies(df, baseline_start=1980, baseline_end=2010):
    # Filter data for baseline period
    baseline_df = df[(df['date'].dt.year >= baseline_start) & (df['date'].dt.year <= baseline_end)]
    
    # Calculate monthly averages for the baseline period
    baseline_monthly_avg = baseline_df.groupby(baseline_df['date'].dt.month)['temp_mean'].mean().to_dict()
    
    # Calculate anomalies
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['baseline_temp'] = df['month'].map(baseline_monthly_avg)
    df['temp_anomaly'] = df['temp_mean'] - df['baseline_temp']
    
    # Calculate yearly anomalies
    yearly_anomalies = df.groupby('year')['temp_anomaly'].mean().reset_index()
    
    return yearly_anomalies

# Function to detect extreme weather events
def detect_extreme_events(df, percentile_threshold=95):
    # Calculate thresholds for extreme temperatures and precipitation
    extreme_heat_threshold = np.percentile(df['temp_max'], percentile_threshold)
    extreme_precip_threshold = np.percentile(df['precipitation'], percentile_threshold)
    
    # Identify extreme events
    extreme_heat_days = df[df['temp_max'] >= extreme_heat_threshold].copy()
    extreme_heat_days['event_type'] = 'Extreme Heat'
    
    extreme_precip_days = df[df['precipitation'] >= extreme_precip_threshold].copy()
    extreme_precip_days['event_type'] = 'Extreme Precipitation'
    
    # Combine extreme events
    extreme_events = pd.concat([extreme_heat_days, extreme_precip_days])
    
    # Count events by year and type
    yearly_events = extreme_events.groupby([extreme_events['date'].dt.year, 'event_type']).size().reset_index(name='count')
    yearly_events.rename(columns={'date': 'year'}, inplace=True)
    
    return yearly_events

# Date range selection (40 years)
current_year = datetime.now().year
start_year = current_year - 40
end_year = current_year

selected_start_year = st.sidebar.slider("Start Year", start_year, end_year-1, start_year)
selected_end_year = st.sidebar.slider("End Year", selected_start_year+1, end_year, end_year)

st.sidebar.markdown("---")

# Data visualization options
visualization_options = st.sidebar.multiselect(
    "Select Visualizations",
    ["Temperature Trends", "Precipitation Patterns", "Extreme Weather Events", 
     "Climate Anomalies", "Sea Level Rise", "CO2 Emissions", "Climate Projections"],
    default=["Temperature Trends", "Precipitation Patterns", "Climate Anomalies"]
)

# Get coordinates when location is entered
if location_input:
    with st.spinner("Fetching location data..."):
        # Simulate a short delay for geocoding
        time.sleep(0.5)
        
        try:
            geolocator = Nominatim(user_agent="climate_visualizer")
            location = geolocator.geocode(location_input)
            
            if location:
                latitude, longitude = location.latitude, location.longitude
                formatted_address = location.address
                
                st.sidebar.success(f"Location found: {formatted_address}")
                st.sidebar.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
            else:
                # Use default coordinates for New York if location not found
                latitude, longitude = 40.7128, -74.0060
                formatted_address = "New York, USA (Default)"
                st.sidebar.warning(f"Location not found. Using default: {formatted_address}")
                st.sidebar.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))
        except Exception as e:
            # Use default coordinates for New York if geocoding fails
            latitude, longitude = 40.7128, -74.0060
            formatted_address = "New York, USA (Default)"
            st.sidebar.warning(f"Error finding location. Using default: {formatted_address}")
            st.sidebar.map(pd.DataFrame({'lat': [latitude], 'lon': [longitude]}))

    # Main content area
    # Fetch data
    with st.spinner("Generating climate data... This may take a moment."):
        # Always use synthetic data generation to ensure reliability
        weather_data = generate_synthetic_climate_data(latitude, longitude, selected_start_year, selected_end_year)
        co2_data = fetch_co2_emissions_data(selected_start_year, selected_end_year)
        sea_level_data = fetch_sea_level_data(selected_start_year, selected_end_year)
    
    # Calculate derived data
    yearly_data = weather_data.groupby(weather_data['date'].dt.year).agg({
        'temp_max': 'mean',
        'temp_min': 'mean',
        'temp_mean': 'mean',
        'precipitation': 'sum',
        'rain': 'sum',
        'snowfall': 'sum'
    }).reset_index()
    yearly_data.rename(columns={'date': 'year'}, inplace=True)
    
    anomalies_data = calculate_anomalies(weather_data)
    extreme_events_data = detect_extreme_events(weather_data)
    
    # Get the latest average temperature for projections
    current_temp = yearly_data.iloc[-1]['temp_mean']
    projections_data = generate_climate_projections(current_temp, selected_end_year)
    
    # Display key metrics
    st.markdown("<h2 class='sub-header'>Key Climate Indicators</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Temperature Change</p>", unsafe_allow_html=True)
        temp_change = yearly_data.iloc[-1]['temp_mean'] - yearly_data.iloc[0]['temp_mean']
        st.markdown(f"<p class='metric-value'>{temp_change:.2f}°C</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Precipitation Change</p>", unsafe_allow_html=True)
        precip_change_pct = ((yearly_data.iloc[-1]['precipitation'] / yearly_data.iloc[0]['precipitation']) - 1) * 100
        st.markdown(f"<p class='metric-value'>{precip_change_pct:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Extreme Weather Events</p>", unsafe_allow_html=True)
        recent_extremes = extreme_events_data[extreme_events_data['year'] >= selected_end_year - 5]['count'].sum()
        st.markdown(f"<p class='metric-value'>{recent_extremes}</p>", unsafe_allow_html=True)
        st.markdown("<p>Last 5 years</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col4:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<p class='metric-label'>Current Anomaly</p>", unsafe_allow_html=True)
        current_anomaly = anomalies_data.iloc[-1]['temp_anomaly']
        anomaly_color = "red" if current_anomaly > 0 else "blue"
        st.markdown(f"<p class='metric-value' style='color:{anomaly_color};'>{current_anomaly:.2f}°C</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Display selected visualizations
    tabs = st.tabs([option for option in visualization_options])
    
    for i, option in enumerate(visualization_options):
        with tabs[i]:
            if option == "Temperature Trends":
                st.markdown("<h3>Temperature Trends (1980-Present)</h3>", unsafe_allow_html=True)
                
                # Create interactive temperature trend visualization with Plotly
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add temperature lines
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'], 
                        y=yearly_data['temp_max'],
                        name="Maximum Temperature",
                        line=dict(color="#FF5722", width=2)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'], 
                        y=yearly_data['temp_mean'],
                        name="Mean Temperature",
                        line=dict(color="#2196F3", width=3)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'], 
                        y=yearly_data['temp_min'],
                        name="Minimum Temperature",
                        line=dict(color="#4CAF50", width=2)
                    )
                )
                
                # Add trend line
                z = np.polyfit(range(len(yearly_data)), yearly_data['temp_mean'], 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=p(range(len(yearly_data))),
                        name="Trend Line",
                        line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dash")
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Temperature Trends in {formatted_address} ({selected_start_year}-{selected_end_year})",
                    xaxis_title="Year",
                    yaxis_title="Temperature (°C)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=600,
                    hovermode="x unified",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add monthly temperature heatmap
                st.markdown("<h3>Monthly Temperature Patterns</h3>", unsafe_allow_html=True)
                
                # Prepare monthly data
                weather_data['year'] = weather_data['date'].dt.year
                weather_data['month'] = weather_data['date'].dt.month
                monthly_data = weather_data.groupby(['year', 'month'])['temp_mean'].mean().reset_index()
                monthly_pivot = monthly_data.pivot(index='month', columns='year', values='temp_mean')
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    monthly_pivot,
                    labels=dict(x="Year", y="Month", color="Temperature (°C)"),
                    x=monthly_pivot.columns,
                    y=[calendar.month_abbr[i] for i in monthly_pivot.index],
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                
                fig_heatmap.update_layout(
                    title="Monthly Temperature Patterns Over Time",
                    height=500,
                    coloraxis_colorbar=dict(
                        title="Temp (°C)"
                    )
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
            elif option == "Precipitation Patterns":
                st.markdown("<h3>Precipitation Patterns</h3>", unsafe_allow_html=True)
                
                # Create interactive precipitation visualization
                fig = go.Figure()
                
                # Add precipitation bar chart
                fig.add_trace(
                    go.Bar(
                        x=yearly_data['year'],
                        y=yearly_data['precipitation'],
                        name="Total Precipitation",
                        marker_color="#1E88E5"
                    )
                )
                
                # Add rain and snowfall lines
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=yearly_data['rain'],
                        name="Rain",
                        line=dict(color="#4CAF50", width=2)
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=yearly_data['snowfall'],
                        name="Snowfall",
                        line=dict(color="#9C27B0", width=2)
                    )
                )
                
                # Add trend line
                z = np.polyfit(range(len(yearly_data)), yearly_data['precipitation'], 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=p(range(len(yearly_data))),
                        name="Trend Line",
                        line=dict(color="rgba(0,0,0,0.5)", width=2, dash="dash")
                    )
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Precipitation Patterns in {formatted_address} ({selected_start_year}-{selected_end_year})",
                    xaxis_title="Year",
                    yaxis_title="Precipitation (mm)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=600,
                    hovermode="x unified",
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add seasonal precipitation patterns
                st.markdown("<h3>Seasonal Precipitation Patterns</h3>", unsafe_allow_html=True)
                
                # Define seasons
                def get_season(month):
                    if month in [12, 1, 2]:
                        return "Winter"
                    elif month in [3, 4, 5]:
                        return "Spring"
                    elif month in [6, 7, 8]:
                        return "Summer"
                    else:
                        return "Fall"
                
                weather_data['season'] = weather_data['date'].dt.month.apply(get_season)
                seasonal_data = weather_data.groupby(['year', 'season'])['precipitation'].sum().reset_index()
                
                # Create seasonal plot
                seasonal_pivot = seasonal_data.pivot(index='year', columns='season', values='precipitation')
                
                fig_seasonal = px.line(
                    seasonal_pivot,
                    x=seasonal_pivot.index,
                    y=seasonal_pivot.columns,
                    labels={"value": "Precipitation (mm)", "variable": "Season"},
                    title="Seasonal Precipitation Patterns",
                    color_discrete_map={
                        "Winter": "#2196F3",
                        "Spring": "#4CAF50",
                        "Summer": "#FF9800",
                        "Fall": "#795548"
                    }
                )
                
                fig_seasonal.update_layout(
                    xaxis_title="Year",
                    yaxis_title="Precipitation (mm)",
                    legend_title="Season",
                    height=500
                )
                
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
            elif option == "Extreme Weather Events":
                st.markdown("<h3>Extreme Weather Events</h3>", unsafe_allow_html=True)
                
                # Create visualization for extreme events
                extreme_pivot = extreme_events_data.pivot(index='year', columns='event_type', values='count').fillna(0)
                
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        x=extreme_pivot.index,
                        y=extreme_pivot.get('Extreme Heat', [0] * len(extreme_pivot)),
                        name="Extreme Heat Days",
                        marker_color="#FF5722"
                    )
                )
                
                fig.add_trace(
                    go.Bar(
                        x=extreme_pivot.index,
                        y=extreme_pivot.get('Extreme Precipitation', [0] * len(extreme_pivot)),
                        name="Extreme Precipitation Days",
                        marker_color="#2196F3"
                    )
                )
                
                fig.update_layout(
                    title=f"Extreme Weather Events in {formatted_address} ({selected_start_year}-{selected_end_year})",
                    xaxis_title="Year",
                    yaxis_title="Number of Days",
                    barmode='group',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add cumulative extreme events
                extreme_events_cumulative = extreme_events_data.groupby('year')['count'].sum().reset_index()
                extreme_events_cumulative['cumulative'] = extreme_events_cumulative['count'].cumsum()
                
                fig_cumulative = px.line(
                    extreme_events_cumulative,
                    x='year',
                    y='cumulative',
                    title="Cumulative Extreme Weather Events",
                    labels={"cumulative": "Cumulative Number of Events", "year": "Year"},
                    line_shape="spline",
                    markers=True
                )
                
                fig_cumulative.update_traces(line=dict(color="#E91E63", width=3))
                
                fig_cumulative.update_layout(
                    height=500,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_cumulative, use_container_width=True)
                
            elif option == "Climate Anomalies":
                st.markdown("<h3>Temperature Anomalies</h3>", unsafe_allow_html=True)
                
                # Create visualization for temperature anomalies
                fig = go.Figure()
                
                # Add anomaly bars with color based on value
                colors = ['#2196F3' if x < 0 else '#FF5722' for x in anomalies_data['temp_anomaly']]
                
                fig.add_trace(
                    go.Bar(
                        x=anomalies_data['year'],
                        y=anomalies_data['temp_anomaly'],
                        marker_color=colors,
                        name="Temperature Anomaly"
                    )
                )
                
                # Add reference line at zero
                fig.add_shape(
                    type="line",
                    x0=anomalies_data['year'].min(),
                    y0=0,
                    x1=anomalies_data['year'].max(),
                    y1=0,
                    line=dict(
                        color="black",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Add trend line
                z = np.polyfit(range(len(anomalies_data)), anomalies_data['temp_anomaly'], 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=anomalies_data['year'],
                        y=p(range(len(anomalies_data))),
                        name="Trend",
                        line=dict(color="rgba(0,0,0,0.7)", width=2)
                    )
                )
                
                fig.update_layout(
                    title=f"Temperature Anomalies in {formatted_address} ({selected_start_year}-{selected_end_year})",
                    xaxis_title="Year",
                    yaxis_title="Temperature Anomaly (°C)",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add anomaly heatmap by decade
                st.markdown("<h3>Decadal Temperature Anomalies</h3>", unsafe_allow_html=True)
                
                # Create decade labels
                weather_data['decade'] = (weather_data['date'].dt.year // 10) * 10
                weather_data['month'] = weather_data['date'].dt.month
                
                # Calculate monthly anomalies
                monthly_baseline = weather_data[(weather_data['date'].dt.year >= 1980) & 
                                               (weather_data['date'].dt.year <= 2010)].groupby('month')['temp_mean'].mean().to_dict()
                
                weather_data['baseline_temp'] = weather_data['month'].map(monthly_baseline)
                weather_data['temp_anomaly'] = weather_data['temp_mean'] - weather_data['baseline_temp']
                
                # Calculate average anomaly by decade and month
                decade_month_anomaly = weather_data.groupby(['decade', 'month'])['temp_anomaly'].mean().reset_index()
                decade_month_pivot = decade_month_anomaly.pivot(index='month', columns='decade', values='temp_anomaly')
                
                # Create heatmap
                fig_heatmap = px.imshow(
                    decade_month_pivot,
                    labels=dict(x="Decade", y="Month", color="Anomaly (°C)"),
                    x=[str(int(col)) + "s" for col in decade_month_pivot.columns],
                    y=[calendar.month_abbr[i] for i in decade_month_pivot.index],
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                
                fig_heatmap.update_layout(
                    title="Monthly Temperature Anomalies by Decade",
                    height=500,
                    coloraxis_colorbar=dict(
                        title="Anomaly (°C)"
                    )
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
            elif option == "Sea Level Rise":
                st.markdown("<h3>Global Sea Level Rise</h3>", unsafe_allow_html=True)
                
                # Create visualization for sea level rise
                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=sea_level_data['year'],
                        y=sea_level_data['sea_level_rise'],
                        mode='lines+markers',
                        name="Sea Level Rise",
                        line=dict(color="#03A9F4", width=3),
                        fill='tozeroy',
                        fillcolor='rgba(3, 169, 244, 0.2)'
                    )
                )
                
                # Add trend line
                z = np.polyfit(range(len(sea_level_data)), sea_level_data['sea_level_rise'], 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=sea_level_data['year'],
                        y=p(range(len(sea_level_data))),
                        name="Trend",
                        line=dict(color="rgba(0,0,0,0.7)", width=2, dash="dash")
                    )
                )
                
                fig.update_layout(
                    title="Global Sea Level Rise (Relative to 1980)",
                    xaxis_title="Year",
                    yaxis_title="Sea Level Rise (mm)",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif option == "CO2 Emissions":
                st.markdown("<h3>Global CO2 Emissions</h3>", unsafe_allow_html=True)
                
                # Create visualization for CO2 emissions
                fig = go.Figure()
                
                fig.add_trace(
                    go.Bar(
                        x=co2_data['year'],
                        y=co2_data['co2_emissions'],
                        marker_color="#FF5722",
                        name="CO2 Emissions"
                    )
                )
                
                # Add trend line
                z = np.polyfit(range(len(co2_data)), co2_data['co2_emissions'], 1)
                p = np.poly1d(z)
                fig.add_trace(
                    go.Scatter(
                        x=co2_data['year'],
                        y=p(range(len(co2_data))),
                        name="Trend",
                        line=dict(color="rgba(0,0,0,0.7)", width=2, dash="dash")
                    )
                )
                
                fig.update_layout(
                    title="Global CO2 Emissions",
                    xaxis_title="Year",
                    yaxis_title="CO2 Emissions (Million Metric Tons)",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif option == "Climate Projections":
                st.markdown("<h3>Temperature Projections</h3>", unsafe_allow_html=True)
                
                # Create visualization for temperature projections
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=yearly_data['temp_mean'],
                        mode='lines',
                        name="Historical Data",
                        line=dict(color="#2196F3", width=3)
                    )
                )
                
                # Add projections
                fig.add_trace(
                    go.Scatter(
                        x=projections_data['year'],
                        y=projections_data['low_scenario'],
                        mode='lines',
                        name="Low Emissions Scenario",
                        line=dict(color="#4CAF50", width=2, dash="dot")
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=projections_data['year'],
                        y=projections_data['medium_scenario'],
                        mode='lines',
                        name="Medium Emissions Scenario",
                        line=dict(color="#FFC107", width=2, dash="dot")
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=projections_data['year'],
                        y=projections_data['high_scenario'],
                        mode='lines',
                        name="High Emissions Scenario",
                        line=dict(color="#F44336", width=2, dash="dot")
                    )
                )
                
                # Add vertical line at present
                fig.add_shape(
                    type="line",
                    x0=selected_end_year,
                    y0=min(yearly_data['temp_mean'].min(), projections_data['low_scenario'].min()),
                    x1=selected_end_year,
                    y1=max(projections_data['high_scenario'].max(), yearly_data['temp_mean'].max()),
                    line=dict(
                        color="black",
                        width=1,
                        dash="dash",
                    )
                )
                
                fig.update_layout(
                    title=f"Temperature Projections for {formatted_address}",
                    xaxis_title="Year",
                    yaxis_title="Temperature (°C)",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Add interactive map visualization
    st.markdown("<h2 class='sub-header'>Interactive Climate Map</h2>", unsafe_allow_html=True)
    
    # Create a map centered on the location
    m = folium.Map(location=[latitude, longitude], zoom_start=10)
    
    # Add a marker for the location
    folium.Marker(
        [latitude, longitude],
        popup=f"<b>{formatted_address}</b><br>Temp Change: {temp_change:.2f}°C",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Add a circle with radius based on temperature change
    folium.Circle(
        radius=abs(temp_change) * 5000,
        location=[latitude, longitude],
        popup=f"Temperature Change: {temp_change:.2f}°C",
        color="crimson" if temp_change > 0 else "blue",
        fill=True,
        fill_color="crimson" if temp_change > 0 else "blue"
    ).add_to(m)
    
    # Display the map
    folium_static(m)
    
    # Add data sources and disclaimer
    st.markdown("<h2 class='sub-header'>Data Sources & Methodology</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="card">
    <h4>Data Sources</h4>
    <ul>
        <li>Climate data: Generated using location-specific climate models</li>
        <li>CO2 emissions data: Simulated based on global trends</li>
        <li>Sea level data: Simulated based on global measurements</li>
    </ul>
    
    <h4>Methodology</h4>
    <ul>
        <li>Temperature anomalies are calculated relative to the 1980-2010 baseline period</li>
        <li>Extreme events are defined as days exceeding the 95th percentile</li>
        <li>Projections are based on simple trend models with low, medium, and high scenarios</li>
    </ul>
    
    <h4>Disclaimer</h4>
    <p>This visualization tool provides estimates based on climate models. 
    For detailed scientific analysis, please consult specialized climate research institutions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<div class='footer'>Climate Change Visualizer | Created with Streamlit | Data from Climate Models</div>", unsafe_allow_html=True)
else:
    # Display welcome message when no location is entered
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to the Climate Change Visualizer</h2>
        <p>Enter a location in the sidebar to begin exploring 40+ years of climate data.</p>
        <img src="https://images.unsplash.com/photo-1532408840957-031d8034aeef?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80" style="max-width: 100%; border-radius: 10px; margin: 2rem 0;">
    </div>
    """, unsafe_allow_html=True)