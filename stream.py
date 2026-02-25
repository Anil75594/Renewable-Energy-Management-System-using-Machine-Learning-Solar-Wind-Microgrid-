import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import queue
import io
import math
import requests
from contextlib import redirect_stdout
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

try:
    from model1 import SimpleSolarPredictor
    from model2 import WindEnergyPredictor, main as wind_main
    from realtime import MicrogridController, House, WindTurbine, SolarPanel, BatterySystem
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Make sure model1.py, model2.py, and realtime.py are in the same directory")
    st.stop()

st.set_page_config(
    page_title="Energy Management Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric-container { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0; }
.success-metric { border-left: 5px solid #28a745; }
.warning-metric { border-left: 5px solid #ffc107; }
.danger-metric { border-left: 5px solid #dc3545; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'simulation_running' not in st.session_state: st.session_state.simulation_running = False
    if 'simulation_data' not in st.session_state: st.session_state.simulation_data = []
    if 'simulation_logs' not in st.session_state: st.session_state.simulation_logs = []
    if 'controller' not in st.session_state: st.session_state.controller = None
    if 'data_queue' not in st.session_state: st.session_state.data_queue = None
    if 'log_queue' not in st.session_state: st.session_state.log_queue = None
    if 'simulation_thread' not in st.session_state: st.session_state.simulation_thread = None
    if 'low_demand_mode' not in st.session_state: st.session_state.low_demand_mode = False

# ---------------------------------------------------------
# Forecasting Logic using OpenWeatherMap Data
# ---------------------------------------------------------
def get_3_day_forecast_from_api(houses, api_key, lat=18.5204, lon=73.8567, low_demand_mode=False):
    """Fetches real forecast data from OpenWeatherMap and projects energy supply, demand, and battery SOC"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Check if API returned an error message
        if 'cod' in data and data['cod'] != '200':
            print(f"OpenWeatherMap API error: {data.get('message', 'Unknown error')}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Failed to fetch forecast from OpenWeatherMap: {e}")
        return pd.DataFrame()

    weather_points = []
    for item in data['list']:
        weather_points.append({
            'Time': datetime.fromtimestamp(item['dt']),
            'temp': float(item['main']['temp']),
            'wind_speed': float(item['wind']['speed']),
            'cloud_cover': float(item['clouds']['all'] / 100.0)
        })
        
    df_weather = pd.DataFrame(weather_points)
    df_weather.set_index('Time', inplace=True)
    
    # Isolate only numeric columns, resample, and fill missing edges
    df_numeric = df_weather[['temp', 'wind_speed', 'cloud_cover']]
    df_hourly = df_numeric.resample('1H').mean().interpolate(method='linear').bfill().ffill()
    df_hourly.reset_index(inplace=True)
    
    # Ensure exactly 72 hours
    df_hourly = df_hourly.head(72)
    
    forecast_data = []
    wind_turbine = WindTurbine(capacity=8.0, hub_height=100.0) 
    solar_panel = SolarPanel(capacity=6.0)
    battery = BatterySystem(capacity=50.0) 
    
    for _, row in df_hourly.iterrows():
        current_time = row['Time']
        hour = current_time.hour
        
        temp = float(row['temp']) if pd.notna(row['temp']) else 25.0
        wind_speed = float(row['wind_speed']) if pd.notna(row['wind_speed']) else 5.0
        cloud_cover = float(row['cloud_cover']) if pd.notna(row['cloud_cover']) else 0.5
        
        # --- 1. Demand Calculation ---
        if 6 <= hour <= 8 or 18 <= hour <= 22:
            multiplier = 1.35
        elif 9 <= hour <= 17:
            multiplier = 0.95
        else:
            multiplier = 0.55
            
        # Apply Low Demand Mode settings matching realtime.py behavior
        if low_demand_mode:
            multiplier = min(multiplier, 0.6)
            demand_factor = 0.4
        else:
            demand_factor = 1.0
            
        temp_penalty = 1.15 if temp > 30.0 else 1.0
        total_demand = sum(h.base_demand * demand_factor * multiplier * temp_penalty for h in houses)
        
        # --- 2. Supply Calculation ---
        if 6 <= hour <= 18:
            sun_angle = math.sin((hour - 6) * math.pi / 12)
            clear_sky_ghi = 1000 * sun_angle
            cloud_factor = 1 - (cloud_cover * 0.75)
            ghi = max(0, clear_sky_ghi * cloud_factor)
        else:
            ghi = 0.0
            
        solar_power = solar_panel.calculate_power(ghi)
        wind_power = wind_turbine.calculate_power(wind_speed)
        total_supply = solar_power + wind_power
        
        # --- 3. Battery Calculation ---
        net_power = total_supply - total_demand
        if net_power > 0:
            # Note the is_hourly_forecast=True flag uses real physics instead of visual multiplier!
            battery.charge(net_power, is_hourly_forecast=True)
        else:
            deficit = abs(net_power)
            battery.discharge(deficit, is_hourly_forecast=True)
        
        forecast_data.append({
            "Time": current_time, 
            "Demand (kW)": total_demand,
            "Solar (kW)": solar_power,
            "Wind (kW)": wind_power,
            "Total Supply (kW)": total_supply,
            "Temp (°C)": temp,
            "Battery SOC (%)": battery.soc * 100
        })
        
    return pd.DataFrame(forecast_data)

# ---------------------------------------------------------
# Realtime Simulation Threading Component
# ---------------------------------------------------------
class SimulationWorker:
    def __init__(self, controller, data_queue, log_queue):
        self.controller = controller
        self.data_queue = data_queue
        self.log_queue = log_queue
        self.running = True
        self.output_buffer = io.StringIO()
    
    def stop(self):
        self.running = False
    
    def run(self):
        try:
            while self.running:
                with redirect_stdout(self.output_buffer):
                    data = self.controller.run_control_cycle()
                    self.controller.display_status(data)
                
                output = self.output_buffer.getvalue()
                self.output_buffer = io.StringIO()
                
                if output.strip():
                    lines = output.strip().split('\n')
                    for line in lines:
                        if line.strip():
                            self.log_queue.put(line)
                
                self.data_queue.put(data)
                time.sleep(2)
                
        except Exception as e:
            self.log_queue.put(f"ERROR: {str(e)}")
            self.data_queue.put({"error": str(e)})

# ---------------------------------------------------------
# UI Core Components
# ---------------------------------------------------------
def run_solar_model():
    try:
        st.subheader("Solar Energy Prediction")
        with st.spinner("Initializing Solar Predictor..."):
            predictor = SimpleSolarPredictor(18.5204, 73.8567)
            
        col1, col2 = st.columns(2)
        with col1: st.info("Location: Pune, Maharashtra, India (18.5204°N, 73.8567°E)")
        with col2: days_back = st.selectbox("Historical Data Period", [30, 60, 90, 180], index=3)
        
        if st.button("Run Solar Analysis", type="primary"):
            with st.spinner("Fetching weather data and training model..."):
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days_back)
                
                try:
                    historical_data = predictor.get_weather_data(
                        start_date.strftime('%Y-%m-%d'), 
                        end_date.strftime('%Y-%m-%d')
                    )
                    
                    if len(historical_data) > 0:
                        prepared_data = predictor.prepare_data(historical_data)
                        model = predictor.train_model(prepared_data)
                        prediction = predictor.predict_tomorrow()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("Data Points", len(prepared_data))
                        with col2: st.metric("Model Score", f"{model.score(prepared_data[['temp', 'day_of_year', 'month']], prepared_data['solar_potential']):.3f}")
                        with col3: st.metric("Tomorrow's Prediction", f"{prediction:.2f} kWh/m²")
                        with col4: 
                            avg_temp = prepared_data['temp'].mean() - 273.15
                            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=prepared_data['date'], y=prepared_data['solar_potential'], name='Solar Potential', line=dict(color='orange', width=2), yaxis='y'))
                        fig.add_trace(go.Scatter(x=prepared_data['date'], y=prepared_data['temp'] - 273.15, name='Temperature', line=dict(color='red', width=1), yaxis='y2'))
                        fig.update_layout(title="Historical Solar Potential and Temperature", yaxis=dict(title="Solar Potential (kWh/m²)", side="left"), yaxis2=dict(title="Temperature (°C)", side="right", overlaying="y"), template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else: raise Exception("No historical data available")
                        
                except Exception as e:
                    st.warning(f"Using fallback data: {e}")
                    
    except Exception as e:
        st.error(f"Error in solar model: {e}")

def run_wind_model():
    try:
        st.subheader("Wind Energy Prediction")
        with st.spinner("Initializing Wind Predictor..."):
            wind_predictor = WindEnergyPredictor(18.5204, 73.8567, turbine_height=100)
        
        col1, col2 = st.columns(2)
        with col1: st.info("Wind Turbine: 100m hub height, 2MW rated capacity")
        with col2: analysis_days = st.selectbox("Analysis Period", [365, 730], index=1)
        
        if st.button("Run Wind Analysis", type="primary"):
            with st.spinner("Analyzing wind patterns and training model..."):
                try:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=analysis_days)).strftime('%Y-%m-%d')
                    
                    historical_data = wind_predictor.get_historical_wind_data(start_date, end_date)
                    
                    if historical_data is not None and len(historical_data) > 0:
                        wind_data = wind_predictor.prepare_wind_data(historical_data)
                        model = wind_predictor.train_wind_model(wind_data)
                        wind_stats = wind_predictor.analyze_wind_statistics(wind_data)
                        prediction = wind_predictor.predict_wind_tomorrow()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("Avg Wind Speed", f"{wind_stats['mean']:.1f} m/s")
                        with col2: st.metric("Max Wind Speed", f"{wind_stats['max']:.1f} m/s")
                        with col3: st.metric("Avg Capacity Factor", f"{wind_data['capacity_factor'].mean():.1%}")
                        with col4: st.metric("Tomorrow's Power", f"{prediction:.0f} kW")
                        
                        # 1. Wind Speed Distribution
                        fig1 = px.histogram(wind_data, x='wind_speed_hub', nbins=30, title="Wind Speed Distribution at Hub Height")
                        st.plotly_chart(fig1, use_container_width=True)

                        # 2. Power Curve
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatter(x=wind_data['date'], y=wind_data['wind_power_kw'], mode='lines', name='Wind Power', line=dict(color='blue')))
                        fig2.update_layout(title="Wind Power Generation Over Time", xaxis_title="Date", yaxis_title="Power (kW)", template="plotly_white")
                        st.plotly_chart(fig2, use_container_width=True)

                        # 3. Seasonal Analysis
                        if 'season' in wind_data.columns:
                            seasonal_avg = wind_data.groupby('season')['wind_power_kw'].mean().reset_index()
                            season_names = {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}
                            seasonal_avg['season_name'] = seasonal_avg['season'].map(season_names)
                            
                            fig3 = px.bar(seasonal_avg, x='season_name', y='wind_power_kw', title="Average Wind Power by Season", labels={'wind_power_kw': 'Average Power (kW)', 'season_name': 'Season'})
                            st.plotly_chart(fig3, use_container_width=True)
                        
                    else: st.warning("No wind data available")
                        
                except Exception as e:
                    st.error(f"Wind analysis error: {e}")
                    
    except Exception as e:
        st.error(f"Error initializing wind model: {e}")

def run_realtime_simulation():
    st.subheader("Real-Time Microgrid Simulation")

    api_key = st.text_input("OpenWeatherMap API Key", value="a8f972578da0e255cc8973902bcb0c7a", type="password")

    st.markdown("---")
    col_toggle, col_explain = st.columns([1, 3])
    with col_toggle:
        low_demand_on = st.toggle("🔋 Low Demand Mode", value=st.session_state.low_demand_mode, disabled=st.session_state.simulation_running)
        st.session_state.low_demand_mode = low_demand_on
    with col_explain:
        if low_demand_on: st.success("**Low Demand Mode ON** — House demand is reduced. Battery will charge.")
        else: st.info("**Normal Mode** — All houses run at full demand.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1: start_clicked = st.button("Start Simulation", type="primary", disabled=st.session_state.simulation_running)
    with col2: stop_clicked = st.button("Stop Simulation", disabled=not st.session_state.simulation_running)
    with col3: clear_clicked = st.button("Clear Data")

    if start_clicked:
        try:
            st.session_state.controller = MicrogridController(api_key=api_key, low_demand_mode=st.session_state.low_demand_mode)
            st.session_state.simulation_running = True
            st.session_state.simulation_data = []
            st.session_state.simulation_logs = []
            st.session_state.data_queue = queue.Queue()
            st.session_state.log_queue = queue.Queue()

            worker = SimulationWorker(st.session_state.controller, st.session_state.data_queue, st.session_state.log_queue)
            st.session_state.simulation_worker = worker

            t = threading.Thread(target=worker.run, daemon=True)
            st.session_state.simulation_thread = t
            t.start()
            st.success("Simulation started!")
        except Exception as e:
            st.error(f"Failed to start: {e}")

    if stop_clicked:
        st.session_state.simulation_running = False
        if hasattr(st.session_state, 'simulation_worker') and st.session_state.simulation_worker:
            st.session_state.simulation_worker.stop()
        st.success("Simulation stopped!")

    if clear_clicked:
        st.session_state.simulation_data = []
        st.session_state.simulation_logs = []
        st.success("Cleared!")

    if st.session_state.simulation_running:
        if hasattr(st.session_state, 'data_queue') and st.session_state.data_queue:
            while True:
                try:
                    data = st.session_state.data_queue.get_nowait()
                    if "error" in data:
                        st.error(f"Simulation error: {data['error']}")
                        st.session_state.simulation_running = False
                        break
                    st.session_state.simulation_data.append(data)
                    if len(st.session_state.simulation_data) > 50: st.session_state.simulation_data.pop(0)
                except queue.Empty: break

        if hasattr(st.session_state, 'log_queue') and st.session_state.log_queue:
            while True:
                try:
                    line = st.session_state.log_queue.get_nowait()
                    st.session_state.simulation_logs.append(line)
                    if len(st.session_state.simulation_logs) > 200: st.session_state.simulation_logs.pop(0)
                except queue.Empty: break

    if st.session_state.simulation_data:
        latest = st.session_state.simulation_data[-1]
        w = latest['weather']
        gen = latest['generation']

        st.markdown(f"**System Status** — {latest['timestamp']} | 🌤️ {w['temperature']}°C | Wind: {w['wind_speed']} m/s | Solar GHI: {w['ghi']:.0f} W/m²")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Wind Power", f"{gen['wind']:.1f} kW")
        c2.metric("Solar Power", f"{gen['solar']:.1f} kW")

        batt_pct = latest['battery_soc'] * 100
        batt_delta = f"+{latest.get('battery_power', 0):.1f} kW charging" if latest.get('battery_charging') else f"-{latest.get('battery_power', 0):.1f} kW discharging"
        c3.metric("Battery SOC", f"{batt_pct:.1f}%", delta=batt_delta)

        c4.metric("Total Demand", f"{latest['total_demand']:.1f} kW")
        c5.metric("Available", f"{latest['available_power']:.1f} kW")

        if len(st.session_state.simulation_data) > 1:
            times = [d['timestamp'] for d in st.session_state.simulation_data]
            wind_vals = [d['generation']['wind'] for d in st.session_state.simulation_data]
            solar_vals = [d['generation']['solar'] for d in st.session_state.simulation_data]
            demand_vals = [d['total_demand'] for d in st.session_state.simulation_data]
            battery_vals = [d['battery_soc']*100 for d in st.session_state.simulation_data]

            # PLOT 1: Power Graph
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=times, y=wind_vals, name='Wind', line=dict(color='blue')))
            fig1.add_trace(go.Scatter(x=times, y=solar_vals, name='Solar', line=dict(color='orange')))
            fig1.add_trace(go.Scatter(x=times, y=demand_vals, name='Demand', line=dict(color='red', dash='dash')))
            fig1.update_layout(title="Power Generation vs Demand", yaxis_title="Power (kW)", template="plotly_white")
            
            # PLOT 2: RESTORED BATTERY GRAPH FOR REAL-TIME SIMULATION
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=times, y=battery_vals, name='Battery SOC', 
                mode='lines+markers', line=dict(color='green', width=2),
                fill='tozeroy', fillcolor='rgba(40,167,69,0.1)'
            ))
            fig2.add_hline(y=100, line_dash='dot', line_color='gray', annotation_text='Full')
            fig2.add_hline(y=20, line_dash='dot', line_color='red', annotation_text='Low')
            fig2.update_layout(title="Battery State of Charge (%)", yaxis_title="SOC (%)", yaxis_range=[0, 105], template="plotly_white")
            
            ch1, ch2 = st.columns(2)
            with ch1:
                st.plotly_chart(fig1, use_container_width=True)
            with ch2:
                st.plotly_chart(fig2, use_container_width=True)

            # --- NEW: House Status Table ---
        st.subheader("🏠 Individual House Status")
        
        house_data_list = []
        for house_id, active, demand, supplied, state in latest['houses']:
            # Determine emoji and status label
            if not active:
                status_emoji = "🔴 OFFLINE"
            elif state == "curtailed":
                status_emoji = "🟠 CURTAILED"
            else:
                status_emoji = "🟢 NORMAL"
            
            # Find the priority for this house
            priority_map = {1: "1 - Critical", 2: "2 - High", 3: "3 - Normal"}
            h_obj = next((h for h in st.session_state.controller.houses if h.id == house_id), None)
            priority_label = priority_map.get(h_obj.priority, "Unknown") if h_obj else "N/A"

            house_data_list.append({
                "House ID": f"House {house_id}",
                "Priority": priority_label,
                "Status": status_emoji,
                "Demand (kW)": f"{demand:.2f}",
                "Supplied (kW)": f"{supplied:.2f}",
                "Satisfaction": f"{(supplied/demand*100 if demand > 0 else 0):.0f}%"
            })

        # Display as a clean table
        st.table(pd.DataFrame(house_data_list))
        # -------------------------------

        if st.session_state.simulation_logs:
            st.subheader("Console Output")
            console_output = "\n".join(st.session_state.simulation_logs[-30:])
            st.text_area("Live Console", value=console_output, height=300, key="live_console")

    elif st.session_state.simulation_running:
        st.info("Waiting for first data cycle... (takes ~2 seconds)")

    if st.session_state.simulation_running:
        time.sleep(3)
        st.rerun()

def main():
    initialize_session_state()
    
    st.title("⚡ Energy Management Dashboard")
    st.sidebar.title("Navigation")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌞 Solar Model", "💨 Wind Model", "🏘️ Real-Time Simulation", "📈 3-Day Forecast"])
    
    with tab1: run_solar_model()
    with tab2: run_wind_model()
    with tab3: run_realtime_simulation()
        
    with tab4:
        st.subheader("📈 3-Day Forecast (Live OpenWeatherMap Data)")
        st.info("Projected generation uses real 5-day/3-hour forecast data interpolated to hourly increments.")
        
        col_api, col_toggle = st.columns([2, 1])
        with col_api:
            api_key_forecast = st.text_input(
                "OpenWeatherMap API Key (for Forecast)",
                value="a8f972578da0e255cc8973902bcb0c7a",
                type="password",
                key="forecast_api_key"
            )
        
        with col_toggle:
            # NEW FEATURE: Low Demand Mode toggle for the Forecast
            st.write("") # Spacer
            forecast_low_demand = st.toggle("🔋 Enable Low Demand Mode Forecast", value=False)
        
        if st.button("Generate Live 3-Day Forecast", type="primary"):
            with st.spinner("Fetching forecast data from OpenWeatherMap..."):
                houses = [House(1, 2.5, 1), House(2, 2.0, 2), House(3, 1.8, 3), House(4, 1.5, 3), House(5, 1.2, 3)]
                df_forecast = get_3_day_forecast_from_api(houses, api_key_forecast, low_demand_mode=forecast_low_demand)
                
                if not df_forecast.empty:
                    # 1. Plot Supply vs Demand
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=df_forecast["Time"], y=df_forecast["Total Supply (kW)"], mode='lines', name='Total Supply', line=dict(color='#28a745', width=3, shape='spline'), fill='tozeroy', fillcolor='rgba(40,167,69,0.1)'))
                    fig1.add_trace(go.Scatter(x=df_forecast["Time"], y=df_forecast["Demand (kW)"], mode='lines', name='Total Demand', line=dict(color='#dc3545', width=3, dash='dash', shape='spline')))
                    fig1.add_trace(go.Scatter(x=df_forecast["Time"], y=df_forecast["Solar (kW)"], mode='lines', name='Solar Gen', line=dict(color='#ffc107', width=1, shape='spline')))
                    fig1.add_trace(go.Scatter(x=df_forecast["Time"], y=df_forecast["Wind (kW)"], mode='lines', name='Wind Gen', line=dict(color='#007bff', width=1, shape='spline')))
                    fig1.update_layout(title="72-Hour Real Forecast: Supply vs. Demand", xaxis_title="Timeline", yaxis_title="Power (kW)", template="plotly_white", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    
                    # 2. Plot Battery SOC
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=df_forecast["Time"], y=df_forecast["Battery SOC (%)"],
                        mode='lines', name='Battery SOC',
                        line=dict(color='green', width=2, shape='spline'),
                        fill='tozeroy', fillcolor='rgba(40,167,69,0.1)'
                    ))
                    fig2.add_hline(y=100, line_dash='dot', line_color='gray', annotation_text='Full')
                    fig2.add_hline(y=20, line_dash='dot', line_color='red', annotation_text='Low')
                    fig2.update_layout(
                        title="Battery State of Charge (SOC) Forecast", 
                        xaxis_title="Timeline", yaxis_title="SOC (%)", 
                        yaxis_range=[0, 105], template="plotly_white", hovermode="x unified"
                    )

                    st.plotly_chart(fig1, use_container_width=True)
                    st.plotly_chart(fig2, use_container_width=True)
                    
                    avg_supply = df_forecast["Total Supply (kW)"].mean()
                    avg_demand = df_forecast["Demand (kW)"].mean()
                    avg_temp = df_forecast["Temp (°C)"].mean()
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Avg Forecast Supply", f"{avg_supply:.1f} kW")
                    c2.metric("Avg Forecast Demand", f"{avg_demand:.1f} kW")
                    c3.metric("Avg Forecast Temp", f"{avg_temp:.1f} °C")
                    
                    if avg_supply > avg_demand:
                        c4.metric("3-Day Outlook", "Surplus", delta=f"+{(avg_supply-avg_demand):.1f} kW avg")
                    else:
                        c4.metric("3-Day Outlook", "Deficit", delta=f"{(avg_supply-avg_demand):.1f} kW avg", delta_color="inverse")

if __name__ == "__main__":
    main()