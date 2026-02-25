import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import math
import matplotlib.pyplot as plt

# Initialize Google Earth Engine
ee.Authenticate()
ee.Initialize(project='clean-algebra-473715-b0')

class WindEnergyPredictor:
    def __init__(self, latitude, longitude, turbine_height=80):
        self.lat = latitude
        self.lon = longitude
        self.turbine_height = turbine_height
        self.model = None
        self.point = ee.Geometry.Point([longitude, latitude])
        self.region = self.point.buffer(5000)
        
        self.turbine_specs = {
            'rated_power': 2000,
            'cut_in_speed': 3.0,
            'rated_speed': 12.0,
            'cut_out_speed': 25.0,
            'hub_height': turbine_height
        }
    
    def get_historical_wind_data(self, start_date, end_date):
        print(f"Getting wind data from {start_date} to {end_date}...")
        
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date, end_date) \
            .filterBounds(self.point)
        
        def extract_daily_wind_data(image):
            date = image.date()
            u_wind_10m = image.select('u_component_of_wind_10m')
            v_wind_10m = image.select('v_component_of_wind_10m')
            wind_speed_10m = u_wind_10m.pow(2).add(v_wind_10m.pow(2)).sqrt()
            wind_direction = v_wind_10m.atan2(u_wind_10m).multiply(180).divide(math.pi).add(180)
            temperature = image.select('temperature_2m')
            pressure = image.select('surface_pressure')
            
            return ee.Image.cat([
                wind_speed_10m, wind_direction, temperature, pressure
            ]).rename(['wind_speed_10m', 'wind_direction', 'temperature', 'pressure']) \
                .set('date', date.format('YYYY-MM-dd'))
        
        daily_wind_data = era5.map(extract_daily_wind_data)
        
        def get_wind_values(image):
            values = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.region,
                scale=10000
            )
            return ee.Feature(None, values.set('date', image.get('date')))
        
        data_list = daily_wind_data.map(get_wind_values).getInfo()
        data_rows = []
        
        for feature in data_list['features']:
            props = feature['properties']
            if 'wind_speed_10m' in props and props['wind_speed_10m'] is not None:
                data_rows.append(props)
        
        df = pd.DataFrame(data_rows)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()
        
        print(f"Got {len(df)} days of wind data")
        return df
    
    def extrapolate_wind_speed(self, wind_speed_10m, target_height):
        alpha = 0.2
        wind_speed_hub = wind_speed_10m * (target_height / 10) ** alpha
        return wind_speed_hub
    
    def calculate_wind_power(self, wind_speed, air_density=1.225):
        if wind_speed < self.turbine_specs['cut_in_speed']:
            return 0
        elif wind_speed >= self.turbine_specs['cut_out_speed']:
            return 0
        elif wind_speed >= self.turbine_specs['rated_speed']:
            return self.turbine_specs['rated_power']
        else:
            power_ratio = (wind_speed / self.turbine_specs['rated_speed']) ** 3
            return self.turbine_specs['rated_power'] * power_ratio
    
    def calculate_air_density(self, temperature_k, pressure_pa):
        R = 287.05
        air_density = pressure_pa / (R * temperature_k)
        return air_density
    
    def prepare_wind_data(self, df):
        df['wind_speed_hub'] = df['wind_speed_10m'].apply(
            lambda x: self.extrapolate_wind_speed(x, self.turbine_specs['hub_height'])
        )
        
        df['air_density'] = df.apply(
            lambda row: self.calculate_air_density(row['temperature'], row['pressure']), 
            axis=1
        )
        
        df['wind_power_kw'] = df.apply(
            lambda row: self.calculate_wind_power(row['wind_speed_hub'], row['air_density']),
            axis=1
        )
        
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].apply(
            lambda x: 1 if x in [12, 1, 2] else
                     2 if x in [3, 4, 5] else
                     3 if x in [6, 7, 8, 9] else 4
        )
        
        df['capacity_factor'] = df['wind_power_kw'] / self.turbine_specs['rated_power']
        
        return df
    
    def train_wind_model(self, df):
        print("Training wind power model...")
        
        feature_cols = [
            'wind_speed_10m', 'temperature', 'pressure', 'air_density',
            'day_of_year', 'month', 'season'
        ]
        self.feature_names = feature_cols
        
        df_clean = df.dropna(subset=feature_cols + ['wind_power_kw'])
        
        if len(df_clean) == 0:
            print("No valid data for training!")
            return None
        
        X = df_clean[feature_cols]
        y = df_clean['wind_power_kw']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Wind Model trained!")
        print(f"Train R² score: {train_score:.3f}")
        print(f"Test R² score: {test_score:.3f}")
        
        return self.model
    
    def predict_wind_tomorrow(self):
        if self.model is None:
            print("Please train the model first!")
            return None
        
        tomorrow = datetime.now() + timedelta(days=1)
        month = tomorrow.month
        season = 1 if month in [12, 1, 2] else \
                 2 if month in [3, 4, 5] else \
                 3 if month in [6, 7, 8, 9] else 4
        
        seasonal_wind_data = {
            1: {'wind_speed_10m': 4.2, 'temperature': 295, 'pressure': 101325},
            2: {'wind_speed_10m': 5.8, 'temperature': 305, 'pressure': 100800},  
            3: {'wind_speed_10m': 7.2, 'temperature': 301, 'pressure': 100500},
            4: {'wind_speed_10m': 3.8, 'temperature': 298, 'pressure': 101200}
        }
        
        data = seasonal_wind_data[season]
        air_density = self.calculate_air_density(data['temperature'], data['pressure'])
        
        features = {
            'wind_speed_10m': data['wind_speed_10m'],
            'temperature': data['temperature'],
            'pressure': data['pressure'],
            'air_density': air_density,
            'day_of_year': tomorrow.timetuple().tm_yday,
            'month': month,
            'season': season
        }
        
        features_df = pd.DataFrame([features])
        predicted_power = self.model.predict(features_df)[0]
        
        wind_speed_hub = self.extrapolate_wind_speed(
            data['wind_speed_10m'], 
            self.turbine_specs['hub_height']
        )
        capacity_factor = predicted_power / self.turbine_specs['rated_power']
        
        if wind_speed_hub >= self.turbine_specs['rated_speed']:
            suitability = "Excellent"
        elif wind_speed_hub >= 8.0:
            suitability = "Very Good"
        elif wind_speed_hub >= 6.0:
            suitability = "Good"
        elif wind_speed_hub >= self.turbine_specs['cut_in_speed']:
            suitability = "Fair"
        else:
            suitability = "Poor"
        
        print(f"\nTomorrow's Wind Power Prediction ({tomorrow.strftime('%Y-%m-%d')}):")
        print(f"Expected wind speed (10m): {data['wind_speed_10m']:.1f} m/s")
        print(f"Expected wind speed (hub): {wind_speed_hub:.1f} m/s")
        print(f"Predicted power output: {predicted_power:.1f} kW")
        print(f"Capacity factor: {capacity_factor:.1%}")
        print(f"Wind suitability: {suitability}")
        
        return predicted_power
    
    def analyze_wind_statistics(self, df):
        if df is None or len(df) == 0:
            print("No data available for analysis")
            return
        
        print("\nWind Statistics Analysis")
        print("="*40)
        
        wind_stats = df['wind_speed_hub'].describe()
        print(f"Wind Speed Statistics:")
        print(f"  Mean:     {wind_stats['mean']:.2f} m/s")
        print(f"  Median:   {wind_stats['50%']:.2f} m/s")
        print(f"  Std Dev:  {wind_stats['std']:.2f} m/s")
        
        power_stats = df['wind_power_kw'].describe()
        avg_capacity_factor = df['capacity_factor'].mean()
        
        print(f"\nWind Power Statistics:")
        print(f"  Mean Power:           {power_stats['mean']:.0f} kW")
        print(f"  Max Power:            {power_stats['max']:.0f} kW")
        print(f"  Average Capacity:     {avg_capacity_factor:.1%}")
        
        return wind_stats
    
    def save_model(self, filename):
        if self.model is None:
            print("No model to save. Please train the model first.")
            return False
        
        try:
            model_data = {
                'model': self.model,
                'feature_names': self.feature_names,
                'turbine_specs': self.turbine_specs,
                'location': {'lat': self.lat, 'lon': self.lon}
            }
            joblib.dump(model_data, filename)
            print(f"Model saved successfully to {filename}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename):
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.turbine_specs = model_data['turbine_specs']
            print(f"Model loaded successfully from {filename}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# ================== PLOTTING FUNCTIONS ==================
def plot_results(wind_data, model, feature_names):
    # 1. Wind Speed over Time
    plt.figure(figsize=(10, 5))
    plt.plot(wind_data['date'], wind_data['wind_speed_hub'], label="Hub Height Wind Speed", color='blue')
    plt.xlabel("Date")
    plt.ylabel("Wind Speed (m/s)")
    plt.title("Wind Speed at Hub Height Over Time")
    plt.legend()
    plt.show()

    # 2. Predicted vs Actual Wind Power
    X = wind_data[feature_names]
    y_true = wind_data['wind_power_kw']
    y_pred = model.predict(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor='k')
    plt.plot([0, max(y_true)], [0, max(y_true)], 'r--', label="Ideal")
    plt.xlabel("Actual Wind Power (kW)")
    plt.ylabel("Predicted Wind Power (kW)")
    plt.title("Predicted vs Actual Wind Power")
    plt.legend()
    plt.show()

    # 3. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(feature_names)), importances[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.ylabel("Importance")
    plt.title("Feature Importance in Wind Power Prediction")
    plt.tight_layout()
    plt.show()

# ================== MAIN ==================
def main():
    print("Wind Energy Prediction System")
    print("="*30)
    pune_lat, pune_lon = 18.5204, 73.8567
    wind_predictor = WindEnergyPredictor(
        latitude=pune_lat,
        longitude=pune_lon,
        turbine_height=100
    )
    
    print(f"Analyzing wind potential for location: {pune_lat}°N, {pune_lon}°E")
    
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        
        print(f"Fetching historical wind data from {start_date} to {end_date}...")
        historical_data = wind_predictor.get_historical_wind_data(start_date, end_date)
        
        if historical_data is not None and len(historical_data) > 0:
            wind_data = wind_predictor.prepare_wind_data(historical_data)
            wind_predictor.train_wind_model(wind_data)
            wind_predictor.analyze_wind_statistics(wind_data)
            wind_predictor.predict_wind_tomorrow()

            # Call plotting function
            plot_results(wind_data, wind_predictor.model, wind_predictor.feature_names)

            wind_predictor.save_model('wind_energy_model.pkl')
        else:
            print("No historical data available.")
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()