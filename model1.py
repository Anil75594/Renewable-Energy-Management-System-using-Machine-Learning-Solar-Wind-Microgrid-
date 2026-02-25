import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Initialize Google Earth Engine
ee.Authenticate()

class SimpleSolarPredictor:
    def __init__(self, latitude, longitude):
        self.lat = latitude
        self.lon = longitude
        self.model = None
        self.point = ee.Geometry.Point([longitude, latitude])
    
    # -------------------------
    # Historical Data (Earth Engine)
    # -------------------------
    def get_weather_data(self, start_date, end_date):
        """Get historical weather data from Earth Engine"""
        print(f"Getting data from {start_date} to {end_date}...")
        
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date, end_date) \
            .filterBounds(self.point)
        
        def extract_daily_data(image):
            date = image.date()
            solar_rad = image.select('surface_solar_radiation_downwards_sum').divide(3600000)
            temp = image.select('temperature_2m')
            return ee.Image.cat([solar_rad, temp]).rename(['solar_rad', 'temp']) \
                .set('date', date.format('YYYY-MM-dd'))
        
        daily_data = era5.map(extract_daily_data)
        
        def get_values(image):
            values = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=self.point,
                scale=10000
            )
            return ee.Feature(None, values.set('date', image.get('date')))
        data_list = daily_data.map(get_values).getInfo()
        data_rows = []
        
        for feature in data_list['features']:
            props = feature['properties']
            if 'solar_rad' in props and props['solar_rad'] is not None:
                data_rows.append(props)
        
        df = pd.DataFrame(data_rows)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.dropna()
        
        print(f"Got {len(df)} days of weather data")
        return df
    
    def prepare_data(self, df):
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['solar_potential'] = df['solar_rad'] * np.where(
            df['temp'] > 298, 1 - (df['temp'] - 298) * 0.004, 1
        )
        return df
    
    def train_model(self, df):
        print("Training model...")
        feature_cols = ['temp', 'day_of_year', 'month']
        X = df[feature_cols]
        y = df['solar_potential']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Model trained! Train score: {train_score:.3f}, Test score: {test_score:.3f}")
        return self.model
    
    def predict_tomorrow(self):
        """Predict solar energy for tomorrow"""
        if self.model is None:
            print("Please train the model first!")
            return None
        
        tomorrow = datetime.now() + timedelta(days=1)
        seasonal_avgs = {
            1: {'temp': 295}, 2: {'temp': 302},
            3: {'temp': 299}, 4: {'temp': 297}
        }
        month = tomorrow.month
        season = 1 if month in [12, 1, 2] else \
                 2 if month in [3, 4, 5] else \
                 3 if month in [6, 7, 8, 9] else 4
        avg_temp = seasonal_avgs[season]['temp']
        
        features = {
            'temp': avg_temp,
            'day_of_year': tomorrow.timetuple().tm_yday,
            'month': month
        }
        
        features_df = pd.DataFrame([features])
        prediction = self.model.predict(features_df)[0]
        
        if prediction > 5.0:
            suitability = "Excellent"
        elif prediction > 3.5:
            suitability = "Good"
        elif prediction > 2.0:
            suitability = "Fair"
        else:
            suitability = "Poor"
        
        print(f"\nTomorrow's Solar Prediction ({tomorrow.strftime('%Y-%m-%d')}):")
        print(f"Estimated solar potential: {prediction:.2f} kWh/m²")
        print(f"Suitability: {suitability}")
        print(f"Expected temperature: {avg_temp-273.15:.1f}°C")
        
        return prediction

# -------------------------
# Main execution
# -------------------------
if __name__ == "__main__":
    print("=== Simple Solar Energy Predictor ===")
    print("Location: Pune, Maharashtra, India")
    print("Coordinates: 18.5204°N, 73.8567°E\n")
    
    predictor = SimpleSolarPredictor(18.5204, 73.8567)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    try:
        historical_data = predictor.get_weather_data(
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        if len(historical_data) > 0:
            prepared_data = predictor.prepare_data(historical_data)
            predictor.train_model(prepared_data)
            predictor.predict_tomorrow()
        else:
            raise Exception("No historical data available")
    except Exception as e:
        print(f"Using fallback training data due to: {e}")
        seasons = [1, 2, 3, 4]
        sample_data = []
        seasonal_params = {
            1: {'temp': 295, 'solar_rad': 4.8},
            2: {'temp': 305, 'solar_rad': 6.2},
            3: {'temp': 301, 'solar_rad': 3.8},
            4: {'temp': 298, 'solar_rad': 5.1}
        }
        for season in seasons:
            days_in_season = 90 if season != 3 else 120
            for day in range(1, days_in_season + 1):
                params = seasonal_params[season]
                temp = params['temp'] + np.random.normal(0, 3)
                solar_rad = max(0, params['solar_rad'] + np.random.normal(0, 0.5))
                solar_potential = solar_rad * (1 - max(0, (temp - 298) * 0.004))
                sample_data.append({
                    'temp': temp, 'solar_rad': solar_rad,
                    'day_of_year': day + sum([90 if s != 3 else 120 for s in range(1, season)]),
                    'month': 1 if season == 1 else 4 if season == 2 else 7 if season == 3 else 10,
                    'solar_potential': solar_potential
                })
        sample_df = pd.DataFrame(sample_data)
        predictor.train_model(sample_df)
        predictor.predict_tomorrow()
    
    print("\nThank you for using the Solar Energy Predictor!")