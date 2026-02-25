import time
import random
import math
import datetime
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
import requests

class IrregularityType(Enum):
    DEMAND_SPIKE = "demand_spike"
    POWER_FAILURE = "power_failure"

class PowerState(Enum):
    NORMAL = "normal"
    CURTAILED = "curtailed"
    FAILED = "failed"

@dataclass
class House:
    id: int
    base_demand: float
    priority: int  # 1=critical, 2=high, 3=normal
    is_active: bool = True
    current_demand: float = 0.0
    power_supplied: float = 0.0
    power_state: PowerState = PowerState.NORMAL
    failure_probability: float = 0.05
    demand_spike_probability: float = 0.1

class WeatherAPI:
    """Fetches real-time weather data from OpenWeatherMap API"""
    
    def __init__(self, api_key: str, lat: float = 18.5246, lon: float = 73.8786):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
        self.last_fetch_time = 0
        self.cached_data = None
        self.cache_duration = 300  # 5 minutes cache to avoid API rate limits
        
    def fetch_weather_data(self):
        current_time = time.time()
        
        if (self.cached_data and 
            current_time - self.last_fetch_time < self.cache_duration):
            return self.cached_data
            
        try:
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data = {
                'temperature': data['main']['temp'],
                'wind_speed': data['wind']['speed'],
                'cloud_cover': data['clouds']['all'] / 100.0,
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'visibility': data.get('visibility', 10000) / 1000,
                'description': data['weather'][0]['description'],
                'hour': datetime.datetime.now().hour
            }
            
            weather_data['ghi'] = self.calculate_solar_irradiance(
                weather_data['hour'], 
                weather_data['cloud_cover']
            )
            
            self.cached_data = weather_data
            self.last_fetch_time = current_time
            
            return weather_data
            
        except requests.RequestException as e:
            print(f"❌ API Error: {e}")
            return self.get_fallback_weather()
        except Exception as e:
            print(f"❌ Weather API Error: {e}")
            return self.get_fallback_weather()
            
    def calculate_solar_irradiance(self, hour: int, cloud_cover: float) -> float:
        if hour < 6 or hour > 18:
            return 0.0
        sun_angle = math.sin((hour - 6) * math.pi / 12)
        clear_sky_ghi = 1000 * sun_angle
        cloud_factor = 1 - (cloud_cover * 0.75)
        ghi = clear_sky_ghi * cloud_factor
        return max(0, ghi)
        
    def get_fallback_weather(self):
        current_hour = datetime.datetime.now().hour
        print("⚠️  Using fallback weather data (API unavailable)")
        
        return {
            'temperature': 25.0,
            'wind_speed': 5.0,
            'ghi': 500 if 6 <= current_hour <= 18 else 0,
            'cloud_cover': 0.3,
            'humidity': 60,
            'pressure': 1013,
            'visibility': 10,
            'description': 'clear sky (fallback)',
            'hour': current_hour
        }
        
    def get_current_weather(self):
        return self.fetch_weather_data()

class EnergySource:
    def __init__(self, capacity: float):
        self.capacity = capacity
        self.current_output = 0.0

class WindTurbine(EnergySource):
    def __init__(self, capacity: float, hub_height: float = 100.0):
        super().__init__(capacity)
        self.hub_height = hub_height
        self.reference_height = 10.0  
        self.alpha = 0.143            

    def calculate_power(self, wind_speed_10m: float) -> float:
        hub_wind_speed = wind_speed_10m * ((self.hub_height / self.reference_height) ** self.alpha)
        if hub_wind_speed < 3.0 or hub_wind_speed > 25.0:
            return 0.0
        return min(self.capacity * (hub_wind_speed / 12) ** 2, self.capacity)

class SolarPanel(EnergySource):
    def calculate_power(self, ghi: float) -> float:
        if ghi <= 0:
            return 0.0
        return min((ghi / 1000) * self.capacity, self.capacity)

class BatterySystem:
    def __init__(self, capacity: float = 50.0):
        self.capacity = capacity
        self.soc = 0.6 
        
    def charge(self, power: float, is_hourly_forecast: bool = False) -> float:
        # Standard charging logic
        multiplier = 1.0 if is_hourly_forecast else 0.05
        added_energy = (min(power, self.capacity * 0.5) / self.capacity) * multiplier
        self.soc = min(1.0, self.soc + added_energy)
        return min(power, self.capacity * 0.5)
        
    def discharge(self, power: float, is_hourly_forecast: bool = False) -> float:
        if self.soc <= 0.05: return 0.0 # 5% Hard Floor safety
        
        # --- THE FIX FOR 3-DAY SURVIVAL ---
        if is_hourly_forecast:
            # Limit hourly discharge to only 3% of capacity per hour
            # This forces the forecast to "ration" energy over 72 hours
            max_discharge = self.capacity * 0.03 
            multiplier = 1.0
        else:
            # Real-time simulation stays aggressive
            max_discharge = self.capacity * 0.5
            multiplier = 0.05

        actual_withdrawal = min(power, max_discharge)
        removed_energy = (actual_withdrawal / self.capacity) * multiplier
        
        # Ensure we don't drop below the floor
        if self.soc - removed_energy < 0.05:
            actual_withdrawal = (self.soc - 0.05) * self.capacity / multiplier
            self.soc = 0.05
        else:
            self.soc -= removed_energy
            
        return actual_withdrawal

class IrregularityManager:
    def __init__(self):
        self.active_irregularities = []
        self.last_irregularity_time = time.time()
        
    def generate_irregularities(self, houses: List[House]) -> List[Dict]:
        irregularities = []
        current_time = time.time()
        
        if current_time - self.last_irregularity_time >= 30:
            self.last_irregularity_time = current_time
            
            for house in houses:
                if random.random() < house.demand_spike_probability:
                    spike_factor = random.uniform(1.5, 3.0)
                    irregularities.append({
                        'type': IrregularityType.DEMAND_SPIKE,
                        'house_id': house.id,
                        'severity': spike_factor,
                        'duration': random.randint(3, 5),
                        'description': f"House {house.id} demand spike ({spike_factor:.1f}x)"
                    })
                    
            for house in houses:
                if random.random() < house.failure_probability:
                    irregularities.append({
                        'type': IrregularityType.POWER_FAILURE,
                        'house_id': house.id,
                        'duration': random.randint(3, 5),
                        'description': f"House {house.id} power failure"
                    })
            
        return irregularities

class MicrogridController:
    def __init__(self, api_key: str = None, low_demand_mode: bool = False):
        if api_key:
            self.weather_api = WeatherAPI(api_key)
            print("✅ Using OpenWeatherMap API for real-time weather data")
        else:
            print("❌ No API key provided. Using simulated weather data.")
            self.weather_api = None

        self.low_demand_mode = low_demand_mode
        demand_factor = 0.4 if low_demand_mode else 1.0

        if low_demand_mode:
            print("🔋 LOW DEMAND MODE ACTIVE — demand reduced to show battery charging")
            
        self.wind_turbine = WindTurbine(capacity=8.0, hub_height=100.0)
        self.solar_panel = SolarPanel(capacity=6.0)
        self.battery = BatterySystem(capacity=50.0)
        self.irregularity_manager = IrregularityManager()
        
        self.houses = [
            House(1, 2.5 * demand_factor, 1),
            House(2, 2.0 * demand_factor, 2),
            House(3, 1.8 * demand_factor, 3),
            House(4, 1.5 * demand_factor, 3),
            House(5, 1.2 * demand_factor, 3),
        ]
        
        self.active_irregularities = []
        self.running = False
        
    def get_simulated_weather(self):
        current_hour = datetime.datetime.now().hour
        
        if 6 <= current_hour <= 18:
            wind_speed = random.uniform(3, 15)
            ghi = 800 * abs(math.sin((current_hour - 6) * math.pi / 12))
            cloud_cover = random.uniform(0.1, 0.7)
        else:
            wind_speed = random.uniform(2, 10)
            ghi = 0
            cloud_cover = random.uniform(0.3, 0.9)
            
        return {
            'temperature': random.uniform(20, 30),
            'wind_speed': wind_speed,
            'ghi': ghi,
            'cloud_cover': cloud_cover,
            'hour': current_hour,
            'description': 'simulated weather'
        }
        
    def simulate_demand(self, house: House):
        base_demand = house.base_demand
        spike_multiplier = 1.0
        
        for irregularity in self.active_irregularities:
            if (irregularity['type'] == IrregularityType.DEMAND_SPIKE and 
                irregularity.get('house_id') == house.id):
                spike_multiplier = irregularity['severity']
                
        hour = datetime.datetime.now().hour
        if 6 <= hour <= 8 or 18 <= hour <= 22:
            time_multiplier = random.uniform(1.2, 1.5)
        elif 9 <= hour <= 17:
            time_multiplier = random.uniform(0.8, 1.1)
        else:
            time_multiplier = random.uniform(0.4, 0.7)

        if self.low_demand_mode:
            time_multiplier = min(time_multiplier, 0.6)
            
        house.current_demand = base_demand * time_multiplier * spike_multiplier
        
    def manage_power_failures(self):
        for house in self.houses:
            for irregularity in self.active_irregularities:
                if (irregularity['type'] == IrregularityType.POWER_FAILURE and
                    irregularity.get('house_id') == house.id):
                    house.is_active = False
                    house.power_state = PowerState.FAILED
                    
    def update_irregularities(self):
        self.active_irregularities = [
            {**irr, 'duration': irr['duration'] - 1}
            for irr in self.active_irregularities
            if irr['duration'] > 1
        ]
        
        for irr in self.active_irregularities[:]:
            if irr['duration'] <= 0:
                print(f"✅ Resolved: {irr['description']}")
                self.active_irregularities.remove(irr)
                
                if irr['type'] == IrregularityType.POWER_FAILURE:
                    house_id = irr.get('house_id')
                    house = next((h for h in self.houses if h.id == house_id), None)
                    if house:
                        house.is_active = True
                        house.power_state = PowerState.NORMAL
        
        new_irregularities = self.irregularity_manager.generate_irregularities(self.houses)
        for irr in new_irregularities:
            self.active_irregularities.append(irr)
            print(f"🚨 {irr['description']}")
            
    def run_control_cycle(self):
        self.update_irregularities()
        self.manage_power_failures()
        
        if self.weather_api:
            weather = self.weather_api.get_current_weather()
        else:
            weather = self.get_simulated_weather()
        
        total_demand = 0
        for house in self.houses:
            if house.is_active:
                self.simulate_demand(house)
                total_demand += house.current_demand
                
        wind_power = self.wind_turbine.calculate_power(weather['wind_speed'])
        solar_power = self.solar_panel.calculate_power(weather['ghi'])
        total_renewable = wind_power + solar_power
        
        net_power = total_renewable - total_demand
        
        battery_charging = False
        battery_power = 0.0
        
        if net_power > 0:
            battery_power = self.battery.charge(net_power)
            available_power = total_demand
            battery_charging = True
            if self.low_demand_mode:
                print(f"🔋 Battery charging: +{battery_power:.2f}kW surplus | SOC: {self.battery.soc*100:.1f}%")
        else:
            deficit = abs(net_power)
            battery_power = self.battery.discharge(deficit)
            available_power = total_renewable + battery_power
            
        power_deficit = max(0, total_demand - available_power)
        if power_deficit > 0:
            print(f"⚠️  Power deficit: {power_deficit:.2f}kW")
            
        remaining_power = available_power
        for house in sorted(self.houses, key=lambda h: h.priority):
            if not house.is_active:
                house.power_supplied = 0
                continue
                
            if remaining_power >= house.current_demand:
                house.power_supplied = house.current_demand
                remaining_power -= house.current_demand
                house.power_state = PowerState.NORMAL
            else:
                house.power_supplied = remaining_power
                house.power_state = PowerState.CURTAILED
                remaining_power = 0
                
        return {
            'timestamp': datetime.datetime.now().strftime('%H:%M:%S'),
            'weather': weather,
            'generation': {
                'wind': wind_power,
                'solar': solar_power,
                'total': total_renewable
            },
            'battery_soc': self.battery.soc,
            'battery_charging': battery_charging,
            'battery_power': battery_power,
            'low_demand_mode': self.low_demand_mode,
            'houses': [(h.id, h.is_active, h.current_demand, h.power_supplied, h.power_state.value) 
                      for h in self.houses],
            'irregularities': [irr['description'] for irr in self.active_irregularities],
            'total_demand': total_demand,
            'available_power': available_power
        }
        
    def display_status(self, data):
        print("\n" + "="*60)
        mode_label = " [LOW DEMAND MODE 🔋]" if data.get('low_demand_mode') else ""
        print(f"Microgrid Status [{data['timestamp']}]{mode_label}")
        print("="*60)
        
        print(f"🌤️  Wind (10m): {data['weather']['wind_speed']:.1f}m/s | Solar: {data['weather']['ghi']:.0f}W/m²")
        print(f"⚡ Generation: Wind {data['generation']['wind']:.2f}kW | Solar {data['generation']['solar']:.2f}kW | Total {data['generation']['total']:.2f}kW")

        charging_arrow = "⬆️  CHARGING" if data.get('battery_charging') else "⬇️  DISCHARGING"
        print(f"🔋 Battery: {data['battery_soc']*100:.1f}% {charging_arrow} ({data.get('battery_power', 0):.2f}kW)")
        
        if data['irregularities']:
            print(f"\n🚨 Active Irregularities:")
            for irr in data['irregularities']:
                print(f"   ⚠️  {irr}")
        else:
            print(f"\n✅ No active irregularities")
            
        print(f"\n🏠 House Status:")
        for house_id, active, demand, supplied, state in data['houses']:
            status = "🟢" if active and state == "normal" else "🟠" if state == "curtailed" else "🔴"
            priority_text = ["", "CRITICAL", "HIGH", "NORMAL"][next(h.priority for h in self.houses if h.id == house_id)]
            
            if active:
                power_info = f"Demand: {demand:.2f}kW | Supplied: {supplied:.2f}kW"
                if demand > 0:
                    power_info += f" ({supplied/demand*100:.0f}%)"
            else:
                power_info = "OFFLINE"
                    
            print(f"   {status} House {house_id} ({priority_text}): {power_info}")
        
        power_balance = data['available_power'] - data['total_demand']
        balance_status = "✅ SURPLUS" if power_balance > 0.5 else "⚠️  DEFICIT" if power_balance < -0.1 else "⚖️  BALANCED"
        print(f"\n{balance_status} Power Balance: {power_balance:+.2f}kW")
        print(f"📊 Total Demand: {data['total_demand']:.2f}kW | Available: {data['available_power']:.2f}kW")
        
    def start(self):
        print("Starting Microgrid Controller with Irregularity Management")
        print("Press Ctrl+C to stop\n")
        
        self.running = True
        try:
            while self.running:
                data = self.run_control_cycle()
                self.display_status(data)
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\nShutting down microgrid controller...")
            self.running = False

if __name__ == "__main__":
    API_KEY = ""
    
    print("Microgrid Controller - Irregularity Management Demo")
    print("Starting in 3 seconds...")
    time.sleep(3)
    
    try:
        controller = MicrogridController(api_key=API_KEY, low_demand_mode=False)
        controller.start()
    except Exception as e:
        print(f"Error: {e}")