import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

class WeatherService:
    """Service pour récupérer les données météo"""
    
    def __init__(self):
        # Option 1: OpenWeatherMap (gratuit jusqu'à 1000 calls/jour)
        self.openweather_key = os.getenv('OPENWEATHER_API_KEY')
        
        # Option 2: WeatherAPI (gratuit jusqu'à 1M calls/mois)
        self.weatherapi_key = os.getenv('WEATHERAPI_KEY')
        
        # Coordonnées par défaut (à personnaliser selon votre ville)
        self.default_lat = 36.8065  # Tunis
        self.default_lon = 10.1815
        
    def get_weather_data(self, date_str, hour, lat=None, lon=None):
        """
        Récupère les données météo pour une date et heure données
        
        Args:
            date_str: Date au format 'YYYY-MM-DD'
            hour: Heure (0-23)
            lat: Latitude (optionnel)
            lon: Longitude (optionnel)
            
        Returns:
            dict: {
                'temp': float,  # Température en °C
                'humidity': float,  # Humidité (0-1)
                'weather': int,  # 1=Clair, 2=Nuageux, 3=Pluie
                'description': str,  # Description
                'success': bool
            }
        """
        lat = lat or self.default_lat
        lon = lon or self.default_lon
        
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        now = datetime.now()
        
        # Vérifier si c'est une date future ou passée
        if target_date.date() > now.date():
            # Prévisions futures (jusqu'à 7 jours)
            return self._get_forecast_weather(target_date, hour, lat, lon)
        elif target_date.date() >= (now - timedelta(days=5)).date():
            # Données historiques récentes (jusqu'à 5 jours)
            return self._get_historical_weather(target_date, hour, lat, lon)
        else:
            # Date trop ancienne, utiliser des valeurs estimées
            return self._get_estimated_weather(target_date, hour)
    
    def _get_forecast_weather(self, target_date, hour, lat, lon):
        """Récupère les prévisions météo (OpenWeatherMap)"""
        try:
            if not self.openweather_key:
                return self._get_fallback_weather()
            
            # API OpenWeatherMap - Prévisions 5 jours
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.openweather_key,
                'units': 'metric',  # Celsius
                'lang': 'fr'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Trouver la prévision la plus proche de l'heure demandée
            target_datetime = target_date.replace(hour=hour)
            closest_forecast = None
            min_diff = float('inf')
            
            for forecast in data['list']:
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                diff = abs((forecast_time - target_datetime).total_seconds())
                
                if diff < min_diff:
                    min_diff = diff
                    closest_forecast = forecast
            
            if closest_forecast:
                return self._parse_openweather_response(closest_forecast)
            
            return self._get_fallback_weather()
            
        except Exception as e:
            print(f"Erreur API météo: {e}")
            return self._get_fallback_weather()
    
    def _get_historical_weather(self, target_date, hour, lat, lon):
        """Récupère les données historiques (WeatherAPI)"""
        try:
            if not self.weatherapi_key:
                return self._get_fallback_weather()
            
            # API WeatherAPI - Données historiques
            url = f"http://api.weatherapi.com/v1/history.json"
            params = {
                'key': self.weatherapi_key,
                'q': f"{lat},{lon}",
                'dt': target_date.strftime('%Y-%m-%d'),
                'lang': 'fr'
            }
            
            response = requests.get(url, params=params, timeout=5)
            response.raise_for_status()
            data = response.json()
            
            # Trouver l'heure la plus proche
            hours_data = data['forecast']['forecastday'][0]['hour']
            target_hour_data = None
            
            for hour_data in hours_data:
                hour_time = datetime.strptime(hour_data['time'], '%Y-%m-%d %H:%M')
                if hour_time.hour == hour:
                    target_hour_data = hour_data
                    break
            
            if target_hour_data:
                return self._parse_weatherapi_response(target_hour_data)
            
            return self._get_fallback_weather()
            
        except Exception as e:
            print(f"Erreur API météo historique: {e}")
            return self._get_fallback_weather()
    
    def _parse_openweather_response(self, data):
        """Parse la réponse OpenWeatherMap"""
        temp = data['main']['temp']
        humidity = data['main']['humidity'] / 100  # Convertir en 0-1
        
        # Mapper les codes météo OpenWeather vers nos codes
        weather_id = data['weather'][0]['id']
        if weather_id < 600:  # Pluie, orage
            weather_code = 3
        elif weather_id < 800:  # Nuages
            weather_code = 2
        else:  # Clair
            weather_code = 1
        
        return {
            'temp': round(temp, 1),
            'humidity': round(humidity, 2),
            'weather': weather_code,
            'description': data['weather'][0]['description'],
            'success': True
        }
    
    def _parse_weatherapi_response(self, data):
        """Parse la réponse WeatherAPI"""
        temp = data['temp_c']
        humidity = data['humidity'] / 100
        
        # Mapper les conditions vers nos codes
        condition = data['condition']['text'].lower()
        if 'rain' in condition or 'snow' in condition or 'storm' in condition:
            weather_code = 3
        elif 'cloud' in condition or 'overcast' in condition:
            weather_code = 2
        else:
            weather_code = 1
        
        return {
            'temp': round(temp, 1),
            'humidity': round(humidity, 2),
            'weather': weather_code,
            'description': data['condition']['text'],
            'success': True
        }
    
    def _get_estimated_weather(self, target_date, hour):
        """Estimation basée sur les moyennes saisonnières"""
        month = target_date.month
        
        # Moyennes pour Tunis (à adapter selon votre région)
        seasonal_temps = {
            1: 12, 2: 13, 3: 15, 4: 18, 5: 22,
            6: 26, 7: 29, 8: 29, 9: 27, 10: 23,
            11: 18, 12: 14
        }
        
        base_temp = seasonal_temps.get(month, 20)
        
        # Variation jour/nuit
        if 6 <= hour <= 18:
            temp = base_temp + 2
        else:
            temp = base_temp - 3
        
        # Humidité moyenne
        humidity = 0.65
        
        return {
            'temp': round(temp, 1),
            'humidity': round(humidity, 2),
            'weather': 1,  # Par défaut clair
            'description': 'Données estimées',
            'success': False  # Indique que ce sont des estimations
        }
    
    def _get_fallback_weather(self):
        """Valeurs par défaut en cas d'erreur"""
        return {
            'temp': 20.0,
            'humidity': 0.5,
            'weather': 1,
            'description': 'Conditions par défaut',
            'success': False
        }
