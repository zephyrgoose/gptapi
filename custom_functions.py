import requests
import logging
import os
import yaml

# Configure logging
logging.basicConfig(filename="api_debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

def load_yaml(file_path):
    """Loads and parses a YAML file."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {file_path} - {exc}")

def load_api_key():
    """Loads the API keys from a YAML file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    keys_filepath = os.path.join(current_dir, "keys.yaml")
    keys = load_yaml(keys_filepath)
    if "openweather_api" not in keys:
        raise ValueError("Missing 'openweather_api' in keys file.")
    return keys["openweather_api"]

def get_current_weather(city):
    """Fetches the current weather for a specified city using the OpenWeatherMap API."""
    api_key = load_api_key()  # Correctly loads OpenWeather API key from keys.yaml

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,  # Now correctly using OpenWeather API key
        "units": "metric"
    }

    logging.info(f"Calling OpenWeather API for city: {city}")
    print(f"[DEBUG] Calling OpenWeather API for city: {city}")

    try:
        response = requests.get(base_url, params=params)
        logging.debug(f"OpenWeather API request URL: {response.url}")
        print(f"[DEBUG] OpenWeather API request URL: {response.url}")

        response.raise_for_status()
        data = response.json()

        logging.info(f"Received response: {data}")
        print(f"[DEBUG] OpenWeather API Response: {data}")

        return {
            "description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"]
        }

    except requests.RequestException as e:
        logging.error(f"Error fetching weather data: {e}")
        print(f"[ERROR] Error fetching weather data: {e}")
        raise RuntimeError(f"Error fetching weather data: {e}")
