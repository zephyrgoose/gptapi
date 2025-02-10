import requests
import logging
import os
import yaml

# Configure logging
logging.basicConfig(filename="../debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

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
    keys_filepath = os.path.join(current_dir, "../keys.yaml")  # Adjust path to root level
    keys = load_yaml(keys_filepath)
    if "openweather_api" not in keys:
        raise ValueError("Missing 'openweather_api' in keys file.")
    return keys["openweather_api"]

def get_current_weather(city):
    """
    Fetches the current weather for a specified city using the OpenWeatherMap API.

    This function retrieves weather details such as temperature, humidity, pressure, 
    wind speed, and a short description of the current weather conditions.

    Args:
        city (str): The name of the city for which to retrieve weather data.

    Returns:
        dict: A dictionary containing the following weather information:
            - "description" (str): A short description of the weather (e.g., "clear sky").
            - "temperature" (float): The current temperature in degrees Celsius.
            - "humidity" (int): The humidity percentage.
            - "pressure" (int): The atmospheric pressure in hPa.
            - "wind_speed" (float): The wind speed in metres per second.

    Raises:
        RuntimeError: If there is an error fetching weather data from the API.
        ValueError: If the API key is missing or invalid.
    """
    
    api_key = load_api_key()

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    logging.info(f"Calling OpenWeather API for city: {city}")

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        return {
            "description": data["weather"][0]["description"],
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": data["wind"]["speed"]
        }

    except requests.RequestException as e:
        logging.error(f"Error fetching weather data: {e}")
        raise RuntimeError(f"Error fetching weather data: {e}")
