import os
import json
import yaml
import logging
import openai  # Ensure OpenAI is installed
from custom_functions import get_current_weather  # Import custom function

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


def validate_config(config, required_fields):
    """Validates that all required fields are present in the config."""
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required configuration field: {field}")


def load_profile(profile_name, profiles_dir):
    """Loads and validates the profile from a YAML configuration file."""
    profile_filename = os.path.join(profiles_dir, f"{profile_name}.yaml")
    profile = load_yaml(profile_filename)
    required_fields = ["model", "system_prompt", "parameters"]
    validate_config(profile, required_fields)
    return profile


def load_api_key(keys_filename="./keys.yaml"):
    """Loads the API key from a YAML file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    keys_filepath = os.path.join(current_dir, keys_filename)
    keys = load_yaml(keys_filepath)
    if "openai_api" not in keys:
        raise ValueError("Missing 'openai_api' in keys file.")
    return keys["openai_api"]


def gptapi(profile_name, prompt):
    """Main function to interact with the GPT API using the specified profile."""
    # Determine paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_dir = os.path.join(current_dir, "profiles")

    # Load profile and API key
    profile = load_profile(profile_name, profiles_dir)
    api_key = load_api_key(profile.get("credentials_file", "./keys.yaml"))

    # Instantiate OpenAI client
    client = openai.OpenAI(api_key=api_key)

    # Prepare function schemas from the profile, if any
    tools = profile.get("tools", [])

    logging.info(f"Sending prompt to GPT API: {prompt}")
    print(f"[DEBUG] Sending prompt to GPT API: {prompt}")  # Console proof

    try:
        response = client.chat.completions.create(
            model=profile["model"],
            messages=[
                {"role": "system", "content": profile["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            tools=tools,  # Updated from functions to tools
            tool_choice="auto",  # Updated from function_call to tool_choice
            **profile["parameters"]
        )

        if not response.choices:
            logging.warning("GPT API returned no choices.")
            return None

        message = response.choices[0].message

        # Log and print the raw response from OpenAI
        logging.debug(f"GPT API Response: {message}")
        print(f"[DEBUG] GPT API Response: {message}")  # Console proof

        # Handle function calls
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                logging.info(f"Tool call detected: {function_name} with arguments {function_args}")
                print(f"[DEBUG] Tool call detected: {function_name} with arguments {function_args}")  # Console proof

                # Execute the function if recognized
                if function_name == "get_current_weather":
                    city = function_args.get("city")
                    if not city:
                        raise ValueError("City parameter is missing in function arguments.")

                    logging.info(f"Executing function: {function_name} with city: {city}")
                    print(f"[DEBUG] Executing function: {function_name} with city: {city}")  # Console proof

                    weather_result = get_current_weather(city)

                    logging.info(f"Function execution result: {weather_result}")
                    print(f"[DEBUG] Function execution result: {weather_result}")  # Console proof

                    return weather_result

        return message.content

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"[ERROR] An error occurred: {e}")  # Console proof
        return None


if __name__ == "__main__":
    try:
        result = gptapi(
            "weather",  # Assuming you create a 'weather.yaml' profile
            "What's the current weather in Chadstone, Victoria, Australia?"
        )
        print("RESULT:", result)
    except Exception as e:
        print(f"An error occurred: {e}")
