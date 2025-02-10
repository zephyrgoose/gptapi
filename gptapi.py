import os
import json
import yaml
import logging
import openai
import importlib.util
import argparse

# Configure logging
logging.basicConfig(filename="debug.log", level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

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
    
    if not os.path.exists(profile_filename):
        raise FileNotFoundError(f"Profile '{profile_name}' not found at {profile_filename}")

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

def load_function(function_name):
    """Dynamically loads a function from the functions/ directory."""
    function_file = f"{function_name}.py"
    function_path = os.path.join(os.path.dirname(__file__), "functions", function_file)

    if not os.path.exists(function_path):
        logging.error(f"Function module {function_file} not found at path {function_path}.")
        return None

    try:
        spec = importlib.util.spec_from_file_location(function_name, function_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, function_name):
            logging.info(f"Successfully loaded function: {function_name} from {function_file}")
            return getattr(module, function_name)
        else:
            logging.error(f"Function {function_name} not found inside {function_file}. Available attributes: {dir(module)}")
            return None
    except Exception as e:
        logging.error(f"Error loading function {function_name}: {e}")
        return None

def gptapi(profile_name, prompt):
    """Main function to interact with the GPT API using the specified profile."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_dir = os.path.join(current_dir, "profiles")

    try:
        # Load profile and API key
        profile = load_profile(profile_name, profiles_dir)
        api_key = load_api_key(profile.get("credentials_file", "./keys.yaml"))

        client = openai.OpenAI(api_key=api_key)

        tools = profile.get("tools", [])
        tool_choice = profile.get("tool_choice", "auto")

        logging.info(f"Sending prompt to GPT API: {prompt}")

        response = client.chat.completions.create(
            model=profile["model"],
            messages=[
                {"role": "system", "content": profile["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            tools=tools,
            tool_choice=tool_choice,
            **profile["parameters"]
        )

        logging.debug(f"Raw OpenAI response: {response}")

        if not response.choices:
            logging.warning("GPT API returned no choices.")
            return None

        message = response.choices[0].message

        if message.tool_calls:
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                logging.debug(f"Tool call detected: {function_name} with arguments {function_args}")

                available_functions = [f.replace(".py", "") for f in os.listdir("functions") if f.endswith(".py")]
                logging.info(f"Available functions: {available_functions}")

                if function_name in available_functions:
                    function = load_function(function_name)
                    if function:
                        result = function(**function_args)
                        logging.debug(f"Function execution result: {result}")
                        return result
                    else:
                        logging.error(f"Function {function_name} was loaded but could not be executed.")
                        return f"Function {function_name} could not be executed."
                else:
                    logging.warning(f"Function {function_name} is not available.")
                    return f"Function {function_name} is not available."

        return message.content

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None

def main():
    """Command-line interface for GPT API."""
    parser = argparse.ArgumentParser(description="Run GPT API with a specified profile and prompt.")
    
    parser.add_argument(
        "--profile", "-p", required=True, type=str, help="Specify the profile name (e.g., 'worker', 'weather')."
    )
    parser.add_argument(
        "--prompt", "-q", required=True, type=str, help="Specify the user prompt."
    )

    args = parser.parse_args()
    
    result = gptapi(args.profile, args.prompt)
    print("Result:", result)

if __name__ == "__main__":
    main()
