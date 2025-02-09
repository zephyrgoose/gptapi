import yaml
from openai import OpenAI
import json
import os

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
    required_fields = ["model", "system_prompt", "parameters", "structured_output"]
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
    """Main function to interact with the GPT API using the new interface."""
    # Determine paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_dir = os.path.join(current_dir, "profiles")

    # Load profile and API key
    profile = load_profile(profile_name, profiles_dir)
    api_key = load_api_key(profile.get("credentials_file", "./keys.yaml"))

    # Instantiate the client
    client = OpenAI(api_key=api_key)

    # Prepare function schema for structured output
    structured_output = profile["structured_output"]
    function_schema = {
        "name": structured_output["name"],
        "description": "Structured JSON output schema",
        "parameters": structured_output["schema"]
    }

    try:
        response = client.chat.completions.create(
            model=profile["model"],
            messages=[
                {"role": "system", "content": profile["system_prompt"]},
                {"role": "user", "content": prompt}
            ],
            functions=[function_schema],
            function_call="auto",
            **profile["parameters"]
        )

        if not response.choices:
            return None

        message = response.choices[0].message

        # Extract structured output correctly
        if message.function_call:
            return json.loads(message.function_call.arguments)

        return message.content

    except Exception as e:
        return None

if __name__ == "__main__":
    try:
        result = gptapi(
            "cot",
            "Architect, plan, and design a program that facilitates the use of asymmetric encryption protocols."
        )
        print("RESULT:", result)
    except Exception as e:
        print(f"An error occurred: {e}")
