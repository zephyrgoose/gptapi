import yaml
from openai import Client
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
    profile_filename = os.path.join(profiles_dir, f'{profile_name}.yaml')
    profile = load_yaml(profile_filename)

    # Define required fields
    required_fields = ['model', 'system_prompt', 'parameters', 'structured_output']
    validate_config(profile, required_fields)

    return profile


def load_api_key(keys_filename):
    """Loads the API key from a separate YAML file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    keys_filepath = os.path.join(current_dir, keys_filename)

    keys = load_yaml(keys_filepath)

    if 'openai_api' not in keys:
        raise ValueError("Missing 'openai_api' in keys file.")

    return keys['openai_api']


def gptapi(profile_name, prompt):
    """Main function to interact with the GPT API."""

    # Define the directory where profile YAML files are stored
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_dir = os.path.join(current_dir, 'profiles')

    profile = load_profile(profile_name, profiles_dir)
    api_key = load_api_key(profile.get('credentials_file', './keys.yaml'))

    client = Client(api_key=api_key)  # Initialize the OpenAI client

    # Prepare the response format with JSON-compliant formatting
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": profile['structured_output']['name'],
            "schema": {
                "type": "object",
                "properties": profile['structured_output']['schema']['properties'],
                "required": profile['structured_output']['schema'].get('required', []),
                "additionalProperties": profile['structured_output']['schema'].get('additionalProperties', False)
            },
            "strict": profile['structured_output'].get('strict', True)
        }
    }
    # Convert the schema to a JSON string with double quotation marks
    response_format_json = json.dumps(response_format, indent=4)

    # Prepare the API request
    response = client.chat.completions.create(
        model=profile['model'],
        messages=[
            {"role": "system", "content": profile['system_prompt']},
            {"role": "user", "content": prompt}
        ],
        response_format=json.loads(response_format_json),
        **profile['parameters']
    )

    # Handle the response
    content = response.choices[0].message.content
    if content:
        return content


if __name__ == "__main__":
    response = None  # Initialize response to handle undefined variable

    # Example usage
    try:
        result = gptapi(
            'goalplanner',
            "Architect, plan, and design a program that facilitates the use of asymmetric encryption protocols."
        )
        print(result)
        response = result  # Capture the response for further inspection
    except Exception as e:
        print(f"An error occurred: {e}")

    # Print token usage if it's available
    if response and hasattr(response, 'usage'):
        print("Token Usage:", response.usage)

    # Print errors if they exist
    if response and hasattr(response, 'errors'):
        print("Errors:", response.errors)