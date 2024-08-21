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
    keys_filepath = os.path.join(current_dir, keys_filename)  # Use the directory of gptapi.py
    
    keys = load_yaml(keys_filepath)
    
    if 'openai_api' not in keys:
        raise ValueError("Missing 'openai_api' in keys file.")
    
    return keys['openai_api']


def gptapi(profile_name, prompt):
    """Main function to interact with the GPT API."""
    
    # Define the directory where profile YAML files are stored
    current_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_dir = os.path.join(current_dir, 'profiles')  # Ensure this path points to the correct profiles directory
    
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
    
    # Print the response format before the API call
    #print(f"Response Format:\n{response_format_json}")
    
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
    
    # Basic error handling
    #if hasattr(response, 'usage'):
    #    print("Token Usage:", response.usage)
    if hasattr(response, 'errors'):
        print("Errors:", response.errors)
    
    # Handle the response
    content = response.choices[0].message.content
    if content:
        return content

# Example usage:
#result = gptapi('webvulnscraper', "Today, CISA—in partnership with the Federal Bureau of Investigation (FBI)—released an update to joint Cybersecurity Advisory #StopRansomware: Royal Ransomware, etc.")
#print(result)