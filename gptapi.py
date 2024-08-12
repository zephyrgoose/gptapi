import json
import yaml
import openai
import logging
import os
import jsonschema
from pathlib import Path
from pydantic import BaseModel, create_model, ValidationError

# Global constants and variables
CREDENTIALS_FILE = 'keys.yaml'
CREDENTIALS_KEY = 'openai_api'
PROFILES_DIR = 'profiles'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_PROP_TYPE = str
DEFAULT_MODEL_NAME = 'StructuredOutputModel'

def load_yaml(file_path):
    """Load a YAML file and return its contents as a dictionary."""
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        logging.error("Error loading YAML configuration: %s", e)
        raise SystemExit("Failed to load configuration. Exiting.")

def setup_logging(log_file, log_level):
    """Set up logging configuration, ensuring the log directory exists."""
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)  # This will create the directory only if it doesn't exist

    logging.basicConfig(filename=log_file,
                        level=log_level,
                        format=LOG_FORMAT)

def create_openai_client(credentials_file=CREDENTIALS_FILE):
    """Create an instance of the OpenAI client using credentials from a YAML file."""
    current_dir = Path(__file__).parent
    credentials_path = current_dir / credentials_file  # Directly pointing to the keys.yaml in the gptapi submodule
    credentials = load_yaml(credentials_path)
    return openai.OpenAI(api_key=credentials[CREDENTIALS_KEY])


def generate_pydantic_model(schema):
    """Dynamically generate a Pydantic model based on the provided JSON schema."""
    properties = {}
    for prop, details in schema['properties'].items():
        prop_type = DEFAULT_PROP_TYPE  # Default to string; you can extend this with more types if needed
        properties[prop] = (prop_type, ...)
    
    return create_model(DEFAULT_MODEL_NAME, **properties)

def gptapi(profile, prompt):
    """
    Make an API call to OpenAI using the given profile and prompt, adhering to structured output if supported.

    Parameters:
    - profile (str): The name of the YAML profile to use.
    - prompt (str): The prompt to send to the API.

    Returns:
    - dict or str: The API response content, either as a dict (structured) or a string.
    """
    try:
        current_dir = Path(__file__).parent
        profile_path = current_dir / PROFILES_DIR / f"{profile}.yaml"
        config = load_yaml(profile_path)

        if config['logging']['enable']:
            setup_logging(config['logging']['log_file'], config['logging']['log_level'])

        client = create_openai_client(config['credentials_file'])

        messages = [
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": prompt}
        ]

        # Prepare additional parameters
        parameters = {
            "model": config['model'],
            "messages": messages,
            "max_tokens": config['parameters']['max_tokens'],
            "temperature": config['parameters']['temperature'],
            "top_p": config['parameters']['top_p'],
            "n": config['parameters']['n'],
            "stop": config['parameters'].get('stop', None),
            "frequency_penalty": config['parameters'].get('frequency_penalty', 0.0),
            "presence_penalty": config['parameters'].get('presence_penalty', 0.0)
        }

        if config.get('structured_output', {}).get('enable'):
            function_name = "format_response"
            function_schema = config['structured_output']['schema']

            functions = [
                {
                    "name": function_name,
                    "description": "Formats the response according to the specified schema.",
                    "parameters": function_schema
                }
            ]

            parameters.update({
                "functions": functions,
                "function_call": {"name": function_name}
            })

            completion = client.chat.completions.create(**parameters)

            result = completion.choices[0].message.function_call.arguments
        else:
            completion = client.chat.completions.create(**parameters)

            result = completion.choices[0].message.content

        if not result:
            logging.error("Received empty response from API.")
            raise ValueError("Empty response from API.")

        logging.debug("API call successful: %s", result)

        return result

    except ValidationError as e:
        logging.error("ValidationError: %s", e)
        raise SystemExit("Validation error. Please check the structured output schema.")

    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise SystemExit("An unexpected error occurred. Please try again later.")

    except ValidationError as e:
        logging.error("ValidationError: %s", e)
        raise SystemExit("Validation error. Please check the structured output schema.")

    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise SystemExit("An unexpected error occurred. Please try again later.")

# Example Usage
#result = gptapi('goalplanner', "Formulate a plan for beginning a small enterprise technology architecture consulting firm.")
#print(result)
