import json
import yaml
import openai
import logging
import os
import jsonschema
from pathlib import Path
from pydantic import BaseModel, create_model, ValidationError

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
    if not log_dir.exists():
        os.makedirs(log_dir)

    logging.basicConfig(filename=log_file,
                        level=log_level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

def create_openai_client(credentials_file):
    """Create an instance of the OpenAI client using credentials from a YAML file."""
    credentials = load_yaml(credentials_file)
    return openai.OpenAI(api_key=credentials['openai_api'])

def generate_pydantic_model(schema):
    """Dynamically generate a Pydantic model based on the provided JSON schema."""
    properties = {}
    for prop, details in schema['properties'].items():
        prop_type = str  # Default to string; you can extend this with more types if needed
        properties[prop] = (prop_type, ...)
    
    return create_model('StructuredOutputModel', **properties)

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
        # Load the profile configuration
        profile_path = Path(f"./profiles/{profile}.yaml")
        config = load_yaml(profile_path)

        # Setup logging
        if config['logging']['enable']:
            setup_logging(config['logging']['log_file'], config['logging']['log_level'])

        # Create OpenAI client
        client = create_openai_client(config['credentials_file'])

        # Prepare the messages array
        messages = [
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": prompt}
        ]

        # Check if structured output is enabled and set up function calling
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

            completion = client.chat.completions.create(
                model=config['model'],
                messages=messages,
                max_tokens=config['parameters']['max_tokens'],
                temperature=config['parameters']['temperature'],
                top_p=config['parameters']['top_p'],
                n=config['parameters']['n'],
                stop=config['parameters'].get('stop', None),
                functions=functions,
                function_call={"name": function_name}
            )

            # Assuming the model returns the structured output directly
            result = completion.choices[0].message.function_call.arguments
        else:
            completion = client.chat.completions.create(
                model=config['model'],
                messages=messages,
                max_tokens=config['parameters']['max_tokens'],
                temperature=config['parameters']['temperature'],
                top_p=config['parameters']['top_p'],
                n=config['parameters']['n'],
                stop=config['parameters'].get('stop', None)
            )

            result = completion.choices[0].message.content

        if not result:
            logging.error("Received empty response from API.")
            raise ValueError("Empty response from API.")

        # Log the successful API call
        logging.info("API call successful: %s", result)

        return result

    except ValidationError as e:
        logging.error("ValidationError: %s", e)
        raise SystemExit("Validation error. Please check the structured output schema.")

    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
        raise SystemExit("An unexpected error occurred. Please try again later.")

# Example Usage
#result = gptapi('goalplanner', "Formulate a plan for beginning a small enterprise technology architecture consulting firm.")
#print(result)
