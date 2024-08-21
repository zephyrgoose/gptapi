import json
import yaml
import openai
from openai import BadRequestError
import logging
import tiktoken
from pathlib import Path
from pydantic import create_model, ValidationError

# Global Constants
CREDENTIALS_FILE = 'keys.yaml'
CREDENTIALS_KEY = 'openai_api'
PROFILES_DIR = 'profiles'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
DEFAULT_PROP_TYPE = str
DEFAULT_MODEL_NAME = 'StructuredOutputModel'
MAX_INPUT_TOKENS = 120000  # Global variable for maximum input tokens
CUTS = 3  # Maximum number of cuts to handle large inputs

class ConfigurationManager:
    """Handles loading and caching of configuration files."""
    _cache = {}

    @classmethod
    def load_yaml(cls, file_path):
        if file_path not in cls._cache:
            cls._cache[file_path] = cls.load_file(file_path)
        return cls._cache[file_path]

    @staticmethod
    def load_file(file_path):
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except yaml.YAMLError as e:
            Logger.error(f"Error loading YAML configuration: {e}")
            raise SystemExit("Failed to load configuration. Exiting.")

class Logger:
    """Singleton class for consistent logging setup."""
    _configured = False

    @staticmethod
    def setup_logging(log_file, log_level):
        if not Logger._configured:
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            logging.basicConfig(filename=log_file, level=log_level, format=LOG_FORMAT)
            Logger._configured = True

    @staticmethod
    def error(message):
        logging.error(message)
    
    @staticmethod
    def debug(message):
        logging.debug(message)

class OpenAIClientManager:
    """Manages the creation and use of the OpenAI client."""
    def __init__(self, credentials_file=CREDENTIALS_FILE):
        current_dir = Path(__file__).parent
        credentials_path = current_dir / credentials_file
        credentials = ConfigurationManager.load_yaml(credentials_path)
        self.client = openai.OpenAI(api_key=credentials[CREDENTIALS_KEY])

    def create_completion(self, parameters):
        return self.client.chat.completions.create(**parameters)

class ModelFactory:
    """Utility for creating Pydantic models."""
    @staticmethod
    def generate_pydantic_model(schema):
        properties = {}
        for prop, details in schema['properties'].items():
            prop_type = DEFAULT_PROP_TYPE  # Extend this if needed
            properties[prop] = (prop_type, ...)
        return create_model(DEFAULT_MODEL_NAME, **properties)

class CustomException(Exception):
    """Custom exception for handling specific errors."""
    pass

class APICallPreparer:
    """Prepares API call parameters based on configuration and user input."""
    def __init__(self, config, prompt, cut_attempt=0):
        self.config = config
        self.prompt = prompt
        self.cut_attempt = cut_attempt

    def prepare_parameters(self):
        messages = [
            {"role": "system", "content": self.config['system_prompt']},
            {"role": "user", "content": self.prompt}
        ]

        # Initialize tiktoken and count tokens
        encoding = tiktoken.encoding_for_model(self.config['model'])
        total_tokens = sum([len(encoding.encode(m['content'])) for m in messages])

        # Check and log total tokens
        Logger.debug(f"Total tokens in prepared messages: {total_tokens}")
        

        # This is a little fucked, would like to get rid of this later
        if total_tokens > MAX_INPUT_TOKENS:
            if self.cut_attempt >= CUTS:
                Logger.error("Maximum number of cuts reached. Aborting.")
                raise CustomException("Input too large, and maximum cuts exceeded.")
            Logger.debug(f"Input token limit exceeded: {total_tokens} tokens. Attempting to cut input into smaller parts.")

            # Cut the input in half with a 5% overlap
            midpoint = len(self.prompt) // 2
            overlap = int(len(self.prompt) * 0.05)
            start_cut = max(midpoint - overlap, 0)
            end_cut = min(midpoint + overlap, len(self.prompt))

            # Two halves of the input with overlap
            first_half = self.prompt[:end_cut]
            second_half = self.prompt[start_cut:]

            # Prepare parameters for each half
            Logger.debug(f"First cut: '{first_half[:50]}...'")
            Logger.debug(f"Second cut: '{second_half[:50]}...'")

            first_half_preparer = APICallPreparer(self.config, first_half, self.cut_attempt + 1)
            second_half_preparer = APICallPreparer(self.config, second_half, self.cut_attempt + 1)

            first_parameters = first_half_preparer.prepare_parameters()
            second_parameters = second_half_preparer.prepare_parameters()

            # Merge or handle these separately as per your application logic
            # For now, we'll just assume the first one is used
            return first_parameters  # You may want to handle merging results

        parameters = {
            "model": self.config['model'],
            "messages": messages,
            "max_tokens": self.config['parameters']['max_tokens'],  # For output token limit
            "temperature": self.config['parameters']['temperature'],
            "top_p": self.config['parameters']['top_p'],
            "n": self.config['parameters']['n'],
            "stop": self.config['parameters'].get('stop', None),
            "frequency_penalty": self.config['parameters'].get('frequency_penalty', 0.0),
            "presence_penalty": self.config['parameters'].get('presence_penalty', 0.0)
        }

        if self.config.get('structured_output', {}).get('enable'):
            function_name = "format_response"
            function_schema = self.config['structured_output']['schema']

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

        return parameters

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
        config = ConfigurationManager.load_yaml(profile_path)

        if config['logging']['enable']:
            Logger.setup_logging(config['logging']['log_file'], config['logging']['log_level'])

        openai_client = OpenAIClientManager(config['credentials_file'])
        api_preparer = APICallPreparer(config, prompt)
        parameters = api_preparer.prepare_parameters()

        completion = openai_client.create_completion(parameters)
        result = completion.choices[0].message.function_call.arguments if config.get('structured_output', {}).get('enable') else completion.choices[0].message.content

        if not result:
            raise CustomException("Received empty response from API.")
            
        Logger.debug(f"API call successful: {result}")

        return result

    except ValidationError as e:
        Logger.error(f"ValidationError: {e}")
        raise SystemExit("Validation error. Please check the structured output schema.")

    except CustomException as e:
        Logger.error(str(e))
        raise SystemExit(str(e))

    except BadRequestError as e:
        if "context_length_exceeded" in str(e):
            Logger.error("API request failed due to exceeding the token context length.")
            raise SystemExit("Token context length exceeded. Please adjust the input.")
        else:
            Logger.error(f"Unexpected API error: {e}")
            raise SystemExit(f"Unexpected API error: {e}")

    except Exception as e:
        Logger.error(f"An unexpected error occurred: {e}")
        raise SystemExit("An unexpected error occurred. Please try again later.")

# Example Usage
# result = gptapi('goalplanner', "Formulate a plan for beginning a small enterprise technology architecture consulting firm.")
# print(result)
