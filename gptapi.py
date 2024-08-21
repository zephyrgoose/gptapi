import yaml
from openai import Client

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

def load_profile(profile_filename):
    """Loads and validates the profile from a YAML configuration file."""
    profile = load_yaml(profile_filename)
    
    # Define required fields
    required_fields = ['model', 'system_prompt', 'parameters', 'structured_output']
    validate_config(profile, required_fields)
    
    return profile

def load_api_key(keys_filename):
    """Loads the API key from a separate YAML file."""
    keys = load_yaml(keys_filename)
    
    if 'openai_api' not in keys:
        raise ValueError("Missing 'openai_api' in keys file.")
    
    return keys['openai_api']

def gptapi(profile_filename, prompt):
    """Main function to interact with the GPT API."""
    profile = load_profile(profile_filename)
    api_key = load_api_key(profile.get('credentials_file', './keys.yaml'))
    
    client = Client(api_key=api_key)  # Initialize the OpenAI client
    
    # Prepare the response format
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": profile['structured_output']['name'],
            "schema": profile['structured_output']['schema'],
            "strict": profile['structured_output'].get('strict', True)
        }
    }
    
    # Print the response format before the API call
    print(f"Response Format: {response_format}")
    
    # Prepare the API request
    response = client.chat.completions.create(
        model=profile['model'],
        messages=[
            {"role": "system", "content": profile['system_prompt']},
            {"role": "user", "content": prompt}
        ],
        response_format=response_format,
        **profile['parameters']
    )
    
    # Handle the response
    content = response.choices[0].message.content
    print("API Response:", content)
    
    # Basic error handling
    if hasattr(response, 'usage'):
        print("Token Usage:", response.usage)
    if hasattr(response, 'errors'):
        print("Errors:", response.errors)

# Example usage:
gptapi('./profiles/webvulnscraper.yaml', "Today, CISA—in partnership with the Federal Bureau of Investigation (FBI)—released an update to joint Cybersecurity Advisory #StopRansomware: Royal Ransomware,#StopRansomware: BlackSuit (Royal) Ransomware. The updated advisory provides network defenders with recent and historically observed tactics, techniques, and procedures (TTPs) and indicators of compromise (IOCs) associated with BlackSuit and legacy Royal activity. FBI investigations identified these TTPs and IOCs as recently as July 2024. [#StopRansomware: BlackSuit (Royal) Ransomware](https://www.cisa.gov/news-events/cybersecurity-advisories/aa23-061a). BlackSuit ransomware attacks have spread across numerouscritical infrastructure sectorsincluding, but not limited to, commercial facilities, healthcare and public health, government facilities, and critical manufacturing. [critical infrastructure sectors](https://www.cisa.gov/topics/critical-infrastructure-security-and-resilience/critical-infrastructure-sectors). CISA encourages network defenders to review the updated advisory and apply the recommended mitigations. See#StopRansomwarefor additional guidance on ransomware protection, detection, and response. Visit CISA’sCross-Sector Cybersecurity Performance Goalsfor more information on the CPGs, including additional recommended baseline protections. [#StopRansomware](https://www.cisa.gov/stopransomware). [Cross-Sector Cybersecurity Performance Goals](https://www.cisa.gov/cross-sector-cybersecurity-performance-goals). CISA encourages software manufacturers to take ownership of improving the security outcomes of their customers by applying secure by design tactics. For more information on secure by design, see CISA’sSecure by Designwebpage and joint guideShifting the Balance of Cybersecurity Risk: Principles and Approaches for Secure by Design Software.")
