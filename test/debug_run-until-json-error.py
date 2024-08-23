import logging
from pathlib import Path
import json
from gptapi import gptapi

def read_prompt_from_file(file_path):
    """
    Read the prompt text from a given file.
    
    Parameters:
    - file_path (str): The path to the input file containing the prompt.

    Returns:
    - str: The content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read().strip()
        return prompt
    except FileNotFoundError:
        logging.error("The file %s does not exist.", file_path)
        raise SystemExit(f"Error: The file {file_path} was not found.")
    except Exception as e:
        logging.error("An error occurred while reading the file %s: %s", file_path, e)
        raise SystemExit(f"Error: Unable to read the file {file_path}.")

def run_gptapi_until_json_fails(profile_name, prompt):
    """
    Run the gptapi function in a loop until the output cannot be interpreted as JSON.

    Parameters:
    - profile_name (str): The profile name for the GPT API.
    - prompt (str): The prompt to send to the GPT API.

    Returns:
    - str: The last non-JSON output for investigation.
    """
    iteration = 0
    while True:
        try:
            # Run the GPT API
            result = gptapi(profile_name, prompt)
            iteration += 1
            logging.info("Iteration %d: Successfully received result", iteration)

            # Display the result of this iteration
            #print(f"Iteration {iteration}: Result received.")
            #print(result)

            # Attempt to interpret the result as JSON
            json_output = json.loads(result)
            logging.info("Iteration %d: Output is valid JSON", iteration)

            # Display the parsed JSON
            print(f"Iteration {iteration}: Output is valid JSON.")
            print(json.dumps(json_output, indent=2))

        except json.JSONDecodeError:
            logging.error("Iteration %d: Output is not valid JSON. Stopping.", iteration)
            print(f"Stopping at iteration {iteration}. The output is not valid JSON:")
            print(result)
            return result
        except Exception as e:
            logging.error("Iteration %d: An error occurred while running the GPT API: %s", iteration, e)
            print("An error occurred. Please check the logs for more details.")
            return str(e)

def main():
    """
    Main function to run the gptapi with the goalplanner profile, using input from a file.
    """
    try:
        # Define the profile name and input file path
        profile_name = 'goalplanner'  # The YAML profile file name without the extension
        input_file_path = './input.txt'  # Path to the input file containing the prompt

        # Read the prompt from the input file
        prompt = read_prompt_from_file(input_file_path)

        # Run the GPT API until JSON interpretation fails
        last_output = run_gptapi_until_json_fails(profile_name, prompt)

        # Print the last output
        print("Last output before failure:")
        print(last_output)

    except Exception as e:
        logging.error("An error occurred while running the GPT API: %s", e)
        print("An error occurred. Please check the logs for more details.")

if __name__ == "__main__":
    main()
