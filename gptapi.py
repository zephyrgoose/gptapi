#!/usr/bin/env python3
# gptapi.py

import os
import json
import yaml
import logging
import openai
import argparse
import importlib.util
import threading
from collections import defaultdict
import time

# Configure logging
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

###############################################################################
#                           MEMORY MANAGER                                    #
###############################################################################
class MemoryManager:
    """Manages and tracks conversation history with thread safety."""
    def __init__(self):
        self._lock = threading.RLock()
        self._conversations = defaultdict(list)

    def append_message(self, conversation_id: str, role: str, content: str):
        with self._lock:
            self._conversations[conversation_id].append({"role": role, "content": content})
            logging.debug(f"Appended to {conversation_id}: {role[:6]}, {content[:60]}...")

    def get_messages(self, conversation_id: str):
        with self._lock:
            return list(self._conversations[conversation_id])

    def clear_conversation(self, conversation_id: str):
        with self._lock:
            if conversation_id in self._conversations:
                del self._conversations[conversation_id]
                logging.debug(f"Cleared conversation {conversation_id}.")

# Create a single, global instance of MemoryManager
memory_manager = MemoryManager()

def background_memory_thread():
    """Optional thread for periodic housekeeping/logging."""
    while True:
        with memory_manager._lock:
            num_active_conversations = len(memory_manager._conversations)
        logging.info(f"[Background Thread] Active conversations: {num_active_conversations}")
        time.sleep(30)  # Sleep 30 seconds between checks

###############################################################################
#                           HELPER FUNCTIONS                                  #
###############################################################################
def load_yaml(file_path: str) -> dict:
    """Safely load and parse a YAML file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        msg = f"YAML file not found: {file_path}"
        logging.error(msg)
        raise
    except yaml.YAMLError:
        logging.exception(f"Error parsing YAML file: {file_path}")
        raise ValueError(f"Error parsing YAML file: {file_path}")

def load_profile(profile_name: str, profiles_dir: str = "profiles") -> dict:
    """
    Load and validate a YAML profile (e.g., manager.yaml or worker.yaml).
    Assumes 'output' is part of structured_output in the JSON schema.
    """
    profile_path = os.path.join(profiles_dir, f"{profile_name}.yaml")
    if not os.path.exists(profile_path):
        errmsg = f"Profile '{profile_name}' not found at {profile_path}"
        logging.error(errmsg)
        raise FileNotFoundError(errmsg)

    profile = load_yaml(profile_path)
    required = ["model", "system_prompt", "parameters", "structured_output"]
    for field in required:
        if field not in profile:
            errmsg = f"Profile missing required field: {field}"
            logging.error(errmsg)
            raise ValueError(errmsg)

    return profile

def load_api_key(keys_filename: str = "./keys.yaml") -> str:
    """Load the OpenAI API key from a YAML file."""
    keys = load_yaml(keys_filename)
    if "openai_api" not in keys:
        errmsg = "Missing 'openai_api' in keys.yaml."
        logging.error(errmsg)
        raise ValueError(errmsg)
    return keys["openai_api"]

def load_function(function_name: str):
    """Dynamically load a function from the functions/ directory."""
    func_filename = f"{function_name}.py"
    func_path = os.path.join(os.path.dirname(__file__), "functions", func_filename)
    if not os.path.exists(func_path):
        logging.error(f"Function file {func_filename} not found in functions/.")
        return None
    try:
        spec = importlib.util.spec_from_file_location(function_name, func_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, function_name):
            logging.info(f"Successfully loaded function: {function_name} from {func_filename}")
            return getattr(module, function_name)
        else:
            logging.error(f"Function '{function_name}' not in {func_filename}.")
            return None
    except Exception:
        logging.exception(f"Error loading function {function_name}")
        return None

###############################################################################
#                        SINGLE-CALL GPT FUNCTION                             #
###############################################################################
def gptapi(profile_name: str, prompt: str, conversation_id: str = "default"):
    """Single-step GPT call with a chosen profile (e.g., 'worker')."""
    # Load profile & key
    profile = load_profile(profile_name)
    api_key = load_api_key(profile.get("credentials_file", "./keys.yaml"))
    client = openai.OpenAI(api_key=api_key)

    logging.info(f"Single-step GPT call with profile='{profile_name}', prompt='{prompt}'")

    existing_messages = memory_manager.get_messages(conversation_id)
    if not existing_messages:
        memory_manager.append_message(conversation_id, "system", profile["system_prompt"])
        existing_messages = memory_manager.get_messages(conversation_id)

    memory_manager.append_message(conversation_id, "user", prompt)
    updated_messages = memory_manager.get_messages(conversation_id)

    tools = profile.get("tools", [])
    tool_choice = profile.get("tool_choice", "auto")

    response = client.chat.completions.create(
        model=profile["model"],
        messages=updated_messages,
        tools=tools,
        tool_choice=tool_choice,
        **profile["parameters"]
    )

    if not response.choices:
        logging.warning("No response from GPT in single-step call.")
        return "No response from GPT."

    message = response.choices[0].message

    if not message.tool_calls:
        gpt_text = message.content or ""
        memory_manager.append_message(conversation_id, "assistant", gpt_text)
        logging.debug("No tool calls found, returning direct GPT text.")
        return gpt_text

    all_results = []
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        try:
            func_args = json.loads(tool_call.function.arguments)
        except Exception as e:
            errmsg = f"Cannot parse arguments for '{func_name}': {e}"
            logging.error(errmsg)
            all_results.append({func_name: errmsg})
            continue

        loaded_func = load_function(func_name)
        if not loaded_func:
            err = f"Tool '{func_name}' not found or error loading."
            logging.error(err)
            all_results.append({func_name: err})
            continue

        try:
            invoked_result = loaded_func(**func_args)
            all_results.append({func_name: invoked_result})
        except Exception as e:
            errmsg = f"Exception in '{func_name}': {e}"
            logging.error(errmsg)
            all_results.append({func_name: errmsg})

    return all_results

###############################################################################
#                    MULTI-STEP MANAGER CONTROL LOOP                          #
###############################################################################
def try_extract_output_from_content(content):
    """
    Attempt to extract a structured JSON output (with an "output" key) 
    from a JSON code fence in the provided content. 
    Returns the value of "output" if found, else None.
    """
    import re
    import json
    
    if not isinstance(content, str):
        return None
    
    # Look for JSON code fences wrapped in ```json ... ```
    code_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', content, flags=re.DOTALL)
    for block in code_blocks:
        try:
            parsed = json.loads(block)
            # The manager.yaml definitions expect "output" to exist.
            if "output" in parsed and parsed["output"] is not None:
                return parsed["output"]
        except Exception as e:
            logging.debug(f"JSON parse error in code fence: {e}")
    
    return None

def manager_control_loop(user_prompt: str, 
                         max_steps: int = 10, 
                         conversation_id: str = "manager-default") -> str:
    """
    A multi-step Manager loop that uses memory_manager to accumulate conversation
    history and calls GPT repeatedly until an 'output' field is produced in the
    structured response. The manager can spawn workers and gather tool results,
    then finalize when 'output' is found.
    """
    import json
    mgr_profile = load_profile("manager")
    api_key = load_api_key(mgr_profile.get("credentials_file", "./keys.yaml"))
    client = openai.OpenAI(api_key=api_key)

    # Seed the conversation if there isn't one already.
    existing_messages = memory_manager.get_messages(conversation_id)
    if not existing_messages:
        memory_manager.append_message(conversation_id, "system", mgr_profile["system_prompt"])
        memory_manager.append_message(conversation_id, "user", user_prompt)

    tools = mgr_profile.get("tools", [])
    tool_choice = mgr_profile.get("tool_choice", "auto")

    for step_index in range(max_steps):
        logging.info(f"[Manager Loop] Step {step_index + 1}/{max_steps}")

        conversation = memory_manager.get_messages(conversation_id)
        response = client.chat.completions.create(
            model=mgr_profile["model"],
            messages=conversation,
            tools=tools,
            tool_choice=tool_choice,
            **mgr_profile["parameters"]
        )

        if not response.choices:
            logging.warning("No manager response. Nudging GPT.")
            memory_manager.append_message(conversation_id, "assistant", "No GPT response. Retrying...")
            continue

        message = response.choices[0].message

        # Log message content if present
        if message.content is not None:
            logging.debug(f"[Manager Loop] GPT message content: {message.content[:150]}")
        else:
            logging.warning("[Manager Loop] Received message with no content.")

        # Save the assistant's reply to memory.
        memory_manager.append_message(conversation_id, "assistant", message.content or "")

        # ---------------------------------------------------------------------
        # REMOVED the check:
        # if "output" in message and message["output"] is not None:
        #    ...
        # Because 'message' is not a dict, we rely on parsing message.content
        # ---------------------------------------------------------------------

        # Extract structured output from message.content (the main path):
        extracted_output = try_extract_output_from_content(message.content or "")
        if extracted_output is not None:
            logging.info("Extracted 'output' from message content. Finalizing.")
            return extracted_output

        # Handle tool calls if present (function calling)
        if message.tool_calls:
            bundle = []
            for tc in message.tool_calls:
                fname = tc.function.name
                try:
                    fargs = json.loads(tc.function.arguments)
                except Exception as e:
                    errmsg = f"Invalid JSON from Manager: {e}"
                    logging.error(errmsg)
                    bundle.append({fname: errmsg})
                    continue
                logging.info(f"Manager requests tool='{fname}' with {fargs}")
                tool_obj = load_function(fname)
                if not tool_obj:
                    err = f"Could not load tool '{fname}'."
                    logging.error(err)
                    bundle.append({fname: err})
                    continue
                try:
                    result = tool_obj(**fargs)
                    bundle.append({fname: result})
                except Exception as e:
                    errmsg = f"Tool '{fname}' error: {e}"
                    logging.error(errmsg)
                    bundle.append({fname: errmsg})

            tool_result_json = json.dumps(bundle, ensure_ascii=False)
            combined_content = (
                "Tool results: " + tool_result_json +
                "\nIf you can now finalize your answer, include 'output' in your JSON."
            )
            memory_manager.append_message(conversation_id, "assistant", combined_content)
        else:
            # If no tool calls were made, nudge GPT to finalize.
            memory_manager.append_message(
                conversation_id,
                "assistant",
                "You have no tool calls left and did not provide 'output'. Please finalize."
            )

    logging.warning("Manager did not produce 'output' by max_steps. Using fallback.")
    conversation = memory_manager.get_messages(conversation_id)
    return parse_and_combine_tool_results(conversation)

def parse_and_combine_tool_results(conversation: list) -> str:
    """
    A fallback function if the Manager never provides 'output'.
    :param conversation: The entire chain of messages so far.
    :return: A text summary from the last known tool results or a fallback.
    """
    for msg in reversed(conversation):
        if (msg["role"] == "assistant") and ("Tool results:" in msg["content"]):
            content = msg["content"]
            splitted = content.split("Tool results:")
            if len(splitted) < 2:
                continue
            raw_json_part = splitted[1].strip()
            marker = "\nIf you can now finalize"
            idx = raw_json_part.find(marker)
            if idx != -1:
                raw_json_part = raw_json_part[:idx].strip()
            try:
                results = json.loads(raw_json_part)
            except Exception as exc:
                logging.error(f"Fallback: couldn't parse tool data: {exc}")
                return "Could not parse final tool data."
            weather_lines = []
            for item in results:
                for key, val in item.items():
                    if key == "spawn_worker":
                        out_text = val.get("output", "")
                        if out_text.startswith("Result: "):
                            real_json = out_text.replace("Result: ", "", 1)
                            try:
                                subres = json.loads(real_json)
                                for sr in subres:
                                    if "get_current_weather" in sr:
                                        w = sr["get_current_weather"]
                                        desc = w.get("description", "")
                                        temp = w.get("temperature", "")
                                        hum = w.get("humidity", "")
                                        weather_lines.append(
                                            f"Weather: {desc}, {temp}°C, humidity {hum}%"
                                        )
                            except Exception as e:
                                logging.debug(f"Fallback parse error: {e}")
            if weather_lines:
                return " | ".join(weather_lines)
            return "No conclusive final data found in fallback parse."
    return "No tool results found in fallback parse."

###############################################################################
#                         MAIN FUNCTION (CLI)                                 #
###############################################################################
def main():
    """
    Simple CLI for either:
      - manager_control_loop (profile="manager"), or
      - single-call usage (profile="worker" or something else).
    """
    parser = argparse.ArgumentParser(description="Run GPT with a chosen profile and prompt.")
    parser.add_argument(
        "-p", "--profile", required=True,
        help="Profile name (e.g., 'manager' or 'worker')."
    )
    parser.add_argument(
        "-q", "--prompt", required=True,
        help="User prompt or question to GPT."
    )
    parser.add_argument(
        "--max_steps", type=int, default=10,
        help="Maximum manager loop steps."
    )
    parser.add_argument(
        "--conversation_id", default="default",
        help="Conversation ID for memory management."
    )
    parser.add_argument(
        "--start_bg_thread", action="store_true",
        help="If provided, starts a background thread to log conversation stats."
    )
    args = parser.parse_args()

    if args.start_bg_thread:
        threading.Thread(target=background_memory_thread, daemon=True).start()

    if args.profile == "manager":
        answer = manager_control_loop(
            args.prompt,
            max_steps=args.max_steps,
            conversation_id=args.conversation_id
        )
        print(answer)
    else:
        result = gptapi(
            args.profile,
            args.prompt,
            conversation_id=args.conversation_id
        )
        print("Result: " + json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()