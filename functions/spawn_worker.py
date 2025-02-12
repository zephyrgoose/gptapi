# spawn_worker.py

import logging
import subprocess
import os
import sys

# Configure logging
logging.basicConfig(
    filename="../debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

MAX_SPAWN_ATTEMPTS = 20
_spawn_count = 0

def spawn_worker(goal):
    """
    Spawns a new GPT worker instance with a specified goal (prompt).

    Args:
        goal (str): The prompt or goal for the new worker to achieve.

    Returns:
        dict: A dictionary with either {"output": "..."} or {"error": "..."} 
    """
    global _spawn_count
    
    # Enforce a maximum number of worker spawns
    if _spawn_count >= MAX_SPAWN_ATTEMPTS:
        logging.error("Reached maximum sub-worker spawn limit. Aborting.")
        return {"error": "Max sub-worker spawns reached."}
    
    _spawn_count += 1
    logging.info(f"Spawning new worker (count={_spawn_count}/{MAX_SPAWN_ATTEMPTS}) with goal: {goal}")

    script_path = os.path.join(os.path.dirname(__file__), "../gptapi.py")
    command = [
        sys.executable,
        script_path,
        "--profile",
        "worker",  # Assumes worker profile handles specific functions
        "--prompt",
        goal
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info("Spawned worker successfully.")
        return {"output": result.stdout.strip()}
    except subprocess.CalledProcessError as e:
        logging.error(f"Spawn worker returned non-zero exit code:\n{e.stderr}")
        return {"error": e.stderr.strip()}
    except Exception as e:
        logging.error(f"Exception while spawning worker: {e}")
        return {"error": str(e)}