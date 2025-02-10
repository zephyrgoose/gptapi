import logging
import subprocess
import os
import sys

logging.basicConfig(
    filename="../debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def spawn_worker(goal):
    """
    Spawns a new GPT worker instance with a specified goal (prompt).

    Args:
        goal (str): The prompt or goal for the new worker to achieve.

    Returns:
        dict: A dictionary with either:
            {"output": "..."} or {"error": "..."} 
    """
    logging.info(f"Spawning new worker with goal: {goal}")

    script_path = os.path.join(os.path.dirname(__file__), "../gptapi.py")
    command = [
        sys.executable,
        script_path,
        "--profile",
        "worker",            # <--- Points to the updated worker profile
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