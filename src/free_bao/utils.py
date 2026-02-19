import os
import re

def load_keys_from_bashrc():
    """Loads API keys from ~/.bashrc if they are not already set."""
    required_keys = ["OPENAI_API_KEY", "WANDB_API_KEY"]
    missing = [k for k in required_keys if k not in os.environ]
    
    if not missing:
        return

    bashrc_path = os.path.expanduser("~/.bashrc")
    if not os.path.exists(bashrc_path):
        return

    with open(bashrc_path, "r") as f:
        content = f.read()

    for key in missing:
        # Match export KEY="value" or export KEY=value
        match = re.search(f'export {key}=[\'"]?([^\'"\n]+)[\'"]?', content)
        if match:
            os.environ[key] = match.group(1)
            print(f"Loaded {key} from .bashrc")
