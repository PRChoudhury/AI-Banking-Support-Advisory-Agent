import config
from datetime import datetime
from pathlib import Path
import os

def load_role(name: str) -> str:
    """
    Returns the prompt files from specified folder.
    """
    path = config.ROLES_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Role prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Role prompt file is empty: {path}")
    return text

# -------------------------------
# Logging helpers
# -------------------------------
def log_to_file(log_file, version, message: str, error: bool = False) -> None:
    """Write logs immediately to file to avoid buffering issues.

    The caller supplies log_file so each phase logs into its own directory.
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level = "ERROR" if error else "INFO"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{ts} | Prompt: {version} |{level} | {message}\n")
        f.flush()


def reset_log(log_file) -> None:
    """Truncate the log so each run starts clean."""
    Path(log_file).write_text("", encoding="utf-8")


def log_interaction(log_file, version, query: str, response: str, style: str) -> None:
    """Log a single user query and the agent's response."""
    log_to_file(log_file, version, f"QUERY | {query}")
    log_to_file(log_file, version, f"STYLE USED | {style}")
    log_to_file(log_file, version, f"RESPONSE | {response}")
    log_to_file(log_file, version, "=" * 59 + "\n")

# -------------------------------
# CHECK API KEY
# -------------------------------

def check_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file before starting the app."
        )