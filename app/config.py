import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt", "system_prompt.txt")
JUDGE_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt", "judge_prompt.txt")
TEST_CASES_PATH = os.path.join(BASE_DIR, "test_cases.json")
REPORTS_DIR_BASE = os.path.join(BASE_DIR, "reports")

load_dotenv(ENV_PATH)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2:3b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1"))
EVAL_MODEL = os.getenv("EVAL_MODEL", "ggozad/prometheus2")


def get_report_file_path() -> str:
    """Generates the report file path, defaulting to configured MODEL_NAME and TEMPERATURE."""
    clean_model = MODEL_NAME.replace(":", "_").replace("/", "_")
    model_dir = os.path.join(REPORTS_DIR_BASE, clean_model)
    os.makedirs(model_dir, exist_ok=True)
    return os.path.join(model_dir, f"report_temp_{TEMPERATURE}.jsonl")
