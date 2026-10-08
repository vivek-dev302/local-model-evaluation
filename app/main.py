import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import psutil
from pydantic import ValidationError

try:
    from app.schemas import SupportTriageResponse
except ImportError:
    from schemas import SupportTriageResponse

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
SYSTEM_PROMPT_PATH = os.path.join(BASE_DIR, "system_prompt", "system_prompt.txt")
TEST_CASES_PATH = os.path.join(BASE_DIR, "test_cases.json")

# Load environment variables
load_dotenv(ENV_PATH)

base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
api_key = os.getenv("OLLAMA_API_KEY", "ollama")
model_name = os.getenv("MODEL_NAME", "llama3.2:3b")
temperature = float(os.getenv("TEMPERATURE", "0.0"))

# Model reports directory and temperature-specific report file
clean_model_name = model_name.replace(":", "_").replace("/", "_")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", clean_model_name)
os.makedirs(REPORTS_DIR, exist_ok=True)
REPORT_FILE = os.path.join(REPORTS_DIR, f"report_temp_{temperature}.jsonl")

# Helper to measure Ollama RAM usage
def get_ollama_ram_mb() -> float:
    try:
        total_bytes = sum(
            p.info["memory_info"].rss
            for p in psutil.process_iter(["name", "memory_info"])
            if "ollama" in (p.info.get("name") or "").lower()
        )
        return round(total_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0

# Load system prompt and test cases
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
    test_cases = json.load(f)

client = OpenAI(
    base_url=base_url,
    api_key=api_key
)

max_retries = 1

print(f"Loaded {len(test_cases)} test cases | Model: {model_name} | Temp: {temperature}")
print(f"Initial Ollama RAM: {get_ollama_ram_mb()} MB | System RAM: {psutil.virtual_memory().percent}%")
print(f"Saving reports to: {REPORT_FILE}\n")

for test_case in test_cases:
    case_id = test_case["id"]
    category = test_case.get("category", "unknown")
    user_message = test_case["message"]

    print("=" * 60)
    print(f"Test Case #{case_id} [{category.upper()}]")
    print(f"User: {user_message}")
    print("=" * 60)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    for attempt in range(max_retries + 1):
        start_time = time.perf_counter()
        ttft = None
        full_response = ""
        completion_tokens_count = 0
        usage_info = None

        if attempt > 0:
            print(f"\n--- Retrying Attempt {attempt + 1}/{max_retries + 1} ---\n")
        else:
            print("Response:\n")

        stream = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            stream=True,
            stream_options={"include_usage": True}
        )

        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    print(delta.content, end="", flush=True)
                    full_response += delta.content
                    completion_tokens_count += 1
            if hasattr(chunk, "usage") and chunk.usage:
                usage_info = chunk.usage

        total_time = time.perf_counter() - start_time

        # Validate with Pydantic schema
        print("\n\n" + "-" * 40)
        try:
            validated_output = SupportTriageResponse.model_validate_json(full_response)
            print(f"Schema Validation: SUCCESS (Attempt {attempt + 1})")
            print(f"Intent: {validated_output.intent} | Priority: {validated_output.priority} | Requires Human: {validated_output.requires_human}")
            is_valid = True
            validation_error = None
        except (ValidationError, Exception) as e:
            print(f"Schema Validation: FAILED (Attempt {attempt + 1}) - {e}")
            is_valid = False
            validation_error = str(e)

        # Calculate metrics for this attempt
        input_tokens = usage_info.prompt_tokens if usage_info else 0
        output_tokens = usage_info.completion_tokens if usage_info else completion_tokens_count
        total_tokens = usage_info.total_tokens if usage_info else (input_tokens + output_tokens)
        ttft = ttft if ttft is not None else total_time
        tps = output_tokens / total_time if total_time > 0 else 0
        ollama_ram_mb = get_ollama_ram_mb()
        system_ram_percent = psutil.virtual_memory().percent

        if is_valid:
            report_data = {
                "question_id": case_id,
                "attempt": attempt + 1,
                "temperature": temperature,
                "validation_passed": True,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "ttft": round(ttft, 4),
                "total_time": round(total_time, 4),
                "tokens_per_second": round(tps, 2),
                "ollama_ram_mb": ollama_ram_mb,
                "system_ram_percent": system_ram_percent,
                "output": validated_output.model_dump(),
                "error": None
            }
            with open(REPORT_FILE, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(report_data) + "\n")
            break
        elif attempt < max_retries:
            messages.append({"role": "assistant", "content": full_response})
            messages.append({
                "role": "user",
                "content": f"Your previous response violated the JSON schema with error: {validation_error}. Please fix and return strictly valid JSON matching the schema."
            })
        else:
            print("[!] Failed validation after max retries. Saving failed record.")
            report_data = {
                "question_id": case_id,
                "attempt": attempt + 1,
                "temperature": temperature,
                "validation_passed": False,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "ttft": round(ttft, 4),
                "total_time": round(total_time, 4),
                "tokens_per_second": round(tps, 2),
                "ollama_ram_mb": ollama_ram_mb,
                "system_ram_percent": system_ram_percent,
                "output": None,
                "error": validation_error,
                "raw_response": full_response
            }
            with open(REPORT_FILE, "a", encoding="utf-8") as rf:
                rf.write(json.dumps(report_data) + "\n")

    print("\n")
