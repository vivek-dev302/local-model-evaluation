import json
import os
import time
from dotenv import load_dotenv
from openai import OpenAI
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

print(f"Loaded {len(test_cases)} test cases | Target Model: {model_name} | Base URL: {base_url}\n")

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
        except (ValidationError, Exception) as e:
            print(f"Schema Validation: FAILED (Attempt {attempt + 1}) - {e}")
            is_valid = False

        # Calculate metrics
        input_tokens = usage_info.prompt_tokens if usage_info else 0
        output_tokens = usage_info.completion_tokens if usage_info else completion_tokens_count
        total_tokens = usage_info.total_tokens if usage_info else (input_tokens + output_tokens)
        ttft = ttft if ttft is not None else total_time
        tps = output_tokens / total_time if total_time > 0 else 0

        print(f"Tokens: In={input_tokens}, Out={output_tokens}, Total={total_tokens}")
        print(f"TTFT: {ttft:.4f}s | Total Time: {total_time:.4f}s | TPS: {tps:.2f} tokens/s")
        print("-" * 40)

        if is_valid:
            break
        elif attempt < max_retries:
            messages.append({"role": "assistant", "content": full_response})
            messages.append({
                "role": "user",
                "content": f"Your previous response violated the JSON schema with error: {e}. Please fix and return strictly valid JSON matching the schema."
            })
        else:
            print("[!] Failed validation after max retries.")

    print("\n")
