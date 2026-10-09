import json
from openai import OpenAI

try:
    from app.config import (
        MAX_RETRIES,
        MODEL_NAME,
        OLLAMA_API_KEY,
        OLLAMA_BASE_URL,
        SYSTEM_PROMPT_PATH,
        TEMPERATURE,
        TEST_CASES_PATH,
        get_report_file_path,
    )
    from app.metrics import get_ollama_ram_mb, get_system_ram_percent, save_benchmark_record
    from app.triage_service import run_triage_with_retry
except ImportError:
    from config import (
        MAX_RETRIES,
        MODEL_NAME,
        OLLAMA_API_KEY,
        OLLAMA_BASE_URL,
        SYSTEM_PROMPT_PATH,
        TEMPERATURE,
        TEST_CASES_PATH,
        get_report_file_path,
    )
    from metrics import get_ollama_ram_mb, get_system_ram_percent, save_benchmark_record
    from triage_service import run_triage_with_retry


def main():
    # Load system prompt and test dataset
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize client and output report path
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    report_file = get_report_file_path()

    # Clear report file at the start of each run (fresh benchmark)
    open(report_file, "w").close()

    print(f"Loaded {len(test_cases)} test cases | Model: {MODEL_NAME} | Temp: {TEMPERATURE}")
    print(f"Initial Ollama RAM: {get_ollama_ram_mb()} MB | System RAM: {get_system_ram_percent()}%")
    print(f"Saving reports to: {report_file}\n")

    for test_case in test_cases:
        case_id = test_case["id"]
        category = test_case.get("category", "unknown")
        user_message = test_case["message"]

        print("=" * 60)
        print(f"Test Case #{case_id} [{category.upper()}]")
        print(f"User: {user_message}")
        print("=" * 60)

        record = run_triage_with_retry(
            client=client,
            model_name=MODEL_NAME,
            system_prompt=system_prompt,
            user_message=user_message,
            case_id=case_id,
            temperature=TEMPERATURE,
            max_retries=MAX_RETRIES,
        )

        save_benchmark_record(report_file, record)
        print(f"Metrics: In={record.input_tokens}, Out={record.output_tokens}, TTFT={record.ttft}s, TPS={record.tokens_per_second} tokens/s")
        print("\n")


if __name__ == "__main__":
    main()
