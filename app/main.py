import json
from openai import OpenAI

try:
    from app.scripts.config import (
        MAX_RETRIES,
        MODEL_NAME,
        OLLAMA_API_KEY,
        OLLAMA_BASE_URL,
        SYSTEM_PROMPT_PATH,
        TEMPERATURES,
        TEST_CASES_PATH,
        get_report_file_path,
    )
    from app.scripts.metrics import get_ollama_ram_mb, save_benchmark_record
    from app.scripts.triage_service import run_triage_with_retry
except ImportError:
    from scripts.config import (
        MAX_RETRIES,
        MODEL_NAME,
        OLLAMA_API_KEY,
        OLLAMA_BASE_URL,
        SYSTEM_PROMPT_PATH,
        TEMPERATURES,
        TEST_CASES_PATH,
        get_report_file_path,
    )
    from scripts.metrics import get_ollama_ram_mb, save_benchmark_record
    from scripts.triage_service import run_triage_with_retry


def main():
    # Load system prompt and test dataset
    with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()

    with open(TEST_CASES_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    # Initialize client
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)

    print(f"Loaded {len(test_cases)} test cases | Model: {MODEL_NAME} | Temperatures: {TEMPERATURES}")
    print(f"Initial Ollama RAM: {get_ollama_ram_mb()} MB\n")

    for temp in TEMPERATURES:
        report_file = get_report_file_path(temp)

        # Clear report file at the start of each temperature run (fresh benchmark)
        open(report_file, "w").close()

        print("=" * 70)
        print(f"   STARTING BENCHMARK: {MODEL_NAME} @ Temperature: {temp}")
        print(f"   Saving output to: {report_file}")
        print("=" * 70 + "\n")

        for test_case in test_cases:
            case_id = test_case["id"]
            category = test_case.get("category", "unknown")
            user_message = test_case["message"]

            print("=" * 60)
            print(f"Test Case #{case_id} [{category.upper()}] (Temp: {temp})")
            print(f"User: {user_message}")
            print("=" * 60)

            record = run_triage_with_retry(
                client=client,
                model_name=MODEL_NAME,
                system_prompt=system_prompt,
                user_message=user_message,
                case_id=case_id,
                temperature=temp,
                max_retries=MAX_RETRIES,
            )

            save_benchmark_record(report_file, record)
            print(f"Metrics: In={record.input_tokens}, Out={record.output_tokens}, TTFT={record.ttft}s, TPS={record.tokens_per_second} tokens/s")
            print("\n")

        print(f"\n[OK] Completed all test cases for {MODEL_NAME} @ Temperature: {temp}\n")

    print("All temperature benchmark runs completed successfully!")


if __name__ == "__main__":
    main()
