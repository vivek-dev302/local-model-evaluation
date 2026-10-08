import time
from typing import Any, Dict, List
from openai import OpenAI
from pydantic import ValidationError

try:
    from app.metrics import get_ollama_ram_mb, get_system_ram_percent
    from app.schemas import BenchmarkRecord, SupportTriageResponse
except ImportError:
    from metrics import get_ollama_ram_mb, get_system_ram_percent
    from schemas import BenchmarkRecord, SupportTriageResponse


def run_triage_with_retry(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_message: str,
    case_id: int,
    temperature: float = 0.0,
    max_retries: int = 1,
) -> BenchmarkRecord:
    """Executes streaming inference, benchmarks performance & RAM, validates schema, and retries upon failure."""
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
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
            stream_options={"include_usage": True},
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

        # Validate against Pydantic schema
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

        # Performance & resource metrics
        input_tokens = usage_info.prompt_tokens if usage_info else 0
        output_tokens = usage_info.completion_tokens if usage_info else completion_tokens_count
        total_tokens = usage_info.total_tokens if usage_info else (input_tokens + output_tokens)
        ttft = ttft if ttft is not None else total_time
        tps = output_tokens / total_time if total_time > 0 else 0
        ollama_ram_mb = get_ollama_ram_mb()
        system_ram_pct = get_system_ram_percent()

        if is_valid:
            return BenchmarkRecord(
                question_id=case_id,
                attempt=attempt + 1,
                temperature=temperature,
                validation_passed=True,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                ttft=round(ttft, 4),
                total_time=round(total_time, 4),
                tokens_per_second=round(tps, 2),
                ollama_ram_mb=ollama_ram_mb,
                system_ram_percent=system_ram_pct,
                output=validated_output.model_dump(),
                error=None,
                raw_response=full_response,
            )
        elif attempt < max_retries:
            messages.append({"role": "assistant", "content": full_response})
            messages.append({
                "role": "user",
                "content": f"Your previous response violated the JSON schema with error: {validation_error}. Please fix and return strictly valid JSON matching the schema.",
            })
        else:
            print("[!] Failed validation after max retries.")
            return BenchmarkRecord(
                question_id=case_id,
                attempt=attempt + 1,
                temperature=temperature,
                validation_passed=False,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                ttft=round(ttft, 4),
                total_time=round(total_time, 4),
                tokens_per_second=round(tps, 2),
                ollama_ram_mb=ollama_ram_mb,
                system_ram_percent=system_ram_pct,
                output=None,
                error=validation_error,
                raw_response=full_response,
            )
