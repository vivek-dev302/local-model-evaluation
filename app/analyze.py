import os
import json
from statistics import mean
from typing import List, Dict, Any, Optional


from app.evaluator import evaluate_report_accuracy


def load_report(file_path: str) -> List[Dict[str, Any]]:
    """Reads a .jsonl report file line-by-line and returns a list of records."""
    records = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Report file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping corrupted line {line_num} in {file_path}: {e}")

    return records


def summarize_report(file_path: str) -> Dict[str, Any]:
    """
    Step 1: Analyzes a single .jsonl report file and computes summary metrics.
    Handles missing or None values safely.
    """
    records = load_report(file_path)
    total_records = len(records)
    if total_records == 0:
        return {"file_path": file_path, "total_records": 0, "error": "No records found in file"}

    # Extract metadata (model name and temperature)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    file_name = os.path.basename(file_path)
    model_name = parent_dir or "unknown"
    temperature = records[0].get("temperature", "unknown")

    # Filter numeric values for safe averages
    tps_vals = [r["tokens_per_second"] for r in records if r.get("tokens_per_second") is not None]
    ttft_vals = [r["ttft"] for r in records if r.get("ttft") is not None]
    latency_vals = [r["total_time"] for r in records if r.get("total_time") is not None]
    out_token_vals = [r["output_tokens"] for r in records if r.get("output_tokens") is not None]

    ollama_ram_vals = [r["ollama_ram_mb"] for r in records if r.get("ollama_ram_mb") is not None]
    sys_ram_vals = [r["system_ram_percent"] for r in records if r.get("system_ram_percent") is not None]

    # Reliability and schema validity counts
    passed_records = [r for r in records if r.get("validation_passed") is True]
    first_attempt_pass = [r for r in passed_records if r.get("attempt") == 1]
    retry_pass = [r for r in passed_records if (r.get("attempt") or 1) > 1]
    failed_records = [r for r in records if r.get("validation_passed") is not True or r.get("error") is not None]

    pass_rate = (len(passed_records) / total_records) * 100 if total_records > 0 else 0.0

    # Calculate unified weighted accuracy against ground truths
    acc_stats = evaluate_report_accuracy(records)

    return {
        "model": model_name,
        "temperature": temperature,
        "file_name": file_name,
        "file_path": file_path,
        "total_records": total_records,
        # Unified Quality & Accuracy
        "overall_accuracy_pct": acc_stats["overall_accuracy_pct"],
        "category_accuracy_pct": acc_stats["category_accuracy_pct"],
        "escalation_accuracy_pct": acc_stats["escalation_accuracy_pct"],
        "priority_accuracy_pct": acc_stats["priority_accuracy_pct"],
        "sentiment_accuracy_pct": acc_stats["sentiment_accuracy_pct"],
        # Reliability & Schema
        "passed_count": len(passed_records),
        "pass_rate_pct": round(pass_rate, 2),
        "first_attempt_pass_count": len(first_attempt_pass),
        "retry_pass_count": len(retry_pass),
        "failed_count": len(failed_records),
        # Latency & Speed
        "avg_tokens_per_second": round(mean(tps_vals), 2) if tps_vals else 0.0,
        "avg_ttft_sec": round(mean(ttft_vals), 3) if ttft_vals else 0.0,
        "avg_total_time_sec": round(mean(latency_vals), 3) if latency_vals else 0.0,
        "avg_output_tokens": round(mean(out_token_vals), 1) if out_token_vals else 0.0,
        # Resource footprint
        "avg_ollama_ram_mb": round(mean(ollama_ram_vals), 2) if ollama_ram_vals else 0.0,
        "peak_ollama_ram_mb": round(max(ollama_ram_vals), 2) if ollama_ram_vals else 0.0,
        "avg_system_ram_pct": round(mean(sys_ram_vals), 2) if sys_ram_vals else 0.0,
    }


def print_single_summary(summary: Dict[str, Any]) -> None:
    """Pretty prints the summary of a single report run."""
    print("=" * 60)
    print(f"REPORT SUMMARY: {summary['model']} (Temp: {summary['temperature']})")
    print(f"File: {summary['file_name']} ({summary['total_records']} test cases)")
    print("-" * 60)
    print("Decision Accuracy (Ground Truth):")
    print(f"  - Overall Triage Accuracy: {summary['overall_accuracy_pct']}% (Weighted)")
    print(f"  - Category Routing:        {summary['category_accuracy_pct']}%")
    print(f"  - Escalation Correctness:  {summary['escalation_accuracy_pct']}%")
    print("\nReliability & Schema Validity:")
    print(f"  - Total Passed:            {summary['passed_count']} / {summary['total_records']} ({summary['pass_rate_pct']}%)")
    print(f"  - 1st Attempt Pass:        {summary['first_attempt_pass_count']}")
    print(f"  - Retry Pass:              {summary['retry_pass_count']}")
    print(f"  - Total Failed:            {summary['failed_count']}")
    print("\nSpeed & Latency:")
    print(f"  - Avg Throughput:     {summary['avg_tokens_per_second']} tokens/sec")
    print(f"  - Avg TTFT:           {summary['avg_ttft_sec']} s")
    print(f"  - Avg Total Time:     {summary['avg_total_time_sec']} s")
    print(f"  - Avg Output Tokens:  {summary['avg_output_tokens']} tokens")
    print("\nResource Footprint:")
    print(f"  - Avg Ollama RAM:     {summary['avg_ollama_ram_mb']} MB")
    print(f"  - Peak Ollama RAM:    {summary['peak_ollama_ram_mb']} MB")
    print(f"  - Avg System RAM:     {summary['avg_system_ram_pct']}%")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    from app.config import REPORTS_DIR_BASE

    # If user passes a specific file path, use it. Otherwise test on the first available report.
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        sample_path = os.path.join(REPORTS_DIR_BASE, "llama3.2_3b", "report_temp_0.0.jsonl")
        target_file = sample_path

    if os.path.exists(target_file):
        summary = summarize_report(target_file)
        print_single_summary(summary)
    else:
        print(f"File not found: {target_file}")
