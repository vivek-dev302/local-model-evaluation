import psutil

try:
    from app.schemas import BenchmarkRecord
except ImportError:
    from schemas import BenchmarkRecord


def get_ollama_ram_mb() -> float:
    """Calculates total physical RAM consumed by Ollama and its llama-server runners."""
    target_keywords = ("ollama", "llama")
    try:
        total_bytes = sum(
            p.info["memory_info"].rss
            for p in psutil.process_iter(["name", "memory_info"])
            if any(k in (p.info.get("name") or "").lower() for k in target_keywords)
        )
        return round(total_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0



def prime_ollama_cpu() -> None:
    """Initializes the CPU percentage counters on all Ollama & llama-server processes."""
    target_keywords = ("ollama", "llama")
    try:
        for p in psutil.process_iter(["name"]):
            if any(k in (p.info.get("name") or "").lower() for k in target_keywords):
                p.cpu_percent(interval=None)
    except Exception:
        pass


def get_ollama_cpu_percent() -> float:
    """
    Calculates total CPU utilization percentage across Ollama and llama-server processes,
    normalized by total logical cores so the result is a clean 0% - 100% range.
    """
    target_keywords = ("ollama", "llama")
    try:
        cores = psutil.cpu_count(logical=True) or 1
        total_cpu = sum(
            p.cpu_percent(interval=None)
            for p in psutil.process_iter(["name"])
            if any(k in (p.info.get("name") or "").lower() for k in target_keywords)
        )
        normalized = total_cpu / cores
        return round(normalized, 1)
    except Exception:
        return 0.0


def save_benchmark_record(file_path: str, record: BenchmarkRecord) -> None:
    """Appends a validated BenchmarkRecord as a JSON line to the target report file."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json(exclude_none=True) + "\n")
