import psutil

try:
    from app.schemas import BenchmarkRecord
except ImportError:
    from schemas import BenchmarkRecord


def get_ollama_ram_mb() -> float:
    """Calculates the physical RSS memory consumed by local Ollama processes in MB."""
    try:
        total_bytes = sum(
            p.info["memory_info"].rss
            for p in psutil.process_iter(["name", "memory_info"])
            if "ollama" in (p.info.get("name") or "").lower()
        )
        return round(total_bytes / (1024 * 1024), 2)
    except Exception:
        return 0.0


def get_system_ram_percent() -> float:
    """Returns the host system RAM utilization percentage."""
    try:
        return psutil.virtual_memory().percent
    except Exception:
        return 0.0


def save_benchmark_record(file_path: str, record: BenchmarkRecord) -> None:
    """Appends a validated BenchmarkRecord as a JSON line to the target report file."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")
