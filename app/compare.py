import os
import glob
from typing import List, Dict, Any

from app.scripts.config import REPORTS_DIR_BASE
from app.scripts.analyze import summarize_report
from app.scripts.judge import run_judge_on_all_reports
from app.scripts.visualize import generate_all_charts
from app.scripts.report_generator import (
    format_markdown_table,
    generate_comparison_views,
    save_benchmark_report,
)


def discover_all_reports() -> List[str]:
    """Finds all report_temp_*.jsonl files inside reports/ directory."""
    pattern = os.path.join(REPORTS_DIR_BASE, "*", "report_temp_*.jsonl")
    report_files = glob.glob(pattern)
    report_files.sort()
    return report_files


def run_comparisons() -> List[Dict[str, Any]]:
    """Loads and summarizes all discovered report files."""
    files = discover_all_reports()
    if not files:
        print(f"No report files found matching {os.path.join(REPORTS_DIR_BASE, '*', 'report_temp_*.jsonl')}")
        return []

    summaries = []
    for f in files:
        summary = summarize_report(f)
        summaries.append(summary)

    # Sort primarily by model name, secondarily by temperature
    summaries.sort(key=lambda s: (s["model"], str(s["temperature"])))
    return summaries


def run_comparison_pipeline():
    """Main comparison workflow: runs judge, summarizes runs, generates charts & markdown report."""
    print("=" * 70)
    print("       MODEL COMPARISON BENCHMARK STUDY")
    print("=" * 70)

    # First evaluate quality for all reports with the judge
    run_judge_on_all_reports()
    summaries = run_comparisons()
    if not summaries:
        return

    tables_text = generate_comparison_views(summaries)
    print("\n" + tables_text + "\n")

    # Generate separate individual charts
    charts = generate_all_charts(summaries)
    print("Generated separate benchmark charts in reports/charts/:")
    for name, path in charts.items():
        print(f"  - {name}: {path}")

    # Save benchmark markdown report with separate embedded charts
    report_path = save_benchmark_report(summaries)
    print("-" * 70)
    print(f"Benchmark report generated successfully at:\n{report_path}")
    print("=" * 70)


def main():
    run_comparison_pipeline()


if __name__ == "__main__":
    main()
