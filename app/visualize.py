import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from app.config import REPORTS_DIR_BASE


def get_charts_dir() -> str:
    charts_dir = os.path.join(REPORTS_DIR_BASE, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir


def plot_throughput(summaries: List[Dict[str, Any]], output_path: str) -> str:
    """Creates a standalone throughput (tokens/sec) bar chart."""
    labels = [f"{s['model']}\n(T={s['temperature']})" for s in summaries]
    x = np.arange(len(labels))
    palette = ["#2563EB", "#3B82F6", "#7C3AED", "#A855F7"]
    colors = [palette[i % len(palette)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(8, 5))
    tps_values = [s["avg_tokens_per_second"] for s in summaries]
    bars = ax.bar(x, tps_values, color=colors, width=0.5, edgecolor="#1E293B", linewidth=1)

    ax.set_title("Inference Throughput Comparison (Tokens / Sec)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Tokens / Sec", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            f"{h:.2f} t/s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_latency(summaries: List[Dict[str, Any]], output_path: str) -> str:
    """Creates a standalone latency profile (TTFT vs Total Latency) bar chart."""
    labels = [f"{s['model']}\n(T={s['temperature']})" for s in summaries]
    x = np.arange(len(labels))
    width = 0.35

    ttft_vals = [s["avg_ttft_sec"] for s in summaries]
    tot_vals = [s["avg_total_time_sec"] for s in summaries]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - width / 2, ttft_vals, width=width, label="Avg TTFT (s)", color="#0EA5E9", edgecolor="#0369A1")
    b2 = ax.bar(x + width / 2, tot_vals, width=width, label="Avg Total Time (s)", color="#F97316", edgecolor="#C2410C")

    ax.set_title("Latency Profile: TTFT vs Total Response Time", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Seconds", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reliability_pie(summaries: List[Dict[str, Any]], output_path: str) -> str:
    """
    Creates pie/donut charts for schema validity across each model & temperature configuration.
    Displays 1st Attempt Pass, Retry Pass, and Failed slices.
    """
    n = len(summaries)
    if n == 0:
        return output_path

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("Reliability & Schema Validity Breakdown", fontsize=14, fontweight="bold", y=1.03)

    category_colors = {
        "1st Pass": "#10B981",
        "Retry Pass": "#F59E0B",
        "Failed": "#EF4444"
    }

    for idx, s in enumerate(summaries):
        ax = axes[idx]
        counts = [
            ("1st Pass", s["first_attempt_pass_count"]),
            ("Retry Pass", s["retry_pass_count"]),
            ("Failed", s["failed_count"]),
        ]

        # Filter out 0 count items for clean pie display
        active_counts = [(label, count) for label, count in counts if count > 0]
        labels = [item[0] for item in active_counts]
        values = [item[1] for item in active_counts]
        colors = [category_colors[label] for label in labels]

        # Draw donut/pie chart
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct=lambda pct: f"{int(round(pct*sum(values)/100))}\n({pct:.0f}%)",
            colors=colors,
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.65},  # donut style
            textprops={"fontsize": 9}
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_fontweight("bold")

        ax.set_title(f"{s['model']}\n(T={s['temperature']})", fontsize=11, fontweight="bold", pad=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_memory(summaries: List[Dict[str, Any]], output_path: str) -> str:
    """Creates a standalone memory footprint (Avg vs Peak RAM) bar chart."""
    labels = [f"{s['model']}\n(T={s['temperature']})" for s in summaries]
    x = np.arange(len(labels))
    width = 0.35

    avg_ram = [s["avg_ollama_ram_mb"] for s in summaries]
    peak_ram = [s["peak_ollama_ram_mb"] for s in summaries]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    b1 = ax.bar(x - width / 2, avg_ram, width=width, label="Avg Ollama RAM (MB)", color="#6366F1", edgecolor="#4338CA")
    b2 = ax.bar(x + width / 2, peak_ram, width=width, label="Peak Ollama RAM (MB)", color="#EC4899", edgecolor="#BE185D")

    ax.set_title("Ollama Memory Footprint Comparison (MB)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Megabytes (MB)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in b1:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in b2:
        h = bar.get_height()
        ax.annotate(f"{h:.1f} MB", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_charts(summaries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generates all individual charts separately and returns a dictionary of their paths."""
    charts_dir = get_charts_dir()

    throughput_path = os.path.join(charts_dir, "throughput_comparison.png")
    latency_path = os.path.join(charts_dir, "latency_comparison.png")
    reliability_path = os.path.join(charts_dir, "reliability_pie_chart.png")
    memory_path = os.path.join(charts_dir, "memory_footprint.png")

    plot_throughput(summaries, throughput_path)
    plot_latency(summaries, latency_path)
    plot_reliability_pie(summaries, reliability_path)
    plot_memory(summaries, memory_path)

    return {
        "throughput": throughput_path,
        "latency": latency_path,
        "reliability": reliability_path,
        "memory": memory_path,
    }
