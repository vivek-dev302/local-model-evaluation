# 🛡️ Local AI Support Triage Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Inference-Ollama-black?logo=ollama&logoColor=white)](https://ollama.ai/)
[![Pydantic](https://img.shields.io/badge/Validation-Pydantic%20v2-e92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Hardware](https://img.shields.io/badge/Tested%20On-Ryzen%205%207520U%20(CPU%20Only)-orange)](#-test-hardware)
[![Offline](https://img.shields.io/badge/Offline-100%25%20Local-success)](#)

A 100% offline, privacy-first AI customer support triage engine powered by local Small Language Models (SLMs). It classifies incoming support tickets, determines urgency and routing, drafts empathetic responses, and enforces deterministic JSON output via **Pydantic** with an automated **self-healing retry loop**.

Includes an empirical benchmark comparing **`phi4-mini:3.8b`**, **`qwen2.5:3b`**, and **`llama3.2:3b`** on commodity CPU-only laptop hardware.

---

### 💻 Test Hardware
All benchmarks were performed on a consumer budget laptop with **zero dedicated GPU**:
- **CPU:** AMD Ryzen 5 7520U (4 Cores / 8 Threads, up to 4.3 GHz)
- **RAM:** 16 GB LPDDR5
- **GPU:** AMD Radeon 610M (512 MB Integrated VRAM — **CPU-only inference**)
- **OS:** Windows 11

---

## ⚡ Key Benchmark Results

> 📊 Detailed breakdown with per-temperature analysis available in **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**.

Evaluated over **50 real-world customer tickets** at temperature 0.0 & 0.7:

| Model | Size | Throughput (tok/s) | Avg Latency | Schema Reliability | Category Accuracy | Judge Quality | Ollama RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **qwen2.5:3b** | 3.0B | **8.40 tok/s** | **11.3s** | 100% | 76.0% | 88.6% | **2.2 GB** |
| **phi4-mini:3.8b** | 3.8B | 6.70 tok/s | 14.7s | **100% (0 retries)** | **84.0%** | **89.8%** | 2.7–3.3 GB |
| **llama3.2:3b** | 3.0B | 7.63 tok/s | 14.5s | 98–100% | 78.0% | 89.4% | 2.6 GB |

<p align="center">
  <img src="reports/charts/throughput_comparison.png" width="48%" alt="Throughput" />
  <img src="reports/charts/accuracy_comparison.png" width="48%" alt="Accuracy" />
</p>

### TL;DR Takeaways
- **Speed & Efficiency**: `qwen2.5:3b` is the throughput leader (**8.4 tok/s**, lowest latency) with the smallest RAM footprint (**2.2 GB**).
- **Decision Quality**: `phi4-mini:3.8b` achieves the highest routing accuracy (**84%**) and flawless **100% first-pass schema compliance** without ever needing a retry.
- **Feasibility on Budget Hardware**: Even on an entry-level Ryzen 5 laptop without a dGPU, 3B SLMs generate responses at a usable **7–8+ tokens/second**.

---

## 🏗️ How It Works

```
Customer Ticket ──► Ollama (Local SLM) ──► Pydantic JSON Validation
                                                     │
                             ┌─── Failure ◄──────────┴──────────► Success
                             ▼                                      │
                   Self-Healing Retry Prompt                        ▼
                   (re-prompts with error trace)         Structured Triage Output
                                                                    │
                                                         LLM-as-a-Judge Eval
                                                        (Flow-Judge Scoring)
```

1. **Structured Output**: Models are prompted with a strict JSON contract (`intent`, `priority`, `sentiment`, `requires_human`, `category`, `suggested_response`).
2. **Self-Healing Loop**: If output is invalid JSON, the validator catches the error and automatically reprompts the model with the exact validation message.
3. **Automated LLM Judge**: `Flow-Judge-v0.1` runs locally to evaluate response empathy, actionability, and conciseness on a 1–5 scale.

---

## 🚀 Quickstart

### 1. Setup Environment
```bash
git clone https://github.com/your-username/local-ai-assistant.git
cd local-ai-assistant

python -m venv .venv
.venv\Scripts\activate      # Windows (or: source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

### 2. Pull Models in Ollama
Make sure [Ollama](https://ollama.ai/) is running, then pull candidate models and the evaluation judge:
```bash
# SLMs being benchmarked
ollama pull phi4-mini:3.8b
ollama pull qwen2.5:3b
ollama pull llama3.2:3b

# Evaluator Judge (runs offline)
ollama pull hf.co/flowaicom/Flow-Judge-v0.1-GGUF:Q4_K_M
```

### 3. Run Benchmark (`app.main`)
Select the model in `.env`:
```env
MODEL_NAME=phi4-mini:3.8b
TEMPERATURES=0.0,0.7
```
Run the test suite (50 cases across both temperatures):
```bash
python -m app.main
```
> **To test other models**: change `MODEL_NAME` in `.env` (e.g., `qwen2.5:3b`) and re-run `python -m app.main`.

### 4. Compare & Generate Reports (`app.compare`)
```bash
python -m app.compare
```
This automatically:
- Runs `Flow-Judge` on all completed model outputs.
- Scores accuracy against ground truth.
- Generates charts in `reports/charts/`.
- Updates **[BENCHMARK_REPORT.md](BENCHMARK_REPORT.md)**.

---

## 📂 Project Structure

```text
├── BENCHMARK_REPORT.md     # Full benchmark tables and deep analysis
├── test_cases.json         # 50 labeled ground-truth customer tickets
├── system_prompt/          # Triage & LLM-Judge system prompts
├── app/
│   ├── main.py             # Runs benchmark on active model in .env
│   ├── compare.py          # Runs judge, generates charts, builds report
│   └── scripts/            # Modular evaluation, metrics, and visualization logic
└── reports/
    ├── charts/             # Generated comparison charts (.png)
    └── <model_name>/       # Raw benchmark logs (.jsonl)
```

---

## 📄 License
MIT
