# Model Comparison Benchmark Study

This report evaluates local small language models (SLMs) on offline customer triage tasks using Ollama.

---

## 1. Throughput & Latency Performance

<p align="center">
  <img src="reports/charts/throughput_comparison.png" width="49%" alt="Inference Throughput" />
  <img src="reports/charts/latency_comparison.png" width="49%" alt="Latency Profile" />
</p>

| Model          | Temp | Tokens/sec | Avg TTFT (s) | Avg Latency (s) | Avg Out Tokens |
| -------------- | ---- | ---------- | ------------ | --------------- | -------------- |
| llama3.2_3b    | 0.0  | 7.19       | 3.288        | 15.21           | 108.3          |
| llama3.2_3b    | 0.7  | 5.98       | 3.599        | 18.66           | 111.7          |
| phi4-mini_3.8b | 0.0  | 6.16       | 3.513        | 16.39           | 98.2           |
| phi4-mini_3.8b | 0.7  | 5.86       | 3.121        | 16.88           | 98.0           |

---

## 2. Reliability & Schema Adherence

![Reliability Pie Charts](reports/charts/reliability_pie_chart.png)

| Model          | Temp | Tests | 1st Pass | Retry Pass | Failed | Pass Rate (%) |
| -------------- | ---- | ----- | -------- | ---------- | ------ | ------------- |
| llama3.2_3b    | 0.0  | 50    | 46       | 3          | 1      | 98.0%         |
| llama3.2_3b    | 0.7  | 50    | 46       | 4          | 0      | 100.0%        |
| phi4-mini_3.8b | 0.0  | 50    | 50       | 0          | 0      | 100.0%        |
| phi4-mini_3.8b | 0.7  | 50    | 50       | 0          | 0      | 100.0%        |

---

## 3. Decision Quality & Accuracy (Ground Truth)

![Accuracy Comparison](reports/charts/accuracy_comparison.png)

| Model          | Temp | Overall Accuracy (%) | Category Routing (%) | Escalation Match (%) |
| -------------- | ---- | -------------------- | -------------------- | -------------------- |
| llama3.2_3b    | 0.0  | 73.4%                | 72.0%                | 88.0%                |
| llama3.2_3b    | 0.7  | 75.2%                | 76.0%                | 92.0%                |
| phi4-mini_3.8b | 0.0  | 77.2%                | 82.0%                | 84.0%                |
| phi4-mini_3.8b | 0.7  | 76.8%                | 84.0%                | 84.0%                |

*Overall Accuracy is a unified weighted composite score calibrated across triage requirements: 40% Department Routing (Category), 30% Human Escalation (`requires_human`), 20% Queue Priority, and 10% Customer Sentiment.*

---

## 4. Resource Footprint (RAM Usage)

![Memory Footprint](reports/charts/memory_footprint.png)

| Model          | Temp | Avg Ollama RAM | Peak Ollama RAM | Avg System RAM |
| -------------- | ---- | -------------- | --------------- | -------------- |
| llama3.2_3b    | 0.0  | 105.5 MB       | 106.0 MB        | 74.2%          |
| llama3.2_3b    | 0.7  | 238.2 MB       | 271.1 MB        | 86.1%          |
| phi4-mini_3.8b | 0.0  | 92.1 MB        | 225.4 MB        | 73.4%          |
| phi4-mini_3.8b | 0.7  | 76.8 MB        | 78.9 MB         | 78.2%          |

---

## Key Observations

- **Decision Quality & Accuracy**:
  `phi4-mini:3.8b` leads with **~77% overall weighted triage accuracy** and **82–84% category routing**, compared to `llama3.2:3b` (**~73–75% overall accuracy** and **72–76% category routing**).
- **Throughput vs Parameter Size**:
  `llama3.2:3b` achieved higher throughput (~7.19 tokens/sec at T=0.0) compared to `phi4-mini:3.8b` (~6.16 tokens/sec).
- **Schema Reliability & Determinism**:
  `phi4-mini:3.8b` demonstrated superior schema adherence, passing **100% of test cases on the first attempt (50/50)** with **0 retries** across both temperature settings. `llama3.2:3b` required retry reprompts to repair JSON schemas.
- **Temperature Impact**:
  Higher temperature (`temp=0.7`) slightly degraded throughput and increased total latency across both models due to increased stochastic token exploration.
- **Memory Footprint**:
  Both models stayed within compact local memory limits (<250 MB Ollama process RAM), making them suitable for constrained edge deployments.
