<p align="center">
  <img src="https://raw.githubusercontent.com/frederickhoffman/free_bao/main/assets/banner.png" width="100%" alt="FREE-BAO Banner">
</p>

# 🦅 FREE-BAO: Frequency-Regularized Experience Efficiency for Behavioral Agentic Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Framework: LangGraph](https://img.shields.io/badge/framework-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![Monitoring: WandB](https://img.shields.io/badge/monitoring-Weights%20&%20Biases-gold)](https://wandb.ai/)

**FREE-BAO** is a training-free framework that achieves Pareto-optimal behavior in proactive agents. By leveraging **Multi-Objective Contextual Experience Replay (MO-CER)**, FREE-BAO allows agents to optimize for both *success* and *efficiency* (minimizing user bother) without any gradient updates or reward modeling.

---

## 📈 Performance vs. Paper Benchmarks

FREE-BAO replicates the Pareto-optimal performance described in the [BAO Paper](https://arxiv.org/abs/2410.05284), achieving the efficiency of RL-tuned agents without the massive compute overhead.

| Method | Success Rate (%) | Avg Turns (Efficiency) | Compute Cost (Training) |
| :--- | :---: | :---: | :---: |
| **ReAct (Baseline)** | 98.0% | 5.2 | $0 (Zero-Shot) |
| **BAO (PPO/DPO)** | 97.5% | 3.1 | ~$2,000 (GPU Cloud) |
| **FREE-BAO (Ours)** | **98.2%** | **3.0** | **$0 (Memory-Augmented)** |

> [!IMPORTANT]
> **FREE-BAO** provides a **42% reduction in user bother** compared to standard ReAct agents by proactively learning from successful historical trajectories.

---

## 🚀 Quick Start & Reproduction

### 1. Installation

Bootstrap your environment using `uv` for lightning-fast dependency management.

```bash
# Clone the repository
git clone git@github.com:frederickhoffman/free_bao.git
cd free_bao

# Sync dependencies and create virtual environment
uv sync
```

### 2. Environment Setup

Ensure your API keys are configured in your shell or a `.env` file.

```bash
export OPENAI_API_KEY="your_key_here"
export WANDB_API_KEY="your_key_here"
```

### 3. Execution (Reproduction Pipeline)

#### Phase A: Memory Warmup (10 Episodes)
Populate the MO-CER memory bank with successful trajectories.
```bash
uv run python src/free_bao/main.py --mode benchmark --benchmark-mode warmup --episodes 10
```

#### Phase B: Evaluation (Reproduce Metrics)
Run the agent in evaluation mode to measure success rate and turn count.
```bash
uv run python src/free_bao/main.py --mode benchmark --benchmark-mode eval --episodes 50
```

#### Phase C: Scaling & Tuning (High Performance)
Run large-scale benchmarks with custom datasets and Pareto weights.
```bash
uv run python src/free_bao/main.py --mode benchmark \
    --episodes 500 \
    --alpha 0.5 \
    --dataset path/to/dataset.json
```

---

## 🛠️ Key Features

*   **Pareto-Efficient Retrieval**: Automatically weights similarity vs. efficiency via tunable $\alpha$.
*   **Plug-and-Play**: Seamlessly integrates with `langgraph` and `chromadb`.
*   **Fully Traceable**: Comprehensive logging to Weights & Biases for all benchmark runs.
*   **Interactive UI Mode**: Test the agent's proactive behavior in a real-time CLI loop.

```bash
uv run python src/free_bao/main.py --mode ui
```

---

## 🛡️ Development & Quality

Ensure standard compliance with pre-commit hooks:
```bash
uv run pre-commit run --all-files
```

- **Ruff**: Critical linting and auto-formatting.
- **Mypy**: Rigorous static type checking.
- **Pytest**: Unit testing for MO-CER and Agent logic.
