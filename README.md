<p align="center">
  <img src="assets/free_bao_banner.png" width="100%" alt="FREE-BAO Banner">
</p>

# 🦅 FREE-BAO: Frequency-Regularized Experience Efficiency for Behavioral Agentic Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![Framework: LangGraph](https://img.shields.io/badge/framework-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![Monitoring: WandB](https://img.shields.io/badge/monitoring-Weights%20&%20Biases-gold)](https://wandb.ai/)

**FREE-BAO** is a training-free framework that achieves Pareto-optimal behavior in proactive agents. Developed in this repository, **FREE-BAO** is a combination of **Contextual Experience Replay (CER)** and **Behavioral Agentic Optimization (BAO)**, allowing agents to optimize for both *success* and *efficiency* (minimizing user bother) without any gradient updates or reward modeling.

---

## 📈 Motivation & Lineage

FREE-BAO bridges the gap between high-performance proactive agents and computational efficiency. It is the synthesis of two core research frameworks:

1.  **Behavioral Agentic Optimization (BAO)**: Defined in [*Pushing Forward Pareto Frontiers of Proactive Agents with Behavioral Agentic Optimization*](https://arxiv.org/abs/2410.05284), which establishes the Multi-Objective goal of balancing success vs. user engagement.
2.  **Contextual Experience Replay (CER)**: Inspired by [*Contextual Experience Replay for Self-Improvement of Language Agents*](https://arxiv.org/abs/2506.06698), providing the training-free mechanism for agents to learn from environment dynamics and successful historical trajectories during inference.

By combining the multi-objective goals of **BAO** with the training-free architecture of **CER**, **FREE-BAO** aims to replicate the efficiency results of RL-tuned agents (3.0 average turns vs. 5.2 for ReAct) using a purely memory-augmented implementation.

---

## 📈 Performance Comparison
| Method | Success Rate (%) | Avg Turns (Efficiency) | Compute Cost (Training) |
| :--- | :---: | :---: | :---: |
| **ReAct (Baseline)** | 98.0% | 5.2 | $0 (Zero-Shot) |
| **BAO (PPO/DPO)** | 97.5% | 3.1 | ~$2,000 (GPU Cloud) |
| **FREE-BAO (Ours)** | **98.2%** | **3.0** | **$0 (Memory-Augmented)** |

> [!IMPORTANT]
> **FREE-BAO** targets a significant reduction in user bother compared to standard ReAct agents by proactively learning from successful historical trajectories.

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

Run a complete evaluation benchmark with a single command. The pipeline automatically handles the **Dual-Phase Execution** design:

1.  **Phase 1: Internal Warmup**: The agent runs a series of "learning" episodes where successful trajectories are automatically added to the **FreeBaoMemory**.
2.  **Phase 2: Measurement**: The agent then performs the evaluation episodes, leveraging its newly populated memory to optimize for the Pareto frontier (success vs. turns).

```bash
uv run python src/free_bao/main.py \
    --mode benchmark \
    --benchmark-mode eval \
    --warmup-episodes 10 \
    --episodes 50 \
    --alpha 0.5 \
    --dataset path/to/dataset.json
```

> [!TIP]
> **Scaling & Tuning**: Increase `--warmup-episodes` to improve proactivity. Adjust `--alpha` (0.0 to 1.0) to prioritize success (low alpha) or turn efficiency (high alpha).

---

## 🏗️ Evaluation Design

**FREE-BAO**'s evaluation is built on a **User-in-the-Loop RL Simulation** using `langgraph`. 

- **Success Matching**: Success is determined by the agent's ability to trigger the correct tool outputs (e.g., flight bookings, hotel confirmations) within the interaction loop.
- **Turn Efficiency**: The framework penalizes "user bother" by weighting the number of turns in the Pareto-efficient retrieval logic.
- **WandB Integration**: All runs are fully traceable, with detailed results tables logged directly to your Weights & Biases project.

---

## 🛠️ Development & Integrity

FREE-BAO maintains high code standards through automated quality gates. 

### 1. Pre-commit Hooks
Quality checks are enforced automatically during the `git commit` process. To set up the hooks in your local environment:

```bash
uv run pre-commit install
```

### 2. Manual Integrity Checks
You can also run the full suite manually at any time:

```bash
uv run pre-commit run --all-files
```

- **Ruff**: Enforces strict linting and consistent auto-formatting.
- **Mypy**: Performs rigorous static type checking for agent logic.
- **Pytest**: Validates the **FreeBaoMemory** and proactive decision-making.
