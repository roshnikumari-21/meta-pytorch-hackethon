---
title: Corporate Expense OpenEnv
emoji: 🏦
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# 🏦 Corporate Expense Approval — OpenEnv Environment

> **Meta × PyTorch Hackathon Submission**  
> A production-style [OpenEnv](https://github.com/openenv-org/openenv-core) RL environment where an LLM agent acts as a corporate auditor — reviewing a batch of employee expenses one by one, deciding to **approve** or **reject** each claim while citing receipts, duplicates, and policy violations.

---

## 📋 Table of Contents

- [What This Project Does](#-what-this-project-does)
- [Architecture Overview](#-architecture-overview)
- [Project Structure](#-project-structure)
- [Quickstart](#-quickstart)
- [Environment Reference](#-environment-reference)
- [Running the LLM Agent (inference.py)](#-running-the-llm-agent-inferencepy)
- [REST API Reference](#-rest-api-reference)
- [Tests](#-tests)
- [Deploying to Hugging Face Spaces](#-deploying-to-hugging-face-spaces)
- [Validating Your Submission](#-validating-your-submission)
- [Reward System Deep Dive](#-reward-system-deep-dive)
- [License](#-license)

---

## 🎯 What This Project Does

Finance teams route thousands of expense claims through policy checks and fraud screens every day. This project distills that workflow into a **multi-step reinforcement learning problem**:

- An LLM agent receives a batch of `ExpenseRecord` objects one at a time
- For each expense it must output a JSON `{"decision": "approve"|"reject", "reason": "..."}` 
- A **deterministic grader** scores every decision based on correctness, reasoning quality, and fraud-detection skill
- At the end of the episode an aggregate `episode_score` in **[0, 1]** is emitted

The server implements the full **OpenEnv HTTP protocol** (FastAPI + uvicorn), so any OpenEnv-compatible agent can connect to it locally, via Docker, or as a live **Hugging Face Space**.

---

## 🏗 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                     inference.py                     │
│  (Async LLM agent: OpenAI-compatible API → actions)  │
└────────────────────────┬─────────────────────────────┘
                         │  OpenEnv WebSocket/HTTP
                         ▼
┌──────────────────────────────────────────────────────┐
│               server/app.py  (FastAPI)               │
│  POST /reset   POST /step   GET /state   GET /health │
└────────────────────────┬─────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   env/env.py        │  CorporateExpenseEnvironment
              │   env/policy.py     │  Deterministic ground-truth
              │   env/grader.py     │  Step reward + episode score
              │   env/tasks.py      │  Fixed expense fixtures
              │   env/models.py     │  Pydantic schemas
              └─────────────────────┘
```

---

## 📁 Project Structure

```
.
├── env/
│   ├── env.py          # CorporateExpenseEnvironment (reset / step / state)
│   ├── models.py       # Pydantic models: Action, Observation, State, ExpenseRecord
│   ├── policy.py       # Deterministic ground-truth policy (what the correct answer is)
│   ├── grader.py       # Step reward computation + episode_score aggregation
│   └── tasks.py        # Fixed expense lists for easy / medium / hard
├── server/
│   └── app.py          # FastAPI app: /reset, /step, /state, /health endpoints
├── scripts/
│   └── validate-submission.sh   # End-to-end HF + Docker + openenv validator
├── tests/
│   └── test_env.py     # Pytest unit tests (policy, grader, episode flow)
├── models.py           # Root re-export of env.models (used by OpenEnv tooling)
├── client.py           # Async WebSocket client (CorporateExpenseEnv)
├── inference.py        # LLM agent entry point
├── openenv.yaml        # OpenEnv manifest
├── Dockerfile          # Container for local runs & HF Spaces
├── pyproject.toml      # Build system + dependencies
└── uv.lock             # Reproducible lock file (uv)
```

---

## ⚡ Quickstart

### Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | ≥ 3.10 | [python.org](https://python.org) |
| Docker | any | [docs.docker.com](https://docs.docker.com/get-docker/) |
| uv *(optional but recommended)* | latest | `pip install uv` |

---

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/meta-pytorch-hackathon.git
cd meta-pytorch-hackathon

# Install with dev dependencies (pytest, etc.)
pip install -e ".[dev]"
```

Or with **uv** (faster, fully reproducible):

```bash
uv sync --extra dev
```

---

### 2. Run the Environment Server Locally

```bash
# Option A — via uv entry point
uv run server

# Option B — direct uvicorn
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Verify it's alive:

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

### 3. Run via Docker

```bash
# Build the image
docker build -t corporate-expense-openenv:latest .

# Run the server
docker run --rm -p 8000:8000 corporate-expense-openenv:latest
```

---

### 4. Run the LLM Agent

Set the required environment variables, then run:

```bash
export IMAGE_NAME="corporate-expense-openenv:latest"
export HF_TOKEN="hf_..."                          # Your Hugging Face token
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export CORPORATE_EXPENSE_TASK="medium"            # easy | medium | hard

python inference.py
```

**Sample output:**

```
[START] task=medium env=corporate_expense_approval model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=expense_action({"decision": "approve", "reason": "..."}) reward=0.90 done=false error=null
[STEP] step=2 action=expense_action({"decision": "reject", "reason": "Missing receipt..."}) reward=0.80 done=false error=null
[STEP] step=3 action=expense_action({"decision": "reject", "reason": "..."}) reward=0.70 done=false error=null
[STEP] step=4 action=expense_action({"decision": "approve", "reason": "..."}) reward=0.90 done=true error=null
[END] success=true steps=4 score=0.88 rewards=0.90,0.80,0.70,0.90
```

---

## 🌍 Environment Reference

### Tasks

| Task | Expenses | Focus |
|------|----------|-------|
| `easy` | 3 claims | Small amounts, all receipted — basic approve path |
| `medium` | 4 claims | Missing receipts at moderate amounts; stricter policy |
| `hard` | 5 claims | Duplicate lines, very high amounts, travel/entertainment edge cases |

Select via `reset(task="hard")` or the `CORPORATE_EXPENSE_TASK` env var.

---

### Action Space — `CorporateExpenseAction`

```json
{
  "decision": "approve",
  "reason": "Amount is within policy limits and receipt is attached."
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `decision` | `"approve"` \| `"reject"` | ✅ | Outcome for the **current** expense |
| `reason` | `str` | ✅ | Short justification (min ~12 chars for partial credit) |
| `metadata` | `dict` | ❌ | Optional OpenEnv metadata |

---

### Observation Space — `CorporateExpenseObservation`

```json
{
  "pending_expenses": [...],
  "current_expense_index": 1,
  "task": "medium",
  "done": false,
  "reward": 0.70,
  "episode_score": null,
  "last_action_error": null,
  "step_reward_detail": {
    "total": 0.70,
    "breakdown": {
      "correct": 0.5,
      "incorrect_penalty": 0.0,
      "reasoning": 0.15,
      "fraud": 0.05,
      "risky_approval_penalty": 0.0,
      "raw_total": 0.70
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `pending_expenses` | Full list of `ExpenseRecord` for the episode |
| `current_expense_index` | 0-based index of the expense to act on next |
| `task` | Active task difficulty |
| `done` | `true` when all expenses have been processed |
| `reward` | Per-step reward in **[0, 1]** |
| `episode_score` | Final aggregate score in **[0, 1]** (only when `done=true`) |
| `last_action_error` | Server-side validation message, if any |
| `step_reward_detail` | Typed breakdown (`ExpenseStepReward`) |

---

### ExpenseRecord Schema

```json
{
  "id": "M2",
  "amount": 65.0,
  "category": "meals",
  "receipt_provided": false,
  "description": "Working lunch with vendor",
  "timestamp": "2024-04-11T12:30:00Z"
}
```

---

## 🤖 Running the LLM Agent (`inference.py`)

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `IMAGE_NAME` | *(required)* | Docker image tag for the environment server |
| `HF_TOKEN` | *(required)* | Hugging Face API key (also accepted as `API_KEY`) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | OpenAI-compatible API base |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model ID for chat completions |
| `CORPORATE_EXPENSE_TASK` | `easy` | Task difficulty: `easy` / `medium` / `hard` |

### How It Works

1. Spins up the Docker env server via `CorporateExpenseEnv.from_docker_image(IMAGE_NAME)`
2. Calls `env.reset(task=TASK_NAME)` to start an episode
3. For each step, formats the current expense + prior rows into a prompt
4. Sends the prompt to the LLM via OpenAI-compatible chat completions API
5. Parses the JSON response into a `CorporateExpenseAction`
6. Sends the action to `env.step(action)` and logs the reward
7. Repeats until `done=true`, then prints the final `[END]` line

### Using a Different LLM Provider

Any OpenAI-compatible provider works — just change `API_BASE_URL` and `MODEL_NAME`:

```bash
# Meta Llama via HF Inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
export HF_TOKEN="hf_..."

# OpenAI (GPT-4o)
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o"
export HF_TOKEN="sk-..."  # OpenAI key also accepted via HF_TOKEN
```

---

## 🔌 REST API Reference

The FastAPI server exposes the standard OpenEnv HTTP protocol:

| Endpoint | Method | Body | Description |
|----------|--------|------|-------------|
| `/health` | `GET` | — | Liveness check → `{"status": "ok"}` |
| `/reset` | `POST` | `{"task": "easy"}` | Start a new episode |
| `/step` | `POST` | `{"decision": "...", "reason": "..."}` | Submit an action |
| `/state` | `GET` | — | Get current episode state |

### Example: Manual curl session

```bash
# 1. Start an episode
curl -s -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard"}' | python -m json.tool

# 2. Submit a decision
curl -s -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"decision": "approve", "reason": "Small amount within policy, receipt attached."}' | python -m json.tool

# 3. Check state
curl -s http://localhost:8000/state | python -m json.tool
```

### Using the Python Client

```python
import asyncio
from client import CorporateExpenseEnv
from models import CorporateExpenseAction

async def run():
    # Connect to locally running server
    env = CorporateExpenseEnv(base_url="http://localhost:8000")

    # Or spin up a fresh Docker container automatically
    env = await CorporateExpenseEnv.from_docker_image("corporate-expense-openenv:latest")

    result = await env.reset(task="medium")
    obs = result.observation

    while not result.done:
        action = CorporateExpenseAction(
            decision="approve",
            reason="Receipt present and amount within policy.",
        )
        result = await env.step(action)
        print(f"Reward: {result.reward:.2f}  Done: {result.done}")

    print(f"Episode score: {result.observation.episode_score:.4f}")
    await env.close()

asyncio.run(run())
```

---

## 🧪 Tests

```bash
# Run all tests
pytest tests/test_env.py -v

# Run a specific test
pytest tests/test_env.py::test_easy_episode_high_score -v
```

### Test Coverage

| Test | What It Checks |
|------|---------------|
| `test_ground_truth_duplicate_hard` | Duplicate expense H2 is correctly flagged as `reject` |
| `test_medium_missing_receipt` | M3 (no receipt, moderate amount) → `reject` |
| `test_easy_episode_high_score` | Oracle policy on `easy` achieves ≥ 0.85 episode score |
| `test_episode_score_deterministic` | Same trajectory always yields the same score |
| `test_reset_task_kw_overrides_env` | `reset(task="hard")` overrides the env var |
| `test_step_advances_and_done` | Cursor advances correctly; `done` triggers at last step |

---

## 🚀 Deploying to Hugging Face Spaces

This project is designed to run as a **Hugging Face Space** (Docker SDK). Follow these steps:

### Step 1 — Create a New Space

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. **Space name**: e.g. `corporate-expense-openenv`
3. **SDK**: Select **Docker**
4. **Visibility**: Public *(required for hackathon submission)*
5. Click **Create Space**

---

### Step 2 — Authenticate & Push the Repository

> ⚠️ **HF no longer supports password authentication over HTTPS.** You must use a **User Access Token** instead.

#### Get a Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token** → Role: **Write** → copy the `hf_...` value

#### Option A — Embed token in the remote URL *(quick)*

```bash
# Add remote with token baked into the URL
git remote add hf https://<your-username>:<hf_token>@huggingface.co/spaces/<your-username>/corporate-expense-openenv

# Push main branch (HF Spaces auto-deploys from main)
git push hf main
```

If you already added the remote without a token, update it first:

```bash
git remote set-url hf https://<your-username>:<hf_token>@huggingface.co/spaces/<your-username>/corporate-expense-openenv
git push hf main
```

#### Option B — HF CLI credential helper *(cleaner, token stored securely)*

```bash
pip install huggingface_hub
huggingface-cli login   # prompts for your token; stores it in ~/.cache/huggingface

# Now plain HTTPS push works, no token in URL
git remote add hf https://huggingface.co/spaces/<your-username>/corporate-expense-openenv
git push hf main
```

> **Tip:** Push to both GitHub and your HF Space:
> ```bash
> git push origin main   # GitHub
> git push hf main       # HF Space (triggers a rebuild)
> ```

---

### Step 3 — Set Space Secrets (if needed)

If your server needs secrets (e.g. `CORPORATE_EXPENSE_TASK`):

1. Open your Space → **Settings** → **Repository secrets**
2. Add key-value pairs; they become environment variables inside the container

---

### Step 4 — Verify the Space is Live

```bash
# Replace with your actual Space URL
curl -s -X POST https://<your-username>-corporate-expense-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{}' | python -m json.tool
```

A `200 OK` with a JSON observation confirms everything is working.

---

### Step 5 — Point `inference.py` at Your Live Space

```bash
export IMAGE_NAME="corporate-expense-openenv:latest"   # still needed for docker client
export HF_TOKEN="hf_..."
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export CORPORATE_EXPENSE_TASK="hard"

python inference.py
```

---

### HF Spaces — How the Dockerfile Works

The `Dockerfile` at the repo root is what HF Spaces builds and runs:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md openenv.yaml ./
COPY models.py client.py ./
COPY env ./env
COPY server ./server
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s ... CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

HF Spaces automatically maps port `7860` externally to your exposed port. The `openenv.yaml` manifest tells OpenEnv tooling to use port `8000`:

```yaml
spec_version: 1
name: corporate_expense_approval
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

---

## ✅ Validating Your Submission

Use the included shell script to run all 3 pre-submission checks:

```bash
chmod +x scripts/validate-submission.sh

./scripts/validate-submission.sh \
  https://<your-username>-corporate-expense-openenv.hf.space \
  .
```

**What it checks:**

| Step | Check |
|------|-------|
| 1/3 | HF Space is live and `/reset` returns HTTP 200 |
| 2/3 | `docker build` completes successfully |
| 3/3 | `openenv validate --verbose` passes |

All three must pass before submitting to the hackathon.

---

## 📊 Reward System Deep Dive

### Per-Step Reward (clamped to [0, 1])

| Component | Value | Condition |
|-----------|-------|-----------|
| **Correct decision** | +0.50 | `decision == ground_truth` |
| **Wrong decision** | −0.30 | `decision != ground_truth` |
| **Reasoning quality** | +0.00 to +0.20 | Length ≥ 40 chars (+0.10) + policy/fraud keywords (+0.10) |
| **Fraud detection** | +0.00 to +0.20 | Correct rejection + fraud keywords mentioned |
| **Risky approval** | −0.20 | Approved a claim that should be rejected **and** has fraud flags |

### Episode Score (deterministic aggregate in [0, 1])

```
score = 0.45 × correctness
      + 0.30 × reasoning_quality
      + 0.25 × fraud_recall
```

- **Correctness**: fraction of steps with correct decisions
- **Reasoning quality**: mean reasoning score normalized to [0, 1]
- **Fraud recall**: fraction of fraud-flagged expenses correctly rejected with keywords

### Baseline Scores

| Agent | Easy | Medium | Hard |
|-------|------|--------|------|
| **Oracle (ground-truth)** | 1.00 | 1.00 | 1.00 |
| Always-approve | ~0.30 | ~0.20 | ~0.15 |
| Random | ~0.35 | ~0.35 | ~0.30 |
| **LLM baseline (Qwen 72B)** | ~0.85 | ~0.75 | ~0.65 |

> Score ≥ 0.55 is considered `success=true` in `inference.py`.

---

## 🪪 License

BSD-3-Clause — consistent with OpenEnv core. See your organization's submission requirements for attribution details.
