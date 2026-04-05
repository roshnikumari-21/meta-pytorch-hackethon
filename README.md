# Corporate Expense Approval (OpenEnv)

Production-style OpenEnv environment where an agent reviews a batch of employee expenses sequentially: approve or reject each line, justify the decision, and surface fraud or policy issues (duplicates, missing receipts, anomalous amounts).

## Real-world motivation

Finance operations teams route thousands of claims through policy checks and fraud screens. This environment distills that workflow into a multi-step RL problem: **decision quality**, **audit-grade reasoning**, and **fraud/policy compliance** under a fixed, inspectable rule set.

## Action space

Pydantic `CorporateExpenseAction` (extends OpenEnv `Action`):

| Field | Type | Description |
|--------|------|-------------|
| `decision` | `"approve"` \| `"reject"` | Outcome for the **current** expense (`current_expense_index`). |
| `reason` | `str` | Short justification (receipts, duplicates, limits, etc.). |
| `metadata` | `dict` | Optional OpenEnv metadata (usually empty). |

## Observation space

Pydantic `CorporateExpenseObservation`:

| Field | Description |
|--------|-------------|
| `pending_expenses` | Full list of `ExpenseRecord` in the episode (id, amount, category, receipt_provided, description, timestamp). |
| `current_expense_index` | Index of the expense to act on next (0-based). |
| `task` | `easy`, `medium`, or `hard`. |
| `done` | `true` when every expense has been processed. |
| `reward` | Scalar step reward in **[0, 1]** after each `step`. |
| `episode_score` | Deterministic **[0, 1]** aggregate when `done` is `true` (otherwise `null`). |
| `last_action_error` | Server-side validation message, if any. |
| `step_reward_detail` | Typed `ExpenseStepReward` (total + `StepRewardBreakdown`). |

## State (`state()` / `/state`)

`CorporateExpenseState` extends OpenEnv `State` with `task`, `total_expenses`, `processed_count`, and `episode_complete`.

## Tasks

| Task | Focus |
|------|--------|
| **easy** | Small, receipted claims; basic approve path. |
| **medium** | Missing receipts at moderate amounts; stricter receipt rules. |
| **hard** | Duplicate lines, very high amounts, travel/client edge cases. |

Select with `reset(task="hard")` or environment variable `CORPORATE_EXPENSE_TASK`.

## Reward logic (per step)

Rewards are **continuous in [0, 1]** (clamped). Components (see `env/grader.py`):

- **+0.5** if the decision matches deterministic ground truth.
- **−0.3** if the decision is wrong.
- **Up to +0.2** for reasoning quality (length + policy/fraud-relevant wording).
- **Up to +0.2** for fraud-detection alignment when fraud signals are present.
- **−0.2** for **risky approval** (approving when ground truth is reject and fraud flags exist).

Typed breakdown is returned as `ExpenseStepReward` on the observation.

## Episode score (grader)

After the last expense, `episode_score` is a **deterministic** aggregate in **[0, 1]** from the full trajectory (`env/grader.py::episode_score`): mix of correctness, reasoning proxy, and fraud-handling recall. Same transcript always yields the same score (no sampling).

## Baseline scores (oracle policy)

With ground-truth decisions and strong reasons, the reference policy achieves **1.00** on `easy`, `medium`, and `hard` episode scores (computed with the shipped rule set).

Random or always-approve agents score much lower; your `inference.py` LLM baseline will land between these depending on the model and prompts.

## Setup

### Python

```bash
pip install -e ".[dev]"
```

### Validate (OpenEnv)

```bash
openenv validate --verbose
```

Requires `uv.lock` (regenerate with `uv lock` after dependency changes).

### Local server

```bash
uv run server
# or
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Health: `GET http://localhost:8000/health`  
HF Space ping (validator): `POST /reset` with JSON body `{}`.

### Docker

```bash
docker build -t corporate-expense-openenv:latest .
docker run --rm -p 8000:8000 corporate-expense-openenv:latest
```

Use with:

```python
from client import CorporateExpenseEnv

env = await CorporateExpenseEnv.from_docker_image("corporate-expense-openenv:latest")
```

### Inference (LLM agent)

Set:

- `IMAGE_NAME` — Docker image for the environment server  
- `HF_TOKEN` (or `API_KEY`)  
- `API_BASE_URL`, `MODEL_NAME`  
- Optional: `CORPORATE_EXPENSE_TASK` (`easy` / `medium` / `hard`)

```bash
python inference.py
```

Stdout follows the hackathon format: `[START]`, one `[STEP]` per `step`, then `[END]` (always, including failures).

## Tests

```bash
pytest tests/test_env.py -v
```

## Layout

| Path | Role |
|------|------|
| `env/env.py` | `CorporateExpenseEnvironment` (`reset`, `step`, `state`). |
| `env/models.py` | Pydantic models including `ExpenseStepReward`. |
| `env/policy.py` | Deterministic ground-truth policy. |
| `env/grader.py` | Step reward + episode score. |
| `env/tasks.py` | Fixed expense lists per task. |
| `models.py` / `client.py` | OpenEnv root modules for tooling and imports. |
| `server/app.py` | `create_app(...)` + `main()` for `uvicorn`. |
| `openenv.yaml` | OpenEnv manifest. |
| `Dockerfile` | Container for Spaces / local runs. |

## License

BSD-3-Clause–style (consistent with OpenEnv core); see your organization’s submission requirements for attribution.
