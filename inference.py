"""
Inference script for the corporate expense approval OpenEnv environment.

Environment variables (see README):
    API_BASE_URL   LLM API base URL (OpenAI-compatible).
    MODEL_NAME     Model id for chat completions.
    HF_TOKEN       API key (falls back to API_KEY).
    IMAGE_NAME     Docker image for CorporateExpenseEnv.from_docker_image().
    CORPORATE_EXPENSE_TASK   Task id: easy | medium | hard (optional).

STDOUT FORMAT (do not change):
    [START] task=<task> env=<env> model=<model>
    [STEP] step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, List, Optional, Tuple

from openai import OpenAI

from client import CorporateExpenseEnv
from models import CorporateExpenseAction

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("CORPORATE_EXPENSE_TASK", os.getenv("EXPENSE_TASK", "easy"))
BENCHMARK = os.getenv("CORPORATE_EXPENSE_BENCHMARK", "corporate_expense_approval")
MAX_STEPS = 64
TEMPERATURE = 0.3
MAX_TOKENS = 400
SUCCESS_SCORE_THRESHOLD = 0.55


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a corporate expense approver. For each claim you must output ONE JSON object only,
    with keys:
      "decision": "approve" or "reject"
      "reason": a concise justification citing receipts, duplicates, policy limits, or fraud cues.

    Rules of thumb:
    - Approve only clearly compliant small claims with documentation.
    - Reject missing receipts when amounts are material, duplicate line items, or suspicious high spend.
    - Mention specific cues (duplicate, receipt, policy, suspicious amount) when they apply.

    Output JSON only. No markdown fences.
    """
).strip()


def _format_observation_for_prompt(obs: Any) -> str:
    idx = obs.current_expense_index
    lines = [
        f"Task: {obs.task}",
        f"Current expense index: {idx} (0-based)",
        f"Total in batch: {len(obs.pending_expenses)}",
    ]
    if idx < len(obs.pending_expenses):
        cur = obs.pending_expenses[idx]
        lines.append("Current expense:")
        lines.append(
            json.dumps(
                {
                    "id": cur.id,
                    "amount": cur.amount,
                    "category": cur.category,
                    "receipt_provided": cur.receipt_provided,
                    "description": cur.description,
                    "timestamp": cur.timestamp,
                },
                indent=2,
            )
        )
    lines.append("Earlier rows in this batch (for duplicate detection):")
    for i, e in enumerate(obs.pending_expenses):
        if i >= idx:
            break
        lines.append(
            f"  - [{e.id}] ${e.amount:.2f} | {e.category} | receipt={e.receipt_provided} | {e.description!r}"
        )
    return "\n".join(lines)


def _parse_decision_json(text: str) -> Tuple[str, str]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    data = json.loads(text)
    decision = str(data.get("decision", "")).strip().lower()
    reason = str(data.get("reason", "")).strip()
    if decision not in ("approve", "reject"):
        raise ValueError(f"invalid decision: {decision!r}")
    return decision, reason


def get_model_action(client: OpenAI, user_block: str) -> CorporateExpenseAction:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_block},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        decision, reason = _parse_decision_json(raw)
        return CorporateExpenseAction(decision=decision, reason=reason)
    except Exception as exc:
        print(f"[DEBUG] Model parse failed, using safe reject: {exc}", flush=True)
        return CorporateExpenseAction(
            decision="reject",
            reason="Fallback: could not produce a valid JSON decision; rejecting for safety.",
        )


def action_log_string(action: CorporateExpenseAction) -> str:
    inner = json.dumps(
        {"decision": action.decision, "reason": action.reason}, ensure_ascii=False
    )
    return f"expense_action({inner})"


async def main() -> None:
    if not IMAGE_NAME:
        raise RuntimeError("IMAGE_NAME must be set for from_docker_image()")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = await CorporateExpenseEnv.from_docker_image(IMAGE_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    last_obs: Any = None
    try:
        result = await env.reset(task=TASK_NAME)
        last_err: Optional[str] = None

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                last_obs = result.observation
                break

            obs = result.observation
            user_block = _format_observation_for_prompt(obs)
            action = get_model_action(client, user_block)
            action_str = action_log_string(action)

            result = await env.step(action)
            obs = result.observation
            last_obs = obs
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            last_err = obs.last_action_error

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=last_err)

            if done:
                break

        if last_obs is not None and last_obs.episode_score is not None:
            score = float(last_obs.episode_score)
        elif rewards:
            score = sum(rewards) / len(rewards)
        else:
            score = 0.0
        score = max(0.0, min(1.0, score))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        success = False
        if not rewards:
            score = 0.0
    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
