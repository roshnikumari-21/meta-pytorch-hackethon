"""Unit tests for deterministic policy, grader, and environment."""

from __future__ import annotations

import os

import pytest

from env.env import CorporateExpenseEnvironment  # noqa: E402
from env.grader import episode_score  # noqa: E402
from env.models import CorporateExpenseAction, TrajectoryStep  # noqa: E402
from env.policy import ground_truth_for_expense  # noqa: E402
from env.tasks import HARD_EXPENSES, MEDIUM_EXPENSES  # noqa: E402


def test_ground_truth_duplicate_hard() -> None:
    gt0 = ground_truth_for_expense(
        HARD_EXPENSES[0], task="hard", prior_in_episode=[]
    )
    assert gt0.decision == "approve"
    gt1 = ground_truth_for_expense(
        HARD_EXPENSES[1], task="hard", prior_in_episode=[HARD_EXPENSES[0]]
    )
    assert gt1.decision == "reject"
    assert "duplicate_claim" in gt1.fraud_flags


def test_medium_missing_receipt() -> None:
    gt = ground_truth_for_expense(
        MEDIUM_EXPENSES[2], task="medium", prior_in_episode=[]
    )
    assert gt.decision == "reject"


def test_easy_episode_high_score(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPORATE_EXPENSE_TASK", "easy")
    env = CorporateExpenseEnvironment()
    obs = env.reset(task="easy")
    while not obs.done:
        idx = obs.current_expense_index
        cur = obs.pending_expenses[idx]
        prior = obs.pending_expenses[:idx]
        gt = ground_truth_for_expense(cur, task=obs.task, prior_in_episode=prior)
        obs = env.step(
            CorporateExpenseAction(
                decision=gt.decision,
                reason="Compliant with policy; receipt and amount are reasonable.",
            )
        )
    assert obs.episode_score is not None
    assert obs.episode_score >= 0.85


def test_episode_score_deterministic() -> None:
    traj = [
        TrajectoryStep(
            expense_id="x",
            decision="approve",
            reason="Within policy and receipt attached.",
            ground_truth_decision="approve",
            fraud_flags=(),
            reward=0.9,
        )
    ]
    a = episode_score(traj)
    b = episode_score(traj)
    assert a == b


def test_reset_task_kw_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORPORATE_EXPENSE_TASK", "easy")
    env = CorporateExpenseEnvironment()
    obs = env.reset(task="hard")
    assert obs.task == "hard"
    assert len(obs.pending_expenses) == len(HARD_EXPENSES)


def test_step_advances_and_done() -> None:
    os.environ["CORPORATE_EXPENSE_TASK"] = "easy"
    env = CorporateExpenseEnvironment()
    env.reset()
    n = len(env._expenses)
    for i in range(n):
        cur = env._expenses[env._cursor]
        prior = env._expenses[: env._cursor]
        gt = ground_truth_for_expense(cur, task=env._task, prior_in_episode=prior)
        obs = env.step(
            CorporateExpenseAction(decision=gt.decision, reason="test reasoning")
        )
        assert obs.current_expense_index == i + 1
        if i + 1 == n:
            assert obs.done
            assert obs.episode_score is not None
