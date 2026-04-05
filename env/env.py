"""Corporate expense approval OpenEnv Environment implementation."""

from __future__ import annotations

import os
import uuid
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

from env.grader import compute_step_reward, episode_score
from env.models import (
    CorporateExpenseAction,
    CorporateExpenseObservation,
    CorporateExpenseState,
    ExpenseRecord,
    ExpenseStepReward,
    TrajectoryStep,
)
from env.policy import ground_truth_for_expense
from env.tasks import get_task_expenses


class CorporateExpenseEnvironment(
    Environment[CorporateExpenseAction, CorporateExpenseObservation, CorporateExpenseState]
):
    """Multi-step expense queue with deterministic rewards and grading."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__(transform=None, rubric=None)
        self._episode_id: Optional[str] = None
        self._step_count = 0
        self._task = "easy"
        self._expenses: list[ExpenseRecord] = []
        self._cursor = 0
        self._trajectory: list[TrajectoryStep] = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: object,
    ) -> CorporateExpenseObservation:
        self._reset_rubric()
        self._episode_id = episode_id or str(uuid.uuid4())
        self._step_count = 0
        self._cursor = 0
        self._trajectory = []

        task_kw = kwargs.get("task")
        if isinstance(task_kw, str) and task_kw.strip():
            self._task = task_kw.strip().lower()
        else:
            self._task = os.environ.get("CORPORATE_EXPENSE_TASK", "easy").strip().lower()

        try:
            self._expenses = get_task_expenses(self._task)
        except ValueError:
            self._task = "easy"
            self._expenses = get_task_expenses("easy")

        return self._build_observation(
            done=False, reward=0.0, err=None, ep_score=None, reward_detail=None
        )

    def step(
        self,
        action: CorporateExpenseAction,
        timeout_s: Optional[float] = None,
        **kwargs: object,
    ) -> CorporateExpenseObservation:
        del timeout_s, kwargs
        if self._cursor >= len(self._expenses):
            return self._build_observation(
                done=True,
                reward=0.0,
                err="episode already complete",
                ep_score=episode_score(self._trajectory),
                reward_detail=None,
            )

        current = self._expenses[self._cursor]
        prior = self._expenses[: self._cursor]
        gt = ground_truth_for_expense(current, task=self._task, prior_in_episode=prior)

        reward, breakdown = compute_step_reward(
            decision=action.decision,
            reason=action.reason,
            gt=gt,
        )
        reward_detail = ExpenseStepReward(total=reward, breakdown=breakdown)

        self._trajectory.append(
            TrajectoryStep(
                expense_id=current.id,
                decision=action.decision,
                reason=action.reason,
                ground_truth_decision=gt.decision,  # type: ignore[arg-type]
                fraud_flags=gt.fraud_flags,
                reward=reward,
            )
        )

        self._cursor += 1
        self._step_count += 1

        done = self._cursor >= len(self._expenses)
        ep_score: Optional[float] = None
        if done:
            ep_score = episode_score(self._trajectory)

        return self._build_observation(
            done=done,
            reward=reward,
            err=None,
            ep_score=ep_score,
            reward_detail=reward_detail,
        )

    @property
    def state(self) -> CorporateExpenseState:
        return CorporateExpenseState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task=self._task,
            total_expenses=len(self._expenses),
            processed_count=self._cursor,
            episode_complete=self._cursor >= len(self._expenses) and len(self._expenses) > 0,
        )

    def _build_observation(
        self,
        *,
        done: bool,
        reward: float,
        err: Optional[str],
        ep_score: Optional[float],
        reward_detail: Optional[ExpenseStepReward],
    ) -> CorporateExpenseObservation:
        return CorporateExpenseObservation(
            pending_expenses=list(self._expenses),
            current_expense_index=self._cursor,
            task=self._task,
            done=done,
            reward=reward,
            episode_score=ep_score,
            last_action_error=err,
            step_reward_detail=reward_detail,
            metadata={
                "processed_expenses": self._cursor,
                "total_expenses": len(self._expenses),
            },
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="corporate_expense_approval",
            description="Corporate expense approval with fraud and policy compliance",
            version="1.0.0",
        )
