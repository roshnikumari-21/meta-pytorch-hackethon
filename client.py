"""WebSocket client for the corporate expense approval environment."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import (
    CorporateExpenseAction,
    CorporateExpenseObservation,
    CorporateExpenseState,
    ExpenseRecord,
    ExpenseStepReward,
)


class CorporateExpenseEnv(
    EnvClient[CorporateExpenseAction, CorporateExpenseObservation, CorporateExpenseState]
):
    """Async client; use ``await CorporateExpenseEnv.from_docker_image(...)`` for containers."""

    def _step_payload(self, action: CorporateExpenseAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[CorporateExpenseObservation]:
        obs_data = payload.get("observation", {}) or {}
        raw_expenses = obs_data.get("pending_expenses", [])
        expenses = [ExpenseRecord.model_validate(x) for x in raw_expenses]
        srd_raw = obs_data.get("step_reward_detail")
        step_reward_detail = (
            ExpenseStepReward.model_validate(srd_raw) if srd_raw else None
        )
        observation = CorporateExpenseObservation(
            pending_expenses=expenses,
            current_expense_index=int(obs_data.get("current_expense_index", 0)),
            task=str(obs_data.get("task", "easy")),
            episode_score=obs_data.get("episode_score"),
            last_action_error=obs_data.get("last_action_error"),
            step_reward_detail=step_reward_detail,
            done=bool(payload.get("done", False)),
            reward=payload.get("reward"),
            metadata={},
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> CorporateExpenseState:
        return CorporateExpenseState.model_validate(payload)
