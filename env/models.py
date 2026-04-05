"""Pydantic models for the corporate expense approval environment."""

from __future__ import annotations

from typing import Any, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


class ExpenseRecord(BaseModel):
    """Single expense line item visible to the agent."""

    model_config = {"extra": "forbid"}

    id: str = Field(..., description="Stable expense identifier")
    amount: float = Field(..., ge=0.0, description="Claimed amount in USD")
    category: str = Field(..., description="Expense category")
    receipt_provided: bool = Field(..., description="Whether a receipt was attached")
    description: str = Field(..., description="Free-text justification")
    timestamp: str = Field(..., description="ISO-8601 timestamp string")


class CorporateExpenseAction(Action):
    """Agent decision for the current expense."""

    decision: Literal["approve", "reject"] = Field(
        ..., description="Approval outcome for the current expense"
    )
    reason: str = Field(
        default="",
        description="Short justification referencing policy, receipts, or fraud cues",
    )


class StepRewardBreakdown(BaseModel):
    """Structured breakdown for debugging and grading (optional metadata)."""

    model_config = {"extra": "forbid"}

    correct: float = 0.0
    reasoning: float = 0.0
    fraud: float = 0.0
    incorrect_penalty: float = 0.0
    risky_approval_penalty: float = 0.0
    raw_total: float = 0.0


class ExpenseStepReward(BaseModel):
    """Typed reward payload for one step (total + decomposition)."""

    model_config = {"extra": "forbid"}

    total: float = Field(..., ge=0.0, le=1.0, description="Clamped step reward")
    breakdown: StepRewardBreakdown = Field(
        default_factory=StepRewardBreakdown,
        description="Additive components before clamping",
    )


class CorporateExpenseObservation(Observation):
    """Observation after reset or step."""

    pending_expenses: list[ExpenseRecord] = Field(
        default_factory=list,
        description="All expenses in the episode (agent uses index to know current)",
    )
    current_expense_index: int = Field(
        default=0,
        ge=0,
        description="Index into pending_expenses for the expense to decide next",
    )
    task: str = Field(default="easy", description="Task difficulty key")
    episode_score: Optional[float] = Field(
        default=None,
        description="Deterministic [0,1] episode score when done is true",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Populated when an internal validation error occurs",
    )
    step_reward_detail: Optional[ExpenseStepReward] = Field(
        default=None,
        description="Structured reward for the last transition (None on reset)",
    )


class CorporateExpenseState(State):
    """Server-side session state exposed via /state."""

    task: str = Field(default="easy", description="Active task id")
    total_expenses: int = Field(default=0, ge=0)
    processed_count: int = Field(default=0, ge=0)
    episode_complete: bool = Field(default=False)


class TrajectoryStep(BaseModel):
    """One row of the episode transcript for grading."""

    model_config = {"extra": "forbid"}

    expense_id: str
    decision: Literal["approve", "reject"]
    reason: str
    ground_truth_decision: Literal["approve", "reject"]
    fraud_flags: tuple[str, ...] = ()
    reward: float = 0.0
