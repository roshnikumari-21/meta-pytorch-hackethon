"""
Deterministic step rewards and episode scores.

Step reward is shaped per problem statement and clamped to [0, 1].
Episode score aggregates correctness, reasoning quality, and fraud handling.
"""

from __future__ import annotations

import re
from typing import Iterable

from env.models import StepRewardBreakdown, TrajectoryStep
from env.policy import GroundTruth, ground_truth_for_expense

_FRAUD_KEYWORDS = (
    "duplicate",
    "receipt",
    "missing",
    "fraud",
    "suspicious",
    "anomal",
    "policy",
    "violation",
    "high",
    "amount",
    "travel",
    "equipment",
)


def _reasoning_quality(reason: str, flags: tuple[str, ...]) -> float:
    """0..0.2 based on length and keyword relevance."""
    r = reason.strip()
    if len(r) < 12:
        return 0.0
    score = 0.0
    if len(r) >= 40:
        score += 0.1
    else:
        score += 0.05
    low = r.lower()
    if flags:
        if any(kw in low for kw in _FRAUD_KEYWORDS):
            score += 0.1
    else:
        if any(kw in low for kw in ("policy", "receipt", "valid", "appropriate", "within")):
            score += 0.1
        else:
            score += 0.05
    return min(0.2, score)


def _fraud_component(
    decision: str,
    gt: GroundTruth,
    reason: str,
) -> float:
    """0..0.2 fraud-detection credit."""
    if not gt.fraud_flags:
        return 0.2
    low = reason.lower()
    mentioned = any(kw in low for kw in _FRAUD_KEYWORDS)
    correct_reject = decision == "reject" and gt.decision == "reject"
    if correct_reject and (mentioned or "duplicate" in gt.fraud_flags):
        return 0.2
    if correct_reject:
        return 0.15
    if mentioned:
        return 0.08
    return 0.0


def compute_step_reward(
    *,
    decision: str,
    reason: str,
    gt: GroundTruth,
) -> tuple[float, StepRewardBreakdown]:
    """Return clamped [0,1] reward and breakdown."""
    breakdown = StepRewardBreakdown()
    total = 0.0

    correct = decision == gt.decision
    if correct:
        total += 0.5
        breakdown.correct = 0.5
    else:
        total -= 0.3
        breakdown.incorrect_penalty = -0.3

    rq = _reasoning_quality(reason, gt.fraud_flags)
    total += rq
    breakdown.reasoning = rq

    fc = _fraud_component(decision, gt, reason)
    total += fc
    breakdown.fraud = fc

    risky = (
        decision == "approve"
        and gt.decision == "reject"
        and bool(gt.fraud_flags)
    )
    if risky:
        total -= 0.2
        breakdown.risky_approval_penalty = -0.2

    breakdown.raw_total = total
    clamped = max(0.0, min(1.0, total))
    return clamped, breakdown


def episode_score(trajectory: Iterable[TrajectoryStep]) -> float:
    """
    Deterministic aggregate in [0, 1].

    Combines mean per-step correctness, mean reasoning proxy from reasons,
    and fraud recall when fraud was present in ground truth.
    """
    steps = list(trajectory)
    if not steps:
        return 0.0

    n = len(steps)
    correct = sum(
        1 for s in steps if s.decision == s.ground_truth_decision
    ) / n

    reasoning = []
    for s in steps:
        gt = GroundTruth(decision=s.ground_truth_decision, fraud_flags=s.fraud_flags)
        reasoning.append(_reasoning_quality(s.reason, s.fraud_flags))
    reasoning_mean = sum(reasoning) / n / 0.2  # normalize to ~[0,1]

    fraud_steps = [s for s in steps if s.fraud_flags]
    if not fraud_steps:
        fraud_score = 1.0
    else:
        hits = 0
        for s in fraud_steps:
            low = s.reason.lower()
            ok = s.decision == "reject" and s.ground_truth_decision == "reject"
            mentioned = any(kw in low for kw in _FRAUD_KEYWORDS)
            if ok and mentioned:
                hits += 1
            elif ok:
                hits += 0.7
        fraud_score = hits / len(fraud_steps)

    # Fixed convex combination (deterministic)
    score = 0.45 * correct + 0.30 * min(1.0, reasoning_mean) + 0.25 * fraud_score
    return max(0.0, min(1.0, round(score, 4)))
