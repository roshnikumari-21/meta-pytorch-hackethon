"""
Deterministic rule-based policy: ground-truth approve/reject and fraud flags.

This is NOT an LLM: same expense + history always yields the same labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from env.models import ExpenseRecord


@dataclass(frozen=True)
class GroundTruth:
    decision: str  # "approve" | "reject"
    fraud_flags: tuple[str, ...]


def _norm_desc(s: str) -> str:
    return " ".join(s.lower().strip().split())


def ground_truth_for_expense(
    expense: ExpenseRecord,
    *,
    task: str,
    prior_in_episode: list[ExpenseRecord],
) -> GroundTruth:
    """
    Compute canonical decision and fraud/anomaly flags for grading.

    prior_in_episode: expenses already decided (lower index than current).
    """
    flags: list[str] = []

    # Duplicate claim: same normalized description and amount as an earlier item
    nd = _norm_desc(expense.description)
    for prev in prior_in_episode:
        if abs(prev.amount - expense.amount) < 1e-6 and _norm_desc(prev.description) == nd:
            flags.append("duplicate_claim")
            return GroundTruth(decision="reject", fraud_flags=tuple(flags))

    # Missing receipt on non-trivial spend
    if not expense.receipt_provided and expense.amount >= 75.0:
        flags.append("missing_receipt")
        return GroundTruth(decision="reject", fraud_flags=tuple(flags))

    # High-value anomaly without receipt
    if expense.amount >= 2500.0 and not expense.receipt_provided:
        flags.append("high_value_no_receipt")
        return GroundTruth(decision="reject", fraud_flags=tuple(flags))

    # Suspicious high spend even with receipt (policy cap simulation)
    if expense.amount >= 8000.0:
        flags.append("high_value_suspicious")
        return GroundTruth(decision="reject", fraud_flags=tuple(flags))

    # Category-specific policy
    if expense.category.lower() in {"travel", "equipment"} and expense.amount > 1500.0:
        if not expense.receipt_provided:
            flags.append("category_high_value_no_receipt")
            return GroundTruth(decision="reject", fraud_flags=tuple(flags))

    # Task-specific tightening
    if task == "medium":
        if not expense.receipt_provided and expense.amount >= 40.0:
            flags.append("medium_tier_receipt_required")
            return GroundTruth(decision="reject", fraud_flags=tuple(flags))

    if task == "hard":
        if expense.amount >= 500.0 and "client" in nd and not expense.receipt_provided:
            flags.append("client_entertainment_no_receipt")
            return GroundTruth(decision="reject", fraud_flags=tuple(flags))

    return GroundTruth(decision="approve", fraud_flags=tuple(flags))
