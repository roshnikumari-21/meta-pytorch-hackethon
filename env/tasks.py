"""Deterministic task definitions: easy, medium, hard."""

from __future__ import annotations

from env.models import ExpenseRecord


def _e(
    eid: str,
    amount: float,
    category: str,
    receipt: bool,
    description: str,
    ts: str,
) -> ExpenseRecord:
    return ExpenseRecord(
        id=eid,
        amount=amount,
        category=category,
        receipt_provided=receipt,
        description=description,
        timestamp=ts,
    )


# Easy: small, receipted, no conflicts
EASY_EXPENSES: list[ExpenseRecord] = [
    _e("E1", 32.5, "meals", True, "Coffee for interview candidate", "2024-03-01T09:00:00Z"),
    _e("E2", 18.0, "parking", True, "Parking near main office", "2024-03-01T11:30:00Z"),
    _e("E3", 44.0, "supplies", True, "Printer paper ream", "2024-03-02T14:00:00Z"),
]

# Medium: missing receipts on moderate amounts, stricter policy path
MEDIUM_EXPENSES: list[ExpenseRecord] = [
    _e("M1", 120.0, "meals", True, "Team lunch downtown", "2024-04-10T12:00:00Z"),
    _e("M2", 65.0, "meals", False, "Working lunch with vendor", "2024-04-11T12:30:00Z"),
    _e("M3", 40.0, "supplies", False, "Adapters for conference room", "2024-04-12T09:00:00Z"),
    _e("M4", 95.0, "software", True, "Monthly SaaS subscription", "2024-04-13T00:00:00Z"),
]

# Hard: duplicates, high-value anomalies, entertainment edge cases
HARD_EXPENSES: list[ExpenseRecord] = [
    _e("H1", 220.0, "meals", True, "Team dinner after release", "2024-05-01T19:00:00Z"),
    _e("H2", 220.0, "meals", True, "Team dinner after release", "2024-05-02T19:05:00Z"),
    _e("H3", 9500.0, "equipment", True, "Bulk laptop order", "2024-05-03T10:00:00Z"),
    _e("H4", 600.0, "travel", False, "Client dinner — no receipt scanned", "2024-05-04T20:00:00Z"),
    _e("H5", 42.0, "meals", True, "Breakfast before onsite", "2024-05-05T07:30:00Z"),
]


TASK_EXPENSES: dict[str, list[ExpenseRecord]] = {
    "easy": EASY_EXPENSES,
    "medium": MEDIUM_EXPENSES,
    "hard": HARD_EXPENSES,
}


def get_task_expenses(task: str) -> list[ExpenseRecord]:
    key = task.lower().strip()
    if key not in TASK_EXPENSES:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_EXPENSES)}")
    return list(TASK_EXPENSES[key])
