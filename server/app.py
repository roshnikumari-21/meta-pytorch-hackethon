# Copyright (c) Contributors to the corporate expense OpenEnv environment.
"""FastAPI application exposing the corporate expense approval environment."""

from __future__ import annotations

from openenv.core.env_server import create_app

from env.env import CorporateExpenseEnvironment
from models import CorporateExpenseAction, CorporateExpenseObservation

app = create_app(
    CorporateExpenseEnvironment,
    CorporateExpenseAction,
    CorporateExpenseObservation,
    env_name="corporate_expense_approval",
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
