"""
client.py — Python client for the SQL Data Quality Environment.

Provides a synchronous client (SQLDataQualityClient) that wraps the HTTP API,
following the OpenEnv EnvClient pattern.

Example:
    from client import SQLDataQualityClient, Action, ActionType

    with SQLDataQualityClient(base_url="http://localhost:7860") as client:
        obs = client.reset(task_id="easy")
        result = client.step(Action(action_type=ActionType.LIST_TABLES))
        state = client.get_state()
"""

from __future__ import annotations

from typing import Optional

import requests

from models import Action, Observation, State, StepResult


class SQLDataQualityClient:
    """
    Synchronous HTTP client for the SQL Data Quality Environment.

    Compatible with the OpenEnv EnvClient interface.
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._session.close()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Observation:
        """Reset the environment and start a new episode."""
        payload = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        if episode_id is not None:
            payload["episode_id"] = episode_id
        resp = self._session.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return Observation(**resp.json())

    def step(self, action: Action) -> StepResult:
        """Execute one action and return the step result."""
        resp = self._session.post(
            f"{self.base_url}/step",
            json=action.model_dump(exclude_none=True),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return StepResult(
            observation=Observation(**data["observation"]),
            reward=data["reward"],
            done=data["done"],
            info=data.get("info", {}),
        )

    def get_state(self) -> State:
        """Return the current episode state."""
        resp = self._session.get(f"{self.base_url}/state", timeout=10)
        resp.raise_for_status()
        return State(**resp.json())

    def health(self) -> dict:
        """Check server health."""
        resp = self._session.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def list_tasks(self) -> dict:
        """Return all available tasks."""
        resp = self._session.get(f"{self.base_url}/tasks", timeout=10)
        resp.raise_for_status()
        return resp.json()
