"""
models.py — Typed Pydantic models for the SQL Data Quality Environment.

Action:       Agent submits a structured action (tool call + arguments).
Observation:  Environment returns execution results, current DB state, feedback.
State:        Episode-level metadata (task_id, step, score, etc.).
StepResult:   Wrapper used by HTTP endpoints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    LIST_TABLES = "list_tables"
    DESCRIBE_TABLE = "describe_table"
    QUERY = "query"
    SUBMIT_FIX = "submit_fix"
    FINISH = "finish"


class Action(BaseModel):
    """An action the agent can take in the SQL Data Quality environment."""

    action_type: ActionType = Field(
        ...,
        description=(
            "Type of action. One of: list_tables, describe_table, query, "
            "submit_fix, finish."
        ),
    )
    # Used by describe_table and submit_fix
    table_name: Optional[str] = Field(
        None, description="Target table name (for describe_table or submit_fix)."
    )
    # Used by query — a raw SQL SELECT / WITH query (read-only)
    sql: Optional[str] = Field(
        None,
        description=(
            "A read-only SQL query (SELECT / WITH) to inspect the database. "
            "DDL / DML is not allowed here; use submit_fix instead."
        ),
    )
    # Used by submit_fix — one or more SQL fix statements
    fix_sql: Optional[str] = Field(
        None,
        description=(
            "SQL fix statement(s) to apply (UPDATE / DELETE / ALTER). "
            "Each statement must end with a semicolon."
        ),
    )
    # Optional free-text reasoning (does not affect grader score)
    reasoning: Optional[str] = Field(
        None, description="Optional chain-of-thought reasoning (not graded)."
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """Result returned to the agent after each step."""

    done: bool = Field(False, description="True when the episode has ended.")
    reward: float = Field(0.0, description="Step reward (0.0 – 1.0).")
    observation_text: str = Field(
        "", description="Human-readable description of what happened."
    )
    # Structured payload — varies by action type
    data: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Structured data payload. Schema depends on action_type: "
            "list_tables → {tables:[str]}, "
            "describe_table → {columns:[{name,type,nullable}], sample_rows:[dict]}, "
            "query → {columns:[str], rows:[list], row_count:int}, "
            "submit_fix → {affected_rows:int, validation:dict}, "
            "finish → {final_score:float, breakdown:dict}."
        ),
    )
    error: Optional[str] = Field(
        None, description="Error message if the action failed."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(BaseModel):
    """Episode-level state exposed via the /state endpoint."""

    episode_id: str = Field(..., description="Unique identifier for this episode.")
    task_id: str = Field(..., description="Active task key (easy / medium / hard).")
    step_count: int = Field(0, description="Number of steps taken so far.")
    max_steps: int = Field(30, description="Maximum steps allowed per episode.")
    cumulative_reward: float = Field(
        0.0, description="Reward accumulated so far in this episode."
    )
    issues_found: int = Field(
        0, description="Number of data quality issues identified by the agent."
    )
    fixes_applied: int = Field(
        0, description="Number of fix statements successfully applied."
    )
    task_description: str = Field(
        "", description="Natural-language description of the current task."
    )
    available_tables: List[str] = Field(
        default_factory=list,
        description="Tables available in the current task database.",
    )


# ---------------------------------------------------------------------------
# StepResult (HTTP envelope)
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Envelope returned by the /step and /reset HTTP endpoints."""

    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)
