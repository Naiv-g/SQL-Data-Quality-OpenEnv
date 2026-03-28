"""
environment.py — Core environment logic for the SQL Data Quality environment.

Implements reset(), step(), and state property following the OpenEnv spec.
Uses an in-memory SQLite database per episode so each reset() is fully clean.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Dict, Optional

from models import Action, ActionType, Observation, State, StepResult
from tasks import TASKS, Task, _exec_ddl


# ---------------------------------------------------------------------------
# Reward shaping constants
# ---------------------------------------------------------------------------

# Partial reward given each time the agent successfully executes a fix
REWARD_FIX_APPLIED = 0.05
# Penalty per step to encourage efficiency (very small)
REWARD_STEP_PENALTY = -0.005
# Bonus for finishing under half the step budget
REWARD_EFFICIENCY_BONUS = 0.10
# Penalty for exceeding max_steps
REWARD_TIMEOUT_PENALTY = -0.05


class SQLDataQualityEnvironment:
    """
    OpenEnv-compatible environment for SQL Data Quality tasks.

    State is held entirely in an in-memory SQLite database that is
    re-created on every reset() call, guaranteeing clean episodes.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._task: Optional[Task] = None
        self._state: State = State(episode_id="", task_id="", step_count=0)

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        task_id: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> Observation:
        """
        Start a new episode.

        Args:
            task_id: One of 'easy', 'medium', 'hard'. Defaults to 'easy'.
            seed: Optional random seed (currently unused – task is deterministic).
            episode_id: Optional explicit episode ID.

        Returns:
            Initial Observation with task description.
        """
        if task_id not in TASKS:
            task_id = "easy"

        self._task = TASKS[task_id]
        eid = episode_id or str(uuid.uuid4())

        # Build fresh in-memory SQLite
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        _exec_ddl(self._conn, self._task.schema_sql)

        self._state = State(
            episode_id=eid,
            task_id=task_id,
            step_count=0,
            max_steps=self._task.max_steps,
            cumulative_reward=0.0,
            issues_found=0,
            fixes_applied=0,
            task_description=self._task.description,
            available_tables=list(self._task.tables),
        )

        return Observation(
            done=False,
            reward=0.0,
            observation_text=(
                f"Episode started. Task: {task_id.upper()}.\n\n"
                + self._task.description
                + f"\n\nAvailable tables: {', '.join(self._task.tables)}"
            ),
            data={"task_id": task_id, "tables": self._task.tables},
        )

    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResult:
        """
        Execute one agent action and return (observation, reward, done, info).

        Args:
            action: Typed Action from the agent.

        Returns:
            StepResult with observation, reward, done flag, and info dict.
        """
        if self._task is None or self._conn is None:
            obs = Observation(
                done=True,
                reward=0.0,
                observation_text="Environment not initialised. Call reset() first.",
                error="not_initialised",
            )
            return StepResult(observation=obs, reward=0.0, done=True, info={})

        self._state.step_count += 1
        step_reward = REWARD_STEP_PENALTY  # small per-step penalty

        # ---- Route by action type ----------------------------------------
        if action.action_type == ActionType.LIST_TABLES:
            obs, extra_reward = self._act_list_tables()

        elif action.action_type == ActionType.DESCRIBE_TABLE:
            obs, extra_reward = self._act_describe_table(action.table_name)

        elif action.action_type == ActionType.QUERY:
            obs, extra_reward = self._act_query(action.sql)

        elif action.action_type == ActionType.SUBMIT_FIX:
            obs, extra_reward = self._act_submit_fix(action.fix_sql)

        elif action.action_type == ActionType.FINISH:
            obs, extra_reward = self._act_finish()

        else:
            obs = Observation(
                done=False,
                reward=0.0,
                observation_text=f"Unknown action type: {action.action_type}",
                error="unknown_action",
            )
            extra_reward = 0.0

        # ---- Accumulate reward -------------------------------------------
        step_reward += extra_reward
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + step_reward, 4
        )

        # ---- Check step budget -------------------------------------------
        if not obs.done and self._state.step_count >= self._state.max_steps:
            obs.done = True
            obs.observation_text += (
                f"\n\n⚠️  Step budget exhausted ({self._state.max_steps} steps). "
                "Episode ending automatically."
            )
            step_reward += REWARD_TIMEOUT_PENALTY
            obs.reward = step_reward

        obs.reward = round(step_reward, 4)

        info: Dict[str, Any] = {
            "step_count": self._state.step_count,
            "cumulative_reward": self._state.cumulative_reward,
            "fixes_applied": self._state.fixes_applied,
        }

        return StepResult(
            observation=obs,
            reward=obs.reward,
            done=obs.done,
            info=info,
        )

    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return the current episode state."""
        return self._state

    # ------------------------------------------------------------------
    # Private action handlers
    # ------------------------------------------------------------------

    def _act_list_tables(self):
        cur = self._conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]
        obs = Observation(
            done=False,
            reward=0.0,
            observation_text=f"Tables in database: {', '.join(tables)}",
            data={"tables": tables},
        )
        return obs, 0.0

    # ------------------------------------------------------------------

    def _act_describe_table(self, table_name: Optional[str]):
        if not table_name:
            obs = Observation(
                done=False,
                reward=0.0,
                observation_text="describe_table requires a table_name.",
                error="missing_table_name",
            )
            return obs, 0.0

        cur = self._conn.cursor()
        try:
            cur.execute(f"PRAGMA table_info({table_name})")
            cols = [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "nullable": not r["notnull"],
                    "default": r["dflt_value"],
                    "pk": bool(r["pk"]),
                }
                for r in cur.fetchall()
            ]
            if not cols:
                raise ValueError(f"Table '{table_name}' not found.")

            cur.execute(f"SELECT * FROM {table_name} LIMIT 5")
            rows = cur.fetchall()
            sample = [dict(r) for r in rows]

            cur.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
            total = cur.fetchone()["cnt"]

            obs = Observation(
                done=False,
                reward=0.0,
                observation_text=(
                    f"Table '{table_name}': {len(cols)} columns, {total} rows.\n"
                    + "\n".join(
                        f"  - {c['name']} ({c['type']}, nullable={c['nullable']})"
                        for c in cols
                    )
                ),
                data={"columns": cols, "sample_rows": sample, "total_rows": total},
            )
            return obs, 0.0

        except Exception as e:
            obs = Observation(
                done=False,
                reward=0.0,
                observation_text=f"describe_table error: {e}",
                error=str(e),
            )
            return obs, 0.0

    # ------------------------------------------------------------------

    def _act_query(self, sql: Optional[str]):
        if not sql:
            obs = Observation(
                done=False,
                reward=0.0,
                observation_text="query requires sql parameter.",
                error="missing_sql",
            )
            return obs, 0.0

        # Safety: only allow read-only queries
        normalized = sql.strip().upper()
        forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "REPLACE")
        for kw in forbidden:
            if normalized.startswith(kw) or f"\n{kw}" in normalized:
                obs = Observation(
                    done=False,
                    reward=0.0,
                    observation_text=(
                        "Only read-only SELECT/WITH queries are allowed via 'query'. "
                        "Use 'submit_fix' for data modification."
                    ),
                    error="write_op_in_query",
                )
                return obs, 0.0

        cur = self._conn.cursor()
        try:
            cur.execute(sql)
            rows = cur.fetchall()
            if rows:
                col_names = [d[0] for d in cur.description]
                data_rows = [list(r) for r in rows]
            else:
                col_names = []
                data_rows = []

            obs = Observation(
                done=False,
                reward=0.0,
                observation_text=(
                    f"Query returned {len(data_rows)} rows."
                    if data_rows
                    else "Query returned 0 rows."
                ),
                data={
                    "columns": col_names,
                    "rows": data_rows[:50],  # cap at 50
                    "row_count": len(data_rows),
                },
            )
            return obs, 0.0

        except Exception as e:
            obs = Observation(
                done=False,
                reward=0.0,
                observation_text=f"Query error: {e}",
                error=str(e),
            )
            return obs, 0.0

    # ------------------------------------------------------------------

    def _act_submit_fix(self, fix_sql: Optional[str]):
        if not fix_sql:
            obs = Observation(
                done=False,
                reward=0.0,
                observation_text="submit_fix requires fix_sql parameter.",
                error="missing_fix_sql",
            )
            return obs, 0.0

        # Evaluate grader score BEFORE applying the fix
        score_before = self._task.grader(self._conn)

        cur = self._conn.cursor()
        affected_total = 0
        errors = []

        # Execute each semicolon-separated statement
        statements = [s.strip() for s in fix_sql.split(";") if s.strip()]
        for stmt in statements:
            try:
                cur.execute(stmt)
                affected_total += cur.rowcount if cur.rowcount > 0 else 0
            except Exception as e:
                errors.append(str(e))

        self._conn.commit()

        # Grader score AFTER
        score_after = self._task.grader(self._conn)
        delta = round(score_after - score_before, 4)

        extra_reward = REWARD_FIX_APPLIED if affected_total > 0 else 0.0
        # Additional reward proportional to grader improvement
        extra_reward += max(delta, 0.0) * 0.5

        self._state.fixes_applied += 1

        validation = {
            "score_before": score_before,
            "score_after": score_after,
            "delta": delta,
            "errors": errors,
        }

        text = (
            f"Fix applied. Rows affected: {affected_total}. "
            f"Quality score: {score_before:.3f} → {score_after:.3f} (Δ {delta:+.3f})."
        )
        if errors:
            text += f" Errors: {'; '.join(errors)}"

        obs = Observation(
            done=False,
            reward=0.0,
            observation_text=text,
            data={"affected_rows": affected_total, "validation": validation},
        )
        return obs, extra_reward

    # ------------------------------------------------------------------

    def _act_finish(self):
        final_score = self._task.grader(self._conn)
        steps_used = self._state.step_count
        efficiency_bonus = (
            REWARD_EFFICIENCY_BONUS
            if steps_used <= self._state.max_steps // 2
            else 0.0
        )

        text = (
            f"Episode finished.\n"
            f"Final data quality score: {final_score:.4f}\n"
            f"Steps used: {steps_used}/{self._state.max_steps}\n"
            f"Efficiency bonus: {efficiency_bonus:.2f}"
        )

        obs = Observation(
            done=True,
            reward=0.0,
            observation_text=text,
            data={
                "final_score": final_score,
                "steps_used": steps_used,
                "efficiency_bonus": efficiency_bonus,
                "breakdown": {
                    "task_id": self._task.task_id,
                    "cumulative_reward": self._state.cumulative_reward,
                },
            },
        )
        return obs, final_score + efficiency_bonus
