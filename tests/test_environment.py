"""
tests/test_environment.py — Unit tests for the SQL Data Quality Environment.

Tests:
  - reset() returns clean state for each task
  - step() with each action type
  - Grader functions produce correct scores
  - Edge cases (bad SQL, unknown tables, step budget exhaustion)

Run with: pytest tests/ -v
"""

from __future__ import annotations

import sqlite3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from environment import SQLDataQualityEnvironment
from models import Action, ActionType
from tasks import TASKS, _easy_grader, _medium_grader, _hard_grader, _exec_ddl


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return SQLDataQualityEnvironment()


def make_conn(schema_sql: str) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _exec_ddl(conn, schema_sql)
    return conn


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_easy_returns_observation(self, env):
        obs = env.reset(task_id="easy")
        assert obs.done is False
        assert obs.reward == 0.0
        assert "customers" in obs.observation_text.lower()

    def test_reset_sets_state(self, env):
        env.reset(task_id="medium")
        s = env.state
        assert s.task_id == "medium"
        assert s.step_count == 0
        assert s.fixes_applied == 0

    def test_reset_hard(self, env):
        obs = env.reset(task_id="hard")
        assert "employees" in obs.observation_text.lower()
        assert env.state.max_steps == 35

    def test_reset_invalid_task_fallback(self, env):
        obs = env.reset(task_id="impossible")
        assert env.state.task_id == "easy"

    def test_reset_clears_previous_state(self, env):
        env.reset(task_id="easy")
        # Dirty the DB
        env.step(Action(action_type=ActionType.SUBMIT_FIX,
                        fix_sql="UPDATE customers SET email='x@x.com' WHERE email IS NULL;"))
        env.reset(task_id="easy")
        # Should start fresh
        assert env.state.step_count == 0
        assert env.state.fixes_applied == 0


# ---------------------------------------------------------------------------
# step() tests
# ---------------------------------------------------------------------------

class TestStep:
    def test_list_tables(self, env):
        env.reset(task_id="easy")
        result = env.step(Action(action_type=ActionType.LIST_TABLES))
        assert result.done is False
        assert "tables" in result.observation.data
        assert "customers" in result.observation.data["tables"]

    def test_describe_table(self, env):
        env.reset(task_id="easy")
        result = env.step(Action(action_type=ActionType.DESCRIBE_TABLE,
                                 table_name="customers"))
        assert result.observation.error is None
        assert "columns" in result.observation.data

    def test_describe_table_missing_name(self, env):
        env.reset(task_id="easy")
        result = env.step(Action(action_type=ActionType.DESCRIBE_TABLE))
        assert result.observation.error == "missing_table_name"

    def test_query_returns_rows(self, env):
        env.reset(task_id="easy")
        result = env.step(Action(
            action_type=ActionType.QUERY,
            sql="SELECT * FROM customers WHERE email IS NULL"
        ))
        assert result.observation.data["row_count"] == 3  # 3 nulls seeded

    def test_query_blocks_dml(self, env):
        env.reset(task_id="easy")
        result = env.step(Action(
            action_type=ActionType.QUERY,
            sql="UPDATE customers SET email='x' WHERE id=1"
        ))
        assert result.observation.error == "write_op_in_query"

    def test_submit_fix_improves_score(self, env):
        env.reset(task_id="easy")
        # Apply fix for email nulls
        result = env.step(Action(
            action_type=ActionType.SUBMIT_FIX,
            fix_sql="UPDATE customers SET email='unknown@example.com' WHERE email IS NULL;"
        ))
        assert result.observation.data["validation"]["delta"] > 0

    def test_submit_fix_applies_correctly(self, env):
        env.reset(task_id="easy")
        env.step(Action(
            action_type=ActionType.SUBMIT_FIX,
            fix_sql="UPDATE customers SET email='x@x.com' WHERE email IS NULL;"
        ))
        result = env.step(Action(
            action_type=ActionType.QUERY,
            sql="SELECT COUNT(*) as cnt FROM customers WHERE email IS NULL"
        ))
        assert result.observation.data["rows"][0][0] == 0

    def test_finish_returns_score(self, env):
        env.reset(task_id="easy")
        result = env.step(Action(action_type=ActionType.FINISH))
        assert result.done is True
        assert "final_score" in result.observation.data
        score = result.observation.data["final_score"]
        assert 0.0 <= score <= 1.0

    def test_step_count_increments(self, env):
        env.reset(task_id="easy")
        for _ in range(3):
            env.step(Action(action_type=ActionType.LIST_TABLES))
        assert env.state.step_count == 3

    def test_episode_ends_at_max_steps(self, env):
        env.reset(task_id="easy")
        result = None
        for _ in range(25):  # easy max_steps=20
            result = env.step(Action(action_type=ActionType.LIST_TABLES))
            if result.done:
                break
        assert result.done is True


# ---------------------------------------------------------------------------
# Grader tests — pure unit tests against known data
# ---------------------------------------------------------------------------

class TestGraders:
    def test_easy_grader_zero_before_fix(self):
        conn = make_conn(TASKS["easy"].schema_sql)
        score = _easy_grader(conn)
        # Initial data has many issues → score < 1.0
        assert score < 1.0

    def test_easy_grader_perfect_after_all_fixes(self):
        conn = make_conn(TASKS["easy"].schema_sql)
        cur = conn.cursor()
        cur.execute("UPDATE customers SET email='x@x.com' WHERE email IS NULL")
        cur.execute("UPDATE customers SET phone='000-0000' WHERE phone IS NULL")
        cur.execute("UPDATE customers SET city='Unknown' WHERE city IS NULL")
        cur.execute("UPDATE customers SET country='Unknown' WHERE country IS NULL")
        cur.execute("UPDATE customers SET age=NULL WHERE CAST(age AS INTEGER)=0 AND age NOT IN ('0')")
        conn.commit()
        score = _easy_grader(conn)
        assert score == 1.0

    def test_medium_grader_zero_before_fix(self):
        conn = make_conn(TASKS["medium"].schema_sql)
        score = _medium_grader(conn)
        assert score < 1.0

    def test_medium_grader_perfect_after_all_fixes(self):
        conn = make_conn(TASKS["medium"].schema_sql)
        cur = conn.cursor()
        # Remove duplicate SKUs (keep lower product_id)
        cur.execute(
            "DELETE FROM products WHERE product_id NOT IN "
            "(SELECT MIN(product_id) FROM products GROUP BY sku)"
        )
        # Remove orphan orders
        cur.execute(
            "DELETE FROM orders WHERE product_id NOT IN "
            "(SELECT product_id FROM products)"
        )
        # Fix invalid quantities
        cur.execute("UPDATE orders SET quantity=1 WHERE quantity <= 0")
        conn.commit()
        score = _medium_grader(conn)
        assert score >= 0.99

    def test_hard_grader_zero_to_perfect(self):
        conn = make_conn(TASKS["hard"].schema_sql)
        before = _hard_grader(conn)
        assert before < 1.0

        cur = conn.cursor()
        cur.execute("UPDATE employees SET hire_date='2018-06-15' WHERE emp_id=4")
        cur.execute("UPDATE employees SET department=NULL WHERE emp_id=6")
        cur.execute("UPDATE employees SET manager_id=NULL WHERE emp_id=7")
        cur.execute("UPDATE employees SET salary=NULL WHERE emp_id=8")
        cur.execute("UPDATE employees SET department='HR' WHERE emp_id=11")
        cur.execute(
            "UPDATE payroll SET net_pay=gross_pay-deductions "
            "WHERE ABS(net_pay-(gross_pay-deductions))>0.01"
        )
        cur.execute(
            "DELETE FROM payroll WHERE emp_id NOT IN (SELECT emp_id FROM employees)"
        )
        conn.commit()
        after = _hard_grader(conn)
        assert after == 1.0

    def test_grader_scores_in_range(self):
        for task_id, task in TASKS.items():
            conn = make_conn(task.schema_sql)
            score = task.grader(conn)
            assert 0.0 <= score <= 1.0, f"Task {task_id} score out of range: {score}"
            conn.close()

    def test_grader_deterministic(self):
        """Same DB state → same score every time."""
        for task_id, task in TASKS.items():
            conn = make_conn(task.schema_sql)
            s1 = task.grader(conn)
            s2 = task.grader(conn)
            assert s1 == s2, f"Task {task_id} grader is not deterministic"
            conn.close()
