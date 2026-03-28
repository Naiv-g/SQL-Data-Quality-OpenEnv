"""
tasks.py — Task definitions, database schemas, and programmatic graders.

Each task is a self-contained Python object with:
  - description            : shown to the agent at episode start
  - schema_sql             : DDL + seed data to build the in-memory SQLite DB
  - max_steps              : episode budget
  - grader(db_conn) → float: deterministic scoring function (0.0 – 1.0)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable, List


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _exec_ddl(conn: sqlite3.Connection, ddl: str) -> None:
    """Execute a multi-statement DDL script."""
    cursor = conn.cursor()
    for stmt in ddl.split(";"):
        stmt = stmt.strip()
        if stmt:
            cursor.execute(stmt)
    conn.commit()


# ---------------------------------------------------------------------------
# Task 1 — EASY: Single-table null/type audit
# ---------------------------------------------------------------------------

_EASY_DDL = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT,
    email       TEXT,
    phone       TEXT,
    age         TEXT,      -- BUG: should be INTEGER (stored as TEXT)
    city        TEXT,
    country     TEXT
);

INSERT INTO customers VALUES (1,  'Alice Johnson',   'alice@example.com',  '555-0101', '29',  'New York',  'USA');
INSERT INTO customers VALUES (2,  'Bob Smith',       NULL,                 '555-0102', '34',  'London',    'UK');
INSERT INTO customers VALUES (3,  'Carol White',     'carol@example.com',  NULL,       'abc', 'Paris',     'France');
INSERT INTO customers VALUES (4,  'Dave Brown',      'dave@example.com',   '555-0104', '41',  NULL,        'Germany');
INSERT INTO customers VALUES (5,  'Eve Davis',       NULL,                 '555-0105', '27',  'Toronto',   'Canada');
INSERT INTO customers VALUES (6,  'Frank Miller',    'frank@example.com',  '555-0106', '',    'Sydney',    'Australia');
INSERT INTO customers VALUES (7,  'Grace Wilson',    'grace@example.com',  '555-0107', '52',  'Tokyo',     'Japan');
INSERT INTO customers VALUES (8,  'Henry Moore',     NULL,                 NULL,       '38',  NULL,        NULL);
INSERT INTO customers VALUES (9,  'Irene Taylor',    'irene@example.com',  '555-0109', '19',  'Dubai',     'UAE');
INSERT INTO customers VALUES (10, 'Jack Anderson',   'jack@example.com',   '555-0110', 'N/A', 'Mumbai',    'India');
"""

_EASY_DESCRIPTION = """
# Task 1 (EASY): Customer Table Null & Type Audit

You are a data quality agent. Audit the `customers` table and fix all data quality issues.

**Known issues to find and fix:**
1. NULL values in `email` (rows where email IS NULL)
2. NULL values in `phone`
3. NULL values in `city` or `country`
4. The `age` column contains non-numeric values (e.g. 'abc', '', 'N/A') — these should be set to NULL

**Actions available:**
- `list_tables`: See all tables
- `describe_table` (table_name): Schema + sample rows
- `query` (sql): Run a read-only SELECT
- `submit_fix` (fix_sql): Apply an UPDATE/DELETE fix
- `finish`: End the episode and receive your final score

**Scoring:**
- 25 pts per category fixed (up to 100 pts total → score 0.0–1.0)
- Bonus 0.1 for finishing under 15 steps
"""


def _easy_grader(conn: sqlite3.Connection) -> float:
    """
    Score 0.0-1.0 based on how many issue categories were fixed.
    Categories:
      A) email nulls   → 2 rows fixed
      B) phone nulls   → 2 rows fixed (id=3,8)
      C) city/country nulls → rows 4,8 for city; row 8 for country
      D) invalid age values (abc, '', N/A) → set to NULL
    Each category worth 0.25.
    """
    cur = conn.cursor()
    score = 0.0

    # A) No nulls in email
    cur.execute("SELECT COUNT(*) FROM customers WHERE email IS NULL")
    if cur.fetchone()[0] == 0:
        score += 0.25

    # B) No nulls in phone
    cur.execute("SELECT COUNT(*) FROM customers WHERE phone IS NULL")
    if cur.fetchone()[0] == 0:
        score += 0.25

    # C) No nulls in city or country
    cur.execute(
        "SELECT COUNT(*) FROM customers WHERE city IS NULL OR country IS NULL"
    )
    if cur.fetchone()[0] == 0:
        score += 0.25

    # D) Non-numeric ages set to NULL or corrected
    # Original non-numeric: 'abc' (id=3), '' (id=6), 'N/A' (id=10)
    cur.execute(
        "SELECT COUNT(*) FROM customers "
        "WHERE age IS NOT NULL AND CAST(age AS INTEGER) = 0 AND age NOT IN ('0')"
    )
    invalid_count = cur.fetchone()[0]
    if invalid_count == 0:
        score += 0.25

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: Duplicate detection + referential integrity
# ---------------------------------------------------------------------------

_MEDIUM_DDL = """
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    sku          TEXT NOT NULL,
    name         TEXT,
    category     TEXT,
    price        REAL,
    stock        INTEGER
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    product_id   INTEGER,
    quantity     INTEGER,
    unit_price   REAL,
    status       TEXT
);

-- Seed products — SKU duplicates exist
INSERT INTO products VALUES (1,  'SKU-001', 'Widget A',    'electronics', 29.99, 100);
INSERT INTO products VALUES (2,  'SKU-002', 'Gadget B',    'electronics', 49.99,  50);
INSERT INTO products VALUES (3,  'SKU-001', 'Widget A v2', 'electronics', 31.99,  75);  -- SKU duplicate!
INSERT INTO products VALUES (4,  'SKU-003', 'Thingamajig', 'tools',       14.99, 200);
INSERT INTO products VALUES (5,  'SKU-004', 'Doohickey',   'tools',        9.99,   0);
INSERT INTO products VALUES (6,  'SKU-002', 'Gadget B Pro','electronics', 59.99,  30);  -- SKU duplicate!
INSERT INTO products VALUES (7,  'SKU-005', 'Sprocket',    'mechanical',  19.99,  80);
INSERT INTO products VALUES (8,  'SKU-006', 'Bracket',     'mechanical',   4.99, 150);
INSERT INTO products VALUES (9,  'SKU-007', 'Cog',         'mechanical',   7.99,  60);
INSERT INTO products VALUES (10, 'SKU-008', 'Flange',      'mechanical',   3.49,  90);

-- Seed orders — some reference non-existent product_ids (orphan rows)
INSERT INTO orders VALUES (101, 1,   5, 29.99, 'completed');
INSERT INTO orders VALUES (102, 2,   3, 49.99, 'completed');
INSERT INTO orders VALUES (103, 99,  1, 10.00, 'pending');   -- orphan: product 99 missing
INSERT INTO orders VALUES (104, 3,   2, 31.99, 'shipped');
INSERT INTO orders VALUES (105, 4,   8, 14.99, 'completed');
INSERT INTO orders VALUES (106, 100, 4, 20.00, 'pending');   -- orphan: product 100 missing
INSERT INTO orders VALUES (107, 7,  10, 19.99, 'shipped');
INSERT INTO orders VALUES (108, 8,   6,  4.99, 'completed');
INSERT INTO orders VALUES (109, 5,   0,  9.99, 'pending');   -- quantity = 0 (invalid)
INSERT INTO orders VALUES (110, 9,  -1,  7.99, 'shipped');   -- quantity = -1 (invalid)
INSERT INTO orders VALUES (111, 1,   2, 29.99, 'completed');
INSERT INTO orders VALUES (112, 2,   1, 49.99, 'cancelled');
"""

_MEDIUM_DESCRIPTION = """
# Task 2 (MEDIUM): Products & Orders Integrity

You are a data quality agent. Two tables need cleanup: `products` and `orders`.

**Known issues to find and fix:**
1. Duplicate SKUs in `products` — keep the row with the LOWER product_id, delete the rest
2. Orphan orders — orders where `product_id` does not exist in `products` (delete them)
3. Invalid quantities — orders where `quantity <= 0` (set quantity to 1)

**Actions available:**
- `list_tables`, `describe_table`, `query`, `submit_fix`, `finish`

**Scoring:**
- 33 pts per category fixed (≈ 0.33 each → 1.0 total with partial credit)
- Partial credit given if some but not all rows of a category are fixed
"""


def _medium_grader(conn: sqlite3.Connection) -> float:
    cur = conn.cursor()
    score = 0.0

    # A) Duplicate SKUs — SKU-001 and SKU-002 should appear exactly once
    cur.execute(
        "SELECT sku, COUNT(*) as cnt FROM products GROUP BY sku HAVING cnt > 1"
    )
    dup_skus = cur.fetchall()
    sku_score = 1.0 - min(len(dup_skus) / 2.0, 1.0)
    score += sku_score * 0.333

    # B) Orphan orders — should be 0
    cur.execute(
        "SELECT COUNT(*) FROM orders "
        "WHERE product_id NOT IN (SELECT product_id FROM products)"
    )
    orphans = cur.fetchone()[0]
    orphan_score = 1.0 if orphans == 0 else max(0.0, 1.0 - orphans / 2.0)
    score += orphan_score * 0.333

    # C) Invalid quantities — should all be > 0
    cur.execute("SELECT COUNT(*) FROM orders WHERE quantity <= 0")
    bad_qty = cur.fetchone()[0]
    qty_score = 1.0 if bad_qty == 0 else max(0.0, 1.0 - bad_qty / 2.0)
    score += qty_score * 0.334

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Task 3 — HARD: Multi-table schema audit + business-rule violations
# ---------------------------------------------------------------------------

_HARD_DDL = """
CREATE TABLE employees (
    emp_id       INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    department   TEXT,
    manager_id   INTEGER,   -- self-referencing FK (may be invalid)
    salary       REAL,
    hire_date    TEXT        -- stored as TEXT, should be ISO 8601
);

CREATE TABLE departments (
    dept_id   INTEGER PRIMARY KEY,
    dept_name TEXT UNIQUE NOT NULL,
    budget    REAL
);

CREATE TABLE payroll (
    payroll_id   INTEGER PRIMARY KEY,
    emp_id       INTEGER,
    month        TEXT,   -- 'YYYY-MM'
    gross_pay    REAL,
    deductions   REAL,
    net_pay      REAL    -- should equal gross_pay - deductions
);

-- Departments
INSERT INTO departments VALUES (1, 'Engineering', 500000.0);
INSERT INTO departments VALUES (2, 'Marketing',   200000.0);
INSERT INTO departments VALUES (3, 'HR',          150000.0);
INSERT INTO departments VALUES (4, 'Finance',     300000.0);

-- Employees — several issues
INSERT INTO employees VALUES (1,  'Alice',   'Engineering', NULL, 95000.0,  '2019-03-15');
INSERT INTO employees VALUES (2,  'Bob',     'Engineering', 1,    85000.0,  '2020-07-01');
INSERT INTO employees VALUES (3,  'Carol',   'Marketing',   1,    72000.0,  '2021-01-10');
INSERT INTO employees VALUES (4,  'Dave',    'Marketing',   3,    68000.0,  '15/06/2018');  -- bad date format
INSERT INTO employees VALUES (5,  'Eve',     'HR',          3,    60000.0,  '2022-03-01');
INSERT INTO employees VALUES (6,  'Frank',   'Logistics',   2,    77000.0,  '2020-09-14');  -- dept 'Logistics' not in departments
INSERT INTO employees VALUES (7,  'Grace',   'Finance',     99,   90000.0,  '2018-11-05');  -- manager 99 doesn't exist
INSERT INTO employees VALUES (8,  'Henry',   'Engineering', 1,    -5000.0,  '2023-01-20');  -- negative salary!
INSERT INTO employees VALUES (9,  'Irene',   'Finance',     4,    82000.0,  '2021-06-30');
INSERT INTO employees VALUES (10, 'Jack',    'HR',          5,    58000.0,  '2020-02-28');
INSERT INTO employees VALUES (11, 'Kate',    NULL,          1,    70000.0,  '2022-07-01');  -- NULL department
INSERT INTO employees VALUES (12, 'Leo',     'Engineering', 1,    88000.0,  '2023-05-15');

-- Payroll — net_pay errors and orphan records
INSERT INTO payroll VALUES (1,  1,  '2024-01', 8000.0,  1200.0, 6800.0);  -- correct
INSERT INTO payroll VALUES (2,  2,  '2024-01', 7200.0,  1080.0, 5500.0);  -- net_pay wrong! (should be 6120)
INSERT INTO payroll VALUES (3,  3,  '2024-01', 6100.0,   900.0, 5200.0);  -- correct
INSERT INTO payroll VALUES (4,  4,  '2024-01', 5800.0,   870.0, 4930.0);  -- correct
INSERT INTO payroll VALUES (5,  5,  '2024-01', 5100.0,   760.0, 4000.0);  -- net_pay wrong! (should be 4340)
INSERT INTO payroll VALUES (6,  6,  '2024-01', 6600.0,   990.0, 5610.0);  -- correct
INSERT INTO payroll VALUES (7,  7,  '2024-01', 7700.0,  1150.0, 6550.0);  -- correct
INSERT INTO payroll VALUES (8,  8,  '2024-01', 4800.0,   720.0, 4080.0);  -- correct (salary in employees is wrong)
INSERT INTO payroll VALUES (9,  99, '2024-01', 5000.0,   750.0, 4250.0);  -- orphan payroll: emp 99 doesn't exist
INSERT INTO payroll VALUES (10, 9,  '2024-01', 7000.0,  1050.0, 5950.0);  -- correct
INSERT INTO payroll VALUES (11, 10, '2024-01', 5000.0,   750.0, 4250.0);  -- correct
INSERT INTO payroll VALUES (12, 11, '2024-01', 6000.0,   900.0, 5100.0);  -- correct
INSERT INTO payroll VALUES (13, 12, '2024-01', 7500.0,  1125.0, 6375.0);  -- correct
"""

_HARD_DESCRIPTION = """
# Task 3 (HARD): Multi-Table Schema & Business-Rule Audit

You are a senior data quality engineer. Three tables need comprehensive auditing.

**Issues to identify and fix:**

**employees table:**
1. Row 4 (Dave): `hire_date` in wrong format ('15/06/2018') — convert to ISO 8601 ('2018-06-15')
2. Row 6 (Frank): department 'Logistics' does not exist in `departments` — set to NULL or valid dept
3. Row 7 (Grace): `manager_id = 99` does not exist — set to NULL
4. Row 8 (Henry): `salary = -5000` — invalid, set to NULL (flag for HR review)
5. Row 11 (Kate): NULL department — set to 'HR' (HR manages unassigned employees)

**payroll table:**
6. Rows 2, 5: `net_pay ≠ gross_pay - deductions` — recalculate and fix
7. Row 9: orphan payroll record (emp_id=99 doesn't exist) — delete it

**Scoring:**
- Each of the 7 issue categories worth ~14.3 pts (0.143 each → max 1.0)
- Partial credit within categories where applicable
"""


def _hard_grader(conn: sqlite3.Connection) -> float:
    cur = conn.cursor()
    score = 0.0
    per_issue = 1.0 / 7.0

    # 1. Dave hire_date format
    cur.execute("SELECT hire_date FROM employees WHERE emp_id = 4")
    row = cur.fetchone()
    if row and row[0] == "2018-06-15":
        score += per_issue

    # 2. Frank invalid department fixed
    cur.execute("SELECT department FROM employees WHERE emp_id = 6")
    row = cur.fetchone()
    if row:
        dept = row[0]
        # Accept NULL or any valid department name
        if dept is None or dept in ("Engineering", "Marketing", "HR", "Finance"):
            score += per_issue

    # 3. Grace manager_id fixed
    cur.execute("SELECT manager_id FROM employees WHERE emp_id = 7")
    row = cur.fetchone()
    if row and row[0] is None:
        score += per_issue

    # 4. Henry negative salary fixed
    cur.execute("SELECT salary FROM employees WHERE emp_id = 8")
    row = cur.fetchone()
    if row and (row[0] is None or row[0] >= 0):
        score += per_issue

    # 5. Kate NULL department fixed
    cur.execute("SELECT department FROM employees WHERE emp_id = 11")
    row = cur.fetchone()
    if row and row[0] is not None:
        score += per_issue

    # 6. payroll net_pay errors fixed (both rows 2 AND 5)
    cur.execute(
        "SELECT COUNT(*) FROM payroll "
        "WHERE ABS(net_pay - (gross_pay - deductions)) > 0.01"
    )
    wrong_net = cur.fetchone()[0]
    if wrong_net == 0:
        score += per_issue
    elif wrong_net <= 1:
        score += per_issue * 0.5  # partial credit

    # 7. Orphan payroll deleted
    cur.execute(
        "SELECT COUNT(*) FROM payroll "
        "WHERE emp_id NOT IN (SELECT emp_id FROM employees)"
    )
    orphans = cur.fetchone()[0]
    if orphans == 0:
        score += per_issue

    return round(min(score, 1.0), 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

@dataclass
class Task:
    task_id: str
    difficulty: str
    description: str
    schema_sql: str
    max_steps: int
    grader: Callable[[sqlite3.Connection], float]
    tables: List[str] = field(default_factory=list)


TASKS: dict[str, Task] = {
    "easy": Task(
        task_id="easy",
        difficulty="easy",
        description=_EASY_DESCRIPTION,
        schema_sql=_EASY_DDL,
        max_steps=20,
        grader=_easy_grader,
        tables=["customers"],
    ),
    "medium": Task(
        task_id="medium",
        difficulty="medium",
        description=_MEDIUM_DESCRIPTION,
        schema_sql=_MEDIUM_DDL,
        max_steps=25,
        grader=_medium_grader,
        tables=["products", "orders"],
    ),
    "hard": Task(
        task_id="hard",
        difficulty="hard",
        description=_HARD_DESCRIPTION,
        schema_sql=_HARD_DDL,
        max_steps=35,
        grader=_hard_grader,
        tables=["employees", "departments", "payroll"],
    ),
}
