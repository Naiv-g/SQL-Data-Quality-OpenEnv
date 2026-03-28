---
title: SQL Data Quality Env
emoji: 🗄️
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# 🗄️ SQL Data Quality Environment

<div align="center">

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-6c63ff)](https://github.com/meta-pytorch/OpenEnv)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Space-orange)](https://huggingface.co/spaces)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**A real-world OpenEnv environment where AI agents audit SQL databases for data quality issues and generate corrective SQL queries.**

</div>

---

## 🎯 What This Environment Simulates

Data quality issues cost enterprises an estimated **$12.9M per year** on average (Gartner). Every data engineer and analytics team spends significant time:

- Detecting NULL values in critical columns
- Finding and deduplicating records with the same business key
- Identifying referential integrity violations (orphan foreign keys)
- Fixing type mismatches and format inconsistencies
- Correcting calculation errors in derived columns

This environment reproduces exactly these real workflows through an interactive SQLite database that agents explore using `list_tables`, `describe_table`, `query`, and `submit_fix` actions — the same tools a real data engineer would use.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Agent (LLM / RL)                      │
│          Observes text + structured JSON data            │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP  (reset / step / state)
┌────────────────────▼─────────────────────────────────────┐
│              FastAPI Server  (port 7860)                  │
│  ┌────────────────────────────────────────────────────┐  │
│  │         SQLDataQualityEnvironment                   │  │
│  │  reset() → fresh in-memory SQLite DB per episode   │  │
│  │  step()  → routes action, returns Observation      │  │
│  │  state   → episode metadata (step_count, score)    │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐   │
│  │ EASY     │  │ MEDIUM   │  │ HARD                 │   │
│  │1 table   │  │2 tables  │  │3 tables              │   │
│  │null/type │  │dedup+FK  │  │multi-table+biz rules │   │
│  └──────────┘  └──────────┘  └──────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

---

## 📐 OpenEnv Spec Compliance

| Requirement | Status |
|---|---|
| Typed `Action` Pydantic model | ✅ |
| Typed `Observation` Pydantic model | ✅ |
| Typed `State` Pydantic model | ✅ |
| `POST /reset` endpoint | ✅ |
| `POST /step` endpoint | ✅ |
| `GET /state` endpoint | ✅ |
| `openenv.yaml` manifest | ✅ |
| 3+ tasks with graders (0.0–1.0) | ✅ |
| Graders are deterministic | ✅ |
| Baseline inference script | ✅ |
| Working Dockerfile | ✅ |
| HF Spaces deployable | ✅ |

---

## 🎮 Action Space

The agent communicates through structured `Action` objects:

```python
class Action(BaseModel):
    action_type: ActionType        # required
    table_name:  Optional[str]     # for describe_table 
    sql:         Optional[str]     # for query (read-only SELECT)
    fix_sql:     Optional[str]     # for submit_fix (UPDATE/DELETE)
    reasoning:   Optional[str]     # optional chain-of-thought (not graded)
```

### Action Types

| `action_type` | Description | Required Fields |
|---|---|---|
| `list_tables` | List all tables in the database | — |
| `describe_table` | Schema + sample rows for a table | `table_name` |
| `query` | Execute a read-only SELECT/WITH | `sql` |
| `submit_fix` | Apply UPDATE/DELETE/ALTER fix statements | `fix_sql` |
| `finish` | End the episode, receive final score | — |

### Example Actions

```json
// Explore
{"action_type": "list_tables"}
{"action_type": "describe_table", "table_name": "customers"}
{"action_type": "query", "sql": "SELECT * FROM customers WHERE email IS NULL"}

// Fix
{"action_type": "submit_fix", "fix_sql": "UPDATE customers SET email='unknown@example.com' WHERE email IS NULL;"}
{"action_type": "finish"}
```

---

## 👁️ Observation Space

Each step returns an `Observation`:

```python
class Observation(BaseModel):
    done:             bool                   # True when episode ends
    reward:           float                  # Step reward (with shaping)
    observation_text: str                    # Human-readable description
    data: Optional[Dict[str, Any]]           # Structured payload (varies by action)
    error: Optional[str]                     # Error message if action failed
```

### Data Payload by Action

| Action | `data` structure |
|---|---|
| `list_tables` | `{tables: [str]}` |
| `describe_table` | `{columns: [{name, type, nullable}], sample_rows: [dict], total_rows: int}` |
| `query` | `{columns: [str], rows: [[...]], row_count: int}` |
| `submit_fix` | `{affected_rows: int, validation: {score_before, score_after, delta, errors}}` |
| `finish` | `{final_score: float, steps_used: int, efficiency_bonus: float, breakdown: dict}` |

---

## 📊 Episode State

```python
class State(BaseModel):
    episode_id:        str        # unique episode identifier
    task_id:           str        # 'easy' | 'medium' | 'hard'
    step_count:        int        # steps taken so far
    max_steps:         int        # budget (easy=20, medium=25, hard=35)
    cumulative_reward: float      # total reward accumulated
    issues_found:      int        # issues the agent has found
    fixes_applied:     int        # successful fix statements applied
    task_description:  str        # full task instructions
    available_tables:  List[str]  # tables in this episode's DB
```

---

## 📋 Tasks

### Task 1 — EASY: Customer Table Null & Type Audit
**Difficulty:** Easy | **Max steps:** 20 | **Table:** `customers`

The `customers` table has 10 rows with four categories of data quality issues:

| Category | Issue | Points |
|---|---|---|
| A | 3 rows missing `email` | 25% |
| B | 2 rows missing `phone` | 25% |
| C | 2 rows missing `city` or `country` | 25% |
| D | 3 rows with non-numeric `age` values (`'abc'`, `''`, `'N/A'`) | 25% |

**Expected agent strategy:** Describe table → query for NULLs by column → submit UPDATE fixes → finish

**Baseline score (GPT-4o-mini):** ~0.75

---

### Task 2 — MEDIUM: Products & Orders Integrity
**Difficulty:** Medium | **Max steps:** 25 | **Tables:** `products`, `orders`

Two related tables with three categories of issues:

| Category | Issue | Points |
|---|---|---|
| A | 2 duplicate SKUs (SKU-001, SKU-002) — keep lower product_id | 33% |
| B | 2 orphan orders referencing non-existent products | 33% |
| C | 2 orders with `quantity ≤ 0` (0 and -1) | 34% |

**Expected agent strategy:** Detect cross-table relationships → find duplicates with GROUP BY → identify FK violations → fix sequentially

**Baseline score (GPT-4o-mini):** ~0.60

---

### Task 3 — HARD: Multi-Table Schema & Business Rules
**Difficulty:** Hard | **Max steps:** 35 | **Tables:** `employees`, `departments`, `payroll`

Seven distinct issue categories across three tables:

| # | Category | Table | Issue |
|---|---|---|---|
| 1 | Date format | `employees` | Row 4: `hire_date='15/06/2018'` → ISO 8601 |
| 2 | Invalid FK | `employees` | Row 6: department `'Logistics'` not in `departments` |
| 3 | Self-ref integrity | `employees` | Row 7: `manager_id=99` doesn't exist |
| 4 | Business rule | `employees` | Row 8: negative `salary=-5000` |
| 5 | NULL constraint | `employees` | Row 11: NULL `department` |
| 6 | Calculation error | `payroll` | Rows 2, 5: `net_pay ≠ gross_pay - deductions` |
| 7 | Orphan record | `payroll` | Row 9: `emp_id=99` doesn't exist in `employees` |

**Expected agent strategy:** Deep multi-table analysis, verify self-referencing integrity, check derived column calculations, fix issues in dependency order

**Baseline score (GPT-4o-mini):** ~0.43

---

## 🏆 Reward Function

The reward function provides **dense, shaped signals** throughout the episode:

```
step_reward = -0.005                          # per-step efficiency penalty
            + 0.05                            # if fix affected ≥ 1 row(s)
            + max(grader_delta, 0.0) × 0.5   # proportional to quality improvement
```

On `finish()`:
```
finish_reward = final_grader_score            # 0.0 – 1.0
              + 0.10                          # efficiency bonus (≤ half step budget)
```

On timeout (step budget exhausted):
```
timeout_penalty = -0.05
```

**Design rationale:**
- The per-step penalty discourages aimless exploration without preventing necessary investigation
- `submit_fix` gives immediate feedback even before the episode ends
- The grader delta component rewards *meaningful* fixes, not just any SQL execution
- The efficiency bonus incentivises concise, targeted agents over brute-force approaches

---

## 🚀 Quick Start

### Local Setup

```bash
# Clone / download the project
cd scaler/

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Open http://localhost:7860 in your browser for the web UI
# OpenAPI docs at http://localhost:7860/docs
```

### Docker

```bash
docker build -t sql-data-quality-env .
docker run -p 7860:7860 sql-data-quality-env

# Health check
curl http://localhost:7860/health
```

### Python Client

```python
from client import SQLDataQualityClient
from models import Action, ActionType

with SQLDataQualityClient("http://localhost:7860") as client:
    # Start easy task
    obs = client.reset(task_id="easy")
    print(obs.observation_text)

    # Explore
    result = client.step(Action(action_type=ActionType.LIST_TABLES))
    result = client.step(Action(action_type=ActionType.DESCRIBE_TABLE, table_name="customers"))

    # Fix
    result = client.step(Action(
        action_type=ActionType.SUBMIT_FIX,
        fix_sql="UPDATE customers SET email='unknown@example.com' WHERE email IS NULL;"
    ))
    print(result.observation.data["validation"])

    # Finish
    result = client.step(Action(action_type=ActionType.FINISH))
    print(f"Final score: {result.observation.data['final_score']}")
```

---

## 🤖 Baseline Inference Script

The baseline script runs a GPT-4o-mini agent through all three tasks:

```bash
# Required environment variables
export HF_TOKEN="your-api-key"
export API_BASE_URL="https://api.openai.com/v1"   # or your custom endpoint
export MODEL_NAME="gpt-4o-mini"                    # or your model

# Run all tasks (server must be running)
python inference.py --url http://localhost:7860

# Run specific tasks
python inference.py --tasks easy medium

# Quiet mode (just scores)
python inference.py --quiet
```

### Baseline Scores (Reproducible)

| Task | GPT-4o-mini | Notes |
|---|---|---|
| Easy | ~0.75 | Misses some edge cases in age validation |
| Medium | ~0.60 | Struggles with ordering of operations (must delete dupes before orphan check) |
| Hard | ~0.43 | Date format conversion and multi-table coordination are challenging |
| **Average** | **~0.59** | Room for significant improvement |

---

## 🧪 Running Tests

```bash
pytest tests/ -v

# Expected output:
# test_reset_easy_returns_observation PASSED
# test_reset_sets_state PASSED
# ...
# 22 passed in X.XX seconds
```

---

## 🐳 Deploying to Hugging Face Spaces

1. Create a new HF Space with **Docker** SDK
2. Push this repository to the Space
3. The Space will automatically build and start on port 7860
4. Tag your Space with `openenv` for discoverability

The web UI at `/` provides a no-code interface for manual interaction.

---

## 📁 Project Structure

```
scaler/
├── openenv.yaml              # OpenEnv manifest
├── models.py                 # Pydantic models (Action, Observation, State, StepResult)
├── tasks.py                  # Task definitions, schemas, seed data, graders
├── environment.py            # Core environment logic (reset/step/state)
├── client.py                 # Synchronous HTTP client
├── inference.py              # Baseline inference script (OpenAI client)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Package metadata
├── Dockerfile                # Container definition
├── server/
│   └── app.py                # FastAPI application
└── tests/
    └── test_environment.py   # Test suite (pytest)
```

---

## 🔧 Environment Variables

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | LLM API endpoint | `https://api.openai.com/v1` |
| `MODEL_NAME` | Model identifier for inference | `gpt-4o-mini` |
| `HF_TOKEN` | API key (used as OpenAI `api_key`) | *(required)* |

---

## 🙏 Acknowledgments

Built for the [OpenEnv Community Challenge](https://github.com/meta-pytorch/OpenEnv) by Meta PyTorch × Hugging Face.

The SQL schema designs are inspired by real data quality issues encountered in production data warehouses.
