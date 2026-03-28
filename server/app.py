"""
server/app.py — FastAPI server for the SQL Data Quality Environment.

Exposes the OpenEnv standard endpoints:
  POST /reset   → returns initial Observation
  POST /step    → executes one Action, returns StepResult
  GET  /state   → returns current State
  GET  /health  → liveness probe
  GET  /tasks   → list available tasks with descriptions
  GET  /        → HTML web interface (for HF Space UI)
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on path when running inside server/ sub-package
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from environment import SQLDataQualityEnvironment
from models import Action, Observation, State, StepResult
from tasks import TASKS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SQL Data Quality Environment",
    description=(
        "An OpenEnv-compatible environment where AI agents audit SQL databases "
        "for data quality issues and generate corrective SQL. "
        "Implements the standard step() / reset() / state() API."
    ),
    version="1.0.0",
)

# Single shared environment instance (stateful, session-based)
env = SQLDataQualityEnvironment()


# ---------------------------------------------------------------------------
# Request schemas (thin wrappers so swagger docs are clear)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe — always returns 200."""
    return {"status": "ok", "env": "sql_data_quality_env"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = None):
    """
    Reset the environment and start a new episode.

    Args:
        task_id: 'easy' | 'medium' | 'hard' (default: 'easy')
        seed: optional random seed
        episode_id: optional explicit episode ID
    """
    if req is None:
        req = ResetRequest()
    obs = env.reset(
        task_id=req.task_id,
        seed=req.seed,
        episode_id=req.episode_id,
    )
    return obs


@app.post("/step", response_model=StepResult)
def step(action: Action):
    """
    Execute one agent action and return the result.

    Action types:
    - list_tables: show available tables
    - describe_table: show schema + sample rows
    - query: run a read-only SELECT
    - submit_fix: apply UPDATE/DELETE/ALTER fix statements
    - finish: end the episode and get final score
    """
    result = env.step(action)
    return result


@app.get("/state", response_model=State)
def state():
    """Return the current episode state (metadata only, not DB contents)."""
    return env.state


@app.get("/tasks")
def list_tasks():
    """Return all available tasks with their descriptions and difficulty."""
    return {
        tid: {
            "task_id": t.task_id,
            "difficulty": t.difficulty,
            "max_steps": t.max_steps,
            "tables": t.tables,
            "description": t.description,
        }
        for tid, t in TASKS.items()
    }


@app.get("/", response_class=HTMLResponse)
def web_ui():
    """Minimal HTML interface for manual interaction via HF Space."""
    return HTMLResponse(content=_WEB_UI_HTML, status_code=200)


# ---------------------------------------------------------------------------
# Minimal interactive HTML UI
# ---------------------------------------------------------------------------

_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>SQL Data Quality Environment</title>
<style>
  :root{--bg:#0f1117;--card:#1a1d27;--accent:#6c63ff;--text:#e2e8f0;--dim:#8892a4;--green:#4ade80;--red:#f87171;--yellow:#fbbf24}
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;min-height:100vh;padding:24px}
  h1{font-size:1.8rem;background:linear-gradient(135deg,var(--accent),#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
  .subtitle{color:var(--dim);font-size:.9rem;margin-bottom:24px}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:20px;max-width:1200px;margin:0 auto}
  .card{background:var(--card);border-radius:12px;padding:20px;border:1px solid rgba(108,99,255,.2)}
  label{font-size:.85rem;color:var(--dim);display:block;margin-bottom:6px;margin-top:14px}
  select,textarea,input{width:100%;background:#0f1117;border:1px solid rgba(255,255,255,.1);border-radius:8px;color:var(--text);padding:10px;font-size:.9rem;outline:none;transition:border .2s}
  select:focus,textarea:focus,input:focus{border-color:var(--accent)}
  textarea{resize:vertical;min-height:80px;font-family:monospace}
  button{background:linear-gradient(135deg,var(--accent),#a78bfa);color:#fff;border:none;border-radius:8px;padding:10px 20px;cursor:pointer;font-size:.9rem;font-weight:600;margin-top:12px;transition:opacity .2s;width:100%}
  button:hover{opacity:.85}
  .output{background:#0f1117;border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:14px;font-family:monospace;font-size:.82rem;white-space:pre-wrap;min-height:120px;max-height:400px;overflow-y:auto;margin-top:12px;color:var(--green)}
  .badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:700;margin-left:8px}
  .easy{background:rgba(74,222,128,.15);color:var(--green)}
  .medium{background:rgba(251,191,36,.15);color:var(--yellow)}
  .hard{background:rgba(248,113,113,.15);color:var(--red)}
  .score-bar{height:8px;background:rgba(255,255,255,.1);border-radius:4px;margin-top:8px}
  .score-fill{height:100%;border-radius:4px;background:linear-gradient(90deg,var(--accent),var(--green));transition:width .4s ease}
  h2{font-size:1rem;font-weight:700;margin-bottom:2px}
  .dimtext{font-size:.8rem;color:var(--dim)}
  @media(max-width:768px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div style="max-width:1200px;margin:0 auto">
  <h1>🗄️ SQL Data Quality Environment</h1>
  <p class="subtitle">OpenEnv-compatible · Real-world agent benchmark · 3 difficulty levels</p>

  <div class="grid">
    <!-- Left panel: actions -->
    <div>
      <div class="card">
        <h2>1. Start Episode</h2>
        <label>Task Difficulty</label>
        <select id="taskId">
          <option value="easy">Easy — Single-table null audit</option>
          <option value="medium">Medium — Duplicate + referential integrity</option>
          <option value="hard">Hard — Multi-table schema + business rules</option>
        </select>
        <button onclick="doReset()">▶ Reset / New Episode</button>
        <div class="output" id="resetOut">Click Reset to start a new episode...</div>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>2. Take Action</h2>
        <label>Action Type</label>
        <select id="actionType" onchange="updateFields()">
          <option value="list_tables">list_tables</option>
          <option value="describe_table">describe_table</option>
          <option value="query">query</option>
          <option value="submit_fix">submit_fix</option>
          <option value="finish">finish</option>
        </select>

        <div id="field_table" style="display:none">
          <label>Table Name</label>
          <input id="tableName" placeholder="e.g. customers"/>
        </div>

        <div id="field_sql" style="display:none">
          <label>SQL Query (read-only)</label>
          <textarea id="sqlQuery" rows="3" placeholder="SELECT * FROM customers WHERE email IS NULL LIMIT 10"></textarea>
        </div>

        <div id="field_fix" style="display:none">
          <label>Fix SQL (UPDATE / DELETE)</label>
          <textarea id="fixSql" rows="4" placeholder="UPDATE customers SET email = 'unknown@example.com' WHERE email IS NULL;"></textarea>
        </div>

        <button onclick="doStep()">⚡ Execute Action</button>
        <div class="output" id="stepOut">Execute an action to see results...</div>
      </div>
    </div>

    <!-- Right panel: state -->
    <div>
      <div class="card">
        <h2>Episode State <span id="taskBadge" class="badge easy">easy</span></h2>
        <div id="stateDisplay" class="dimtext">No episode started</div>
        <button onclick="doState()" style="margin-top:10px;background:#1a1d27;border:1px solid rgba(108,99,255,.4);color:var(--accent)">↻ Refresh State</button>
      </div>

      <div class="card" style="margin-top:16px">
        <h2>Available Tasks</h2>
        <div id="tasksDisplay" class="dimtext">Loading...</div>
      </div>
    </div>
  </div>
</div>

<script>
const API = '';

async function doReset(){
  const task_id = document.getElementById('taskId').value;
  const r = await fetch(API+'/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task_id})});
  const d = await r.json();
  document.getElementById('resetOut').textContent = d.observation_text || JSON.stringify(d,null,2);
  updateBadge(task_id);
  doState();
}

async function doStep(){
  const action_type = document.getElementById('actionType').value;
  const payload = {action_type};
  if(action_type==='describe_table') payload.table_name = document.getElementById('tableName').value;
  if(action_type==='query') payload.sql = document.getElementById('sqlQuery').value;
  if(action_type==='submit_fix') payload.fix_sql = document.getElementById('fixSql').value;
  const r = await fetch(API+'/step',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
  const d = await r.json();
  const obs = d.observation || d;
  let txt = obs.observation_text || JSON.stringify(d,null,2);
  if(d.reward!=null) txt += `\\n\\n[reward: ${d.reward.toFixed(4)}, done: ${d.done}]`;
  if(obs.data) txt += `\\n\\nData: ${JSON.stringify(obs.data,null,2)}`;
  document.getElementById('stepOut').textContent = txt;
  doState();
}

async function doState(){
  const r = await fetch(API+'/state');
  const d = await r.json();
  const pct = d.step_count && d.max_steps ? Math.round(d.step_count/d.max_steps*100) : 0;
  document.getElementById('stateDisplay').innerHTML = `
    <table style="width:100%;font-size:.82rem;border-collapse:collapse">
      <tr><td style="color:var(--dim);padding:3px 0">Episode ID</td><td>${d.episode_id||'-'}</td></tr>
      <tr><td style="color:var(--dim)">Task</td><td>${d.task_id||'-'}</td></tr>
      <tr><td style="color:var(--dim)">Steps</td><td>${d.step_count||0} / ${d.max_steps||'-'}</td></tr>
      <tr><td style="color:var(--dim)">Cumulative Reward</td><td>${(d.cumulative_reward||0).toFixed(4)}</td></tr>
      <tr><td style="color:var(--dim)">Fixes Applied</td><td>${d.fixes_applied||0}</td></tr>
    </table>
    <div class="score-bar" style="margin-top:10px"><div class="score-fill" style="width:${pct}%"></div></div>
    <div style="font-size:.75rem;color:var(--dim);margin-top:4px">Step budget: ${pct}% used</div>
  `;
  updateBadge(d.task_id);
}

async function loadTasks(){
  const r = await fetch(API+'/tasks');
  const d = await r.json();
  const html = Object.values(d).map(t=>`
    <div style="margin-bottom:12px;padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,.05)">
      <strong>${t.task_id}</strong> <span class="badge ${t.difficulty}">${t.difficulty}</span>
      <div style="font-size:.78rem;color:var(--dim);margin-top:4px">Tables: ${t.tables.join(', ')} · Max steps: ${t.max_steps}</div>
    </div>
  `).join('');
  document.getElementById('tasksDisplay').innerHTML = html;
}

function updateFields(){
  const val = document.getElementById('actionType').value;
  document.getElementById('field_table').style.display = val==='describe_table'?'block':'none';
  document.getElementById('field_sql').style.display = val==='query'?'block':'none';
  document.getElementById('field_fix').style.display = val==='submit_fix'?'block':'none';
}

function updateBadge(task_id){
  const b = document.getElementById('taskBadge');
  b.textContent = task_id||'easy';
  b.className = 'badge '+(task_id||'easy');
}

loadTasks();
</script>
</body>
</html>
"""

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
