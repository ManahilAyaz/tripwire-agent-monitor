# Tripwire: mid-trajectory failure detection for LLM agents

Tripwire watches an LLM agent (ReAct, Llama via Groq) while it works on long-horizon ALFWorld tasks
and flags the earliest sign of failure, using only observable behaviour (no training, no model internals).

**Layer 1** deterministic rule monitor (repeated actions, repeated failures, stagnation) ->
**Layer 2** LLM monitor (Gemini Flash), called only when Layer 1 flags -> fused risk score and alarm.
FYP-II adds a correction layer.

## Quickstart
```bash
git clone https://github.com/ManahilAyaz/tripwire-agent-monitor.git && cd tripwire-agent-monitor
pip install -r requirements.txt && pip install -e .
cp .env.example .env        # add GROQ_API_KEY / GEMINI_API_KEY (never commit .env)
pytest
```

## Layout
`src/tripwire/` (env, agent, trajectory, runner, representation, signals, monitors, baselines, eval),
`configs/`, `docs/`, `scripts/`, `data/`, `results/`, `notebooks/`, `report/`, `tests/`.

## Team workflow
See [docs/branching_rules.md](docs/branching_rules.md). One commit per task ID, one PR per week minimum, never push to main.

## Team
Manahil, Kashmala, Aqsa. Supervisor: Dr. Fahad Sherwani, FAST-NUCES Karachi.
