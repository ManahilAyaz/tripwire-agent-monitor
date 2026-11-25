# Setup guide (T11)

Companion to `notebooks/colab_setup.ipynb`. Packages Manahil's real, working T01 setup
(`scripts/t01_sanity_test.py`, PR #2, branch `w1/t01-sanity-test`) into a guide the whole team
can follow — not a separate reimplementation.

## Platform: Colab (or WSL2/Linux), with an isolated venv — not native Windows, not Colab's default Python

**Two issues were hit and confirmed while setting this up, independent of whose code it is:**

1. **Native Windows doesn't work at all.** ALFWorld depends on `textworld`, which depends on
   `jericho`. `jericho` has no prebuilt Windows wheel and fails to build from source:
   ```
   Getting requirements to build wheel: finished with status 'error'
   ...
   FileNotFoundError: [WinError 2] The system cannot find the file specified
   ```
   Its build step shells out to a Unix build tool that isn't on Windows/PATH. Use Colab or a
   real Linux environment (WSL2 Ubuntu — not Docker Desktop's internal WSL distro).

2. **Colab's own default Python (3.13) also doesn't work.** `alfworld.agents.environment`
   crashes on `env.reset()` with `NameError: name 'r' is not defined` inside
   `textworld/envs/pddl/textgen/__init__.py` — an upstream TextWorld bug on very new Python.
   Fix: install everything into an isolated venv pinned to an older Python (Manahil uses 3.10;
   we tested 3.11 too, both avoid it) via `uv`, and run ALFWorld code through that venv's
   interpreter, never Colab's system Python.

## Model: not Llama-3.1-8B-Instruct — confirmed independently by two people

CLAUDE.md fixes the monitored agent as Llama-3.1-8B-Instruct via Groq. That model is currently
unreachable:
- `llama-3.1-8b-instant` → `404 model_not_found`; a live `GET /v1/models` check showed no Llama
  model available at all on a tested Groq key.
- `openai/gpt-oss-20b` was tried next — Groq's docs confirm gpt-oss models ship **built-in
  tools** (`browser_search`, `code_interpreter`) that fire automatically with no way to disable
  them, causing `400` errors (`"Tool choice is none, but model called a tool"`) in a plain
  ReAct loop with no real tool use.
- **`qwen/qwen3.8-27b`** works, and is what Manahil's merged-pending T01 PR already uses
  independently — two separate attempts landed on the same model. **Still flagged to the team
  as a real change to a documented project assumption**, not something either of us decided
  unilaterally.

## Prerequisites

- A Groq API key ([console.groq.com](https://console.groq.com)) — set via `GROQ_API_KEY`,
  entered through a secure prompt (`getpass`), never hard-coded or committed. If a key has ever
  been pasted into a notebook, chat, or committed file in plaintext, treat it as compromised and
  rotate it first.

## Install (Colab)

```bash
pip install -q uv
uv venv --python 3.10 /content/venv
uv pip install --python /content/venv/bin/python alfworld openai pyyaml

export ALFWORLD_DATA=/content/alfworld_data
/content/venv/bin/alfworld-download

git clone https://github.com/alfworld/alfworld
cp alfworld/configs/base_config.yaml .
```

## Run the sanity test

Open `notebooks/colab_setup.ipynb` and run all cells. It clones the team repo's `main` branch
(PR #2 has merged), sets up the venv above, prompts for `GROQ_API_KEY`, then runs:

```bash
/content/venv/bin/python scripts/t01_sanity_test.py --model qwen/qwen3.8-27b --steps 5 --out notebooks/t01_sample_output.json
```

This is Manahil's real script, not a copy — the whole point of T11 is to make her already-working
setup reproducible for everyone else, not to maintain a second implementation that can drift
from it.

## Where T12 picks up

`scripts/t01_sanity_test.py` currently dumps one whole-episode JSON file (`{"goal", "steps":
[{t, thought, action, observation}]}`). T12's job is reshaping that into one JSONL line per
step plus an episode-summary line, once `docs/log_schema.md` (T03) is agreed — see the schema
draft proposal (shared with Manahil directly, not part of this repo) which already uses this
exact script's real field names (`thought`, `action`, `observation`) as its basis. The
provisional logger scaffold for that is in `src/tripwire/trajectory/` — separate from this
setup guide, and not something you need to touch just to get the environment running.

## react_v3 prompting (T13 follow-up, 2026-11-25)

The 4/40 (10%) baseline success rate on the first real Llama 3.1 8B pilot batch prompted a
prompting rework, `configs/prompts/react_v3.txt` + `configs/prompts/react_alfworld_examples.json`
(`src/tripwire/agent/react_agent.py`'s `load_system_prompt`/`load_task_examples`). No report
chapter exists yet to record this, so it's noted here instead - this section should move into
`report/02_environment_agent.md` once that's being written.

**Source and what we changed vs. the original.** The few-shot examples
(`configs/prompts/react_alfworld_examples.json`) are verbatim text from `ysymyth/ReAct`
(`prompts/alfworld_3prompts.json`, MIT-licensed), 2 of the 3 given examples per ALFWorld task
type, unmodified. But two things in the live prompt are **our own additions, not part of the
original ReAct setup**, and matter for interpreting results:
- The per-turn `Valid actions: ...` list (ALFWorld's own `admissible_commands`) - this comes
  from the earlier admissible-commands fix (schema v1.1, PR #22), not from ReAct.
- The `You are at: X. You are holding: Y.` line - new in this change, tracked from the
  executed action itself (`react_agent._update_state`), not parsed from observation text.

Neither appears in ReAct's original prompts or paper. If this agent's behavior is compared
against ReAct's own reported numbers later, that difference should be called out - this isn't
a like-for-like reproduction of ReAct, it's ReAct's few-shot/think-step format plus Tripwire's
own admissible-action grounding on top.

**Task contamination.** The few-shot examples are themselves real ALFWorld *training-split*
tasks (not synthetic), e.g. "put some spraybottle on toilet." Pilot/full batches also sample
from the train split (`env.reset()`), so there's a real chance of resampling one of the exact
tasks the agent was already shown solved in its own prompt - which would quietly inflate the
success rate on that episode. `scripts/run_batch.py` and `scripts/run_episode.py` both guard
against this: `reset_fn` is wrapped with
`runner.batch_runner.avoid_example_tasks(reset_fn, react_agent.example_task_inputs())`, which
resamples (up to 50 times) whenever the drawn task's goal text matches one of the 12 example
tasks, and raises loudly instead of silently letting one through if it can't find a clean task.
This matches on goal text only (not room layout), so it's intentionally conservative - a
different physical trial that happens to share an example's exact task description also gets
excluded, not just the literal same trial.

## Running the Week 1 unit tests (schema-independent code only)

The retry/backoff wrapper (T13 core) and the logger scaffold (T12, provisional field names) run
anywhere — no ALFWorld needed:

```bash
pip install -r requirements.txt && pip install -e .
pytest
```
