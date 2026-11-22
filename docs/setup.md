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

## Running the Week 1 unit tests (schema-independent code only)

The retry/backoff wrapper (T13 core) and the logger scaffold (T12, provisional field names) run
anywhere — no ALFWorld needed:

```bash
pip install -r requirements.txt && pip install -e .
pytest
```
