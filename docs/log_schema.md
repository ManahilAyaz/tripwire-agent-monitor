# Trajectory log schema (JSONL), v1.0

Shared contract between Kashmala (logger, runner), Manahil (representations, Layer 2, eval) and Aqsa (rules).
Change it only by PR, agreed by all three. If you change it, bump `schema_version`.

## Rules
1. One file per episode: `data/raw/<pilot|full>/<episode_id>.jsonl` (UTF-8, one JSON object per line).
2. Every line has a `type`: `"step"` (one per agent action) or `"episode_summary"` (exactly one, always the LAST line).
3. The summary line is written only when the episode finishes. A file without it is unfinished, and the batch runner re-runs it.
4. Log only what is externally observable: task input, the agent's text output, the action, the environment's reply, timestamps, outcome.
   No hidden states, no logprobs.
5. **No leaks.** Monitors (Layer 1, Layer 2) only ever receive a prefix: `task_input`, `initial_observation` and steps 1..k.
   They never see `outcome` or anything after step k. Evaluation code is the only code that reads `outcome`.
      One loader enforces this: `load_prefix(path, k)` in `src/tripwire/trajectory/` (T12). Everyone calls it; nobody writes their own truncation.
6. Ground-truth labels (decisive step, failure category) are NOT in these files. They live in `data/annotations/`.

## Step line (required keys)
| key | type | meaning |
|---|---|---|
| `type` | str | always `"step"` |
| `episode_id` | str | e.g. `ep_0001` |
| `step` | int | 1-indexed. Step 1 is the agent's first action |
| `timestamp` | str | ISO-8601 UTC with milliseconds and `Z`, taken when the environment replied |
| `thought` | str | the agent's reasoning text for this step, `""` if none |
| `action` | str | the command actually sent to the environment |
| `observation` | str | the environment's reply to that action |
| `parse_ok` | bool | `false` if the model reply had no valid `Action:` and the logger substituted `look` |
| `retry_count` | int | API retries needed for this step's LLM call (0 if none) |

```jsonl-step
{"type":"step","episode_id":"ep_0001","step":1,"timestamp":"2026-11-22T10:14:52.101Z","thought":"The potato may be in the fridge.","action":"go to fridge 1","observation":"You arrive at fridge 1. The fridge 1 is closed.","parse_ok":true,"retry_count":0}
{"type":"step","episode_id":"ep_0001","step":2,"timestamp":"2026-11-22T10:14:54.380Z","thought":"Open it.","action":"open fridge 1","observation":"You open the fridge 1. In it, you see a potato 1.","parse_ok":true,"retry_count":0}
{"type":"step","episode_id":"ep_0001","step":3,"timestamp":"2026-11-22T10:14:57.012Z","thought":"Take the potato.","action":"take potato 1 from fridge 1","observation":"You pick up the potato 1 from the fridge 1.","parse_ok":true,"retry_count":1}
```

## Episode summary line (required keys)
| key | type | meaning |
|---|---|---|
| `type` | str | always `"episode_summary"` |
| `schema_version` | str | `"1.0"` |
| `episode_id` | str | matches the step lines |
| `task_id` | str | ALFWorld game name taken from the gamefile path |
| `task_input` | str | the goal text, e.g. `heat some potato and put it in sidetable` |
| `initial_observation` | str | the room description shown before step 1 |
| `num_steps` | int | number of step lines |
| `outcome` | str | one of the four values below |
| `model` | str | the agent's LLM name |
| `max_steps` | int | the step cap for this run |
| `start_timestamp` | str | ISO-8601 UTC |
| `end_timestamp` | str | ISO-8601 UTC |

`outcome` values:
- `success`: the environment reported the task won.
- `failure`: the environment ended the episode without a win.
- `max_steps`: the agent hit the step cap without a win.
- `error`: infrastructure problem (API down, environment crash). Not a task result. `validate_logs.py` excludes these from the corpus.

`failure` and `max_steps` both count as "task failed" in analysis. `outcome` says how the episode ended; the failure *category* (why it went wrong) comes from the taxonomy and annotations, not from this field.

```jsonl-summary
{"type":"episode_summary","schema_version":"1.0","episode_id":"ep_0001","task_id":"pick_heat_then_place_in_recep-Potato-None-SideTable-10","task_input":"heat some potato and put it in sidetable","initial_observation":"You are in the middle of a room. Looking quickly around you, you see a fridge 1, a microwave 1, a sidetable 1.","num_steps":12,"outcome":"success","model":"qwen/qwen3.8-27b","max_steps":40,"start_timestamp":"2026-11-22T10:14:50.000Z","end_timestamp":"2026-11-22T10:15:40.200Z"}
```
The example lines are illustrative. Replace them with real lines once the T12 logger exists.

## Decisions (answers to the T03 open questions)
1. Timestamps: ISO-8601 UTC strings. Readable, sortable, one format everywhere.
2. `outcome`: four values above, kept separate from Aqsa's failure taxonomy (outcome = how it ended, taxonomy = why it failed).
3. `thought` is its own field. It is the agent's own output, so it is observable. Layer 2's prompt and LANGUTORY need it.
4. `retry_count` is per step. Aggregate it per episode when needed.
5. One JSONL file per episode, so a finished episode is a file with a summary line and a batch run can resume.
6. Added by Manahil, please confirm: `type` on every line, `parse_ok`, `initial_observation`, `model`, `max_steps`, `schema_version`.
