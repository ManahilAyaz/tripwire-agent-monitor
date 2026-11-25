# Trajectory log schema (JSONL), v1.4

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
| `step` | int | 1-indexed, increments on every turn including think turns (see `is_think`). Not the same as the summary's `num_steps`, which counts real actions only |
| `timestamp` | str | ISO-8601 UTC with milliseconds and `Z`, taken when the environment replied (or immediately, for a think turn) |
| `thought` | str | the agent's reasoning text for this step, `""` if none - for a think turn (v1.2), this is the "think: ..." text itself |
| `action` | str | the command actually sent to the environment (mapped to the closest ALFWorld-admissible command when the model's raw text wasn't an exact match). `""` for a think turn - nothing was sent |
| `observation` | str | the environment's reply to that action. The fixed string `"OK."` for a think turn |
| `parse_ok` | bool | `false` if the model reply had no valid `Action:` and the logger substituted `look` |
| `retry_count` | int | API retries needed for this step's LLM call (0 if none) |
| `raw_action` | str | (v1.1) what the model actually produced, before mapping to `action`. Equal to `action` when `mapping` is `"exact"` |
| `mapping` | str | (v1.3) `"exact"` (`raw_action` was already admissible, or normalized-identical to one - see `normalized_action`), `"fuzzy"` (mapped to the closest same-verb admissible command, similarity >= 0.8 on normalized text), or `"none"` (no same-verb match good enough - `raw_action` sent to ALFWorld unchanged - or this is a think turn, where no mapping is attempted) |
| `normalized_action` | str | (v1.4) `raw_action` lowercased, punctuation-stripped, filler words removed ("over", "the", "a", "an", "please", "now") - what matching actually compares. `""` for a think/forced turn, where no mapping was attempted |
| `similarity` | float | (v1.4) the matched (or best attempted) same-verb candidate's score: `1.0` for `"exact"`, the real ratio for `"fuzzy"`, the best same-verb candidate's score (if any, else `0.0`) for `"none"` |
| `is_think` | bool | (v1.2) `true` for a ReAct-style "think: ..." reasoning turn - never sent to ALFWorld, doesn't count toward `max_steps`/ALFWorld's own step cap. Capped at 2 consecutive; a 3rd in a row is forced into a real action instead (see Decisions #8) |

```jsonl-step
{"type":"step","episode_id":"ep_0001","step":1,"timestamp":"2026-11-22T10:14:51.500Z","thought":"To solve the task, I need to find the potato, then heat it, then put it in sidetable.","action":"","observation":"OK.","parse_ok":true,"retry_count":0,"raw_action":"think: To solve the task, I need to find the potato, then heat it, then put it in sidetable.","mapping":"none","normalized_action":"","similarity":0.0,"is_think":true}
{"type":"step","episode_id":"ep_0001","step":2,"timestamp":"2026-11-22T10:14:52.101Z","thought":"","action":"go to fridge 1","observation":"You arrive at fridge 1. The fridge 1 is closed.","parse_ok":true,"retry_count":0,"raw_action":"go over to the fridge 1","mapping":"exact","normalized_action":"go to fridge 1","similarity":1.0,"is_think":false}
{"type":"step","episode_id":"ep_0001","step":3,"timestamp":"2026-11-22T10:14:54.380Z","thought":"","action":"open fridge 1","observation":"You open the fridge 1. In it, you see a potato 1.","parse_ok":true,"retry_count":0,"raw_action":"open fridge 1","mapping":"exact","normalized_action":"open fridge 1","similarity":1.0,"is_think":false}
{"type":"step","episode_id":"ep_0001","step":4,"timestamp":"2026-11-22T10:14:57.012Z","thought":"","action":"take potato 1 from fridge 1","observation":"You pick up the potato 1 from the fridge 1.","parse_ok":true,"retry_count":1,"raw_action":"take potato 2 from fridge 1","mapping":"fuzzy","normalized_action":"take potato 2 from fridge 1","similarity":0.963,"is_think":false}
```

## Episode summary line (required keys)
| key | type | meaning |
|---|---|---|
| `type` | str | always `"episode_summary"` |
| `schema_version` | str | `"1.1"` |
| `episode_id` | str | matches the step lines |
| `task_id` | str | ALFWorld game name taken from the gamefile path |
| `task_input` | str | the goal text, e.g. `heat some potato and put it in sidetable` |
| `initial_observation` | str | the room description shown before step 1 |
| `num_steps` | int | number of real environment actions taken (v1.2: think turns are step lines too, but are *not* counted here - this is what `max_steps` bounds) |
| `outcome` | str | one of the four values below |
| `model` | str | the agent's LLM name |
| `max_steps` | int | the step cap for this run (real actions only, per `num_steps` above) |
| `start_timestamp` | str | ISO-8601 UTC |
| `end_timestamp` | str | ISO-8601 UTC |
| `task_type` | str | (v1.2) the ALFWorld task-type bucket this episode's gamefile matched (`put`/`clean`/`heat`/`cool`/`puttwo`/`examine`) - which few-shot examples the agent was prompted with |
| `prompt_version` | str | (v1.2) which `configs/prompts/react_*.txt` template built this episode's system prompt, e.g. `"react_v3"` |

`outcome` values:
- `success`: the environment reported the task won.
- `failure`: the environment ended the episode without a win.
- `max_steps`: the agent hit the step cap without a win.
- `error`: infrastructure problem (API down, environment crash). Not a task result. `validate_logs.py` excludes these from the corpus.

`failure` and `max_steps` both count as "task failed" in analysis. `outcome` says how the episode ended; the failure *category* (why it went wrong) comes from the taxonomy and annotations, not from this field.

```jsonl-summary
{"type":"episode_summary","schema_version":"1.4","episode_id":"ep_0001","task_id":"pick_heat_then_place_in_recep-Potato-None-SideTable-10","task_input":"heat some potato and put it in sidetable","initial_observation":"You are in the middle of a room. Looking quickly around you, you see a fridge 1, a microwave 1, a sidetable 1.","num_steps":12,"outcome":"success","model":"qwen/qwen3.8-27b","max_steps":40,"start_timestamp":"2026-11-22T10:14:50.000Z","end_timestamp":"2026-11-22T10:15:40.200Z","task_type":"heat","prompt_version":"react_v3"}
```
The example lines are illustrative. Replace them with real lines once the T12 logger exists.

## Decisions (answers to the T03 open questions)
1. Timestamps: ISO-8601 UTC strings. Readable, sortable, one format everywhere.
2. `outcome`: four values above, kept separate from Aqsa's failure taxonomy (outcome = how it ended, taxonomy = why it failed).
3. `thought` is its own field. It is the agent's own output, so it is observable. Layer 2's prompt and LANGUTORY need it.
4. `retry_count` is per step. Aggregate it per episode when needed.
5. One JSONL file per episode, so a finished episode is a file with a summary line and a batch run can resume.
6. Added by Manahil, please confirm: `type` on every line, `parse_ok`, `initial_observation`, `model`, `max_steps`, `schema_version`.
7. **v1.1.** Added `raw_action` to the step line: the pilot runs showed a small local model (Llama 3.1 8B) frequently producing invalid/invented actions that ALFWorld silently turns into "Nothing happens." The agent now maps these to the closest admissible command so the episode can continue, but logging only the corrected `action` would erase exactly the kind of failure signal this project exists to monitor - so `raw_action` keeps the model's real, uncorrected output alongside it.
8. **v1.2.** Added task-type-matched ReAct few-shot prompting (`configs/prompts/`) to raise the pilot success rate off its 10% (4/40) baseline. Two new fields: `task_type` (summary) picks which few-shot examples an episode was prompted with; `is_think` (step) marks a ReAct-style "think: ..." reasoning turn, which is never sent to ALFWorld and doesn't count toward `max_steps` - so `num_steps` now means "real environment actions" specifically, not "step lines in the file" (a think turn is still its own step line, just excluded from that count). `prompt_version` (summary) records which prompt template (`configs/prompts/react_v3.txt`) built the episode, for comparing pilot batches collected under different prompt versions later.
9. **v1.3.** A 4-episode spot-check under v1.2 found `map_to_admissible_action` could map a "take" action to an unrelated "examine" command purely on string similarity (e.g. "take tomato 4 from fridge 1" -> "examine fridge 1"), since plain fuzzy-matching doesn't know "take" and "examine" are different kinds of action. Fixed by restricting fuzzy matches to admissible commands sharing the same first word (verb) as `raw_action`, and raising the acceptance threshold to similarity >= 0.8; anything that still doesn't clear both is sent to ALFWorld unchanged (`mapping: "none"`) instead of being silently substituted. New `mapping` step field (`"exact"`/`"fuzzy"`/`"none"`) records which case applied, since `raw_action` vs `action` alone doesn't say *why* they differ (or don't).
10. **v1.4.** The 0.8 threshold from #9 rejected harmless phrasing differences too (e.g. "go over to the fridge 1" vs "go to fridge 1", similarity ~0.76 - same verb, same meaning, just an extra filler word). Added normalization (lowercase, strip punctuation, drop filler words "over"/"the"/"a"/"an"/"please"/"now" - object names, numbers, and ALFWorld's own prepositions are left alone) before comparing: a normalized-exact match is now `mapping: "exact"` (the real admissible command is sent, not the normalized text), and the 0.8 fuzzy threshold applies to normalized text for everything else. New `normalized_action`/`similarity` step fields make this auditable - `similarity` is logged even for a rejected (`"none"`) near-miss, not just accepted matches.
11. Removed the "You are at: X. You are holding: Y." line added in v1.2 - it isn't part of standard ReAct, and telling the agent back its own derived state is the future correction layer's job (FYP-II), not this baseline agent's. Also removed the "if Nothing happens., try something different, don't repeat it" rule from `configs/prompts/react_v3.txt` for the same reason - steering the agent away from repetition after a failure signal is itself a correction behavior. No schema change for this one (no field relied on either).
