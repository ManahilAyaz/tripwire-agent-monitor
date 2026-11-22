# Failure Taxonomy v1.1 (T07)

| Item | Value |
|---|---|
| Project | Tripwire: Mid-Trajectory Failure Detection for Long-Horizon AI Agents |
| Task | T07, Week 1 (November 16-22, 2026) |
| Author | Aqsa Zaib |
| Status | Draft v1.1, updated for Trajectory Log Schema v1.0. Not yet reviewed or approved by the team or the supervisor. |
| Related files | `docs/candidate_signals.md` (T08), `docs/annotation_guideline.md` (T09), `configs/rules.yaml` and `SRS/Tripwire/monitors/rules.py` (T10) |

## 1. Purpose and scope

This taxonomy lists patterns of suspicious or failing behaviour that can be seen in an agent trajectory, mainly in ALFWorld text tasks. It is used by:

- the annotation guideline (T09), which assigns these category IDs to steps;
- the candidate signals (T08), which say how each pattern could be measured;
- the Layer 1 rules (T10 implements TX-01 only).

Scope limits:

- FYP-I is detection and evaluation only. This document does not describe correction, recovery, rollback, steering or replanning (FYP-II).
- Only externally observable information from the log is used (section 2.3). Hidden states are never used. Whether the logged `thought` text may be used is **not decided** (section 2.5).
- The taxonomy is a first version built from reasoning about ALFWorld-style trajectories. It has not been checked against real trajectories yet. All examples are illustrative and made up, not real runs. Real ALFWorld wording (for example how "You arrive at ..." is phrased, or which no-effect messages exist) must be checked on pilot data.
- The category IDs TX-01 to TX-08 and TX-99 are unchanged from v1.

## 2. Terms and conventions

### 2.1 Terms

| Term | Meaning |
|---|---|
| Pattern | A behaviour in the trajectory that matches one category below. |
| Suspicious pattern | A pattern with some evidence but not enough to say the agent is failing. This is what the monitor can raise. |
| Failure-related pattern | A pattern that reached the evidence bar for its category (defined in the annotation guideline) in an episode that ended in failure. |
| Episode failure | A property of the whole episode: the task was not completed. Decided from the raw `outcome` (`failure` or `max_steps`), not from patterns. |
| Recovery | The agent leaves a pattern by its own behaviour and continues sensibly. Only observed and recorded in FYP-I, never produced by the system. |

Suspicious behaviour and episode failure are separate things:

| Situation | Pattern seen? | Episode outcome |
|---|---|---|
| Agent repeats an action twice, then continues and finishes the task | Yes (suspicious) | `success` |
| Agent repeats a no-effect action until the step limit | Yes (failure-related) | `max_steps` |
| Agent behaves reasonably but runs out of steps | No clear pattern | `max_steps` (a miss for any pattern-based monitor) |
| Agent loops for many steps, then escapes and succeeds | Yes (suspicious, recovered) | `success` |

A pattern is evidence that something may go wrong. It never proves the episode will fail. A single unusual action must not be called a failure.

### 2.2 Action normalization (one definition for all documents and code)

To compare two actions, `norm(a)` is used everywhere (taxonomy, signals, rules, annotation):

1. trim leading and trailing whitespace;
2. collapse every run of whitespace into one space;
3. convert to lowercase.

Example: `"  Go to  Cabinet 1 "` becomes `"go to cabinet 1"`. Arguments are never removed, so `go to cabinet 1` and `go to cabinet 2` stay different actions. The code is `normalize_action()` in `SRS/Tripwire/monitors/rules.py`. Normalization is for comparison only; the log keeps the action exactly as sent.

### 2.3 Schema v1.0 fields this taxonomy relies on

Source of truth: `Trajectory log schema (JSONL), v1.0`. Step numbers are **1-indexed**. One JSONL file per episode; the `episode_summary` line is always last, and a file without it is unfinished.

| Field | Where | Used here |
|---|---|---|
| `step`, `action`, `observation` | step line | Yes (all categories) |
| `parse_ok` | step line | Yes (section 2.4) |
| `retry_count` | step line | No. It counts API retries for the LLM call. It is not an agent action and is never evidence of repetition. |
| `timestamp` | step line | No |
| `thought` | step line | Not used until the team/supervisor decide (section 2.5) |
| `task_input`, `initial_observation` | summary line | Yes for TX-07, TX-08 (through `load_prefix`, see below) |
| `outcome` | summary line | Annotation and evaluation only. Never given to a monitor. |
| `max_steps`, `num_steps`, `model`, timestamps | summary line | Not used by categories |

Monitors only receive a prefix (`task_input`, `initial_observation`, steps 1..k) from the team's `load_prefix(path, k)` (T12). Nothing in this taxonomy may need `outcome` or steps after k to *detect* a pattern. `outcome` is used only afterwards, to decide whether a pattern was failure-related.

Not in schema v1.0 (so unavailable): tool name, tool arguments, tool result or error status, an "action was valid" flag, and any GRAPHECTORY feature (for example `cycle_length`).

### 2.4 Parse failures versus rejected or unsuccessful actions

These three situations look alike in a quick read but are different and must not be mixed up.

| Situation | Evidence in the log | What really happened | Taxonomy treatment |
|---|---|---|---|
| **Parse failure** | `parse_ok = false` | The model reply had no valid `Action:`. The logger sent `look` instead. The logged `action` is the logger's substitution, not the agent's choice. `observation` is the environment's reply to that `look`. | Not an agent action. It never counts as evidence for TX-01, TX-03, TX-04 or TX-05, and the substituted `look` must not be treated as a repeated `look`. It is a data-quality event. The annotation guideline records it as TX-99 with note `parse_failure` so that its frequency can be measured. |
| **Environment-rejected or no-effect action** | `parse_ok = true` and the observation indicates that the action had no effect or was rejected | The agent produced a well-formed command, but the observation alone may not establish whether the action was invalid, impossible, or simply unhelpful. | TX-04 only when the available context clearly supports an invalid or impossible action. Repeated unsuccessful attempts may support TX-03. |
| **Valid but unhelpful action** | `parse_ok = true` and a normal observation | The environment executed it, but it did not move the task forward (for example a useless `look`). | No category for a single step. A stretch of them may form TX-01, TX-05 or TX-06. |

Two cautions:

- `parse_ok = true` only says the reply contained an `Action:` line. It does **not** say ALFWorld accepted the action. Acceptance can only be judged from the observation text.
- By default, observations that follow a substituted `look` (`parse_ok = false`) are excluded from observation-based signals and patterns (TX-05 and TX-06). This is a conservative convention because the agent did not choose the substituted action. The convention requires team confirmation (open point 5).

### 2.5 Thoughts (`thought`)

- Schema v1.0 logs the agent's reasoning text per step as `thought`. The schema decision says it is observable because it is the agent's own output.
- **Whether `thought` may be used for monitoring and for human annotation is a team/supervisor decision that has not been made.** Until it is made, nothing in T07 to T10 reads `thought`, and annotators do not read it (see the annotation guideline).
- Repeated reasoning is **not** a category here. In Manahil's GRAPHECTORY design, thoughts are stored on graph nodes (not as separate nodes), LivePlan adds thought nodes to detect repeated reasoning, and LANGUTORY (T06) covers text analysis. Adding a thought-repetition category or signal here would duplicate that work.

## 3. Overview

| ID | Name | Main evidence | Detection approach | Layer (planned) |
|---|---|---|---|---|
| TX-01 | Consecutive Action Repetition | Same action in neighbouring steps | Rule-based | 1 (T10, implemented, provisional) |
| TX-02 | Cyclic Action Loop | A short block of actions repeats | GRAPHECTORY `cycle_length` feature plus rule threshold (pending feature contract) | 1 (later) |
| TX-03 | Repeated Unsuccessful Action | Same action fails again and again | Rule-based, manual | 1 (later) |
| TX-04 | Invalid or Impossible Action | Environment rejects or ignores the action | Rule-based, manual | 1 (later) |
| TX-05 | Repeated Observation Without State Change | Identical observations over several steps | Rule-based | 1 (later) |
| TX-06 | Stagnation (Lack of Progress) | No new information or state change over a window | Rule-based proxy, semantic, manual | 1 proxy, 2 |
| TX-07 | Goal or Instruction Drift | Actions pursue something other than the task | Semantic, manual | 2 |
| TX-08 | Disregarded Observation | Useful information in an observation is ignored | Semantic, manual | 2 |
| TX-99 | Other / Unclassified | Does not fit above | Manual | none |

TX-99 exists so annotators do not force a bad fit. Frequent TX-99 use means the taxonomy needs a new category.

## 4. Categories

Notation: step t has action a_t and the observation o_t returned after it. Examples use ALFWorld-style text and are made up for illustration. Unless stated, `parse_ok` is true in every example.

### TX-01 Consecutive Action Repetition

| Field | Content |
|---|---|
| Definition | The same action is issued by the agent in neighbouring steps. |
| Observable evidence | `norm(a_t)` equal at steps t, t+1, ..., t+k-1 with k >= 2, all with `parse_ok = true` and consecutive step numbers. |
| Illustrative example | Steps 7, 8, 9: `look`, `look`, `look`. Observation each time: "You are in the middle of a room." |
| Not an example | Steps 7, 8, 9 are `look` but `parse_ok = false` at step 8: the logger substituted that `look`, so the agent did not repeat anything. |
| Detection approach | Rule-based (T10). Flag when a run reaches a configurable number of total occurrences (provisional default 3, see `configs/rules.yaml`). |
| Limitations and false positives | A repeat of two can be harmless (the agent double-checks). A repeat can be legitimate when the environment changed in between, though with ALFWorld text actions this is rare. Different arguments are different actions. API retries do not create extra steps in schema v1.0 (one record per step, retries only in `retry_count`), so they cannot create a false run. Parse-failure substitutions could, which is why they are excluded. |

### TX-02 Cyclic Action Loop

| Field | Content |
|---|---|
| Definition | A short block of two or more different actions is repeated back to back, so the agent goes round in a circle. |
| Observable evidence | Action sequence such as A, B, A, B, A, B, with observations also repeating. |
| Illustrative example | `open drawer 1`, `close drawer 1`, `open drawer 1`, `close drawer 1`, `open drawer 1`, `close drawer 1`, each time showing the same drawer contents. |
| Detection approach | Not implemented in FYP-I Week 1. GRAPHECTORY does not do full graph-wide cycle detection; it supplies a per-step `cycle_length` value. The rule monitor will decide when that value reaches an alert threshold. The exact format of `cycle_length` is **not yet confirmed** and it is **not** part of the raw JSONL schema. |
| Limitations and false positives | Searching many receptacles can look like "go to, open, close" repeated, but the arguments differ, so full action strings must be compared. A cycle with changing observations (for example the agent moves objects around) may be legitimate. |

### TX-03 Repeated Unsuccessful Action

| Field | Content |
|---|---|
| Definition | The same action is tried again after it already had no effect or returned an error, with nothing changed in between. |
| Observable evidence | The same `norm(action)` at several steps (not necessarily neighbouring), each followed by a no-effect or error observation, and no state-changing action between the attempts. |
| Illustrative example | `take mug 1 from countertop 1` at steps 12, 14 and 16, each followed by "Nothing happens.", with `look` at steps 13 and 15. |
| Detection approach | Rule-based: match observations against a list of no-effect messages, then count per action. Manual check for "nothing changed in between". |
| Limitations and false positives | The first failure is normal. A retry can be reasonable if the agent did something that could change the outcome in between (for example opened a container first). "Nothing happens." is a general ALFWorld no-effect reply, so the message list must be checked against real observations before use. Steps with `parse_ok = false` are not attempts. |

### TX-04 Invalid or Impossible Action

| Field | Content |
|---|---|
| Definition | A well-formed action the environment cannot carry out: a nonexistent object, or a precondition that is not met. |
| Observable evidence | `parse_ok = true`, with an observation and available task context that provide sufficient evidence that the environment rejected the action as invalid or impossible. A generic no-effect message alone is not sufficient. |
| Illustrative example | `put mug 1 in/on coffeemachine 1` while the agent is not holding the mug. Observation: "Nothing happens." In this illustrative example, the preceding task context and observations establish that the agent has not picked up the mug. The no-effect observation alone would not be sufficient evidence for TX-04. |
| Limitations and false positives | One invalid action is common and often harmless. A no-effect reply does not always mean the action was invalid (it can be valid but not useful). Schema v1.0 has no validity flag, so the observation text is the only evidence. `parse_ok = false` is a different thing (section 2.4). Tool-call errors cannot be detected because schema v1.0 has no tool fields, and ALFWorld actions are text commands, not tool calls. |

### TX-05 Repeated Observation Without State Change

| Field | Content |
|---|---|
| Definition | The environment returns the same normalized observation at consecutive agent steps, potentially indicating that the environment state has not meaningfully changed. |
| Observable evidence | Identical observation text after normalization at consecutive logged steps, with `parse_ok = true` at every step in the sequence. The actions may differ. A repeated observation is a candidate signal of no state change, but must be interpreted in context. |
| Illustrative example | Steps 20 to 23: `put mug 1 in/on coffeemachine 1`, `use coffeemachine 1`, `heat mug 1 with microwave 1`, `put mug 1 in/on coffeemachine 1`. The observation is "Nothing happens." every time. The actions differ, the observation does not, and nothing changes. |
| Correction note (v1.1) | v1 used `look`, `inventory`, `look`, `inventory`. That was wrong: `look` and `inventory` return *different* texts, so neighbouring observations alternate and are never identical. That sequence is an A-B-A-B alternation (TX-02 territory), not TX-05 under this definition. A window-based variant ("the same observation keeps coming back within a window") is a possible signal (SIG-03, window form) but is not the definition of TX-05. |
| Detection approach | Rule-based (exact match of normalized observation text). |
| Limitations and false positives | Some actions legitimately give the same answer (`inventory` when nothing changed). If observations are truncated in the log, matches can be wrong. Exact string match misses repeats that differ in a small detail. Observations from steps with `parse_ok = false` are excluded from TX-05 comparisons. Such steps also break an otherwise consecutive observation sequence (section 2.4). |

### TX-06 Stagnation (Lack of Progress)

| Field | Content |
|---|---|
| Definition | Over a stretch of steps, the agent gains no new information and changes nothing relevant to the task. |
| Observable evidence | No new observation text, no new location, no change in what is carried, over a window of steps. |
| Illustrative example | Ten steps of moving between two already visited places and examining the same objects, with no new object seen and nothing picked up. |
| Detection approach | Rule-based proxy (novelty of observations), semantic check for real progress, manual annotation. |
| Limitations and false positives | Progress is hard to define. Early search can look like "no progress" and is still fine. A novelty proxy can be fooled by observations that change in trivial ways. |

### TX-07 Goal or Instruction Drift

| Field | Content |
|---|---|
| Definition | The agent starts working on something that does not serve the task instruction. |
| Observable evidence | Several consecutive actions aimed at objects or receptacles unrelated to the task input, with no plausible link to a sub-goal. |
| Illustrative example | Task: "put a mug in the coffee machine". The agent takes a knife and a fork and carries them to the dining table. |
| Detection approach | Semantic (Layer 2) and manual. Needs `task_input`. |
| Limitations and false positives | Sub-goals look off-topic at first (go to the sink to clean the mug before placing it). Short exploration is normal. `task_input` is stored only in the `episode_summary` line, which does not exist for an unfinished episode. It is available through `load_prefix` when replaying a finished episode, but a monitor reading a still-growing file would not have it (see candidate signals, section 5, and open point 4). |

### TX-08 Disregarded Observation

| Field | Content |
|---|---|
| Definition | An observation contains information the agent needs, but the following actions behave as if it was not there. |
| Observable evidence | An observation shows the target object or the next needed condition, and the agent keeps searching or acting elsewhere for several steps. |
| Illustrative example | Observation at step 5: "On the countertop 1, you see a mug 1." The task asks for a mug. Steps 6 to 10: the agent visits other receptacles and never takes the mug. |
| Detection approach | Semantic (Layer 2) and manual. |
| Limitations and false positives | The seen object may not be the right one (a different mug, or one that is already used). Observations can be long or truncated. This needs real understanding of the task, so a rule cannot do it reliably. Same `task_input` availability issue as TX-07. |

### TX-99 Other / Unclassified

Used only by annotators for suspicious or failing behaviour that matches no category. Each use needs a note. Repeated use of the same kind of note is a reason to add a category in the next version. Parse failures (section 2.4) are recorded here with note `parse_failure` until the team decides whether they deserve their own category.

## 5. How categories overlap

Several categories can describe the same steps. One pattern often has a main category and secondary ones:

- Repeating a no-effect action three times in a row matches TX-01, TX-03 and possibly TX-04. The main category is the one that best explains why the behaviour is a problem (usually TX-03). The annotation guideline gives the precedence rules.
- TX-05 and TX-06 may overlap with TX-01 and TX-02 when repeated actions or cycles also produce repeated observations or little new information. However, each category must be supported by its own observable evidence.
- TX-06 is a broad label. Use a more specific category when one fits.

## 6. Mapping to signals

| Category | Signals in `docs/candidate_signals.md` |
|---|---|
| TX-01 | SIG-01, SIG-06 |
| TX-02 | SIG-02 (conditional on GRAPHECTORY), SIG-06 |
| TX-03 | SIG-04, SIG-05 |
| TX-04 | SIG-04 (candidate evidence only; a no-effect message does not independently establish invalidity) |
| TX-05 | SIG-03 |
| TX-06 | SIG-07, SIG-06, SIG-11 |
| TX-07 | SIG-08 |
| TX-08 | SIG-10 |

## 7. Open points for team review

1. The "no-effect" message list for ALFWorld has to be confirmed on real pilot observations (including the real wording of normal replies).
2. TX-02 to TX-08 evidence bars (in the annotation guideline) are provisional and need a pilot annotation round.
3. Whether to add categories after seeing real failures (for example wrong object choice, giving up early, or repeated parse failures). None are added now because there is no real data yet.
4. How `task_input` reaches a monitor on a run that is still in progress (it is only in the summary line). Question for the T12 loader owners.
5. Convention for `parse_ok = false` steps: exclude them from action-based and observation-based behavioural patterns, as described in this document. The team must decide whether parse failures should be recorded as a separate data-quality event or retained under TX-99 with note `parse_failure`. They must not automatically count as behavioural failure evidence.
6. Whether the `thought` field may be used by monitors and by annotators (supervisor/team decision).
7. The GRAPHECTORY `cycle_length` contract (needed before TX-02 can be implemented).
